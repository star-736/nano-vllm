import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
# from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.models.models import model_dict
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


"""
llm_engine.generate() -> llm_engine.step() -> scheduler.schedule() -> block_manager.allocate()
以上操作会为传入的seqs分配对应的block，以及搜索seq可能的缓存前缀，加到seq.num_cached_tokens中
-> model_runner.run() -> model_runner.prepare_prefill() / prepare_decode()
"""

class ModelRunner:
    """
    nano-vllm的模型运行类，负责模型的前向传播、下一token采样等。
    """
    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kv_cache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank # 当前进程的rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank) # 设置当前进程的GPU设备
        default_dtype = torch.get_default_dtype() # 默认数据类型
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")

        # self.model = Qwen3ForCausalLM(hf_config) # 初始化模型
        self.model = model_dict[hf_config.model_type](hf_config) # 初始化模型
        load_model(self.model, config.model) # 根据不同组件的权重加载方法，加载模型权重
        self.sampler = Sampler() # 初始化采样器
        self.warmup_model() # 预跑一遍模型
        self.allocate_kv_cache() # 分配kv缓存，对应config.py中num_kvcache_blocks: int = -1
        if not self.enforce_eager:
            self.capture_cudagraph() # enforce_eager为true，不会开cuda_graph
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1: # 单卡不会跑以下判断
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20) # 创建共享内存
                dist.barrier() # 等待所有进程都到达这一步
            else:
                dist.barrier() # 等待所有进程都到达这一步
                self.shm = SharedMemory(name="nanovllm") # 连接共享内存
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        """读取共享内存"""
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        """写入共享内存"""
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        """预热模型：用模拟数据让模型完整跑一遍前向流程，触发 CUDA 懒加载初始化"""
        torch.cuda.empty_cache() # 释放 PyTorch 不再使用、但被 CUDA 运行时缓存占用的《空闲显存》，将其归还给 GPU
        torch.cuda.reset_peak_memory_stats() # 重置显存峰值统计
        # max_num_batched_tokens：总token上限；max_model_len：单序列最大长度
        # num_seqs：能塞下的最大序列数（不超过配置的max_num_seqs）
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len # 16384, 4096
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs) # min(4, 512)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)] # 构造模拟数据：num_seqs个序列，每个序列是max_model_len个0
        self.run(seqs, True) # 前向计算一遍，is_prefill=True
        torch.cuda.empty_cache() # 预热完成后，再次清空缓存

    def allocate_kv_cache(self):
        """为所有层的attn分配kv缓存，kv cache block在内存空间是连续的"""
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info() # 获取GPU内存信息
        used = total - free # 已使用的内存
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"] # 内存峰值
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"] # 当前内存使用情况
        num_kv_heads = hf_config.num_key_value_heads // self.world_size # 每个GPU分配到的的kv头数量
        assert hf_config.hidden_size // hf_config.num_attention_heads == 0
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads) # 每个头的维度（这么写的原因是qwen2没有head_dim这个属性）
        # block_bytes：每个kv block占用的字节数 = 单个block存放的token数 * [(k + v) * attn层数 * kv头数 * 每个头的维度] * 数据类型大小
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes # 计算能分配的kv block数量
        assert config.num_kvcache_blocks > 0
        # 分配kv_cache，共num_kvcache_blocks块，《是连续的！！！》
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules(): # 对每层layer分配kv cache
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"): # Attention类中有k_cache和v_cache
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        # 给 FlashAttention/变长批处理用，告诉 kernel 每个序列 query/key 的起止
        cu_seqlens_q = [0] # 每个序列在拼接后的 input_ids 中的起始和结束位置，q的累积长度，只包含需要计算的token
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq) # 当前序列长度
            input_ids.extend(seq[seq.num_cached_tokens:]) # 去掉当前序列中已经缓存的token后的token_ids
            positions.extend(list(range(seq.num_cached_tokens, seqlen))) # extend需要计算的token在seq里的下标
            seqlen_q = seqlen - seq.num_cached_tokens # 当前序列中需要计算的部分长度
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q) # 加入q的累积长度
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k) # 加入k的累积长度
            max_seqlen_q = max(seqlen_q, max_seqlen_q) # seqs里需要新算的部分的最大长度
            max_seqlen_k = max(seqlen_k, max_seqlen_k) # seqs里最大长度
            if not seq.block_table: # warmup不需要slot_mapping 构建
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks): # (已使用的block数, 需要的总block数)
                # 为还没写入 kv cache  的 block 生成逐 token 的物理位置映射
                start = seq.block_table[i] * self.block_size # 起始索引
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    # 最后一个block可能未填满，用 seq.last_block_num_tokens 精确到实际 token 数
                    end = start + seq.last_block_num_tokens
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]: # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        """为decode输出做准备"""
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [] # 采样温度列表
        for seq in seqs:
            temperatures.append(seq.temperature) # 添加不同seq的采样温度
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """
        处理送来的seqs，根据is_prefill来决定是prefill还是decode
        来自：llm_engine.py token_ids = self.model_runner.call("run", seqs, is_prefill)
        """
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None # 采样温度
        logits = self.run_model(input_ids, positions, is_prefill) # 模型前向计算，返回logits
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None # 采样token_id，只会用主进程来做
        reset_context() # 重置全局变量
        return token_ids # 返回生成的token_ids列表

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
