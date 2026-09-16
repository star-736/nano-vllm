# main 同步前的核心代码学习快照

来源：`6156279`。这是完整历史快照，不是当前可执行代码。原文件的行尾空白也按原样保留。
上游 `bb823b3` 已引入 chunked prefill、新的 block hashing 和序列传输语义，以下旧注释不应直接用于解释新版调度。当前说明见 [同步说明](upstream-sync-20260917.md)。

## nanovllm/config.py

```python
import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str # 模型路径
    max_num_batched_tokens: int = 16384 # 总token上限
    max_num_seqs: int = 512 # 最大batch数
    max_model_len: int = 4096 # 最大序列长度
    gpu_memory_utilization: float = 0.9 # 总显存的 90% 减去当前/峰值占用后的预算
    tensor_parallel_size: int = 1 # 张量并行大小
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None # HuggingFace配置
    eos: int = -1 # 结束符
    kv_cache_block_size: int = 256 # KV Cache块大小
    num_kvcache_blocks: int = -1 # 自动计算需要的KV Cache块数

    def __post_init__(self): # 实例化后自动调用
        assert os.path.isdir(self.model)
        assert self.kv_cache_block_size % 256 == 0 # KV Cache块大小必须是256的倍数
        assert 1 <= self.tensor_parallel_size <= 8 # 张量并行大小必须在1到8之间
        self.hf_config = AutoConfig.from_pretrained(self.model) # config.json也会加载，传递给具体的model对象
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings) # 最大序列长度不能超过模型最大位置嵌入长度
        assert self.max_num_batched_tokens >= self.max_model_len # 批处理的总token上限至少能容纳一个最大长度的序列，目前16384>4096

```

## nanovllm/engine/block_manager.py

```python
from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence

"""
重点解析：
    prefix cache：
        复用不同序列之间的相同前缀的kv cache block，但必须是放满的block
        在allocate(seq)中去尝试去复用缓存的block，如果找不到，则分配新的block
"""

class Block:
    """用于存储每个block的信息"""
    def __init__(self, block_id):
        self.block_id = block_id # 当前block的id
        self.ref_count = 0 # 引用的次数
        self.hash = -1 # hash值
        self.token_ids = [] # 该block里的token_ids列表

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash # block满了以后，会更新hash值
        self.token_ids = token_ids # 更新token_ids列表

    def reset(self):
        self.ref_count = 1 # 重置引用次数
        self.hash = -1 # 重置hash值
        self.token_ids = [] # 重置token_ids列表


class BlockManager:
    """用于管理所有block的分配和释放"""
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size # 每个block的大小
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)] # 总block列表，num_blocks是传入的可用block数量
        self.hash_to_block_id: dict[int, int] = dict() # hash值到block_id的映射
        self.free_block_ids: deque[int] = deque(range(num_blocks)) # 可用的block_id集合，是个双向队列
        self.used_block_ids: set[int] = set() # 已使用的block_id集合，不能重复

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """
        一个block放满了，才计算hash值，
        hash可以快速比对两个block是否相同
        """
        h = xxhash.xxh64()
        if prefix != -1: # 有前缀，会把前缀加到hash值里
            h.update(prefix.to_bytes(8, "little"))
        # 若prefix为-1，则不添加任何前缀
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        """
        分配一个新block，
        确保 block 未被占用，重置后从空闲 队列转入已用集合并返回
        """
        block = self.blocks[block_id] # 获取block
        assert block.ref_count == 0 # 确保block没用过
        block.reset() # 初始化block
        self.free_block_ids.remove(block_id) # 从队列中移除
        self.used_block_ids.add(block_id) # 加到已用集合
        return self.blocks[block_id] # 返回block

    def _deallocate_block(self, block_id: int) -> Block:
        """
        释放一个block，
        确保引用数为 0，把 block 从已用集合放回空闲队列
        """
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        """
        判断空闲block数量是否能覆盖传入序列需要的block数
        在scheduler里会有判断
        """
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        """
        为seq分配block
        在prefill阶段执行，只会执行一次
        """
        assert not seq.block_table # 确保seq的block_table为空，即第一次分配blocks
        h = -1 # 初始化hash值
        cache_miss = False # 初始化缓存miss标志
        for i in range(seq.num_blocks): # 遍历seq所需的block数
            token_ids = seq.block(i) # 获取seq当前block的token_ids列表
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1 # 如果一个block放满了，计算hash值
            block_id = self.hash_to_block_id.get(h, -1) # 去字典里找hash值对应的block_id，如果找不到，返回-1
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                # 如果没找到，或者找到了但是token_ids列表不一样（不同的token_ids算出了同一个hash），说明缓存miss
                cache_miss = True          
            if cache_miss: # 如果缓存miss了，需要分配新的block
                block_id = self.free_block_ids[0] # 从空闲队列里取一个block_id
                block = self._allocate_block(block_id) # 返回新分配的block
            else: # 如果缓存hit了，直接使用block
                seq.num_cached_tokens += self.block_size # 更新seq的缓存token总数
                if block_id in self.used_block_ids:
                    # 如果block_id在已用集合里，说明这个block之前被分配过，直接指向对应的block，同时需要增加引用次数
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    # 说明只是查字典查到了，但之前被释放了，只是字典里没有清除
                    # block_id不在已用集合里，调用_allocate_block分配新的block
                    block = self._allocate_block(block_id) 
            if h != -1: # 如果hash值不是-1，说明这个block是满的，把compute_hash里计算的hash值赋值进去
                # 如果是-1，分配完新的block，不走后续逻辑
                block.update(h, token_ids) # 赋值hash值
                self.hash_to_block_id[h] = block_id # 在字典里把hash值和block_id对应起来
            seq.block_table.append(block_id) # 把block_id加到seq的block_table里

    def deallocate(self, seq: Sequence):
        """为seq释放block"""
        for block_id in reversed(seq.block_table): # 从后往前遍历seq的block_table
            block = self.blocks[block_id] # 取出block
            block.ref_count -= 1 # 减少引用次数
            if block.ref_count == 0: # 如果引用次数为0，说明这个block可以释放了
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0 # 清零seq的缓存token总数
        seq.block_table.clear() # 清空seq的block_table

    def can_append(self, seq: Sequence) -> bool:
        """
        剩余的block块的个数 >= （最后一个block的token数 == 1）
        只有取余后发现多出一个token的时候才需要再分配一个整块
        在decode执行之前会去判断
        """
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        """
        确定can_append了，才会执行may_append
        may_append是在每次decode之前做的准备
        """
        block_table = seq.block_table # 拿到当前seq的block_table
        last_block = self.blocks[block_table[-1]] # 拿到最后一个block
        if len(seq) % self.block_size == 1: # 如果取余发现多出一个token
            assert last_block.hash != -1 # 确保最后一个block是满的，有hash值
            block_id = self.free_block_ids[0] # 从空闲队列里取一个block_id
            self._allocate_block(block_id) # 分配新的block
            block_table.append(block_id) # 把新的block_id加到block_table里
        elif len(seq) % self.block_size == 0: # 如果最后一个满了，需要更新hash值
            assert last_block.hash == -1 # 确保最后一个block是没满，没有hash值
            token_ids = seq.block(seq.num_blocks-1) # 获取最后一个block的token_ids列表
            # 获取前一个block的hash值，如果没有前一个block，返回-1
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix) # 计算新的hash值
            last_block.update(h, token_ids) # 更新最后一个block的hash值
            self.hash_to_block_id[h] = last_block.block_id # 在字典里把新的hash值和block_id对应起来
        else:
            assert last_block.hash == -1 # 最后一个block没满，没有hash值，确认一下

```

## nanovllm/engine/llm_engine.py

```python
import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner

"""
LLMEngine是nano-vllm的入口类，负责初始化model、tokenizer、scheduler和model_runner等，并提供生成文本的接口。
"""

class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.ps = [] # 进程列表
        self.events = [] # config.tensor_parallel_size - 1 个event，传入主进程
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size): # 单卡不会进行以下循环操作，也就是不会开子进程
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event)) # 创建子进程，每个机器上都跑一个ModelRunner
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events) # 主进程model_runner对象，传入config，rank为0，会在其中加载模型权重
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True) # 创建tokenizer
        config.eos = self.tokenizer.eos_token_id # 设置结束符
        self.scheduler = Scheduler(config) # 初始化调度器
        atexit.register(self.exit) # 注册exit函数

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    # 添加请求到waiting的双端队列里
    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str): # 判断是否为str类型
            prompt = self.tokenizer.encode(prompt) # 将prompt转换为token_ids | str -> list[int]
        seq = Sequence(prompt, sampling_params) # 创建序列对象
        self.scheduler.add(seq) # 调用scheduler的add方法，将序列对象添加到等待队列中

    def step(self):
        """nano-vllm是prefill优先的思路，优先为所有序列做预填充，直到全部填充完，才去为序列做解码"""
        # 对序列进行调度，如果还有没做prefill的序列，则优先返回prefill的序列列表，同时is_prefill=True
        # 如果都做完prefill了，就返回decode的序列列表，同时is_prefill=False
        seqs, is_prefill = self.scheduler.schedule()
        # 将需要处理的seqs送到model_runner的run函数中处理，每个seq返回预测的一个token_id，组成token_ids
        token_ids = self.model_runner.call("run", seqs, is_prefill) # [seq_num] 每个seq生成一个新的token的id列表
        # 将生成的token_id添加到seq的token_ids列表中，并更新seq状态
        self.scheduler.postprocess(seqs, token_ids) # postprocess每次只处理每个seq的一个token
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished] # 获取已完成序列的seq_id和回答的token_ids
         # 计算一次step生成的总token数，正数说明是prefill生成的token，负数说明是decode生成的token，等于abs(len(seqs))说明是decode阶段，每次只生成一个token
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens # 返回完成序列的id和回答的token_ids，以及当前step生成的总token数

    def is_finished(self):
        return self.scheduler.is_finished() # 是否所有序列都完成

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        # 传入prompt列表 
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            # 复制多次sampling_params对象
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp) # prompt和sampling_params一一绑定后送到add_request函数中，添加到等待队列中
        outputs = {} # 用于存储完成序列的id和回答的token_ids
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished(): # 判断当前任务是否都完成（waiting和running队列都为空，则完成）
            t = perf_counter() # 记录当前时间
            output, num_tokens = self.step() # 执行任务，包括waiting和running里的所有序列
            # 返回完成序列的id和回答的token_ids（该列表可能为空），以及当前step生成的总token数
            # （num_tokens是正数说明是prefill阶段，负数说明是decode阶段）

            # 数据统计部分：prefill和decode的吞吐量
            if use_tqdm:
                if num_tokens > 0: # 正数，说明是prefill阶段
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else: # 负数，说明是decode阶段
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            # 如果有完成的seq，就更新完成的seq生成的回答的内容
            # 如果没有，说明当前step没有完成任何seq，继续下一个step，以下逻辑不会被执行
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        
        # 按seq_id排序，组成一个二维列表，每个元素代表一个已完成seq的回答的token_ids列表
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs] # 字典列表
        if use_tqdm:
            pbar.close()
        return outputs # 返回所有完成的seq的回答文本和token_ids

```

## nanovllm/engine/model_runner.py

```python
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
1 初始化：
初始化LLMEngine时，会为每张卡初始化一个ModelRunner
初始化ModelRunner时，会调用warmup_model() | allocate_kv_cache() | capture_cudagraph() if enforce_eager is False
warmup_model()会使用config设置的最大批次的模拟数据跑一边模型的前向计算
allocate_kv_cache()会计算出 基于当前显存 能分配的最大kv cache block数量，并分配一块连续显存空间，并为每层分配对应的block
capture_cudagraph()

2 前向计算：
llm_engine.generate() -> llm_engine.step() -> scheduler.schedule() -> block_manager.allocate()
以上操作会为传入的seqs分配对应的block，以及搜索seq可能的缓存前缀，加到seq.num_cached_tokens中
-> model_runner.run() -> model_runner.prepare_prefill() / prepare_decode()
-> model_runner.run_model() -> model_runner.sampler()
"""

class ModelRunner:
    """
    nano-vllm的模型运行类，负责模型的前向传播、下一token采样等。
    """
    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kv_cache_block_size # KV Cache块大小(256)
        self.enforce_eager = config.enforce_eager # 是否强制使用eager模式，不开启cuda_graph
        self.world_size = config.tensor_parallel_size # TP数
        self.rank = rank # 当前进程的rank
        self.event = event # 事件，用于同步不同进程之间的操作

        # 初始化进程组，使用nccl后端，通信地址为localhost:2333，进程数为world_size
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank) # 设置当前进程的GPU设备
        default_dtype = torch.get_default_dtype() # 默认数据类型：torch.float32
        torch.set_default_dtype(hf_config.dtype) # 设置默认数据类型为模型的torch_dtype：bfloat16，用于加载模型
        torch.set_default_device("cuda") # 设置默认设备为cuda

        # self.model = Qwen3ForCausalLM(hf_config) # 初始化模型
        self.model = model_dict[hf_config.model_type](hf_config) # 初始化模型
        load_model(self.model, config.model) # 根据不同组件的权重加载方法，加载模型权重
        self.sampler = Sampler() # 初始化采样器
        self.warmup_model() # 预跑一遍模型
        self.allocate_kv_cache() # 分配kv缓存，对应config.py中num_kvcache_blocks: int = -1
        if not self.enforce_eager: # enforce_eager为true，不会开cuda_graph
            self.capture_cudagraph()
        torch.set_default_device("cpu") # 默认设备设置回cpu，后续会从CPU 列表构建 Tensor 并异步传输到 GPU
        torch.set_default_dtype(default_dtype) # 默认数据类型设置回torch.float32

        if self.world_size > 1: # 单卡不会跑以下逻辑
            # 主进程把shm(shared memory)创建之后，子进程才能连接shm
            # 主进程往下执行调度
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20) # 主进程创建共享内存
                dist.barrier() # 等待所有进程都到达这一步
            else:
                dist.barrier() # 等待所有进程都到达这一步
                self.shm = SharedMemory(name="nanovllm") # 子进程连接共享内存
                self.loop() # 子进程进入循环

    def exit(self):
        """退出model_runner / 主进程关闭shm"""
        if self.world_size > 1:
            self.shm.close()
            dist.barrier() # 等待所有进程都到达这一步
            if self.rank == 0: # 主进程负责删除shm
                self.shm.unlink()
        if not self.enforce_eager: # 如果不是eager模式，删除cuda_graph相关的变量
            del self.graphs, self.graph_pool
        torch.cuda.synchronize() # 等待所有GPU操作完成
        dist.destroy_process_group() # 销毁进程组，释放资源

    def loop(self):
        """子进程循环，读取共享内存中的任务并执行"""
        while True:
            method_name, args = self.read_shm() # 读取共享内存中的任务，方法名
            self.call(method_name, *args) # 调用方法
            if method_name == "exit": # 如果方法名是"exit"，子进程退出循环
                break

    def read_shm(self):
        """子进程读取共享内存"""
        assert self.world_size > 1 and self.rank > 0 # 子进程读取共享内存
        self.event.wait() # 等待主进程的通知
        n = int.from_bytes(self.shm.buf[0:4], "little") # 从共享内存前4个字节获取数据长度
        method_name, *args = pickle.loads(self.shm.buf[4:n+4]) # 解析方法名和参数
        self.event.clear() # 清除标志位
        return method_name, args # 返回方法名和参数

    def write_shm(self, method_name, *args):
        """主进程写入共享内存"""
        assert self.world_size > 1 and self.rank == 0 # 只有主进程负责写入共享内存
        data = pickle.dumps([method_name, *args]) # 序列化方法名和参数
        n = len(data) # 获取序列化数据的长度
        self.shm.buf[0:4] = n.to_bytes(4, "little") # 将数据长度写入共享内存前4个字节
        self.shm.buf[4:n+4] = data # 将数据写入共享内存
        for event in self.event:
            event.set() # 通知子进程去读共享内存

    def call(self, method_name, *args):
        """
        在llm_engine.step()中调用，传入方法及参数
        主进程写入共享内存前，子进程处于等待状态，
        等主进程写入方法并调用set()通知子进程后，子进程解析出方法名和参数并清除标志位，
        然后调用对应方法，之后再次进入read_shm()等待下一次任务。
        主进程也需要调用对应方法：model_runner.run(seqs, is_prefill)
        """
        if self.world_size > 1 and self.rank == 0: # 主进程负责写入共享内存
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None) # 获取方法
        return method(*args) # 调用方法

    def warmup_model(self):
        """
        预热模型：用模拟数据让模型完整跑一遍前向流程，触发 CUDA 懒加载初始化
        模拟数据维度：(num_seqs, max_model_len)
        """
        torch.cuda.empty_cache() # 释放 PyTorch 不再使用、但被 CUDA 运行时缓存占用的《空闲显存》，将其归还给 GPU
        torch.cuda.reset_peak_memory_stats() # 重置显存峰值统计
        # max_num_batched_tokens：总token上限
        # max_model_len：单序列最大长度
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
        assert hf_config.hidden_size % hf_config.num_attention_heads == 0
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads) # 每个头的维度（这么写的原因是qwen2没有head_dim这个属性）
        # block_bytes：每个kv block占用的字节数 = 单个block存放的token数 * [(k + v) * attn层数 * kv头数 * 每个头的维度] * 数据类型大小
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
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
        """
        为prefix cache准备block表
        1. 找本 batch 里最长的 block_table 长度 max_len
        2. 把每条序列的 block_table 右侧补 -1 到 max_len（padding）
        3. 转成 GPU 上的 int32 tensor，形状大概是 (batch_size, max_len)    
        """
        max_len = max(len(seq.block_table) for seq in seqs) # 所有seq中的block_table个数最大值
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs] # 每个seq的block_table长度补齐到最大长度，不足的用-1填充
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True) # 将block_tables转换为Tensor
        return block_tables # (seq_num, max_num_blocks)

    def prepare_prefill(self, seqs: list[Sequence]):
        """
        把一批 seq 打平成一次前向要用的张量，同时构造“每个 token 的 KV 写入位置”
        为prefill准备的输入数据，用于flash_attn
            1 input_ids：所有序列 真正需要计算 的 token id 的拼接后的列表
            2 positions：所有序列 真正需要计算 的 token 在 seq 中的下标的拼接列表
            3 cu_seqlens_q：FlashAttention 专用的“累积长度”数组（Offset 数组）
            cu_seqlens_q[i] 表示第 i 个序列在拼接后的 input_ids 中的 起始下标
            4 cu_seqlens_k：意义同上，针对kv，但包含整个序列的长度（包括已缓存的token）
            5 因为k与v存在一起，因此只记k的下标即可
            6 为什么q可以少记录，但k的下标必须都记录？
                如果有前缀了，q_old就不用参与计算了，但是q_new还得和全部可见上下文的 k（旧的 + 新的）做注意力计算
            7 注意seq.block_table和block_tables的区别
        """
        input_ids = []
        positions = []
        # 给 flash-attention/变长批处理用，告诉 kernel 每个序列 query/key 的起止
        cu_seqlens_q = [0] # 每个序列在拼接后的 input_ids 中的起始和结束位置
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = [] # 新增token的物理位置
        block_tables = None
        
        for seq in seqs:
            # 此时seq.block_table已分配好（id值）
            seqlen = len(seq) # 当前序列长度
            input_ids.extend(seq[seq.num_cached_tokens:]) # （去掉当前序列中已经缓存的token）后的token_ids
            positions.extend(list(range(seq.num_cached_tokens, seqlen))) # extend需要计算的token在当前seq里的下标
            seqlen_q = seqlen - seq.num_cached_tokens # 当前序列中需要计算的部分长度
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q) # 加入q的累积长度
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k) # 加入k的累积长度
            max_seqlen_q = max(seqlen_q, max_seqlen_q) # seqs里需要新算的部分的最大长度
            max_seqlen_k = max(seqlen_k, max_seqlen_k) # seqs里最大长度
            if not seq.block_table: # warmup不需要slot_mapping 构建
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks): # (已使用的block数, 需要的总block数)
                # 为还没写入 kv cache  的 block 生成逐 token 的物理位置映射，每一块都有一个start和一个end
                start = seq.block_table[i] * self.block_size # 起始索引，因为prefix cache都是一块一块的，所以start肯定是从某一块的第一个开始
                if i != seq.num_blocks - 1:
                    end = start + self.block_size # 不是最后一个，就一块一块的加
                else:
                    # 最后一个block可能未填满，用 seq.last_block_num_tokens 精确到实际 token 数
                    end = start + seq.last_block_num_tokens
                # slot_mapping是每个token的kv cache的物理槽位映射，一个token就是一个slot,
                slot_mapping.extend(list(range(start, end))) # 给当前seq这段要处理的 token，逐个分配它们在KV cache里的物理槽位 index
        
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]: # 有前缀需要从KV cache读取，没有就是全新的，只有有seq有前缀缓存，就一起送给attn的kernel
            block_tables = self.prepare_block_tables(seqs) # 为kernel准备的所有seq的block_table，都是block id值
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True) # 从 CPU 列表构建 Tensor 并异步传输到 GPU
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables) # 设置全局变量，供flash_attn使用
        return input_ids, positions # 返回input_ids和positions，做前向运算，输出下一token（也就是prefill完的第一个token）

    def prepare_decode(self, seqs: list[Sequence]):
        """为decode输出做准备，用于flash_attn_with_kvcache，不再需要cu_seqlens_q和cu_seqlens_k"""
        input_ids = [] # 每个seq最后一个token_id的列表
        positions = [] # 每个seq最后一个token的位置索引的列表
        slot_mapping = []
        context_lens = [] # 每个seq长度的列表
        for seq in seqs:
            input_ids.append(seq.last_token) # 每个seq的最后一个token_id
            positions.append(len(seq) - 1) # 每个seq的最后一个token的位置索引
            context_lens.append(len(seq)) # 每个seq的长度
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs) # 同prepare_prefill，给到kernel前缀的block id值，用于取kv cache
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        """采样参数（温度）准备"""
        temperatures = [] # 采样温度列表
        for seq in seqs:
            temperatures.append(seq.temperature) # 添加不同seq的采样温度
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        """计算模型输出下一个token的logits"""
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            # input_ids: [total_tokens]
            # positions: [total_tokens]
            # 遇到以下三种情况：1）prefill阶段，
            # 2）enforce_eager为true（prefill / decode都可），
            # 3）seq_num大于512，直接走普通调用
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            # input_ids: [seq_num]
            # positions: [seq_num]
            # decode阶段且enforce_eager为false：
            #   只传入每个seq的最后一个token_id和位置索引，kv_cache由context.get_context()提供
            bs = input_ids.size(0) # bs = seq_num
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)] # 对当前seq_num向上取整
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
        来自：llm_engine.py 
            token_ids = self.model_runner.call("run", seqs, is_prefill)
        计算logits -> 采样token_id -> 重置kv状态 -> 返回seqs的新生成token_id列表
        """
        # 根据is_prefill标识符为prefill / decode 准备输入数据，返回
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs) # 拿到输入数据，给到run_model
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None # 采样温度
        logits = self.run_model(input_ids, positions, is_prefill) # 模型前向计算，返回logits
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None # 采样token_id，只会用主进程来做，子进程回到loop()下一轮
        reset_context() # 重置全局变量
        return token_ids # 返回生成的token_ids列表

    @torch.inference_mode()
    def capture_cudagraph(self):
        """针对decode阶段进行优化"""
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512) # 单次推理最大的seq数
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        # 开辟好cuda graph需要的全部最大空间
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16)) # 预定义不同批量大小
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs): # 倒叙遍历，先开辟最大的
            graph = torch.cuda.CUDAGraph()
            set_context(
                False, # False表示decode阶段
                slot_mapping=slot_mapping[:bs], 
                context_lens=context_lens[:bs], 
                block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs]) # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs]) # capture
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

```

## nanovllm/engine/scheduler.py

```python
from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:
    """
    调度prefill和decode阶段，管理序列的kv cache block块的分配和释放
    """
    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs # 最大并行的序列数
        self.max_num_batched_tokens = config.max_num_batched_tokens # 最大总token数
        self.eos = config.eos # 结束符
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kv_cache_block_size) # kv cache块管理器
        self.waiting: deque[Sequence] = deque() # 等待队列里的seq
        self.running: deque[Sequence] = deque() # 运行队列里的seq

    def is_finished(self):
        """若等待队列和运行队列都为空，则所有序列都已完成"""
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        """将序列添加到等待队列"""
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """调度prefill和decode阶段，管理序列的kv cache block块的分配和释放"""
        scheduled_seqs = [] # 准备做prefill的序列列表
        num_seqs = 0 # 打算做prefill的序列数量
        num_batched_tokens = 0 # 当前batch的累计token数

        # prefill
        # 等待队列不为空（还有需要做prefill的）且准备做prefill的序列数小于最大并行序列数
        while self.waiting and num_seqs < self.max_num_seqs:
            # 不断去waiting队列里取出序列加到prefill列表
            seq = self.waiting[0] # 取等待队列的第一个序列
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                # 如果没有足够block块提供给当前序列，或者当前batch的累计token数超过了最大总token数，则停止加入到prefill
                break
            num_seqs += 1 # 准备做prefill的序列数加1
            self.block_manager.allocate(seq) # 给当前序列分配block块
            num_batched_tokens += len(seq) - seq.num_cached_tokens # 当前batch的累计token数加上当前序列的token数减去已缓存的token数
            seq.status = SequenceStatus.RUNNING # 将当前序列的状态设置为RUNNING
            self.waiting.popleft() # 从等待队列中移除当前序列
            self.running.append(seq) # 将当前序列添加到运行队列,running是所有在运行的序列
            scheduled_seqs.append(seq) # 将当前序列添加到准备做prefill的序列列表，scheduled_seqs是当前batch的准备运行的序列列表
        if scheduled_seqs: # 如果prefill列表不为空
            return scheduled_seqs, True # 返回prefill列表和True

        # decode
        # 运行队列不为空且准备做decode的序列数小于最大并行序列数
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft() # 从运行队列中取出第一个序列
            while not self.block_manager.can_append(seq): # 如果当前block数不够了
                if self.running: # 如果运行队列不为空
                    self.preempt(self.running.pop()) # 从运行队列队尾拿出最后一个序列移到等待队列，并释放对应的blocks
                else:
                    self.preempt(seq) # 如果运行队列为空，则将当前序列移到等待队列，并释放对应的blocks
                    break
            else:
                num_seqs += 1 # 准备做decode的序列数加1
                self.block_manager.may_append(seq) # 如果seq当前block数不够了，则为seq分配一个新的block块
                scheduled_seqs.append(seq) # 将当前序列添加到decode列表
        assert scheduled_seqs # scheduled_seqs列表不为空
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False # 返回decode列表和False

    def preempt(self, seq: Sequence):
        """将序列从运行队列移到等待队列，并释放对应的blocks"""
        seq.status = SequenceStatus.WAITING # 输入序列的状态设置为WAITING
        self.block_manager.deallocate(seq) # 释放输入序列的block块
        self.waiting.appendleft(seq) # 将输入序列添加到等待队列

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        """更新每个seq的token_ids列表，并更新seq状态"""
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id) # 将生成的token_id添加到seq的token_ids列表
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                # 如果seq的token_id是结束标识，或者seq的token_ids列表的长度达到了最大token数，则将seq的状态设置为FINISHED
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq) # 释放seq的block块
                self.running.remove(seq) # 从运行队列中移除seq

```

## nanovllm/engine/sequence.py

```python
from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    """seq状态"""
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    """管理任意一条序列的状态，包括token_ids列表，kv cache块表，已缓存的token数等"""
    block_size = 256 # kv cache块大小
    counter = count() # 序列id计数器

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter) # 序列id
        self.status = SequenceStatus.WAITING # 序列状态，初始化为waiting
        self.token_ids = copy(token_ids) # 序列的token_ids列表
        self.last_token = token_ids[-1] # 最后一个token的id
        self.num_tokens = len(self.token_ids) # 序列长度
        self.num_prompt_tokens = len(token_ids) # prompt的token数
        self.num_cached_tokens = 0 # 缓存的token数
        self.block_table = [] # block表，用于记录kv cache块的id
        self.temperature = sampling_params.temperature # 温度，seq之间可不同
        self.max_tokens = sampling_params.max_tokens # 单条样本的最大token数
        self.ignore_eos = sampling_params.ignore_eos # 是否忽略EOS

    def __len__(self):
        return self.num_tokens # 返回序列长度

    def __getitem__(self, key):
        return self.token_ids[key] # 返回序列的token_ids列表中的第key个token

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED # 返回是否完成生成

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens # 返回生成的token数

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens] # 返回prompt的token_ids列表

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:] # 返回回答的token_ids列表

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size  # 返回已使用的kv cache blocks数量

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size # 当前序列需要的总block数

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size # 最后一个block实际的token数（不足一整块）

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size] # 返回第i个block的token_ids

    def append_token(self, token_id: int):
        """追加新的token_id到token_ids列表，更新最后一个token的token_id，并更新总token数"""
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]

```
