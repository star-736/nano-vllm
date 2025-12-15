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


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.ps = [] # 进程列表
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size): # 单卡不会进行以下循环操作，也就是不会开子进程
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events) # 创建model_runner对象，传入config，rank为0，会在其中加载模型权重
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
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids) # 将生成的token_id添加到seq的token_ids列表中，postprocess每次只处理一个token
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished] # 获取已完成序列的seq_id和回答的token_ids
         # 计算一次step生成的总token数，正数说明是prefill生成的token，负数说明是decode生成的token，等于len(seqs)说明decode每次只生成一个token
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens # 返回完成序列的id和回答的token_ids以及当前step生成的总token数

    def is_finished(self):
        return self.scheduler.is_finished() # 是否所有序列都完成

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        # 传入多条prompt 
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            # 复制多次sampling_params对象
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp) # prompt和sampling_params一一绑定后送到add_request函数中
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished(): # 判断当前任务是否都完成（waiting和running队列都为空，则完成）
            t = perf_counter()
            output, num_tokens = self.step() # 执行任务，包括waiting和running里的所有序列
            # 返回完成序列的id和回答的token_ids以及当前step生成的总token数

            # 数据统计部分
            if use_tqdm:
                if num_tokens > 0: # 如果为正数，说明是prefill阶段
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else: # 如果为负数，说明是decode阶段
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
    
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
