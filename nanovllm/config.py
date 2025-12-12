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
    kvcache_block_size: int = 256 # KV Cache块大小
    num_kvcache_blocks: int = -1 # 自动计算需要的KV Cache块数

    def __post_init__(self): # 实例化后自动调用
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0 # KV Cache块大小必须是256的倍数
        assert 1 <= self.tensor_parallel_size <= 8 # 张量并行大小必须在1到8之间
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings) # 最大序列长度不能超过模型最大位置嵌入长度
        assert self.max_num_batched_tokens >= self.max_model_len # 批处理的总token上限至少能容纳一个最大长度的序列，目前16384>4096
