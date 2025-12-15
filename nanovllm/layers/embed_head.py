import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):
    """
    token id -> embedding
    """
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank() # 当前GPU编号 
        self.tp_size = dist.get_world_size() # TP值
        assert num_embeddings % self.tp_size == 0 # vocab_size可以被整除
        self.num_embeddings = num_embeddings # vocab_size
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size # 每个GPU的vocab_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank # 当前GPU的起始索引
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition # 当前GPU的结束索引
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim)) # 分配权重参数
        self.weight.weight_loader = self.weight_loader # 权重加载方法

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """分布式加载权重"""
        param_data = param.data # (num_embeddings, embedding_dim)
        shard_size = param_data.size(0) # num_embeddings/2，param_data.shape = self.weight.shape
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size) # (num_embeddings/2, embedding_dim)
        # 若tp_size为2，则
        # GPU 0: 处理 [0 : vocab_size/2]行
        # GPU 1: 处理 [vocab_size/2 : vocab_size]行
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1: # weights分布在多卡上
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)

        y = F.embedding(x, self.weight) # 单卡直接计算并返回

        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):
    """
    embedding -> logit
    和VocabParallelEmbedding一样分布式加载权重
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, bias: bool = False):
        assert not bias # Qwen3默认没有bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        context = get_context() # 从全局变量中获取context.is_prefill，cu_seqlens_q
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        logits = F.linear(x, self.weight)

        if self.tp_size > 1: # TP all-gather
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None # 在GPU 0上创建一个和logits形状相同的空张量列表，其他GPU上为None
            dist.gather(logits, all_logits, 0) # 在GPU 0上收集所有GPU的logits
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None # 只在GPU 0上进行cat
        return logits
