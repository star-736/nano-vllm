import torch
from torch import nn


class Sampler(nn.Module):
    """
    用于从logits中采样下一个token
    在model_runner.py中被调用
    """
    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        """
        logits: [batch_size, vocab_size]
        temperatures: [batch_size]
        """
        logits = logits.float().div_(temperatures.unsqueeze(dim=1)) # 温度缩放
        probs = torch.softmax(logits, dim=-1) # softmax转换为概率
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1) # 指数分布采样
        return sample_tokens # 返回采样后的token [batch_size]
