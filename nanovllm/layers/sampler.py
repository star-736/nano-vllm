import torch
from torch import nn


class Sampler(nn.Module):
    """
    用于从logits中采样下一个token
    在model_runner.py中被调用
    """
    def __init__(self):
        super().__init__()

    def compute_temperature_scaled_probs(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.to(torch.float)
        safe_temperatures = torch.where(temperatures == 0, torch.ones_like(temperatures), temperatures)
        logits.div_(safe_temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        return probs

    def sample_from_probs(self, probs: torch.Tensor, temperatures: torch.Tensor):
        probs = probs.to(torch.float)
        greedy_tokens = probs.argmax(dim=-1)
        sample_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        return torch.where(temperatures == 0, greedy_tokens, sample_tokens)

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor, return_probs: bool = False):
        logits = logits.to(torch.float)
        greedy_tokens = logits.argmax(dim=-1)
        safe_temperatures = torch.where(temperatures == 0, torch.ones_like(temperatures), temperatures)
        logits.div_(safe_temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        sample_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        tokens = torch.where(temperatures == 0, greedy_tokens, sample_tokens)
        if return_probs:
            return tokens, probs
        return tokens
