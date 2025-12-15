import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):
    """SwiGLU activation function"""

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1) # 把gate_up_proj的输出拆分成gate_proj和up_proj两部分
        return F.silu(x) * y # silu激活函数
