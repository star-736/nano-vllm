from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 1.0 # 温度
    max_tokens: int = 64 # 单条样本最大token数
    ignore_eos: bool = False # 是否忽略EOS

    def __post_init__(self): # 实例化后自动调用
        assert self.temperature > 1e-10, "greedy sampling is not permitted" # 温度必须大于1e-10
