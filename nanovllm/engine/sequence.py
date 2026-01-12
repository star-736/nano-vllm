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
