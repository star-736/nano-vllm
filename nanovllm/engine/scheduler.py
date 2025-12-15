from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs # 最大并行的序列数
        self.max_num_batched_tokens = config.max_num_batched_tokens # 最大总token数
        self.eos = config.eos # 结束符
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size) # kv cache块管理器
        self.waiting: deque[Sequence] = deque() # 等待队列里的seq
        self.running: deque[Sequence] = deque() # 运行队列里的seq

    def is_finished(self):
        return not self.waiting and not self.running # 等待队列和运行队列都为空，则所有序列都已完成

    def add(self, seq: Sequence):
        self.waiting.append(seq) # 将序列添加到等待队列

    def schedule(self) -> tuple[list[Sequence], bool]:
        # prefill
        scheduled_seqs = [] # 准备做prefill的序列列表
        num_seqs = 0 # 打算做prefill的序列数量
        num_batched_tokens = 0 # 当前batch的累计token数
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
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
