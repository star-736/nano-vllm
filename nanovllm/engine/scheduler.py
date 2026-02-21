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
        self.enable_chunked_prefill = config.enable_chunked_prefill
        self.speculative_decoding = config.speculative_model is not None and config.num_speculative_tokens > 0
        self.num_speculative_tokens = config.num_speculative_tokens
        self.block_manager = BlockManager(
            config.num_kvcache_blocks,
            config.kv_cache_block_size,
            num_draft_blocks=config.num_draft_kvcache_blocks,
            speculative_decoding=self.speculative_decoding,
            num_speculative_tokens=self.num_speculative_tokens
        ) # kv cache块管理器
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
        if self.enable_chunked_prefill:
            return self._chunked_prefill_schedule()
        else:
            return self._default_schedule()
    
    def _default_schedule(self) -> tuple[list[Sequence], bool]:
        # prefill
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
            seq.num_tokens_to_process = len(seq) - seq.num_processed_tokens
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
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False # 返回decode列表和False

    def _chunked_prefill_schedule(self) -> tuple[list[Sequence], bool]:
        token_budget = self.max_num_batched_tokens

        # prefill
        temp_waiting = deque()
        scheduled_seqs = []
        num_seqs = 0
        while self.waiting and token_budget > 0 and num_seqs < self.max_num_seqs:
            seq = self.waiting.popleft()

            if (not seq.block_table) and (not self.block_manager.can_allocate(seq)):
                temp_waiting.append(seq)
                continue

            if not seq.block_table:
                self.block_manager.allocate(seq)

            prompt_tokens_left = len(seq) - seq.num_processed_tokens
            assert prompt_tokens_left > 0
            if prompt_tokens_left <= token_budget:
                seq.status = SequenceStatus.RUNNING
                scheduled_seqs.append(seq)
                num_seqs += 1
                seq.num_tokens_to_process = prompt_tokens_left
                token_budget -= prompt_tokens_left
                self.running.append(seq)
            else:
                # Chunk the prompt
                chunk_size = token_budget
                seq.status = SequenceStatus.RUNNING
                scheduled_seqs.append(seq)
                num_seqs += 1
                seq.num_tokens_to_process = chunk_size
                token_budget = 0
                # temp_waiting.append(seq)

        if temp_waiting:
            self.waiting.extendleft(reversed(temp_waiting))

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and token_budget >= 1 and num_seqs < self.max_num_seqs:
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
                token_budget -= 1
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        """将序列从运行队列移到等待队列，并释放对应的blocks"""
        seq.status = SequenceStatus.WAITING # 输入序列的状态设置为WAITING
        self.block_manager.deallocate(seq) # 释放输入序列的block块
        self.waiting.appendleft(seq) # 将输入序列添加到等待队列

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        """更新每个seq的token_ids列表，并更新seq状态"""
        def reach_end(seq: Sequence):
            return (not seq.ignore_eos and seq.last_token == self.eos) or seq.num_completion_tokens == seq.max_tokens

        for seq, token_id in zip(seqs, token_ids):
            if seq.is_speculative:  # Speculative decoding
                accepted_count = len(seq.pending_accepted_tokens)
                if accepted_count > 0:
                    end_index = -1
                    for i, t in enumerate(seq.pending_accepted_tokens):
                        if (not seq.ignore_eos and t == self.eos) or (seq.num_completion_tokens - self.num_speculative_tokens + i + 1 == seq.max_tokens):
                            end_index = i
                            break
                    if end_index != -1:
                        to_pop = self.num_speculative_tokens - (end_index + 1)
                        if to_pop > 0:
                            seq.pop_last_n_tokens(to_pop)
                        seq.pending_accepted_tokens.clear()
                        seq.status = SequenceStatus.FINISHED
                        self.block_manager.deallocate(seq)
                        if seq in self.running:
                            self.running.remove(seq)
                        continue

                seq.num_processed_tokens += 1 + accepted_count
                if accepted_count < self.num_speculative_tokens:
                    to_pop = self.num_speculative_tokens - accepted_count
                    seq.pop_last_n_tokens(to_pop)
                    seq.draft_num_processed_tokens = seq.num_processed_tokens
                else:  # the last draft token is not processed by draft model
                    seq.draft_num_processed_tokens = seq.num_processed_tokens - 1
                seq.pending_accepted_tokens.clear()

                # append next token
                seq.append_token(token_id)
                if reach_end(seq):
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq)
                    if seq in self.running:
                        self.running.remove(seq)
                # seq.is_speculative = False
            elif seq.num_tokens_to_process is not None:  # Prefill
                seq.num_processed_tokens += seq.num_tokens_to_process
                seq.num_tokens_to_process = None  # Reset for next iteration
                if seq.num_processed_tokens >= seq.num_prompt_tokens:  # Move to decode phase
                    seq.status = SequenceStatus.RUNNING
                    if seq in self.waiting:
                        self.waiting.remove(seq)
                    if seq not in self.running:
                        self.running.append(seq)

                    seq.append_token(token_id)
                    if reach_end(seq):
                        seq.status = SequenceStatus.FINISHED
                        self.block_manager.deallocate(seq)
                        self.running.remove(seq)
                else:  # Still in prefill phase
                    seq.status = SequenceStatus.WAITING
                    if seq in self.running:
                        self.running.remove(seq)
                    if seq not in self.waiting:
                        self.waiting.append(seq)
            else:  # Naive decoding
                seq.append_token(token_id) # 将生成的token_id添加到seq的token_ids列表
                if reach_end(seq):
                    # 如果seq的token_id是结束标识，或者seq的token_ids列表的长度达到了最大token数，则将seq的状态设置为FINISHED
                    seq.status = SequenceStatus.FINISHED
                    self.block_manager.deallocate(seq) # 释放seq的block块
                    self.running.remove(seq) # 从运行队列中移除seq
