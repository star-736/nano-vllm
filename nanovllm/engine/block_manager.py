from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence

"""
重点解析：
    prefix cache：
        复用不同序列之间的相同前缀的kv cache block，但必须是放满的block
        在allocate(seq)中去尝试去复用缓存的block，如果找不到，则分配新的block
"""

class Block:
    """用于存储每个block的信息"""
    def __init__(self, block_id):
        self.block_id = block_id # 当前block的id
        self.ref_count = 0 # 引用的次数
        self.hash = -1 # hash值
        self.token_ids = [] # 该block里的token_ids列表

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash # block满了以后，会更新hash值
        self.token_ids = token_ids # 更新token_ids列表

    def reset(self):
        self.ref_count = 1 # 重置引用次数
        self.hash = -1 # 重置hash值
        self.token_ids = [] # 重置token_ids列表


class BlockManager:
    """用于管理所有block的分配和释放"""
    def __init__(self, num_blocks: int, block_size: int, num_draft_blocks: int = 0, speculative_decoding: bool = False, num_speculative_tokens: int = 0):
        assert num_blocks > 0
        self.block_size = block_size # 每个block的大小
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)] # 总block列表，num_blocks是传入的可用block数量
        self.hash_to_block_id: dict[int, int] = dict() # hash值到block_id的映射
        self.free_block_ids: deque[int] = deque(range(num_blocks)) # 可用的block_id集合，是个双向队列
        self.used_block_ids: set[int] = set() # 已使用的block_id集合，不能重复

        self.speculative_decoding = speculative_decoding
        self.num_speculative_tokens = num_speculative_tokens

        self.draft_blocks = None
        self.hash_to_draft_block_id: dict[int, int] = dict()
        self.free_draft_block_ids = None
        self.used_draft_block_ids = None
        if self.speculative_decoding and num_draft_blocks > 0:
            self.draft_blocks: list[Block] = [Block(i) for i in range(num_draft_blocks)]
            self.free_draft_block_ids: deque[int] = deque(range(num_draft_blocks))
            self.used_draft_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """
        一个block放满了，才计算hash值，
        hash可以快速比对两个block是否相同
        """
        h = xxhash.xxh64()
        if prefix != -1: # 有前缀，会把前缀加到hash值里
            h.update(prefix.to_bytes(8, "little"))
        # 若prefix为-1，则不添加任何前缀
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        """
        分配一个新block，
        确保 block 未被占用，重置后从空闲 队列转入已用集合并返回
        """
        block = self.blocks[block_id] # 获取block
        assert block.ref_count == 0 # 确保block没用过
        block.reset() # 初始化block
        self.free_block_ids.remove(block_id) # 从队列中移除
        self.used_block_ids.add(block_id) # 加到已用集合
        return self.blocks[block_id] # 返回block

    def _allocate_draft_block(self, block_id: int) -> Block:
        block = self.draft_blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_draft_block_ids.remove(block_id)
        self.used_draft_block_ids.add(block_id)
        return self.draft_blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        """
        释放一个block，
        确保引用数为 0，把 block 从已用集合放回空闲队列
        """
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def _deallocate_draft_block(self, block_id: int) -> Block:
        assert self.draft_blocks[block_id].ref_count == 0
        self.used_draft_block_ids.remove(block_id)
        self.free_draft_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        """
        判断空闲block数量是否能覆盖传入序列需要的block数
        在scheduler里会有判断
        """
        if not self.speculative_decoding:
            return len(self.free_block_ids) >= seq.num_blocks
        return len(self.free_block_ids) >= seq.num_blocks and len(self.free_draft_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        """
        为seq分配block
        在prefill阶段执行，只会执行一次
        """
        assert not seq.block_table # 确保seq的block_table为空，即第一次分配blocks
        h = -1 # 初始化hash值
        cache_miss = False # 初始化缓存miss标志
        for i in range(seq.num_blocks): # 遍历seq所需的block数
            token_ids = seq.block(i) # 获取seq当前block的token_ids列表
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1 # 如果一个block放满了，计算hash值
            block_id = self.hash_to_block_id.get(h, -1) # 去字典里找hash值对应的block_id，如果找不到，返回-1
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                # 如果没找到，或者找到了但是token_ids列表不一样（不同的token_ids算出了同一个hash），说明缓存miss
                cache_miss = True          
            if cache_miss: # 如果缓存miss了，需要分配新的block
                block_id = self.free_block_ids[0] # 从空闲队列里取一个block_id
                block = self._allocate_block(block_id) # 返回新分配的block
            else: # 如果缓存hit了，直接使用block
                seq.num_cached_tokens += self.block_size # 更新seq的缓存token总数
                if block_id in self.used_block_ids:
                    # 如果block_id在已用集合里，说明这个block之前被分配过，直接指向对应的block，同时需要增加引用次数
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    # 说明只是查字典查到了，但之前被释放了，只是字典里没有清除
                    # block_id不在已用集合里，调用_allocate_block分配新的block
                    block = self._allocate_block(block_id) 
            if h != -1: # 如果hash值不是-1，说明这个block是满的，把compute_hash里计算的hash值赋值进去
                # 如果是-1，分配完新的block，不走后续逻辑
                block.update(h, token_ids) # 赋值hash值
                self.hash_to_block_id[h] = block_id # 在字典里把hash值和block_id对应起来
            seq.block_table.append(block_id) # 把block_id加到seq的block_table里
        
        # Allocate draft blocks
        if self.speculative_decoding:
            assert not seq.draft_block_table
            h = -1
            cache_miss = False
            for i in range(seq.num_blocks):
                token_ids = seq.block(i)
                h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
                block_id = self.hash_to_draft_block_id.get(h, -1)
                if block_id == -1 or self.draft_blocks[block_id].token_ids != token_ids:
                    cache_miss = True
                if cache_miss:
                    block_id = self.free_draft_block_ids[0]
                    block = self._allocate_draft_block(block_id)
                else:
                    if block_id in self.used_draft_block_ids:
                        block = self.draft_blocks[block_id]
                        block.ref_count += 1
                    else:
                        block = self._allocate_draft_block(block_id)
                if h != -1:
                    block.update(h, token_ids)
                    self.hash_to_draft_block_id[h] = block_id
                seq.draft_block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        """为seq释放block"""
        for block_id in reversed(seq.block_table): # 从后往前遍历seq的block_table
            block = self.blocks[block_id] # 取出block
            block.ref_count -= 1 # 减少引用次数
            if block.ref_count == 0: # 如果引用次数为0，说明这个block可以释放了
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0 # 清零seq的缓存token总数
        seq.block_table.clear() # 清空seq的block_table

        # Deallocate draft blocks
        if self.speculative_decoding:
            for block_id in reversed(seq.draft_block_table):
                block = self.draft_blocks[block_id]
                block.ref_count -= 1
                if block.ref_count == 0:
                    self._deallocate_draft_block(block_id)
            seq.draft_block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """
        剩余的block块的个数 >= （最后一个block的token数 == 1）
        只有取余后发现多出一个token的时候才需要再分配一个整块
        在decode执行之前会去判断
        """
        # return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)
        if self.speculative_decoding:
            target_len = len(seq) + self.num_speculative_tokens
            needed_blocks = (target_len + self.block_size - 1) // self.block_size
            required_blocks = max(0, needed_blocks - len(seq.block_table))
            return len(self.free_block_ids) >= required_blocks and len(self.free_draft_block_ids) >= required_blocks
        else:
            required_blocks = 1 if (len(seq) % self.block_size == 1) else 0
            return len(self.free_block_ids) >= required_blocks

    def may_append(self, seq: Sequence):
        """
        确定can_append了，才会执行may_append
        may_append是在每次decode之前做的准备
        """
        block_table = seq.block_table # 拿到当前seq的block_table
        draft_block_table = seq.draft_block_table
        last_block = self.blocks[block_table[-1]] # 拿到最后一个block

        if self.speculative_decoding:
            need_new_block = (len(seq) + self.num_speculative_tokens > len(block_table) * self.block_size)
            if need_new_block:
                if last_block.hash == -1:
                    token_ids = seq.block(seq.num_blocks - 1)
                    if len(token_ids) == self.block_size:
                        prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
                        h = self.compute_hash(token_ids, prefix)
                        last_block.update(h, token_ids)
                        self.hash_to_block_id[h] = last_block.block_id
                    else:
                        assert last_block.hash == -1

                main_block_id = self.free_block_ids[0]
                self._allocate_block(main_block_id)
                block_table.append(main_block_id)
                last_block = self.blocks[block_table[-1]]

                # draft block
                draft_block_id = self.free_draft_block_ids[0]
                self._allocate_draft_block(draft_block_id)
                draft_block_table.append(draft_block_id)
            else:
                assert last_block.hash == -1
        else:
            if len(seq) % self.block_size == 1:  # a new block needs to be allocated
                assert last_block.hash != -1 # 确保最后一个block是满的，有hash值
                block_id = self.free_block_ids[0] # 从空闲队列里取一个block_id
                self._allocate_block(block_id) # 分配新的block
                block_table.append(block_id) # 把新的block_id加到block_table里
            elif len(seq) % self.block_size == 0:  # the last block gets finalized with a hash
                assert last_block.hash == -1 # 确保最后一个block是没满，没有hash值
                token_ids = seq.block(seq.num_blocks - 1) # 获取最后一个block的token_ids列表
                # 获取前一个block的hash值，如果没有前一个block，返回-1
                prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
                h = self.compute_hash(token_ids, prefix) # 计算新的hash值
                last_block.update(h, token_ids) # 更新最后一个block的hash值
                self.hash_to_block_id[h] = last_block.block_id # 在字典里把新的hash值和block_id对应起来
            else:
                assert last_block.hash == -1 # 最后一个block没满，没有hash值，确认一下
