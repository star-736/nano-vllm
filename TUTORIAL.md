# Nano-vLLM 入门教程

## 概述

Nano-vLLM 是一个轻量级的 vLLM 实现，从零开始构建，代码简洁易读（约 1200 行 Python 代码），同时保持了与 vLLM 相当的推理速度。本教程将带你了解整个推理流程的架构和实现细节。

## 项目结构

```
nano-vllm/
├── nanovllm/                    # 核心库
│   ├── __init__.py             # 导出 LLM 和 SamplingParams
│   ├── llm.py                  # LLM 类入口
│   ├── sampling_params.py      # 采样参数配置
│   ├── config.py               # 系统配置
│   ├── engine/                 # 推理引擎核心
│   │   ├── llm_engine.py       # 主引擎逻辑
│   │   ├── model_runner.py     # 模型执行器
│   │   ├── scheduler.py        # 请求调度器
│   │   ├── sequence.py         # 序列管理
│   │   └── block_manager.py    # KV Cache 块管理
│   ├── models/                 # 模型实现
│   │   └── qwen3.py            # Qwen3 模型架构
│   ├── layers/                 # 神经网络层
│   │   ├── attention.py        # 注意力机制
│   │   ├── sampler.py          # 采样器
│   │   ├── activation.py       # 激活函数
│   │   ├── layernorm.py        # 层归一化
│   │   ├── linear.py           # 线性层
│   │   ├── rotary_embedding.py # 旋转位置编码
│   │   └── embed_head.py       # 嵌入层和输出头
│   └── utils/                  # 工具函数
│       ├── context.py          # 全局变量管理
│       └── loader.py           # 模型加载
├── example.py                  # 使用示例
├── bench.py                    # 性能基准测试
└── pyproject.toml             # 项目配置
```

## 入口点

### 1. 用户接口（`nanovllm/llm.py`）

用户通过 `LLM` 类与系统交互，它继承自 `LLMEngine`：

```python
from nanovllm import LLM, SamplingParams

llm = LLM("/path/to/model", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
```

### 2. 配置系统（`nanovllm/config.py`）

`Config` 类管理所有系统参数：

- `max_num_batched_tokens`: 最大批处理 总token 数（默认 16384）
- `max_num_seqs`: 最大并发序列数（默认 512）
- `max_model_len`: 最大序列长度（默认 4096）
- `gpu_memory_utilization`: GPU 内存利用率（默认 0.9）
- `tensor_parallel_size`: 张量并行大小（默认 1）
- `enforce_eager`: 是否强制使用 eager 模式（默认 False）
- `kvcache_block_size`: KV Cache 块大小（默认 256）

## 推理流程详解

### 阶段 1: 初始化（`LLMEngine.__init__`）

1. **配置加载**: 解析用户传入参数，创建 `Config` 对象
2. **模型并行初始化**: 
   - 如果 `tensor_parallel_size > 1`，创建多个进程
   - 每个进程初始化一个 `ModelRunner` 实例
   - 使用 NCCL 进行进程间通信
3. **Tokenizer 加载**: 加载 HuggingFace tokenizer
4. **调度器创建**: 创建 `Scheduler` 管理请求队列

### 阶段 2: 请求添加（`LLMEngine.add_request`）

1. **文本编码**: 如果是字符串 prompt，使用 tokenizer 编码为 token IDs
2. **序列创建**: 创建 `Sequence` 对象，包含：
   - token IDs 列表
   - 采样参数（温度、最大长度等）
   - 序列状态（WAITING/RUNNING/FINISHED）
   - kv cache 块表（block table）用于 KV Cache 管理

### 阶段 3: 调度（`Scheduler.schedule`）

调度器采用两阶段策略：

#### Prefill 阶段（首次执行）

1. **从等待队列选择序列**：
   - 检查 batch 大小限制（`max_num_seqs`）
   - 检查 token 数限制（`max_num_batched_tokens`）
   - 检查 KV Cache 块是否足够
2. **分配 KV Cache 块**：
   - 调用 `BlockManager.allocate()`
   - 使用前缀缓存（prefix caching）优化重复文本
   - 计算块哈希，复用相同内容的块
3. **更新序列状态**：WAITING → RUNNING

#### Decode 阶段（迭代生成）

1. **从运行队列选择序列**：
   - 检查是否需要抢占（preemption）
   - 如果 KV Cache 不足，抢占低优先级序列
2. **追加 KV Cache 块**：
   - 如果当前块已满，分配新块
   - 更新块哈希表

### 阶段 4: 模型执行（`ModelRunner.run`）

#### Prefill 准备（`prepare_prefill`）

1. **输入构建**：
   - `input_ids`: 所有序列的新 token（未缓存部分）
   - `positions`: 每个 token 的位置
   - `cu_seqlens_q/k`: 累积序列长度（用于 flash attention）
   - `slot_mapping`: token 到 KV Cache 槽位的映射
   - `block_tables`: 块表（用于 prefix cache）

2. **上下文设置**：
   - 设置 `is_prefill=True`
   - 存储 attention 所需的元数据

#### Decode 准备（`prepare_decode`）

1. **输入构建**：
   - `input_ids`: 每个序列的最后一个 token
   - `positions`: 每个序列的当前位置
   - `slot_mapping`: 新 token 的 KV Cache 槽位
   - `context_lens`: 每个序列的上下文长度
   - `block_tables`: 块表

2. **上下文设置**：
   - 设置 `is_prefill=False`
   - 存储 decode 所需的元数据

### 阶段 5: 注意力计算（`layers/attention.py`）

使用 Flash Attention 优化：

#### Prefill 阶段
```python
# 使用 varlen 接口处理变长序列
o = flash_attn_varlen_func(
    q, k, v,
    max_seqlen_q, cu_seqlens_q,
    max_seqlen_k, cu_seqlens_k,
    softmax_scale, causal=True,
    block_table=block_tables  # 用于 prefix cache
)
```

#### Decode 阶段
```python
# 使用 kvcache 接口，自动读取缓存
o = flash_attn_with_kvcache(
    q.unsqueeze(1),  # 添加序列维度
    k_cache, v_cache,
    cache_seqlens=context_lens,
    block_table=block_tables,
    softmax_scale=scale,
    causal=True
)
```

### 阶段 6: KV Cache 存储

使用 Triton kernel 高效存储 KV：

```python
@triton.jit
def store_kvcache_kernel(
    key_ptr, key_stride,
    value_ptr, value_stride,
    k_cache_ptr, v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,  # 隐藏维度
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    # 从输入加载 key/value
    key = tl.load(key_ptr + idx * key_stride + tl.arange(0, D))
    value = tl.load(value_ptr + idx * value_stride + tl.arange(0, D))
    # 存储到 cache
    tl.store(k_cache_ptr + slot * D + tl.arange(0, D), key)
    tl.store(v_cache_ptr + slot * D + tl.arange(0, D), value)
```

### 阶段 7: 采样（`layers/sampler.py`）

使用 torch.compile 优化采样过程：

```python
@torch.compile
def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
    # 温度缩放
    logits = logits.float().div_(temperatures.unsqueeze(dim=1))
    # softmax 转换为概率
    probs = torch.softmax(logits, dim=-1)
    # 指数分布采样
    sample_tokens = probs.div_(
        torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
    ).argmax(dim=-1)
    return sample_tokens
```

### 阶段 8: 后处理（`Scheduler.postprocess`）

1. **追加 token**：将生成的 token 添加到序列
2. **检查完成条件**：
   - 达到最大长度（`max_tokens`）
   - 遇到 EOS token（除非 `ignore_eos=True`）
3. **释放资源**：如果序列完成，释放 KV Cache 块
4. **状态更新**：RUNNING → FINISHED

### 阶段 9: 迭代生成

主循环（`LLMEngine.generate`）：

```python
while not self.is_finished():
    output, num_tokens = self.step()
    # 更新进度条和吞吐量统计
    for seq_id, token_ids in output:
        outputs[seq_id] = token_ids
```

直到所有序列完成，返回解码后的文本。

## 关键优化技术

### 1. Prefix Caching（前缀缓存）

- 计算每个块的哈希值（使用 xxhash）
- 维护 `hash_to_block_id` 映射
- 相同内容的块复用，减少计算和内存分配

### 2. PagedAttention

- KV Cache 分块管理（默认 256 tokens/块）
- 块表（block table）记录每个序列的块位置
- 非连续存储，支持动态增长

### 3. CUDA Graph

- 捕获 decode 阶段的计算图
- 消除 Python 开销，提高小 batch 性能
- 支持动态 batch size（1, 2, 4, 8, 16, 32, ...）

### 4. Tensor Parallelism

- 使用 PyTorch DDP 实现张量并行
- 线性层分片：QKVParallelLinear、RowParallelLinear
- 注意力头均匀分配到各 GPU

### 5. Flash Attention

- 融合 attention 计算，减少内存读写
- 支持变长序列（prefill）和缓存读取（decode）
- 使用 block table 支持 paged KV Cache

## 示例运行流程

以 `example.py` 为例：

1. **初始化**：
   ```python
   llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)
   # 加载 Qwen3-0.6B 模型，分配 KV Cache
   ```

2. **准备 prompt**：
   ```python
   prompts = [
       "introduce yourself",
       "list all prime numbers within 100",
   ]
   # 应用 chat template
   ```

3. **生成**：
   ```python
   outputs = llm.generate(prompts, sampling_params)
   # Prefill: 处理两个 prompt（约 30 tokens）
   # Decode: 迭代生成最多 256 tokens
   ```

4. **输出**：
   ```python
   for prompt, output in zip(prompts, outputs):
       print(f"Prompt: {prompt!r}")
       print(f"Completion: {output['text']!r}")
   ```

## 性能特点

- **吞吐量**: 与 vLLM 相当（测试中略高）
- **内存效率**: Prefix caching 减少重复计算
- **低延迟**: CUDA Graph 优化 decode 阶段
- **可扩展性**: 支持张量并行（最多 8 GPU）

## 扩展阅读

- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)
- [vLLM Paper](https://arxiv.org/abs/2309.06180)
- [PagedAttention 博客](https://blog.vllm.ai/)
