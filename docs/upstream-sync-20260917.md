# main 上游同步说明（2026-09-17）

本次从个人 main `6156279` 合并原始上游 `bb823b3e06983d71485a8e1f23715ebd87d98ef8`。未合入 `feature/speculative-decoding`，也未修改该分支及其工作区。

## 保留内容

- `QA.md`、`README.md` 保留完整原文，只在顶部增加版本提示。`INSTALL.md` 保持原样。
- `server.py` 保留模型列表、聊天完成及模拟 SSE 输出。增加参数校验：不支持的零/负温度、非有限温度，以及非正 `max_tokens` / `n` 返回 400，避免触发引擎断言返回 500。此版本仍不支持贪心采样，SSE 仍为完整生成后再发送分片。
- 保留 `model_dict` 的 Qwen3、Qwen2、Llama、Qwen3MoE 和 MiniCPM4；保留模型加载、服务依赖和安装文档。
- Qwen2、Llama、Qwen3MoE 的 attention 适配共享层要求的 `[tokens, heads, head_dim]` 输入和投影前展平。仅以真实 forward 方法配合 NumPy 形状替身测试，未验证真实模型数值。
- Qwen3 使用上游对配置字典 `rope_theta` 的处理；其他模型原有 RoPE 能力限制保持，不代表所有架构变种都受支持。

## 新的调度和缓存语义

采用上游 `num_scheduled_tokens` 和 chunked prefill：每轮只处理预算允许的 token；部分 prefill 结果不添加到输出；完整预填充后才产生首个回答 token。缓存 hash 在实际处理完成后登记，末尾块保留计算以生成 logits。抢占后重新进入 prefill，序列跨进程传输保留所需完整历史。

`max_num_batched_tokens` 可以小于 `max_model_len`。原 main 要求总 token 预算大于序列上限的断言已移除。

配置采用上游名称 `kvcache_block_size`，保留原 main 的 `kv_cache_block_size` 作为兼容别名；旧名称传入时归一化为新字段，并设置两个读取属性。显式非默认新值与旧值冲突时抛出 `ValueError`。引擎同步设置 `Sequence.block_size`，避免配置与序列分块大小不一致。

## 学习资料

[同步前核心代码完整快照](main-learning-snapshot-6156279.md) 保存 `6156279` 的 config、block manager、LLM engine、model runner、scheduler 和 sequence 六个文件原文及中文注释。它是历史学习文档，不是可执行模块；不要按旧注释解释新调度。其他源码中的有效学习注释原位保留。

## 验证

使用 Python 3.12 和轻量隔离依赖运行：

```powershell
uv run --no-project --python 3.12 --with pytest --with numpy --with xxhash --with fastapi --with uvicorn --with httpx python -m pytest tests -q
```

CPU 测试覆盖分块预填充完成时机、hash 登记、prefix 复用、抢占与序列传输、EOS 清理、回收块旧 hash 清理、配置别名、服务响应和错误码，以及三个自定义 attention 的共享层形状。测试替换了模型配置加载、CUDA 引擎和模型前向，不下载模型。

未运行实际 GPU 推理、FlashAttention/Triton kernel、CUDA graph、张量并行或完整模型数值/性能验证；需在支持的 Linux/CUDA 环境后续验证。这次 main 同步不包含推测解码。
