import ast
import importlib.util
import pickle
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

def make_seq(tokens, max_tokens=2):
    from nanovllm.engine.sequence import Sequence
    from nanovllm.sampling_params import SamplingParams
    return Sequence(tokens, SamplingParams(max_tokens=max_tokens))

def test_chunked_prefill_only_emits_after_last_chunk(scheduler):
    seq = make_seq(list(range(8)))
    scheduler.add(seq)
    for expected_cached, expected_completion in [(3, 0), (6, 0), (8, 1)]:
        batch, prefill = scheduler.schedule()
        assert prefill and batch == [seq]
        assert seq.num_scheduled_tokens <= 3
        scheduler.postprocess(batch, [88], prefill)
        assert seq.num_cached_tokens == expected_cached
        assert seq.num_completion_tokens == expected_completion
    batch, prefill = scheduler.schedule()
    assert not prefill
    scheduler.postprocess(batch, [89], prefill)
    assert seq.completion_token_ids == [88, 89]
    assert scheduler.is_finished()
    assert not scheduler.block_manager.used_block_ids

def test_hash_only_computed_after_tokens_processed(scheduler):
    seq = make_seq(list(range(8)))
    scheduler.add(seq)
    batch, prefill = scheduler.schedule()
    manager = scheduler.block_manager
    assert not manager.hash_to_block_id
    scheduler.postprocess(batch, [80], prefill)
    assert not manager.hash_to_block_id  # only 3 of the 4 tokens computed
    batch, prefill = scheduler.schedule()
    scheduler.postprocess(batch, [80], prefill)
    assert manager.blocks[seq.block_table[0]].hash != -1
    assert manager.blocks[seq.block_table[1]].hash == -1

def test_prefix_reuse_keeps_final_block_for_logits(scheduler):
    scheduler.max_num_batched_tokens = 16
    first = make_seq(list(range(8)), max_tokens=1)
    scheduler.add(first)
    batch, prefill = scheduler.schedule()
    scheduler.postprocess(batch, [88], prefill)
    second = make_seq(list(range(8)), max_tokens=1)
    scheduler.add(second)
    batch, prefill = scheduler.schedule()
    assert second.num_cached_tokens == 4
    assert second.num_scheduled_tokens == 4
    scheduler.postprocess(batch, [89], prefill)
    assert second.completion_token_ids == [89]
    assert not scheduler.block_manager.used_block_ids

def test_preempt_replays_full_history_and_releases_blocks(scheduler):
    seq = make_seq(list(range(5)))
    scheduler.add(seq)
    batch, prefill = scheduler.schedule()
    scheduler.postprocess(batch, [88], prefill)
    batch, prefill = scheduler.schedule()
    scheduler.postprocess(batch, [88], prefill)
    scheduler.running.remove(seq)
    seq.is_prefill = False
    scheduler.preempt(seq)
    assert seq.is_prefill and seq.num_cached_tokens == 0 and not seq.block_table
    restored = pickle.loads(pickle.dumps(seq))
    assert restored.token_ids == list(range(5)) + [88]
    batch, prefill = scheduler.schedule()
    assert prefill and batch == [seq]

def test_eos_finishes_and_releases_cache(scheduler):
    seq = make_seq([1, 2])
    scheduler.add(seq)
    batch, prefill = scheduler.schedule()
    scheduler.postprocess(batch, [999], prefill)
    assert seq.is_finished and scheduler.is_finished()
    assert len(scheduler.block_manager.free_block_ids) == 16

def test_recycled_block_removes_stale_hash(cpu_modules):
    from nanovllm.engine.block_manager import BlockManager
    manager = BlockManager(1, 4)
    block_id = manager._allocate_block()
    manager.blocks[block_id].update(123, [1, 2, 3, 4])
    manager.hash_to_block_id[123] = block_id
    manager.blocks[block_id].ref_count = 0
    manager._deallocate_block(block_id)
    assert manager._allocate_block() == block_id
    assert 123 not in manager.hash_to_block_id

def test_config_supports_old_alias_and_small_chunk_budget(cpu_modules, tmp_path):
    from nanovllm.config import Config
    config = Config(str(tmp_path), kv_cache_block_size=512, max_num_batched_tokens=64)
    assert config.kvcache_block_size == config.kv_cache_block_size == 512
    assert config.max_model_len > config.max_num_batched_tokens
    with pytest.raises(ValueError, match="Conflicting"):
        Config(str(tmp_path), kv_cache_block_size=512, kvcache_block_size=1024)

def test_config_new_spelling(cpu_modules, tmp_path):
    from nanovllm.config import Config
    config = Config(str(tmp_path), kvcache_block_size=512)
    assert config.kv_cache_block_size == 512

@pytest.fixture
def api(cpu_modules):
    from fastapi.testclient import TestClient
    spec = importlib.util.spec_from_file_location("server", ROOT / "server.py")
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)
    class Tokenizer:
        name_or_path = "fake-model"
        def apply_chat_template(self, *args, **kwargs): return "prompt"
        def encode(self, text): return [1, 2]
        def decode(self, tokens, **kwargs): return "ok"
    class Model:
        tokenizer = Tokenizer()
        def generate(self, prompts, params, **kwargs):
            return [{"text": "ok", "token_ids": [3]} for _ in prompts]
    with TestClient(server.create_app(Model())) as client:
        yield client

def test_server_normal_and_streaming(api):
    assert api.get("/healthz").status_code == 200
    assert api.get("/v1/models").json()["data"][0]["id"] == "fake-model"
    payload = {"messages": [{"role": "user", "content": "hello"}], "temperature": 0.7, "n": 2}
    response = api.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    assert len(response.json()["choices"]) == 2
    assert response.json()["usage"]["completion_tokens"] == 2
    response = api.post("/v1/chat/completions", json={**payload, "stream": True, "n": 1})
    assert response.status_code == 200 and "data: [DONE]" in response.text

@pytest.mark.parametrize("field,value", [("temperature", 0), ("temperature", "oops"), ("temperature", "nan"), ("max_tokens", 0), ("n", -1)])
def test_server_invalid_sampling_returns_client_error(api, field, value):
    response = api.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hello"}], field: value})
    assert response.status_code == 400

class Tensor:
    """Shape-only NumPy stand-in; does not validate CUDA/numerical attention."""
    def __init__(self, data): self.data = np.asarray(data)
    @property
    def shape(self): return self.data.shape
    def split(self, sizes, dim=-1): return [Tensor(x) for x in np.split(self.data, np.cumsum(sizes)[:-1], axis=dim)]
    def view(self, *shape): return Tensor(self.data.reshape(*shape))
    def flatten(self, start_dim, end_dim): return Tensor(self.data.reshape(self.shape[0], -1))

@pytest.mark.parametrize("filename,classname", [("qwen2.py", "Qwen2Attention"), ("llama.py", "LlamaAttention"), ("qwen3_moe.py", "Qwen3MoeAttention")])
@pytest.mark.parametrize("decode", [False, True])
def test_custom_attention_shared_layer_shapes(filename, classname, decode):
    # Execute the actual forward method with shape-checking shared layer doubles.
    tree = ast.parse((ROOT / "nanovllm/models" / filename).read_text(encoding="utf8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == classname)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
    module = ast.parse("from __future__ import annotations")
    module.body.append(method)
    namespace = {}
    exec(compile(ast.fix_missing_locations(module), filename, "exec"), namespace)
    def rope(positions, q, k):
        assert q.shape == (3, 4, 2) and k.shape == (3, 2, 2)
        return q, k
    def attention(q, k, v):
        assert v.shape == (3, 2, 2)
        return Tensor(np.zeros((3, 1, 4, 2) if decode else (3, 4, 2)))
    def project(output):
        assert output.shape == (3, 8)
        return output
    model = SimpleNamespace(q_size=8, kv_size=4, num_heads=4, num_kv_heads=2, head_dim=2,
        qkv_proj=lambda x: Tensor(np.zeros((3, 16))), rotary_emb=rope, attn=attention,
        o_proj=project, q_norm=lambda x:x, k_norm=lambda x:x)
    assert namespace["forward"](model, None, None).shape == (3, 8)
