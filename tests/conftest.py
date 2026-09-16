"""Load CPU modules without importing the CUDA engine or downloading models."""
import sys
import types
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture(scope="session")
def cpu_modules():
    names = ["nanovllm", "transformers", "server"]
    saved = {name: mod for name, mod in sys.modules.items() if any(name == n or name.startswith(n + ".") for n in names)}
    package = types.ModuleType("nanovllm")
    package.__path__ = [str(ROOT / "nanovllm")]
    package.LLM = object
    transformers = types.ModuleType("transformers")
    class AutoConfig:
        @classmethod
        def from_pretrained(cls, path):
            return types.SimpleNamespace(max_position_embeddings=4096)
    transformers.AutoConfig = AutoConfig
    sys.modules["nanovllm"] = package
    sys.modules["transformers"] = transformers
    from nanovllm.sampling_params import SamplingParams
    package.SamplingParams = SamplingParams
    yield package
    for name in list(sys.modules):
        if any(name == n or name.startswith(n + ".") for n in names):
            sys.modules.pop(name, None)
    sys.modules.update(saved)

@pytest.fixture
def scheduler(cpu_modules):
    from nanovllm.engine.sequence import Sequence
    from nanovllm.engine.scheduler import Scheduler
    old_size = Sequence.block_size
    Sequence.block_size = 4
    config = types.SimpleNamespace(max_num_seqs=8, max_num_batched_tokens=3, eos=999, kvcache_block_size=4, num_kvcache_blocks=16)
    yield Scheduler(config)
    Sequence.block_size = old_size
