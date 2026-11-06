"""
Numerical parity tests: PagedTransformerEngine vs. Hugging Face LlamaForCausalLM.

A tiny random Llama model is saved as safetensors, loaded through FileBackedPager
(out-of-core path), and compared against the reference implementation.
"""
import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from hivemind.core.file_pager import FileBackedPager  # noqa: E402
from hivemind.core.transformer_engine import PagedTransformerEngine, TransformerConfig  # noqa: E402

ATOL = 1e-4


class _Tok:
    eos_token_id = -1  # never stop early


def _build(tmp_path, num_kv_heads, tie=False):
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=1000, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=num_kv_heads, max_position_embeddings=256,
                      rms_norm_eps=1e-6, tie_word_embeddings=tie)
    ref = LlamaForCausalLM(cfg).eval()
    ref.save_pretrained(tmp_path, safe_serialization=True)
    files = list(tmp_path.glob("*.safetensors"))
    assert len(files) == 1
    engine = PagedTransformerEngine(FileBackedPager(str(files[0])), _Tok(),
                                    TransformerConfig.from_hf_config(str(tmp_path / "config.json")))
    return ref, engine


@pytest.fixture(params=[4, 2], ids=["mha", "gqa"])
def models(request, tmp_path):
    return _build(tmp_path, request.param)


def test_full_forward_matches_hf(models):
    ref, engine = models
    ids = torch.tensor([[1, 5, 9, 20, 33, 7, 101, 4]])
    with torch.no_grad():
        expected = ref(ids).logits
        got = engine.forward(ids)
    assert got.shape == expected.shape
    assert torch.allclose(got, expected, atol=ATOL), (got - expected).abs().max()


def test_kv_cache_incremental_matches_full_forward(models):
    ref, engine = models
    ids = torch.tensor([[1, 5, 9, 20, 33, 7, 101, 4]])
    with torch.no_grad():
        expected = ref(ids).logits
        engine.reset_cache()
        prefix = engine.forward(ids[:, :5], use_cache=True)
        steps = [engine.forward(ids[:, i:i + 1], use_cache=True) for i in range(5, ids.size(1))]
    got = torch.cat([prefix] + steps, dim=1)
    assert torch.allclose(got, expected, atol=ATOL), (got - expected).abs().max()
    assert engine.current_position == ids.size(1)
    assert all(k is not None and k.size(2) == ids.size(1) for k in engine.kv_cache['k'])


def test_kv_cache_multi_token_chunk(models):
    """A chunk of several tokens after a cached prefix needs an offset causal mask."""
    ref, engine = models
    ids = torch.tensor([[3, 14, 15, 92, 65, 35, 89, 79]])
    with torch.no_grad():
        expected = ref(ids).logits
        engine.reset_cache()
        a = engine.forward(ids[:, :3], use_cache=True)
        b = engine.forward(ids[:, 3:], use_cache=True)
    assert torch.allclose(torch.cat([a, b], dim=1), expected, atol=ATOL)


def test_greedy_generation_matches_hf(models):
    ref, engine = models
    prompt = torch.tensor([[1, 5, 9, 20]])
    n = 12
    with torch.no_grad():
        expected = ref.generate(prompt, max_new_tokens=n, do_sample=False,
                                eos_token_id=None, pad_token_id=0)[0, prompt.size(1):]
    engine.tokenizer = type("T", (), {
        "eos_token_id": -1,
        "encode": staticmethod(lambda s, return_tensors=None: prompt.clone()),
        "decode": staticmethod(lambda t, skip_special_tokens=True: t.tolist()),
    })()
    got = engine.generate("ignored", max_new_tokens=n, temperature=0)
    assert got == expected.tolist()


def test_generate_resets_cache_between_calls(models):
    ref, engine = models
    prompt = torch.tensor([[7, 8, 9]])
    engine.tokenizer = type("T", (), {
        "eos_token_id": -1,
        "encode": staticmethod(lambda s, return_tensors=None: prompt.clone()),
        "decode": staticmethod(lambda t, skip_special_tokens=True: t.tolist()),
    })()
    first = engine.generate("x", max_new_tokens=5, temperature=0)
    second = engine.generate("x", max_new_tokens=5, temperature=0)
    assert first == second


def test_tied_embeddings(tmp_path):
    ref, engine = _build(tmp_path, 4, tie=True)
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        assert torch.allclose(engine.forward(ids), ref(ids).logits, atol=ATOL)


def test_loader_reads_metadata_without_loading_tensors(tmp_path, monkeypatch):
    """Building tensor infos must not materialize tensor data (out-of-core requirement)."""
    from safetensors import safe_open
    _build(tmp_path, 4)
    path = str(next(tmp_path.glob("*.safetensors")))
    from hivemind.core.direct_loader import DirectFileModelLoader

    calls = {"n": 0}
    orig = safe_open

    class Spy:
        def __init__(self, *a, **k):
            self._cm = orig(*a, **k)

        def __enter__(self):
            f = self._cm.__enter__()

            class F:
                def __getattr__(_, name):
                    if name == "get_tensor":
                        calls["n"] += 1
                    return getattr(f, name)
            return F()

        def __exit__(self, *exc):
            return self._cm.__exit__(*exc)

    import safetensors
    monkeypatch.setattr(safetensors, "safe_open", Spy)
    loader = DirectFileModelLoader(path)
    assert calls["n"] == 0
    info = loader.get_tensor_info("model.embed_tokens.weight")
    assert info["shape"] == (1000, 64) and info["size"] == 1000 * 64 * 4
