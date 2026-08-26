import sys
from types import ModuleType

import pytest

from sglang.srt.layers.attention import qwen_sparse_attn_backend


@pytest.fixture(autouse=True)
def clear_decode_resolver_cache():
    qwen_sparse_attn_backend._resolve_trtllm_sparse_decode.cache_clear()
    yield
    qwen_sparse_attn_backend._resolve_trtllm_sparse_decode.cache_clear()


def _install_fake_flashinfer(monkeypatch, sentinel):
    decode = ModuleType("flashinfer.decode")
    decode.trtllm_batch_decode_with_kv_cache = sentinel
    flashinfer = ModuleType("flashinfer")
    flashinfer.decode = decode
    monkeypatch.setitem(sys.modules, "flashinfer", flashinfer)
    monkeypatch.setitem(sys.modules, "flashinfer.decode", decode)


@pytest.mark.parametrize(
    ("sm100", "sm120"),
    [(True, False), (False, True), (True, True)],
)
def test_trtllm_sparse_decode_enabled_on_supported_blackwell(
    monkeypatch, sm100, sm120
):
    sentinel = object()
    _install_fake_flashinfer(monkeypatch, sentinel)
    monkeypatch.setattr("sglang.srt.utils.is_sm100_supported", lambda: sm100)
    monkeypatch.setattr("sglang.srt.utils.is_sm120_supported", lambda: sm120)

    assert qwen_sparse_attn_backend._resolve_trtllm_sparse_decode() is sentinel


def test_trtllm_sparse_decode_stays_disabled_elsewhere(monkeypatch):
    sentinel = object()
    _install_fake_flashinfer(monkeypatch, sentinel)
    monkeypatch.setattr("sglang.srt.utils.is_sm100_supported", lambda: False)
    monkeypatch.setattr("sglang.srt.utils.is_sm120_supported", lambda: False)

    assert qwen_sparse_attn_backend._resolve_trtllm_sparse_decode() is None
