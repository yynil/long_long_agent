from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from src.model.generation_confirmation import load_confirmation_config, native_metrics, probe_tokens
from src.model.generation_parity import PROMPTS
from src.training.tokenizer import RWKVByteTokenizer

ROOT = Path(__file__).resolve().parents[1]


def test_confirmation_is_closed_and_prompt_material_is_new():
    config = load_confirmation_config(ROOT / "configs/generation_confirmation.yaml")
    assert len(config["prompts"]) == 3
    assert all(p["task"] not in PROMPTS for p in config["prompts"])
    tokenizer = RWKVByteTokenizer(ROOT / "external/RWKV-LM/RWKV-v7/rwkv_vocab_v20230424.txt")
    for prompt in config["prompts"]:
        for length in (128, 256, 1024, 4096, 8192, 16384):
            first = probe_tokens(tokenizer, prompt, length, config["seed"])
            assert len(first) == length
            assert first == probe_tokens(tokenizer, prompt, length, config["seed"])


def test_high_confidence_argmax_change_is_rejected():
    thresholds = load_confirmation_config(ROOT / "configs/generation_confirmation.yaml")[
        "native_thresholds"
    ]
    reference = torch.tensor([[[10.0, 0.0, 0.0]]])
    metrics, failures = native_metrics(reference, reference.clone(), thresholds)
    assert metrics["confident_positions"] == 1 and not failures
    _, failures = native_metrics(reference, reference.flip(-1), thresholds)
    assert "confident_argmax_changes" in failures
