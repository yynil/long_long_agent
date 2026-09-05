import pytest

torch = pytest.importorskip("torch")

from src.model.generation import GenerationConfig, _valid_utf8, sample_token
from src.model.generation_parity import comparison, metric_failures


def test_sampling_rejects_reserved_and_invalid_values():
    logits = torch.tensor([0.0, 1.0, 2.0, 100.0])
    assert sample_token(logits, GenerationConfig(), None, 2) == 2
    logits[1] = torch.nan
    with pytest.raises(ValueError, match="nonfinite"):
        sample_token(logits, GenerationConfig(), None, 2)
    for values in ({"seed": True}, {"temperature": -1}, {"max_new_tokens": 0}, {"top_p": 0}):
        with pytest.raises(ValueError):
            GenerationConfig(**values)


def test_sampling_seed_and_utf8_are_explicit():
    config = GenerationConfig(temperature=0.8, top_p=0.9)
    logits = torch.tensor([0.0, 1.0, 1.2, 1.3])
    first = torch.Generator().manual_seed(123)
    second = torch.Generator().manual_seed(123)
    assert [sample_token(logits, config, first, 3) for _ in range(20)] == [
        sample_token(logits, config, second, 3) for _ in range(20)
    ]
    assert _valid_utf8("修复".encode())
    assert not _valid_utf8("修复".encode()[:1])


def test_parity_metrics_and_fail_closed_thresholds():
    logits = torch.tensor([[[1.0, 2.0, 3.0], [2.0, 1.0, 3.0]]])
    thresholds = {
        "relative_rms_max": 0.02,
        "mean_kl_max": 0.002,
        "p95_kl_max": 0.01,
        "top1_agreement_min": 0.95,
    }
    assert not metric_failures(comparison(logits, logits.clone()), thresholds)
    assert set(metric_failures(comparison(logits, logits.flip(-1)), thresholds)) == {
        "relative_rms",
        "mean_kl",
        "p95_kl",
        "top1_agreement",
    }
