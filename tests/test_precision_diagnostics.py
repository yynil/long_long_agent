import pytest

torch = pytest.importorskip("torch")

from src.model.precision_diagnostics import (
    diagnostic_projection_accumulation,
    error_metrics,
    precision_controls,
    projection_controls,
)


def test_precision_metrics_and_controls_fail_closed():
    values = torch.tensor([1.0, 2.0], dtype=torch.float64)
    assert error_metrics(values, values.clone()) == {
        "unequal_fraction": 0.0,
        "absolute_max": 0.0,
        "relative_rms": 0.0,
    }
    assert error_metrics(values, values + 1)["absolute_max"] == 1
    with pytest.raises(ValueError, match="nonfinite"):
        error_metrics(values, torch.tensor([float("nan"), 0.0]))
    with pytest.raises(ValueError, match="matching shapes"):
        error_metrics(values, values[:1])
    with pytest.raises(ValueError, match="BF16"):
        projection_controls(values[None], values[None])


def test_precision_flags_restore_even_after_failure():
    flags = torch.backends.cuda.matmul
    original = flags.fp32_precision, flags.allow_bf16_reduced_precision_reduction
    with pytest.raises(RuntimeError, match="probe"), precision_controls(False):
        assert flags.fp32_precision == "ieee"
        assert flags.allow_bf16_reduced_precision_reduction is False
        raise RuntimeError("probe")
    assert (flags.fp32_precision, flags.allow_bf16_reduced_precision_reduction) == original


def test_projection_probe_preserves_operands_and_all_modes():
    values = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
    weight = torch.eye(4, dtype=torch.bfloat16)
    before = values.clone(), weight.clone()
    rows = projection_controls(values, weight, lengths=(1, 2))
    assert len(rows) == 10
    assert all(row["single_vs_batch"]["absolute_max"] == 0 for row in rows)
    assert torch.equal(values, before[0]) and torch.equal(weight, before[1])
    with pytest.raises(ValueError, match="lengths"):
        projection_controls(values, weight, lengths=(3,))


def test_projection_arithmetic_context_is_inference_only_and_restores():
    from src.model import rwkv7_stateful

    original = rwkv7_stateful.inference_linear, rwkv7_stateful.inference_matmul
    with (
        pytest.raises(ValueError, match="requires inference"),
        diagnostic_projection_accumulation(torch.float64),
    ):
        pass
    with torch.inference_mode(), diagnostic_projection_accumulation(torch.float64):
        module = torch.nn.Linear(4, 2, bias=False).bfloat16()
        value = torch.ones(1, 4, dtype=torch.bfloat16)
        assert rwkv7_stateful.inference_linear(module, value).dtype == torch.bfloat16
        assert rwkv7_stateful.inference_matmul(value, module.weight.T).shape == (1, 2)
    assert (rwkv7_stateful.inference_linear, rwkv7_stateful.inference_matmul) == original
