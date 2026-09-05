import pytest
import torch

from src.training.resume_diagnostics import gradient_differences


def test_gradient_comparison_counts_actual_values_and_rejects_missing_or_nonfinite():
    expected = {"weight": torch.ones(4, dtype=torch.bfloat16), "unused": None}
    actual = {"weight": torch.ones(4, dtype=torch.bfloat16), "unused": None}
    assert gradient_differences(expected, actual) == []
    actual["weight"][0] += 1
    assert gradient_differences(expected, actual) == [
        {
            "parameter": "weight",
            "different_elements": 1,
            "numel": 4,
            "max_abs": 1.0,
        }
    ]
    with pytest.raises(ValueError, match="coverage"):
        gradient_differences(expected, {})
    actual["weight"][1] = float("nan")
    with pytest.raises(FloatingPointError):
        gradient_differences(expected, actual)
