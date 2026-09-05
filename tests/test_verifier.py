import pytest

from src.evaluation.verifier import test_outcome as outcome


def test_verifier_counts_missing_and_failed_tests_as_regressions():
    result = outcome({"fix": "PASSED", "keep1": "PASSED"}, ["fix"], ["keep1", "keep2"])
    assert result["missing_tests"] == 1
    assert result["regressions"] == 1
    assert result["progress"] == 1
    assert result["required_tests_passed"] is False
    assert result["official_passed_match"] is False


def test_verifier_preserves_official_exact_passed_set_condition():
    result = outcome({"fix": "PASSED", "keep": "PASSED", "extra": "PASSED"}, ["fix"], ["keep"])
    assert result["required_tests_passed"] is True
    assert result["official_passed_match"] is False
    with pytest.raises(ValueError):
        outcome({}, ["overlap"], ["overlap"])
