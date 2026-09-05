"""Outcome metrics keep every required test in the denominator."""


def test_outcome(parsed: dict[str, str], fail_to_pass: list[str], pass_to_pass: list[str]) -> dict:
    if not fail_to_pass or not pass_to_pass or set(fail_to_pass) & set(pass_to_pass):
        raise ValueError("verifier requires disjoint nonempty test sets")
    if len(set(fail_to_pass)) != len(fail_to_pass) or len(set(pass_to_pass)) != len(pass_to_pass):
        raise ValueError("duplicate verifier test identity")
    actual_passed = {name for name, status in parsed.items() if status == "PASSED"}
    ftp_passed = sum(parsed.get(name) == "PASSED" for name in fail_to_pass)
    ptp_passed = sum(parsed.get(name) == "PASSED" for name in pass_to_pass)
    expected = set(fail_to_pass) | set(pass_to_pass)
    return {
        "fail_to_pass_total": len(fail_to_pass),
        "fail_to_pass_passed": ftp_passed,
        "fail_to_pass_failed": sum(parsed.get(name) == "FAILED" for name in fail_to_pass),
        "pass_to_pass_total": len(pass_to_pass),
        "pass_to_pass_passed": ptp_passed,
        "missing_tests": sum(name not in parsed for name in expected),
        "regressions": len(pass_to_pass) - ptp_passed,
        "progress": ftp_passed / len(fail_to_pass),
        "required_tests_passed": ftp_passed == len(fail_to_pass)
        and ptp_passed == len(pass_to_pass),
        "official_passed_match": actual_passed == expected,
    }
