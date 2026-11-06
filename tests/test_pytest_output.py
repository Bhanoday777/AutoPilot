"""Unit tests for the pytest output trimmer."""

from __future__ import annotations

from autopilot.pytest_output import trim_pytest_output

TWO_FAILURES = """\
..F.F.                                                                   [100%]
=================================== FAILURES ===================================
___________________________ test_median_even_length ____________________________
tests/test_autopilot_repro.py:5: in test_median_even_length
    assert median([1, 2, 3, 4]) == 2.5
E   assert 3 == 2.5
E    +  where 3 = median([1, 2, 3, 4])
_________________________________ test_other __________________________________
tests/test_stats.py:9: in test_other
    assert mean([]) == 0
E   ZeroDivisionError: division by zero
=============================== warnings summary ===============================
tests/test_stats.py::test_a
  /usr/lib/python3.11/site-packages/x.py:1: DeprecationWarning: old
    warnings.warn("old")

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
FAILED tests/test_autopilot_repro.py::test_median_even_length - assert 3 == 2.5
FAILED tests/test_stats.py::test_other - ZeroDivisionError: division by zero
2 failed, 4 passed, 1 warning in 0.05s
"""


def test_keeps_first_traceback_and_summary() -> None:
    out = trim_pytest_output(TWO_FAILURES)
    assert "E   assert 3 == 2.5" in out
    assert "FAILED tests/test_stats.py::test_other" in out  # still listed in the summary
    assert "2 failed, 4 passed, 1 warning in 0.05s" in out


def test_drops_other_tracebacks_progress_and_warnings() -> None:
    out = trim_pytest_output(TWO_FAILURES)
    assert "ZeroDivisionError: division by zero\n" not in out.split("short test summary")[0]
    assert "1 more failure" in out
    assert "[100%]" not in out
    assert "DeprecationWarning" not in out
    assert len(out) < len(TWO_FAILURES)


def test_all_passing_is_one_line() -> None:
    raw = "....                                                                     [100%]\n4 passed in 0.01s\n"
    assert trim_pytest_output(raw) == "4 passed in 0.01s"


def test_progress_lines_with_many_rows() -> None:
    raw = ("." * 80 + " [ 50%]\n") + ("." * 80 + " [100%]\n") + "160 passed in 1.2s\n"
    assert trim_pytest_output(raw) == "160 passed in 1.2s"


def test_collection_error_is_kept() -> None:
    raw = """\

==================================== ERRORS ====================================
_______________ ERROR collecting tests/test_autopilot_repro.py ________________
ImportError while importing test module '/workspace/repo/tests/test_autopilot_repro.py'.
E   ModuleNotFoundError: No module named 'calc.statz'
=========================== short test summary info ============================
ERROR tests/test_autopilot_repro.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.08s
"""
    out = trim_pytest_output(raw)
    assert "No module named 'calc.statz'" in out
    assert "1 error in 0.08s" in out


def test_long_summary_is_capped() -> None:
    lines = [f"FAILED tests/test_x.py::test_{i} - assert 0" for i in range(60)]
    raw = ("=" * 10 + " short test summary info " + "=" * 10 + "\n" + "\n".join(lines) + "\n60 failed in 1s\n")
    out = trim_pytest_output(raw)
    assert "test_0 " in out and "test_59" not in out and "40 more" in out
    assert out.endswith("60 failed in 1s")


def test_non_pytest_output_passes_through() -> None:
    raw = "/usr/local/bin/python: No module named pytest\n"
    assert trim_pytest_output(raw) == raw.strip()


def test_long_first_traceback_is_capped() -> None:
    body = "\n".join(f"    line {i}" for i in range(200))
    raw = f"F\n{'=' * 5} FAILURES {'=' * 5}\n{'_' * 5} test_a {'_' * 5}\n{body}\n1 failed in 0.1s\n"
    out = trim_pytest_output(raw)
    assert "line 0" in out and "line 199" in out and "line 100" not in out
