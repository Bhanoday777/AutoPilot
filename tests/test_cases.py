"""Unit tests for bug case loading (no network, no Docker)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from autopilot.cases import BugCase, CaseError, load_case, run_hidden_tests
from autopilot.sandbox import CmdResult
from tests.fakes import FakeSandbox

GOOD: dict[str, Any] = {
    "repo": "https://github.com/example/lib.git",
    "base_commit": "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678",
    "fix_commit": "0123456789abcdef0123456789abcdef01234567",
    "issue_title": "parse() drops the last item",
    "issue_body": "Calling `parse('a,b')` returns `['a']`, expected `['a', 'b']`.",
    "issue_url": "https://github.com/example/lib/issues/1",
    "hidden_tests": ["tests/test_parse.py"],
}


def write_case(tmp_path: Path, data: dict[str, Any], name: str = "lib-1") -> Path:
    d = tmp_path / name
    d.mkdir()
    (d / "case.json").write_text(json.dumps(data), encoding="utf-8")
    return d


def test_load_from_directory_and_file(tmp_path: Path) -> None:
    d = write_case(tmp_path, GOOD)
    case = load_case(d)
    assert isinstance(case, BugCase)
    assert case.case_id == "lib-1" and case.test_cmd is None
    assert case.hidden_tests == ("tests/test_parse.py",) and case.difficulty is None
    assert load_case(d / "case.json") == case


def test_issue_text_never_contains_the_fix(tmp_path: Path) -> None:
    case = load_case(write_case(tmp_path, GOOD))
    assert case.issue_text.startswith("# parse() drops the last item\n")
    assert "parse('a,b')" in case.issue_text
    assert GOOD["fix_commit"] not in case.issue_text and GOOD["fix_commit"][:7] not in case.issue_text


def test_optional_fields(tmp_path: Path) -> None:
    case = load_case(write_case(tmp_path, {**GOOD, "case_id": "custom", "test_cmd": "python -m pytest -q tests",
                                           "difficulty": "easy", "notes": "why"}))
    assert case.case_id == "custom" and case.test_cmd == "python -m pytest -q tests"
    assert case.difficulty == "easy" and case.notes == "why"


@pytest.mark.parametrize("change,message", [
    ({"base_commit": ""}, "missing: base_commit"),
    ({"issue_body": None}, "missing: issue_body"),
    ({"base_commit": "main"}, "base_commit must be a commit hash"),
    ({"repo": "git@github.com:x/y.git"}, "https://"),
    ({"fix_diff": "..."}, "unknown fields fix_diff"),
    ({"hidden_tests": []}, "missing: hidden_tests"),
    ({"hidden_tests": "tests/test_parse.py"}, "list of .py paths"),
    ({"difficulty": "trivial"}, "difficulty must be one of"),
])
def test_invalid_cases(tmp_path: Path, change: dict[str, Any], message: str) -> None:
    with pytest.raises(CaseError, match=message):
        load_case(write_case(tmp_path, {**GOOD, **change}))


def test_missing_and_bad_json(tmp_path: Path) -> None:
    with pytest.raises(CaseError, match="No case.json"):
        load_case(tmp_path / "nope")
    d = tmp_path / "bad"
    d.mkdir()
    (d / "case.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(CaseError, match="not valid JSON"):
        load_case(d)


def test_hidden_text_never_contains_hidden_test_paths(tmp_path: Path) -> None:
    case = load_case(write_case(tmp_path, GOOD))
    assert "test_parse.py" not in case.issue_text


class HiddenSandbox(FakeSandbox):
    """Records which targets were run; '__init__.py'-only runs collect nothing."""

    def __init__(self, outcome: dict[str, CmdResult]) -> None:
        super().__init__()
        self.outcome = outcome
        self.runs: list[str] = []

    def run_tests(self, target: str = "", timeout: int = 300) -> CmdResult:
        self.runs.append(target)
        return self.outcome.get(target, CmdResult(0, "ok"))


def test_hidden_tests_are_written_and_run() -> None:
    sb = HiddenSandbox({"tests/test_parse.py": CmdResult(1, "1 failed")})
    res = run_hidden_tests(sb, {"tests/test_parse.py": "def test_x(): assert False\n"})
    assert not res.ok and sb.runs == ["tests/test_parse.py"]
    assert sb.files["tests/test_parse.py"].startswith("def test_x")


def test_hidden_mixin_falls_back_to_full_suite() -> None:
    sb = HiddenSandbox({"": CmdResult(0, "312 passed")})
    res = run_hidden_tests(sb, {"tests/__init__.py": "class Mixin: ..."})
    assert res.ok and sb.runs == [""]


def test_repo_cases_are_valid() -> None:
    """Every committed case must load."""
    root = Path(__file__).resolve().parent.parent / "cases"
    for d in sorted(root.glob("*/")) if root.is_dir() else []:
        load_case(d)
