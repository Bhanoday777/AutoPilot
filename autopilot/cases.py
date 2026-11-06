"""Reproducible real-world bug cases (SWE-bench style): cases/<case_id>/case.json."""

from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from autopilot.sandbox import CmdResult

CASE_FILE = "case.json"
REQUIRED = ("repo", "base_commit", "issue_title", "issue_body", "fix_commit", "hidden_tests")
OPTIONAL = ("case_id", "issue_url", "test_cmd", "difficulty", "notes")
DIFFICULTIES = ("easy", "medium", "hard")
_SHA_RE = re.compile(r"^[0-9a-f]{7,40}$")
_TMP = ".autopilot_hidden_tmp"
NO_TESTS_COLLECTED = 5  # pytest exit code


class CaseError(ValueError):
    """Raised when a case.json is missing or malformed."""


@dataclass(frozen=True)
class BugCase:
    """One real bug: a repo at the commit just before the upstream fix, plus its issue.

    `fix_commit` and `hidden_tests` are for checking the agent's work only. Neither is
    part of `issue_text`, which is the only thing the agent sees.
    """

    case_id: str
    repo: str
    base_commit: str
    issue_title: str
    issue_body: str
    fix_commit: str
    hidden_tests: tuple[str, ...] = field(default_factory=tuple)  # test files changed upstream
    issue_url: str | None = None
    test_cmd: str | None = None  # replaces "python -m pytest -q --tb=short" if the repo needs it
    difficulty: str | None = None
    notes: str | None = None

    @property
    def issue_text(self) -> str:
        """Issue markdown for the agent (title as the first line)."""
        return f"# {self.issue_title}\n\n{self.issue_body.strip()}\n"


def load_case(path: str | Path) -> BugCase:
    """Load a case from its directory or its case.json path, validating every field."""
    p = Path(path)
    file = p / CASE_FILE if p.is_dir() else p
    if not file.is_file():
        raise CaseError(f"No {CASE_FILE} found at {p}")
    try:
        data: Any = json.loads(file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CaseError(f"{file} is not valid JSON: {exc}") from None
    if not isinstance(data, dict):
        raise CaseError(f"{file} must contain a JSON object")

    missing = [k for k in REQUIRED if not data.get(k) or (isinstance(data[k], str) and not data[k].strip())]
    if missing:
        raise CaseError(f"{file} is missing: {', '.join(missing)}")
    unknown = sorted(set(data) - set(REQUIRED) - set(OPTIONAL))
    if unknown:
        raise CaseError(f"{file}: unknown fields {', '.join(unknown)}")
    for key in ("base_commit", "fix_commit"):
        if not _SHA_RE.match(str(data[key]).strip()):
            raise CaseError(f"{file}: {key} must be a commit hash, got {data[key]!r}")
    repo = str(data["repo"]).strip()
    if not repo.startswith("https://"):
        raise CaseError(f"{file}: repo must be an https:// git URL, got {repo!r}")
    hidden = data["hidden_tests"]
    if not isinstance(hidden, list) or not all(isinstance(h, str) and h.endswith(".py") for h in hidden):
        raise CaseError(f"{file}: hidden_tests must be a list of .py paths")
    difficulty = data.get("difficulty")
    if difficulty is not None and difficulty not in DIFFICULTIES:
        raise CaseError(f"{file}: difficulty must be one of {', '.join(DIFFICULTIES)}")

    return BugCase(
        case_id=str(data.get("case_id") or file.parent.name),
        repo=repo,
        base_commit=str(data["base_commit"]).strip(),
        issue_title=str(data["issue_title"]).strip(),
        issue_body=str(data["issue_body"]),
        fix_commit=str(data["fix_commit"]).strip(),
        hidden_tests=tuple(hidden),
        issue_url=data.get("issue_url") or None,
        test_cmd=data.get("test_cmd") or None,
        difficulty=difficulty,
        notes=data.get("notes") or None,
    )


# ---- sandbox steps -----------------------------------------------------------


def checkout_case(sb: Any, case: BugCase) -> tuple[CmdResult, dict[str, str]]:
    """Clone the repo at base_commit, stash the hidden tests on the host, and drop history.

    The clone contains every later commit, including the upstream fix, so `.git` is
    replaced by a fresh single-commit repo: the agent cannot `git log` its way to the fix.
    Returns (result, {hidden test path: upstream content}).
    """
    res = sb.clone(case.repo, case.base_commit)
    if not res.ok:
        return res, {}
    hidden: dict[str, str] = {}
    for path in case.hidden_tests:
        got = sb.exec(f"git show {shlex.quote(case.fix_commit + ':' + path)} > {_TMP}")
        if not got.ok:
            sb.exec(f"rm -f {_TMP}")
            return CmdResult(1, f"hidden test {path} not found at {case.fix_commit}: {got.output}"), {}
        hidden[path] = sb.read_text(_TMP)
    sb.exec(f"rm -f {_TMP}")
    res = sb.exec(
        "rm -rf .git && git init -q && git config core.autocrlf false && git add -A && "
        f"git commit -q -m {shlex.quote('base ' + case.base_commit[:12])}"
    )
    return res, hidden


def run_hidden_tests(sb: Any, hidden: dict[str, str]) -> CmdResult:
    """Copy the upstream tests over the agent's tree and run them.

    If they define no tests themselves (e.g. a shared mixin in tests/__init__.py),
    pytest collects nothing, so the full suite is run instead.
    """
    for path, content in hidden.items():
        res = sb.write_file(path, content)
        if not res.ok:
            return CmdResult(1, f"could not write hidden test {path}: {res.output}")
    targets = [p for p in hidden if not p.endswith("__init__.py")]
    results = [sb.run_tests(t) for t in targets]
    if not results or all(r.exit_code == NO_TESTS_COLLECTED for r in results):
        return sb.run_tests()
    failed = next((r for r in results if not r.ok and r.exit_code != NO_TESTS_COLLECTED), None)
    return failed or CmdResult(0, "\n".join(r.output.strip() for r in results))
