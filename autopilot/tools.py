"""Agent tools: OpenAI-format schemas plus implementations that call the Sandbox."""

from __future__ import annotations

import json
import posixpath
import re
import shlex
from typing import Any, Callable

from autopilot.sandbox import truncate

TOOL_OUTPUT_LIMIT = 8_000
READ_MAX_LINES = 200
SEARCH_MAX_MATCHES = 50
SKIP_DIRS = (".git", "__pycache__", ".venv", "node_modules")


class ToolError(Exception):
    """A tool was called with invalid input; the message is shown to the agent."""


# ---- helpers ---------------------------------------------------------------


def safe_path(path: str | None) -> str:
    """Normalise a repo-relative path; reject absolute paths and escapes via '..'."""
    p = (path or ".").strip().replace("\\", "/")
    if "\0" in p:
        raise ToolError("Path contains a null byte.")
    if p.startswith(("/", "~")) or re.match(r"^[A-Za-z]:", p):
        raise ToolError(f"Absolute paths are not allowed: {path!r}. Use a path relative to the repo root.")
    norm = posixpath.normpath(p)
    if norm == ".." or norm.startswith("../"):
        raise ToolError(f"Path escapes the repository: {path!r}.")
    return norm


def is_test_path(path: str) -> bool:
    """Heuristic: does this path look like a test file?"""
    name = posixpath.basename(path)
    parts = path.split("/")
    return name.startswith("test_") or name.endswith("_test.py") or "tests" in parts or "test" in parts


def is_error(result: str) -> bool:
    """True if a tool result signals failure (an error, or a non-zero exit code)."""
    if result.startswith("ERROR"):
        return True
    m = re.match(r"exit_code=(-?\d+)", result)
    return bool(m and m.group(1) != "0")


def _numbered(lines: list[str], start: int) -> str:
    return "\n".join(f"{n:>5}\t{line}" for n, line in enumerate(lines, start))


# ---- tool implementations --------------------------------------------------


def list_files(sb: Any, path: str = ".", max_depth: int = 2) -> str:
    """Directory listing (dirs end in '/'), skipping noise directories."""
    p = safe_path(path)
    depth = max(1, min(int(max_depth), 6))
    prune = " -o ".join(f"-name {d}" for d in SKIP_DIRS)
    cmd = (
        f"find {shlex.quote(p)} -mindepth 1 -maxdepth {depth} \\( {prune} \\) -prune "
        "-o -mindepth 1 -printf '%y %P\\n'"
    )
    res = sb.exec(cmd, timeout=30)
    if not res.ok:
        return f"ERROR: {res.output.strip() or 'list_files failed'}"
    entries = []
    for line in res.output.splitlines():
        kind, _, rel = line.partition(" ")
        if rel:
            entries.append(rel + ("/" if kind == "d" else ""))
    if not entries:
        return f"(empty directory: {p})"
    return f"{p}/\n" + "\n".join(f"  {e}" for e in sorted(entries))


def read_file(sb: Any, path: str, start_line: int = 1, end_line: int | None = None) -> str:
    """File contents with line numbers, at most READ_MAX_LINES per call."""
    p = safe_path(path)
    try:
        lines = sb.read_text(p).splitlines()
    except FileNotFoundError:
        raise ToolError(f"File not found: {p}. Use list_files or search_code to find it.") from None
    total = len(lines)
    start = max(1, int(start_line))
    end = total if end_line is None else min(int(end_line), total)
    end = min(end, start + READ_MAX_LINES - 1)
    if total == 0:
        return f"{p} is empty."
    if start > total:
        raise ToolError(f"start_line {start} is past the end of {p} ({total} lines).")
    out = f"{p} (lines {start}-{end} of {total})\n" + _numbered(lines[start - 1 : end], start)
    if end < total:
        out += f"\n... {total - end} more lines. Call read_file with start_line={end + 1} to continue."
    return out


def search_code(sb: Any, pattern: str, path: str = ".", file_glob: str | None = None) -> str:
    """ripgrep search with line numbers, capped at SEARCH_MAX_MATCHES."""
    if not pattern:
        raise ToolError("pattern must not be empty.")
    p = safe_path(path)
    glob = f"-g {shlex.quote(file_glob)} " if file_glob else ""
    cmd = (
        f"rg -n --no-heading --color never {glob}-e {shlex.quote(pattern)} -- {shlex.quote(p)} "
        f"| head -n {SEARCH_MAX_MATCHES + 1}"
    )
    res = sb.exec(cmd, timeout=30)
    lines = [ln for ln in res.output.splitlines() if ln.strip()]
    if not lines:
        return f"No matches for {pattern!r} in {p}."
    out = "\n".join(lines[:SEARCH_MAX_MATCHES])
    if len(lines) > SEARCH_MAX_MATCHES:
        out += f"\n... more than {SEARCH_MAX_MATCHES} matches; narrow the pattern, path or file_glob."
    return out


def edit_file(sb: Any, path: str, old_str: str, new_str: str) -> str:
    """Exact, unique string replacement. Returns a snippet around the edit."""
    p = safe_path(path)
    if not old_str:
        raise ToolError("old_str must not be empty. Use create_file for new files.")
    if old_str == new_str:
        raise ToolError("old_str and new_str are identical; nothing to change.")
    if is_test_path(p) and sb.exec(f"git cat-file -e {shlex.quote('HEAD:' + p)}").ok:
        raise ToolError(
            f"{p} is an existing test file. Never modify existing tests; put new tests in "
            "tests/test_autopilot_repro.py."
        )
    try:
        text = sb.read_text(p).replace("\r\n", "\n")
    except FileNotFoundError:
        raise ToolError(f"File not found: {p}.") from None
    old, new = old_str.replace("\r\n", "\n"), new_str.replace("\r\n", "\n")
    count = text.count(old)
    if count == 0:
        raise ToolError(
            f"old_str was not found in {p} (0 matches). Read the file again and copy the exact text, "
            "including indentation, without line-number prefixes."
        )
    if count > 1:
        raise ToolError(
            f"old_str matches {count} times in {p}. Include more surrounding lines so it matches exactly once."
        )
    idx = text.index(old)
    updated = text[:idx] + new + text[idx + len(old) :]
    res = sb.write_file(p, updated)
    if not res.ok:
        raise ToolError(f"Writing {p} failed: {res.output}")
    lines = updated.splitlines()
    first = updated[:idx].count("\n") + 1
    last = first + new.count("\n")
    lo, hi = max(1, first - 3), min(len(lines), last + 3)
    return f"Edited {p}. Snippet (lines {lo}-{hi}):\n" + _numbered(lines[lo - 1 : hi], lo)


def create_file(sb: Any, path: str, content: str) -> str:
    """Create a new file; fails if it already exists."""
    p = safe_path(path)
    if p == ".":
        raise ToolError("A file path is required.")
    if sb.exists(p):
        raise ToolError(f"{p} already exists. Use edit_file to change it.")
    res = sb.write_file(p, content)
    if not res.ok:
        raise ToolError(f"Creating {p} failed: {res.output}")
    return f"Created {p} ({len(content.splitlines())} lines)."


def run_tests(sb: Any, target: str = "") -> str:
    """Run pytest (optionally on one file or node id)."""
    if target:
        file_part, sep, node = target.partition("::")
        target = safe_path(file_part) + sep + node
    res = sb.run_tests(target)
    status = " (timed out)" if res.timed_out else ""
    return f"exit_code={res.exit_code}{status}\n{res.output}"


ALLOWED_SIMPLE = {"python", "python3", "pytest", "rg", "ls", "cat", "head", "tail", "grep", "find"}
ALLOWED_GIT = {"status", "diff", "log"}
FIND_FORBIDDEN = {"-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprint", "-fprintf", "-fls"}
ALLOWED_HELP = (
    "Allowed: python, pytest, pip install, rg, ls, cat, head, tail, grep, find, git status/diff/log. "
    "One command only: no pipes, redirects, ';', '&&' or subshells."
)


def check_command(command: str) -> list[str]:
    """Validate a run_command string against the allowlist; return argv."""
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    try:
        tokens = list(lexer)
    except ValueError as exc:
        raise ToolError(f"Could not parse command: {exc}") from None
    if not tokens:
        raise ToolError("Empty command. " + ALLOWED_HELP)
    if any(t and set(t) <= set("();<>|&") for t in tokens) or "\n" in command:
        raise ToolError("Shell operators are not supported. " + ALLOWED_HELP)
    argv = shlex.split(command)
    cmd, rest = argv[0], argv[1:]
    if cmd in ("pip", "pip3"):
        if not rest or rest[0] != "install":
            raise ToolError("Only 'pip install' is allowed. " + ALLOWED_HELP)
    elif cmd in ("python", "python3") and rest[:2] == ["-m", "pip"]:
        if rest[2:3] != ["install"]:
            raise ToolError("Only 'python -m pip install' is allowed. " + ALLOWED_HELP)
    elif cmd == "git":
        if not rest or rest[0] not in ALLOWED_GIT:
            raise ToolError("Only git status, git diff and git log are allowed. " + ALLOWED_HELP)
    elif cmd == "find":
        bad = FIND_FORBIDDEN.intersection(rest)
        if bad:
            raise ToolError(f"find with {', '.join(sorted(bad))} is not allowed. " + ALLOWED_HELP)
    elif cmd not in ALLOWED_SIMPLE:
        raise ToolError(f"Command '{cmd}' is not allowed. " + ALLOWED_HELP)
    return argv


def run_command(sb: Any, command: str) -> str:
    """Run one allowlisted command in the repo."""
    argv = check_command(command)
    res = sb.exec(shlex.join(argv), timeout=120)
    status = " (timed out)" if res.timed_out else ""
    return f"exit_code={res.exit_code}{status}\n{res.output}"


def git_diff(sb: Any) -> str:
    """Current diff of the repo (including new files)."""
    out = sb.diff().output
    return out if out.strip() else "(no changes)"


def finish(sb: Any, summary: str) -> str:
    """Signal completion. The agent loop stops when this succeeds."""
    if not summary or len(summary.strip()) < 20:
        raise ToolError("Provide a real summary: root cause, the fix, and the test evidence.")
    return "Finished. Summary recorded."


# ---- schemas ---------------------------------------------------------------


def _schema(name: str, description: str, props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }


_STR = {"type": "string"}
_INT = {"type": "integer"}

TOOL_SCHEMAS: list[dict[str, Any]] = [
    _schema("list_files", "List files and directories (dirs end with '/').",
            {"path": {**_STR, "description": "Directory relative to repo root. Default '.'"},
             "max_depth": {**_INT, "description": "Depth to list (1-6). Default 2"}}, []),
    _schema("read_file", f"Read a file with line numbers (max {READ_MAX_LINES} lines per call).",
            {"path": _STR, "start_line": {**_INT, "description": "1-based. Default 1"},
             "end_line": {**_INT, "description": "Inclusive. Default: end of file"}}, ["path"]),
    _schema("search_code", f"Search code with ripgrep (regex). Returns up to {SEARCH_MAX_MATCHES} matches.",
            {"pattern": {**_STR, "description": "Regex pattern"},
             "path": {**_STR, "description": "Directory or file. Default '.'"},
             "file_glob": {**_STR, "description": "Optional glob, e.g. '*.py'"}}, ["pattern"]),
    _schema("edit_file", "Replace old_str with new_str in a file. old_str must match exactly once "
            "(copy it from read_file output without line numbers, with exact indentation).",
            {"path": _STR, "old_str": _STR, "new_str": _STR}, ["path", "old_str", "new_str"]),
    _schema("create_file", "Create a NEW file (fails if it exists). The repro test must be tests/test_autopilot_repro.py.",
            {"path": _STR, "content": _STR}, ["path", "content"]),
    _schema("run_tests", "Run pytest. Empty target runs the full suite.",
            {"target": {**_STR, "description": "e.g. 'tests/test_x.py' or 'tests/test_x.py::test_y'"}}, []),
    _schema("run_command", "Run ONE allowlisted command: python, pytest, pip install, rg, ls, cat, head, "
            "tail, grep, find, git status/diff/log. No pipes or redirects.",
            {"command": _STR}, ["command"]),
    _schema("git_diff", "Show the current diff of the repository.", {}, []),
    _schema("finish", "Call when done (or stuck). Summary must cover root cause, the fix and test evidence.",
            {"summary": _STR}, ["summary"]),
]

TOOLS: dict[str, Callable[..., str]] = {
    "list_files": list_files,
    "read_file": read_file,
    "search_code": search_code,
    "edit_file": edit_file,
    "create_file": create_file,
    "run_tests": run_tests,
    "run_command": run_command,
    "git_diff": git_diff,
    "finish": finish,
}


def execute_tool(sb: Any, name: str, args: dict[str, Any] | str | None) -> str:
    """Run a tool by name. Never raises: errors come back as 'ERROR: ...' strings."""
    func = TOOLS.get(name)
    if func is None:
        return f"ERROR: unknown tool '{name}'. Available: {', '.join(TOOLS)}"
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError as exc:
            return f"ERROR: arguments are not valid JSON ({exc})."
    if not isinstance(args, dict):
        return "ERROR: arguments must be a JSON object."
    try:
        result = func(sb, **args)
    except ToolError as exc:
        result = f"ERROR: {exc}"
    except TypeError as exc:
        result = f"ERROR: bad arguments for {name}: {exc}"
    except Exception as exc:  # noqa: BLE001 - the loop must never crash on a tool
        result = f"ERROR: {name} failed: {type(exc).__name__}: {exc}"
    return truncate(result, TOOL_OUTPUT_LIMIT)
