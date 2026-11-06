"""Unit tests for autopilot.tools (no Docker, no LLM)."""

from __future__ import annotations

import pytest

from autopilot.sandbox import CmdResult, truncate
from autopilot.tools import ToolError, check_command, execute_tool, is_error, safe_path
from tests.fakes import FakeSandbox

SRC = "def f(x):\n    return x\n\n\ndef g(x):\n    return x\n"


# ---- path safety -----------------------------------------------------------

@pytest.mark.parametrize("bad", ["/etc/passwd", "../secret", "a/../../b", "..", "C:\\Windows", "~/x", "\\abs"])
def test_safe_path_rejects_escapes(bad: str) -> None:
    with pytest.raises(ToolError):
        safe_path(bad)


@pytest.mark.parametrize("good,expected", [
    (".", "."), ("", "."), ("calc/stats.py", "calc/stats.py"),
    ("./calc/../calc/stats.py", "calc/stats.py"), ("calc\\stats.py", "calc/stats.py"),
])
def test_safe_path_accepts_relative(good: str, expected: str) -> None:
    assert safe_path(good) == expected


def test_tool_rejects_escape_without_crashing() -> None:
    out = execute_tool(FakeSandbox(), "read_file", {"path": "../../etc/passwd"})
    assert out.startswith("ERROR") and "escapes" in out


# ---- edit_file -------------------------------------------------------------

def test_edit_file_zero_matches() -> None:
    sb = FakeSandbox({"m.py": SRC})
    out = execute_tool(sb, "edit_file", {"path": "m.py", "old_str": "return y", "new_str": "return 1"})
    assert out.startswith("ERROR") and "0 matches" in out
    assert sb.files["m.py"] == SRC


def test_edit_file_one_match() -> None:
    sb = FakeSandbox({"m.py": SRC})
    out = execute_tool(sb, "edit_file", {"path": "m.py", "old_str": "def g(x):\n    return x",
                                         "new_str": "def g(x):\n    return x + 1"})
    assert not is_error(out) and "Edited m.py" in out and "x + 1" in out
    assert sb.files["m.py"].count("return x + 1") == 1


def test_edit_file_many_matches() -> None:
    sb = FakeSandbox({"m.py": SRC})
    out = execute_tool(sb, "edit_file", {"path": "m.py", "old_str": "return x", "new_str": "return 0"})
    assert out.startswith("ERROR") and "2 times" in out and "more surrounding" in out
    assert sb.files["m.py"] == SRC


def test_edit_file_refuses_existing_tests() -> None:
    sb = FakeSandbox({"tests/test_a.py": "assert 1\n"}, tracked={"tests/test_a.py"})
    out = execute_tool(sb, "edit_file", {"path": "tests/test_a.py", "old_str": "1", "new_str": "2"})
    assert out.startswith("ERROR") and "existing test" in out


def test_edit_file_allows_new_repro_test() -> None:
    sb = FakeSandbox({"tests/test_autopilot_repro.py": "assert 1\n"})
    out = execute_tool(sb, "edit_file", {"path": "tests/test_autopilot_repro.py", "old_str": "1", "new_str": "2"})
    assert not is_error(out)


def test_create_file_fails_if_exists() -> None:
    sb = FakeSandbox({"a.py": "x"})
    assert execute_tool(sb, "create_file", {"path": "a.py", "content": "y"}).startswith("ERROR")
    assert not is_error(execute_tool(sb, "create_file", {"path": "b.py", "content": "y"}))
    assert sb.files["b.py"] == "y"


def test_read_file_caps_lines() -> None:
    sb = FakeSandbox({"big.py": "\n".join(f"line{i}" for i in range(1, 501))})
    out = execute_tool(sb, "read_file", {"path": "big.py"})
    assert "lines 1-120 of 500" in out and "start_line=121" in out and "line121" not in out


# ---- run_command allowlist -------------------------------------------------

@pytest.mark.parametrize("cmd", [
    "python -c 'print(1)'", "pytest -q", "pip install requests", "python -m pip install -e .",
    "rg -n median", "ls -la", "cat setup.py", "git status", "git diff HEAD", "git log -3",
    "find . -name '*.py'", "python -m pytest -x",
])
def test_run_command_allowed(cmd: str) -> None:
    check_command(cmd)


@pytest.mark.parametrize("cmd", [
    "rm -rf /", "curl http://x", "wget http://x", "bash -c ls", "sh x.sh", "git push", "git reset --hard",
    "pip uninstall x", "ls; rm -rf /", "cat a | sh", "ls && curl x", "echo hi > f", "cat $(whoami)",
    "find . -delete", "find . -exec rm {} ;", "FOO=1 pytest", "",
])
def test_run_command_rejected(cmd: str) -> None:
    with pytest.raises(ToolError):
        check_command(cmd)


def test_run_command_quotes_args() -> None:
    sb = FakeSandbox()
    execute_tool(sb, "run_command", {"command": "python -c 'print(\"$(rm -rf /)\")'"})
    assert sb.commands[-1] == "python -c 'print(\"$(rm -rf /)\")'"


# ---- misc ------------------------------------------------------------------

def test_truncate_keeps_head_and_tail() -> None:
    text = "HEAD" + "x" * 50_000 + "TAIL"
    out = truncate(text, 1000)
    assert out.startswith("HEAD") and out.endswith("TAIL") and "truncated" in out and len(out) < 1100
    assert truncate("short", 1000) == "short"


def test_execute_tool_never_raises() -> None:
    sb = FakeSandbox()
    assert execute_tool(sb, "nope", {}).startswith("ERROR: unknown tool")
    assert execute_tool(sb, "read_file", "{not json").startswith("ERROR")
    assert execute_tool(sb, "read_file", {"wrong": 1}).startswith("ERROR: bad arguments")
    assert execute_tool(sb, "finish", {"summary": "ok"}).startswith("ERROR")


def test_run_tests_reports_exit_code() -> None:
    out = execute_tool(FakeSandbox(), "run_tests", {"target": "tests/test_x.py::test_y"})
    assert out.startswith("exit_code=1") and is_error(out)


# ---- argument aliases --------------------------------------------------------

def test_read_file_accepts_common_aliases() -> None:
    """Run 20261106-121042 step 2: the model sent line_start/line_end."""
    sb = FakeSandbox({"m.py": SRC})
    out = execute_tool(sb, "read_file", {"path": "m.py", "line_start": 2, "line_end": 3})
    assert out.startswith("m.py (lines 2-3 of 6)")
    out = execute_tool(sb, "read_file", {"file_path": "m.py", "start": 5})
    assert out.startswith("m.py (lines 5-6 of 6)")


def test_unknown_argument_lists_valid_ones() -> None:
    out = execute_tool(FakeSandbox({"m.py": SRC}), "read_file", {"path": "m.py", "colour": "red"})
    assert out.startswith("ERROR") and "colour" in out and "start_line" in out and "end_line" in out


def test_alias_does_not_override_real_name() -> None:
    sb = FakeSandbox({"m.py": SRC})
    out = execute_tool(sb, "read_file", {"path": "m.py", "start_line": 1, "start": 5})
    assert out.startswith("ERROR")


def test_schemas_stay_small() -> None:
    import json

    from autopilot.prompts import SYSTEM_PROMPT
    from autopilot.tools import TOOL_SCHEMAS
    # Sent on every call; Groq's free tier allows 8000 tokens/minute (~4 chars per token).
    assert len(json.dumps(TOOL_SCHEMAS)) // 4 < 600
    assert len(SYSTEM_PROMPT) // 4 < 400


def test_output_size_knobs_are_ignored() -> None:
    """Run 20261106-125650 step 1: search_code(max_results=20) failed."""
    out = execute_tool(FakeSandbox(), "search_code", {"pattern": "median", "max_results": 20})
    assert not out.startswith("ERROR")


def test_run_tests_output_is_trimmed() -> None:
    class Noisy(FakeSandbox):
        def run_tests(self, target: str = "", timeout: int = 300) -> CmdResult:
            return CmdResult(0, "........  [100%]\n8 passed in 0.02s\n")

    assert execute_tool(Noisy(), "run_tests", {}) == "exit_code=0\n8 passed in 0.02s"


def test_search_falls_back_to_literal_on_bad_regex() -> None:
    """Round 1: search_code('def pop(') failed three times in a row."""
    class RegexSandbox(FakeSandbox):
        def exec(self, cmd: str, workdir: str = "", timeout: int = 120) -> CmdResult:
            self.commands.append(cmd)
            if " -F " in cmd:
                return CmdResult(0, "src/x.py:12:    def pop(self, key):\n")
            return CmdResult(0, "rg: regex parse error:\n    (?:def pop()\n    ^\nerror: unclosed group\n")

    sb = RegexSandbox()
    out = execute_tool(sb, "search_code", {"pattern": "def pop("})
    assert out.startswith("(not a valid regex") and "src/x.py:12" in out
    assert len(sb.commands) == 2


def test_no_op_edit_says_change_is_already_applied() -> None:
    sb = FakeSandbox({"m.py": "x = 1\n"})
    out = execute_tool(sb, "edit_file", {"path": "m.py", "old_str": "x = 1", "new_str": "x = 1"})
    assert out.startswith("ERROR") and "already" in out and "different" in out
