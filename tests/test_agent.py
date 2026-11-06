"""Unit tests for the agent loop and message conversion (fake LLM, fake sandbox)."""

from __future__ import annotations

import json

from openai.types.chat import ChatCompletionMessage

from typing import Any

from autopilot.agent import BROKEN_TEST_MARK, REPEAT_MARK, Agent, message_to_dict
from autopilot.prompts import EXPLORE_NUDGE
from autopilot.sandbox import CmdResult
from autopilot.trajectory import Trajectory
from tests.fakes import FakeLLM, FakeSandbox, text_msg, tool_msg

FINISH = tool_msg("finish", json.dumps({"summary": "Root cause X, fixed Y, repro + suite pass."}), "call_f")


def test_message_to_dict_tool_call() -> None:
    d = message_to_dict(tool_msg("read_file", '{"path": "a.py"}', "abc"))
    assert d == {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "abc", "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}],
    }
    json.dumps(d)  # must be plain JSON-serialisable


def test_message_to_dict_text_only_and_empty() -> None:
    assert message_to_dict(text_msg("hi")) == {"role": "assistant", "content": "hi"}
    assert message_to_dict(text_msg("")) == {"role": "assistant", "content": ""}


def test_message_to_dict_fills_missing_id_and_args_and_keeps_extra_content() -> None:
    msg = ChatCompletionMessage.model_validate({
        "role": "assistant", "content": "",
        "tool_calls": [{"id": "", "type": "function", "function": {"name": "git_diff", "arguments": ""},
                        "extra_content": {"google": {"thought_signature": "sig"}}}],
    })
    call = message_to_dict(msg)["tool_calls"][0]
    assert call["id"].startswith("call_") and call["function"]["arguments"] == "{}"
    assert call["extra_content"] == {"google": {"thought_signature": "sig"}}


def test_agent_finishes_and_pairs_tool_ids(tmp_path) -> None:
    llm = FakeLLM([tool_msg("list_files", "{}", "c1"), FINISH])
    traj = Trajectory(root=tmp_path)
    res = Agent(llm, trajectory=traj).run("# Bug\nbody", FakeSandbox())
    assert res.status == "finished" and res.steps == 2 and "Root cause" in res.summary
    last = llm.sent[-1]
    assert last[-1] == {"role": "tool", "tool_call_id": "c1", "name": "list_files", "content": last[-1]["content"]}
    saved = json.loads((traj.dir / "trajectory.json").read_text(encoding="utf-8"))
    assert [s["tool"] for s in saved["steps"]] == ["list_files", "finish"]


def test_agent_nudges_then_stops() -> None:
    llm = FakeLLM([text_msg("I think it's fine.")])
    res = Agent(llm).run("# Bug\nbody", FakeSandbox())
    assert res.status == "error" and res.steps == 3
    assert sum(1 for m in llm.sent[-1] if m["role"] == "user") == 3  # task + 2 nudges


def test_agent_step_limit() -> None:
    res = Agent(FakeLLM([tool_msg("list_files", "{}")]), max_steps=4).run("# Bug", FakeSandbox())
    assert res.status == "step_limit" and res.steps == 4


def test_agent_compacts_old_tool_outputs() -> None:
    llm = FakeLLM([tool_msg("list_files", "{}", f"c{i}") for i in range(5)] + [FINISH])
    Agent(llm, keep_full_results=2).run("# Bug", FakeSandbox())
    tool_contents = [m["content"] for m in llm.sent[-1] if m["role"] == "tool"]
    assert len(tool_contents) == 5
    assert all(c == "[output of list_files truncated, ok]" for c in tool_contents[:3])
    assert all("truncated" not in c for c in tool_contents[3:])
    assert llm.sent[-1][0]["role"] == "system"


def test_failed_finish_does_not_stop() -> None:
    llm = FakeLLM([tool_msg("finish", '{"summary": "done"}'), FINISH])
    res = Agent(llm).run("# Bug", FakeSandbox())
    assert res.status == "finished" and res.steps == 2



def test_text_summary_after_green_suite_counts_as_finish() -> None:
    """Run 20261106-121042 step 10: a good summary as plain text cost an extra call."""
    summary = "Root cause: median returned the upper middle. Fixed in calc/stats.py; 4 passed."
    llm = FakeLLM([tool_msg("run_tests", "{}", "c1"), text_msg(summary)])
    res = Agent(llm).run("# Bug", FakeSandbox(tests_pass=True))
    assert res.status == "finished" and res.steps == 2 and res.summary == summary


def test_text_reply_without_green_suite_is_nudged() -> None:
    llm = FakeLLM([tool_msg("run_tests", "{}", "c1"), text_msg("All good, I think the bug is fixed now."), FINISH])
    res = Agent(llm).run("# Bug", FakeSandbox(tests_pass=False))
    assert res.status == "finished" and res.steps == 3


def test_edit_after_green_suite_requires_rerun() -> None:
    sb = FakeSandbox({"m.py": "x = 1\n"}, tests_pass=True)
    llm = FakeLLM([
        tool_msg("run_tests", "{}", "c1"),
        tool_msg("edit_file", json.dumps({"path": "m.py", "old_str": "x = 1", "new_str": "x = 2"}), "c2"),
        text_msg("Everything is fixed and the suite passes."),
        FINISH,
    ])
    res = Agent(llm, require_repro=False).run("# Bug", sb)
    assert res.steps == 4  # the text reply was nudged, not accepted


def test_targeted_test_run_does_not_count_as_green_suite() -> None:
    llm = FakeLLM([tool_msg("run_tests", '{"target": "tests/test_autopilot_repro.py"}', "c1"),
                   text_msg("The repro passes so we are done here."), FINISH])
    res = Agent(llm).run("# Bug", FakeSandbox(tests_pass=True))
    assert res.steps == 3


def test_agent_compacts_old_tool_call_arguments() -> None:
    big = "x" * 2000
    llm = FakeLLM([tool_msg("create_file", json.dumps({"path": f"f{i}.py", "content": big}), f"c{i}")
                   for i in range(4)] + [FINISH])
    Agent(llm, keep_full_results=2).run("# Bug", FakeSandbox())
    calls = [m["tool_calls"][0] for m in llm.sent[-1] if m.get("tool_calls")]
    old = [json.loads(c["function"]["arguments"]) for c in calls[:2]]
    recent = [json.loads(c["function"]["arguments"]) for c in calls[2:]]
    assert all(a["content"].startswith("[2000 chars omitted") and a["path"] for a in old)
    assert all(a["content"] == big for a in recent)


def test_gemini_signed_calls_are_not_stubbed() -> None:
    from autopilot.agent import Agent as A
    msgs = [{"role": "assistant", "content": None, "tool_calls": [
        {"id": "g1", "type": "function", "extra_content": {"google": {"thought_signature": "s"}},
         "function": {"name": "create_file", "arguments": json.dumps({"path": "a", "content": "y" * 500})}}]},
        {"role": "tool", "tool_call_id": "g1", "name": "create_file", "content": "ok"}]
    A(FakeLLM([FINISH]), keep_full_results=0)._compact(msgs, [(1, "create_file", "ok")])
    assert json.loads(msgs[0]["tool_calls"][0]["function"]["arguments"])["content"] == "y" * 500
    assert msgs[1]["content"].startswith("[output of create_file truncated")


SRC_EDIT = tool_msg("edit_file", json.dumps({"path": "m.py", "old_str": "x = 1", "new_str": "x = 2"}), "ce")


def test_source_edit_refused_until_repro_fails() -> None:
    """Round 1: both cachetools runs edited source before reproducing the bug."""
    sb = FakeSandbox({"m.py": "x = 1\n"}, tests_pass=False)
    llm = FakeLLM([
        SRC_EDIT,
        tool_msg("create_file", json.dumps({"path": "tests/test_autopilot_repro.py", "content": "def test(): ..."}), "cc"),
        tool_msg("run_tests", '{"target": "tests/test_autopilot_repro.py"}', "cr"),
        SRC_EDIT,
        FINISH,
    ])
    Agent(llm).run("# Bug", sb)
    results = {m["tool_call_id"]: m["content"] for m in llm.sent[-1] if m["role"] == "tool"}
    assert results["ce"].startswith("Edited m.py") and sb.files["m.py"] == "x = 2\n"
    first_try = [m for m in llm.sent[1] if m["role"] == "tool"][0]["content"]
    assert first_try.startswith("ERROR: Reproduce first")


def test_test_files_can_be_edited_before_repro() -> None:
    sb = FakeSandbox({"tests/test_autopilot_repro.py": "x = 1\n"})
    edit = tool_msg("edit_file", json.dumps({"path": "tests/test_autopilot_repro.py", "old_str": "x = 1",
                                             "new_str": "x = 2"}), "ct")
    Agent(FakeLLM([edit, FINISH])).run("# Bug", sb)
    assert sb.files["tests/test_autopilot_repro.py"] == "x = 2\n"


def test_passing_repro_does_not_unlock_edits() -> None:
    sb = FakeSandbox({"m.py": "x = 1\n"}, tests_pass=True)
    llm = FakeLLM([tool_msg("run_tests", '{"target": "tests/test_autopilot_repro.py"}', "cr"), SRC_EDIT, FINISH])
    Agent(llm).run("# Bug", sb)
    assert sb.files["m.py"] == "x = 1\n"


def test_low_steps_warning_is_sent_once() -> None:
    llm = FakeLLM([tool_msg("list_files", "{}", f"c{i}") for i in range(5)])
    Agent(llm, max_steps=5).run("# Bug", FakeSandbox())
    warnings = [m for m in llm.sent[-1] if m["role"] == "user" and "tool calls left" in m["content"]]
    assert len(warnings) == 1 and "Only 3 tool calls left" in warnings[0]["content"]
    assert not any("tool calls left" in str(m.get("content")) for m in llm.sent[1])


def test_repro_failing_via_run_command_unlocks_edits() -> None:
    """Round 2: the repro was run with run_command('python3 tests/test_autopilot_repro.py')."""
    class CmdSandbox(FakeSandbox):
        def exec(self, cmd: str, workdir: str = "", timeout: int = 120) -> CmdResult:
            if "test_autopilot_repro.py" in cmd:
                self.commands.append(cmd)
                return CmdResult(1, "AssertionError")
            return super().exec(cmd, workdir, timeout)

    sb = CmdSandbox({"m.py": "x = 1\n"})
    run = tool_msg("run_command", json.dumps({"command": "python3 tests/test_autopilot_repro.py"}), "cr")
    Agent(FakeLLM([run, SRC_EDIT, FINISH])).run("# Bug", sb)
    assert sb.files["m.py"] == "x = 2\n"


def test_unrelated_failing_command_does_not_unlock_edits() -> None:
    from autopilot.agent import _repro_command_failed

    assert not _repro_command_failed({"command": "python -c 'import sys; sys.exit(1)'"}, "exit_code=1\n")
    assert not _repro_command_failed({"command": "cat tests/test_autopilot_repro.py"}, "exit_code=1\n")
    assert not _repro_command_failed({"command": "pytest tests/test_autopilot_repro.py"}, "exit_code=0\n")
    assert _repro_command_failed({"command": "pytest -q tests/test_autopilot_repro.py"}, "exit_code=1\nF")


# --- Round 3 (cachetools-405, semver-484): steps wasted on repeats; existing tests ignored ---

def _read(path: str, start: int, end: int, call_id: str) -> Any:
    return tool_msg("read_file", json.dumps({"path": path, "start_line": start, "end_line": end}), call_id)


def _results(llm: FakeLLM) -> dict[str, str]:
    return {m["tool_call_id"]: m["content"] for m in llm.sent[-1] if m["role"] == "tool"}


def test_repeated_read_of_unchanged_lines_is_flagged() -> None:
    """semver-484 re-read version.py lines 280-370 four times without acting."""
    sb = FakeSandbox({"m.py": "".join(f"x{i} = {i}\n" for i in range(200))})
    llm = FakeLLM([_read("m.py", 10, 50, "r1"), _read("m.py", 20, 40, "r2"), _read("m.py", 60, 80, "r3"), FINISH])
    Agent(llm).run("# Bug", sb)
    res = _results(llm)
    assert REPEAT_MARK not in res["r1"] and REPEAT_MARK not in res["r3"]
    assert REPEAT_MARK in res["r2"] and "step 1" in res["r2"]
    assert "x20 = 20" in res["r2"]  # the content is still returned


def test_read_after_edit_is_not_flagged() -> None:
    sb = FakeSandbox({"m.py": "x = 1\n"})
    llm = FakeLLM([
        tool_msg("create_file", json.dumps({"path": "tests/test_autopilot_repro.py", "content": "def test(): ..."}), "cc"),
        tool_msg("run_tests", '{"target": "tests/test_autopilot_repro.py"}', "cr"),
        _read("m.py", 1, 5, "r1"), SRC_EDIT, _read("m.py", 1, 5, "r2"), FINISH,
    ])
    Agent(llm).run("# Bug", sb)
    assert REPEAT_MARK not in _results(llm)["r2"]


def test_long_exploration_without_repro_gets_one_nudge() -> None:
    """semver-484 explored for 15 steps before writing a repro test."""
    llm = FakeLLM([tool_msg("list_files", "{}", f"c{i}") for i in range(12)])
    Agent(llm, max_steps=12).run("# Bug", FakeSandbox())
    nudges = [m for m in llm.sent[-1] if m["role"] == "user" and m["content"] == EXPLORE_NUDGE]
    assert len(nudges) == 1


def test_no_exploration_nudge_once_repro_exists() -> None:
    script = [tool_msg("create_file", json.dumps({"path": "tests/test_autopilot_repro.py", "content": "x"}), "cc")]
    script += [tool_msg("list_files", "{}", f"c{i}") for i in range(11)]
    llm = FakeLLM(script)
    Agent(llm, max_steps=12).run("# Bug", FakeSandbox())
    assert not any(m["content"] == EXPLORE_NUDGE for m in llm.sent[-1] if m["role"] == "user")


def test_broken_existing_test_gets_a_hint_naming_it() -> None:
    """cachetools-405: the fix broke test_getsizeof_param and the agent kept re-editing."""
    class SuiteSandbox(FakeSandbox):
        def run_tests(self, target: str = "", timeout: int = 300) -> CmdResult:
            return CmdResult(1, "F.\n=== short test summary info ===\n"
                                "FAILED tests/__init__.py::CacheTest::test_getsizeof_param - AssertionError\n"
                                "FAILED tests/test_autopilot_repro.py::test_repro - AssertionError\n"
                                "2 failed, 1 passed in 0.1s")

    suite = tool_msg("run_tests", "{}", "rt")
    before = FakeLLM([suite, FINISH])
    Agent(before).run("# Bug", SuiteSandbox({"m.py": "x = 1\n"}))
    assert BROKEN_TEST_MARK not in _results(before)["rt"]  # no source edit yet: not the agent's doing

    llm = FakeLLM([tool_msg("run_tests", '{"target": "tests/test_autopilot_repro.py"}', "cr"),
                   SRC_EDIT, suite, FINISH])
    Agent(llm).run("# Bug", SuiteSandbox({"m.py": "x = 1\n"}))
    res = _results(llm)["rt"]
    assert BROKEN_TEST_MARK in res and "tests/__init__.py::CacheTest::test_getsizeof_param" in res
    assert "test_autopilot_repro.py::test_repro" not in res.split(BROKEN_TEST_MARK, 1)[1]


def test_system_prompt_asks_to_read_existing_tests() -> None:
    from autopilot.prompts import SYSTEM_PROMPT

    assert "existing tests" in SYSTEM_PROMPT and "expected" in SYSTEM_PROMPT
