"""Unit tests for the agent loop and message conversion (fake LLM, fake sandbox)."""

from __future__ import annotations

import json

from openai.types.chat import ChatCompletionMessage

from autopilot.agent import Agent, message_to_dict
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


def test_retry_after_parsing() -> None:
    from autopilot.llm import retry_after_seconds

    assert retry_after_seconds(Exception("Please try again in 11.79s. Need more")) == 11.79
    assert retry_after_seconds(Exception("Please try again in 142.5ms.")) == 0.1425
    assert retry_after_seconds(Exception("nope")) is None
