"""The agent loop: LLM -> tool calls in the sandbox -> repeat until finish."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from autopilot.llm import LLM, LLMError
from autopilot.prompts import NUDGE, SYSTEM_PROMPT, task_prompt
from autopilot.tools import TOOL_SCHEMAS, execute_tool, is_error
from autopilot.trajectory import Trajectory

MAX_NUDGES = 2


@dataclass
class AgentResult:
    """Outcome of one agent run."""

    status: str  # "finished" | "step_limit" | "error"
    summary: str
    diff: str
    steps: int
    usage: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        """Return the result as a plain dict."""
        return asdict(self)


@dataclass
class StepEvent:
    """One step, emitted to the UI callback."""

    step: int
    max_steps: int
    provider: str
    text: str
    tool: str | None
    args: dict[str, Any]
    result: str
    ok: bool


def message_to_dict(msg: Any) -> dict[str, Any]:
    """Convert an SDK assistant message into a plain dict all three providers accept."""
    calls: list[dict[str, Any]] = []
    for tc in getattr(msg, "tool_calls", None) or []:
        args = tc.function.arguments
        if not isinstance(args, str):
            args = json.dumps(args or {})
        call: dict[str, Any] = {
            "id": tc.id or f"call_{uuid.uuid4().hex[:12]}",
            "type": "function",
            "function": {"name": tc.function.name, "arguments": args.strip() or "{}"},
        }
        # Gemini attaches thought signatures here; it needs them echoed back.
        extra = getattr(tc, "model_extra", None) or {}
        if "extra_content" in extra:
            call["extra_content"] = extra["extra_content"]
        calls.append(call)
    content = getattr(msg, "content", None) or None
    out: dict[str, Any] = {"role": "assistant", "content": content if (content or calls) else ""}
    if calls:
        out["tool_calls"] = calls
    return out


def _parse_args(raw: str) -> dict[str, Any] | str:
    try:
        parsed = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        return raw  # execute_tool turns this into a clear error for the model
    return parsed if isinstance(parsed, dict) else raw


def _status_of(name: str, result: str) -> str:
    if result.startswith("exit_code="):
        return "exit=" + result.split("\n", 1)[0].split("=", 1)[1]
    return "error" if is_error(result) else "ok"


class Agent:
    """Runs the reproduce -> fix -> verify loop against a Sandbox."""

    def __init__(
        self,
        llm: LLM | None = None,
        max_steps: int = 25,
        keep_full_results: int = 6,
        trajectory: Trajectory | None = None,
        on_event: Callable[[StepEvent], None] | None = None,
    ) -> None:
        self.llm = llm or LLM()
        self.max_steps = max_steps
        self.keep_full_results = keep_full_results
        self.trajectory = trajectory
        self.on_event = on_event

    def run(self, issue_text: str, sandbox: Any) -> AgentResult:
        """Run the agent on an issue until finish, the step limit, or an LLM failure."""
        start = time.monotonic()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task_prompt(issue_text, self.max_steps)},
        ]
        tool_log: list[tuple[int, str, str]] = []  # (message index, tool name, status)
        status, summary, steps, nudges = "step_limit", "", 0, 0

        while steps < self.max_steps:
            self._compact(messages, tool_log)
            try:
                msg, provider = self.llm.chat(messages, tools=TOOL_SCHEMAS)
            except LLMError as exc:
                status, summary = "error", str(exc)
                break
            steps += 1
            assistant = message_to_dict(msg)
            text = (assistant.get("content") or "").strip()
            calls = assistant.get("tool_calls", [])

            if not calls:
                self._emit(steps, provider, text, None, {}, "(no tool call)", False)
                if nudges >= MAX_NUDGES:
                    status, summary = "error", "Model stopped calling tools. Last reply: " + text[:500]
                    break
                nudges += 1
                if text:
                    messages.append(assistant)
                messages.append({"role": "user", "content": NUDGE})
                continue

            nudges = 0
            messages.append(assistant)
            finished = False
            for i, call in enumerate(calls):
                name = call["function"]["name"]
                args = _parse_args(call["function"]["arguments"])
                result = execute_tool(sandbox, name, args)
                messages.append({"role": "tool", "tool_call_id": call["id"], "name": name, "content": result})
                tool_log.append((len(messages) - 1, name, _status_of(name, result)))
                shown_args = args if isinstance(args, dict) else {"raw": args}
                self._emit(steps, provider, text if i == 0 else "", name, shown_args, result, not is_error(result))
                if name == "finish" and not is_error(result):
                    finished, summary = True, str(shown_args.get("summary", ""))
            if finished:
                status = "finished"
                break

        if status == "step_limit":
            summary = f"Step limit ({self.max_steps}) reached without calling finish."
        return AgentResult(
            status=status,
            summary=summary,
            diff=sandbox.diff().output,
            steps=steps,
            usage=self.llm.usage.as_dict(),
            seconds=round(time.monotonic() - start, 1),
        )

    def _compact(self, messages: list[dict[str, Any]], tool_log: list[tuple[int, str, str]]) -> None:
        """Replace all but the last N tool outputs with a one-line summary."""
        cutoff = len(tool_log) - self.keep_full_results
        for idx, name, status in tool_log[: max(cutoff, 0)]:
            messages[idx]["content"] = f"[output of {name} truncated, {status}]"

    def _emit(self, step: int, provider: str, text: str, tool: str | None,
              args: dict[str, Any], result: str, ok: bool) -> None:
        if self.trajectory:
            self.trajectory.add_step(step, provider, text, tool, args, result, ok)
        if self.on_event:
            self.on_event(StepEvent(step, self.max_steps, provider, text, tool, args, result, ok))
