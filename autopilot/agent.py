"""The agent loop: LLM -> tool calls in the sandbox -> repeat until finish."""

from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from autopilot.llm import LLM, LLMError
from autopilot.prompts import (
    EXPLORE_NUDGE, LOW_STEPS_WARNING, NUDGE, REPRO_FIRST, REPRO_TEST_PATH, SYSTEM_PROMPT, task_prompt,
)
from autopilot.tools import (
    TOOL_SCHEMAS, ToolError, execute_tool, is_error, is_test_path, normalize_args, safe_path,
)
from autopilot.trajectory import Trajectory

MAX_NUDGES = 2
ARG_KEEP_CHARS = 200  # string arguments longer than this are stubbed once their result is compacted
MIN_SUMMARY_CHARS = 20
FILE_CHANGING_TOOLS = {"edit_file", "create_file", "run_command"}
WARN_STEPS_LEFT = 3
REPEAT_MARK = "NOTE: you already read"
BROKEN_TEST_MARK = "NOTE: your change broke existing test(s)"
READ_WINDOW = 120  # read_file returns this many lines when end_line is omitted
_FAILED_RE = re.compile(r"^FAILED (\S+)", re.MULTILINE)


@dataclass
class AgentResult:
    """Outcome of one agent run."""

    status: str  # "finished" | "step_limit" | "error"
    summary: str
    diff: str
    steps: int
    usage: dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0
    # Estimated prompt tokens per category, summed over all calls (see prompt_breakdown).
    token_breakdown: dict[str, int] = field(default_factory=dict)

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
    tokens: dict[str, Any] | None = None  # set on the first event of each LLM call


BREAKDOWN_KEYS = ("system_prompt", "tool_schemas", "task", "tool_outputs", "history")


def prompt_breakdown(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> dict[str, int]:
    """Estimate (~4 chars/token) how a request's prompt splits into categories.

    history = the model's own earlier replies (text + tool-call arguments) and nudges.
    """
    chars = dict.fromkeys(BREAKDOWN_KEYS, 0)
    chars["tool_schemas"] = len(json.dumps(tools)) if tools else 0
    for i, msg in enumerate(messages):
        size = len(json.dumps(msg, default=str))
        if msg["role"] == "system":
            chars["system_prompt"] += size
        elif msg["role"] == "tool":
            chars["tool_outputs"] += size
        elif i == 1 and msg["role"] == "user":
            chars["task"] += size
        else:
            chars["history"] += size
    return {k: v // 4 for k, v in chars.items()}


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


def _stub_long_args(raw: str) -> str:
    """Shorten long string values in a JSON arguments string, keeping it valid JSON."""
    try:
        args = json.loads(raw)
    except json.JSONDecodeError:
        return raw if len(raw) <= ARG_KEEP_CHARS else json.dumps({"raw": f"[{len(raw)} chars omitted]"})
    if not isinstance(args, dict):
        return raw
    changed = False
    for key, val in args.items():
        if isinstance(val, str) and len(val) > ARG_KEEP_CHARS:
            args[key] = f"[{len(val)} chars omitted]"
            changed = True
    return json.dumps(args) if changed else raw


def _needs_repro_first(name: str, args: dict[str, Any] | str) -> str | None:
    """Refuse source edits until the repro test has been seen failing (workflow step 3)."""
    if name != "edit_file" or not isinstance(args, dict):
        return None
    try:
        path = safe_path(str(args.get("path") or args.get("file_path") or ""))
    except ToolError:
        return None  # let the tool report the bad path
    return None if is_test_path(path) else "ERROR: " + REPRO_FIRST


def _repro_seen_failing(target: str, result: str) -> bool:
    """True if a test run shows the repro test failing (exit 1 = tests ran and failed)."""
    if not result.startswith("exit_code=1"):
        return False
    return target.startswith(REPRO_TEST_PATH) or (not target and REPRO_TEST_PATH in result)


def _repro_command_failed(args: dict[str, Any], result: str) -> bool:
    """True if run_command ran the repro test file (pytest or plain python) and it failed.

    Round 2: a model reproduced the bug with `python3 tests/test_autopilot_repro.py`
    instead of run_tests, which left source edits locked.
    """
    command = str(args.get("command") or "")
    runs_python = command.split(maxsplit=1)[:1] in (["python"], ["python3"], ["pytest"])
    return runs_python and REPRO_TEST_PATH in command and result.startswith("exit_code=1")


class _ReadLog:
    """Remembers which line ranges were read, so re-reading unchanged code can be flagged.

    Round 3: semver-484 re-read the same ranges of version.py up to four times and made its
    first edit on its last step. The content is still returned (old outputs get compacted,
    so a re-read can be legitimate), but with a note that pushes the agent to act.
    """

    def __init__(self) -> None:
        self.reads: dict[str, list[tuple[int, int, int]]] = {}  # path -> (start, end, step)

    def note(self, args: dict[str, Any] | str, step: int) -> str | None:
        try:
            a = normalize_args("read_file", args) if isinstance(args, dict) else None
            if a is None:
                return None
            path = safe_path(str(a.get("path") or ""))
            start = max(1, int(a.get("start_line") or 1))
            end = int(a.get("end_line") or start + READ_WINDOW - 1)
        except (ToolError, TypeError, ValueError):
            return None
        earlier = next((n for s, e, n in self.reads.get(path, []) if s <= start and end <= e), None)
        self.reads.setdefault(path, []).append((start, end, step))
        if earlier is None:
            return None
        return (f"{REPEAT_MARK} lines {start}-{end} of {path} at step {earlier} and the file has not "
                "changed since. Stop re-reading: write or run the repro test, or make an edit.")

    def changed(self, path: str | None) -> None:
        """Forget reads of a changed file (None = anything may have changed)."""
        if path is None:
            self.reads.clear()
        else:
            self.reads.pop(path, None)


def _broken_tests_note(result: str) -> str | None:
    """Name existing tests that fail in a full-suite run (the repro test is not one of them)."""
    broken = [t for t in _FAILED_RE.findall(result) if not t.startswith(REPRO_TEST_PATH)]
    if not broken:
        return None
    return (f"{BROKEN_TEST_MARK}: {', '.join(broken[:5])}. Their expected values are requirements: "
            "read the failing assertion, then change your fix so these tests and the repro pass. "
            "Never edit existing tests.")


def _changed_path(name: str, args: dict[str, Any]) -> str | None:
    try:
        return safe_path(str(args.get("path") or args.get("file_path") or ""))
    except ToolError:
        return None


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
        keep_full_results: int = 4,
        trajectory: Trajectory | None = None,
        on_event: Callable[[StepEvent], None] | None = None,
        require_repro: bool = True,
    ) -> None:
        self.llm = llm or LLM()
        self.require_repro = require_repro
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
        breakdown_total = dict.fromkeys(BREAKDOWN_KEYS, 0)
        suite_green = False  # full suite passed and nothing has changed since
        repro_failed = not self.require_repro  # the repro test has been seen failing
        repro_created = source_edited = explore_nudged = False
        read_log = _ReadLog()
        explore_limit = max(3, self.max_steps // 3)  # steps allowed before a repro test exists

        while steps < self.max_steps:
            self._compact(messages, tool_log)
            estimate = prompt_breakdown(messages, TOOL_SCHEMAS)
            try:
                msg, provider = self.llm.chat(messages, tools=TOOL_SCHEMAS)
            except LLMError as exc:
                status, summary = "error", str(exc)
                break
            steps += 1
            for key, val in estimate.items():
                breakdown_total[key] += val
            prompt_tokens, completion_tokens = getattr(self.llm, "last_usage", (0, 0))
            tokens = {"prompt": prompt_tokens, "completion": completion_tokens, "estimate": estimate}
            assistant = message_to_dict(msg)
            text = (assistant.get("content") or "").strip()
            calls = assistant.get("tool_calls", [])

            if not calls:
                if suite_green and len(text) >= MIN_SUMMARY_CHARS:
                    # The model wrote its summary as text instead of calling finish. The full
                    # suite already passed with no changes since, so accept it rather than
                    # spending another call (and free-tier tokens) on a nudge.
                    self._emit(steps, provider, text, None, {},
                               "(accepted as finish: full suite passed, no changes since)", True, tokens)
                    status, summary = "finished", text
                    break
                self._emit(steps, provider, text, None, {}, "(no tool call)", False, tokens)
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
                blocked = None if repro_failed else _needs_repro_first(name, args)
                result = blocked or execute_tool(sandbox, name, args)
                note = None
                if name == "read_file" and not is_error(result):
                    note = read_log.note(args, steps)
                elif (name == "run_tests" and source_edited and isinstance(args, dict)
                      and not str(args.get("target") or "").strip() and result.startswith("exit_code=1")):
                    note = _broken_tests_note(result)
                if note:
                    result = f"{result}\n\n{note}"
                messages.append({"role": "tool", "tool_call_id": call["id"], "name": name, "content": result})
                tool_log.append((len(messages) - 1, name, _status_of(name, result)))
                shown_args = args if isinstance(args, dict) else {"raw": args}
                self._emit(steps, provider, text if i == 0 else "", name, shown_args, result,
                           not is_error(result), tokens if i == 0 else None)
                if name == "finish" and not is_error(result):
                    finished, summary = True, str(shown_args.get("summary", ""))
                if name == "run_tests":
                    target = str(shown_args.get("target") or "").strip()
                    if not target:
                        suite_green = result.startswith("exit_code=0")
                    repro_failed = repro_failed or _repro_seen_failing(target, result)
                elif name in FILE_CHANGING_TOOLS and not result.startswith("ERROR"):
                    suite_green = False
                    changed = None if name == "run_command" else _changed_path(name, shown_args)
                    read_log.changed(changed)
                    if changed == REPRO_TEST_PATH:
                        repro_created = True
                    elif name == "edit_file" and changed and not is_test_path(changed):
                        source_edited = True
                    if name == "run_command":
                        repro_failed = repro_failed or _repro_command_failed(shown_args, result)
            if finished:
                status = "finished"
                break
            if not repro_created and not explore_nudged and steps >= explore_limit:
                explore_nudged = True
                messages.append({"role": "user", "content": EXPLORE_NUDGE})
            if self.max_steps - steps == WARN_STEPS_LEFT:
                messages.append({"role": "user", "content": LOW_STEPS_WARNING.format(n=WARN_STEPS_LEFT)})

        if status == "step_limit":
            summary = f"Step limit ({self.max_steps}) reached without calling finish."
        return AgentResult(
            status=status,
            summary=summary,
            diff=sandbox.diff().output,
            steps=steps,
            usage=self.llm.usage.as_dict(),
            seconds=round(time.monotonic() - start, 1),
            token_breakdown=breakdown_total,
        )

    def _compact(self, messages: list[dict[str, Any]], tool_log: list[tuple[int, str, str]]) -> None:
        """Replace all but the last N tool outputs with a one-line summary, and stub the
        long arguments (file contents, edit text) of the calls that produced them."""
        cutoff = len(tool_log) - self.keep_full_results
        old_ids: set[str] = set()
        for idx, name, status in tool_log[: max(cutoff, 0)]:
            messages[idx]["content"] = f"[output of {name} truncated, {status}]"
            old_ids.add(messages[idx]["tool_call_id"])
        if not old_ids:
            return
        for msg in messages:
            for call in msg.get("tool_calls") or []:
                # Gemini-signed calls must be echoed back unmodified, so leave those alone.
                if call["id"] in old_ids and "extra_content" not in call:
                    call["function"]["arguments"] = _stub_long_args(call["function"]["arguments"])

    def _emit(self, step: int, provider: str, text: str, tool: str | None,
              args: dict[str, Any], result: str, ok: bool, tokens: dict[str, Any] | None = None) -> None:
        if self.trajectory:
            self.trajectory.add_step(step, provider, text, tool, args, result, ok, tokens)
        if self.on_event:
            self.on_event(StepEvent(step, self.max_steps, provider, text, tool, args, result, ok, tokens))
