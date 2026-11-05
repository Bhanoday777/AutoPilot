"""Test doubles: a dict-backed sandbox and a scripted LLM. No Docker, no network."""

from __future__ import annotations

import shlex
from typing import Any

from openai.types.chat import ChatCompletionMessage

from autopilot.llm import Usage
from autopilot.sandbox import CmdResult


class FakeSandbox:
    """Implements the subset of Sandbox the tools use, backed by a dict of files."""

    def __init__(self, files: dict[str, str] | None = None, tracked: set[str] | None = None) -> None:
        self.files = dict(files or {})
        self.tracked = set(tracked or ())
        self.commands: list[str] = []

    def read_text(self, path: str) -> str:
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]

    def write_file(self, path: str, content: str) -> CmdResult:
        self.files[path] = content
        return CmdResult(0, "")

    def exists(self, path: str) -> bool:
        return path in self.files

    def exec(self, cmd: str, workdir: str = "", timeout: int = 120) -> CmdResult:
        self.commands.append(cmd)
        if cmd.startswith("git cat-file -e"):
            path = shlex.split(cmd)[-1].split(":", 1)[1]
            return CmdResult(0 if path in self.tracked else 1, "")
        return CmdResult(0, "ran")

    def run_tests(self, target: str = "", timeout: int = 300) -> CmdResult:
        self.commands.append(f"pytest {target}".strip())
        return CmdResult(1, "1 failed")

    def diff(self) -> CmdResult:
        return CmdResult(0, "fake diff")


def tool_msg(name: str, arguments: str, call_id: str = "call_1", content: str | None = None) -> Any:
    """Build a real SDK assistant message containing one tool call."""
    return ChatCompletionMessage.model_validate({
        "role": "assistant",
        "content": content,
        "tool_calls": [{"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}],
    })


def text_msg(content: str) -> Any:
    """Build a real SDK assistant message with text only."""
    return ChatCompletionMessage.model_validate({"role": "assistant", "content": content})


class FakeLLM:
    """Returns scripted messages in order (repeating the last one); records what it was sent."""

    def __init__(self, script: list[Any]) -> None:
        self.script = script
        self.sent: list[list[dict[str, Any]]] = []
        self.usage = Usage()

    def chat(self, messages: list[dict[str, Any]], tools: Any = None, temperature: float = 0.0) -> tuple[Any, str]:
        self.sent.append([dict(m) for m in messages])
        msg = self.script[min(len(self.sent) - 1, len(self.script) - 1)]
        self.usage.calls += 1
        self.usage.calls_per_provider["fake"] += 1
        return msg, "fake"
