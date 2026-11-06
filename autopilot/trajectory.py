"""Records every agent step to runs/<timestamp>/ for debugging and demos."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from autopilot.config import PROJECT_ROOT
from autopilot.sandbox import truncate

RESULT_LIMIT = 4_000


class Trajectory:
    """Writes trajectory.json after every step, so a crash still leaves a usable log."""

    def __init__(self, root: Path | None = None, meta: dict[str, Any] | None = None) -> None:
        root = root or PROJECT_ROOT / "runs"
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = root / stamp
        n = 1
        while run_dir.exists():
            n += 1
            run_dir = root / f"{stamp}-{n}"
        run_dir.mkdir(parents=True)
        self.dir = run_dir
        self.data: dict[str, Any] = {"meta": meta or {}, "steps": []}
        self._flush()

    def add_step(
        self,
        step: int,
        provider: str,
        text: str,
        tool: str | None,
        args: dict[str, Any] | None,
        result: str,
        ok: bool,
        tokens: dict[str, Any] | None = None,
    ) -> None:
        """Append one step (one tool call, or a reply with no tool call).

        tokens = {"prompt", "completion", "estimate": {category: tokens}} for the LLM call
        that produced this step (only on the first tool call of a multi-call reply).
        """
        self.data["steps"].append(
            {
                "step": step,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "provider": provider,
                "text": text,
                "tool": tool,
                "args": args,
                "result": truncate(result, RESULT_LIMIT),
                "ok": ok,
                "tokens": tokens,
            }
        )
        self._flush()

    def save_result(self, result: dict[str, Any], diff: str) -> None:
        """Write result.json and final.diff."""
        (self.dir / "final.diff").write_text(diff, encoding="utf-8", newline="\n")
        (self.dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    def _flush(self) -> None:
        (self.dir / "trajectory.json").write_text(json.dumps(self.data, indent=2), encoding="utf-8")
