"""Command-line entry point: python -m autopilot.cli fix ..."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from autopilot.agent import Agent, StepEvent
from autopilot.config import ConfigError, load_providers
from autopilot.llm import LLM
from autopilot.sandbox import Sandbox, SandboxError
from autopilot.trajectory import Trajectory

console = Console()


def _preview(args: dict[str, Any], limit: int = 90) -> str:
    """Short one-line preview of tool arguments."""
    parts = []
    for key, val in args.items():
        s = str(val).replace("\n", "⏎")
        parts.append(f"{key}={s[:40] + '…' if len(s) > 40 else s}")
    out = " ".join(parts)
    return out[:limit] + ("…" if len(out) > limit else "")


def show_step(ev: StepEvent) -> None:
    """Print one line per step, with the model's reasoning dimmed above it."""
    if ev.text:
        console.print(Text("    " + ev.text.replace("\n", " ")[:300], style="dim"))
    line = Text.assemble(
        (f"[{ev.step:>2}/{ev.max_steps}] ", "bold"),
        (f"{ev.provider:<10} ", "cyan"),
        (f"{ev.tool or '(text only)':<12} ", "magenta"),
        (_preview(ev.args) + " ", ""),
        ("✓" if ev.ok else "✗", "green" if ev.ok else "red"),
    )
    console.print(line)


def cmd_fix(args: argparse.Namespace) -> int:
    """Run the agent on an issue against a local folder or a git repo."""
    issue_path = Path(args.issue)
    if not issue_path.is_file():
        console.print(f"[red]Issue file not found: {issue_path}[/]")
        return 2
    issue_text = issue_path.read_text(encoding="utf-8")

    try:
        providers = load_providers()
        llm = LLM(providers)
    except ConfigError as exc:
        console.print(f"[red]{exc}[/]")
        return 2
    console.print("Free providers: " + ", ".join(f"[cyan]{p.name}[/] ({p.model})" for p in providers))

    traj = Trajectory(meta={
        "issue_file": str(issue_path),
        "source": args.local or args.repo,
        "ref": args.ref,
        "max_steps": args.max_steps,
        "providers": [f"{p.name}:{p.model}" for p in providers],
    })

    try:
        with Sandbox() as sb:
            if args.local:
                console.print(f"Copying [bold]{args.local}[/] into the sandbox…")
                res = sb.copy_in(args.local)
            else:
                console.print(f"Cloning [bold]{args.repo}[/]…")
                res = sb.clone(args.repo, args.ref)
            if not res.ok:
                console.print(Panel(Text(res.output), title="[red]Setup failed"))
                return 1
            console.print("Installing dependencies (best effort)…")
            sb.install()

            console.rule("[bold]Agent")
            agent = Agent(llm, max_steps=args.max_steps, trajectory=traj, on_event=show_step)
            result = agent.run(issue_text, sb)

            console.rule("[bold]Independent verification")
            verify = sb.run_tests()
            verified = verify.ok
    except SandboxError as exc:
        console.print(f"[red]{exc}[/]")
        return 1

    traj.save_result({**result.as_dict(), "verified": verified, "verify_output": verify.output}, result.diff)
    console.print(Panel(Text(verify.output.strip()[-2000:]), title="pytest (full suite)"))
    console.print("[bold green]PASS[/] full test suite" if verified else "[bold red]FAIL[/] full test suite")

    u = result.usage
    table = Table.grid(padding=(0, 2))
    colour = {"finished": "green", "step_limit": "yellow"}.get(result.status, "red")
    table.add_row("Status", Text(result.status, style=f"bold {colour}"))
    table.add_row("Verified", Text("PASS" if verified else "FAIL", style="green" if verified else "red"))
    table.add_row("Steps", f"{result.steps}/{args.max_steps}")
    table.add_row("LLM calls", f"{u['calls']} {u['calls_per_provider']}")
    table.add_row("Tokens", f"prompt={u['prompt_tokens']} completion={u['completion_tokens']}")
    table.add_row("Time", f"{result.seconds}s")
    table.add_row("Run log", str(traj.dir))
    table.add_row("Summary", Text(result.summary))
    console.print(Panel(table, title="Result"))
    console.print(Panel(Syntax(result.diff.strip() or "(no changes)", "diff"), title="Diff"))
    return 0 if result.status == "finished" and verified else 1


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch."""
    parser = argparse.ArgumentParser(prog="autopilot", description="Fix a bug from an issue, in a sandbox.")
    sub = parser.add_subparsers(dest="command", required=True)
    fix = sub.add_parser("fix", help="reproduce, fix and verify an issue")
    src = fix.add_mutually_exclusive_group(required=True)
    src.add_argument("--local", help="local folder to copy into the sandbox")
    src.add_argument("--repo", help="git URL to clone into the sandbox")
    fix.add_argument("--issue", required=True, help="markdown file with the issue (first line = title)")
    fix.add_argument("--ref", default=None, help="branch/tag/commit to check out (with --repo)")
    fix.add_argument("--max-steps", type=int, default=25, help="hard cap on LLM steps (default 25)")
    args = parser.parse_args(argv)
    if args.command == "fix":
        return cmd_fix(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
