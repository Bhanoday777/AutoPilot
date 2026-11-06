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
from autopilot.cases import BugCase, CaseError, checkout_case, load_case, run_hidden_tests
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
    if ev.tokens:
        line.append(f"  {ev.tokens['prompt']}+{ev.tokens['completion']} tok", style="dim")
    console.print(line)


def token_table(breakdown: dict[str, int], prompt_total: int) -> Table:
    """Where the prompt tokens went: estimated share per category, scaled to the real total."""
    table = Table(title="Where prompt tokens went (estimated split of the real total)", title_justify="left")
    table.add_column("category")
    table.add_column("tokens", justify="right")
    table.add_column("share", justify="right")
    est_total = sum(breakdown.values()) or 1
    for key, est in sorted(breakdown.items(), key=lambda kv: -kv[1]):
        share = est / est_total
        table.add_row(key.replace("_", " "), f"{round(share * prompt_total):,}", f"{share:.0%}")
    return table


def cmd_fix(args: argparse.Namespace) -> int:
    """Run the agent on an issue against a local folder, a git repo or a bug case."""
    case: BugCase | None = None
    if args.case:
        try:
            case = load_case(args.case)
        except CaseError as exc:
            console.print(f"[red]{exc}[/]")
            return 2
        issue_text, issue_label = case.issue_text, f"case {case.case_id}"
    else:
        if not args.issue:
            console.print("[red]--issue is required with --local or --repo[/]")
            return 2
        issue_path = Path(args.issue)
        if not issue_path.is_file():
            console.print(f"[red]Issue file not found: {issue_path}[/]")
            return 2
        issue_text, issue_label = issue_path.read_text(encoding="utf-8"), str(issue_path)

    try:
        providers = load_providers()
        llm = LLM(providers)
    except ConfigError as exc:
        console.print(f"[red]{exc}[/]")
        return 2
    console.print("Free providers: " + ", ".join(f"[cyan]{p.name}[/] ({p.model})" for p in providers))

    traj = Trajectory(meta={
        "issue": issue_label,
        "case_id": case.case_id if case else None,
        "source": case.repo if case else (args.local or args.repo),
        "ref": case.base_commit if case else args.ref,
        "max_steps": args.max_steps,
        "providers": [f"{p.name}:{p.model}" for p in providers],
    })

    hidden: dict[str, str] = {}
    hidden_ok: bool | None = None  # None = no hidden tests (not a case)
    hidden_output = ""
    try:
        with Sandbox(test_cmd=case.test_cmd if case else None) as sb:
            if case:
                console.print(f"Case [bold]{case.case_id}[/]: cloning {case.repo} at {case.base_commit[:10]}…")
                res, hidden = checkout_case(sb, case)
            elif args.local:
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
            if case:
                # Last, so the agent's tree is judged first; the agent never sees these files.
                console.rule("[bold]Upstream hidden tests")
                hidden_run = run_hidden_tests(sb, hidden)
                hidden_ok, hidden_output = hidden_run.ok, hidden_run.output
    except SandboxError as exc:
        console.print(f"[red]{exc}[/]")
        return 1

    resolved = verified and hidden_ok is not False
    record: dict[str, Any] = {**result.as_dict(), "verified": verified, "verify_output": verify.output}
    if case:
        record |= {"case_id": case.case_id, "hidden_tests_passed": hidden_ok,
                   "hidden_output": hidden_output, "resolved": resolved}
    traj.save_result(record, result.diff)
    console.print(Panel(Text(verify.output.strip()[-2000:]), title="pytest (full suite)"))
    console.print("[bold green]PASS[/] full test suite" if verified else "[bold red]FAIL[/] full test suite")
    if hidden_ok is not None:
        console.print(Panel(Text(hidden_output.strip()[-1500:]), title="pytest (upstream hidden tests)"))
        console.print("[bold green]PASS[/] upstream hidden tests" if hidden_ok else "[bold red]FAIL[/] upstream hidden tests")

    u = result.usage
    table = Table.grid(padding=(0, 2))
    colour = {"finished": "green", "step_limit": "yellow"}.get(result.status, "red")
    table.add_row("Status", Text(result.status, style=f"bold {colour}"))
    table.add_row("Full suite", Text("PASS" if verified else "FAIL", style="green" if verified else "red"))
    if hidden_ok is not None:
        table.add_row("Hidden tests", Text("PASS" if hidden_ok else "FAIL", style="green" if hidden_ok else "red"))
        table.add_row("Resolved", Text("YES" if resolved else "NO", style="bold green" if resolved else "bold red"))
    table.add_row("Steps", f"{result.steps}/{args.max_steps}")
    table.add_row("LLM calls", f"{u['calls']} {u['calls_per_provider']}")
    table.add_row("Tokens", f"prompt={u['prompt_tokens']} completion={u['completion_tokens']}")
    table.add_row("Time", f"{result.seconds}s")
    table.add_row("Run log", str(traj.dir))
    table.add_row("Summary", Text(result.summary))
    console.print(Panel(table, title="Result"))
    if result.token_breakdown:
        console.print(token_table(result.token_breakdown, u["prompt_tokens"]))
    console.print(Panel(Syntax(result.diff.strip() or "(no changes)", "diff"), title="Diff"))
    return 0 if result.status == "finished" and resolved else 1


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch."""
    parser = argparse.ArgumentParser(prog="autopilot", description="Fix a bug from an issue, in a sandbox.")
    sub = parser.add_subparsers(dest="command", required=True)
    fix = sub.add_parser("fix", help="reproduce, fix and verify an issue")
    src = fix.add_mutually_exclusive_group(required=True)
    src.add_argument("--local", help="local folder to copy into the sandbox")
    src.add_argument("--repo", help="git URL to clone into the sandbox")
    src.add_argument("--case", help="bug case directory, e.g. cases/<case_id> (repo, commit and issue)")
    fix.add_argument("--issue", help="markdown file with the issue (first line = title); not used with --case")
    fix.add_argument("--ref", default=None, help="branch/tag/commit to check out (with --repo)")
    fix.add_argument("--max-steps", type=int, default=25, help="hard cap on LLM steps (default 25)")
    args = parser.parse_args(argv)
    if args.command == "fix":
        return cmd_fix(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
