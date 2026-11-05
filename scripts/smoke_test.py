"""Day 1 smoke test: free LLM providers and the Docker sandbox.

Usage:
    python scripts/smoke_test.py            # both checks
    python scripts/smoke_test.py --llm      # LLM only (2 calls)
    python scripts/smoke_test.py --sandbox  # sandbox only (no LLM calls)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.syntax import Syntax  # noqa: E402
from rich.table import Table  # noqa: E402
from rich.text import Text  # noqa: E402

from autopilot.config import ConfigError, load_providers  # noqa: E402

console = Console()

REPRO_TEST = '''from calc.stats import median


def test_median_even_length() -> None:
    assert median([1, 2, 3, 4]) == 2.5
'''

READ_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a file from the repository and return its contents.",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path relative to repo root"}},
            "required": ["path"],
        },
    },
}


def ok(msg: str) -> None:
    console.print(f"[bold green]PASS[/] {msg}")


def fail(msg: str) -> None:
    console.print(f"[bold red]FAIL[/] {msg}")


def warn(msg: str) -> None:
    console.print(f"[bold yellow]WARN[/] {msg}")


def check_llm() -> bool:
    """Two LLM calls: a plain prompt and a tool-call prompt."""
    from autopilot.llm import LLM, LLMError

    console.rule("[bold]LLM check")
    try:
        providers = load_providers()
        llm = LLM(providers)
    except ConfigError as exc:
        fail(str(exc))
        return False

    table = Table("order", "provider", "model", title="Configured free providers")
    for i, p in enumerate(providers, 1):
        table.add_row(str(i), p.name, p.model)
    console.print(table)

    try:
        msg, provider = llm.chat([{"role": "user", "content": "Reply with exactly: autopilot ready"}])
        ok(f"call 1 answered by [cyan]{provider}[/]: {(msg.content or '').strip()!r}")

        msg, provider = llm.chat(
            [
                {"role": "system", "content": "You are a coding agent. Use tools to inspect files."},
                {"role": "user", "content": "Read the file calc/stats.py using the read_file tool."},
            ],
            tools=[READ_FILE_TOOL],
        )
    except LLMError as exc:
        fail(str(exc))
        return False

    if msg.tool_calls:
        tc = msg.tool_calls[0]
        ok(f"call 2 answered by [cyan]{provider}[/] with tool call {tc.function.name}({tc.function.arguments})")
    else:
        warn(f"call 2 ({provider}) returned no tool call; content: {(msg.content or '')[:200]!r}")

    u = llm.usage.as_dict()
    console.print(
        f"Usage: calls={u['calls']} prompt_tokens={u['prompt_tokens']} "
        f"completion_tokens={u['completion_tokens']} per_provider={u['calls_per_provider']}"
    )
    return True


def check_sandbox() -> bool:
    """Reproduce the toy bug inside the sandbox. No LLM calls."""
    from autopilot.sandbox import Sandbox, SandboxError

    console.rule("[bold]Sandbox check")
    passed = True
    try:
        sb = Sandbox()
        with sb:
            ok(f"container started: {sb.container_id[:12]}")

            res = sb.copy_in(ROOT / "examples" / "toy_repo")
            if not res.ok:
                fail(f"copy_in failed:\n{res.output}")
                return False
            ok("toy repo copied into /workspace/repo and committed")

            sb.install()
            res = sb.run_tests()
            if res.ok:
                ok("existing tests pass")
            else:
                fail(f"existing tests should pass:\n{res.output}")
                passed = False

            sb.write_file("tests/test_repro.py", REPRO_TEST)
            res = sb.run_tests("tests/test_repro.py")
            if not res.ok and "assert 3 == 2.5" in res.output:
                ok("tests/test_repro.py FAILS as expected: bug reproduced (median([1,2,3,4]) == 3)")
            else:
                fail(f"repro test should fail with 'assert 3 == 2.5':\n{res.output}")
                passed = False

            res = sb.exec("rg -n 'def median'")
            console.print(Panel(Text(res.output.strip()), title="rg -n 'def median'"))

            res = sb.diff()
            console.print(Panel(Syntax(res.output.strip() or "(empty)", "diff"), title="git diff"))

        if sb.is_removed():
            ok("container removed")
        else:
            fail("container still exists")
            passed = False
    except SandboxError as exc:
        fail(str(exc))
        return False
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description="Autopilot Day 1 smoke test")
    parser.add_argument("--llm", action="store_true", help="run the LLM check only")
    parser.add_argument("--sandbox", action="store_true", help="run the sandbox check only")
    args = parser.parse_args()
    run_all = not (args.llm or args.sandbox)

    results: dict[str, bool] = {}
    if args.llm or run_all:
        results["llm"] = check_llm()
    if args.sandbox or run_all:
        results["sandbox"] = check_sandbox()

    console.rule("[bold]Summary")
    for name, good in results.items():
        (ok if good else fail)(name)
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
