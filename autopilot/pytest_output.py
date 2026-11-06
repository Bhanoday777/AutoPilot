"""Trim pytest output down to what the agent needs: the first failure and the summary."""

from __future__ import annotations

import re

SECTION_RE = re.compile(r"^={3,} (.+?) ={3,}$")
BLOCK_RE = re.compile(r"^_{3,} (.+?) _{3,}$")
PROGRESS_RE = re.compile(r"^[.FEsxXp]+\s*(\[\s*\d+%\])?$")
RESULT_RE = re.compile(r"\b\d+ (passed|failed|errors?|skipped|xfailed|xpassed|warnings?)\b.* in [\d.]+s")
SUMMARY_PREFIXES = ("FAILED", "ERROR", "XFAIL", "XPASS", "SKIPPED")


def trim_pytest_output(output: str, max_summary_lines: int = 20, max_traceback_lines: int = 60) -> str:
    """Keep the first failing traceback, the short summary and the result line.

    Drops progress dots, passed tests, the session header, warnings and every traceback
    after the first (those tests are still listed in the short summary). Output that
    does not look like pytest passes through unchanged apart from stripping.
    """
    sections: list[tuple[str | None, list[str]]] = [(None, [])]
    for line in output.strip("\n").splitlines():
        m = SECTION_RE.match(line)
        if m and RESULT_RE.search(m.group(1)):
            line = m.group(1)  # non -q runs wrap the result line in '=' banners
        elif m:
            sections.append((m.group(1).strip(), []))
            continue
        sections[-1][1].append(line)

    out: list[str] = []
    for title, body in sections:
        key = (title or "").lower()
        if title is None:
            out += [ln for ln in body if ln.strip() and not PROGRESS_RE.match(ln.strip())]
        elif key in ("failures", "errors"):
            out.append(f"== {title} ==")
            out += _first_block(body, max_traceback_lines)
        elif key.startswith("short test summary"):
            out.append("== short test summary ==")
            out += _cap_summary(body, max_summary_lines)
        elif "warnings summary" in key or "test session starts" in key:
            continue
        else:
            out.append(f"== {title} ==")
            out += [ln for ln in body if ln.strip()]
    return "\n".join(out).strip()


def _first_block(body: list[str], max_lines: int) -> list[str]:
    """First failure block (header + traceback, capped) and a count of the rest."""
    blocks: list[list[str]] = [[]]
    for line in body:
        if BLOCK_RE.match(line):
            blocks.append([line])
        else:
            blocks[-1].append(line)
    preamble, failures = blocks[0], blocks[1:]
    if not failures:
        return [ln for ln in preamble if ln.strip()]
    first = failures[0]
    if len(first) > max_lines:
        head, tail = max_lines * 2 // 3, max_lines // 3
        first = first[:head] + [f"... [{len(first) - head - tail} traceback lines omitted] ..."] + first[-tail:]
    rest = len(failures) - 1
    if rest:
        first = first + [f"... {rest} more failure{'s' if rest > 1 else ''} not shown (listed in the summary)"]
    return first


def _cap_summary(body: list[str], max_lines: int) -> list[str]:
    """Cap the FAILED/ERROR list; keep everything else (result line, 'Interrupted')."""
    entries = [ln for ln in body if ln.startswith(SUMMARY_PREFIXES)]
    others = [ln for ln in body if ln.strip() and not ln.startswith(SUMMARY_PREFIXES)]
    if len(entries) > max_lines:
        entries = entries[:max_lines] + [f"... {len(entries) - max_lines} more"]
    return entries + others
