"""Unit tests for the LLM client's pacing, retries and fall-through (no network)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import openai
try:  # openai >= 3 ships on httpx2; older releases on httpx
    import httpx2 as httpx
except ImportError:  # pragma: no cover
    import httpx  # type: ignore[no-redef]
import pytest

from autopilot.config import ProviderConfig
from autopilot.llm import (
    BAD_TOOL_CALL, FATAL, LLM, RATE_LIMIT, TRANSIENT, LLMError, RateLimiter,
    classify_error, retry_after_seconds,
)
from tests.fakes import text_msg

TOOLS = [{"type": "function", "function": {"name": n, "parameters": {}}} for n in ("read_file", "finish")]

# Error texts copied from the failed runs in runs/20261105-112701 and runs/20261105-112805.
GROQ_TPM = ("Rate limit reached for model `openai/gpt-oss-120b` in organization `org_x` service tier "
            "`on_demand` on tokens per minute (TPM): Limit 8000, Used 6344, Requested 2413. "
            "Please try again in 5.6775s.")
GROQ_TPD = ("Rate limit reached for model `openai/gpt-oss-120b` on tokens per day (TPD): "
            "Limit 200000, Used 199000, Requested 2413. Please try again in 7m12.5s.")
GROQ_BAD_TOOL = ("Error code: 400 - Tool call validation failed: tool call validation failed: "
                 "attempted to call tool 'commentary' which was not in request.tools")


def _status_error(cls: type, code: int, text: str) -> Exception:
    resp = httpx.Response(code, request=httpx.Request("POST", "https://example.invalid"))
    return cls(text, response=resp, body=None)


def rate_limited(text: str = GROQ_TPM) -> Exception:
    return _status_error(openai.RateLimitError, 429, text)


def bad_tool(text: str = GROQ_BAD_TOOL) -> Exception:
    return _status_error(openai.BadRequestError, 400, text)


def ok(content: str = "done", completion_tokens: int = 10) -> Any:
    return SimpleNamespace(choices=[SimpleNamespace(message=text_msg(content))],
                           usage=SimpleNamespace(prompt_tokens=100, completion_tokens=completion_tokens))


class FakeClient:
    """Stands in for OpenAI(); each create() pops the next scripted response or raises it."""

    def __init__(self, script: list[Any]) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, s: float) -> None:
        self.slept.append(s)
        self.now += s


def make_llm(scripts: dict[str, list[Any]], tpm: int = 0, rpm: int = 0, **kw: Any) -> tuple[LLM, dict[str, FakeClient], Clock]:
    clock = Clock()
    providers = [ProviderConfig(n, "https://example.invalid", "k", f"{n}-model", tpm=tpm, rpm=rpm) for n in scripts]
    llm = LLM(providers, verbose=False, sleep=clock.sleep, clock=clock, **kw)
    clients = {n: FakeClient(s) for n, s in scripts.items()}
    llm._clients = clients  # type: ignore[assignment]
    return llm, clients, clock


MSGS = [{"role": "user", "content": "fix it"}]


# --- parsing / classification -------------------------------------------------------

def test_retry_after_parsing() -> None:
    assert retry_after_seconds(Exception("Please try again in 11.79s. Need more")) == 11.79
    assert retry_after_seconds(Exception("Please try again in 142.5ms.")) == pytest.approx(0.1425)
    assert retry_after_seconds(Exception("Please try again in 7m12.5s.")) == pytest.approx(432.5)
    assert retry_after_seconds(Exception("Please try again in 1h2m3s.")) == pytest.approx(3723)
    assert retry_after_seconds(Exception("nope")) is None


def test_classify_error() -> None:
    assert classify_error(rate_limited()) == RATE_LIMIT
    assert classify_error(bad_tool()) == BAD_TOOL_CALL
    assert classify_error(bad_tool("Error code: 400 - tool_use_failed: bad json")) == BAD_TOOL_CALL
    assert classify_error(_status_error(openai.InternalServerError, 503, "down")) == TRANSIENT
    assert classify_error(_status_error(openai.BadRequestError, 400, "context too long")) == FATAL
    assert classify_error(_status_error(openai.AuthenticationError, 401, "bad key")) == FATAL


# --- the two real failures ------------------------------------------------------------

def test_tpm_429_is_waited_out_on_same_provider() -> None:
    """Run 20261105-112701: a 5.7 s TPM wait used to kill the run."""
    llm, clients, clock = make_llm({"groq": [rate_limited(), rate_limited(), ok()]})
    msg, provider = llm.chat(MSGS, tools=TOOLS)
    assert provider == "groq" and msg.content == "done"
    assert clock.slept == [pytest.approx(6.6775)] * 2
    assert llm.usage.seconds_waiting == pytest.approx(13.355)


def test_invented_tool_is_retried_with_correction() -> None:
    """Run 20261105-112805: gpt-oss called a non-existent 'commentary' tool -> fatal 400."""
    llm, clients, _ = make_llm({"groq": [bad_tool(), ok()]})
    msg, provider = llm.chat(MSGS, tools=TOOLS)
    assert provider == "groq"
    retry = clients["groq"].calls[1]
    note = retry["messages"][-1]
    assert note["role"] == "user" and "'commentary'" in note["content"]
    assert "read_file, finish" in note["content"]
    assert retry["temperature"] > 0
    assert MSGS == [{"role": "user", "content": "fix it"}]  # caller's history untouched


# --- fall-through ---------------------------------------------------------------------

def test_daily_quota_falls_through_immediately() -> None:
    llm, clients, clock = make_llm({"groq": [rate_limited(GROQ_TPD)], "gemini": [ok()]})
    _, provider = llm.chat(MSGS)
    assert provider == "gemini" and clock.slept == [] and len(clients["groq"].calls) == 1


def test_rate_limit_waits_are_capped() -> None:
    llm, clients, _ = make_llm({"groq": [rate_limited()] * 3, "gemini": [ok()]}, max_rate_limit_waits=2)
    _, provider = llm.chat(MSGS)
    assert provider == "gemini" and len(clients["groq"].calls) == 3


def test_repeated_bad_tool_calls_fall_through() -> None:
    llm, clients, _ = make_llm({"groq": [bad_tool()] * 4, "gemini": [ok()]}, max_retries=3)
    _, provider = llm.chat(MSGS, tools=TOOLS)
    assert provider == "gemini" and len(clients["groq"].calls) == 3
    assert clients["gemini"].calls[0]["messages"] == MSGS  # correction is not carried over


def test_all_fail_raises() -> None:
    llm, _, _ = make_llm({"groq": [_status_error(openai.AuthenticationError, 401, "bad key")]})
    with pytest.raises(LLMError, match="bad key"):
        llm.chat(MSGS)


# --- pacing ---------------------------------------------------------------------------

def test_rate_limiter_tokens_window() -> None:
    clock = Clock()
    rl = RateLimiter(tpm=1000, clock=clock)
    assert rl.wait_time(600) == 0
    rl.record(600)
    clock.now = 10
    assert rl.wait_time(600) == pytest.approx(50)  # must wait for the first request to expire
    assert rl.wait_time(400) == 0
    clock.now = 60
    assert rl.wait_time(600) == 0


def test_rate_limiter_requests_window_and_oversized_request() -> None:
    clock = Clock()
    rl = RateLimiter(rpm=2, tpm=100, clock=clock)
    rl.record(10)
    clock.now = 5
    rl.record(10)
    clock.now = 20
    assert rl.wait_time(10) == pytest.approx(40)
    clock.now = 100
    assert rl.wait_time(5000) == 0  # bigger than the budget: go once the window is empty


def test_llm_paces_before_hitting_the_limit() -> None:
    big = [{"role": "user", "content": "x" * 20_000}]  # ~5k tokens
    llm, clients, clock = make_llm({"groq": [ok(), ok()]}, tpm=8000)
    llm.chat(big)
    llm.chat(big)
    assert len(clock.slept) == 1 and 55 < clock.slept[0] <= 60
