"""LLM client that rotates across free, OpenAI-compatible providers."""

from __future__ import annotations

import json
import re
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Callable

import openai
from openai import OpenAI

from autopilot.config import ConfigError, ProviderConfig, free_key_help, load_providers


class LLMError(RuntimeError):
    """Raised when every configured provider failed."""


@dataclass
class Usage:
    """Running totals for eval reporting."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls_per_provider: Counter[str] = field(default_factory=Counter)
    seconds_waiting: float = 0.0  # time spent pacing or backing off on rate limits

    def as_dict(self) -> dict[str, Any]:
        """Return usage as a plain dict."""
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "calls_per_provider": dict(self.calls_per_provider),
            "seconds_waiting": round(self.seconds_waiting, 1),
        }


# --- error classification -------------------------------------------------------------

RATE_LIMIT = "rate_limit"      # 429: wait and retry on the same provider
BAD_TOOL_CALL = "bad_tool_call"  # model emitted an unknown or malformed tool call
TRANSIENT = "transient"        # timeouts, connection errors, 5xx
FATAL = "fatal"                # anything else: fall through to the next provider

_BAD_TOOL_MARKERS = ("tool_use_failed", "tool call validation failed", "which was not in request.tools")


def classify_error(exc: Exception) -> str:
    """Decide how chat() should react to a provider error."""
    if isinstance(exc, openai.RateLimitError):
        return RATE_LIMIT
    if isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError)):
        return TRANSIENT
    if isinstance(exc, openai.APIStatusError):
        text = str(exc).lower()
        if exc.status_code == 400 and any(m in text for m in _BAD_TOOL_MARKERS):
            return BAD_TOOL_CALL
        if exc.status_code == 429:
            return RATE_LIMIT
        if exc.status_code >= 500:
            return TRANSIENT
    return FATAL


_DURATION_RE = re.compile(r"try again in ((?:\d+(?:\.\d+)?(?:h|ms|m|s))+)", re.IGNORECASE)
_UNIT_SECONDS = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001}


def retry_after_seconds(exc: Exception) -> float | None:
    """Wait time the provider asked for, from the Retry-After header or the error text.

    Understands Groq's formats: "11.79s", "142.5ms", "7m12.5s", "1h2m3s".
    """
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            return float(response.headers.get("retry-after"))
        except (TypeError, ValueError):
            pass
    m = _DURATION_RE.search(str(exc))
    if not m:
        return None
    return sum(float(v) * _UNIT_SECONDS[u.lower()]
               for v, u in re.findall(r"(\d+(?:\.\d+)?)(h|ms|m|s)", m.group(1), re.IGNORECASE))


def unknown_tool_name(exc: Exception) -> str | None:
    """Name of the tool the model invented, if the provider said so."""
    m = re.search(r"attempted to call tool '([^']+)'", str(exc))
    return m.group(1) if m else None


def estimate_tokens(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> int:
    """Rough token count (~4 characters per token) used for pacing before a request."""
    chars = len(json.dumps(messages, default=str))
    if tools:
        chars += len(json.dumps(tools))
    return chars // 4 + 1


# --- client-side pacing ---------------------------------------------------------------


class RateLimiter:
    """Sliding 60-second window of requests and tokens for one provider.

    Waiting a few seconds before a request is much cheaper than eating a 429 and
    falling through to a weaker provider mid-run.
    """

    def __init__(self, tpm: int = 0, rpm: int = 0, window: float = 60.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.tpm = tpm
        self.rpm = rpm
        self.window = window
        self.clock = clock
        self._events: deque[tuple[float, int]] = deque()  # (timestamp, tokens)

    def _prune(self, now: float) -> None:
        while self._events and now - self._events[0][0] >= self.window:
            self._events.popleft()

    def wait_time(self, tokens: int) -> float:
        """Seconds to wait before a request of `tokens` fits in the window."""
        now = self.clock()
        self._prune(now)
        events = list(self._events)
        wait = 0.0
        if self.rpm and len(events) >= self.rpm:
            wait = max(wait, events[len(events) - self.rpm][0] + self.window - now)
        if self.tpm:
            # A request bigger than the whole budget can only wait for an empty window.
            budget = max(self.tpm - tokens, 0)
            used = sum(t for _, t in events)
            for ts, t in events:
                if used <= budget:
                    break
                used -= t
                wait = max(wait, ts + self.window - now)
        return max(wait, 0.0)

    def record(self, tokens: int) -> None:
        """Count a request that was just sent."""
        self._events.append((self.clock(), tokens))


# --- client ---------------------------------------------------------------------------


class LLM:
    """Chat client with pacing, retries and fall-through across free providers."""

    def __init__(
        self,
        providers: list[ProviderConfig] | None = None,
        max_retries: int = 4,
        max_rate_limit_waits: int = 6,
        base_delay: float = 2.0,
        max_delay: float = 60.0,
        timeout: float = 120.0,
        verbose: bool = True,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.providers = providers if providers is not None else load_providers()
        if not self.providers:
            raise ConfigError(
                "No LLM API key configured. Add at least one FREE key to .env:\n" + free_key_help()
            )
        self.max_retries = max_retries
        self.max_rate_limit_waits = max_rate_limit_waits
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.verbose = verbose
        self._sleep = sleep
        self.usage = Usage()
        self.last_usage: tuple[int, int] = (0, 0)  # (prompt, completion) tokens of the last call
        # max_retries=0: we handle retries ourselves so fall-through stays predictable.
        self._clients = {
            p.name: OpenAI(base_url=p.base_url, api_key=p.api_key, timeout=timeout, max_retries=0)
            for p in self.providers
        }
        self._limiters = {p.name: RateLimiter(p.tpm, p.rpm, clock=clock) for p in self.providers}

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[llm] {msg}")

    def _wait(self, seconds: float) -> None:
        if seconds > 0:
            self.usage.seconds_waiting += seconds
            self._sleep(seconds)

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
    ) -> tuple[Any, str]:
        """Send a chat request; return (assistant_message, provider_name).

        Per provider: rate limits are waited out (unless the provider asks for longer
        than max_delay, which means a daily quota is gone), invented tool calls are
        retried with a correction, and transient errors are retried with backoff.
        Anything else falls through to the next provider.
        """
        failures: list[str] = []
        tool_names = [t["function"]["name"] for t in tools or [] if "function" in t]
        for p in self.providers:
            outcome = self._try_provider(p, messages, tools, tool_names, temperature)
            if isinstance(outcome, str):
                failures.append(outcome)
                continue
            return outcome
        raise LLMError("All free providers failed:\n" + "\n".join(f"  - {f}" for f in failures))

    def _try_provider(
        self,
        p: ProviderConfig,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_names: list[str],
        temperature: float,
    ) -> tuple[Any, str] | str:
        """Return (message, provider) on success, or a failure description."""
        client, limiter = self._clients[p.name], self._limiters[p.name]
        rate_waits = bad_calls = transient = 0
        correction: list[dict[str, Any]] = []
        temp = temperature
        attempt = 0
        while True:
            attempt += 1
            sent = messages + correction
            kwargs: dict[str, Any] = {"model": p.model, "messages": sent, "temperature": temp}
            if tools:
                kwargs["tools"] = tools
            estimate = estimate_tokens(sent, tools)
            pause = limiter.wait_time(estimate)
            if pause > 0:
                self._log(f"{p.name}: pacing {pause:.1f}s to stay under its free-tier limits")
                self._wait(pause)
            limiter.record(estimate)
            try:
                resp = client.chat.completions.create(**kwargs)
            except Exception as exc:  # noqa: BLE001 - classified below
                kind = classify_error(exc)
                desc = f"{p.name} ({p.model}) attempt {attempt}: {type(exc).__name__}: {exc}"
                if kind == RATE_LIMIT:
                    hinted = retry_after_seconds(exc)
                    if hinted is not None and hinted > self.max_delay:
                        self._log(f"{p.name}: quota exhausted (retry in {hinted:.0f}s) -> next provider")
                        return desc
                    if rate_waits < self.max_rate_limit_waits:
                        rate_waits += 1
                        delay = hinted + 1.0 if hinted is not None else self.base_delay * 2 ** (rate_waits - 1)
                        delay = min(delay, self.max_delay)
                        self._log(f"{p.name}: rate limited -> waiting {delay:.1f}s ({rate_waits}/{self.max_rate_limit_waits})")
                        self._wait(delay)
                        continue
                elif kind == BAD_TOOL_CALL:
                    if bad_calls < self.max_retries - 1:
                        bad_calls += 1
                        correction = [{"role": "user", "content": _tool_correction(exc, tool_names)}]
                        temp = max(temperature, 0.4)  # same prompt at temp 0 tends to repeat the mistake
                        self._log(f"{p.name}: invalid tool call -> retrying with a correction ({bad_calls})")
                        continue
                elif kind == TRANSIENT:
                    if transient < self.max_retries - 1:
                        transient += 1
                        delay = min(self.base_delay * 2 ** (transient - 1), self.max_delay)
                        self._log(f"{desc} -> retrying in {delay:.0f}s")
                        self._wait(delay)
                        continue
                self._log(f"{desc} -> falling through to next provider")
                return desc
            if not resp.choices:
                return f"{p.name} ({p.model}): empty response"
            self._record(p.name, resp)
            if resp.usage and resp.usage.completion_tokens:
                limiter.record(resp.usage.completion_tokens)  # output counts toward TPM too
            return resp.choices[0].message, p.name

    def _record(self, provider: str, resp: Any) -> None:
        self.usage.calls += 1
        self.usage.calls_per_provider[provider] += 1
        prompt = completion = 0
        if resp.usage:
            prompt, completion = resp.usage.prompt_tokens or 0, resp.usage.completion_tokens or 0
        self.usage.prompt_tokens += prompt
        self.usage.completion_tokens += completion
        self.last_usage = (prompt, completion)


def _tool_correction(exc: Exception, tool_names: list[str]) -> str:
    bad = unknown_tool_name(exc)
    what = f"called a tool named '{bad}', which does not exist" if bad else "contained a malformed tool call"
    allowed = ", ".join(tool_names) if tool_names else "the tools provided"
    return (
        f"Your previous reply {what}. Call exactly one of these tools with valid JSON "
        f"arguments: {allowed}. If the work is done, call `finish`."
    )
