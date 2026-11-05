"""LLM client that rotates across free, OpenAI-compatible providers."""

from __future__ import annotations

import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

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

    def as_dict(self) -> dict[str, Any]:
        """Return usage as a plain dict."""
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "calls_per_provider": dict(self.calls_per_provider),
        }


def _is_retryable(exc: Exception) -> bool:
    """Rate limits, timeouts, connection errors and 5xx are worth retrying."""
    if isinstance(exc, (openai.RateLimitError, openai.APITimeoutError, openai.APIConnectionError)):
        return True
    if isinstance(exc, openai.APIStatusError):
        if exc.status_code == 400 and "tool_use_failed" in str(exc):
            return True  # Groq: the model emitted a malformed tool call; a retry usually works
        return exc.status_code == 429 or exc.status_code >= 500
    return False


def retry_after_seconds(exc: Exception) -> float | None:
    """Wait time the provider asked for, from the Retry-After header or the error text."""
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            return float(response.headers.get("retry-after"))
        except (TypeError, ValueError):
            pass
    m = re.search(r"try again in (\d+(?:\.\d+)?)(ms|s)", str(exc))
    if m:
        value = float(m.group(1))
        return value / 1000 if m.group(2) == "ms" else value
    return None


class LLM:
    """Chat client with retries and fall-through across free providers."""

    def __init__(
        self,
        providers: list[ProviderConfig] | None = None,
        max_retries: int = 4,
        base_delay: float = 2.0,
        max_delay: float = 60.0,
        timeout: float = 120.0,
        verbose: bool = True,
    ) -> None:
        self.providers = providers if providers is not None else load_providers()
        if not self.providers:
            raise ConfigError(
                "No LLM API key configured. Add at least one FREE key to .env:\n" + free_key_help()
            )
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.verbose = verbose
        self.usage = Usage()
        # max_retries=0: we handle retries ourselves so fall-through stays predictable.
        self._clients = {
            p.name: OpenAI(base_url=p.base_url, api_key=p.api_key, timeout=timeout, max_retries=0)
            for p in self.providers
        }

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[llm] {msg}")

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
    ) -> tuple[Any, str]:
        """Send a chat request; return (assistant_message, provider_name)."""
        failures: list[str] = []
        for p in self.providers:
            client = self._clients[p.name]
            kwargs: dict[str, Any] = {"model": p.model, "messages": messages, "temperature": temperature}
            if tools:
                kwargs["tools"] = tools
            for attempt in range(1, self.max_retries + 1):
                try:
                    resp = client.chat.completions.create(**kwargs)
                except Exception as exc:  # noqa: BLE001 - classified below
                    desc = f"{p.name} ({p.model}) attempt {attempt}: {type(exc).__name__}: {exc}"
                    if _is_retryable(exc) and attempt < self.max_retries:
                        delay = self.base_delay * 2 ** (attempt - 1)
                        hinted = retry_after_seconds(exc)
                        if hinted is not None:
                            delay = max(delay, hinted + 1.0)
                        delay = min(delay, self.max_delay)
                        self._log(f"{desc} -> retrying in {delay:.0f}s")
                        time.sleep(delay)
                        continue
                    failures.append(desc)
                    self._log(f"{desc} -> falling through to next provider")
                    break
                if not resp.choices:
                    failures.append(f"{p.name} ({p.model}): empty response")
                    break
                self._record(p.name, resp)
                return resp.choices[0].message, p.name
        raise LLMError("All free providers failed:\n" + "\n".join(f"  - {f}" for f in failures))

    def _record(self, provider: str, resp: Any) -> None:
        self.usage.calls += 1
        self.usage.calls_per_provider[provider] += 1
        if resp.usage:
            self.usage.prompt_tokens += resp.usage.prompt_tokens or 0
            self.usage.completion_tokens += resp.usage.completion_tokens or 0
