"""Configuration: free LLM providers and sandbox limits, loaded from .env."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


class ConfigError(RuntimeError):
    """Raised when the configuration is invalid."""


@dataclass(frozen=True)
class ProviderSpec:
    """Static description of a free provider."""

    name: str
    base_url: str
    key_env: str
    model_env: str
    default_model: str
    key_url: str
    # Free-tier limits for the default model, used to pace requests (0 = no limit known).
    default_tpm: int = 0
    default_rpm: int = 0


# Only free-tier, no-credit-card providers. Never add paid providers here.
PROVIDERS: dict[str, ProviderSpec] = {
    "gemini": ProviderSpec(
        name="gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        key_env="GEMINI_API_KEY",
        model_env="GEMINI_MODEL",
        default_model="gemini-2.5-flash",
        key_url="https://aistudio.google.com/apikey",
        default_tpm=250_000,
        default_rpm=10,
    ),
    "groq": ProviderSpec(
        name="groq",
        base_url="https://api.groq.com/openai/v1",
        key_env="GROQ_API_KEY",
        model_env="GROQ_MODEL",
        default_model="openai/gpt-oss-120b",
        key_url="https://console.groq.com/keys",
        default_tpm=8_000,
        default_rpm=30,
    ),
    "openrouter": ProviderSpec(
        name="openrouter",
        base_url="https://openrouter.ai/api/v1",
        key_env="OPENROUTER_API_KEY",
        model_env="OPENROUTER_MODEL",
        default_model="meta-llama/llama-3.3-70b-instruct:free",
        key_url="https://openrouter.ai/keys",
        default_rpm=20,
    ),
}


@dataclass(frozen=True)
class ProviderConfig:
    """A provider that has a key configured and is ready to use."""

    name: str
    base_url: str
    api_key: str
    model: str
    tpm: int = 0  # tokens per minute we allow ourselves (0 = unlimited)
    rpm: int = 0  # requests per minute (0 = unlimited)


@dataclass(frozen=True)
class SandboxConfig:
    """Resource limits for the Docker sandbox."""

    image: str = "autopilot-sandbox:latest"
    memory: str = "2g"
    cpus: float = 2.0
    pids_limit: int = 512


@dataclass(frozen=True)
class Config:
    """Top-level configuration."""

    providers: list[ProviderConfig] = field(default_factory=list)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    github_token: str | None = None


def free_key_help() -> str:
    """Human-readable list of where to get free keys."""
    return "\n".join(f"  - {s.key_env}: {s.key_url}" for s in PROVIDERS.values())


def load_providers() -> list[ProviderConfig]:
    """Load providers in PROVIDER_ORDER, skipping any without an API key."""
    order = os.getenv("PROVIDER_ORDER", "gemini,groq,openrouter")
    result: list[ProviderConfig] = []
    for raw in order.split(","):
        name = raw.strip().lower()
        if not name:
            continue
        spec = PROVIDERS.get(name)
        if spec is None:
            raise ConfigError(
                f"Unknown provider '{name}' in PROVIDER_ORDER. "
                f"Allowed (free only): {', '.join(PROVIDERS)}"
            )
        key = os.getenv(spec.key_env, "").strip()
        if not key:
            continue
        model = os.getenv(spec.model_env, "").strip() or spec.default_model
        if name == "openrouter" and not model.endswith(":free"):
            raise ConfigError(
                f"OPENROUTER_MODEL='{model}' does not end in ':free'. "
                "Only free OpenRouter models are allowed."
            )
        prefix = name.upper()
        tpm = _int_env(f"{prefix}_TPM", spec.default_tpm)
        rpm = _int_env(f"{prefix}_RPM", spec.default_rpm)
        result.append(ProviderConfig(name, spec.base_url, key, model, tpm=tpm, rpm=rpm))
    return result


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return max(int(raw), 0)
    except ValueError as exc:
        raise ConfigError(f"{name}='{raw}' must be a whole number (0 = no limit).") from exc


def load_sandbox() -> SandboxConfig:
    """Load sandbox limits from the environment."""
    return SandboxConfig(
        image=os.getenv("SANDBOX_IMAGE", "autopilot-sandbox:latest"),
        memory=os.getenv("SANDBOX_MEMORY", "2g"),
        cpus=float(os.getenv("SANDBOX_CPUS", "2")),
        pids_limit=int(os.getenv("SANDBOX_PIDS_LIMIT", "512")),
    )


def load_config() -> Config:
    """Load the full configuration."""
    return Config(
        providers=load_providers(),
        sandbox=load_sandbox(),
        github_token=os.getenv("GITHUB_TOKEN") or None,
    )
