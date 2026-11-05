# Autopilot

An AI agent that takes a GitHub issue, reproduces the bug inside a Docker sandbox,
fixes it, verifies the fix with tests and opens a pull request. Scope: Python repos.

## Runs 100% on free resources

- **LLMs:** free tiers only, no credit card: Google Gemini (AI Studio), Groq and
  OpenRouter (`:free` models only, which `config.py` enforces). They are called through
  the open-source `openai` SDK as an OpenAI-compatible client, never OpenAI's paid API.
- **Sandbox:** Docker Desktop (free for personal use) and the official `python:3.11-slim` image.
- **GitHub:** the free REST API with a personal access token. No Actions minutes are needed.
- **Libraries:** open source from PyPI only (`openai`, `docker`, `python-dotenv`, `rich`, `PyGithub`).
- **Hosting:** none. Everything runs locally.

## Setup (Windows, PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r requirements.txt
Copy-Item .env.example .env         # then add at least one free API key
docker build -t autopilot-sandbox:latest -f docker/Dockerfile.sandbox .
python scripts/smoke_test.py --sandbox   # no API key needed
python scripts/smoke_test.py             # full check (2 LLM calls)
```

Free keys: [Gemini](https://aistudio.google.com/apikey) · [Groq](https://console.groq.com/keys) · [OpenRouter](https://openrouter.ai/keys)

## Layout

```
autopilot/
  config.py      # free providers (PROVIDER_ORDER), :free enforcement, sandbox limits
  llm.py         # LLM client: backoff, provider fall-through, usage tracking
  sandbox.py     # Docker sandbox: exec with timeout, copy_in, install, tests, diff
docker/Dockerfile.sandbox
examples/toy_repo/   # planted bug (median of an even-length list) + ISSUE.md
scripts/smoke_test.py
```

## Roadmap

- [x] Day 1: LLM client + sandbox
- [ ] Day 2: agent loop + tools
- [ ] Day 3: real bugs + prompt tuning
- [ ] Day 4: verifier + open PR
- [ ] Day 5: eval set of 8–10 bugs
- [ ] Day 6: fix failure patterns + results table
- [ ] Day 7: README polish + demo video
