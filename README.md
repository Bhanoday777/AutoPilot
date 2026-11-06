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

## Usage

```powershell
# Fix the toy bug (a local folder is copied into the sandbox)
python -m autopilot.cli fix --local examples/toy_repo --issue examples/toy_repo/ISSUE.md

# Fix a bug in a public repo (cloned into the sandbox, dependencies installed)
python -m autopilot.cli fix --repo https://github.com/user/repo.git --issue path/to/issue.md --ref main

# Options: --max-steps 25 (hard cap on LLM calls per run)
python -m pytest            # offline unit tests, no LLM calls
```

Each run writes `runs/<timestamp>/trajectory.json` (every step), `final.diff` and `result.json`.

## How the agent works

The agent loop sends the issue to a free LLM along with a set of tools, runs each tool call
inside the Docker sandbox and feeds the result back, until the agent calls `finish` or hits
the step cap. The system prompt enforces a strict workflow:

1. **Understand** the issue.
2. **Explore** the code (list, search, read).
3. **Reproduce:** write `tests/test_autopilot_repro.py` and confirm it **fails**.
4. **Fix:** the smallest change to the source code. Existing tests can't be edited.
5. **Verify:** the repro test and the full suite pass.
6. **Finish** with a summary of the root cause, the fix and the test evidence.

After the agent stops, the CLI **re-runs the full test suite itself** instead of trusting the agent's claim.

The loop also steers the agent away from patterns seen in real runs: re-reading unchanged code
gets a note to act instead, a fix that breaks an existing test gets a note naming that test, and a
long exploration with no repro test yet gets one nudge to write it.

| Tool | What it does |
|---|---|
| `list_files` | directory tree (skips .git, caches, venvs) |
| `read_file` | file with line numbers, 120 lines per call |
| `search_code` | ripgrep, max 30 matches |
| `edit_file` | exact, unique string replacement; refuses to edit existing tests |
| `create_file` | new files only (for the repro test) |
| `run_tests` | pytest (`-q --tb=short`), full suite or one target; output trimmed to the first failure + summary |
| `run_command` | allowlisted: python, pytest, pip install, rg, ls, cat, head, tail, grep, find, git status/diff/log |
| `git_diff` | current changes |
| `finish` | ends the run with a summary |

**Staying inside free-tier limits.** The LLM client tracks each provider's requests and tokens
over a 60-second window and waits *before* a request would exceed its free limit (defaults in
`config.py`, overridable with `GROQ_TPM`, `GEMINI_RPM`, etc.). If a 429 still arrives, it waits as
long as the provider asks; if that's longer than a minute (daily quota gone) it falls through to
the next provider. When a model calls a tool that doesn't exist, the request is retried with a
short correction. Lines like `[llm] groq: pacing 12.3s` mean it is waiting, not stuck.

To save free-tier tokens, tool outputs are truncated (head + tail), and only the last 4 tool
outputs stay in full. Older ones are replaced with a one-line summary, and the long arguments of
those older calls (file contents, edit text) are stubbed too. Tools accept common argument-name
guesses (`line_start`, `file_path`, …), and if the model writes its summary as plain text right
after a passing full suite with no changes since, that is accepted as `finish`.

## Layout

```
autopilot/
  config.py      # free providers (PROVIDER_ORDER), :free enforcement, sandbox limits
  llm.py         # LLM client: backoff, provider fall-through, usage tracking
  sandbox.py     # Docker sandbox: exec with timeout, copy_in, install, tests, diff
  tools.py       # agent tools (JSON schemas + implementations) with path/command safety
  prompts.py     # system prompt (reproduce -> fix -> verify) + task template
  agent.py       # the agent loop, context compaction, AgentResult
  trajectory.py  # step-by-step log in runs/<timestamp>/
  cli.py         # python -m autopilot.cli fix ...
docker/Dockerfile.sandbox
examples/toy_repo/   # planted bug (median of an even-length list) + ISSUE.md
scripts/smoke_test.py
tests/               # offline unit tests (fake sandbox + fake LLM, no API calls)
```

## Roadmap

- [x] Day 1: LLM client + sandbox
- [x] Day 2: agent loop + tools
- [ ] Day 3: real bugs + prompt tuning
- [ ] Day 4: verifier + open PR
- [ ] Day 5: eval set of 8–10 bugs
- [ ] Day 6: fix failure patterns + results table
- [ ] Day 7: README polish + demo video
