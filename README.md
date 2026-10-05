# Evaboot Agent Runtime

A bounded SDR prospecting and learning runtime. It is not an official Evaboot integration and does not imply Evaboot lacks similar capabilities.

## What it does

- A real OpenAI planner turns a prospecting goal into a typed plan. The operator sees the scope and budgets before execution.
- The UI can route GPT-6 Luna through either the OpenAI API key or the signed-in Codex/ChatGPT subscription. Python drives an explicit plan → act → observe → revise loop over typed search, export, poll, inspect, verify, and feedback tools.
- The default simulator combines Evaboot's public sample export (when downloaded locally) with synthetic fixtures. Its asynchronous export includes an injected transient timeout and an incomplete poll so recovery is visible. It does not access LinkedIn.
- Python gates filter mismatches, unsafe/missing email status, duplicates, suppressions, missing evidence, stale evidence when enabled, citations, tool permissions, retries, and token-cost estimates.
- TypeSafe Jev supplies separate Noul checks for role fit, company fit, claim support, and contradictions. Reply intent uses Choice's complete probability distribution; opt-out text and unclear results suppress the prospect.
- Assisted mode pauses at human review. Autonomous mode skips ambiguous work rather than asking for approval. Both modes can only record sandbox deliveries.
- SQLite persists runs, tool observations, judgments, feedback, policies, outcome events, and an idempotent sandbox outbox.
- An improvement action uses Python to calibrate question thresholds on the calibration split (including reply fixtures), asks OpenAI to propose one whitelisted targeting change, then freezes the thresholds before comparing baseline and candidate on a locked synthetic held-out set. The report includes precision, false-approval denominator, coverage, skip rate, Wilson intervals, per-question reliability bins and Brier score, and calibration/held-out reply confusion matrices. Assisted candidates require explicit approval. Autonomous candidates enter a post-creation shadow canary: only new candidate-scoped reply, bounce, and conversion observations count, and adverse-rate regression is bounded before promotion. OpenAI cannot alter thresholds, labels, or acceptance rules.

The model is pinned to GPT-6 Luna for both backends; no other model can be selected. Luna supports structured output and function calling ([model documentation](https://developers.openai.com/api/docs/models/gpt-6-luna)). Jev integration uses TypeSafe's current Python SDK and `TYPESAFE_API_KEY` ([SDK documentation](https://docs.typesafe.ai/sdk/python)). The optional live Evaboot adapter uses the documented MCP server ([Evaboot MCP docs](https://docs.evaboot.com/mcp-server)).

## Run locally

Python 3.11 or newer is required. Automated tests are offline and do not need credentials:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install -e . --no-deps
pytest
```

For subscription mode, sign in to Codex once with `codex login`; the health panel checks for ChatGPT sign-in. The Codex option launches an ephemeral `codex exec` pinned to GPT-6 Luna, in a temporary directory with a read-only sandbox and user MCP configuration disabled. It never falls back to an API key. Codex usage consumes the applicable ChatGPT plan allowance/credits and is not a free API key; access and limits depend on the account and workspace.

For the separate API-billed option, add `OPENAI_KEY` (or the compatibility alias `OPENAI_API_KEY`) to `.env`. Both options also require `TYPESAFE_API_KEY` for live Jev judgments. `.env` is git-ignored. The UI defaults to Codex subscription mode. Its usage ceiling is an estimate at GPT-6 Luna API-equivalent rates, not a subscription bill or a replacement for Codex account limits. Start the local-only server with:

```bash
evaboot-agent
```

Then open <http://127.0.0.1:8000>. The dashboard only reports whether backends are available; it never displays credential values. Interactive mode has no canned Luna or Jev fallback. API mode requires an API credential; subscription mode requires Codex ChatGPT sign-in. Both require TypeSafe for Jev judgments.

Keep credentials only in the ignored `.env` file or an approved secret manager. See `SECURITY.md` for the local and CI secret-scanning workflow.

Mutable state defaults to `.evaboot-agent/runtime.sqlite3` in the working directory. Existing checkouts with the legacy `data/runtime.sqlite3` continue using it unless `DATABASE_PATH` is set.

For an optional live Evaboot export, install the adapter and configure `EVABOOT_API_KEY`:

```bash
python -m pip install -e '.[live]'
```

Choose “Live Evaboot” and explicitly authorize the small export in the form. The adapter checks export quota and credit balance before starting. It does not build a scraper and cannot send email. A separate live delivery integration is intentionally absent; delivery remains sandbox-only even when live Evaboot data is selected.

## 90-second walkthrough

1. Enter a narrow ICP, choose Assisted + CSV simulator, and create a plan. Review the search criteria and caps.
2. Approve the plan and run. The trace shows a simulated async export, a transient timeout, a bounded retry, and a later completion observation.
3. Open Prospects to inspect deterministic gates, Jev probabilities, source-field excerpts, and a draft. Approve or reject a draft; approval records a sandbox outbox event only.
4. Choose Learning and evaluate a candidate. The report shows calibration/test sample counts, baseline vs candidate, reply errors, uncertainty intervals, and acceptance failures. Assisted promotion requires human approval; autonomous promotion is gated by observed outcome volume. Rejected candidates leave the active policy unchanged.
5. Open the engineering report from the sidebar or visit `/api/report.md`.

## Boundaries and limitations

- The public sample file is loaded only if present at `.context/evaboot-sample-export.csv`; it is an ignored local file. Synthetic rows, policies, and UI assets are packaged under `src/evaboot_agent/resources/` so wheels and editable installs behave the same. Public sample prospects are never sent anything; there is no sender in this app.
- Main app interactions call real OpenAI and TypeSafe APIs. Deterministic agent/Jev doubles exist only in pytest. Test results are not model accuracy claims.
- The held-out evaluation data is small and synthetic. Even zero errors would not establish production reliability. Thresholds, labels, and acceptance criteria are versioned and outside the improvement agent's permissions.
- The live Evaboot adapter relies on the connected MCP tool schema and may need adjustment if the server's schema changes. Export credit quotas are checked immediately before export; verify calls may use credits too.
- No CRM writes, follow-up scheduler, real send provider, or production analytics are included. Live delivery, mass outreach, and LinkedIn scraping are out of scope.
- The local interface is intended for one operator on loopback, not multi-user deployment. Add authentication and operational controls before exposing it to a network.

## Engineering report and evaluation

The report at `/api/report.md` uses persisted run events and token-cost estimates; it labels simulator calls and fixture outcomes. The packaged evaluation dataset has separate calibration and held-out groups; related examples have distinct group IDs. Run `pytest` for offline gate, persistence, retry, evaluation, packaging, and idempotency checks. Calling the interactive improvement endpoint makes real Jev calls over the fixed evaluation fixtures; it is deliberately not run as part of offline tests.

### Direct agent evaluation

Run `evaboot-eval` to replay every packaged labeled prospect through the actual GPT-6 Luna planner, tool selector, drafter, and TypeSafe Jev. It defaults to the signed-in Codex subscription backend. Use `evaboot-eval --case syn-grounded --case syn-headline-only` for a small run. This command uses one isolated SQLite database and one-row simulator export per case, with assisted mode and sandbox delivery. It does not send messages. The first draft stops at human review.

The command writes each trace database and scored result, plus `results.json`, `report.md`, and `review_queue.md`, under `.context/evals/<UTC timestamp>/`. Pass `--output-dir` to choose another new directory. If interrupted, run the same command with `--resume`; add `--retry-incomplete` to continue cases that ended without a usable lead decision. A plain `--resume` also rescans saved traces with the current scoring code without making model calls. It compares each qualification decision with a fixture label; checks tool order and budgets, source excerpt existence, errors, review boundary, and no delivery; and classifies fixture replies separately through Jev. The review queue shows each draft and its cited source fields but hides the expected label from the reviewer. A cited excerpt existing in a field is a syntax check, not proof that the entire claim is true; Jev's in-run claim judgment and a human reviewer address meaning. Synthetic fixture labels and Jev scores are not independent production outcome evidence.

## Engineering documentation

- `AGENTS.md` defines repository invariants and the workflow for coding agents.
- `docs/architecture.md` documents orchestration, persistence, and policy lifecycle.
- `docs/threat-model.md` records trust boundaries, controls, and residual sandbox risk.
- `docs/adr/0001-bounded-agent-runtime.md` explains why deterministic Python owns authority.
- `.conductor/settings.toml` provides reproducible setup, test, and per-workspace development-server scripts.
