# ADR 0001: Python-owned bounded agent runtime

Status: accepted

## Context

Prospecting combines untrusted external text, probabilistic model decisions, paid tools, personal data, and actions that could affect real people. Prompt instructions alone are not an authorization system.

## Decision

Use models only for typed planning, action selection, drafting, and semantic estimation. Keep permissions, budgets, hard gates, idempotency, review checkpoints, policy evaluation, and delivery modes in deterministic Python. Persist each transition so a run can be inspected and replayed.

Codex subscription calls run ephemerally from a temporary directory with user configuration and rules disabled, a read-only sandbox, no inherited shell environment, and a minimal parent-process environment. Application credentials remain in the Python control plane.

Autonomous policy changes are shadow-canary candidates rather than immediately active policies. Only post-candidate observations may promote them.

## Consequences

- The runtime is more verbose than a free-form agent loop, but behavior is testable and auditable.
- New tools require explicit schemas, permission-state integration, redaction, and tests.
- The local Codex CLI sandbox reduces exposure but is not a VM boundary; sensitive production workloads need externally isolated compute and network policy.
- The runtime remains sandbox-delivery-only until a separate delivery authorization design is approved.
