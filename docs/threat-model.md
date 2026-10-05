# Threat model

## Assets

- OpenAI, TypeSafe, and Evaboot credentials
- Lead contact and profile data
- Suppression decisions and reviewer feedback
- Policy versions, evaluation labels, and acceptance criteria
- Operator workstation files and network access

## Trust boundaries

External lead fields, replies, MCP payloads, model output, and reviewer text are untrusted. The FastAPI/Python process is the control plane. OpenAI, Codex, TypeSafe, and Evaboot are external processors. SQLite and the sandbox outbox are local persistence.

## Principal threats and controls

| Threat | Control |
|---|---|
| Prompt injection requests tools or secrets | Typed outputs, Python allowlists, untrusted-data instructions, no application secrets in Codex environment |
| Codex shell reads inherited credentials | Minimal subprocess environment plus `shell_environment_policy.inherit="none"` |
| Agent stops before completing deterministic work | `done` is withheld while actionable leads remain |
| Unsupported personalization | Exact excerpt validation, Jev claim checks, one bounded revision, human review |
| Unsafe or repeated contact | Safe-email gate, suppression ledger, duplicate gate, sandbox-only outbox |
| Policy overfits fixtures | Calibration/held-out split, fixed acceptance rules, candidate-scoped post-creation canary |
| Historical outcomes falsely validate a candidate | Candidate observation table keyed by candidate and event; occurrence must postdate candidate |
| Package works only from a checkout | UI, policy, and fixtures are wheel resources; CI installs and imports the wheel |

## Residual risk

The Codex CLI read-only sandbox is not equivalent to a dedicated VM or container confidentiality boundary. Subscription mode should run only on a workstation account without unrelated sensitive readable files. Production use should move agent execution to isolated compute with restricted egress and brokered credentials. Official OpenAI guidance notes that agent-generated code can access credentials and resources available in its environment: <https://developers.openai.com/api/docs/guides/agents-api/environments/security>.

The held-out set is intentionally small and synthetic. Passing it and the shadow canary does not establish production safety or model accuracy. Live delivery, multi-user exposure, and CRM writes remain out of scope.
