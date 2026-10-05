# Security policy

## Secrets

Never commit API keys, tokens, exported lead data, local databases, evaluation traces, or authentication output. Copy `.env.example` to `.env` and keep real values only in the ignored local file or an approved secret manager.

Every push and pull request runs Gitleaks against the complete Git history. Run the same check locally before publishing:

```bash
gitleaks detect --source . --no-git --redact
gitleaks git . --redact
```

If a credential is exposed, revoke or rotate it immediately. Removing it from a later commit is not sufficient because it remains in Git history.

## Reporting a vulnerability

Report security issues privately to the repository owner. Do not include live credentials, personal lead data, or unredacted runtime traces in an issue.
