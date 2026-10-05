from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from .config import RESOURCE_ROOT


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def json_load(value: str | None, default: Any = None) -> Any:
    if value is None:
        return default
    return json.loads(value)


class Store:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self.connect() as db:
            db.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    goal_json TEXT NOT NULL,
                    plan_json TEXT,
                    policy_version TEXT NOT NULL,
                    counters_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL REFERENCES runs(id),
                    occurred_at TEXT NOT NULL,
                    step TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    arguments_json TEXT NOT NULL DEFAULT '{}',
                    result_json TEXT NOT NULL DEFAULT '{}',
                    duration_ms INTEGER,
                    error_status TEXT,
                    model TEXT,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    cost_usd REAL
                );
                CREATE INDEX IF NOT EXISTS events_run_idx ON events(run_id, id);
                CREATE TABLE IF NOT EXISTS leads (
                    run_id TEXT NOT NULL REFERENCES runs(id),
                    lead_id TEXT NOT NULL,
                    fields_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    reason TEXT,
                    draft_json TEXT,
                    judgments_json TEXT NOT NULL DEFAULT '{}',
                    review_json TEXT,
                    PRIMARY KEY (run_id, lead_id)
                );
                CREATE TABLE IF NOT EXISTS exports (
                    run_id TEXT PRIMARY KEY REFERENCES runs(id),
                    job_id TEXT UNIQUE NOT NULL,
                    status TEXT NOT NULL,
                    poll_count INTEGER NOT NULL DEFAULT 0,
                    fail_once INTEGER NOT NULL DEFAULT 1,
                    leads_json TEXT
                );
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL REFERENCES runs(id),
                    lead_id TEXT,
                    correction TEXT NOT NULL,
                    reason_code TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS policies (
                    version TEXT PRIMARY KEY,
                    config_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    promoted_at TEXT
                );
                CREATE TABLE IF NOT EXISTS candidates (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES runs(id),
                    proposal_json TEXT NOT NULL,
                    evaluation_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS outbox (
                    idempotency_key TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES runs(id),
                    lead_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS outcomes (
                    external_event_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES runs(id),
                    lead_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    reply_class TEXT,
                    probabilities_json TEXT,
                    text_hash TEXT
                );
                CREATE TABLE IF NOT EXISTS suppressions (
                    suppression_key TEXT PRIMARY KEY,
                    reason TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    run_id TEXT NOT NULL REFERENCES runs(id)
                );
                CREATE TABLE IF NOT EXISTS candidate_observations (
                    candidate_id TEXT NOT NULL REFERENCES candidates(id),
                    external_event_id TEXT NOT NULL REFERENCES outcomes(external_event_id),
                    candidate_approved INTEGER NOT NULL,
                    baseline_approved INTEGER NOT NULL,
                    adverse INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (candidate_id, external_event_id)
                );
                CREATE INDEX IF NOT EXISTS candidate_observations_candidate_idx
                    ON candidate_observations(candidate_id, created_at);
                """
            )
            policy = RESOURCE_ROOT / "config" / "policy_v1.json"
            if policy.exists():
                config = json_load(policy.read_text(encoding="utf-8"))
                db.execute(
                    "INSERT OR IGNORE INTO policies(version, config_json, created_at, promoted_at) VALUES (?, ?, ?, ?)",
                    (config["version"], json_dump(config), now_iso(), now_iso()),
                )

    def create_run(self, run_id: str, goal: dict[str, Any], policy_version: str) -> None:
        stamp = now_iso()
        with self.connect() as db:
            db.execute(
                "INSERT INTO runs(id, created_at, updated_at, status, goal_json, policy_version, counters_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (run_id, stamp, stamp, "planning", json_dump(goal), policy_version, json_dump({"tool_calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0})),
            )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if not row:
            return None
        return {
            "id": row["id"], "created_at": row["created_at"], "updated_at": row["updated_at"],
            "status": row["status"], "goal": json_load(row["goal_json"], {}),
            "plan": json_load(row["plan_json"], None), "policy_version": row["policy_version"],
            "counters": json_load(row["counters_json"], {}),
        }

    def update_run(self, run_id: str, *, status: str | None = None, plan: dict[str, Any] | None = None, counters: dict[str, Any] | None = None) -> None:
        assignments = ["updated_at = ?"]
        values: list[Any] = [now_iso()]
        if status is not None:
            assignments.append("status = ?")
            values.append(status)
        if plan is not None:
            assignments.append("plan_json = ?")
            values.append(json_dump(plan))
        if counters is not None:
            assignments.append("counters_json = ?")
            values.append(json_dump(counters))
        values.append(run_id)
        with self.connect() as db:
            db.execute(f"UPDATE runs SET {', '.join(assignments)} WHERE id = ?", values)

    def add_event(self, run_id: str, *, step: str, summary: str, arguments: dict[str, Any] | None = None,
                  result: dict[str, Any] | None = None, duration_ms: int | None = None,
                  error_status: str | None = None, model: str | None = None,
                  input_tokens: int | None = None, output_tokens: int | None = None,
                  cost_usd: float | None = None) -> int:
        with self.connect() as db:
            cursor = db.execute(
                "INSERT INTO events(run_id, occurred_at, step, summary, arguments_json, result_json, duration_ms, error_status, model, input_tokens, output_tokens, cost_usd) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (run_id, now_iso(), step, summary, json_dump(arguments or {}), json_dump(result or {}), duration_ms,
                 error_status, model, input_tokens, output_tokens, cost_usd),
            )
            return int(cursor.lastrowid)

    def list_events(self, run_id: str, limit: int = 500) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute("SELECT * FROM events WHERE run_id = ? ORDER BY id LIMIT ?", (run_id, limit)).fetchall()
        return [{"id": row["id"], "occurred_at": row["occurred_at"], "step": row["step"], "summary": row["summary"],
                 "arguments": json_load(row["arguments_json"], {}), "result": json_load(row["result_json"], {}),
                 "duration_ms": row["duration_ms"], "error_status": row["error_status"], "model": row["model"],
                 "input_tokens": row["input_tokens"], "output_tokens": row["output_tokens"], "cost_usd": row["cost_usd"]} for row in rows]

    def save_lead(self, run_id: str, lead_id: str, fields: dict[str, str], metadata: dict[str, Any], status: str, reason: str | None = None,
                  draft: dict[str, Any] | None = None, judgments: dict[str, Any] | None = None, review: dict[str, Any] | None = None) -> None:
        with self.connect() as db:
            db.execute(
                """INSERT INTO leads(run_id, lead_id, fields_json, metadata_json, status, reason, draft_json, judgments_json, review_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(run_id, lead_id) DO UPDATE SET fields_json=excluded.fields_json, metadata_json=excluded.metadata_json,
                   status=excluded.status, reason=excluded.reason, draft_json=excluded.draft_json, judgments_json=excluded.judgments_json, review_json=excluded.review_json""",
                (run_id, lead_id, json_dump(fields), json_dump(metadata), status, reason,
                 json_dump(draft) if draft is not None else None, json_dump(judgments or {}),
                 json_dump(review) if review is not None else None),
            )

    def get_lead(self, run_id: str, lead_id: str) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM leads WHERE run_id = ? AND lead_id = ?", (run_id, lead_id)).fetchone()
        return self._lead_row(row) if row else None

    def list_leads(self, run_id: str) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute("SELECT * FROM leads WHERE run_id = ? ORDER BY lead_id", (run_id,)).fetchall()
        return [self._lead_row(row) for row in rows]

    @staticmethod
    def _lead_row(row: sqlite3.Row) -> dict[str, Any]:
        return {"lead_id": row["lead_id"], "fields": json_load(row["fields_json"], {}),
                "metadata": json_load(row["metadata_json"], {}), "status": row["status"], "reason": row["reason"],
                "draft": json_load(row["draft_json"], None), "judgments": json_load(row["judgments_json"], {}),
                "review": json_load(row["review_json"], None)}

    def create_or_get_export(self, run_id: str, job_id: str) -> dict[str, Any]:
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO exports(run_id, job_id, status, poll_count, fail_once) VALUES (?, ?, 'queued', 0, 1)", (run_id, job_id))
            row = db.execute("SELECT * FROM exports WHERE run_id = ?", (run_id,)).fetchone()
        return dict(row)

    def get_export(self, run_id: str) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM exports WHERE run_id = ?", (run_id,)).fetchone()
        if not row:
            return None
        item = dict(row)
        item["leads"] = json_load(item.pop("leads_json"), None)
        return item

    def update_export(self, run_id: str, *, status: str, poll_count: int, leads: list[dict[str, Any]] | None = None, fail_once: int = 0) -> None:
        with self.connect() as db:
            db.execute("UPDATE exports SET status = ?, poll_count = ?, fail_once = ?, leads_json = COALESCE(?, leads_json) WHERE run_id = ?",
                       (status, poll_count, fail_once, json_dump(leads) if leads is not None else None, run_id))

    def add_feedback(self, run_id: str, lead_id: str | None, correction: str, reason_code: str) -> None:
        with self.connect() as db:
            db.execute("INSERT INTO feedback(run_id, lead_id, correction, reason_code, created_at) VALUES (?, ?, ?, ?, ?)",
                       (run_id, lead_id, correction[:700], reason_code[:80], now_iso()))

    def list_feedback(self, limit: int = 100) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute("SELECT * FROM feedback ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [{"run_id": row["run_id"], "lead_id": row["lead_id"], "correction": row["correction"],
                 "reason_code": row["reason_code"], "created_at": row["created_at"]} for row in rows]

    def feedback_summary(self) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute("SELECT reason_code, COUNT(*) AS n FROM feedback GROUP BY reason_code ORDER BY n DESC").fetchall()
        return [{"reason_code": row["reason_code"], "count": row["n"]} for row in rows]

    def active_policy(self) -> dict[str, Any]:
        with self.connect() as db:
            row = db.execute("SELECT * FROM policies WHERE promoted_at IS NOT NULL ORDER BY promoted_at DESC LIMIT 1").fetchone()
        if not row:
            raise RuntimeError("No active policy version is installed")
        return json_load(row["config_json"], {})

    def list_policies(self) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute("SELECT version, created_at, promoted_at FROM policies ORDER BY created_at").fetchall()
        return [dict(row) for row in rows]

    def save_candidate(self, candidate_id: str, run_id: str, proposal: dict[str, Any], evaluation: dict[str, Any], status: str) -> None:
        with self.connect() as db:
            db.execute("INSERT INTO candidates(id, run_id, proposal_json, evaluation_json, status, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                       (candidate_id, run_id, json_dump(proposal), json_dump(evaluation), status, now_iso()))

    def get_candidate(self, candidate_id: str) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM candidates WHERE id = ?", (candidate_id,)).fetchone()
        if not row:
            return None
        return {"id": row["id"], "run_id": row["run_id"], "proposal": json_load(row["proposal_json"], {}),
                "evaluation": json_load(row["evaluation_json"], {}), "status": row["status"],
                "created_at": row["created_at"], "canary": self.candidate_observation_summary(row["id"])}

    def update_candidate_status(self, candidate_id: str, status: str) -> None:
        with self.connect() as db:
            db.execute("UPDATE candidates SET status = ? WHERE id = ?", (status, candidate_id))

    def promote_policy(self, config: dict[str, Any]) -> None:
        with self.connect() as db:
            db.execute("UPDATE policies SET promoted_at = NULL")
            db.execute("INSERT OR REPLACE INTO policies(version, config_json, created_at, promoted_at) VALUES (?, ?, ?, ?)",
                       (config["version"], json_dump(config), now_iso(), now_iso()))

    def sandbox_outbox(self, run_id: str, lead_id: str, payload: dict[str, Any]) -> bool:
        key = f"{run_id}:{lead_id}"
        with self.connect() as db:
            cursor = db.execute("INSERT OR IGNORE INTO outbox(idempotency_key, run_id, lead_id, status, created_at, payload_json) VALUES (?, ?, ?, 'sandbox_recorded', ?, ?)",
                                (key, run_id, lead_id, now_iso(), json_dump(payload)))
        return cursor.rowcount > 0

    def list_outbox(self, run_id: str | None = None) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute("SELECT * FROM outbox WHERE (? IS NULL OR run_id = ?) ORDER BY created_at", (run_id, run_id)).fetchall()
        return [{"run_id": row["run_id"], "lead_id": row["lead_id"], "status": row["status"],
                 "created_at": row["created_at"], "payload": json_load(row["payload_json"], {})} for row in rows]

    def add_outcome(self, external_event_id: str, run_id: str, lead_id: str, event_type: str,
                    occurred_at: str, reply_class: str | None, probabilities: dict[str, float] | None, text_hash: str | None) -> bool:
        with self.connect() as db:
            cursor = db.execute("INSERT OR IGNORE INTO outcomes(external_event_id, run_id, lead_id, event_type, occurred_at, reply_class, probabilities_json, text_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                                (external_event_id, run_id, lead_id, event_type, occurred_at, reply_class,
                                 json_dump(probabilities) if probabilities is not None else None, text_hash))
        return cursor.rowcount > 0

    def outcome_exists(self, external_event_id: str) -> bool:
        with self.connect() as db:
            row = db.execute("SELECT 1 FROM outcomes WHERE external_event_id = ?", (external_event_id,)).fetchone()
        return row is not None

    def list_candidates(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute("SELECT * FROM candidates ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [{"id": row["id"], "run_id": row["run_id"],
                 "proposal": json_load(row["proposal_json"], {}),
                 "evaluation": json_load(row["evaluation_json"], {}),
                 "status": row["status"], "created_at": row["created_at"],
                 "canary": self.candidate_observation_summary(row["id"])} for row in rows]

    def add_candidate_observation(self, candidate_id: str, external_event_id: str, *,
                                  candidate_approved: bool, baseline_approved: bool,
                                  adverse: bool) -> bool:
        with self.connect() as db:
            cursor = db.execute(
                """INSERT OR IGNORE INTO candidate_observations(
                       candidate_id, external_event_id, candidate_approved,
                       baseline_approved, adverse, created_at
                   ) VALUES (?, ?, ?, ?, ?, ?)""",
                (candidate_id, external_event_id, int(candidate_approved),
                 int(baseline_approved), int(adverse), now_iso()),
            )
        return cursor.rowcount > 0

    def candidate_observation_summary(self, candidate_id: str) -> dict[str, Any]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT candidate_approved, baseline_approved, adverse FROM candidate_observations WHERE candidate_id = ?",
                (candidate_id,),
            ).fetchall()
        candidate_rows = [row for row in rows if row["candidate_approved"]]
        baseline_rows = [row for row in rows if row["baseline_approved"]]
        candidate_adverse = sum(bool(row["adverse"]) for row in candidate_rows)
        baseline_adverse = sum(bool(row["adverse"]) for row in baseline_rows)
        return {
            "observed_events": len(rows),
            "candidate_observations": len(candidate_rows),
            "candidate_adverse": candidate_adverse,
            "candidate_adverse_rate": candidate_adverse / len(candidate_rows) if candidate_rows else None,
            "baseline_observations": len(baseline_rows),
            "baseline_adverse": baseline_adverse,
            "baseline_adverse_rate": baseline_adverse / len(baseline_rows) if baseline_rows else None,
        }

    def add_suppression(self, key: str, reason: str, run_id: str) -> None:
        if not key:
            return
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO suppressions(suppression_key, reason, created_at, run_id) VALUES (?, ?, ?, ?)",
                       (key.strip().lower(), reason, now_iso(), run_id))

    def is_suppressed(self, key: str) -> bool:
        if not key:
            return False
        with self.connect() as db:
            row = db.execute("SELECT 1 FROM suppressions WHERE suppression_key = ?", (key.strip().lower(),)).fetchone()
        return row is not None

    def outcomes_summary(self) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute("SELECT event_type, reply_class, COUNT(*) AS n FROM outcomes GROUP BY event_type, reply_class").fetchall()
        return [dict(row) for row in rows]

    def outcome_count(self, run_id: str) -> int:
        with self.connect() as db:
            row = db.execute("SELECT COUNT(*) AS n FROM outcomes WHERE run_id = ? AND event_type IN ('reply', 'bounce', 'conversion')", (run_id,)).fetchone()
        return int(row["n"] if row else 0)

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute("SELECT id, created_at, updated_at, status, policy_version FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(row) for row in rows]
