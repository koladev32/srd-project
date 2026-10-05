from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Lead:
    lead_id: str
    fields: dict[str, str]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def email(self) -> str:
        return self.fields.get("Email", "").strip().lower()

    @property
    def full_name(self) -> str:
        return (self.fields.get("Full Name") or " ".join(
            part for part in (self.fields.get("First Name"), self.fields.get("Last Name")) if part
        )).strip()

    @property
    def profile_key(self) -> str:
        return (self.fields.get("Linkedin URL Unique ID") or self.fields.get("LinkedIn URL", "")).strip().lower()

    def public_summary(self) -> dict[str, str]:
        """A trace-safe summary: no names, emails, URLs, or raw profile text."""
        return {
            "lead_id": self.lead_id,
            "matches_filters": _clean(self.fields.get("Matches Filters", "")),
            "email_status": _clean(self.fields.get("Email Status", "")),
            "current_job_present": str(bool(self.fields.get("Current Job", "").strip())).lower(),
            "company_industry": self.fields.get("Company Industry", "")[:80],
        }


def _clean(value: Any) -> str:
    return str(value or "").strip().strip('"').strip()


def normalize_row(row: dict[str, Any], index: int, source: str) -> Lead:
    fields = {str(key).strip(): _clean(value) for key, value in row.items() if key and not str(key).startswith("Fixture ")}
    fixture_id = _clean(row.get("Fixture ID"))
    if fixture_id:
        lead_id = fixture_id
    else:
        provider_id = _clean(row.get("Linkedin URL Unique ID"))
        token = provider_id or str(index)
        digest = hashlib.sha256(f"{source}:{token}".encode()).hexdigest()[:10]
        lead_id = f"csv-{digest}"
    metadata = {
        "source": source,
        "fixture_group": _clean(row.get("Fixture Group")),
        "fixture_reply": _clean(row.get("Fixture Reply")),
        "source_timestamp": _clean(row.get("Source Updated At")) or None,
        "verification_result": _clean(row.get("Fixture Verification")) or None,
    }
    return Lead(lead_id=lead_id, fields=fields, metadata=metadata)


def read_csv(path: Path, source: str) -> list[Lead]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = csv.DictReader(handle)
        return [normalize_row(row, index, source) for index, row in enumerate(rows, start=1)]


def load_simulator_leads(public_csv: Path, synthetic_csv: Path, limit: int = 25) -> list[Lead]:
    leads: list[Lead] = []
    if public_csv.exists():
        leads.extend(read_csv(public_csv, "evaboot_public_sample_simulated"))
    leads.extend(read_csv(synthetic_csv, "synthetic_fixture"))
    unique: dict[str, Lead] = {}
    for lead in leads:
        unique.setdefault(lead.lead_id, lead)
    return list(unique.values())[:limit]
