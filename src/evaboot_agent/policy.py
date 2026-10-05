from __future__ import annotations

import re
from datetime import date, datetime, timezone
from typing import Any

from .normalizer import Lead


EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
OPTOUT_RE = re.compile(r"\b(stop|unsubscribe|opt\s*out|remove me|do not contact|don't contact|no further contact)\b", re.I)


def deterministic_gate(lead: Lead, *, seen_emails: set[str], seen_profiles: set[str],
                       user_suppressions: list[str], is_suppressed: bool) -> tuple[str, str | None]:
    """Hard gates. A missing value is unknown and never treated as a match."""
    match = lead.fields.get("Matches Filters", "").strip().strip('"').upper()
    if match != "YES":
        return "rejected", "filter_mismatch" if match == "NO" else "filter_match_unknown"

    email = lead.email
    if not email or not EMAIL_RE.match(email) or email.endswith((".invalid", ".test.invalid")):
        return "rejected", "missing_or_malformed_email"
    if is_suppressed or _matches_user_suppression(email, user_suppressions):
        return "suppressed", "suppression_list_match"
    if email in seen_emails or (lead.profile_key and lead.profile_key in seen_profiles):
        return "rejected", "duplicate_lead"

    status = lead.fields.get("Email Status", "").strip().lower()
    if status in {"riskier", "invalid", "unverified", "unknown", "unsafe", "bad"}:
        return "rejected", "email_not_safe"
    if not status:
        return "needs_verification", "email_status_missing"
    if status != "safe":
        return "rejected", "email_status_not_safe"
    return "eligible", None


def _matches_user_suppression(email: str, entries: list[str]) -> bool:
    email = email.lower().strip()
    domain = email.rsplit("@", 1)[-1] if "@" in email else ""
    for item in entries:
        token = item.lower().strip()
        if token and (token == email or token == domain or token == f"@{domain}"):
            return True
    return False


def validate_claim_references(claims: list[dict[str, Any]], fields: dict[str, str]) -> list[str]:
    failures: list[str] = []
    for index, claim in enumerate(claims):
        refs = claim.get("evidence", [])
        if not refs:
            failures.append(f"claim_{index}:missing_source_reference")
            continue
        for ref in refs:
            source = fields.get(ref.get("field", ""), "")
            excerpt = ref.get("excerpt", "").strip()
            if not source or not excerpt or excerpt.casefold() not in source.casefold():
                failures.append(f"claim_{index}:invalid_source_reference")
                break
    return failures


def required_evidence_missing(lead: Lead, *, require_current_job_match: bool = False) -> list[str]:
    required = ["Company Industry", "Company Employee Exact Count", "Company Location", "Company Description"]
    if require_current_job_match:
        required.insert(0, "Current Job")
    return [field for field in required if not lead.fields.get(field, "").strip()]


def evidence_is_stale(lead: Lead, *, today: date | None = None, max_age_days: int = 365) -> bool:
    timestamp = lead.metadata.get("source_timestamp") or lead.fields.get("Source Updated At")
    if not timestamp:
        return False
    try:
        parsed = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00")).date()
    except ValueError:
        return False
    return ((today or datetime.now(timezone.utc).date()) - parsed).days > max_age_days


def route_judgment(judgment: dict[str, Any], policy: dict[str, Any], *, missing_fields: list[str], stale: bool,
                   claim_count: int = 0) -> tuple[str, str]:
    answers = judgment.get("answers", {})
    thresholds = policy["question_thresholds"]
    role = answers.get("current_role_fit", {}).get("probability_yes")
    company = answers.get("company_fit", {}).get("probability_yes")
    contradiction = answers.get("evidence_contradictions", {}).get("probability_yes")
    claim_values = [answers.get(f"claim_support_{i}", {}).get("probability_yes") for i in range(claim_count)]

    if missing_fields:
        return "insufficient", "missing_evidence:" + ",".join(missing_fields)
    if stale:
        return "insufficient", "source_evidence_stale"
    if role is None or company is None or contradiction is None:
        return "deferred", "required_jev_answer_missing"
    if contradiction > thresholds["contradiction_risk_max"]:
        return "reject", "evidence_contradiction"
    if role <= 1 - thresholds["current_role_fit"]:
        return "reject", "current_role_mismatch"
    if company <= 1 - thresholds["company_fit"]:
        return "reject", "company_mismatch"
    if any(value is not None and value < 1 - thresholds["claim_support"] for value in claim_values):
        return "revise", "claim_not_supported"
    if (role < thresholds["current_role_fit"] or company < thresholds["company_fit"] or
            any(value is None or value < thresholds["claim_support"] for value in claim_values)):
        return "insufficient", "judgment_below_action_threshold"
    return "qualified", "all_semantic_thresholds_passed"


def classify_reply(judgment: dict[str, Any], policy: dict[str, Any], reply_text: str) -> tuple[str, str]:
    if OPTOUT_RE.search(reply_text):
        return "opt_out", "deterministic_opt_out_signal"
    answer = judgment.get("answers", {}).get("intent", {})
    probabilities = answer.get("probabilities", {})
    if not probabilities:
        return "unclear", "reply_probabilities_missing"
    opt_out_floor = policy["reply_policy"]["opt_out_possible_floor"]
    if float(probabilities.get("opt_out", 0)) >= opt_out_floor:
        return "opt_out", "opt_out_probability_requires_pause"
    choice = answer.get("choice", "unclear")
    max_probability = max(float(value) for value in probabilities.values())
    min_clear = policy["reply_policy"]["clear_label_probability"]
    if choice not in {"positive", "negative"} or max_probability < min_clear:
        return "unclear", "ambiguous_reply_suppresses_contact"
    return choice, "clear_jev_reply_intent"
