from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from evaboot_agent.agents import AIResult
from evaboot_agent.config import Settings
from evaboot_agent.judgments import JevResult
from evaboot_agent.models import (EmailDraft, EvidenceRef, GoalRequest, LeadStatus,
                                  PersonalizedClaim, Plan, ToolAction, ToolName)
from evaboot_agent.runtime import Runtime
from evaboot_agent.store import Store


class OfflineAgent:
    def __init__(self, store: Store, *, invented: bool = False):
        self.store = store
        self.invented = invented

    def create_plan(self, request, *, run_id, prior_feedback):
        plan = Plan(summary="A small bounded finance-leader search plan.",
                    search_prompt="European finance leaders at fintech businesses",
                    target_criteria=["Finance leadership", "European fintech", "200+ employees"],
                    exclusions=request.audience_exclusions,
                    evidence_requirements=["Current role", "Company fit", "Cited claim"],
                    stopping_conditions=["Lead cap reached", "No permitted work remains"],
                    max_leads=request.max_leads, max_tool_calls=request.max_tool_calls,
                    max_search_revisions=request.max_search_revisions,
                    approval_required=request.execution_mode.value == "assisted")
        return AIResult(plan, "offline-test-double", 0, 0, 0, 0)

    def next_action(self, *, run_id, request, plan, state, allowed_tools):
        export = self.store.get_export(run_id)
        leads = self.store.list_leads(run_id)
        if ToolName.BUILD_SEARCH in allowed_tools and not state.get("search_builds"):
            tool = ToolName.BUILD_SEARCH
            return self._result(ToolAction(tool=tool, rationale="Build the bounded search query.", revised_search_prompt=plan.search_prompt))
        if ToolName.START_EXPORT in allowed_tools:
            return self._result(ToolAction(tool=ToolName.START_EXPORT, rationale="Start the bounded async export."))
        if ToolName.GET_EXPORT_STATUS in allowed_tools:
            return self._result(ToolAction(tool=ToolName.GET_EXPORT_STATUS, rationale="Observe the existing export job.", job_id=export["job_id"]))
        if ToolName.VERIFY_EMAIL in allowed_tools:
            lead = next(item for item in leads if item["status"] == LeadStatus.NEEDS_VERIFICATION.value)
            return self._result(ToolAction(tool=ToolName.VERIFY_EMAIL, rationale="Verify missing email status.", lead_id=lead["lead_id"]))
        if ToolName.INSPECT_LEAD in allowed_tools:
            lead = next(item for item in leads if item["status"] in {LeadStatus.ELIGIBLE.value, LeadStatus.NEEDS_RESEARCH.value})
            detail = "deep" if lead["status"] == LeadStatus.NEEDS_RESEARCH.value else "standard"
            return self._result(ToolAction(tool=ToolName.INSPECT_LEAD, rationale="Inspect current source evidence.",
                                           lead_id=lead["lead_id"], detail_level=detail))
        return self._result(ToolAction(tool=ToolName.DONE, rationale="No permitted work remains."))

    def draft_claims(self, *, run_id, request, lead, evidence, correction=None):
        industry = evidence.get("Company Industry", "Financial Services")
        if self.invented:
            text = "Your company serves 10000 customers across five continents."
            source_field = "Company Industry"
            excerpt = industry
        else:
            text = f"Your company operates in {industry}."
            source_field = "Company Industry"
            excerpt = industry
        draft = EmailDraft(subject="A question about the company", claims=[PersonalizedClaim(
            text=text, evidence=[EvidenceRef(field=source_field, excerpt=excerpt)])])
        return AIResult(draft, "offline-test-double", 0, 0, 0, 0)

    def propose_improvement(self, *, run_id, request, feedback, calibration_summary):
        return AIResult({"change_type": "exclude_stale_evidence",
                         "rationale": "Exclude old evidence from candidate qualification decisions.",
                         "expected_tradeoff": "Lower coverage on stale rows while reducing risky approvals."},
                        "offline-test-double", 0, 0, 0, 0)

    @staticmethod
    def _result(value):
        return AIResult(value, "offline-test-double", 0, 0, 0, 0)


class OfflineJev:
    def assess_lead(self, lead, *, goal, claims=None):
        fields = lead.fields
        current = fields.get("Current Job", "").lower()
        industry = fields.get("Company Industry", "").lower()
        headline = fields.get("Profile Headline", "").lower()
        role = 0.97 if any(term in current for term in ("finance", "cfo", "chief financial")) else (0.03 if current else 0.50)
        company = 0.97 if "financial" in industry else (0.03 if industry else 0.50)
        contradiction = 0.98 if ("sales manager" in current and ("cfo" in headline or "chief financial officer" in headline)) else 0.01
        answers = {
            "current_role_fit": {"type": "noul", "probability_yes": role},
            "company_fit": {"type": "noul", "probability_yes": company},
            "evidence_contradictions": {"type": "noul", "probability_yes": contradiction},
        }
        for index, claim in enumerate(claims or []):
            supported = "10000" not in claim.get("text", "") and all(
                ref.get("excerpt", "").casefold() in fields.get(ref.get("field", ""), "").casefold()
                for ref in claim.get("evidence", []))
            answers[f"claim_support_{index}"] = {"type": "noul", "probability_yes": 0.97 if supported else 0.03}
        return JevResult("offline-jev-double", answers, 0, 0, 1)

    def classify_reply(self, text):
        lower = text.lower()
        if any(term in lower for term in ("unsubscribe", "stop", "do not contact")):
            probs, choice = {"opt_out": .96, "unclear": .02, "negative": .01, "positive": .01}, "opt_out"
        elif "not a priority" in lower:
            probs, choice = {"opt_out": .01, "unclear": .04, "negative": .91, "positive": .04}, "negative"
        elif "maybe" in lower or "who is this" in lower or "remind me" in lower:
            probs, choice = {"opt_out": .02, "unclear": .88, "negative": .06, "positive": .04}, "unclear"
        else:
            probs, choice = {"opt_out": .01, "unclear": .04, "negative": .05, "positive": .90}, "positive"
        return JevResult("offline-jev-double", {"intent": {"type": "choice", "choice": choice,
                           "probabilities": probs, "confidence": max(probs.values())}}, 0, 0, 1)


@pytest.fixture
def runtime_factory(tmp_path, monkeypatch):
    from evaboot_agent import runtime as runtime_module

    monkeypatch.setattr(runtime_module, "has_openai_key", lambda: True)
    monkeypatch.setattr(runtime_module, "has_typesafe_key", lambda: True)

    def make(*, autonomous=False, invented=False, public_csv=None, max_leads=25):
        db_path = tmp_path / f"runtime-{len(list(tmp_path.glob('*.sqlite3')))}.sqlite3"
        settings = Settings(database_path=db_path, openai_model="gpt-6-luna",
                            openai_input_rate=.10, openai_output_rate=.50,
                            typesafe_model="jev-latest", typesafe_timeout_seconds=12,
                            public_csv_path=public_csv or (tmp_path / "no-public-sample.csv"))
        store = Store(db_path)
        agent = OfflineAgent(store, invented=invented)
        runtime = Runtime(settings, store, agent=agent, jev=OfflineJev())
        request = GoalRequest(goal="Finance leaders at European fintech companies with 200 or more employees",
                              execution_mode="autonomous" if autonomous else "assisted",
                              max_leads=max_leads, max_tool_calls=35, max_model_cost_usd=.25)
        return runtime, request
    return make
