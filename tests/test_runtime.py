from pathlib import Path

import pytest

from evaboot_agent.models import GoalRequest, LeadStatus, Plan, ReviewRequest, ToolName
from evaboot_agent.runtime import RuntimeRequestError


def test_assisted_run_recovers_export_and_requires_human_checkpoint(runtime_factory):
    runtime, request=runtime_factory(max_leads=1)
    run=runtime.create_run(request)
    result=runtime.execute(run["id"])
    assert result["status"] == "awaiting_human_review"
    lead=next(item for item in result["leads"] if item["status"] == LeadStatus.AWAITING_REVIEW.value)
    assert lead["draft"]["evidence"]
    assert "ava.quinn@northstar.test" not in str(lead)
    assert "ava.quinn@northstar.test" not in str(result)
    assert any(event["error_status"] == "temporary_timeout" for event in result["events"])
    assert any(event["step"] == "drafter.claims_created" for event in result["events"])
    replay=runtime.replay(result["id"],lead["lead_id"])
    assert replay["lead"]["lead_id"]==lead["lead_id"]
    assert replay["events"]
    assert "ava.quinn@northstar.test" not in str(replay)
    approved=runtime.review(result["id"],lead["lead_id"],ReviewRequest(decision="approve"))
    assert approved["leads"][0]["status"] == LeadStatus.SANDBOXED.value or any(x["status"] == LeadStatus.APPROVED.value for x in approved["leads"])
    assert len(runtime.store.list_outbox(result["id"])) == 1
    assert runtime.store.list_outbox(result["id"])[0]["payload"]["recipient_stored"] is False


def test_invented_claim_is_not_approved_even_when_lead_fit_is_high(runtime_factory):
    runtime, request=runtime_factory(invented=True,max_leads=1)
    run=runtime.create_run(request)
    result=runtime.execute(run["id"])
    lead=next(item for item in result["leads"] if item["lead_id"] == "syn-grounded")
    assert lead["status"] != LeadStatus.APPROVED.value
    assert not runtime.store.list_outbox(result["id"])
    assert any(event["step"] == "jev.assess_lead" and event["result"].get("judgments",{}).get("claim_support_0",{}).get("probability_yes",1) < .82
               for event in result["events"])


def test_autonomous_uncertain_lead_is_researched_then_skipped(runtime_factory):
    runtime, request=runtime_factory(autonomous=True)
    result=runtime.execute(runtime.create_run(request)["id"])
    lead=next(item for item in result["leads"] if item["lead_id"] == "syn-uncertain")
    assert lead["status"] == LeadStatus.SKIPPED.value
    events=[event for event in result["events"] if event["arguments"].get("lead_id")=="syn-uncertain"]
    inspections=[event for event in events if event["step"]=="tool.inspect_lead"]
    assert len(inspections) >= 2
    assert inspections[-1]["arguments"]["detail_level"] == "deep"
    assert not any(item["status"] == LeadStatus.AWAITING_REVIEW.value for item in result["leads"])


def test_jev_timeout_defers_without_fallback_or_draft(runtime_factory):
    runtime, request=runtime_factory(max_leads=1)
    class BrokenJev:
        def assess_lead(self,*args,**kwargs):
            raise TimeoutError("offline Jev timeout")
    runtime.jev=BrokenJev()
    result=runtime.execute(runtime.create_run(request)["id"])
    lead=result["leads"][0]
    assert lead["status"] == LeadStatus.DEFERRED.value
    assert lead["draft"] is None
    assert any(event["step"]=="policy.jev_unavailable" and event["error_status"]=="JevConfigurationError" for event in result["events"])
    assert not runtime.store.list_outbox(result["id"])


def test_tool_budget_stops_before_an_extra_action(runtime_factory):
    runtime, request=runtime_factory(max_leads=1)
    request=request.model_copy(update={"max_tool_calls":4})
    run=runtime.create_run(request)
    result=runtime.execute(run["id"])
    assert result["status"]=="budget_exhausted"
    assert result["counters"]["tool_calls"]==4
    assert any(event["step"]=="budget.stop" for event in result["events"])


def test_done_is_not_available_while_actionable_leads_remain(runtime_factory):
    runtime, request = runtime_factory(max_leads=1)
    run = runtime.create_run(request)
    runtime.store.update_run(run["id"], counters={"search_builds": 1, "tool_calls": 3})
    runtime.store.create_or_get_export(run["id"], "export-test")
    runtime.store.update_export(run["id"], status="complete", poll_count=1, leads=[])
    runtime.store.save_lead(
        run["id"], "lead-actionable",
        {"Email": "lead@example.com", "Email Status": "safe", "Matches Filters": "YES"},
        {}, LeadStatus.ELIGIBLE.value,
    )
    plan = Plan.model_validate(runtime.store.get_run(run["id"])["plan"])

    allowed = runtime._allowed_tools(run["id"], request, plan, runtime.store.get_run(run["id"])["counters"])

    assert allowed == [ToolName.INSPECT_LEAD]


def test_ambiguous_reply_suppresses_and_duplicate_event_is_ignored(runtime_factory):
    runtime, request=runtime_factory()
    run=runtime.create_run(request)
    runtime.store.save_lead(run["id"],"lead-x",{"Email":"prospect@example.com","Matches Filters":"YES","Email Status":"safe"},{},LeadStatus.APPROVED.value)
    payload={"external_event_id":"reply-event-0001","lead_id":"lead-x","event_type":"reply","text":"Maybe. Who is this again?"}
    first=runtime.ingest_outcome(run["id"],payload)
    second=runtime.ingest_outcome(run["id"],payload)
    assert first["reply_class"] == "unclear" and first["suppressed"]
    assert second["duplicate"]
    assert runtime.store.is_suppressed("prospect@example.com")
    assert runtime.store.get_lead(run["id"],"lead-x")["status"] == LeadStatus.SUPPRESSED.value
    # An out-of-order later delivery event does not clear a suppression.
    runtime.ingest_outcome(run["id"],{"external_event_id":"late-delivery-0001","lead_id":"lead-x","event_type":"delivered"})
    assert runtime.store.is_suppressed("prospect@example.com")


def test_opt_out_signal_is_deterministic_even_if_jev_would_be_positive(runtime_factory):
    runtime, request=runtime_factory()
    run=runtime.create_run(request)
    runtime.store.save_lead(run["id"],"lead-x",{"Email":"prospect@example.com"},{},LeadStatus.APPROVED.value)
    result=runtime.ingest_outcome(run["id"],{"external_event_id":"optout-event-01","lead_id":"lead-x",
                                               "event_type":"reply","text":"Please stop and unsubscribe."})
    assert result["reply_class"] == "opt_out"
    assert runtime.store.is_suppressed("prospect@example.com")


def test_edit_or_approval_without_grounded_draft_is_blocked(runtime_factory):
    runtime, request=runtime_factory()
    run=runtime.create_run(request)
    runtime.store.save_lead(run["id"],"blank-draft",{"Email":"x@example.com"},{},LeadStatus.AWAITING_REVIEW.value)
    with pytest.raises(RuntimeRequestError,match="no grounded draft"):
        runtime.review(run["id"],"blank-draft",ReviewRequest(decision="approve"))


def test_autonomous_candidate_requires_locked_evaluation_and_observed_outcomes(runtime_factory):
    runtime, request=runtime_factory(autonomous=True)
    run=runtime.create_run(request)
    # An initial empty plan is enough; offline doubles are test-only and never used in the app.
    for index in range(4):
        runtime.store.add_outcome(f"historical-{index}", run["id"], f"old-lead-{index}",
                                  "conversion", "2026-09-01T00:00:00Z", None, None, None)
    canary=runtime.improve(run["id"])
    assert canary["status"] == "awaiting_canary"
    assert canary["canary"]["observed_events"] == 0
    assert canary["evaluation"]["heldout_case_count"] >= 8
    assert runtime.store.active_policy()["version"] == "policy-v1"
    for index in range(4):
        lead_id = f"lead-{index}"
        fields = {
            "Current Job": "VP Finance", "Company Industry": "Financial Services",
            "Company Employee Exact Count": "450", "Company Location": "London, Europe",
            "Company Description": "European payments platform.", "Source Updated At": "2026-10-01",
        }
        judgments = {"answers": {
            "current_role_fit": {"type": "noul", "probability_yes": .97},
            "company_fit": {"type": "noul", "probability_yes": .97},
            "evidence_contradictions": {"type": "noul", "probability_yes": .01},
            "claim_support_0": {"type": "noul", "probability_yes": .97},
        }}
        draft = {"claims": [{"text": "European payments platform.",
                              "evidence": [{"field": "Company Description",
                                            "excerpt": "European payments platform."}]}]}
        runtime.store.save_lead(run["id"], lead_id, fields, {"source_timestamp": "2026-10-01"},
                                LeadStatus.APPROVED.value, draft=draft, judgments=judgments)
        runtime.ingest_outcome(run["id"], {"external_event_id": f"conversion-{index:04d}",
                                            "lead_id": lead_id, "event_type": "conversion"})
    promoted=runtime.store.get_candidate(canary["id"])
    assert promoted["status"] == "auto_promoted"
    assert runtime.store.active_policy()["version"] == promoted["evaluation"]["candidate_policy"]["version"]
    rolled_back=runtime.rollback_candidate(promoted["id"])
    assert rolled_back["status"] == "rolled_back"
    assert runtime.store.active_policy()["version"] == "policy-v1"
    report=runtime.report_markdown(run["id"])
    assert "Candidate evaluation report" in report
    assert "Held-out qualification cases" in report


def test_autonomous_candidate_is_rejected_when_shadow_canary_regresses(runtime_factory):
    runtime, request = runtime_factory(autonomous=True)
    run = runtime.create_run(request)
    candidate = runtime.improve(run["id"])

    judgments = {"answers": {
        "current_role_fit": {"type": "noul", "probability_yes": .97},
        "company_fit": {"type": "noul", "probability_yes": .97},
        "evidence_contradictions": {"type": "noul", "probability_yes": .01},
        "claim_support_0": {"type": "noul", "probability_yes": .97},
    }}
    draft = {"claims": [{"text": "European payments platform.",
                          "evidence": [{"field": "Company Description",
                                        "excerpt": "European payments platform."}]}]}
    base_fields = {
        "Current Job": "VP Finance", "Company Industry": "Financial Services",
        "Company Employee Exact Count": "450", "Company Location": "London, Europe",
        "Company Description": "European payments platform.",
    }
    # Baseline-only, non-adverse observations establish a better comparator first.
    for index in range(4):
        lead_id = f"stale-{index}"
        fields = {**base_fields, "Source Updated At": "2020-01-01"}
        runtime.store.save_lead(run["id"], lead_id, fields, {"source_timestamp": "2020-01-01"},
                                LeadStatus.APPROVED.value, draft=draft, judgments=judgments)
        runtime.ingest_outcome(run["id"], {"external_event_id": f"old-conversion-{index}",
                                            "lead_id": lead_id, "event_type": "conversion"})
    # Candidate-approved observations are all adverse, exceeding the 10% bound.
    for index in range(4):
        lead_id = f"current-{index}"
        fields = {**base_fields, "Source Updated At": "2026-10-01"}
        runtime.store.save_lead(run["id"], lead_id, fields, {"source_timestamp": "2026-10-01"},
                                LeadStatus.APPROVED.value, draft=draft, judgments=judgments)
        runtime.ingest_outcome(run["id"], {"external_event_id": f"bounce-{index}",
                                            "lead_id": lead_id, "event_type": "bounce"})

    rejected = runtime.store.get_candidate(candidate["id"])
    assert rejected["status"] == "rejected_by_canary"
    assert runtime.store.active_policy()["version"] == "policy-v1"
    assert rejected["canary"]["candidate_adverse_rate"] == 1
    assert rejected["canary"]["baseline_adverse_rate"] == .5
