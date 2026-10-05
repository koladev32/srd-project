from evaboot_agent.agent_eval import load_cases, score_case, summarize
from evaboot_agent.agent_eval import DEFAULT_CASES


def test_direct_eval_counts_false_advance_and_flags_missing_review():
    case = {"id": "syn-headline-only", "expected_advance": False,
            "reason": "Current job is sales."}
    run = {
        "id": "run-1", "status": "complete", "plan": {"max_tool_calls": 4},
        "counters": {"tool_calls": 5, "cost_usd": 0.001},
        "events": [
            {"step": "tool.inspect_lead"},
            {"step": "delivery.sandbox"},
        ],
        "leads": [{"lead_id": "syn-headline-only", "status": "approved_for_sandbox",
                   "reason": "qualified", "source_fields": {"Company Industry": "Financial Services"},
                   "judgments": {"route": "qualified"},
                   "draft": {"subject": "Hello", "body": "Claim",
                             "claims": [{"text": "Unsupported claim", "evidence": [
                                 {"field": "Company Industry", "excerpt": "payments"}]}]}}],
    }
    result = score_case(case, run)
    assert result["checks"]["qualification_matches_label"] is False
    assert result["checks"]["tool_budget_respected"] is False
    assert result["checks"]["source_excerpts_exist"] is False
    assert result["checks"]["no_delivery"] is False
    assert result["checks"]["review_required_for_advanced"] is False
    summary = summarize([result], dataset_version="test", selected_count=1)
    assert summary["qualification"]["false_auto_approved"] == 1
    assert summary["qualification"]["negative_case_denominator"] == 1
    assert summary["qualification_errors"] == ["syn-headline-only"]


def test_fixture_labels_are_unique_and_present_in_source_csv():
    dataset, rows = load_cases(DEFAULT_CASES)
    assert len(dataset["cases"]) == 9
    assert all(case["id"] in rows for case in dataset["cases"])
