import json

from evaboot_agent.config import RESOURCE_ROOT
from evaboot_agent.evaluation import (binary_metrics, calibrate_reply_threshold, calibrate_thresholds, policy_metrics,
                                      promotion_gate, reply_metrics)
from evaboot_agent.models import LeadStatus
from evaboot_agent.normalizer import Lead, normalize_row
from evaboot_agent.policy import (classify_reply, deterministic_gate, required_evidence_missing,
                                  route_judgment, validate_claim_references)


def test_normalize_and_public_summary_omit_contact_details():
    lead = normalize_row({"Fixture ID":"synthetic-1", "First Name":"Ava", "Email":"ava@example.test",
                          "LinkedIn URL":"https://example.test/profile", "Matches Filters":"\"YES\"",
                          "Current Job":"VP Finance", "Company Industry":"Financial Services"}, 1, "fixture")
    assert lead.lead_id == "synthetic-1"
    assert lead.fields["Matches Filters"] == "YES"
    summary = lead.public_summary()
    assert summary["matches_filters"] == "YES"
    assert "ava@example.test" not in str(summary)
    assert "https://" not in str(summary)


def test_hard_gates_reject_filter_mismatch_missing_unsafe_duplicate_and_suppressed():
    base = {"Matches Filters":"YES", "Email":"a@example.com", "Email Status":"safe"}
    def gate(fields, **kwargs):
        return deterministic_gate(Lead("x", fields), seen_emails=kwargs.get("seen", set()),
                                 seen_profiles=set(), user_suppressions=kwargs.get("suppressions", []),
                                 is_suppressed=kwargs.get("suppressed", False))
    assert gate({**base,"Matches Filters":"NO"}) == ("rejected", "filter_mismatch")
    assert gate({"Email":"a@example.com", "Email Status":"safe"}) == ("rejected", "filter_match_unknown")
    assert gate({**base,"Email Status":"riskier"}) == ("rejected", "email_not_safe")
    assert gate(base, seen={"a@example.com"}) == ("rejected", "duplicate_lead")
    assert gate(base, suppressions=["example.com"]) == ("suppressed", "suppression_list_match")
    assert gate(base, suppressed=True) == ("suppressed", "suppression_list_match")


def test_absent_evidence_is_unknown_and_citations_must_exist_verbatim():
    lead = Lead("missing", {"Current Job":"", "Company Industry":"Financial Services"})
    assert "Current Job" not in required_evidence_missing(lead)
    assert "Current Job" in required_evidence_missing(lead, require_current_job_match=True)
    assert validate_claim_references([{"text":"Finance company", "evidence":[{"field":"Company Industry","excerpt":"financial services"}]}], lead.fields) == []
    assert validate_claim_references([{"text":"Made up", "evidence":[{"field":"Company Industry","excerpt":"10,000 employees"}]}], lead.fields)


def test_policy_threshold_boundaries_and_missing_judgment():
    policy={"question_thresholds":{"current_role_fit":.8,"company_fit":.8,"claim_support":.82,"contradiction_risk_max":.15}}
    judgment={"answers":{"current_role_fit":{"probability_yes":.8},"company_fit":{"probability_yes":.8},
                         "evidence_contradictions":{"probability_yes":.15},"claim_support_0":{"probability_yes":.82}}}
    assert route_judgment(judgment,policy,missing_fields=[],stale=False,claim_count=1)[0] == "qualified"
    judgment["answers"]["claim_support_0"]["probability_yes"] = .10
    assert route_judgment(judgment,policy,missing_fields=[],stale=False,claim_count=1)[0] == "revise"
    assert route_judgment({"answers":{}},policy,missing_fields=[],stale=False)[0] == "deferred"
    assert route_judgment(judgment,policy,missing_fields=["Current Job"],stale=False)[0] == "insufficient"


def test_reply_opt_out_and_unclear_always_pause_contact():
    policy={"reply_policy":{"clear_label_probability":.8,"opt_out_possible_floor":.05}}
    positive={"answers":{"intent":{"choice":"positive","probabilities":{"positive":.9,"negative":.04,"opt_out":.01,"unclear":.05}}}}
    assert classify_reply(positive,policy,"Please unsubscribe me") == ("opt_out","deterministic_opt_out_signal")
    assert classify_reply({"answers":{"intent":{"choice":"unclear","probabilities":{"positive":.2,"negative":.2,"opt_out":.04,"unclear":.56}}}},policy,"Maybe") [0] == "unclear"
    possible_optout={"answers":{"intent":{"choice":"positive","probabilities":{"positive":.8,"negative":.1,"opt_out":.06,"unclear":.04}}}}
    assert classify_reply(possible_optout,policy,"Thanks") [0] == "opt_out"


def test_evaluation_reports_denominator_intervals_reliability_and_regression_gate():
    metrics=binary_metrics([True,False,True],[True,True,False],[.9,.8,.2])
    assert metrics["negative_case_denominator"] == 1
    assert metrics["false_approval_rate"] == 1
    assert metrics["precision_interval_95"] is not None
    assert metrics["brier_score"] is not None
    assert metrics["reliability_bins"]
    baseline=policy_metrics([True,False],[True,False])
    candidate=policy_metrics([True,False],[True,True])
    passed, failures=promotion_gate(baseline,candidate,{"minimum_heldout_cases":8,
        "minimum_auto_approved_precision":.85,"maximum_false_approval_rate":.1,
        "minimum_coverage_non_regression":-.2,"maximum_critical_violations":0,
        "maximum_heldout_reply_class_errors":0},critical_violations=1,reply_class_errors=1)
    assert not passed
    assert "critical_policy_violation" in failures
    assert "heldout_reply_errors_above_threshold" in failures
    assert reply_metrics(["positive","opt_out"],["positive","unclear"])["error_count"] == 1


def test_calibration_and_locked_test_groups_do_not_overlap():
    path=RESOURCE_ROOT/"data"/"evaluation_cases.json"
    cases=json.loads(path.read_text())["cases"]
    calibration={item["group_id"] for item in cases if item["split"]=="calibration"}
    heldout={item["group_id"] for item in cases if item["split"]=="heldout"}
    assert calibration.isdisjoint(heldout)
    assert sum(1 for item in cases if item["split"]=="heldout" and "reply" not in item)>=8
    assert sum(1 for item in cases if item["split"]=="calibration" and "reply" in item)>=4


def test_calibration_selects_thresholds_without_touching_acceptance_rules():
    results=[
        {"labels":{"role_match":True,"company_match":True,"claim_supported":True,"contradiction":False},
         "judgment":{"answers":{"current_role_fit":{"probability_yes":.91},"company_fit":{"probability_yes":.88},
                                  "claim_support_0":{"probability_yes":.90},"evidence_contradictions":{"probability_yes":.03}}}},
        {"labels":{"role_match":False,"company_match":True,"claim_supported":False,"contradiction":True},
         "judgment":{"answers":{"current_role_fit":{"probability_yes":.15},"company_fit":{"probability_yes":.86},
                                  "claim_support_0":{"probability_yes":.12},"evidence_contradictions":{"probability_yes":.94}}}},
        {"labels":{"role_match":True,"company_match":False,"claim_supported":True,"contradiction":False},
         "judgment":{"answers":{"current_role_fit":{"probability_yes":.89},"company_fit":{"probability_yes":.18},
                                  "claim_support_0":{"probability_yes":.88},"evidence_contradictions":{"probability_yes":.06}}}},
    ]
    policy={"question_thresholds":{"current_role_fit":.8,"company_fit":.8,"claim_support":.82,"contradiction_risk_max":.15}}
    calibrated=calibrate_thresholds(results,policy,minimum_precision=.85)
    assert calibrated["thresholds"]["current_role_fit"]==.89
    assert calibrated["thresholds"]["company_fit"]==.86
    assert calibrated["thresholds"]["claim_support"]==.88
    assert calibrated["thresholds"]["contradiction_risk_max"]==.5
    assert policy["question_thresholds"]["current_role_fit"]==.8
    assert calibrated["selection"]["evidence_contradictions"]["labeled_case_count"]==3


def test_reply_threshold_calibrates_clear_label_while_preserving_optout_floor():
    cases=[
        {"actual":"positive","choice":"positive","probabilities":{"positive":.9,"negative":.02,"opt_out":.01,"unclear":.07},"max_probability":.9,"deterministic_opt_out":False},
        {"actual":"negative","choice":"negative","probabilities":{"positive":.02,"negative":.91,"opt_out":.01,"unclear":.06},"max_probability":.91,"deterministic_opt_out":False},
        {"actual":"opt_out","choice":"opt_out","probabilities":{"positive":.01,"negative":.01,"opt_out":.96,"unclear":.02},"max_probability":.96,"deterministic_opt_out":True},
        {"actual":"unclear","choice":"unclear","probabilities":{"positive":.1,"negative":.1,"opt_out":.01,"unclear":.79},"max_probability":.79,"deterministic_opt_out":False},
    ]
    policy={"reply_policy":{"clear_label_probability":.8,"opt_out_possible_floor":.05}}
    calibrated=calibrate_reply_threshold(cases,policy)
    assert calibrated["selected_threshold"]==.9
    assert calibrated["error_count"]==0
    assert policy["reply_policy"]["clear_label_probability"]==.8
    assert policy["reply_policy"]["opt_out_possible_floor"]==.05
