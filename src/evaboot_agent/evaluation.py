from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def load_evaluation_cases(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float] | None:
    if total <= 0:
        return None
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = (z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)) / denom
    return [max(0.0, center - margin), min(1.0, center + margin)]


def binary_metrics(labels: list[bool], predictions: list[bool], probabilities: list[float]) -> dict[str, Any]:
    count = min(len(labels), len(predictions), len(probabilities))
    labels, predictions, probabilities = labels[:count], predictions[:count], probabilities[:count]
    approved = sum(predictions)
    true_approved = sum(1 for actual, predicted in zip(labels, predictions) if actual and predicted)
    false_approved = sum(1 for actual, predicted in zip(labels, predictions) if not actual and predicted)
    negatives = sum(not actual for actual in labels)
    brier = sum((float(score) - float(actual)) ** 2 for actual, score in zip(labels, probabilities)) / count if count else None
    bins: list[dict[str, Any]] = []
    for lower in (0.0, 0.2, 0.4, 0.6, 0.8):
        upper = lower + 0.2
        members = [(actual, score) for actual, score in zip(labels, probabilities)
                   if lower <= score < upper or (upper == 1.0 and score == 1.0)]
        if members:
            bins.append({"range": [lower, upper], "count": len(members),
                         "mean_predicted": round(sum(score for _, score in members) / len(members), 4),
                         "observed_rate": round(sum(actual for actual, _ in members) / len(members), 4)})
    return {
        "sample_count": count,
        "auto_approved_count": approved,
        "true_auto_approved": true_approved,
        "false_auto_approved": false_approved,
        "negative_case_denominator": negatives,
        "precision_among_auto_approved": true_approved / approved if approved else None,
        "precision_interval_95": wilson_interval(true_approved, approved),
        "false_approval_rate": false_approved / negatives if negatives else None,
        "false_approval_interval_95": wilson_interval(false_approved, negatives),
        "automatic_action_coverage": approved / count if count else None,
        "skip_rate": 1 - approved / count if count else None,
        "brier_score": brier,
        "reliability_bins": bins,
    }


def policy_metrics(labels: list[bool], predictions: list[bool]) -> dict[str, Any]:
    count = min(len(labels), len(predictions))
    labels, predictions = labels[:count], predictions[:count]
    approved = sum(predictions)
    true_approved = sum(1 for label, prediction in zip(labels, predictions) if label and prediction)
    false_approved = sum(1 for label, prediction in zip(labels, predictions) if not label and prediction)
    negatives = sum(not label for label in labels)
    return {
        "sample_count": count,
        "auto_approved_count": approved,
        "true_auto_approved": true_approved,
        "false_auto_approved": false_approved,
        "negative_case_denominator": negatives,
        "precision_among_auto_approved": true_approved / approved if approved else None,
        "precision_interval_95": wilson_interval(true_approved, approved),
        "false_approval_rate": false_approved / negatives if negatives else None,
        "false_approval_interval_95": wilson_interval(false_approved, negatives),
        "automatic_action_coverage": approved / count if count else None,
        "skip_rate": 1 - approved / count if count else None,
    }


def reply_metrics(labels: list[str], predictions: list[str]) -> dict[str, Any]:
    count = min(len(labels), len(predictions))
    labels, predictions = labels[:count], predictions[:count]
    classes = sorted(set(labels) | set(predictions))
    matrix = {actual: {predicted: 0 for predicted in classes} for actual in classes}
    for actual, predicted in zip(labels, predictions):
        matrix[actual][predicted] += 1
    errors = [{"actual": actual, "predicted": predicted}
              for actual, predicted in zip(labels, predictions) if actual != predicted]
    return {"sample_count": count, "classes": classes, "confusion_matrix": matrix,
            "error_count": len(errors), "errors": errors}


def calibrate_thresholds(results: list[dict[str, Any]], policy: dict[str, Any], *,
                         minimum_labeled_cases: int = 3,
                         minimum_precision: float = 0.85) -> dict[str, Any]:
    """Select thresholds from calibration labels only; never edits acceptance rules."""
    observations: dict[str, list[tuple[bool, float]]] = {
        "current_role_fit": [], "company_fit": [], "claim_support": [], "evidence_contradictions": []
    }
    for item in results:
        labels = item.get("labels", {})
        answers = item.get("judgment", {}).get("answers", {})
        role = labels.get("role_match")
        role_probability = answers.get("current_role_fit", {}).get("probability_yes")
        if role is not None and role_probability is not None:
            observations["current_role_fit"].append((bool(role), float(role_probability)))
        company = labels.get("company_match")
        company_probability = answers.get("company_fit", {}).get("probability_yes")
        if company is not None and company_probability is not None:
            observations["company_fit"].append((bool(company), float(company_probability)))
        claim = labels.get("claim_supported")
        claim_probability = answers.get("claim_support_0", {}).get("probability_yes")
        if claim is not None and claim_probability is not None:
            observations["claim_support"].append((bool(claim), float(claim_probability)))
        contradiction = labels.get("contradiction")
        contradiction_probability = answers.get("evidence_contradictions", {}).get("probability_yes")
        if contradiction is not None and contradiction_probability is not None:
            observations["evidence_contradictions"].append((bool(contradiction), float(contradiction_probability)))

    thresholds = dict(policy["question_thresholds"])
    selection: dict[str, Any] = {}
    for question in ("current_role_fit", "company_fit", "claim_support"):
        values = observations[question]
        selected = thresholds[question]
        reason = "insufficient_labeled_calibration_cases"
        if len(values) >= minimum_labeled_cases:
            candidate = _positive_threshold(values, minimum_precision)
            if candidate is not None:
                selected = candidate
                reason = "max_coverage_at_minimum_calibration_precision"
            else:
                reason = "no_threshold_met_calibration_precision"
        thresholds[question] = selected
        selection[question] = {"labeled_case_count": len(values), "selected_threshold": selected,
                               "minimum_precision": minimum_precision, "reason": reason}

    values = observations["evidence_contradictions"]
    selected = thresholds["contradiction_risk_max"]
    reason = "insufficient_labeled_calibration_cases"
    if len(values) >= minimum_labeled_cases:
        negatives = [score for label, score in values if not label]
        positives = [score for label, score in values if label]
        if negatives and positives and max(negatives) < min(positives):
            selected = (max(negatives) + min(positives)) / 2
            reason = "separating_calibration_boundary"
        else:
            reason = "contradiction_classes_not_separable_on_calibration"
    thresholds["contradiction_risk_max"] = selected
    selection["evidence_contradictions"] = {"labeled_case_count": len(values),
                                             "selected_threshold": selected, "reason": reason}
    return {"thresholds": thresholds, "selection": selection,
            "calibration_case_count": len(results),
            "minimum_labeled_cases": minimum_labeled_cases}


def calibrate_reply_threshold(case_results: list[dict[str, Any]], policy: dict[str, Any], *,
                              minimum_labeled_cases: int = 4) -> dict[str, Any]:
    """Choose the reply clear-label floor using calibration replies only."""
    current = float(policy["reply_policy"]["clear_label_probability"])
    if len(case_results) < minimum_labeled_cases:
        return {"selected_threshold": current, "labeled_case_count": len(case_results),
                "reason": "insufficient_labeled_calibration_cases", "error_count": None}
    candidates = sorted({current, *(float(item["max_probability"]) for item in case_results)})
    ranked = []
    for threshold in candidates:
        predictions = [_reply_prediction(item, threshold,
                                         float(policy["reply_policy"]["opt_out_possible_floor"]))
                       for item in case_results]
        errors = sum(prediction != item["actual"] for prediction, item in zip(predictions, case_results))
        actionable = sum(prediction in {"positive", "negative"} for prediction in predictions)
        ranked.append((errors, -actionable, -threshold, threshold))
    best = min(ranked)
    return {"selected_threshold": best[3], "labeled_case_count": len(case_results),
            "reason": "minimize_calibration_class_errors_then_maximize_clear_coverage",
            "error_count": best[0], "clear_label_coverage": -best[1] / len(case_results)}


def _reply_prediction(case: dict[str, Any], clear_threshold: float, opt_out_floor: float) -> str:
    if case.get("deterministic_opt_out"):
        return "opt_out"
    probabilities = case.get("probabilities", {})
    if not probabilities:
        return "unclear"
    if float(probabilities.get("opt_out", 0)) >= opt_out_floor:
        return "opt_out"
    choice = case.get("choice", "unclear")
    if choice not in {"positive", "negative"} or float(case.get("max_probability", 0)) < clear_threshold:
        return "unclear"
    return choice


def _positive_threshold(values: list[tuple[bool, float]], minimum_precision: float) -> float | None:
    candidates = sorted({score for _, score in values})
    valid: list[tuple[int, float]] = []
    for threshold in candidates:
        selected = [(label, score) for label, score in values if score >= threshold]
        if not selected:
            continue
        true_positive = sum(label for label, _ in selected)
        precision = true_positive / len(selected)
        if precision >= minimum_precision:
            valid.append((len(selected), threshold))
    if not valid:
        return None
    # Maximize covered calibration cases; tie-break toward the stricter threshold.
    return max(valid, key=lambda item: (item[0], item[1]))[1]


def promotion_gate(baseline: dict[str, Any], candidate: dict[str, Any], acceptance: dict[str, Any], *,
                   critical_violations: int, reply_class_errors: int = 0) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if candidate["sample_count"] < acceptance["minimum_heldout_cases"]:
        failures.append("insufficient_heldout_cases")
    precision = candidate.get("precision_among_auto_approved")
    if precision is None or precision < acceptance["minimum_auto_approved_precision"]:
        failures.append("heldout_precision_below_threshold")
    false_rate = candidate.get("false_approval_rate")
    if false_rate is None or false_rate > acceptance["maximum_false_approval_rate"]:
        failures.append("false_approval_rate_above_threshold")
    if critical_violations > acceptance["maximum_critical_violations"]:
        failures.append("critical_policy_violation")
    max_reply_errors = acceptance.get("maximum_heldout_reply_class_errors")
    if max_reply_errors is not None and reply_class_errors > max_reply_errors:
        failures.append("heldout_reply_errors_above_threshold")
    base_coverage = float(baseline.get("automatic_action_coverage") or 0)
    candidate_coverage = float(candidate.get("automatic_action_coverage") or 0)
    if candidate_coverage - base_coverage < acceptance["minimum_coverage_non_regression"]:
        failures.append("coverage_regression_exceeds_limit")
    return not failures, failures
