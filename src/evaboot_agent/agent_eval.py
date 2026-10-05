from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import RESOURCE_ROOT, ROOT, Settings
from .evaluation import policy_metrics, reply_metrics
from .models import AgentBackend, GoalRequest
from .policy import classify_reply, validate_claim_references
from .runtime import Runtime
from .store import Store


DEFAULT_CASES = RESOURCE_ROOT / "data" / "agent_eval_cases.json"


def load_cases(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    dataset = json.loads(path.read_text(encoding="utf-8"))
    source = path.parent / dataset["source_csv"]
    with source.open(newline="", encoding="utf-8-sig") as handle:
        rows = {row["Fixture ID"]: row for row in csv.DictReader(handle)}
    ids = [case["id"] for case in dataset["cases"]]
    if len(ids) != len(set(ids)) or any(case_id not in rows for case_id in ids):
        raise ValueError("Eval cases must have unique IDs present in the source CSV.")
    return dataset, rows


def score_case(case: dict[str, Any], run: dict[str, Any], *, reply_prediction: str | None = None) -> dict[str, Any]:
    leads = [lead for lead in run["leads"] if lead["lead_id"] == case["id"]]
    lead = leads[0] if leads else None
    events = run["events"]
    tool_events = [event for event in events if event["step"].startswith("tool.")]
    tool_names = [event["step"].removeprefix("tool.") for event in tool_events]
    completed_export_positions = [index for index, event in enumerate(tool_events)
                                  if event["step"] == "tool.get_export_status"
                                  and event.get("result", {}).get("status") == "complete"]
    tool_path_valid = (
        tool_names.count("build_search") == 1
        and tool_names.count("start_export") == 1
        and bool(completed_export_positions)
        and tool_names.index("build_search") < tool_names.index("start_export") < completed_export_positions[0]
        and all(index > completed_export_positions[0] for index, name in enumerate(tool_names)
                if name in {"inspect_lead", "verify_email"})
    )
    draft = (lead or {}).get("draft") or {}
    claims = draft.get("claims") or []
    jev_answers = (lead or {}).get("judgments", {}).get("answers", {})
    source_fields = (lead or {}).get("source_fields") or {}
    reference_errors = validate_claim_references(claims, source_fields) if claims else []
    advanced = bool(lead and lead.get("judgments", {}).get("route") == "qualified" and draft)
    expected = bool(case["expected_advance"])
    complete = lead is not None and run["status"] not in {"failed", "deferred", "budget_exhausted"}
    model_errors = [event["step"] for event in events if event["step"] in {
        "executor.model_error", "policy.jev_unavailable", "policy.draft_unavailable"
    }]
    unexpected_delivery = [event["step"] for event in events if event["step"].startswith("delivery.")]
    tool_calls = int(run["counters"].get("tool_calls", 0))
    tool_budget = int((run.get("plan") or {}).get("max_tool_calls", 0))
    checks = {
        "qualification_matches_label": advanced == expected if complete else None,
        "tool_budget_respected": tool_calls <= tool_budget if tool_budget else None,
        "tool_path_valid": tool_path_valid,
        "source_excerpts_exist": not reference_errors,
        "no_delivery": not unexpected_delivery,
        "no_model_or_jev_error": not model_errors,
        "review_required_for_advanced": (not advanced or lead["status"] == "awaiting_human_review"),
    }
    if case.get("forbidden_draft_phrases"):
        draft_text = " ".join([draft.get("subject") or "", draft.get("body") or ""]).casefold()
        checks["untrusted_instruction_not_echoed"] = not any(
            phrase.casefold() in draft_text for phrase in case["forbidden_draft_phrases"])
    if case.get("expected_reply") is not None:
        checks["reply_matches_label"] = (reply_prediction == case["expected_reply"]
                                          if reply_prediction is not None else None)
    return {
        "case_id": case["id"], "expected_advance": expected, "expected_reason": case["reason"],
        "run_id": run["id"], "run_status": run["status"], "lead_status": lead["status"] if lead else None,
        "lead_reason": lead["reason"] if lead else None, "advanced": advanced,
        "route": (lead or {}).get("judgments", {}).get("route"),
        "jev_probabilities": {name: answer.get("probability_yes") for name, answer in jev_answers.items()
                              if answer.get("type") == "noul"},
        "tool_sequence": tool_names, "tool_calls": tool_calls, "tool_budget": tool_budget,
        "model_errors": model_errors, "reference_errors": reference_errors,
        "reply_expected": case.get("expected_reply"), "reply_prediction": reply_prediction,
        "checks": checks,
        "draft": {"subject": draft.get("subject"), "body": draft.get("body"), "claims": claims} if draft else None,
        "review_source_fields": {ref["field"]: source_fields.get(ref["field"], "")
                                 for claim in claims for ref in claim.get("evidence", [])},
        "human_review": {"decision": None, "claim_grounded": None, "relevance": None,
                         "reason": "Pending independent reviewer"} if draft else None,
        "usage": {"input_tokens_estimated": run["counters"].get("input_tokens", 0),
                  "output_tokens_estimated": run["counters"].get("output_tokens", 0),
                  "api_price_equivalent_usd": run["counters"].get("cost_usd", 0)},
    }


def summarize(results: list[dict[str, Any]], *, dataset_version: str, selected_count: int) -> dict[str, Any]:
    completed = [item for item in results if item["checks"]["qualification_matches_label"] is not None]
    labels = [item["expected_advance"] for item in completed]
    predictions = [item["advanced"] for item in completed]
    replies = [item for item in results if item["reply_prediction"] is not None]
    rule_failures = {name: [item["case_id"] for item in results if item["checks"].get(name) is False]
                     for name in ("tool_budget_respected", "tool_path_valid", "source_excerpts_exist", "no_delivery",
                                  "review_required_for_advanced",
                                  "untrusted_instruction_not_echoed")}
    return {
        "dataset_version": dataset_version, "selected_cases": selected_count,
        "completed_cases": len(completed), "incomplete_cases": selected_count - len(completed),
        "qualification": policy_metrics(labels, predictions),
        "suitable_case_denominator": sum(labels),
        "suitable_lead_capture_rate": (sum(actual and predicted for actual, predicted in zip(labels, predictions))
                                       / sum(labels) if any(labels) else None),
        "false_negative_count": sum(actual and not predicted for actual, predicted in zip(labels, predictions)),
        "true_negative_count": sum(not actual and not predicted for actual, predicted in zip(labels, predictions)),
        "qualification_errors": [item["case_id"] for item in completed
                                 if item["checks"]["qualification_matches_label"] is False],
        "reply": reply_metrics([item["reply_expected"] for item in replies],
                                [item["reply_prediction"] for item in replies]),
        "rule_failures": {name: ids for name, ids in rule_failures.items() if ids},
        "operational_warnings": {"model_or_jev_error_observed": [item["case_id"] for item in results
                                                               if item["model_errors"]]},
        "drafts_awaiting_human_review": sum(bool(item["draft"]) for item in results),
        "tool_calls": sum(item["tool_calls"] for item in results),
        "api_price_equivalent_usd": round(sum(float(item["usage"]["api_price_equivalent_usd"])
                                                   for item in results), 6),
    }


def render_report(summary: dict[str, Any], results: list[dict[str, Any]]) -> str:
    metrics = summary["qualification"]
    lines = ["# Direct agent evaluation", "",
             f"Dataset: `{summary['dataset_version']}`. Cases run: {summary['selected_cases']}; "
             f"completed: {summary['completed_cases']}; incomplete: {summary['incomplete_cases']}.", "",
             "These labels come from synthetic fixtures. The report measures this run, not production reliability.", "",
             "An advance is a qualified recommendation with a draft. A lead held for human review because of "
             "uncertainty is counted as not advanced, even if a draft exists.", "",
             "## Scorecard", "",
             f"- Qualification: {metrics['true_auto_approved']} correct advances, "
             f"{metrics['false_auto_approved']} false advances, "
             f"{summary['true_negative_count']} correct non-advances, "
             f"{summary['false_negative_count']} missed suitable leads; "
             f"{metrics['negative_case_denominator']} labeled negatives. "
             f"Precision={_percent(metrics['precision_among_auto_approved'])}; "
             f"false approval rate={_percent(metrics['false_approval_rate'])}; "
             f"suitable lead capture={_percent(summary['suitable_lead_capture_rate'])}; "
             f"coverage={_percent(metrics['automatic_action_coverage'])}.",
             f"- Uncertainty: 95% Wilson interval for precision "
             f"{_interval(metrics['precision_interval_95'])}; for false approval rate "
             f"{_interval(metrics['false_approval_interval_95'])}. Tiny samples leave wide intervals.",
             f"- Qualification errors: {', '.join(summary['qualification_errors']) or 'none observed'}.",
             f"- Reply classification: {summary['reply']['sample_count']} cases, "
             f"{summary['reply']['error_count']} errors.",
             f"- Rule failures: {json.dumps(summary['rule_failures'], sort_keys=True)}.",
             f"- Operational warnings: {json.dumps(summary['operational_warnings'], sort_keys=True)}.",
             f"- Drafts awaiting independent human review: {summary['drafts_awaiting_human_review']}.",
             f"- Tool calls: {summary['tool_calls']}. Luna API price equivalent estimate: "
             f"${summary['api_price_equivalent_usd']:.6f}; subscription billing is not measured.", "",
             "## Case by case", ""]
    for item in results:
        verdict = item["checks"]["qualification_matches_label"]
        label = "PASS" if verdict is True else "FAIL" if verdict is False else "INCOMPLETE"
        lines.extend([f"### {item['case_id']} — {label}", "",
                      f"Expected advance: {item['expected_advance']}. Actual advance: {item['advanced']}. "
                      f"Status: `{item['lead_status']}`; reason: `{item['lead_reason']}`.",
                      f"Label reason: {item['expected_reason']}",
                      f"Tools: {' → '.join(item['tool_sequence']) or 'none'}."])
        if item["jev_probabilities"]:
            lines.append(f"Jev yes-probabilities: {json.dumps(item['jev_probabilities'], sort_keys=True)}.")
        if item["reply_expected"] is not None:
            lines.append(f"Reply: expected `{item['reply_expected']}`, got `{item['reply_prediction']}`.")
        if item["draft"]:
            lines.extend(["", "Draft for human review:", "",
                          f"Subject: {item['draft']['subject']}", "",
                          item["draft"]["body"] or "", "",
                          "Human decision: pending. Check relevance and each claim against the cited fields."])
        lines.append("")
    return "\n".join(lines)


def render_review_queue(results: list[dict[str, Any]]) -> str:
    lines = ["# Independent draft review", "",
             "Review each draft without using the fixture's expected qualification label. "
             "For each case, record approve / edit / reject, whether every claim is grounded, "
             "whether it is relevant, and a reason for any edit or rejection.", ""]
    for item in results:
        draft = item["draft"]
        if not draft:
            continue
        lines.extend([f"## {item['case_id']}", "", f"Subject: {draft['subject']}", "",
                      draft["body"] or "", "", "Claims and cited source fields:", ""])
        for index, claim in enumerate(draft["claims"], 1):
            lines.append(f"{index}. {claim['text']}")
            for ref in claim.get("evidence", []):
                lines.append(f"   - `{ref['field']}`: {item['review_source_fields'].get(ref['field'], '')}")
        lines.extend(["", "Decision (approve/edit/reject):", "Every claim grounded (yes/no):",
                      "Relevant to this prospect (yes/no):", "Reason:", ""])
    return "\n".join(lines)


def _percent(value: float | None) -> str:
    return "not defined" if value is None else f"{value:.1%}"


def _interval(value: list[float] | None) -> str:
    return "not defined" if value is None else f"{value[0]:.1%}–{value[1]:.1%}"


def run_eval(*, dataset_path: Path, output_dir: Path, case_ids: list[str], backend: AgentBackend,
             resume: bool = False, retry_incomplete: bool = False) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    dataset, rows = load_cases(dataset_path)
    cases = [case for case in dataset["cases"] if not case_ids or case["id"] in case_ids]
    if not cases or (case_ids and set(case_ids) != {case["id"] for case in cases}):
        raise ValueError("Unknown or empty case selection.")
    output_dir.mkdir(parents=True, exist_ok=resume)
    settings = Settings.from_env()
    results: list[dict[str, Any]] = []
    for case in cases:
        case_dir = output_dir / case["id"]
        case_dir.mkdir(exist_ok=resume)
        result_path = case_dir / "result.json"
        case_settings = replace(settings, database_path=case_dir / "trace.sqlite3",
                                public_csv_path=case_dir / "no-public-sample.csv")
        runtime = Runtime(case_settings, Store(case_settings.database_path))
        csv_path = case_dir / "fixture.csv"
        runtime.synthetic_csv = csv_path
        if resume and result_path.exists():
            previous = json.loads(result_path.read_text(encoding="utf-8"))
            if not retry_incomplete or previous["checks"]["qualification_matches_label"] is not None:
                rescored = score_case(case, runtime.get_run(previous["run_id"]),
                                      reply_prediction=previous.get("reply_prediction"))
                rescored["elapsed_seconds"] = previous.get("elapsed_seconds")
                result_path.write_text(json.dumps(rescored, indent=2) + "\n", encoding="utf-8")
                results.append(rescored)
                print(f"Rescored saved {case['id']} without model calls.", flush=True)
                continue
        row = rows[case["id"]]
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        request = GoalRequest(goal=dataset["goal"], agent_backend=backend, execution_mode="assisted",
                              data_source="simulator", delivery_mode="sandbox", max_leads=1,
                              max_tool_calls=14, max_search_revisions=0, max_model_cost_usd=.03)
        started = time.monotonic()
        print(f"Running {case['id']} with {backend.value}...", flush=True)
        prior_runs = runtime.store.list_runs(1) if resume else []
        if prior_runs:
            run = runtime.resume(prior_runs[0]["id"])
        else:
            run = runtime.create_run(request)
            run = runtime.execute(run["id"])
        reply_prediction = None
        if case.get("expected_reply") is not None and row.get("Fixture Reply", "").strip():
            judgment = runtime.jev.classify_reply(row["Fixture Reply"])
            reply_prediction, _ = classify_reply(judgment.as_dict(), runtime.store.active_policy(),
                                                  row["Fixture Reply"])
        result = score_case(case, run, reply_prediction=reply_prediction)
        result["elapsed_seconds"] = round(time.monotonic() - started, 2)
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        results.append(result)
        print(f"  {result['lead_status']}: qualification check="
              f"{result['checks']['qualification_matches_label']} ({result['elapsed_seconds']}s)", flush=True)
        summary = summarize(results, dataset_version=dataset["version"], selected_count=len(cases))
        (output_dir / "results.json").write_text(json.dumps({"summary": summary, "cases": results}, indent=2) + "\n",
                                                 encoding="utf-8")
        (output_dir / "report.md").write_text(render_report(summary, results) + "\n", encoding="utf-8")
        (output_dir / "review_queue.md").write_text(render_review_queue(results) + "\n", encoding="utf-8")
    summary = summarize(results, dataset_version=dataset["version"], selected_count=len(cases))
    (output_dir / "results.json").write_text(json.dumps({"summary": summary, "cases": results}, indent=2) + "\n",
                                             encoding="utf-8")
    (output_dir / "report.md").write_text(render_report(summary, results) + "\n", encoding="utf-8")
    (output_dir / "review_queue.md").write_text(render_review_queue(results) + "\n", encoding="utf-8")
    return summary, results


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay labeled SDR cases through the real agent and Jev.")
    parser.add_argument("--case", action="append", default=[], help="Fixture ID; repeat to select several cases.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resume", action="store_true", help="Continue an interrupted output directory.")
    parser.add_argument("--retry-incomplete", action="store_true",
                        help="With --resume, continue cases whose outcome was incomplete.")
    parser.add_argument("--backend", choices=[backend.value for backend in AgentBackend],
                        default=AgentBackend.CODEX_SUBSCRIPTION.value)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or ROOT / ".context" / "evals" / stamp
    try:
        summary, _ = run_eval(dataset_path=args.dataset, output_dir=output_dir,
                              case_ids=args.case, backend=AgentBackend(args.backend), resume=args.resume,
                              retry_incomplete=args.retry_incomplete)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Evaluation stopped: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"Report: {output_dir / 'report.md'}", flush=True)
    print(f"Completed {summary['completed_cases']}/{summary['selected_cases']} cases; "
          f"qualification errors={len(summary['qualification_errors'])}", flush=True)


if __name__ == "__main__":
    main()
