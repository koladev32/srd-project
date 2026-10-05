from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from .agents import (AgentConfigurationError, AgentOutputError, CodexSubscriptionAgent,
                     CostLimitReached, OpenAIAgent)
from .config import (RESOURCE_ROOT, Settings, has_codex_chatgpt_login, has_evaboot_key,
                     has_openai_key, has_typesafe_key)
from .evaluation import (binary_metrics, calibrate_reply_threshold, calibrate_thresholds, load_evaluation_cases,
                         policy_metrics, promotion_gate, reply_metrics)
from .evaboot_mcp import EvabootMCPAdapter, EvabootMCPError
from .judgments import JevConfigurationError, JevJudge, JevResult
from .models import (AgentBackend, DataSource, DeliveryMode, EmailDraft, ExecutionMode, GoalRequest,
                     LeadStatus, Plan, ReviewRequest, RunStatus, ToolAction, ToolName,
                     ToolResult)
from .normalizer import Lead
from .policy import (classify_reply, deterministic_gate, evidence_is_stale,
                     required_evidence_missing, route_judgment,
                     validate_claim_references)
from .store import Store, now_iso
from .tools import ToolRegistry


class RuntimeRequestError(RuntimeError):
    pass


class Runtime:
    """Persistent plan/act/observe/revise coordinator."""

    def __init__(self, settings: Settings | None = None, store: Store | None = None,
                 agent: OpenAIAgent | None = None, jev: JevJudge | None = None):
        self.settings = settings or Settings.from_env()
        self.store = store or Store(self.settings.database_path)
        self._injected_agent = agent
        self.agent = agent or OpenAIAgent(self.settings, self.store)
        self._subscription_agent: CodexSubscriptionAgent | None = None
        self.jev = jev or JevJudge(self.settings)
        self.synthetic_csv = RESOURCE_ROOT / "data" / "synthetic_fixtures.csv"
        self.acceptance_file = RESOURCE_ROOT / "config" / "evaluation_acceptance.json"

    def health(self) -> dict[str, Any]:
        try:
            sample_count = len(self._load_public_and_synthetic(25))
        except Exception:
            sample_count = 0
        codex_subscription_available = has_codex_chatgpt_login()
        api_available = has_openai_key()
        return {
            "openai_configured": has_openai_key(),
            "codex_subscription_available": codex_subscription_available,
            "typesafe_configured": has_typesafe_key(),
            "evaboot_configured": has_evaboot_key(),
            "interactive_ready": (api_available or codex_subscription_available) and has_typesafe_key(),
            "openai_model": self.settings.openai_model,
            "typesafe_model": self.settings.typesafe_model,
            "csv_rows_available": sample_count,
            "live_delivery": False,
        }

    def _agent_for_request(self, request: GoalRequest) -> OpenAIAgent:
        if self._injected_agent is not None:
            return self._injected_agent
        if request.agent_backend == AgentBackend.CODEX_SUBSCRIPTION:
            if self._subscription_agent is None:
                self._subscription_agent = CodexSubscriptionAgent(self.settings, self.store)
            return self._subscription_agent
        return self.agent

    @staticmethod
    def _ensure_agent_backend(request: GoalRequest) -> None:
        if request.agent_backend == AgentBackend.CODEX_SUBSCRIPTION:
            if not has_codex_chatgpt_login():
                raise RuntimeRequestError("Codex is not signed in with ChatGPT. Run `codex login`, then retry.")
        elif not has_openai_key():
            raise RuntimeRequestError("OPENAI_KEY or OPENAI_API_KEY is missing for OpenAI API mode.")

    def _load_public_and_synthetic(self, max_leads: int):
        from .normalizer import load_simulator_leads
        return load_simulator_leads(self.settings.public_csv_path, self.synthetic_csv, max_leads)

    def create_run(self, request: GoalRequest) -> dict[str, Any]:
        self._ensure_agent_backend(request)
        if not has_typesafe_key():
            raise RuntimeRequestError("TYPESAFE_API_KEY is required. Jev judgments are never stubbed in interactive modes.")
        if request.delivery_mode == DeliveryMode.LIVE:
            raise RuntimeRequestError("No live sender is configured. This runtime records sandbox deliveries only.")
        if request.data_source == DataSource.EVABOOT:
            if not request.live_evaboot_authorized:
                raise RuntimeRequestError("Check the explicit Evaboot export authorization to use live data.")
            if not has_evaboot_key():
                raise RuntimeRequestError("EVABOOT_API_KEY is required for live Evaboot MCP. Simulator mode remains available.")
        run_id = str(uuid.uuid4())
        policy = self.store.active_policy()
        self.store.create_run(run_id, request.model_dump(mode="json"), policy["version"])
        self.store.add_event(
            run_id, step="run.created", summary="Owner goal and execution bounds recorded.",
            arguments={"agent_backend": request.agent_backend.value,
                       "execution_mode": request.execution_mode.value, "data_source": request.data_source.value,
                       "delivery_mode": request.delivery_mode.value, "max_leads": request.max_leads,
                       "max_tool_calls": request.max_tool_calls, "max_search_revisions": request.max_search_revisions},
            result={"live_evaboot_authorized": request.live_evaboot_authorized,
                    "sandbox_only": True, "policy_version": policy["version"]},
        )
        try:
            feedback = self.store.list_feedback(12)
            result = self._agent_for_request(request).create_plan(request, run_id=run_id, prior_feedback=feedback)
            plan: Plan = result.value
            self.store.update_run(run_id, status=RunStatus.PLANNED.value, plan=plan.model_dump())
            self.store.add_event(
                run_id, step="planner.plan_saved", summary="Bounded plan is ready for owner review before execution.",
                arguments={"goal_character_count": len(request.goal)},
                result={"target_criteria_count": len(plan.target_criteria), "evidence_requirements_count": len(plan.evidence_requirements),
                        "max_leads": plan.max_leads, "max_tool_calls": plan.max_tool_calls,
                        "max_search_revisions": plan.max_search_revisions, "approval_required": plan.approval_required},
            )
        except Exception as exc:
            self.store.update_run(run_id, status=RunStatus.FAILED.value)
            if isinstance(exc, (AgentConfigurationError, AgentOutputError, CostLimitReached)):
                raise RuntimeRequestError(str(exc)) from exc
            raise
        return self.get_run(run_id)

    def get_run(self, run_id: str) -> dict[str, Any]:
        run = self.store.get_run(run_id)
        if not run:
            raise RuntimeRequestError("Run not found.")
        leads = self.store.list_leads(run_id)
        run["leads"] = [self._public_lead(lead) for lead in leads]
        run["events"] = self.store.list_events(run_id)
        run["pending_reviews"] = [lead["lead_id"] for lead in leads if lead["status"] == LeadStatus.AWAITING_REVIEW.value]
        export = self.store.get_export(run_id)
        run["exports"] = ({key: value for key, value in export.items() if key != "leads"} if export else None)
        return run

    def execute(self, run_id: str) -> dict[str, Any]:
        run = self.store.get_run(run_id)
        if not run:
            raise RuntimeRequestError("Run not found.")
        request = GoalRequest.model_validate(run["goal"])
        plan = Plan.model_validate(run["plan"])
        if request.delivery_mode != DeliveryMode.SANDBOX:
            raise RuntimeRequestError("Only sandbox delivery is implemented; this runtime cannot send a message.")
        self._ensure_agent_backend(request)
        if not has_typesafe_key():
            raise RuntimeRequestError("TYPESAFE_API_KEY is required for interactive execution.")
        if run["status"] == RunStatus.AWAITING_REVIEW.value:
            return self.get_run(run_id)
        if run["status"] == RunStatus.COMPLETE.value:
            return self.get_run(run_id)

        evaboot = None
        if request.data_source == DataSource.EVABOOT:
            if not request.live_evaboot_authorized:
                raise RuntimeRequestError("Live Evaboot work was not explicitly authorized.")
            evaboot = EvabootMCPAdapter()
        registry = ToolRegistry(
            self.store, goal=request, plan=plan,
            public_csv_path=self.settings.public_csv_path,
            synthetic_csv_path=self.synthetic_csv,
            evaboot=evaboot,
        )
        self.store.update_run(run_id, status=RunStatus.RUNNING.value)
        self._recover_completed_export(run_id, request)

        while True:
            run = self.store.get_run(run_id) or {}
            counters = dict(run.get("counters", {}))
            if counters.get("stop_requested"):
                self.store.update_run(run_id, status=RunStatus.COMPLETE.value)
                self.store.add_event(run_id, step="run.stopped", summary="Owner stop control ended the run.")
                break
            if int(counters.get("tool_calls", 0)) >= plan.max_tool_calls:
                self._budget_event(run_id, "tool_call_budget_exhausted")
                self.store.update_run(run_id, status=RunStatus.BUDGET_EXHAUSTED.value)
                break
            allowed = self._allowed_tools(run_id, request, plan, counters)
            state = self._agent_state(run_id, request, plan, allowed, counters)
            try:
                selected = self._agent_for_request(request).next_action(run_id=run_id, request=request, plan=plan,
                                                  state=state, allowed_tools=allowed)
            except CostLimitReached as exc:
                self._budget_event(run_id, str(exc))
                self.store.update_run(run_id, status=RunStatus.BUDGET_EXHAUSTED.value)
                break
            except (AgentConfigurationError, AgentOutputError) as exc:
                self.store.add_event(run_id, step="executor.model_error", summary="Tool-selection call failed; no fallback action was used.",
                                     error_status=type(exc).__name__)
                self.store.update_run(run_id, status=RunStatus.DEFERRED.value)
                break

            action: ToolAction = selected.value
            if action.tool not in allowed:
                counters["policy_rejections"] = int(counters.get("policy_rejections", 0)) + 1
                self.store.add_event(run_id, step="policy.action_rejected", summary="Python rejected a tool outside the current state permissions.",
                                     arguments={"tool": action.tool.value}, error_status="tool_not_allowed")
                self.store.update_run(run_id, counters=counters)
                if counters["policy_rejections"] >= 2:
                    self.store.update_run(run_id, status=RunStatus.DEFERRED.value)
                    break
                continue
            if action.tool == ToolName.DONE:
                pending = [lead for lead in self.store.list_leads(run_id) if lead["status"] == LeadStatus.AWAITING_REVIEW.value]
                export = self.store.get_export(run_id)
                if pending:
                    status = RunStatus.AWAITING_REVIEW.value
                elif export and export["status"] != "complete":
                    status = RunStatus.DEFERRED.value
                    self.store.add_event(run_id, step="executor.deferred", summary="Run stopped with an incomplete export after the bounded polling window.",
                                         result={"export_status": export["status"], "poll_count": export["poll_count"]})
                else:
                    status = RunStatus.COMPLETE.value
                self.store.add_event(run_id, step="executor.finished", summary="Agent selected its bounded stopping condition.",
                                     result={"pending_review_count": len(pending)})
                self.store.update_run(run_id, status=status)
                break

            if action.tool == ToolName.BUILD_SEARCH:
                if counters.get("search_builds", 0) > 0:
                    action.revised_search_prompt = action.revised_search_prompt or ""
                    if action.revised_search_prompt.strip() == (run.get("plan") or {}).get("search_prompt", "").strip():
                        counters["policy_rejections"] = int(counters.get("policy_rejections", 0)) + 1
                        self.store.add_event(run_id, step="policy.search_revision_rejected", summary="Revision did not change the query; the revision budget was preserved.",
                                             error_status="no_query_change")
                        self.store.update_run(run_id, counters=counters)
                        if counters["policy_rejections"] >= 2:
                            self.store.update_run(run_id, status=RunStatus.DEFERRED.value)
                            break
                        continue
                    if counters.get("search_revisions", 0) >= plan.max_search_revisions:
                        counters["policy_rejections"] = int(counters.get("policy_rejections", 0)) + 1
                        self.store.add_event(run_id, step="policy.search_revision_rejected", summary="Python enforced the search revision cap.",
                                             error_status="search_revision_budget_exhausted")
                        self.store.update_run(run_id, counters=counters)
                        if counters["policy_rejections"] >= 2:
                            self.store.update_run(run_id, status=RunStatus.DEFERRED.value)
                            break
                        continue

            result = registry.call(run_id, action)
            counters["tool_calls"] = int(counters.get("tool_calls", 0)) + 1
            counters["last_tool"] = action.tool.value
            if action.tool == ToolName.BUILD_SEARCH and result.ok:
                if counters.get("search_builds", 0) > 0:
                    counters["search_revisions"] = int(counters.get("search_revisions", 0)) + 1
                counters["search_builds"] = int(counters.get("search_builds", 0)) + 1
            if action.tool == ToolName.READ_PRIOR_FEEDBACK and result.ok:
                counters["feedback_read"] = True
                counters["feedback_reason_counts"] = result.data.get("reason_counts", [])
            if not result.ok:
                failures = dict(counters.get("tool_failures", {}))
                key = action.tool.value
                failures[key] = int(failures.get(key, 0)) + 1
                counters["tool_failures"] = failures
            self.store.update_run(run_id, counters=counters)
            self.store.add_event(
                run_id, step=f"tool.{action.tool.value}", summary=result.summary,
                arguments=self._safe_action(action), result=self._safe_tool_result(result),
                error_status=result.error_code if not result.ok else None,
            )

            if not result.ok:
                if result.retryable and counters.get("tool_failures", {}).get(action.tool.value, 0) < 2:
                    continue
                self._handle_tool_failure(run_id, action, result)
                if action.tool in {ToolName.BUILD_SEARCH, ToolName.START_EXPORT, ToolName.GET_EXPORT_STATUS}:
                    self.store.add_event(run_id, step="tool.retry_exhausted", summary="The bounded retry budget for a required search/export operation was exhausted; the run was deferred.",
                                         arguments={"tool": action.tool.value,
                                                    "attempts": counters.get("tool_failures", {}).get(action.tool.value, 0)},
                                         error_status=result.error_code or "tool_error")
                    self.store.update_run(run_id, status=RunStatus.DEFERRED.value)
                    break
                continue

            if action.tool == ToolName.GET_EXPORT_STATUS and result.data.get("status") == "complete":
                self._ingest_export_leads(run_id, result.data.get("leads", []), request)
            elif action.tool == ToolName.VERIFY_EMAIL:
                status = result.data.get("email_status", "unknown")
                self.store.add_event(run_id, step="policy.email_verification", summary="Email verification hard gate evaluated.",
                                     arguments={"lead_id": action.lead_id},
                                     result={"status": status, "actionable": status == "safe"})
            elif action.tool == ToolName.INSPECT_LEAD:
                try:
                    self._judge_inspection(run_id, request, plan, action, result)
                except CostLimitReached as exc:
                    self._budget_event(run_id, str(exc))
                    self.store.update_run(run_id, status=RunStatus.BUDGET_EXHAUSTED.value)
                    break
                except JevConfigurationError as exc:
                    current = self.store.get_lead(run_id, action.lead_id or "")
                    if current:
                        self.store.save_lead(run_id, current["lead_id"], current["fields"], current["metadata"],
                                             LeadStatus.DEFERRED.value, "jev_unavailable_no_fallback",
                                             current["draft"], {"route": "deferred", "route_reason": "jev_unavailable"})
                    self.store.add_event(run_id, step="policy.jev_unavailable", summary="Prospect deferred because Jev was unavailable; no fallback or unreviewed draft was permitted.",
                                         arguments={"lead_id": action.lead_id}, error_status=type(exc).__name__)
                except (AgentConfigurationError, AgentOutputError) as exc:
                    current = self.store.get_lead(run_id, action.lead_id or "")
                    if current:
                        status = (LeadStatus.AWAITING_REVIEW.value if request.execution_mode == ExecutionMode.ASSISTED
                                  else LeadStatus.SKIPPED.value)
                        self.store.save_lead(run_id, current["lead_id"], current["fields"], current["metadata"],
                                             status, "draft_unavailable_after_retry", current["draft"], current["judgments"])
                    self.store.add_event(run_id, step="policy.draft_unavailable", summary="Drafting failed after the bounded model retry; assisted mode is reviewable, autonomous mode skips.",
                                         arguments={"lead_id": action.lead_id}, error_status=type(exc).__name__)
                    if request.execution_mode == ExecutionMode.ASSISTED:
                        self.store.update_run(run_id, status=RunStatus.AWAITING_REVIEW.value)
                        break
                if request.execution_mode == ExecutionMode.ASSISTED:
                    current = self.store.get_lead(run_id, action.lead_id or "")
                    if current and current["status"] == LeadStatus.AWAITING_REVIEW.value:
                        self.store.update_run(run_id, status=RunStatus.AWAITING_REVIEW.value)
                        break

        return self.get_run(run_id)

    def resume(self, run_id: str) -> dict[str, Any]:
        run = self.store.get_run(run_id)
        if not run:
            raise RuntimeRequestError("Run not found.")
        if run["status"] in {RunStatus.COMPLETE.value, RunStatus.BUDGET_EXHAUSTED.value}:
            return self.get_run(run_id)
        export = self.store.get_export(run_id)
        if export and export["status"] != "complete" and export["poll_count"] >= 4:
            self.store.update_export(run_id, status=export["status"], poll_count=0,
                                     fail_once=export["fail_once"])
            self.store.add_event(run_id, step="export.resumed", summary="Resumed the existing asynchronous export in a fresh bounded polling window; the job was not duplicated.",
                                 result={"job_id": export["job_id"], "previous_poll_count": export["poll_count"]})
        return self.execute(run_id)

    def stop(self, run_id: str) -> dict[str, Any]:
        run = self.store.get_run(run_id)
        if not run:
            raise RuntimeRequestError("Run not found.")
        counters = dict(run["counters"])
        counters["stop_requested"] = True
        self.store.update_run(run_id, counters=counters)
        self.store.add_event(run_id, step="run.stop_requested", summary="Owner requested the persistent stop control.")
        return self.get_run(run_id)

    def review(self, run_id: str, lead_id: str, request: ReviewRequest) -> dict[str, Any]:
        run = self.store.get_run(run_id)
        lead = self.store.get_lead(run_id, lead_id)
        if not run or not lead:
            raise RuntimeRequestError("Run or lead not found.")
        goal = GoalRequest.model_validate(run["goal"])
        if lead["status"] != LeadStatus.AWAITING_REVIEW.value:
            raise RuntimeRequestError("This lead is not waiting for human review.")
        if request.decision in {"reject", "edit"} and not request.reason.strip():
            raise RuntimeRequestError("Add a short correction reason so the feedback can be evaluated later.")
        if request.decision == "reject":
            reason_code = _feedback_reason_code(request.reason)
            self.store.add_feedback(run_id, lead_id, request.reason, reason_code)
            self.store.save_lead(run_id, lead_id, lead["fields"], lead["metadata"], LeadStatus.REJECTED.value,
                                 f"reviewer_rejected:{reason_code}", lead["draft"], lead["judgments"],
                                 {"decision": "reject", "reason": request.reason, "at": now_iso()})
            self.store.add_event(run_id, step="review.rejected", summary="Reviewer correction was persisted for future policy analysis.",
                                 arguments={"lead_id": lead_id, "reason_code": reason_code})
            self.store.update_run(run_id, status=RunStatus.PLANNED.value)
            return self.get_run(run_id)
        if request.decision == "edit":
            reason_code = _feedback_reason_code(request.reason)
            self.store.add_feedback(run_id, lead_id, request.reason, reason_code)
            lead_object = Lead(lead_id=lead_id, fields=lead["fields"], metadata=lead["metadata"])
            safe_fields = {key: value for key, value in lead["fields"].items() if key not in {"Email", "LinkedIn URL", "Sales Navigator URL"}}
            result = self._agent_for_request(goal).draft_claims(run_id=run_id, request=goal, lead=lead_object,
                                            evidence=safe_fields, correction=request.reason)
            proposal: EmailDraft = result.value
            refs = validate_claim_references([item.model_dump() for item in proposal.claims], lead["fields"])
            if refs:
                raise RuntimeRequestError("Edited draft could not be grounded to exact source fields; no draft was updated.")
            jev_result = self._assess(run_id, lead_object, goal.goal, [item.model_dump() for item in proposal.claims])
            policy = self.store.active_policy()
            missing = required_evidence_missing(
                lead_object,
                require_current_job_match=bool(policy.get("targeting_rules", {}).get("require_current_job_match")),
            )
            route, reason = route_judgment(jev_result.as_dict(), policy, missing_fields=missing,
                                           stale=evidence_is_stale(lead_object) if policy.get("targeting_rules", {}).get("exclude_stale_evidence") else False,
                                           claim_count=len(proposal.claims))
            if route != "qualified":
                raise RuntimeRequestError(f"Edited draft failed Jev/policy checks ({reason}); it remains blocked.")
            draft = self._draft_storage(proposal, lead_object)
            judgments = {**jev_result.as_dict(), "route": route, "route_reason": reason}
            self.store.save_lead(run_id, lead_id, lead["fields"], lead["metadata"], LeadStatus.AWAITING_REVIEW.value,
                                 "edited_draft_revalidated", draft, judgments)
            self.store.add_event(run_id, step="review.edited_and_revalidated", summary="Edited draft was rebuilt and Jev-checked against source fields.",
                                 arguments={"lead_id": lead_id, "reason_code": reason_code},
                                 result={"claim_count": len(proposal.claims), "jev_model": jev_result.model})
            self.store.update_run(run_id, status=RunStatus.AWAITING_REVIEW.value)
            return self.get_run(run_id)

        draft = lead.get("draft") or {}
        if not draft.get("body"):
            raise RuntimeRequestError("There is no grounded draft to approve; reject with a reason or resume the run after the failure is resolved.")
        inserted = self.store.sandbox_outbox(run_id, lead_id, {
            "subject": draft.get("subject", ""), "body": draft.get("body", ""),
            "policy_version": run["policy_version"], "recipient_stored": False,
        })
        self.store.save_lead(run_id, lead_id, lead["fields"], lead["metadata"], LeadStatus.APPROVED.value,
                             "reviewer_approved_sandbox", lead["draft"], lead["judgments"],
                             {"decision": "approve", "reason": request.reason, "at": now_iso()})
        self.store.add_event(run_id, step="review.approved", summary="Draft approved; idempotent sandbox outbox recorded without contacting a recipient.",
                             arguments={"lead_id": lead_id}, result={"outbox_created": inserted, "delivery_mode": "sandbox"})
        self._record_sandbox_delivery(run_id, lead_id, lead["fields"], lead["metadata"], draft)
        self.store.update_run(run_id, status=RunStatus.PLANNED.value)
        return self.get_run(run_id)

    def ingest_outcome(self, run_id: str, data: dict[str, Any]) -> dict[str, Any]:
        lead = self.store.get_lead(run_id, data["lead_id"])
        if not lead:
            raise RuntimeRequestError("Outcome lead is not part of this run.")
        external_id = data["external_event_id"]
        if self.store.outcome_exists(external_id):
            return {"accepted": False, "duplicate": True, "external_event_id": external_id}
        event_type = data["event_type"]
        occurred = data.get("occurred_at") or now_iso()
        reply_class = None
        probabilities = None
        body_hash = None
        if event_type == "reply":
            text = data.get("text") or ""
            try:
                result = self.jev.classify_reply(text)
                self._record_jev_event(run_id, "jev.classify_reply", result,
                                       arguments={"lead_id": data["lead_id"], "reply_character_count": len(text)})
                reply_class, reason = classify_reply(result.as_dict(), self.store.active_policy(), text)
                probabilities = result.answers["intent"].get("probabilities", {})
                body_hash = result.answers["intent"].get("text_sha256")
            except Exception as exc:
                # No canned classifier: uncertain/unavailable replies pause future contact.
                reply_class = "unclear"
                reason = "jev_unavailable_contact_paused"
                if text:
                    body_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
                self.store.add_event(run_id, step="outcome.classification_deferred",
                                     summary="Jev reply classification was unavailable; automated contact was paused.",
                                     arguments={"lead_id": data["lead_id"]}, error_status=type(exc).__name__)
            inserted = self.store.add_outcome(external_id, run_id, data["lead_id"], event_type, occurred,
                                              reply_class, probabilities, body_hash)
            if not inserted:
                return {"accepted": False, "duplicate": True, "external_event_id": external_id}
            if reply_class == "opt_out":
                self.store.add_suppression(lead["fields"].get("Email", ""), "opt_out", run_id)
            elif reply_class == "unclear":
                self.store.add_suppression(lead["fields"].get("Email", ""), "ambiguous_reply_pause", run_id)
            self.store.add_event(run_id, step="outcome.reply_classified", summary="Reply intent stored; ambiguity or opt-out suppresses future contact.",
                                 arguments={"lead_id": data["lead_id"]},
                                 result={"reply_class": reply_class, "reason": reason,
                                         "probabilities": probabilities or {}, "suppressed": reply_class in {"opt_out", "unclear"}})
            if reply_class in {"opt_out", "unclear"}:
                self.store.save_lead(run_id, data["lead_id"], lead["fields"], lead["metadata"], LeadStatus.SUPPRESSED.value,
                                     f"reply_{reply_class}", lead["draft"], lead["judgments"], lead["review"])
            self._observe_candidate_canaries(run_id, external_id, lead, event_type, occurred, reply_class)
            return {"accepted": True, "duplicate": False, "reply_class": reply_class,
                    "probabilities": probabilities or {}, "suppressed": reply_class in {"opt_out", "unclear"}}

        inserted = self.store.add_outcome(external_id, run_id, data["lead_id"], event_type, occurred,
                                          None, None, None)
        if not inserted:
            return {"accepted": False, "duplicate": True, "external_event_id": external_id}
        if event_type == "bounce":
            self.store.add_suppression(lead["fields"].get("Email", ""), "bounce", run_id)
        self.store.add_event(run_id, step=f"outcome.{event_type}", summary="Outcome event stored without inferring a business result.",
                             arguments={"lead_id": data["lead_id"], "occurred_at": occurred})
        self._observe_candidate_canaries(run_id, external_id, lead, event_type, occurred, None)
        return {"accepted": True, "duplicate": False, "event_type": event_type}

    def list_runs(self) -> dict[str, Any]:
        outbox = [{key: item[key] for key in ("run_id", "lead_id", "status", "created_at")}
                  for item in self.store.list_outbox()]
        return {"runs": self.store.list_runs(), "policy": self.store.active_policy(),
                "policies": self.store.list_policies(), "feedback_summary": self.store.feedback_summary(),
                "outcomes": self.store.outcomes_summary(), "outbox": outbox,
                "candidates": self.store.list_candidates()}

    def improve(self, run_id: str) -> dict[str, Any]:
        run = self.store.get_run(run_id)
        if not run:
            raise RuntimeRequestError("Run not found.")
        request = GoalRequest.model_validate(run["goal"])
        pending = [item for item in self.store.list_candidates()
                   if item["status"] in {"awaiting_human_approval", "awaiting_canary"}]
        if pending:
            raise RuntimeRequestError(
                "Resolve the existing policy candidate before evaluating another; policy baselines must remain stable."
            )
        self._ensure_agent_backend(request)
        if not has_typesafe_key():
            raise RuntimeRequestError("TYPESAFE_API_KEY is required for interactive evaluation.")
        dataset = load_evaluation_cases(RESOURCE_ROOT / "data" / "evaluation_cases.json")
        calibration_rows = [case for case in dataset["cases"] if case["split"] == "calibration" and "reply" not in case]
        calibration_reply_rows = [case for case in dataset["cases"] if case["split"] == "calibration" and "reply" in case]
        heldout_rows = [case for case in dataset["cases"] if case["split"] == "heldout" and "reply" not in case]
        reply_rows = [case for case in dataset["cases"] if case["split"] == "heldout" and "reply" in case]
        if len(heldout_rows) < 8:
            raise RuntimeRequestError("The locked held-out set has fewer than eight qualification cases; promotion is disabled.")

        active_policy = deepcopy(self.store.active_policy())
        calibration = self._evaluate_cases(run_id, request.goal, calibration_rows)
        calibrated = calibrate_thresholds(calibration, active_policy,
                                          minimum_precision=json.loads(self.acceptance_file.read_text(encoding="utf-8"))["minimum_auto_approved_precision"])
        evaluation_policy = deepcopy(active_policy)
        evaluation_policy["question_thresholds"] = calibrated["thresholds"]
        calibration_replies = self._evaluate_replies(run_id, calibration_reply_rows)
        reply_calibration = calibrate_reply_threshold(calibration_replies["case_results"], active_policy)
        evaluation_policy["reply_policy"]["clear_label_probability"] = reply_calibration["selected_threshold"]
        calibration_summary = {**self._summarize_case_evaluation(calibration, policy=evaluation_policy),
                               "threshold_calibration": calibrated,
                               "reply_threshold_calibration": reply_calibration}
        calibration_summary["reply_metrics"] = self._summarize_replies(calibration_replies["case_results"], evaluation_policy)
        feedback = [item for item in self.store.list_feedback(100) if item["run_id"] == run_id]
        proposal_result = self._agent_for_request(request).propose_improvement(run_id=run_id, request=request,
                                                         feedback=feedback,
                                                         calibration_summary=calibration_summary)
        proposal = proposal_result.value
        if proposal.get("change_type") not in {"require_current_job_match", "exclude_stale_evidence"}:
            raise RuntimeRequestError("The proposed change is outside the Python whitelist.")

        heldout = self._evaluate_cases(run_id, request.goal, heldout_rows)
        heldout_replies = self._evaluate_replies(run_id, reply_rows)
        reply_results = self._summarize_replies(heldout_replies["case_results"], evaluation_policy)
        baseline_reply_results = self._summarize_replies(heldout_replies["case_results"], active_policy)
        baseline = active_policy
        candidate_policy = deepcopy(evaluation_policy)
        candidate_policy["version"] = f"{baseline['version']}-candidate-{uuid.uuid4().hex[:6]}"
        candidate_policy.setdefault("targeting_rules", {})[proposal["change_type"]] = True
        candidate = self._summarize_case_evaluation(heldout, policy=candidate_policy)
        base_metrics = self._summarize_case_evaluation(heldout, policy=baseline)
        calibrated_baseline = self._summarize_case_evaluation(heldout, policy=evaluation_policy)
        acceptance = json.loads(self.acceptance_file.read_text(encoding="utf-8"))
        critical_violations = candidate["critical_violations"]
        passed, failures = promotion_gate(base_metrics["qualification"], candidate["qualification"],
                                          acceptance, critical_violations=critical_violations,
                                          reply_class_errors=reply_results["error_count"])
        evaluation = {
            "dataset_version": dataset["version"], "warning": dataset["warning"],
            "heldout_case_count": len(heldout_rows), "reply_metrics": reply_results,
            "baseline_reply_metrics": baseline_reply_results,
            "baseline": base_metrics, "candidate": candidate,
            "calibrated_baseline": calibrated_baseline,
            "acceptance_passed": passed, "acceptance_failures": failures,
            "critical_violations": critical_violations,
            "calibration": calibration_summary,
            "threshold_calibration": calibrated,
            "evaluation_policy": evaluation_policy,
            "policy_before": baseline,
        }
        candidate_id = str(uuid.uuid4())
        status = "awaiting_human_approval" if request.execution_mode == ExecutionMode.ASSISTED else "awaiting_canary"
        if not passed:
            status = "rejected_by_gate"
        if request.execution_mode == ExecutionMode.AUTONOMOUS and not passed:
            self.store.add_event(run_id, step="policy.rollback", summary="Candidate failed the fixed held-out guardrails; the prior active version remains unchanged.",
                                 result={"candidate_id": candidate_id, "active_version": baseline["version"],
                                         "failures": failures})
        self.store.save_candidate(candidate_id, run_id, proposal,
                                  {**evaluation, "candidate_policy": candidate_policy}, status)
        if status == "awaiting_canary":
            self.store.add_event(
                run_id, step="policy.canary_started",
                summary="Candidate passed locked held-out checks and entered a post-creation shadow canary.",
                result={"candidate_id": candidate_id, "observed_outcomes": 0,
                        "required_observations": acceptance["canary_minimum_observations"],
                        "maximum_regression": acceptance["canary_maximum_regression"]},
            )
        self.store.add_event(run_id, step="policy.candidate_evaluated", summary="Candidate change compared with the locked held-out set; labels and acceptance rules were not editable.",
                             arguments={"candidate_id": candidate_id, "change_type": proposal["change_type"]},
                             result={"status": status, "acceptance_passed": passed,
                                     "failures": failures, "heldout_case_count": len(heldout_rows)})
        return self.store.get_candidate(candidate_id) or {}

    def decide_candidate(self, candidate_id: str, decision: str) -> dict[str, Any]:
        candidate = self.store.get_candidate(candidate_id)
        if not candidate:
            raise RuntimeRequestError("Candidate not found.")
        run = self.store.get_run(candidate["run_id"])
        if not run:
            raise RuntimeRequestError("Candidate run not found.")
        if run["goal"].get("execution_mode") != ExecutionMode.ASSISTED.value:
            raise RuntimeRequestError("Human policy approval is available only for assisted runs.")
        if candidate["status"] != "awaiting_human_approval":
            raise RuntimeRequestError("Candidate is not awaiting human approval.")
        if decision not in {"approve", "reject"}:
            raise RuntimeRequestError("Decision must be approve or reject.")
        if decision == "approve":
            if not candidate["evaluation"].get("acceptance_passed"):
                raise RuntimeRequestError("Candidate failed the fixed evaluation gate and cannot be promoted.")
            policy = candidate["evaluation"]["candidate_policy"]
            previous = self.store.active_policy()["version"]
            self.store.promote_policy(policy)
            status = "human_approved"
            step = "policy.promoted"
            result = {"candidate_id": candidate_id, "version": policy["version"], "previous_version": previous}
        else:
            status = "human_rejected"
            step = "policy.candidate_rejected"
            result = {"candidate_id": candidate_id, "active_version": self.store.active_policy()["version"]}
        self.store.update_candidate_status(candidate_id, status)
        self.store.add_event(candidate["run_id"], step=step, summary="Policy candidate decision recorded; active policy history retained.", result=result)
        return self.store.get_candidate(candidate_id) or {}

    def rollback_candidate(self, candidate_id: str) -> dict[str, Any]:
        candidate = self.store.get_candidate(candidate_id)
        if not candidate:
            raise RuntimeRequestError("Candidate not found.")
        evaluation = candidate["evaluation"]
        previous = evaluation.get("policy_before")
        if not previous:
            raise RuntimeRequestError("Candidate has no recorded prior policy for rollback.")
        if self.store.active_policy()["version"] != evaluation.get("candidate_policy", {}).get("version"):
            raise RuntimeRequestError("This candidate is not the active policy; rollback would overwrite a newer decision.")
        self.store.promote_policy(previous)
        self.store.update_candidate_status(candidate_id, "rolled_back")
        self.store.add_event(candidate["run_id"], step="policy.rollback", summary="Active candidate policy reverted to its recorded prior version.",
                             result={"candidate_id": candidate_id, "restored_version": previous["version"]})
        return self.store.get_candidate(candidate_id) or {}

    def _observe_candidate_canaries(self, run_id: str, external_event_id: str, lead: dict[str, Any],
                                    event_type: str, occurred_at: str,
                                    reply_class: str | None) -> None:
        if event_type not in {"reply", "bounce", "conversion"}:
            return
        occurred = _parse_iso_datetime(occurred_at)
        if occurred is None:
            return
        acceptance = json.loads(self.acceptance_file.read_text(encoding="utf-8"))
        for candidate in self.store.list_candidates():
            if candidate["run_id"] != run_id or candidate["status"] != "awaiting_canary":
                continue
            created = _parse_iso_datetime(candidate["created_at"])
            if created is None or occurred < created:
                continue
            evaluation = candidate["evaluation"]
            baseline = evaluation.get("policy_before", {})
            proposed = evaluation.get("candidate_policy", {})
            if self.store.active_policy()["version"] != baseline.get("version"):
                self.store.update_candidate_status(candidate["id"], "superseded")
                self.store.add_event(
                    run_id, step="policy.canary_superseded",
                    summary="Shadow canary stopped because its recorded baseline is no longer active.",
                    result={"candidate_id": candidate["id"],
                            "active_version": self.store.active_policy()["version"]},
                )
                continue
            candidate_approved = self._lead_approved_by_policy(lead, proposed)
            baseline_approved = self._lead_approved_by_policy(lead, baseline)
            adverse = event_type == "bounce" or (event_type == "reply" and reply_class != "positive")
            inserted = self.store.add_candidate_observation(
                candidate["id"], external_event_id,
                candidate_approved=candidate_approved,
                baseline_approved=baseline_approved,
                adverse=adverse,
            )
            if not inserted:
                continue
            summary = self.store.candidate_observation_summary(candidate["id"])
            self.store.add_event(
                run_id, step="policy.canary_observation",
                summary="A post-candidate outcome was scored under candidate and baseline policies.",
                arguments={"candidate_id": candidate["id"], "event_type": event_type},
                result={**summary, "candidate_approved": candidate_approved,
                        "baseline_approved": baseline_approved, "adverse": adverse},
            )
            minimum = int(acceptance["canary_minimum_observations"])
            if summary["candidate_observations"] < minimum:
                continue
            if not summary["baseline_observations"]:
                self.store.update_candidate_status(candidate["id"], "rejected_canary_incomparable")
                self.store.add_event(
                    run_id, step="policy.canary_rejected",
                    summary="Candidate canary had no comparable baseline-approved outcomes; promotion failed closed.",
                    result={"candidate_id": candidate["id"], **summary},
                )
                continue
            regression = float(summary["candidate_adverse_rate"] or 0) - float(summary["baseline_adverse_rate"] or 0)
            maximum = float(acceptance["canary_maximum_regression"])
            if regression <= maximum:
                self.store.promote_policy(proposed)
                self.store.update_candidate_status(candidate["id"], "auto_promoted")
                self.store.add_event(
                    run_id, step="policy.promoted",
                    summary="Candidate passed held-out evaluation and its post-creation shadow canary.",
                    result={"candidate_id": candidate["id"], "version": proposed["version"],
                            "previous_version": baseline["version"], "canary_regression": regression,
                            **summary},
                )
            else:
                self.store.update_candidate_status(candidate["id"], "rejected_by_canary")
                self.store.add_event(
                    run_id, step="policy.canary_rejected",
                    summary="Candidate adverse-outcome rate exceeded the configured canary regression bound.",
                    result={"candidate_id": candidate["id"], "canary_regression": regression,
                            "maximum_regression": maximum, **summary},
                )

    @staticmethod
    def _lead_approved_by_policy(lead: dict[str, Any], policy: dict[str, Any]) -> bool:
        if not policy or not lead.get("judgments"):
            return False
        lead_object = Lead(lead_id=lead["lead_id"], fields=lead["fields"], metadata=lead["metadata"])
        rules = policy.get("targeting_rules", {})
        missing = required_evidence_missing(
            lead_object,
            require_current_job_match=bool(rules.get("require_current_job_match")),
        )
        stale = evidence_is_stale(lead_object) and bool(rules.get("exclude_stale_evidence"))
        claim_count = len((lead.get("draft") or {}).get("claims", []))
        route, _ = route_judgment(
            lead["judgments"], policy, missing_fields=missing, stale=stale, claim_count=claim_count,
        )
        return route == "qualified"

    def replay(self, run_id: str, lead_id: str) -> dict[str, Any]:
        run = self.store.get_run(run_id)
        lead = self.store.get_lead(run_id, lead_id)
        if not run or not lead:
            raise RuntimeRequestError("Run or lead not found.")
        events = [item for item in self.store.list_events(run_id) if item["arguments"].get("lead_id") == lead_id]
        return {"run_id": run_id, "lead": self._public_lead(lead), "policy_version": run["policy_version"], "events": events}

    def _evaluate_cases(self, run_id: str, goal: str, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results = []
        for case in cases:
            fields = {key: str(value) for key, value in case.get("fields", {}).items()}
            lead = Lead(lead_id=case["case_id"], fields=fields,
                        metadata={"source": "synthetic_evaluation_fixture", "source_timestamp": fields.get("Source Updated At")})
            label = case["labels"].get("claim_supported")
            claims = []
            if label is not None:
                industry = fields.get("Company Industry", "")
                if label:
                    claim_text = f"Your company operates in {industry}."
                    excerpt = industry
                    field_name = "Company Industry"
                else:
                    source_field = "Company Employee Exact Count" if fields.get("Company Employee Exact Count") else "Company Description"
                    excerpt = fields.get(source_field, "")
                    claim_text = "Your company has more than 10000 employees and operates on five continents."
                    field_name = source_field
                claims = [{"text": claim_text, "evidence": [{"field": field_name, "excerpt": excerpt}]}]
            try:
                judgment = self.jev.assess_lead(lead, goal=goal, claims=claims)
            except Exception as exc:
                self.store.add_event(run_id, step="evaluation.jev_unavailable", summary="Locked-set evaluation stopped because Jev was unavailable; no candidate was promoted.",
                                     arguments={"case_id": case["case_id"], "split": case["split"]},
                                     error_status=type(exc).__name__)
                raise JevConfigurationError("Jev was unavailable during evaluation; no candidate was created.") from exc
            self._record_jev_event(run_id, "jev.evaluation_case", judgment,
                                   arguments={"case_id": case["case_id"], "split": case["split"], "claim_count": len(claims)})
            results.append({"case_id": case["case_id"], "labels": case["labels"], "judgment": judgment.as_dict(),
                            "current_job_present": bool(fields.get("Current Job", "").strip()),
                            "missing_fields": required_evidence_missing(lead), "stale": evidence_is_stale(lead)})
        return results

    def _evaluate_replies(self, run_id: str, cases: list[dict[str, Any]]) -> dict[str, Any]:
        case_results = []
        for case in cases:
            try:
                result = self.jev.classify_reply(case["reply"])
            except Exception as exc:
                self.store.add_event(run_id, step="evaluation.jev_unavailable", summary="Reply fixture evaluation stopped because Jev was unavailable; no candidate was promoted.",
                                     arguments={"case_id": case["case_id"], "split": case["split"]},
                                     error_status=type(exc).__name__)
                raise JevConfigurationError("Jev was unavailable during reply evaluation; no candidate was created.") from exc
            self._record_jev_event(run_id, "jev.evaluation_reply", result,
                                   arguments={"case_id": case["case_id"], "split": case["split"]})
            answer = result.answers.get("intent", {})
            probabilities = answer.get("probabilities", {})
            _, reason = classify_reply(result.as_dict(), self.store.active_policy(), case["reply"])
            case_results.append({"case_id": case["case_id"], "actual": case["labels"]["reply_intent"],
                                 "choice": answer.get("choice", "unclear"),
                                 "probabilities": probabilities,
                                 "max_probability": max((float(value) for value in probabilities.values()), default=0),
                                 "deterministic_opt_out": reason == "deterministic_opt_out_signal"})
        return {"case_results": case_results,
                **self._summarize_replies(case_results, self.store.active_policy())}

    @staticmethod
    def _summarize_replies(case_results: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any]:
        labels, predictions = [], []
        for item in case_results:
            judgment = {"answers": {"intent": {"choice": item["choice"],
                                                   "probabilities": item["probabilities"]}}}
            text = "Please unsubscribe" if item["deterministic_opt_out"] else ""
            prediction, _ = classify_reply(judgment, policy, text)
            labels.append(item["actual"])
            predictions.append(prediction)
        return reply_metrics(labels, predictions)

    @staticmethod
    def _summarize_case_evaluation(results: list[dict[str, Any]], policy: dict[str, Any] | None = None) -> dict[str, Any]:
        policy = policy or {}
        labels, predictions = [], []
        critical = 0
        question_labels: dict[str, list[bool]] = {key: [] for key in ("current_role_fit", "company_fit", "claim_support", "contradiction_free")}
        question_probs: dict[str, list[float]] = {key: [] for key in question_labels}
        for item in results:
            answers = item["judgment"].get("answers", {})
            truth = item["labels"]
            actual = (truth.get("role_match") is True and truth.get("company_match") is True and
                      truth.get("claim_supported") is True and truth.get("contradiction") is False and
                      not item["missing_fields"])
            missing = list(item["missing_fields"])
            rules = policy.get("targeting_rules", {})
            if rules.get("require_current_job_match") and not item.get("current_job_present"):
                missing.append("Current Job")
            stale = item["stale"] and rules.get("exclude_stale_evidence", False)
            evaluation_policy = policy or {"question_thresholds": {"current_role_fit": .8, "company_fit": .8,
                                                                     "claim_support": .82, "contradiction_risk_max": .15}}
            route, _ = route_judgment(item["judgment"], evaluation_policy,
                missing_fields=missing, stale=stale,
                claim_count=sum(1 for key in answers if key.startswith("claim_support_")))
            approved = route == "qualified"
            labels.append(actual)
            predictions.append(approved)
            contradiction = answers.get("evidence_contradictions", {}).get("probability_yes")
            if approved and not actual:
                critical += 1
            mapping = {
                "current_role_fit": (truth.get("role_match"), answers.get("current_role_fit", {}).get("probability_yes")),
                "company_fit": (truth.get("company_match"), answers.get("company_fit", {}).get("probability_yes")),
                "claim_support": (truth.get("claim_supported"), answers.get("claim_support_0", {}).get("probability_yes")),
                "contradiction_free": (None if truth.get("contradiction") is None else not truth.get("contradiction"),
                                       None if contradiction is None else 1 - contradiction),
            }
            for key, (actual_q, probability) in mapping.items():
                if actual_q is not None and probability is not None:
                    question_labels[key].append(bool(actual_q))
                    question_probs[key].append(float(probability))
        question_metrics = {}
        for key in question_labels:
            qlabels = question_labels[key]
            qprobs = question_probs[key]
            question_metrics[key] = binary_metrics(qlabels, [score >= .5 for score in qprobs], qprobs)
        return {"qualification": policy_metrics(labels, predictions),
                "question_metrics": question_metrics,
                "critical_violations": critical,
                "case_decisions": [{"case_id": item["case_id"], "predicted_qualified": predicted,
                                    "actual_qualified": actual}
                                   for item, predicted, actual in zip(results, predictions, labels)]}

    def report_markdown(self, run_id: str | None = None) -> str:
        runs = [self.store.get_run(run_id)] if run_id else [self.store.get_run(item["id"]) for item in self.store.list_runs(5)]
        runs = [run for run in runs if run]
        active = self.store.active_policy()
        lines = ["# Evaboot Agent Runtime — engineering report", "",
                 "A bounded SDR runtime. Simulator tool results are fixtures; sandbox delivery never contacts prospects.", "",
                 "## Architecture", "",
                 "OpenAI plans, selects typed tools, drafts cited claims, and proposes policy changes. Python enforces budgets, hard gates, persistence, and the outbox. Jev supplies typed evidence and reply judgments. SQLite stores runs, traces, feedback, policy versions, exports, outcomes, and idempotency keys.", "",
                 "```mermaid", "flowchart LR\n  Goal --> Planner[OpenAI planner]\n  Planner --> Tools[Typed Evaboot-shaped tools]\n  Tools --> Gates[Python gates + Jev judgments]\n  Gates --> Review[Assisted review or autonomous sandbox]\n  Review --> Memory[SQLite trace and outcomes]\n  Memory --> Eval[Held-out evaluation]\n  Eval --> Policy[Versioned policy / rollback]", "```", "",
                 f"Active policy: `{active['version']}`. Jev model configured as `{self.settings.typesafe_model}`.", "",
                 "## Run summary", ""]
        for run in runs:
            if not run:
                continue
            counts: dict[str, int] = {}
            for lead in self.store.list_leads(run["id"]):
                counts[lead["status"]] = counts.get(lead["status"], 0) + 1
            counters = run["counters"]
            lines.extend([
                f"### `{run['id']}` — {run['status']}", "",
                f"Mode: {run['goal'].get('execution_mode')} / {run['goal'].get('data_source')} / sandbox. Policy: `{run['policy_version']}`.",
                f"Tool calls: {counters.get('tool_calls', 0)}. OpenAI tokens: {counters.get('input_tokens', 0)} in / {counters.get('output_tokens', 0)} out. Estimated OpenAI cost: ${float(counters.get('cost_usd', 0)):.6f}. Jev token cost is not reported by the configured TypeSafe account; token counts are in the trace.",
                "Lead outcomes: " + (", ".join(f"{key}={value}" for key, value in sorted(counts.items())) or "none"), "",
            ])
            trace = self.store.list_events(run["id"], 30)
            recovered = any(event["error_status"] == "temporary_timeout" for event in trace)
            lines.append(f"Recovered tool failure: {'yes' if recovered else 'not observed in this run'}. Reviewer feedback items: {len([x for x in self.store.list_feedback() if x['run_id'] == run['id']])}.")
            lines.append("")
            if trace:
                lines.append("Trace excerpt:")
                lines.append("")
                for event in trace[-8:]:
                    lines.append(f"- `{event['occurred_at']}` **{event['step']}** — {event['summary']}" + (f" (`{event['error_status']}`)" if event["error_status"] else ""))
                lines.append("")
        lines.extend([
            "## Evaluation and limits", "",
                 "The calibration and held-out fixtures are synthetic labels, not production evidence. Python selects thresholds from calibration labels only, freezes that threshold snapshot, then evaluates baseline and candidate on untouched held-out cases. Reports include precision and false-approval Wilson intervals, the negative-case denominator, action coverage, skip rate, per-question reliability bins/Brier scores, and reply-class confusion. Candidate reports are persisted below; no live-model evaluation is fabricated by offline tests.", "",
            "No email provider, CRM write, or outreach sender is configured. Live Evaboot export requires a separate explicit checkbox and quota preflight; Evaboot results are not a delivery authorization.", "",
            "## Next integration steps", "",
            "Add a configured live Evaboot MCP account, verify tool schemas against the connected account, provide an authorized delivery provider only if separately approved, then evaluate on independently labeled and group-separated outcomes.", "",
        ])
        dataset = load_evaluation_cases(RESOURCE_ROOT / "data" / "evaluation_cases.json")
        calibration_count = sum(1 for case in dataset["cases"] if case["split"] == "calibration" and "reply" not in case)
        calibration_reply_count = sum(1 for case in dataset["cases"] if case["split"] == "calibration" and "reply" in case)
        heldout_count = sum(1 for case in dataset["cases"] if case["split"] == "heldout" and "reply" not in case)
        reply_count = sum(1 for case in dataset["cases"] if case["split"] == "heldout" and "reply" in case)
        lines.extend([f"Evaluation dataset `{dataset['version']}`: {calibration_count} calibration qualification and {calibration_reply_count} calibration reply cases; {heldout_count} held-out qualification and {reply_count} held-out reply cases. {dataset['warning']}", ""])
        candidates = self.store.list_candidates(10)
        if candidates:
            lines.extend(["## Candidate evaluation report", ""])
            for item in candidates:
                evaluation = item["evaluation"]
                baseline = evaluation.get("baseline", {}).get("qualification", {})
                candidate = evaluation.get("candidate", {}).get("qualification", {})
                reply = evaluation.get("reply_metrics", {})
                baseline_reply = evaluation.get("baseline_reply_metrics", {})
                reply_threshold = evaluation.get("calibration", {}).get("reply_threshold_calibration", {}).get("selected_threshold")
                canary = item.get("canary", {})
                lines.extend([
                    f"### `{item['id']}` — {item['status']}", "",
                    f"Change: `{item['proposal'].get('change_type')}`. Held-out qualification cases: {evaluation.get('heldout_case_count', 0)}. Acceptance gate: {'pass' if evaluation.get('acceptance_passed') else 'fail'}.",
                    f"Baseline → candidate: precision {baseline.get('precision_among_auto_approved')} → {candidate.get('precision_among_auto_approved')}; false approvals {baseline.get('false_auto_approved', 0)}/{baseline.get('negative_case_denominator', 0)} → {candidate.get('false_auto_approved', 0)}/{candidate.get('negative_case_denominator', 0)}; coverage {baseline.get('automatic_action_coverage')} → {candidate.get('automatic_action_coverage')}; skip rate {baseline.get('skip_rate')} → {candidate.get('skip_rate')}.",
                    f"Candidate 95% intervals: precision {candidate.get('precision_interval_95')}; false-approval rate {candidate.get('false_approval_interval_95')}. Calibration-selected thresholds: {evaluation.get('threshold_calibration', {}).get('thresholds', {})}.",
                    f"Reply fixtures: {reply.get('sample_count', 0)} cases, {baseline_reply.get('error_count', 0)} baseline → {reply.get('error_count', 0)} candidate class errors; calibrated clear-label floor={reply_threshold}. Gate failures: {', '.join(evaluation.get('acceptance_failures', [])) or 'none'}.", "",
                    f"Shadow canary: {canary.get('candidate_observations', 0)} candidate observations with adverse rate {canary.get('candidate_adverse_rate')}; baseline {canary.get('baseline_observations', 0)} observations with adverse rate {canary.get('baseline_adverse_rate')}.", "",
                ])
                for question, metrics in evaluation.get("candidate", {}).get("question_metrics", {}).items():
                    lines.append(f"- `{question}`: n={metrics.get('sample_count')}, Brier={metrics.get('brier_score')}, reliability bins={len(metrics.get('reliability_bins', []))}")
                lines.append("")
        return "\n".join(lines)

    def _allowed_tools(self, run_id: str, request: GoalRequest, plan: Plan, counters: dict[str, Any]) -> list[ToolName]:
        export = self.store.get_export(run_id)
        if not counters.get("search_builds"):
            result = [ToolName.BUILD_SEARCH]
            if self.store.list_feedback() and not counters.get("feedback_read"):
                result.append(ToolName.READ_PRIOR_FEEDBACK)
            return result
        if not export:
            result = [ToolName.START_EXPORT]
            revisions = int(counters.get("search_revisions", 0))
            if revisions < plan.max_search_revisions:
                result.append(ToolName.BUILD_SEARCH)
            if self.store.list_feedback() and not counters.get("feedback_read"):
                result.append(ToolName.READ_PRIOR_FEEDBACK)
            return result
        if export["status"] != "complete":
            if export["poll_count"] >= 4:
                return [ToolName.DONE]
            return [ToolName.GET_EXPORT_STATUS]
        leads = self.store.list_leads(run_id)
        verify = [lead for lead in leads if lead["status"] == LeadStatus.NEEDS_VERIFICATION.value]
        if verify:
            return [ToolName.VERIFY_EMAIL]
        inspectable = []
        for lead in leads:
            if lead["status"] == LeadStatus.ELIGIBLE.value:
                inspectable.append(lead)
            elif lead["status"] == LeadStatus.NEEDS_RESEARCH.value and not lead["metadata"].get("deep_inspected"):
                inspectable.append(lead)
        if inspectable:
            return [ToolName.INSPECT_LEAD]
        return [ToolName.DONE]

    def _agent_state(self, run_id: str, request: GoalRequest, plan: Plan,
                     allowed: list[ToolName], counters: dict[str, Any]) -> dict[str, Any]:
        run = self.store.get_run(run_id) or {}
        leads = self.store.list_leads(run_id)
        export = self.store.get_export(run_id)
        events = self.store.list_events(run_id, 20)
        return {
            "run_status": run.get("status"),
            "search_builds": counters.get("search_builds", 0),
            "search_revisions_used": counters.get("search_revisions", 0),
            "tool_calls_used": counters.get("tool_calls", 0),
            "tool_call_budget": plan.max_tool_calls,
            "model_cost_estimate_usd": round(float(counters.get("cost_usd", 0)), 6),
            "model_cost_ceiling_usd": request.max_model_cost_usd,
            "export": {"job_id": export["job_id"], "status": export["status"], "poll_count": export["poll_count"]} if export else None,
            "leads": [{"lead_id": lead["lead_id"], "status": lead["status"], "reason": lead["reason"],
                       "matches_filters": lead["fields"].get("Matches Filters", ""),
                       "email_status": lead["fields"].get("Email Status", ""),
                       "current_job_present": bool(lead["fields"].get("Current Job", "").strip()),
                       "company_industry": lead["fields"].get("Company Industry", "")[:80]}
                      for lead in leads],
            "feedback_available": bool(self.store.list_feedback()),
            "feedback_read": bool(counters.get("feedback_read")),
            "feedback_reason_counts": counters.get("feedback_reason_counts", []),
            "recent_observations": [{"step": event["step"], "summary": event["summary"],
                                     "error": event["error_status"]} for event in events[-8:]],
            "allowed_tools": [tool.value for tool in allowed],
        }

    def _recover_completed_export(self, run_id: str, request: GoalRequest) -> None:
        export = self.store.get_export(run_id)
        if export and export["status"] == "complete" and export.get("leads") and not self.store.list_leads(run_id):
            self._ingest_export_leads(run_id, export["leads"], request)

    def _ingest_export_leads(self, run_id: str, rows: list[dict[str, Any]], request: GoalRequest) -> None:
        existing = {lead["lead_id"] for lead in self.store.list_leads(run_id)}
        saved = self.store.list_leads(run_id)
        seen_emails = {item["fields"].get("Email", "").strip().lower() for item in saved if item["fields"].get("Email")}
        seen_profiles = {(item["fields"].get("Linkedin URL Unique ID") or item["fields"].get("LinkedIn URL", "")).strip().lower()
                         for item in saved if item["fields"].get("Linkedin URL Unique ID") or item["fields"].get("LinkedIn URL")}
        for item in rows[:request.max_leads]:
            lead_id = str(item.get("lead_id", ""))
            if not lead_id or lead_id in existing:
                continue
            fields = {str(key): str(value) for key, value in (item.get("fields") or {}).items() if value is not None}
            metadata = dict(item.get("metadata") or {})
            lead = Lead(lead_id=lead_id, fields=fields, metadata=metadata)
            status, reason = deterministic_gate(
                lead, seen_emails=seen_emails, seen_profiles=seen_profiles,
                user_suppressions=request.suppression_list,
                is_suppressed=self.store.is_suppressed(lead.email),
            )
            lead_status = {
                "eligible": LeadStatus.ELIGIBLE.value,
                "needs_verification": LeadStatus.NEEDS_VERIFICATION.value,
                "rejected": LeadStatus.REJECTED.value,
                "suppressed": LeadStatus.SUPPRESSED.value,
            }[status]
            self.store.save_lead(run_id, lead_id, fields, metadata, lead_status, reason)
            if lead.email:
                seen_emails.add(lead.email)
            if lead.profile_key:
                seen_profiles.add(lead.profile_key)
            self.store.add_event(run_id, step="policy.hard_gate", summary="Deterministic lead gate completed before semantic judgment.",
                                 arguments={"lead_id": lead_id},
                                 result={"decision": status, "reason_code": reason or "passed", "email_status": fields.get("Email Status", "")})

    def _judge_inspection(self, run_id: str, request: GoalRequest, plan: Plan,
                          action: ToolAction, result: ToolResult) -> None:
        lead_id = action.lead_id or ""
        record = self.store.get_lead(run_id, lead_id)
        if not record:
            return
        inspected = result.data.get("source_fields") or {}
        judged_lead = Lead(lead_id=lead_id, fields=inspected, metadata=record["metadata"])
        if action.detail_level == "deep":
            metadata = dict(record["metadata"])
            metadata["deep_inspected"] = True
            record["metadata"] = metadata
        policy = self.store.active_policy()
        missing = required_evidence_missing(
            judged_lead,
            require_current_job_match=bool(policy.get("targeting_rules", {}).get("require_current_job_match")),
        )
        stale = (evidence_is_stale(judged_lead) and
                 bool(policy.get("targeting_rules", {}).get("exclude_stale_evidence")))
        prior_draft = record.get("draft")
        claims = (prior_draft or {}).get("claims", [])
        judgment = self._assess(run_id, judged_lead, request.goal, claims)
        route, reason = route_judgment(judgment.as_dict(), policy,
                                       missing_fields=missing, stale=stale, claim_count=len(claims))
        if route in {"reject", "deferred"}:
            status = LeadStatus.REJECTED.value if route == "reject" else LeadStatus.DEFERRED.value
            self.store.save_lead(run_id, lead_id, record["fields"], record["metadata"], status, reason,
                                 prior_draft, {**judgment.as_dict(), "route": route, "route_reason": reason})
            self.store.add_event(run_id, step=f"policy.jev_{route}", summary="Jev result routed by thresholds; Python retained the final action authority.",
                                 arguments={"lead_id": lead_id, "question_ids": list(judgment.answers)},
                                 result={"route": route, "reason_code": reason,
                                         "judgments": _probabilities(judgment.answers)},
                                 duration_ms=judgment.duration_ms, model=judgment.model,
                                 input_tokens=judgment.input_tokens, output_tokens=judgment.output_tokens)
            return
        if route == "insufficient" and not prior_draft:
            if action.detail_level == "standard":
                self.store.save_lead(run_id, lead_id, record["fields"], record["metadata"], LeadStatus.NEEDS_RESEARCH.value,
                                     reason, None, {**judgment.as_dict(), "route": route, "route_reason": reason})
            else:
                self._uncertain_route(run_id, request, record, reason, judgment)
            self.store.add_event(run_id, step="policy.jev_insufficient", summary="Evidence remained uncertain; one deeper inspection is available before skip or assisted review.",
                                 arguments={"lead_id": lead_id, "question_ids": list(judgment.answers)},
                                 result={"route": route, "reason_code": reason,
                                         "judgments": _probabilities(judgment.answers)},
                                 duration_ms=judgment.duration_ms, model=judgment.model,
                                 input_tokens=judgment.input_tokens, output_tokens=judgment.output_tokens)
            return
        if route == "qualified" and not prior_draft:
            result_draft = self._agent_for_request(request).draft_claims(run_id=run_id, request=request, lead=judged_lead,
                                                  evidence=inspected)
            proposal: EmailDraft = result_draft.value
            claim_dicts = [claim.model_dump() for claim in proposal.claims]
            invalid_refs = validate_claim_references(claim_dicts, record["fields"])
            if invalid_refs:
                result_draft = self._agent_for_request(request).draft_claims(run_id=run_id, request=request, lead=judged_lead,
                                                      evidence=inspected,
                                                      correction="Correct the invalid evidence reference. Use an exact source-field excerpt or omit unsupported wording.")
                proposal = result_draft.value
                claim_dicts = [claim.model_dump() for claim in proposal.claims]
                invalid_refs = validate_claim_references(claim_dicts, record["fields"])
            if invalid_refs:
                self._uncertain_route(run_id, request, record, "draft_source_reference_invalid", judgment)
                self.store.add_event(run_id, step="policy.draft_blocked", summary="Python blocked a draft because cited field excerpts did not exist in the source row.",
                                     arguments={"lead_id": lead_id}, result={"failure_codes": invalid_refs})
                return
            claims = claim_dicts
            judgment = self._assess(run_id, judged_lead, request.goal, claims)
            route, reason = route_judgment(judgment.as_dict(), policy,
                                           missing_fields=missing, stale=stale, claim_count=len(claims))
            prior_draft = self._draft_storage(proposal, judged_lead)
            self.store.add_event(run_id, step="drafter.claims_created", summary="OpenAI proposed a source-cited draft; Python assembled the fixed email wrapper.",
                                 arguments={"lead_id": lead_id, "claim_count": len(claims)},
                                 result={"evidence_reference_count": sum(len(claim["evidence"]) for claim in claims),
                                         "subject_is_template": True},
                                 model=self.settings.openai_model)

        if route == "revise" and prior_draft:
            revisions = int(record["metadata"].get("draft_revision_count", 0))
            if revisions < 1:
                result_draft = self._agent_for_request(request).draft_claims(
                    run_id=run_id, request=request, lead=judged_lead, evidence=inspected,
                    correction="Jev found a proposed claim unsupported. Remove or rewrite it using only the cited source fields.",
                )
                proposal = result_draft.value
                claims = [claim.model_dump() for claim in proposal.claims]
                invalid_refs = validate_claim_references(claims, record["fields"])
                if not invalid_refs:
                    judgment = self._assess(run_id, judged_lead, request.goal, claims)
                    route, reason = route_judgment(judgment.as_dict(), policy,
                                                   missing_fields=missing, stale=stale, claim_count=len(claims))
                    prior_draft = self._draft_storage(proposal, judged_lead)
                    metadata = dict(record["metadata"])
                    metadata["draft_revision_count"] = revisions + 1
                    record["metadata"] = metadata
            if route != "qualified":
                if action.detail_level == "standard" and not record["metadata"].get("deep_inspected"):
                    self.store.save_lead(run_id, lead_id, record["fields"], record["metadata"], LeadStatus.NEEDS_RESEARCH.value,
                                         reason, prior_draft, {**judgment.as_dict(), "route": route, "route_reason": reason})
                    return
                self._uncertain_route(run_id, request, record, reason, judgment, prior_draft)
                return

        if route == "insufficient":
            if action.detail_level == "standard" and not record["metadata"].get("deep_inspected"):
                self.store.save_lead(run_id, lead_id, record["fields"], record["metadata"], LeadStatus.NEEDS_RESEARCH.value,
                                     reason, prior_draft, {**judgment.as_dict(), "route": route, "route_reason": reason})
            else:
                self._uncertain_route(run_id, request, record, reason, judgment, prior_draft)
            return
        if route != "qualified" or not prior_draft:
            self._uncertain_route(run_id, request, record, reason, judgment, prior_draft)
            return

        target_status = (LeadStatus.AWAITING_REVIEW.value if request.execution_mode == ExecutionMode.ASSISTED
                         else LeadStatus.APPROVED.value)
        judgments = {**judgment.as_dict(), "route": "qualified", "route_reason": reason,
                     "policy_version": self.store.active_policy()["version"]}
        self.store.save_lead(run_id, lead_id, record["fields"], record["metadata"], target_status,
                             "all_hard_and_jev_gates_passed", prior_draft, judgments)
        self.store.add_event(run_id, step="policy.recommendation_qualified", summary="Recommendation passed deterministic gates and every mandatory Jev threshold.",
                             arguments={"lead_id": lead_id, "evidence_fields": sorted({ref["field"] for claim in prior_draft["claims"] for ref in claim["evidence"]})},
                             result={"status": target_status, "claim_count": len(prior_draft["claims"]),
                                     "judgments": _probabilities(judgment.answers)},
                             duration_ms=judgment.duration_ms, model=judgment.model,
                             input_tokens=judgment.input_tokens, output_tokens=judgment.output_tokens)
        if request.execution_mode == ExecutionMode.AUTONOMOUS:
            self._record_sandbox_delivery(run_id, lead_id, record["fields"], record["metadata"], prior_draft)

    def _assess(self, run_id: str, lead: Lead, goal: str, claims: list[dict[str, Any]]) -> JevResult:
        try:
            result = self.jev.assess_lead(lead, goal=goal, claims=claims)
            self._record_jev_event(run_id, "jev.assess_lead", result,
                                   arguments={"lead_id": lead.lead_id,
                                              "evidence_fields": sorted(key for key, value in lead.fields.items() if value),
                                              "claim_count": len(claims)})
            return result
        except Exception as exc:
            self.store.add_event(run_id, step="jev.assess_lead", summary="Jev was unavailable; no semantic fallback was used.",
                                 arguments={"lead_id": lead.lead_id, "claim_count": len(claims)},
                                 error_status=type(exc).__name__)
            raise JevConfigurationError("Jev judgment unavailable; this prospect is deferred or skipped.") from exc

    def _record_jev_event(self, run_id: str, step: str, result: JevResult, arguments: dict[str, Any]) -> None:
        self.store.add_event(run_id, step=step, summary="Typed Jev judgment stored for independent Python routing.",
                             arguments=arguments,
                             result={"judgments": _probabilities(result.answers), "question_ids": list(result.answers)},
                             duration_ms=result.duration_ms, model=result.model,
                             input_tokens=result.input_tokens, output_tokens=result.output_tokens)

    def _uncertain_route(self, run_id: str, request: GoalRequest, record: dict[str, Any], reason: str,
                         judgment: JevResult, draft: dict[str, Any] | None = None) -> None:
        status = (LeadStatus.AWAITING_REVIEW.value if request.execution_mode == ExecutionMode.ASSISTED
                  else LeadStatus.SKIPPED.value)
        final_reason = reason if request.execution_mode == ExecutionMode.ASSISTED else f"{LeadStatus.SKIPPED.value}:{reason}"
        self.store.save_lead(run_id, record["lead_id"], record["fields"], record["metadata"], status,
                             final_reason, draft, {**judgment.as_dict(), "route": "insufficient", "route_reason": reason})

    @staticmethod
    def _draft_storage(proposal: EmailDraft, lead: Lead) -> dict[str, Any]:
        claims = [item.model_dump() for item in proposal.claims]
        return {"subject": proposal.subject, "body": proposal.render_body(lead.fields.get("First Name", "there")),
                "claims": claims,
                "evidence": [{"claim": claim["text"], "field": ref["field"], "excerpt": ref["excerpt"],
                              "source_id": lead.lead_id}
                             for claim in claims for ref in claim["evidence"]]}

    def _record_sandbox_delivery(self, run_id: str, lead_id: str, fields: dict[str, str], metadata: dict[str, Any], draft: dict[str, Any]) -> None:
        created = self.store.sandbox_outbox(run_id, lead_id, {
            "subject": draft.get("subject", ""), "body": draft.get("body", ""),
            "policy_version": (self.store.get_run(run_id) or {}).get("policy_version"),
            "recipient_stored": False,
        })
        self.store.add_event(run_id, step="delivery.sandbox", summary="Simulated sender recorded an idempotent delivery event; no recipient was contacted.",
                             arguments={"lead_id": lead_id}, result={"created": created, "simulated": True})
        delivery_id = f"sandbox:{run_id}:{lead_id}:delivered"
        self.store.add_outcome(delivery_id, run_id, lead_id, "delivered", now_iso(), None, None, None)
        reply = metadata.get("fixture_reply")
        if reply:
            result = self.ingest_outcome(run_id, {
                "external_event_id": f"sandbox:{run_id}:{lead_id}:reply",
                "lead_id": lead_id, "event_type": "reply", "occurred_at": now_iso(), "text": reply,
            })
            self.store.add_event(run_id, step="outcome.fixture", summary="Simulator fixture reply was labeled and passed to Jev.",
                                 arguments={"lead_id": lead_id},
                                 result={"reply_class": result.get("reply_class", "unclear"),
                                         "simulated": True, "text_redacted": True})

    def _handle_tool_failure(self, run_id: str, action: ToolAction, result: ToolResult) -> None:
        if action.tool in {ToolName.INSPECT_LEAD, ToolName.VERIFY_EMAIL} and action.lead_id:
            lead = self.store.get_lead(run_id, action.lead_id)
            if lead:
                self.store.save_lead(run_id, action.lead_id, lead["fields"], lead["metadata"],
                                     LeadStatus.DEFERRED.value, "tool_retry_budget_exhausted", lead["draft"], lead["judgments"])

    def _budget_event(self, run_id: str, detail: str) -> None:
        self.store.add_event(run_id, step="budget.stop", summary="Python stopped work at the configured runtime budget.",
                             result={"reason": detail[:180]})

    @staticmethod
    def _safe_action(action: ToolAction) -> dict[str, Any]:
        return {"tool": action.tool.value, "lead_id": action.lead_id, "job_id": action.job_id,
                "detail_level": action.detail_level,
                "search_prompt_character_count": len(action.revised_search_prompt or ""),
                "rationale_character_count": len(action.rationale)}

    @staticmethod
    def _safe_tool_result(result: ToolResult) -> dict[str, Any]:
        safe = {"ok": result.ok, "retryable": result.retryable, "error_code": result.error_code}
        for key in ("status", "job_id", "lead_id", "email_status", "row_count", "simulated", "source"):
            if key in result.data:
                safe[key] = result.data[key]
        if isinstance(result.data.get("leads"), list):
            safe["lead_count"] = len(result.data["leads"])
        if isinstance(result.data.get("feedback"), list):
            safe["feedback_count"] = len(result.data["feedback"])
        return safe

    @staticmethod
    def _public_lead(lead: dict[str, Any]) -> dict[str, Any]:
        fields = lead["fields"]
        safe_fields = {key: value for key, value in fields.items()
                       if key not in {"Email", "LinkedIn URL", "Sales Navigator URL", "Linkedin URL Unique ID"}}
        return {
            "lead_id": lead["lead_id"], "status": lead["status"], "reason": lead["reason"],
            "name": (fields.get("Full Name") or " ".join(x for x in (fields.get("First Name"), fields.get("Last Name")) if x)).strip(),
            "current_job": fields.get("Current Job", ""), "company": fields.get("Company Name", ""),
            "company_industry": fields.get("Company Industry", ""),
            "employee_count": fields.get("Company Employee Exact Count", ""),
            "email_status": fields.get("Email Status", ""), "source": lead["metadata"].get("source", "unknown"),
            "source_timestamp": lead["metadata"].get("source_timestamp"),
            "source_fields": safe_fields, "draft": lead["draft"], "judgments": lead["judgments"],
            "review": lead["review"],
        }


def _probabilities(answers: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, answer in answers.items():
        if answer.get("type") == "noul":
            result[name] = {"probability_yes": answer.get("probability_yes")}
        else:
            result[name] = {"choice": answer.get("choice"), "probabilities": answer.get("probabilities", {}),
                            "confidence": answer.get("confidence")}
    return result


def _parse_iso_datetime(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _feedback_reason_code(reason: str) -> str:
    text = reason.lower()
    if any(term in text for term in ("current job", "headline", "title")):
        return "current_job_evidence"
    if any(term in text for term in ("unsupported", "claim", "evidence", "source")):
        return "claim_grounding"
    if any(term in text for term in ("wrong company", "industry", "size")):
        return "company_fit"
    return "reviewer_correction"
