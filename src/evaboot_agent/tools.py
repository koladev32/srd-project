from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .evaboot_mcp import EvabootMCPAdapter, EvabootMCPError
from .models import DataSource, GoalRequest, Plan, ToolAction, ToolName, ToolResult
from .normalizer import Lead, load_simulator_leads
from .store import Store, json_dump, json_load


class _ToolArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BuildSearchArgs(_ToolArgs):
    revised_search_prompt: str = Field(min_length=10, max_length=500)


class ExportArgs(_ToolArgs):
    pass


class PollArgs(_ToolArgs):
    job_id: str = Field(min_length=4, max_length=100)


class InspectLeadArgs(_ToolArgs):
    lead_id: str = Field(min_length=3, max_length=100)
    detail_level: str = Field(pattern="^(standard|deep)$")


class VerifyLeadArgs(_ToolArgs):
    lead_id: str = Field(min_length=3, max_length=100)


class FeedbackArgs(_ToolArgs):
    pass


class ToolRegistry:
    def __init__(self, store: Store, *, goal: GoalRequest, plan: Plan,
                 public_csv_path, synthetic_csv_path, evaboot: EvabootMCPAdapter | None = None):
        self.store = store
        self.goal = goal
        self.plan = plan
        self.public_csv_path = public_csv_path
        self.synthetic_csv_path = synthetic_csv_path
        self.evaboot = evaboot

    def call(self, run_id: str, action: ToolAction) -> ToolResult:
        try:
            if action.tool == ToolName.BUILD_SEARCH:
                return self._build_search(run_id, action)
            if action.tool == ToolName.START_EXPORT:
                return self._start_export(run_id)
            if action.tool == ToolName.GET_EXPORT_STATUS:
                return self._get_export_status(run_id, action)
            if action.tool == ToolName.INSPECT_LEAD:
                return self._inspect(run_id, action)
            if action.tool == ToolName.VERIFY_EMAIL:
                return self._verify(run_id, action)
            if action.tool == ToolName.READ_PRIOR_FEEDBACK:
                return self._read_feedback()
            return ToolResult(tool=action.tool, ok=False, summary="This tool is not executable.", error_code="invalid_tool")
        except (ValidationError, EvabootMCPError, ValueError) as exc:
            return ToolResult(tool=action.tool, ok=False, summary="Adapter rejected the request; source data and details were withheld.",
                              error_code=type(exc).__name__)
        except Exception as exc:  # sanitized unexpected adapter failure
            return ToolResult(tool=action.tool, ok=False, summary="Tool adapter failed; details were withheld from the trace.",
                              retryable=True, error_code=type(exc).__name__)

    def _build_search(self, run_id: str, action: ToolAction) -> ToolResult:
        parsed = BuildSearchArgs.model_validate({"revised_search_prompt": action.revised_search_prompt or self.plan.search_prompt})
        if self.goal.data_source == DataSource.EVABOOT:
            if not self.goal.live_evaboot_authorized:
                return ToolResult(tool=action.tool, ok=False, summary="Live Evaboot use was not explicitly authorized.", error_code="live_authorization_required")
            if not self.evaboot:
                return ToolResult(tool=action.tool, ok=False, summary="The live Evaboot adapter is unavailable.", error_code="live_adapter_unavailable")
            response = self.evaboot.build_search(parsed.revised_search_prompt)
            run = self.store.get_run(run_id) or {}
            updated_plan = dict(run.get("plan") or self.plan.model_dump())
            updated_plan["search_prompt"] = parsed.revised_search_prompt
            updated_plan["live_search"] = response
            self.store.update_run(run_id, plan=updated_plan)
            return ToolResult(tool=action.tool, ok=True, summary="Live Evaboot search builder returned a search definition.",
                              data={"search": response, "simulated": False})
        run = self.store.get_run(run_id) or {}
        updated_plan = dict(run.get("plan") or self.plan.model_dump())
        updated_plan["search_prompt"] = parsed.revised_search_prompt
        self.store.update_run(run_id, plan=updated_plan)
        count = len(load_simulator_leads(self.public_csv_path, self.synthetic_csv_path, self.goal.max_leads))
        return ToolResult(tool=action.tool, ok=True, summary=f"Built a simulated Evaboot-shaped search over {count} available sample rows.",
                          data={"search_prompt": parsed.revised_search_prompt, "row_count": count,
                                "simulated": True, "source": "public_sample_plus_synthetic_fixtures"})

    def _start_export(self, run_id: str) -> ToolResult:
        existing = self.store.get_export(run_id)
        if existing:
            return ToolResult(tool=ToolName.START_EXPORT, ok=True,
                              summary="Returned the existing idempotent export job.",
                              data={"job_id": existing["job_id"], "status": existing["status"], "simulated": self.goal.data_source == DataSource.SIMULATOR})
        job_id = f"export-{uuid.uuid4().hex[:12]}"
        if self.goal.data_source == DataSource.EVABOOT:
            if not self.goal.live_evaboot_authorized:
                return ToolResult(tool=ToolName.START_EXPORT, ok=False, summary="Live export was not explicitly authorized.", error_code="live_authorization_required")
            if not self.evaboot:
                return ToolResult(tool=ToolName.START_EXPORT, ok=False, summary="The live Evaboot adapter is unavailable.", error_code="live_adapter_unavailable")
            run = self.store.get_run(run_id) or {}
            search = (run.get("plan") or {}).get("live_search", {})
            export = self.evaboot.start_export(search, self.goal.max_leads)
            job_id = str(export.get("extraction_id") or export.get("job_id") or export.get("id") or job_id)
            self.store.create_or_get_export(run_id, job_id)
            return ToolResult(tool=ToolName.START_EXPORT, ok=True, summary="Started a quota-checked Evaboot MCP export.",
                              data={"job_id": job_id, "status": "running", "simulated": False})
        job = self.store.create_or_get_export(run_id, job_id)
        return ToolResult(tool=ToolName.START_EXPORT, ok=True, summary="Started asynchronous simulated export.",
                          data={"job_id": job["job_id"], "status": job["status"], "simulated": True})

    def _get_export_status(self, run_id: str, action: ToolAction) -> ToolResult:
        parsed = PollArgs.model_validate({"job_id": action.job_id or ""})
        export = self.store.get_export(run_id)
        if not export or export["job_id"] != parsed.job_id:
            return ToolResult(tool=action.tool, ok=False, summary="Export job is not associated with this run.", error_code="unknown_export")
        if self.goal.data_source == DataSource.EVABOOT:
            assert self.evaboot is not None
            result = self.evaboot.get_export(parsed.job_id)
            payload = result if isinstance(result, dict) else {"text": result}
            status = str(payload.get("status", "running")).lower()
            leads = _extract_leads(payload)
            count = export["poll_count"] + 1
            normalized = [Lead(lead_id=f"live-{index + 1}", fields=_string_fields(row),
                               metadata={"source": "evaboot_mcp_live", "source_timestamp": None}).__dict__
                         for index, row in enumerate(leads[:self.goal.max_leads])]
            if status in {"complete", "completed", "done", "success"}:
                status = "complete"
                self.store.update_export(run_id, status=status, poll_count=count, leads=normalized, fail_once=0)
                return ToolResult(tool=action.tool, ok=True, summary=f"Live export completed with {len(normalized)} rows.",
                                  data={"job_id": parsed.job_id, "status": status, "leads": normalized, "simulated": False})
            self.store.update_export(run_id, status=status, poll_count=count, fail_once=0)
            return ToolResult(tool=action.tool, ok=True, summary=f"Live export is {status}.",
                              data={"job_id": parsed.job_id, "status": status, "simulated": False})

        if export["fail_once"]:
            self.store.update_export(run_id, status="queued", poll_count=export["poll_count"] + 1, fail_once=0)
            return ToolResult(tool=action.tool, ok=False, summary="Simulated transient status timeout; the export job remains intact.",
                              data={"job_id": parsed.job_id, "status": "unknown"}, retryable=True, error_code="temporary_timeout")
        if export["poll_count"] < 2:
            self.store.update_export(run_id, status="running", poll_count=export["poll_count"] + 1, fail_once=0)
            return ToolResult(tool=action.tool, ok=True, summary="Export is still running; poll again within the bounded retry budget.",
                              data={"job_id": parsed.job_id, "status": "running", "simulated": True})
        stored = export.get("leads")
        if stored is None:
            rows = load_simulator_leads(self.public_csv_path, self.synthetic_csv_path, self.goal.max_leads)
            stored = [{"lead_id": lead.lead_id, "fields": lead.fields, "metadata": lead.metadata} for lead in rows]
            self.store.update_export(run_id, status="complete", poll_count=export["poll_count"] + 1, leads=stored, fail_once=0)
        return ToolResult(tool=action.tool, ok=True, summary=f"Simulated export completed with {len(stored)} rows.",
                          data={"job_id": parsed.job_id, "status": "complete", "leads": stored,
                                "simulated": True, "source": "public_sample_plus_synthetic_fixtures"})

    def _inspect(self, run_id: str, action: ToolAction) -> ToolResult:
        parsed = InspectLeadArgs.model_validate({"lead_id": action.lead_id or "", "detail_level": action.detail_level})
        lead = self.store.get_lead(run_id, parsed.lead_id)
        if not lead:
            return ToolResult(tool=action.tool, ok=False, summary="Lead is not part of this export.", error_code="unknown_lead")
        allowed = {
            "Current Job", "Profile Headline", "Profile Summary", "Job Description", "Company Name",
            "Company Industry", "Company Employee Exact Count", "Company Employee Range", "Company Description",
            "Company Specialities", "Company Location", "Location", "Matches Filters", "No Match Reasons",
            "Email Status", "Source Updated At",
        }
        evidence = {key: value for key, value in lead["fields"].items()
                    if key in allowed and value and (parsed.detail_level == "deep" or key in {
                        "Current Job", "Profile Headline", "Company Name", "Company Industry",
                        "Company Employee Exact Count", "Company Location", "Company Description", "Profile Summary",
                    })}
        return ToolResult(tool=action.tool, ok=True,
                          summary=f"Inspected {parsed.detail_level} evidence for {parsed.lead_id} across {len(evidence)} source fields.",
                          data={"lead_id": parsed.lead_id, "source_id": parsed.lead_id,
                                "source_timestamp": lead["metadata"].get("source_timestamp"),
                                "source_fields": evidence, "detail_level": parsed.detail_level,
                                "simulated": self.goal.data_source == DataSource.SIMULATOR})

    def _verify(self, run_id: str, action: ToolAction) -> ToolResult:
        parsed = VerifyLeadArgs.model_validate({"lead_id": action.lead_id or ""})
        lead = self.store.get_lead(run_id, parsed.lead_id)
        if not lead:
            return ToolResult(tool=action.tool, ok=False, summary="Lead is not part of this export.", error_code="unknown_lead")
        if self.goal.data_source == DataSource.EVABOOT:
            assert self.evaboot is not None
            result = self.evaboot.verify_email(lead["fields"].get("Email", ""))
            status = _extract_email_status(result)
        else:
            status = lead["metadata"].get("verification_result") or lead["fields"].get("Email Status") or "unknown"
        metadata = dict(lead["metadata"])
        metadata["verified_email_status"] = str(status).lower()
        fields = dict(lead["fields"])
        fields["Email Status"] = str(status).lower()
        self.store.save_lead(run_id, parsed.lead_id, fields, metadata,
                             "eligible_for_judgment" if str(status).lower() == "safe" else "rejected",
                             None if str(status).lower() == "safe" else "email_not_safe")
        return ToolResult(tool=action.tool, ok=True, summary=f"Email verification result: {str(status).lower()}.",
                          data={"lead_id": parsed.lead_id, "email_status": str(status).lower(),
                                "simulated": self.goal.data_source == DataSource.SIMULATOR})

    def _read_feedback(self) -> ToolResult:
        feedback = self.store.list_feedback(40)
        summary = self.store.feedback_summary()
        return ToolResult(tool=ToolName.READ_PRIOR_FEEDBACK, ok=True,
                          summary=f"Loaded {len(feedback)} reviewer corrections from persistent memory.",
                          data={"feedback": feedback, "reason_counts": summary})


def _extract_leads(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        for key in ("leads", "results", "data", "export"):
            if isinstance(value.get(key), list):
                return [item for item in value[key] if isinstance(item, dict)]
        for child in value.values():
            result = _extract_leads(child)
            if result:
                return result
    elif isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _string_fields(row: dict[str, Any]) -> dict[str, str]:
    return {str(key): str(value) for key, value in row.items() if value is not None}


def _extract_email_status(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("status", "email_status", "result"):
            if isinstance(value.get(key), str):
                return value[key]
        for child in value.values():
            if isinstance(child, dict):
                result = _extract_email_status(child)
                if result != "unknown":
                    return result
    return "unknown"
