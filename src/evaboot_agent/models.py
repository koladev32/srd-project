from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ExecutionMode(StrEnum):
    ASSISTED = "assisted"
    AUTONOMOUS = "autonomous"


class DataSource(StrEnum):
    SIMULATOR = "simulator"
    EVABOOT = "evaboot"


class DeliveryMode(StrEnum):
    SANDBOX = "sandbox"
    LIVE = "live"


class AgentBackend(StrEnum):
    OPENAI_API = "openai_api"
    CODEX_SUBSCRIPTION = "codex_subscription"


class RunStatus(StrEnum):
    PLANNING = "planning"
    PLANNED = "planned"
    RUNNING = "running"
    AWAITING_REVIEW = "awaiting_human_review"
    DEFERRED = "deferred"
    COMPLETE = "complete"
    BUDGET_EXHAUSTED = "budget_exhausted"
    FAILED = "failed"


class LeadStatus(StrEnum):
    NEW = "new"
    NEEDS_VERIFICATION = "needs_verification"
    ELIGIBLE = "eligible_for_judgment"
    NEEDS_RESEARCH = "needs_more_evidence"
    AWAITING_REVIEW = "awaiting_human_review"
    APPROVED = "approved_for_sandbox"
    REJECTED = "rejected"
    SKIPPED = "skipped_insufficient_evidence"
    DEFERRED = "deferred_jev_unavailable"
    SANDBOXED = "sandbox_delivery_recorded"
    SUPPRESSED = "suppressed"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GoalRequest(StrictModel):
    goal: str = Field(min_length=12, max_length=1200)
    agent_backend: AgentBackend = AgentBackend.OPENAI_API
    execution_mode: ExecutionMode = ExecutionMode.ASSISTED
    data_source: DataSource = DataSource.SIMULATOR
    delivery_mode: DeliveryMode = DeliveryMode.SANDBOX
    max_leads: int = Field(default=25, ge=1, le=25)
    max_tool_calls: int = Field(default=20, ge=4, le=60)
    max_search_revisions: int = Field(default=1, ge=0, le=2)
    max_model_cost_usd: float = Field(default=0.25, gt=0, le=10)
    audience_exclusions: list[str] = Field(default_factory=list, max_length=30)
    suppression_list: list[str] = Field(default_factory=list, max_length=200)
    live_evaboot_authorized: bool = False

    @field_validator("goal")
    @classmethod
    def clean_goal(cls, value: str) -> str:
        return value.strip()


class Plan(StrictModel):
    summary: str
    search_prompt: str
    target_criteria: list[str]
    exclusions: list[str]
    evidence_requirements: list[str]
    stopping_conditions: list[str]
    max_leads: int = Field(ge=1, le=25)
    max_tool_calls: int = Field(ge=4, le=60)
    max_search_revisions: int = Field(ge=0, le=2)
    approval_required: bool


class ToolName(StrEnum):
    BUILD_SEARCH = "build_search"
    START_EXPORT = "start_export"
    GET_EXPORT_STATUS = "get_export_status"
    INSPECT_LEAD = "inspect_lead"
    VERIFY_EMAIL = "verify_email"
    READ_PRIOR_FEEDBACK = "read_prior_feedback"
    DONE = "done"


class ToolAction(StrictModel):
    tool: ToolName
    rationale: str = Field(min_length=2, max_length=400)
    lead_id: str | None = None
    job_id: str | None = None
    detail_level: Literal["standard", "deep"] = "standard"
    revised_search_prompt: str | None = Field(default=None, max_length=500)


class EvidenceRef(StrictModel):
    field: str
    excerpt: str = Field(min_length=1, max_length=240)


class PersonalizedClaim(StrictModel):
    text: str = Field(min_length=3, max_length=320)
    evidence: list[EvidenceRef] = Field(min_length=1, max_length=4)


class EmailDraft(StrictModel):
    subject: str = Field(min_length=3, max_length=120)
    claims: list[PersonalizedClaim] = Field(min_length=1, max_length=3)

    def render_body(self, first_name: str) -> str:
        claims = "\n\n".join(claim.text for claim in self.claims)
        return f"Hi {first_name},\n\n{claims}\n\nWould a short overview be useful?"


class ReviewRequest(StrictModel):
    decision: Literal["approve", "reject", "edit"]
    reason: str = Field(default="", max_length=700)
    subject: str | None = Field(default=None, max_length=120)
    body: str | None = Field(default=None, max_length=1600)


class OutcomeRequest(StrictModel):
    external_event_id: str = Field(min_length=4, max_length=150)
    lead_id: str
    event_type: Literal["delivered", "bounce", "reply", "conversion"]
    occurred_at: str | None = None
    text: str | None = Field(default=None, max_length=3000)


class PolicyProposal(StrictModel):
    change_type: Literal["require_current_job_match", "exclude_stale_evidence"]
    rationale: str = Field(min_length=12, max_length=800)
    expected_tradeoff: str = Field(min_length=8, max_length=500)


class ToolResult(StrictModel):
    tool: ToolName
    ok: bool
    summary: str
    data: dict[str, Any] = Field(default_factory=dict)
    retryable: bool = False
    error_code: str | None = None


class APIError(BaseModel):
    detail: str
