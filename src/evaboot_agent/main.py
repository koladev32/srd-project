from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

from .config import RESOURCE_ROOT
from .models import GoalRequest, OutcomeRequest, ReviewRequest
from .agents import AgentConfigurationError, AgentOutputError, CostLimitReached
from .judgments import JevConfigurationError
from .runtime import Runtime, RuntimeRequestError


STATIC = RESOURCE_ROOT / "static"
runtime = Runtime()
app = FastAPI(title="Evaboot Agent Runtime", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


class CandidateDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    decision: Literal["approve", "reject"]


def _call(function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except RuntimeRequestError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (AgentConfigurationError, AgentOutputError, JevConfigurationError, CostLimitReached) as exc:
        raise HTTPException(status_code=503, detail="A required model service or configured budget stopped this action; inspect the run trace before retrying.") from exc


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return runtime.health()


@app.get("/api/state")
def state():
    return {"health": runtime.health(), "workspace": runtime.list_runs()}


@app.get("/api/runs")
def runs():
    return runtime.list_runs()


@app.post("/api/runs")
def create_run(request: GoalRequest):
    return _call(runtime.create_run, request)


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    return _call(runtime.get_run, run_id)


@app.post("/api/runs/{run_id}/execute")
def execute(run_id: str):
    return _call(runtime.execute, run_id)


@app.post("/api/runs/{run_id}/resume")
def resume(run_id: str):
    return _call(runtime.resume, run_id)


@app.post("/api/runs/{run_id}/stop")
def stop(run_id: str):
    return _call(runtime.stop, run_id)


@app.post("/api/runs/{run_id}/leads/{lead_id}/review")
def review(run_id: str, lead_id: str, request: ReviewRequest):
    return _call(runtime.review, run_id, lead_id, request)


@app.post("/api/runs/{run_id}/outcomes")
def outcome(run_id: str, request: OutcomeRequest):
    return _call(runtime.ingest_outcome, run_id, request.model_dump(exclude_none=True))


@app.post("/api/runs/{run_id}/improve")
def improve(run_id: str):
    return _call(runtime.improve, run_id)


@app.post("/api/candidates/{candidate_id}/decision")
def candidate_decision(candidate_id: str, request: CandidateDecision):
    return _call(runtime.decide_candidate, candidate_id, request.decision)


@app.post("/api/candidates/{candidate_id}/rollback")
def candidate_rollback(candidate_id: str):
    return _call(runtime.rollback_candidate, candidate_id)


@app.get("/api/runs/{run_id}/replay/{lead_id}")
def replay(run_id: str, lead_id: str):
    return _call(runtime.replay, run_id, lead_id)


@app.get("/api/report.md", response_class=PlainTextResponse)
def report(run_id: str | None = Query(default=None)):
    return runtime.report_markdown(run_id)
