from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from evaboot_agent.agents import OpenAIAgent
from evaboot_agent.agents import CostLimitReached
from evaboot_agent.config import Settings, has_openai_key
from evaboot_agent.main import app
from evaboot_agent.models import GoalRequest, Plan, ToolName
from evaboot_agent.store import Store


def _settings(path):
    return Settings(database_path=path, openai_model="gpt-6-luna", openai_input_rate=.10,
                    openai_output_rate=.50, typesafe_model="jev-latest",
                    typesafe_timeout_seconds=12, public_csv_path=path.parent/"none.csv")


def _response(output, input_tokens=30, output_tokens=12):
    return SimpleNamespace(output=output, usage=SimpleNamespace(input_tokens=input_tokens,output_tokens=output_tokens))


def test_tool_selector_retries_malformed_function_output_once(tmp_path):
    store=Store(tmp_path/"runtime.sqlite3")
    store.create_run("run-a",{"goal":"Finance leaders at European fintech companies"},"policy-v1")
    malformed=SimpleNamespace(type="function_call",name="build_search",arguments='{"rationale":"too short"}')
    valid=SimpleNamespace(type="function_call",name="build_search",arguments='{"rationale":"Build the scoped query.","revised_search_prompt":"European finance leaders at fintech companies"}')
    responses=SimpleNamespace()
    def create(**kwargs):
        count=getattr(responses,"count",0);responses.count=count+1
        return _response([malformed] if count==0 else [valid])
    responses.create=create
    agent=OpenAIAgent(_settings(tmp_path/"runtime.sqlite3"),store)
    agent._client=SimpleNamespace(responses=responses)
    request=GoalRequest(goal="Finance leaders at European fintech companies with 200 employees")
    plan=Plan(summary="A bounded finance leader plan.",search_prompt="European finance leaders at fintech companies",
              target_criteria=["finance"],exclusions=[],evidence_requirements=["current role"],stopping_conditions=["cap"],
              max_leads=5,max_tool_calls=20,max_search_revisions=1,approval_required=True)
    result=agent.next_action(run_id="run-a",request=request,plan=plan,state={},allowed_tools=[ToolName.BUILD_SEARCH])
    assert result.value.tool == ToolName.BUILD_SEARCH
    assert result.attempts == 2
    events=store.list_events("run-a")
    assert len(events)==2 and events[0]["error_status"]=="ValidationError" and events[1]["error_status"] is None


def test_structured_output_retries_once_and_logs_both_attempts(tmp_path):
    store=Store(tmp_path/"runtime.sqlite3")
    store.create_run("run-a",{"goal":"Finance leaders at European fintech companies"},"policy-v1")
    settings=_settings(tmp_path/"runtime.sqlite3")
    agent=OpenAIAgent(settings,store)
    from evaboot_agent.agents import PlanDraft
    valid_plan=PlanDraft(summary="A bounded test plan.",search_prompt="Finance leaders in European fintech",
                         target_criteria=["finance leaders"],evidence_requirements=["Current Job"],
                         stopping_conditions=["Lead cap reached"])
    valid=SimpleNamespace(output_parsed=valid_plan,usage=SimpleNamespace(input_tokens=20,output_tokens=10))
    malformed=SimpleNamespace(output_parsed=None,usage=SimpleNamespace(input_tokens=18,output_tokens=4))
    class Responses:
        count=0
        def parse(self,**kwargs):
            self.count+=1
            return malformed if self.count==1 else SimpleNamespace(output_parsed=valid_plan,usage=valid.usage)
    agent._client=SimpleNamespace(responses=Responses())
    result=agent._parse(PlanDraft,step="test.parse",instructions="Return typed output",payload={"goal":"a test"},run_id="run-a",cost_ceiling=.25)
    assert result.attempts==2
    events=store.list_events("run-a")
    assert [event["error_status"] for event in events]==["AgentOutputError",None]
    assert store.get_run("run-a")["counters"]["input_tokens"]==38


def test_model_cost_ceiling_prevents_request_before_call(tmp_path):
    store=Store(tmp_path/"runtime.sqlite3")
    store.create_run("run-a",{"goal":"Finance leaders at European fintech companies"},"policy-v1")
    agent=OpenAIAgent(_settings(tmp_path/"runtime.sqlite3"),store)
    class Responses:
        called=False
        def create(self,**kwargs):
            self.called=True
            return _response([])
    responses=Responses()
    agent._client=SimpleNamespace(responses=responses)
    request=GoalRequest(goal="Finance leaders at European fintech companies with 200 employees").model_copy(update={"max_model_cost_usd":0.00000001})
    plan=Plan(summary="A bounded finance leader plan.",search_prompt="European finance leaders",
              target_criteria=["finance"],exclusions=[],evidence_requirements=["current role"],stopping_conditions=["cap"],
              max_leads=5,max_tool_calls=20,max_search_revisions=1,approval_required=True)
    import pytest
    with pytest.raises(CostLimitReached):
        agent.next_action(run_id="run-a",request=request,plan=plan,state={},allowed_tools=[ToolName.BUILD_SEARCH])
    assert not responses.called


def test_app_serves_local_dashboard_and_health_without_exposing_key_values():
    client=TestClient(app)
    page=client.get("/")
    assert page.status_code==200 and "Build a qualified pipeline" in page.text
    assert "Python owns the guardrails" not in page.text
    assert "Sandbox only" not in page.text
    health=client.get("/api/health")
    assert health.status_code==200
    body=health.json()
    assert "openai_configured" in body and "OPENAI_API_KEY" not in str(body)


def test_openai_key_alias_is_detected_without_requiring_legacy_name(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_KEY", "test-credential")
    assert has_openai_key()


def test_codex_subscription_output_uses_isolated_luna_cli_and_tracks_estimates(tmp_path, monkeypatch):
    import json
    import subprocess
    from pathlib import Path
    from pydantic import BaseModel, ConfigDict, Field
    from evaboot_agent import agents as agents_module
    from evaboot_agent.agents import CodexSubscriptionAgent

    class SmallOutput(BaseModel):
        model_config = ConfigDict(extra="forbid")
        summary: str = Field(min_length=8, max_length=100)
        note: str | None = None

    store = Store(tmp_path / "subscription.sqlite3")
    store.create_run("run-sub", {"goal": "A synthetic subscription smoke test"}, "policy-v1")
    agent = CodexSubscriptionAgent(_settings(tmp_path / "subscription.sqlite3"), store)
    captured = {}

    def fake_run(command, **kwargs):
        captured.update(command=command, cwd=kwargs["cwd"], env=kwargs["env"])
        schema_path = Path(command[command.index("--output-schema") + 1])
        schema_json = json.loads(schema_path.read_text())
        captured["strict_schema"] = schema_json.get("required") == list(schema_json.get("properties", {}))
        result_path = Path(command[command.index("--output-last-message") + 1])
        result_path.write_text(json.dumps({"summary": "Synthetic output passed schema validation.", "note": None}))
        return subprocess.CompletedProcess(command, 0, '{"type":"turn.completed"}\n', "")

    monkeypatch.setattr(agents_module.shutil, "which", lambda _: "/usr/bin/codex")
    monkeypatch.setattr(agents_module.subprocess, "run", fake_run)
    monkeypatch.setenv("OPENAI_KEY", "do-not-forward")
    monkeypatch.setenv("OPENAI_API_KEY", "do-not-forward-either")
    monkeypatch.setenv("TYPESAFE_API_KEY", "also-do-not-forward")
    monkeypatch.setenv("EVABOOT_API_KEY", "never-forward-this")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "never-forward-cloud-secrets")

    result = agent._parse(SmallOutput, step="test.subscription", instructions="Return a short summary.",
                          payload={"source": "synthetic"}, run_id="run-sub", cost_ceiling=.25)

    assert result.value.summary.startswith("Synthetic output")
    assert result.model == "gpt-6-luna" and result.input_tokens > 0 and result.output_tokens > 0
    assert result.cost_usd > 0 and result.attempts == 1
    assert "--model" in captured["command"] and "gpt-6-luna" in captured["command"]
    assert captured["command"].index("--ask-for-approval") < captured["command"].index("exec")
    assert "--ignore-user-config" in captured["command"] and "--sandbox" in captured["command"]
    assert captured["command"][captured["command"].index("--sandbox") + 1] == "read-only"
    assert captured["command"][captured["command"].index("--ephemeral"):]  # ephemeral session flag is present
    assert captured["strict_schema"] is True
    assert "OPENAI_KEY" not in captured["env"] and "OPENAI_API_KEY" not in captured["env"]
    assert "TYPESAFE_API_KEY" not in captured["env"] and "EVABOOT_API_KEY" not in captured["env"]
    assert "AWS_SECRET_ACCESS_KEY" not in captured["env"]
    assert 'shell_environment_policy.inherit="none"' in captured["command"]
    assert str(tmp_path) not in captured["cwd"]
    assert agent._client is None


def test_codex_subscription_selector_only_returns_python_allowed_action(tmp_path, monkeypatch):
    from evaboot_agent.agents import CodexSubscriptionAgent

    store = Store(tmp_path / "subscription-selector.sqlite3")
    store.create_run("run-sub", {"goal": "Finance leaders at European fintech companies"}, "policy-v1")
    agent = CodexSubscriptionAgent(_settings(tmp_path / "subscription-selector.sqlite3"), store)

    def fake_invoke(schema, prompt):
        return schema.model_validate({
            "tool": "build_search",
            "rationale": "The workflow has no search definition yet.",
            "revised_search_prompt": "European fintech finance leaders",
        }), 300, 40

    monkeypatch.setattr(agent, "_invoke_codex", fake_invoke)
    request = GoalRequest(goal="Finance leaders at European fintech companies with 200 employees",
                          max_model_cost_usd=.25)
    plan = Plan(summary="A bounded finance leader plan.", search_prompt="European finance leaders",
                target_criteria=["finance leaders"], exclusions=[], evidence_requirements=["current role"],
                stopping_conditions=["lead cap reached"], max_leads=5, max_tool_calls=8,
                max_search_revisions=1, approval_required=True)
    result = agent.next_action(run_id="run-sub", request=request, plan=plan,
                               state={"search_exists": False}, allowed_tools=[ToolName.BUILD_SEARCH])

    assert result.value.tool == ToolName.BUILD_SEARCH
    assert result.value.revised_search_prompt == "European fintech finance leaders"
    event = store.list_events("run-sub")[-1]
    assert event["step"] == "executor.choose_tool" and event["model"] == "gpt-6-luna"


def test_model_configuration_rejects_non_luna(monkeypatch):
    import pytest
    from evaboot_agent.config import Settings

    monkeypatch.setenv("OPENAI_MODEL", "gpt-6.1-sol")
    with pytest.raises(ValueError, match="pinned to gpt-6-luna"):
        Settings.from_env()


def test_runtime_routes_subscription_mode_without_openai_api_key(tmp_path, monkeypatch):
    from evaboot_agent import runtime as runtime_module
    from evaboot_agent.agents import CodexSubscriptionAgent
    from evaboot_agent.runtime import Runtime

    monkeypatch.setattr(runtime_module, "has_openai_key", lambda: False)
    monkeypatch.setattr(runtime_module, "has_codex_chatgpt_login", lambda: True)
    monkeypatch.setattr(runtime_module, "has_typesafe_key", lambda: True)
    settings = _settings(tmp_path / "runtime.sqlite3")
    runtime = Runtime(settings, Store(settings.database_path))
    request = GoalRequest(goal="Finance leaders at European fintech companies with 200 employees",
                          agent_backend="codex_subscription")

    runtime._ensure_agent_backend(request)
    assert isinstance(runtime._agent_for_request(request), CodexSubscriptionAgent)
    health = runtime.health()
    assert health["codex_subscription_available"] is True and health["interactive_ready"] is True
