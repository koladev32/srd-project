"""Standalone smoke check used after installing the built wheel in CI."""

from evaboot_agent.config import RESOURCE_ROOT
from evaboot_agent.main import app
from evaboot_agent.store import Store


required = [
    RESOURCE_ROOT / "static" / "index.html",
    RESOURCE_ROOT / "static" / "shadcn.css",
    RESOURCE_ROOT / "config" / "policy_v1.json",
    RESOURCE_ROOT / "config" / "evaluation_acceptance.json",
    RESOURCE_ROOT / "data" / "synthetic_fixtures.csv",
    RESOURCE_ROOT / "data" / "evaluation_cases.json",
]

assert all(path.is_file() for path in required), required
assert app.title == "Evaboot Agent Runtime"
assert Store.__name__ == "Store"
