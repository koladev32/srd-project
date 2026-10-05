from __future__ import annotations

import uvicorn


def main() -> None:
    uvicorn.run("evaboot_agent.main:app", host="127.0.0.1", port=8000)
