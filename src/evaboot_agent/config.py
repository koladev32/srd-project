from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PACKAGE_ROOT = Path(__file__).resolve().parent
RESOURCE_ROOT = PACKAGE_ROOT / "resources"
# Runtime state belongs to the operator's working directory, never inside an
# installed wheel. ROOT remains as a compatibility alias for CLI callers.
ROOT = Path.cwd()
load_dotenv(ROOT / ".env")


def _rate(name: str) -> float | None:
    value = os.getenv(name, "").strip()
    if not value:
        return None
    parsed = float(value)
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


@dataclass(frozen=True)
class Settings:
    database_path: Path
    openai_model: str
    openai_input_rate: float
    openai_output_rate: float
    typesafe_model: str
    typesafe_timeout_seconds: float
    public_csv_path: Path
    codex_timeout_seconds: float = 75

    @classmethod
    def from_env(cls) -> "Settings":
        configured_model = os.getenv("OPENAI_MODEL", "gpt-6-luna").strip()
        if configured_model and configured_model != "gpt-6-luna":
            raise ValueError("This runtime is pinned to gpt-6-luna for both agent backends.")
        model = "gpt-6-luna"
        input_rate = _rate("OPENAI_INPUT_USD_PER_MILLION")
        output_rate = _rate("OPENAI_OUTPUT_USD_PER_MILLION")
        if model == "gpt-6-luna":
            input_rate = input_rate if input_rate is not None else 0.10
            output_rate = output_rate if output_rate is not None else 0.50
        if input_rate is None or output_rate is None:
            raise ValueError(
                "Set OPENAI_INPUT_USD_PER_MILLION and OPENAI_OUTPUT_USD_PER_MILLION "
                "for a custom OPENAI_MODEL so the run cost ceiling can be enforced."
            )

        legacy_database = ROOT / "data" / "runtime.sqlite3"
        default_database = legacy_database if legacy_database.exists() else ROOT / ".evaboot-agent" / "runtime.sqlite3"
        database = Path(os.getenv("DATABASE_PATH", str(default_database)))
        configured_csv = os.getenv("PUBLIC_CSV_PATH", "").strip()
        default_csv = ROOT / ".context" / "evaboot-sample-export.csv"
        if configured_csv:
            csv_path = Path(configured_csv)
        elif default_csv.exists():
            csv_path = default_csv
        else:
            prior_samples = sorted((ROOT / ".context").glob("evaboot-*-export.csv"))
            csv_path = prior_samples[0] if prior_samples else default_csv
        return cls(
            database_path=database,
            openai_model=model,
            openai_input_rate=input_rate,
            openai_output_rate=output_rate,
            typesafe_model=os.getenv("TYPESAFE_MODEL", "jev-latest").strip() or "jev-latest",
            typesafe_timeout_seconds=float(os.getenv("TYPESAFE_TIMEOUT_SECONDS", "12")),
            public_csv_path=csv_path,
            codex_timeout_seconds=float(os.getenv("CODEX_TIMEOUT_SECONDS", "75")),
        )


def get_openai_key() -> str:
    """Return the configured OpenAI credential without exposing it in status output."""
    return os.getenv("OPENAI_KEY", "").strip() or os.getenv("OPENAI_API_KEY", "").strip()


def has_openai_key() -> bool:
    return bool(get_openai_key())


def has_codex_chatgpt_login() -> bool:
    """Check Codex CLI's login mode without exposing its captured output or credentials."""
    executable = shutil.which("codex")
    if not executable:
        return False
    try:
        result = subprocess.run(
            [executable, "login", "status"], capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    status = f"{result.stdout}\n{result.stderr}".lower()
    return result.returncode == 0 and "chatgpt" in status


def has_typesafe_key() -> bool:
    return bool(os.getenv("TYPESAFE_API_KEY", "").strip())


def has_evaboot_key() -> bool:
    return bool(os.getenv("EVABOOT_API_KEY", "").strip())
