import json
import os
from pathlib import Path
from typing import Any, Dict


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def get_openai_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it before running ToneBridge."
        )
    return api_key


def pretty_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, indent=2, ensure_ascii=True)
