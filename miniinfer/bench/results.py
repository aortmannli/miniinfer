"""Append-only benchmark log. Every number in the write-up should come from here."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import torch

RESULTS_FILE = Path(__file__).resolve().parents[2] / "results" / "runs.jsonl"


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def log_result(name: str, metrics: dict, settings: dict) -> dict:
    record = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "name": name,
        "commit": _git("rev-parse", "--short", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "settings": settings,
        "metrics": metrics,
    }
    RESULTS_FILE.parent.mkdir(exist_ok=True)
    with RESULTS_FILE.open("a") as f:
        f.write(json.dumps(record) + "\n")
    print(json.dumps(record, indent=2))
    return record
