"""
Retrain the diabetes-risk model and register it as a brand-new, timestamped
version — never overwrites a previous one.

Usage:
    python src/retrain.py
    python src/retrain.py --extra-rows data/new_patients.csv --note "August intake batch"

This is the same fitting routine as train.py (imported, not duplicated); the
only difference is intent: `train.py` is "build the first model", `retrain.py`
is "the world changed a bit, learn a new version on top of it". Both land in
the same `models/` registry so the API and UI treat every version — bootstrap
or retrain — identically.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))

from train import DEFAULT_DATA_PATH, train_and_save  # noqa: E402


def retrain(note: str = "retrain.py run", extra_rows_path: str | None = None) -> dict:
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    return train_and_save(
        now=now,
        data_path=DEFAULT_DATA_PATH,
        note=note,
        extra_rows_path=extra_rows_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrain and register a new model version.")
    parser.add_argument("--extra-rows", type=str, default=None, help="Optional CSV of new rows to append before retraining.")
    parser.add_argument("--note", type=str, default="manual retrain.py run")
    args = parser.parse_args()

    result = retrain(note=args.note, extra_rows_path=args.extra_rows)
    print(json.dumps(result, indent=2))
