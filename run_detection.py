#!/usr/bin/env python3
"""Score findings/conclusion reports and write a predictions CSV for the labeler.

    python run_detection.py
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional

from dummy_model import load_detector
from paths import MODEL_CHECKPOINT, PREDICTION_COLUMNS, PREDICTIONS_CSV, REPORTS_CSV


def _pick(row: Dict[str, str], *names: str) -> str:
    for name in names:
        if name in row and str(row[name]).strip():
            return str(row[name]).strip()
    return ""


def _gold(row: Dict[str, str]) -> Optional[bool]:
    """True if the report is gold-consistent, False if gold-inconsistent, None if unlabeled."""
    name = _pick(row, "true_label_name", "error_type").lower()
    if name:
        return name == "consistent"
    label = _pick(row, "true_label")
    if label in {"1", "1.0"}:
        return True
    if label in {"0", "0.0"}:
        return False
    return None


def _read_reports(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"Reports CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"No rows in {path}")
    sample = rows[0]
    if not any(name in sample for name in ("findings", "Findings")):
        raise SystemExit(f"{path} needs a findings or Findings column")
    if not any(name in sample for name in ("conclusion", "Conclusion")):
        raise SystemExit(f"{path} needs a conclusion or Conclusion column")
    return rows


def score_reports(reports_csv: Path, checkpoint: Path, predictions_csv: Path) -> List[Dict[str, str]]:
    detector = load_detector(checkpoint)
    scored: List[Dict[str, str]] = []
    for index, row in enumerate(_read_reports(reports_csv)):
        findings = _pick(row, "findings", "Findings")
        conclusion = _pick(row, "conclusion", "Conclusion")
        if not findings or not conclusion:
            continue
        prediction = detector.predict(findings, conclusion)
        gold_consistent = _gold(row)
        pred_inconsistent = prediction["pred_label"] == "0"
        out = {name: "" for name in PREDICTION_COLUMNS}
        out.update(
            {
                "patient_id": _pick(row, "patient_id", "PatientID") or f"row-{index}",
                "source_idx": _pick(row, "source_idx") or str(index),
                "findings": findings,
                "conclusion": conclusion,
                "error_type": _pick(row, "error_type"),
                "incidence": _pick(row, "incidence"),
                "true_label_name": _pick(row, "true_label_name"),
                "true_label": _pick(row, "true_label"),
                **prediction,
            }
        )
        if gold_consistent is not None:
            true_label = "1" if gold_consistent else "0"
            pred_label = prediction["pred_label"]
            correct = true_label == pred_label
            out["true_label"] = true_label
            out["true_label_name"] = "consistent" if gold_consistent else "inconsistent"
            if not out["error_type"]:
                out["error_type"] = out["true_label_name"]
            out["is_correct"] = "1" if correct else "0"
            out["is_tp0"] = "1" if pred_inconsistent and not gold_consistent else "0"
            out["is_fp0"] = "1" if pred_inconsistent and gold_consistent else "0"
            out["is_fn0"] = "1" if (not pred_inconsistent) and (not gold_consistent) else "0"
            out["is_tn0"] = "1" if (not pred_inconsistent) and gold_consistent else "0"
        scored.append(out)

    predictions_csv.parent.mkdir(parents=True, exist_ok=True)
    with predictions_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_COLUMNS)
        writer.writeheader()
        writer.writerows(scored)
    return scored


def main() -> None:
    rows = score_reports(REPORTS_CSV, MODEL_CHECKPOINT, PREDICTIONS_CSV)
    flagged = sum(1 for row in rows if row["pred_label_name"] == "inconsistent")
    print(f"Checkpoint: {MODEL_CHECKPOINT}")
    print(f"Scored {len(rows)} reports from {REPORTS_CSV}")
    print(f"Model flagged {flagged} as inconsistent")
    print(f"Wrote {PREDICTIONS_CSV}")
    print("Next: python label_reports.py")


if __name__ == "__main__":
    main()
