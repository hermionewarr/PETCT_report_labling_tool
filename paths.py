"""Paths for the report -> detect -> label flow. Edit these."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# Reports to score. Minimum columns: findings, conclusion (Findings, Conclusion also accepted).
# Optional: patient_id, source_idx, and a gold label (true_label_name or error_type).
REPORTS_CSV = REPO_ROOT / "data" / "dummy_reports.csv"

# Dummy stand-in for now. Point this at a trained run when real weights are wired in.
MODEL_CHECKPOINT = REPO_ROOT / "checkpoints" / "dummy"

# Written by run_detection.py. This is what the labeling page reads.
PREDICTIONS_CSV = REPO_ROOT / "data" / "predictions.csv"

# Written by label_reports.py as the reviewer clicks.
LABEL_OUTPUT_CSV = REPO_ROOT / "labels" / "fp_labels.csv"

# False: review every report the model called inconsistent.
# True: only gold-consistent rows the model still flagged (needs a gold label column).
ONLY_GOLD_FALSE_POSITIVES = False

HOST = "127.0.0.1"
PORT = 8766

PREDICTION_COLUMNS = [
    "patient_id",
    "source_idx",
    "findings",
    "conclusion",
    "error_type",
    "incidence",
    "true_label",
    "pred_label",
    "true_label_name",
    "pred_label_name",
    "prob_inconsistent",
    "prob_consistent",
    "is_correct",
    "is_tp0",
    "is_fp0",
    "is_fn0",
    "is_tn0",
]
