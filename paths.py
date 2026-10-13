"""Edit these, then run run_detection.py and label_reports.py."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# Findings + conclusion CSV. Findings/Conclusion column names are also accepted.
REPORTS_CSV = REPO_ROOT / "data" / "dummy_reports.csv"

# Directory of a trained inconsistency BERT (config.json + cls_classifier.pt).
# The folder below is only a placeholder. Replace it with the real checkpoint.
MODEL_CHECKPOINT = REPO_ROOT / "checkpoints" / "dummy"
# 4/small, 6/big, 8/8L, 12/12L. Must match the checkpoint.
MODEL_SIZE = "big"
# Token length the checkpoint was trained with.
SEQ_LENGTH = 600

# Written by run_detection.py. Read by label_reports.py.
PREDICTIONS_CSV = REPO_ROOT / "data" / "predictions.csv"
# Written as the reviewer clicks.
LABEL_OUTPUT_CSV = REPO_ROOT / "labels" / "fp_labels.csv"

# False: review every report the model called inconsistent.
# True: only gold-consistent rows the model still flagged.
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
