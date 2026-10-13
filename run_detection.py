#!/usr/bin/env python3
"""Run the inconsistency BERT on a findings/conclusion CSV.

Set MODEL_CHECKPOINT, MODEL_SIZE, and SEQ_LENGTH in paths.py, then:

    python run_detection.py
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer, BertModel

from paths import (
    MODEL_CHECKPOINT,
    MODEL_SIZE,
    PREDICTION_COLUMNS,
    PREDICTIONS_CSV,
    REPORTS_CSV,
    SEQ_LENGTH,
)

CLS_HEAD_FILE = "cls_classifier.pt"
FINDINGS_SPECIAL = "[Findings]"
CONCLUSION_SPECIAL = "[Conclusion]"
SIZE_LAYERS = {"small": 4, "4": 4, "big": 6, "6": 6, "8L": 8, "8": 8, "12L": 12, "12": 12}


def _pick(row: Dict[str, str], *names: str) -> str:
    for name in names:
        if name in row and str(row[name]).strip():
            return str(row[name]).strip()
    return ""


def _gold_consistent(row: Dict[str, str]) -> Optional[bool]:
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


def _check_checkpoint(checkpoint: Path, model_size: str) -> int:
    if not (checkpoint / "config.json").is_file() or not (checkpoint / CLS_HEAD_FILE).is_file():
        raise SystemExit(
            f"MODEL_CHECKPOINT is not a trained BERT directory: {checkpoint}\n"
            "Set it in paths.py to the run's model folder "
            f"(needs config.json and {CLS_HEAD_FILE})."
        )
    expected = SIZE_LAYERS.get(str(model_size).strip())
    if expected is None:
        raise SystemExit(f"MODEL_SIZE must be one of {sorted(set(SIZE_LAYERS))}, got {model_size!r}")
    return expected


def _load(checkpoint: Path, expected_layers: int):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    tokenizer.add_special_tokens({"cls_token": "[CLS]"})
    extra = [tok for tok in (FINDINGS_SPECIAL, CONCLUSION_SPECIAL) if tok not in tokenizer.get_vocab()]
    if extra:
        tokenizer.add_special_tokens({"additional_special_tokens": extra})
    tokenizer._tokenizer.post_processor = TemplateProcessing(
        single="$A",
        pair="$A $B",
        special_tokens=[],
    )
    bert = BertModel.from_pretrained(checkpoint, add_pooling_layer=False)
    if bert.config.num_hidden_layers != expected_layers:
        raise SystemExit(
            f"MODEL_SIZE expects {expected_layers} layers but {checkpoint} "
            f"has {bert.config.num_hidden_layers}. Change MODEL_SIZE in paths.py."
        )
    if bert.get_input_embeddings().weight.shape[0] != len(tokenizer):
        bert.resize_token_embeddings(len(tokenizer))
    head = nn.Linear(bert.config.hidden_size, 1)
    head.load_state_dict(torch.load(checkpoint / CLS_HEAD_FILE, map_location="cpu"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bert.to(device).eval()
    head.to(device).eval()
    return tokenizer, bert, head, device


def _score_one(tokenizer, bert, head, device: str, findings: str, conclusion: str) -> Dict[str, str]:
    text = (
        f"{tokenizer.cls_token} {FINDINGS_SPECIAL} {findings} "
        f"{CONCLUSION_SPECIAL} {conclusion} {tokenizer.sep_token}"
    )
    encoded = tokenizer(
        text,
        truncation=True,
        max_length=SEQ_LENGTH,
        return_tensors="pt",
        add_special_tokens=False,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        cls_repr = bert(**encoded).last_hidden_state[:, 0, :]
        prob_consistent = float(torch.sigmoid(head(cls_repr).squeeze(-1))[0])
    pred_consistent = prob_consistent >= 0.5
    return {
        "pred_label": "1" if pred_consistent else "0",
        "pred_label_name": "consistent" if pred_consistent else "inconsistent",
        "prob_consistent": f"{prob_consistent:.6f}",
        "prob_inconsistent": f"{1.0 - prob_consistent:.6f}",
    }


def score_reports(reports_csv: Path, checkpoint: Path, predictions_csv: Path) -> List[Dict[str, str]]:
    expected_layers = _check_checkpoint(checkpoint, MODEL_SIZE)
    tokenizer, bert, head, device = _load(checkpoint, expected_layers)
    scored: List[Dict[str, str]] = []
    for index, row in enumerate(_read_reports(reports_csv)):
        findings = _pick(row, "findings", "Findings")
        conclusion = _pick(row, "conclusion", "Conclusion")
        if not findings or not conclusion:
            continue
        prediction = _score_one(tokenizer, bert, head, device, findings, conclusion)
        gold_consistent = _gold_consistent(row)
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
            correct = true_label == prediction["pred_label"]
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
    print(f"Checkpoint: {MODEL_CHECKPOINT}  size={MODEL_SIZE}  seq_length={SEQ_LENGTH}")
    print(f"Scored {len(rows)} reports from {REPORTS_CSV}")
    print(f"Flagged {flagged} as inconsistent")
    print(f"Wrote {PREDICTIONS_CSV}")
    print("Next: python label_reports.py")


if __name__ == "__main__":
    main()
