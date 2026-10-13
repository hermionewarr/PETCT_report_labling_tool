"""Stand-in error detector. Replace load_detector when a real checkpoint can be loaded."""
from __future__ import annotations

import json
import re
from pathlib import Path

_LEFT = re.compile(r"\bleft\b", re.IGNORECASE)
_RIGHT = re.compile(r"\bright\b", re.IGNORECASE)
_MILD = re.compile(r"\bmild\b", re.IGNORECASE)
_SEVERE = re.compile(r"\bsevere\b", re.IGNORECASE)


def _pole(text: str, left: re.Pattern[str], right: re.Pattern[str]) -> str:
    """Side or severity of the first matching word. Later mentions are ignored."""
    left_hit = left.search(text)
    right_hit = right.search(text)
    if left_hit and right_hit:
        return "left" if left_hit.start() < right_hit.start() else "right"
    if left_hit:
        return "left"
    if right_hit:
        return "right"
    return ""


class DummyDetector:
    """Keyword mismatch stand-in. Same call shape a BERT scorer would use."""

    def __init__(self, checkpoint: Path) -> None:
        self.checkpoint = checkpoint

    def predict(self, findings: str, conclusion: str) -> dict[str, str]:
        laterality_findings = _pole(findings, _LEFT, _RIGHT)
        laterality_conclusion = _pole(conclusion, _LEFT, _RIGHT)
        severity_findings = _pole(findings, _MILD, _SEVERE)
        severity_conclusion = _pole(conclusion, _MILD, _SEVERE)
        laterality_clash = (
            laterality_findings
            and laterality_conclusion
            and laterality_findings != laterality_conclusion
        )
        severity_clash = (
            severity_findings
            and severity_conclusion
            and severity_findings != severity_conclusion
        )
        inconsistent = bool(laterality_clash or severity_clash)
        if laterality_clash:
            prob_inconsistent = 0.96
        elif severity_clash:
            prob_inconsistent = 0.88
        else:
            prob_inconsistent = 0.12
        prob_consistent = round(1.0 - prob_inconsistent, 4)
        pred_label = "0" if inconsistent else "1"
        return {
            "pred_label": pred_label,
            "pred_label_name": "inconsistent" if inconsistent else "consistent",
            "prob_inconsistent": f"{prob_inconsistent:.4f}",
            "prob_consistent": f"{prob_consistent:.4f}",
        }


def load_detector(checkpoint: Path) -> DummyDetector:
    marker = checkpoint / "dummy_model.json"
    if not marker.is_file():
        raise SystemExit(
            f"{checkpoint} is not the bundled dummy checkpoint "
            f"(missing {marker.name}). Loading a trained BERT error detector "
            "is not wired in yet. Keep MODEL_CHECKPOINT on checkpoints/dummy, "
            "or add that wiring before pointing at real weights."
        )
    payload = json.loads(marker.read_text(encoding="utf-8"))
    if payload.get("type") != "dummy":
        raise SystemExit(f"{marker} is not a dummy detector description.")
    return DummyDetector(checkpoint)
