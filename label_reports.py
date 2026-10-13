#!/usr/bin/env python3
"""Label BERT false-positive reports as error, no error, or unclear.

Edit the three paths below, then:

    python label_reports.py

Open the printed URL. Keys: 1 = error, 0 = no error, x = unclear, u = undo.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import re
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Edit these paths.
# ---------------------------------------------------------------------------
DATA_CSV = REPO_ROOT / "data" / "dummy_bert6_predictions.csv"
MODEL_CHECKPOINT = REPO_ROOT / "checkpoints" / "dummy"
LABEL_OUTPUT_CSV = REPO_ROOT / "labels" / "fp_labels.csv"
# True: only rows the model called inconsistent when the gold label is consistent.
ONLY_FALSE_POSITIVES = True
HOST = "127.0.0.1"
PORT = 8766
# ---------------------------------------------------------------------------

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
LABEL_COLUMNS = PREDICTION_COLUMNS + [
    "human_label",
    "review_note",
    "reviewed_at",
    "model_checkpoint",
]
LABELS = ("error", "no_error", "unclear")
KEY_TO_LABEL = {"1": "error", "0": "no_error", "x": "unclear"}
_LEFT = re.compile(r"\bleft\b", re.IGNORECASE)
_RIGHT = re.compile(r"\bright\b", re.IGNORECASE)


def _row_key(row: Dict[str, Any]) -> Tuple[str, str]:
    return (str(row.get("patient_id", "")).strip(), str(row.get("source_idx", "")).strip())


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in LABEL_COLUMNS})


def _is_false_positive(row: Dict[str, str]) -> bool:
    if str(row.get("is_fp0", "")).strip() in {"1", "1.0", "true", "True"}:
        return True
    true_name = str(row.get("true_label_name", "")).strip().lower()
    pred_name = str(row.get("pred_label_name", "")).strip().lower()
    if true_name == "consistent" and pred_name == "inconsistent":
        return True
    return str(row.get("true_label", "")).strip() == "1" and str(row.get("pred_label", "")).strip() == "0"


def highlight_laterality(text: str) -> str:
    escaped = html.escape(text or "")
    escaped = _LEFT.sub(r'<mark class="hl-left">\g<0></mark>', escaped)
    escaped = _RIGHT.sub(r'<mark class="hl-right">\g<0></mark>', escaped)
    return escaped


def load_queue(data_csv: Path, labels_csv: Path) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    if not data_csv.is_file():
        raise SystemExit(f"Predictions CSV not found: {data_csv}")
    rows = _read_csv(data_csv)
    missing = [name for name in ("patient_id", "source_idx", "findings", "conclusion") if name not in (rows[0] if rows else {})]
    if rows and missing:
        raise SystemExit(f"{data_csv} is missing columns: {', '.join(missing)}")
    if ONLY_FALSE_POSITIVES:
        rows = [row for row in rows if _is_false_positive(row)]
    saved = _read_csv(labels_csv) if labels_csv.is_file() else []
    saved_by_key = {_row_key(row): row for row in saved}
    queue: List[Dict[str, str]] = []
    for index, row in enumerate(rows):
        item = {name: row.get(name, "") for name in PREDICTION_COLUMNS}
        previous = saved_by_key.get(_row_key(item), {})
        item["human_label"] = previous.get("human_label", "")
        item["review_note"] = previous.get("review_note", "")
        item["reviewed_at"] = previous.get("reviewed_at", "")
        item["model_checkpoint"] = str(MODEL_CHECKPOINT)
        item["queue_index"] = str(index)
        queue.append(item)
    labeled = [row for row in queue if row.get("human_label")]
    return queue, labeled


class ReviewSession:
    def __init__(self, data_csv: Path, labels_csv: Path, checkpoint: Path) -> None:
        self.data_csv = data_csv
        self.labels_csv = labels_csv
        self.checkpoint = checkpoint
        self.queue, self.labeled = load_queue(data_csv, labels_csv)
        self.undo_stack: List[Dict[str, str]] = []

    def _unlabeled_index(self) -> Optional[int]:
        for index, row in enumerate(self.queue):
            if not str(row.get("human_label", "")).strip():
                return index
        return None

    def state(self) -> Dict[str, Any]:
        index = self._unlabeled_index()
        current = None
        if index is not None:
            row = self.queue[index]
            current = {
                **row,
                "findings_html": highlight_laterality(row.get("findings", "")),
                "conclusion_html": highlight_laterality(row.get("conclusion", "")),
            }
        reviewed = sum(1 for row in self.queue if str(row.get("human_label", "")).strip())
        return {
            "total": len(self.queue),
            "reviewed": reviewed,
            "remaining": len(self.queue) - reviewed,
            "undo_available": bool(self.undo_stack),
            "current": current,
            "data_csv": str(self.data_csv),
            "label_output_csv": str(self.labels_csv),
            "model_checkpoint": str(self.checkpoint),
            "checkpoint_exists": self.checkpoint.exists(),
        }

    def _persist(self) -> None:
        _write_csv(self.labels_csv, [row for row in self.queue if str(row.get("human_label", "")).strip()])

    def label(self, human_label: str, note: str = "") -> Dict[str, Any]:
        raw = str(human_label).strip().lower()
        human_label = KEY_TO_LABEL.get(raw, raw)
        if human_label not in LABELS:
            raise ValueError("human_label must be error, no_error, or unclear")
        index = self._unlabeled_index()
        if index is None:
            raise ValueError("Nothing left to label")
        row = self.queue[index]
        self.undo_stack.append(dict(row))
        row["human_label"] = human_label
        row["review_note"] = note.strip()
        row["reviewed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        row["model_checkpoint"] = str(self.checkpoint)
        self._persist()
        return self.state()

    def undo(self) -> Dict[str, Any]:
        if not self.undo_stack:
            raise ValueError("Nothing to undo")
        previous = self.undo_stack.pop()
        key = _row_key(previous)
        for row in self.queue:
            if _row_key(row) == key:
                row["human_label"] = ""
                row["review_note"] = previous.get("review_note", "")
                row["reviewed_at"] = ""
                break
        self._persist()
        return self.state()


PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>PET/CT report labeling</title>
<style>
  :root {
    --bg: #10161d; --panel: #1a2430; --text: #e7eef6; --muted: #93a4b8;
    --accent: #7eb6e0; --error: #c45c5c;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--text); font: 16px/1.45 "Segoe UI", sans-serif; }
  .wrap { max-width: 1100px; margin: 0 auto; padding: 1.25rem 1.25rem 3rem; }
  .top, .paths { display: flex; justify-content: space-between; gap: 0.75rem; flex-wrap: wrap; }
  .progress strong { color: var(--text); }
  .progress, .meta, .paths { color: var(--muted); font-size: 0.9rem; }
  .paths { margin-top: 0.35rem; }
  .paths code { color: var(--text); }
  .banner {
    margin: 0.9rem 0 1rem; padding: 0.7rem 1rem; border-radius: 8px; font-weight: 600;
    background: rgba(126, 182, 224, 0.12); border-left: 4px solid var(--accent);
    display: flex; justify-content: space-between; gap: 1rem; flex-wrap: wrap;
  }
  .cols { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
  @media (max-width: 900px) { .cols { grid-template-columns: 1fr; } }
  h2 { margin: 0 0 0.45rem; font-size: 0.85rem; letter-spacing: 0.04em; text-transform: uppercase; color: var(--muted); }
  .text { background: var(--panel); border-radius: 10px; padding: 1rem 1.1rem; min-height: 10rem; line-height: 1.55; }
  mark.hl-left { background: #5b8def66; color: #cfe0ff; padding: 0 0.12em; border-radius: 3px; }
  mark.hl-right { background: #e06c7566; color: #ffd0d4; padding: 0 0.12em; border-radius: 3px; }
  textarea {
    width: 100%; margin-top: 0.9rem; background: var(--panel); color: var(--text);
    border: 1px solid #2e3c4e; border-radius: 8px; padding: 0.55rem 0.75rem; font: inherit; min-height: 2.6rem;
  }
  .actions { display: flex; gap: 0.75rem; margin-top: 1rem; flex-wrap: wrap; }
  button.act { font-size: 1rem; padding: 0.65rem 1.2rem; border: none; border-radius: 8px; cursor: pointer; font-weight: 600; }
  button.k1 { background: var(--error); color: #fff; }
  button.k0 { background: #4a5568; color: #fff; }
  button.kx { background: #6b5a2a; color: #fff; }
  button.ku { background: transparent; color: var(--muted); border: 1px solid #444; }
  button:disabled { opacity: 0.35; cursor: not-allowed; }
  .hint { margin-top: 0.85rem; font-size: 0.8rem; color: var(--muted); }
  kbd { background: #2a3544; padding: 0.12rem 0.4rem; border-radius: 4px; }
  .done { text-align: center; padding: 3rem 1rem; font-size: 1.15rem; }
  .flash {
    position: fixed; bottom: 1rem; right: 1rem; background: #2a3544;
    padding: 0.5rem 0.85rem; border-radius: 6px; opacity: 0; transition: opacity 0.2s;
  }
  .flash.show { opacity: 1; }
  #focus-trap { position: absolute; width: 0; height: 0; opacity: 0; }
</style>
</head>
<body>
<input id="focus-trap" tabindex="0" aria-hidden="true" />
<div id="app" class="wrap"><p>Loading…</p></div>
<div id="flash" class="flash"></div>
<script>
const app = document.getElementById('app');
const flash = document.getElementById('flash');
let state = null;
let busy = false;

function showFlash(msg) {
  flash.textContent = msg;
  flash.classList.add('show');
  setTimeout(() => flash.classList.remove('show'), 700);
}
async function api(path, body) {
  const usePost = path === '/api/label' || path === '/api/undo';
  const opts = usePost ? {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body || {}),
  } : {};
  const response = await fetch(path, opts);
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}
function noteValue() {
  const el = document.getElementById('note');
  return el ? el.value : '';
}
function render() {
  if (!state) return;
  if (!state.current) {
    app.innerHTML = `<div class="done"><p>All ${state.total} reports are labeled.</p><p>Saved to<br><code>${state.label_output_csv}</code></p></div>`;
    return;
  }
  const row = state.current;
  const checkpointNote = state.checkpoint_exists ? '' : ' (path not found yet)';
  app.innerHTML = `
    <div class="top">
      <div class="progress"><strong>${state.reviewed + 1}</strong> of ${state.total} · ${state.remaining} left</div>
      <div class="meta">${row.patient_id} · source ${row.source_idx}</div>
    </div>
    <div class="paths">
      <div>data <code>${state.data_csv}</code></div>
      <div>checkpoint <code>${state.model_checkpoint}</code>${checkpointNote}</div>
    </div>
    <div class="banner">
      <div>Gold ${row.true_label_name || row.true_label} · model ${row.pred_label_name || row.pred_label}</div>
      <div>P(inconsistent) ${Number(row.prob_inconsistent || 0).toFixed(3)} · ${row.error_type || ''}</div>
    </div>
    <div class="cols">
      <div><h2>Findings</h2><div class="text" id="findings"></div></div>
      <div><h2>Conclusion</h2><div class="text" id="conclusion"></div></div>
    </div>
    <textarea id="note" placeholder="Optional note"></textarea>
    <div class="actions">
      <button type="button" class="act k1" data-lab="error">1 — Error</button>
      <button type="button" class="act k0" data-lab="no_error">0 — No error</button>
      <button type="button" class="act kx" data-lab="unclear">x — Unclear</button>
      <button type="button" class="act ku" id="btn-undo" ${state.undo_available ? '' : 'disabled'}>Undo (u)</button>
    </div>
    <div class="hint">Keys: <kbd>1</kbd> error · <kbd>0</kbd> no error · <kbd>x</kbd> unclear · <kbd>u</kbd> undo. <mark class="hl-left">left</mark> <mark class="hl-right">right</mark></div>`;
  document.getElementById('findings').innerHTML = row.findings_html || '';
  document.getElementById('conclusion').innerHTML = row.conclusion_html || '';
  const note = document.getElementById('note');
  if (note) note.value = row.review_note || '';
  app.querySelectorAll('button[data-lab]').forEach((button) => {
    button.onclick = () => submit(button.dataset.lab);
  });
  const undo = document.getElementById('btn-undo');
  if (undo) undo.onclick = () => undoLabel();
  document.getElementById('focus-trap').focus();
}
async function submit(humanLabel) {
  if (busy) return;
  busy = true;
  try {
    state = await api('/api/label', {human_label: humanLabel, note: noteValue()});
    showFlash('Saved: ' + humanLabel);
    render();
  } catch (err) {
    showFlash(String(err));
  } finally {
    busy = false;
  }
}
async function undoLabel() {
  if (busy || !state.undo_available) return;
  busy = true;
  try {
    state = await api('/api/undo', {});
    showFlash('Undone');
    render();
  } catch (err) {
    showFlash(String(err));
  } finally {
    busy = false;
  }
}
document.addEventListener('keydown', (event) => {
  if (!state || event.metaKey || event.ctrlKey || event.altKey) return;
  const target = event.target;
  if (target && (target.tagName === 'TEXTAREA' || target.tagName === 'INPUT')) return;
  const key = event.key.toLowerCase();
  if (key === '1') submit('error');
  else if (key === '0') submit('no_error');
  else if (key === 'x') submit('unclear');
  else if (key === 'u') undoLabel();
});
api('/api/state').then((payload) => { state = payload; render(); }).catch((err) => {
  app.innerHTML = '<p>' + err + '</p>';
});
</script>
</body>
</html>
"""


def make_handler(session: ReviewSession):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            return

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: Dict[str, Any], code: int = 200) -> None:
            self._send(code, json.dumps(payload).encode("utf-8"), "application/json")

        def _read_json(self) -> Dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            data = json.loads(raw.decode("utf-8") or "{}")
            if not isinstance(data, dict):
                raise ValueError("Expected a JSON object")
            return data

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path in {"/", "/index.html"}:
                self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
                return
            if path == "/api/state":
                self._json(session.state())
                return
            self._send(404, b"not found", "text/plain")

        def do_POST(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            try:
                data = self._read_json()
                if path == "/api/label":
                    self._json(session.label(str(data.get("human_label", "")), str(data.get("note", ""))))
                    return
                if path == "/api/undo":
                    self._json(session.undo())
                    return
                self._send(404, b"not found", "text/plain")
            except Exception as exc:  # noqa: BLE001
                self._send(400, str(exc).encode("utf-8"), "text/plain; charset=utf-8")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Open the PET/CT false-positive labeling page.")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--host", default=HOST)
    args = parser.parse_args()
    session = ReviewSession(DATA_CSV, LABEL_OUTPUT_CSV, MODEL_CHECKPOINT)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(session))
    url = f"http://{args.host}:{args.port}/"
    print(f"Labeling {len(session.queue)} reports from {DATA_CSV}")
    print(f"Checkpoint path (not loaded): {MODEL_CHECKPOINT}")
    print(f"Labels will be written to {LABEL_OUTPUT_CSV}")
    print(f"Open {url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
