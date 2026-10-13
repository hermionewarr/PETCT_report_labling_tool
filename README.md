# PET/CT report labeling tool

Local page for marking model false positives as **error**, **no error**, or **unclear**.

The dummy input matches the columns written by a BERT-6 inconsistency run (`predictions.csv`). No clinical reports and no trained weights are included.

## Run

```bash
cd /path/to/PETCT_report_labling_tool
python label_reports.py
```

Open the URL it prints (default http://127.0.0.1:8766/).

Keys: `1` error, `0` no error, `x` unclear, `u` undo.

## Point at your files

Edit the three paths at the top of `label_reports.py`:

| Name | Dummy default | Replace with |
|---|---|---|
| `DATA_CSV` | `data/dummy_bert6_predictions.csv` | BERT predictions CSV |
| `MODEL_CHECKPOINT` | `checkpoints/dummy` | Run directory or `model/` checkpoint |
| `LABEL_OUTPUT_CSV` | `labels/fp_labels.csv` | Where saved labels should go |

`ONLY_FALSE_POSITIVES = True` keeps rows with `is_fp0` of 1 (gold consistent, model called inconsistent). Set it to `False` to label every row in the CSV.

The checkpoint path is stored on each saved row. The page does not load the weights.
