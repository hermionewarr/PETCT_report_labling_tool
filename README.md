# PET/CT report labelling tool

Made for the paper: [Error Detection for PET/CT Radiology Reports: Domain-Specific vs Large Language Models](https://arxiv.org/pdf/2608.30021)

Point labelling tool at reports and a trained domain-specific model, then review the reports the model flagged.

```bash
python run_detection.py    # findings + conclusion -> predictions CSV
python label_reports.py    # review flagged reports in the browser
```

Open the URL the second command prints (default http://127.0.0.1:8766/).

Keys: `1` error, `0` no error, `x` unclear, `u` undo.

## What to edit

All settings live in `paths.py`.

| Name | Replace with |
|---|---|
| `REPORTS_CSV` | Findings and conclusion file. Dummy rows are `data/dummy_reports.csv` |
| `MODEL_CHECKPOINT` | Trained model folder (`config.json` and `cls_classifier.pt`) |
| `MODEL_SIZE` | `4`/`small`, `6`/`big`, `8`/`8L`, or `12`/`12L` |
| `SEQ_LENGTH` | Length that checkpoint was trained with (600 for the paper runs) |
| `PREDICTIONS_CSV` | Where predictions are written |
| `LABEL_OUTPUT_CSV` | Where review labels are written |

`MODEL_CHECKPOINT` currently points at `checkpoints/dummy`, which is empty. `run_detection.py` stops until that path is a real checkpoint.

Reports need `findings` and `conclusion`. `patient_id` and `source_idx` are optional. A gold label (`true_label_name` or `error_type`) is optional. Without one, the page reviews every report the model called inconsistent.

## UI 

<img width="750" height="300" alt="image" src="https://github.com/user-attachments/assets/9c6b28db-8620-4635-a99f-86b0acbc11f0" />

## Paper

Made for the paper: [Error Detection for PET/CT Radiology Reports:Domain-Specific vs Large Language Models](https://arxiv.org/pdf/2608.30021)
