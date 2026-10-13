# PET/CT report labeling tool

Download this, point it at a reports file, and review what the error-detection model flagged.

Two steps, both local. Nothing is sent off the machine.

```bash
python run_detection.py    # findings + conclusion -> predictions CSV
python label_reports.py    # review the flagged reports in the browser
```

Open the URL the second command prints (default http://127.0.0.1:8766/).

Keys: `1` error, `0` no error, `x` unclear, `u` undo.

A label of **error** means the findings and conclusion really disagree. **No error** means the model flag was a false alarm. The counts are the readout of the model's flags.

## What to edit

All paths live in `paths.py`.

| Name | Dummy default | Replace with |
|---|---|---|
| `REPORTS_CSV` | `data/dummy_reports.csv` | Your reports, or a set we give you |
| `MODEL_CHECKPOINT` | `checkpoints/dummy` | Trained weights, once that loader exists |
| `PREDICTIONS_CSV` | `data/predictions.csv` | Where model output should be written |
| `LABEL_OUTPUT_CSV` | `labels/fp_labels.csv` | Where review labels should be written |

Reports need `findings` and `conclusion` (`Findings` and `Conclusion` are also accepted). `patient_id` and `source_idx` are optional. A gold label (`true_label_name` or `error_type`) is optional. Without one, the page reviews every report the model called inconsistent.

The checkpoint in this repo is a stand-in. It flags a report only when findings and conclusion disagree on left/right or mild/severe. It writes the same columns as a BERT-6 `predictions.csv`. Pointing `MODEL_CHECKPOINT` at a real run will stop with an error until that loader is added, so the stand-in is not silently used on real weights.

`ONLY_GOLD_FALSE_POSITIVES = True` limits the review to gold-consistent reports the model still flagged. That needs a gold column.
