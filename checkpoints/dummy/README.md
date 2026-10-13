Placeholder checkpoint. `dummy_model.json` marks this directory as the keyword stand-in, not the paper BERT.

`run_detection.py` refuses any other folder until a real checkpoint loader is added. Point `MODEL_CHECKPOINT` in `paths.py` at that folder when it exists.
