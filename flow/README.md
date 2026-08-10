# 🔄 Pipeline Flow — Core Files Only

This folder contains **only the 7 files** that make up the actual OCR pipeline.
Files are numbered in execution order.

```
  ┌─────────────────────────┐
  │   Scanned Page Image    │
  └────────────┬────────────┘
               ▼
  ┌─────────────────────────┐
  │ 1_line_detection.py     │  Binarize → Horizontal Projection → ACF
  │ (pipeline3.py)          │  → Valley Detection → Line Crops
  └────────────┬────────────┘
               ▼
  ┌─────────────────────────┐
  │ 2_trocr_inference.py    │  Load TrOCR model → Predict text
  │ (run_ocr.py)            │  for each cropped line
  └────────────┬────────────┘
               ▼
  ┌─────────────────────────┐
  │ 3_rule_correction.py    │  ç→z, u↔v, f↔s resolution
  │ (rule_corrector.py)     │  via 1,763-word archaic dictionary
  └────────────┬────────────┘
               ▼
  ┌─────────────────────────┐
  │ 4_t5_correction.py      │  [Optional] T5 seq2seq error correction
  │ (eval_t5.py)            │  "correct: <text>" → corrected text
  └────────────┬────────────┘
               ▼
  ┌─────────────────────────┐
  │ 5_gemini_correction.py  │  [Optional] Gemini 2.5 Flash API
  │ (gemini_corrector.py)   │  Ultra-conservative prompt + word-count guard
  └────────────┬────────────┘
               ▼
  ┌─────────────────────────┐
  │ 6_dynamic_alignment.py  │  Needleman-Wunsch DP alignment
  │ (dynamic_align.py)      │  Predictions ↔ Ground Truth + CER
  └────────────┬────────────┘
               ▼
  ┌─────────────────────────┐
  │ 7_streamlit_app.py      │  Interactive UI — ties all stages together
  │ (app.py)                │  Per-stage CER, annotated images, line crops
  └─────────────────────────┘
```

## What's NOT here (16 support files left in `project/`)

| File | Purpose |
|---|---|
| `train_trocr.py` | TrOCR fine-tuning script |
| `train_T5.py` | T5 training + noise functions |
| `compute_metrics.py` | Standalone aggregate metrics |
| `check_metrics.py` | Quick metric check |
| `generate_trocr_preds.py` | Generate TrOCR prediction pairs for T5 training |
| `prepare_dataset.py` | Dataset preparation |
| `preprocess.py` | Image preprocessing |
| `realign_and_eval.py` | Realignment evaluation |
| `generate_assets.py` | Asset generation |
| `deploy_to_spaces.py` | HuggingFace Spaces deployment |
| `upload_models.py` | Model upload to HuggingFace |
| `upload_models_auto.py` | Auto model upload |
| `speedtest.py` | Performance benchmarking |
| `report.txt` | Technical report |
| `requirements.txt` | Dependencies |
| `.gitignore` | Git ignore rules |
