"""
eval_t5.py — Evaluate T5 OCR corrector on aligned val set
==========================================================
Loads the trained T5 model from models/t5, runs it on TrOCR predictions
(using the realigned val set), and compares CER against rule-based and
Gemini results.

No API calls needed — everything runs locally.
"""

import csv
import json
import time
import types
from pathlib import Path

import editdistance
import torch
from PIL import Image
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)
import evaluate

# ── Local imports ─────────────────────────────────────────────────────────────
from rule_corrector import SpanishDictionary, apply_rules, normalize_for_cer

# ── Config ────────────────────────────────────────────────────────────────────
TROCR_MODEL_DIR = "models/trocr"
T5_MODEL_DIR    = "models/t5"
VAL_CSV         = "data/val.csv"
TASK_PREFIX     = "correct: "
MAX_LEN_TROCR   = 128
BEAM_SIZE_TROCR = 4
MAX_LEN_T5      = 128    # Override the saved max_length=20 (way too short!)
NUM_BEAMS_T5    = 4       # Use beam search for better quality

cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")


# ── TrOCR helpers (same as realign_and_eval.py) ──────────────────────────────

def _patch_embed_positions(model):
    embed_pos = model.decoder.model.decoder.embed_positions
    if type(embed_pos).__name__ == "TrOCRLearnedPositionalEmbedding":
        def _fwd(self, input_ids_shape, past_key_values_length=0):
            bsz, seq_len = input_ids_shape
            positions = torch.arange(
                past_key_values_length, past_key_values_length + seq_len,
                dtype=torch.long, device=self.weight.device,
            )
            return torch.nn.Embedding.forward(self, positions)
        embed_pos.forward = types.MethodType(_fwd, embed_pos)
    elif type(embed_pos).__name__ == "TrOCRSinusoidalPositionalEmbedding":
        def _fwd_sin(self, input_ids, past_key_values_length=0):
            bsz, seq_len = input_ids.size()
            position_ids = self.create_position_ids_from_input_ids(
                input_ids, self.padding_idx, past_key_values_length
            ).to(input_ids.device)
            max_pos = self.padding_idx + 1 + seq_len
            if (self.weights is None or max_pos > self.weights.size(0)
                    or self.weights.device != input_ids.device):
                self.weights = self.get_embedding(
                    max_pos, self.embedding_dim, self.padding_idx
                ).to(input_ids.device)
            return (self.weights
                    .index_select(0, position_ids.view(-1))
                    .view(bsz, seq_len, -1).detach())
        embed_pos.forward = types.MethodType(_fwd_sin, embed_pos)


def load_trocr(model_dir, device):
    print(f"Loading TrOCR from {model_dir}…")
    processor = TrOCRProcessor.from_pretrained(model_dir)
    model = VisionEncoderDecoderModel.from_pretrained(model_dir).to(device).eval()
    _patch_embed_positions(model)
    model.generation_config.max_length = MAX_LEN_TROCR
    model.generation_config.num_beams = BEAM_SIZE_TROCR
    return processor, model


def trocr_predict(processor, model, image_path, device):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        ids = model.generate(**inputs)
    return processor.tokenizer.decode(ids[0], skip_special_tokens=True).strip()


# ── T5 helpers ────────────────────────────────────────────────────────────────

def load_t5(model_dir, device):
    print(f"Loading T5 from {model_dir}…")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir).to(device).eval()
    return tokenizer, model


def t5_correct(text: str, tokenizer, model, device) -> str:
    """Run T5 correction on a single text line."""
    input_text = TASK_PREFIX + text
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=MAX_LEN_T5,
        truncation=True,
        padding=False,
    ).to(device)

    with torch.no_grad():
        ids = model.generate(
            **inputs,
            max_length=MAX_LEN_T5,
            num_beams=NUM_BEAMS_T5,
            early_stopping=True,
            no_repeat_ngram_size=0,
            length_penalty=1.0,
        )
    result = tokenizer.decode(ids[0], skip_special_tokens=True).strip()

    # Word-count guard: reject if T5 changed the number of words
    orig_words = text.split()
    corr_words = result.split()
    if len(corr_words) != len(orig_words):
        print(f"  [T5] word count mismatch: {len(orig_words)} → {len(corr_words)}, keeping original.")
        return text
    return result


# ── Alignment helper ─────────────────────────────────────────────────────────

def find_best_gt_match(trocr_text: str, all_gt_lines: list[str]):
    best_idx = 0
    best_dist = float("inf")
    for i, gt in enumerate(all_gt_lines):
        dist = editdistance.eval(trocr_text.lower(), gt.lower())
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return all_gt_lines[best_idx], best_idx


def compute_scores(preds, refs):
    cer = cer_metric.compute(predictions=preds, references=refs)
    wer = wer_metric.compute(predictions=preds, references=refs)
    return {"CER": round(cer, 4), "WER": round(wer, 4)}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # Load val.csv
    with open(VAL_CSV, newline="", encoding="utf-8") as f:
        all_val_rows = list(csv.DictReader(f))
    val_rows = all_val_rows[:19]  # Same 19 lines as previous eval
    all_gt_lines = [row["transcription"] for row in all_val_rows]
    print(f"Val set: {len(all_val_rows)} total, evaluating {len(val_rows)}")

    # Load models
    trocr_proc, trocr_model = load_trocr(TROCR_MODEL_DIR, device)
    t5_tok, t5_model = load_t5(T5_MODEL_DIR, device)

    # Load dictionary for rule-based corrections
    print("Loading Spanish dictionary…")
    dictionary = SpanishDictionary()

    # ── Run pipeline ──────────────────────────────────────────────────────────
    print("\nRunning 4-stage evaluation:")
    print("  Stage 1: TrOCR → Stage 2: Rules → Stage 3: T5 → Metrics\n")

    trocr_preds = []
    rule_preds  = []
    t5_preds    = []
    aligned_refs = []

    for i, row in enumerate(val_rows):
        # Stage 1: TrOCR
        raw = trocr_predict(trocr_proc, trocr_model, row["image_path"], device)
        trocr_preds.append(raw)

        # Realign GT
        best_gt, _ = find_best_gt_match(raw, all_gt_lines)
        aligned_refs.append(best_gt)

        # Stage 2: Rule-based correction
        ruled = apply_rules(raw, dictionary)
        rule_preds.append(ruled)

        # Stage 3: T5 correction (on rule-corrected text)
        t5_out = t5_correct(ruled, t5_tok, t5_model, device)
        t5_preds.append(t5_out)

        if (i + 1) % 5 == 0 or i == len(val_rows) - 1:
            print(f"  {i + 1}/{len(val_rows)}")

    # ── Compute metrics ───────────────────────────────────────────────────────
    trocr_scores = compute_scores(trocr_preds, aligned_refs)
    rule_scores  = compute_scores(rule_preds,  aligned_refs)
    t5_scores    = compute_scores(t5_preds,    aligned_refs)

    # Accent-normalised CER
    norm_t5   = [normalize_for_cer(p) for p in t5_preds]
    norm_refs = [normalize_for_cer(r) for r in aligned_refs]
    t5_norm_cer = round(cer_metric.compute(predictions=norm_t5, references=norm_refs), 4)

    # Load Gemini results for comparison
    gemini_cer = "N/A"
    gemini_file = Path("data/eval_results_aligned.json")
    if gemini_file.exists():
        with open(gemini_file) as f:
            gdata = json.load(f)
        gemini_cer = gdata["metrics"]["after_gemini"]["CER"]

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  RESULTS — T5 vs Gemini comparison")
    print("═" * 65)
    print(f"  Stage                    │  CER    │  WER")
    print(f"  ─────────────────────────┼─────────┼─────────")
    print(f"  TrOCR (raw)              │ {trocr_scores['CER']:.4f}  │ {trocr_scores['WER']:.4f}")
    print(f"  + Rule-based fixes       │ {rule_scores['CER']:.4f}  │ {rule_scores['WER']:.4f}")
    print(f"  + T5 correction          │ {t5_scores['CER']:.4f}  │ {t5_scores['WER']:.4f}")
    print(f"  (T5 accent-normalised)   │ {t5_norm_cer:.4f}  │  —")
    print(f"  ─────────────────────────┼─────────┼─────────")
    print(f"  Gemini (previous run)    │ {gemini_cer}  │")
    t5_delta = trocr_scores["CER"] - t5_scores["CER"]
    print(f"  T5 total improvement     │ {t5_delta:+.4f}  │")
    print("═" * 65)

    # ── Sample comparisons ────────────────────────────────────────────────────
    print("\n  Sample line-by-line comparisons:")
    for i in range(min(10, len(val_rows))):
        gt = aligned_refs[i]
        trocr = trocr_preds[i]
        rules = rule_preds[i]
        t5 = t5_preds[i]
        # Per-line CER
        trocr_cer_i = round(editdistance.eval(trocr, gt) / max(len(gt), 1), 3)
        t5_cer_i = round(editdistance.eval(t5, gt) / max(len(gt), 1), 3)
        improved = "✓" if t5_cer_i < trocr_cer_i else ("—" if t5_cer_i == trocr_cer_i else "✗")
        print(f"    [{improved}] Line {i+1} (CER: {trocr_cer_i:.3f} → {t5_cer_i:.3f})")
        print(f"        GT    : {gt}")
        print(f"        TrOCR : {trocr}")
        if rules != trocr:
            print(f"        Rules : {rules}")
        print(f"        T5    : {t5}")
        print()

    # ── Save results ──────────────────────────────────────────────────────────
    out = Path("data/eval_results_t5.json")
    samples = []
    for i in range(len(val_rows)):
        samples.append({
            "image": val_rows[i]["image_path"],
            "gt": aligned_refs[i],
            "trocr": trocr_preds[i],
            "after_rules": rule_preds[i],
            "after_t5": t5_preds[i],
        })
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": {
                "trocr_raw": trocr_scores,
                "after_rules": rule_scores,
                "after_t5": t5_scores,
                "t5_accent_norm_cer": t5_norm_cer,
                "gemini_cer_for_comparison": gemini_cer,
            },
            "samples": samples,
        }, f, indent=2, ensure_ascii=False)
    print(f"  Full results → {out}")


if __name__ == "__main__":
    main()
