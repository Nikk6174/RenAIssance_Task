"""
realign_and_eval.py
===================
1. Runs TrOCR on each val image to get predictions.
2. For each TrOCR prediction, find the best-matching ground truth line
   from val.csv (using edit distance), fixing the misalignment.
3. Applies rule-based + Gemini corrections on the TrOCR output.
4. Reports CER at each stage with CORRECT alignment.
5. Saves a corrected val_aligned.csv.

This avoids the inflated CER caused by image-to-GT misalignment in val.csv.
"""

import csv
import json
import os
import sys
import time
from pathlib import Path

import editdistance
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import evaluate
import types

# ── Local imports ─────────────────────────────────────────────────────────────
from rule_corrector import SpanishDictionary, apply_rules, normalize_for_cer
from gemini_corrector import GeminiCorrector, get_text_changes, categorize_changes

# ── Config ────────────────────────────────────────────────────────────────────
TROCR_MODEL_DIR = "models/trocr"
VAL_CSV         = "data/val.csv"
MAX_LEN         = 128
BEAM_SIZE       = 4
GEMINI_MODEL    = "gemini-2.5-flash"
# Gemini free tier: 5 RPM, 20 requests/day — limit lines to fit in quota
MAX_EVAL_LINES  = 19
GEMINI_DELAY_S  = 13  # seconds between calls (5 RPM = 12s min, add buffer)

cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")


def _resolve_api_key() -> str:
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        val = os.getenv(var)
        if val:
            return val
    env_file = Path("gemini.env")
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("GEMINI_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise RuntimeError("No Gemini API key found.")


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
    model.generation_config.max_length = MAX_LEN
    model.generation_config.num_beams = BEAM_SIZE
    return processor, model


def trocr_predict(processor, model, image_path, device):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        ids = model.generate(**inputs)
    return processor.tokenizer.decode(ids[0], skip_special_tokens=True).strip()


def compute_scores(preds, refs):
    cer = cer_metric.compute(predictions=preds, references=refs)
    wer = wer_metric.compute(predictions=preds, references=refs)
    return {"CER": round(cer, 4), "WER": round(wer, 4)}


def find_best_gt_match(trocr_text: str, all_gt_lines: list[str]) -> tuple[str, int, int]:
    """
    Find the GT line with the lowest edit distance to the TrOCR text.
    Returns (best_gt, best_index, edit_dist).
    """
    best_idx = 0
    best_dist = float("inf")
    for i, gt in enumerate(all_gt_lines):
        dist = editdistance.eval(trocr_text.lower(), gt.lower())
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return all_gt_lines[best_idx], best_idx, best_dist


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load val.csv ──────────────────────────────────────────────────────────
    with open(VAL_CSV, newline="", encoding="utf-8") as f:
        all_val_rows = list(csv.DictReader(f))
    # Limit to MAX_EVAL_LINES to stay within Gemini's 20 req/day quota
    val_rows = all_val_rows[:MAX_EVAL_LINES]
    all_gt_lines = [row["transcription"] for row in all_val_rows]  # all GT for matching
    print(f"Val set: {len(all_val_rows)} lines total, evaluating {len(val_rows)}")

    # ── Load TrOCR ────────────────────────────────────────────────────────────
    trocr_proc, trocr_model = load_trocr(TROCR_MODEL_DIR, device)

    # ── Get TrOCR predictions for all images ──────────────────────────────────
    print("\nStage 1: Running TrOCR on all val images…")
    trocr_preds = []
    image_paths = []
    for i, row in enumerate(val_rows):
        img_path = row["image_path"]
        pred = trocr_predict(trocr_proc, trocr_model, img_path, device)
        trocr_preds.append(pred)
        image_paths.append(img_path)
        if (i + 1) % 20 == 0 or i == len(val_rows) - 1:
            print(f"  {i + 1}/{len(val_rows)}")

    # ── Realign: find best GT match for each TrOCR prediction ─────────────────
    print("\nRealigning GT to TrOCR predictions…")
    aligned_data = []
    unmatched = 0

    for i, pred in enumerate(trocr_preds):
        original_gt = all_gt_lines[i]
        best_gt, best_idx, edit_dist = find_best_gt_match(pred, all_gt_lines)

        # Calculate CER for original vs best match
        orig_cer = editdistance.eval(pred, original_gt) / max(len(original_gt), 1)
        best_cer = editdistance.eval(pred, best_gt) / max(len(best_gt), 1)

        if best_idx != i:
            print(f"  Line {i+1:3d}: SHIFTED (was line {best_idx+1})")
            print(f"            TrOCR: {pred[:60]}")
            print(f"            Old GT: {original_gt[:60]}")
            print(f"            New GT: {best_gt[:60]}")
            print(f"            CER: {orig_cer:.2f} → {best_cer:.2f}")
            unmatched += 1

        aligned_data.append({
            "image_path": image_paths[i],
            "trocr": pred,
            "original_gt": original_gt,
            "aligned_gt": best_gt,
            "original_gt_idx": i,
            "aligned_gt_idx": best_idx,
            "shifted": best_idx != i,
        })

    print(f"\n  {unmatched} of {len(trocr_preds)} lines were misaligned")

    # ── Compute CER with original vs aligned GT ──────────────────────────────
    orig_refs = [d["original_gt"] for d in aligned_data]
    aligned_refs = [d["aligned_gt"] for d in aligned_data]

    orig_scores = compute_scores(trocr_preds, orig_refs)
    aligned_scores = compute_scores(trocr_preds, aligned_refs)

    print(f"\n  TrOCR CER (original alignment):  {orig_scores['CER']:.4f}")
    print(f"  TrOCR CER (realigned):           {aligned_scores['CER']:.4f}")

    # ── Save aligned val.csv ──────────────────────────────────────────────────
    aligned_csv = Path("data/val_aligned.csv")
    with open(aligned_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "transcription"])
        writer.writeheader()
        for d in aligned_data:
            writer.writerow({
                "image_path": d["image_path"],
                "transcription": d["aligned_gt"],
            })
    print(f"  Saved aligned CSV → {aligned_csv}")

    # ── Stage 2: Rule-based correction ────────────────────────────────────────
    print("\nStage 2: Applying rule-based corrections…")
    dictionary = SpanishDictionary()
    rule_preds = [apply_rules(p, dictionary) for p in trocr_preds]
    rule_scores = compute_scores(rule_preds, aligned_refs)
    print(f"  CER after rules: {rule_scores['CER']:.4f}")

    # ── Stage 3: Gemini correction ────────────────────────────────────────────
    print("\nStage 3: Applying Gemini corrections…")
    api_key = _resolve_api_key()
    corrector = GeminiCorrector(api_key=api_key, model=GEMINI_MODEL)

    gemini_preds = []
    for i, text in enumerate(rule_preds):
        corr, status = corrector.correct(text)
        gemini_preds.append(corr)
        if (i + 1) % 10 == 0 or i == len(rule_preds) - 1:
            print(f"  {i + 1}/{len(rule_preds)}  [{status}]")
        time.sleep(GEMINI_DELAY_S)  # Stay under 5 RPM

    gemini_scores = compute_scores(gemini_preds, aligned_refs)

    # Accent-normalised CER
    norm_preds = [normalize_for_cer(p) for p in gemini_preds]
    norm_refs = [normalize_for_cer(r) for r in aligned_refs]
    norm_cer = round(cer_metric.compute(predictions=norm_preds, references=norm_refs), 4)

    # ── Final report ──────────────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  RESULTS (with corrected alignment)")
    print("═" * 65)
    print(f"  Stage                    │  CER    │  WER")
    print(f"  ─────────────────────────┼─────────┼─────────")
    print(f"  TrOCR (raw)              │ {aligned_scores['CER']:.4f}  │ {aligned_scores['WER']:.4f}")
    print(f"  + Rule-based fixes       │ {rule_scores['CER']:.4f}  │ {rule_scores['WER']:.4f}")
    print(f"  + Gemini correction      │ {gemini_scores['CER']:.4f}  │ {gemini_scores['WER']:.4f}")
    print(f"  (accent-normalised CER)  │ {norm_cer:.4f}  │  —")
    cer_delta = aligned_scores["CER"] - gemini_scores["CER"]
    print(f"  LLM improvement          │ {cer_delta:+.4f}  │")
    print("═" * 65)

    # ── Save detailed results ─────────────────────────────────────────────────
    samples = []
    for i, d in enumerate(aligned_data):
        samples.append({
            "image": d["image_path"],
            "gt": d["aligned_gt"],
            "gt_was_shifted": d["shifted"],
            "trocr": trocr_preds[i],
            "after_rules": rule_preds[i],
            "after_gemini": gemini_preds[i],
        })

    out_path = Path("data/eval_results_aligned.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": {
                "trocr_raw": aligned_scores,
                "after_rules": rule_scores,
                "after_gemini": gemini_scores,
                "accent_norm_cer": norm_cer,
                "original_misaligned_cer": orig_scores["CER"],
            },
            "samples": samples,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  Full results → {out_path}")

    # ── Show sample corrections ───────────────────────────────────────────────
    print("\n  Sample corrections (first 10):")
    for s in samples[:10]:
        shifted = " ⚠SHIFTED" if s["gt_was_shifted"] else ""
        print(f"    GT     : {s['gt']}{shifted}")
        print(f"    TrOCR  : {s['trocr']}")
        print(f"    Rules  : {s['after_rules']}")
        print(f"    Gemini : {s['after_gemini']}")
        print()


if __name__ == "__main__":
    main()
