"""
Compute aggregate metrics (CER, WER, BLEU, Cosine Similarity) across all
input folders, at each pipeline stage: TrOCR raw → Rule-based → T5 → Accent-normalised.
"""
import sys, json, unicodedata
import numpy as np
from pathlib import Path

import torch
import cv2
import editdistance
from jiwer import wer as compute_wer

# ── Local imports ────────────────────────────────────────────────────────────
from pipeline3 import (
    binarize, horizontal_projection, find_line_segments,
    adaptive_crop, IMAGE_EXTENSIONS,
)
from run_ocr import load_trocr
from eval_t5 import load_t5
from rule_corrector import SpanishDictionary
from dynamic_align import align_with_local_window

# ── Config ───────────────────────────────────────────────────────────────────
INPUT_DIR  = Path("input")
TROCR_DIR  = "models/trocr"
T5_DIR     = "models/t5"
WINDOW     = 5

def normalize_for_cer(text):
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()

def find_source_image(folder):
    for ext in IMAGE_EXTENSIONS:
        for f in folder.iterdir():
            if f.suffix.lower() == ext:
                return f
    return None

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load models
    print("Loading TrOCR...")
    processor, trocr_model = load_trocr(TROCR_DIR, device)
    print("Loading T5...")
    tokenizer, t5_model = load_t5(T5_DIR, device)
    print("Loading dictionary...")
    dictionary = SpanishDictionary()

    # BLEU
    try:
        import evaluate as hf_evaluate
        bleu_metric = hf_evaluate.load("bleu")
        has_bleu = True
    except:
        has_bleu = False
        print("WARNING: could not load BLEU metric")

    # Cosine similarity
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        has_cosine = True
    except:
        has_cosine = False

    folders = sorted(
        [d for d in INPUT_DIR.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name)
    )
    print(f"Found {len(folders)} folders\n")

    # Accumulators per stage
    all_trocr_preds = []
    all_rule_preds  = []
    all_t5_preds    = []
    all_gt          = []
    all_accnorm_preds = []
    all_accnorm_gt    = []

    for folder in folders:
        src_img = find_source_image(folder)
        src_txt = folder / "transcription.txt"
        if src_img is None or not src_txt.exists():
            print(f"  Folder {folder.name}: SKIP (missing files)")
            continue

        gt_lines = [ln.rstrip("\n") for ln in src_txt.read_text(encoding="utf-8").split("\n")]
        while gt_lines and not gt_lines[-1].strip():
            gt_lines.pop()

        image_bgr = cv2.imread(str(src_img))
        if image_bgr is None:
            continue

        # Detect lines
        bw = binarize(image_bgr)
        proj = horizontal_projection(bw)
        segments, est_h, period, conf, method = find_line_segments(proj, image_bgr)
        crops = [adaptive_crop(image_bgr, s, e, est_h) for s, e in segments]

        # TrOCR predictions
        from PIL import Image
        trocr_preds = []
        for crop in crops:
            pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            pixel_values = processor(pil, return_tensors="pt").pixel_values.to(device)
            with torch.no_grad():
                ids = trocr_model.generate(pixel_values, max_new_tokens=128)
            text = processor.decode(ids[0], skip_special_tokens=True)
            trocr_preds.append(text)

        # Rule-based correction
        rule_preds = [dictionary.correct_line(t) for t in trocr_preds]

        # T5 correction
        t5_preds = []
        for t in rule_preds:
            input_text = f"fix: {t}"
            enc = tokenizer(input_text, return_tensors="pt", max_length=128, truncation=True).to(device)
            with torch.no_grad():
                out = t5_model.generate(**enc, max_new_tokens=128)
            corrected = tokenizer.decode(out[0], skip_special_tokens=True)
            if abs(len(corrected.split()) - len(t.split())) <= 2:
                t5_preds.append(corrected)
            else:
                t5_preds.append(t)

        # Align predictions to GT
        pairs = align_with_local_window(trocr_preds, gt_lines, window=WINDOW)

        for i, p in enumerate(pairs):
            if p.get("skipped") or not p["matched_gt"]:
                continue
            gt = p["matched_gt"]
            pred_idx = p["pred_index"]
            all_gt.append(gt)
            all_trocr_preds.append(trocr_preds[pred_idx])
            all_rule_preds.append(rule_preds[pred_idx])
            all_t5_preds.append(t5_preds[pred_idx])
            all_accnorm_preds.append(normalize_for_cer(rule_preds[pred_idx]))
            all_accnorm_gt.append(normalize_for_cer(gt))

        print(f"  Folder {folder.name}: {len(trocr_preds)} lines detected, {len(gt_lines)} GT lines")

    # ── Compute metrics ──────────────────────────────────────────────────────
    def avg_cer(preds, refs):
        total_ed = sum(editdistance.eval(p, r) for p, r in zip(preds, refs))
        total_len = sum(len(r) for r in refs)
        return total_ed / max(total_len, 1)

    def avg_wer(preds, refs):
        try:
            return compute_wer(refs, preds)
        except:
            return 0.0

    def avg_bleu(preds, refs):
        if not has_bleu:
            return None
        try:
            result = bleu_metric.compute(predictions=preds, references=[[r] for r in refs])
            return result["bleu"]
        except:
            return None

    def avg_cosine(preds, refs):
        if not has_cosine:
            return None
        try:
            vectorizer = TfidfVectorizer()
            all_texts = preds + refs
            tfidf = vectorizer.fit_transform(all_texts)
            pred_vecs = tfidf[:len(preds)]
            ref_vecs  = tfidf[len(preds):]
            sims = [cosine_similarity(pred_vecs[i], ref_vecs[i])[0][0] for i in range(len(preds))]
            return np.mean(sims)
        except:
            return None

    print(f"\n{'='*70}")
    print(f"  AGGREGATE METRICS ACROSS {len(folders)} FOLDERS ({len(all_gt)} matched lines)")
    print(f"{'='*70}\n")

    stages = {
        "TrOCR (raw)":       (all_trocr_preds, all_gt),
        "+ Rule-based":      (all_rule_preds,  all_gt),
        "+ T5 corrector":    (all_t5_preds,    all_gt),
        "Accent-normalised": (all_accnorm_preds, all_accnorm_gt),
    }

    print(f"{'Stage':<22} {'CER':>8} {'WER':>8} {'BLEU':>8} {'CosSim':>8}")
    print(f"{'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for name, (preds, refs) in stages.items():
        cer  = avg_cer(preds, refs)
        w    = avg_wer(preds, refs)
        b    = avg_bleu(preds, refs)
        c    = avg_cosine(preds, refs)
        b_str = f"{b:.4f}" if b is not None else "N/A"
        c_str = f"{c:.4f}" if c is not None else "N/A"
        print(f"{name:<22} {cer:>7.2%} {w:>7.2%} {b_str:>8} {c_str:>8}")

    print()

if __name__ == "__main__":
    main()
