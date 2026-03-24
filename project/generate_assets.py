"""
Generate submitted_assets output files with pipeline metrics.
"""
import json, csv, unicodedata, time
from pathlib import Path
import torch, cv2
import editdistance
from PIL import Image

# Local imports
from pipeline3 import detect_text_lines, crop_line, IMAGE_EXTENSIONS
from run_ocr import load_trocr
from eval_t5 import load_t5
from rule_corrector import SpanishDictionary, apply_rules
from dynamic_align import align_with_local_window

INPUT_DIR = Path("input")
OUT_DIR = Path("../submitted_assets")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load models
print("Loading TrOCR...")
processor, trocr_model = load_trocr("models/trocr", device)
print("Loading T5...")
tokenizer, t5_model = load_t5("models/t5", device)
print("Loading dictionary...")
dictionary = SpanishDictionary()

def normalize_for_cer(text):
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()

def find_source_image(folder):
    for ext in IMAGE_EXTENSIONS:
        for f in folder.iterdir():
            if f.suffix.lower() == ext:
                return f
    return None

folders = sorted(
    [d for d in INPUT_DIR.iterdir() if d.is_dir() and d.name.isdigit()],
    key=lambda d: int(d.name)
)
print(f"Found {len(folders)} folders\n")

all_results = []
per_folder_metrics = []

for folder in folders:
    src_img = find_source_image(folder)
    src_txt = folder / "transcription.txt"
    if src_img is None or not src_txt.exists():
        print(f"  Folder {folder.name}: SKIP")
        continue

    gt_lines = [ln.rstrip("\n") for ln in src_txt.read_text(encoding="utf-8").split("\n")]
    while gt_lines and not gt_lines[-1].strip():
        gt_lines.pop()

    image_bgr = cv2.imread(str(src_img))
    if image_bgr is None:
        continue

    # Detect lines
    boxes, stats = detect_text_lines(image_bgr)
    pad_v, pad_h = stats["pad_v"], stats["pad_h"]

    # Per-line processing
    trocr_preds = []
    rule_preds = []
    t5_preds = []

    t0 = time.time()
    for box in boxes:
        crop_bgr, _ = crop_line(image_bgr, box["bbox"], pad_v, pad_h)
        pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))

        # TrOCR
        px = processor(pil, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            ids = trocr_model.generate(px, max_new_tokens=128)
        raw = processor.decode(ids[0], skip_special_tokens=True)
        trocr_preds.append(raw)

        # Rules
        ruled = apply_rules(raw, dictionary)
        rule_preds.append(ruled)

        # T5
        enc = tokenizer(f"fix: {ruled}", return_tensors="pt", max_length=128, truncation=True).to(device)
        with torch.no_grad():
            out = t5_model.generate(**enc, max_new_tokens=128)
        corrected = tokenizer.decode(out[0], skip_special_tokens=True)
        if abs(len(corrected.split()) - len(ruled.split())) <= 2:
            t5_preds.append(corrected)
        else:
            t5_preds.append(ruled)

    elapsed = time.time() - t0

    # Align
    pairs = align_with_local_window(trocr_preds, gt_lines, window=5)

    # Compute per-folder metrics
    trocr_ed, rule_ed, t5_ed, accnorm_ed, total_len, accnorm_len = 0, 0, 0, 0, 0, 0
    line_results = []
    for p in pairs:
        if p.get("skipped") or not p["matched_gt"]:
            continue
        gt = p["matched_gt"]
        trocr_text = p["trocr_text"]
        # Find the index in our arrays
        try:
            idx = trocr_preds.index(trocr_text)
        except ValueError:
            continue
        total_len += len(gt)
        trocr_ed += editdistance.eval(trocr_preds[idx], gt)
        rule_ed += editdistance.eval(rule_preds[idx], gt)
        t5_ed += editdistance.eval(t5_preds[idx], gt)
        norm_pred = normalize_for_cer(rule_preds[idx])
        norm_gt = normalize_for_cer(gt)
        accnorm_len += len(norm_gt)
        accnorm_ed += editdistance.eval(norm_pred, norm_gt)

        line_results.append({
            "ground_truth": gt,
            "trocr_prediction": trocr_preds[idx],
            "rule_corrected": rule_preds[idx],
            "t5_corrected": t5_preds[idx],
            "cer_trocr": round(editdistance.eval(trocr_preds[idx], gt) / max(len(gt), 1), 4),
            "cer_rule": round(editdistance.eval(rule_preds[idx], gt) / max(len(gt), 1), 4),
            "cer_t5": round(editdistance.eval(t5_preds[idx], gt) / max(len(gt), 1), 4),
        })

    cer_trocr = trocr_ed / max(total_len, 1)
    cer_rule = rule_ed / max(total_len, 1)
    cer_t5 = t5_ed / max(total_len, 1)
    cer_accnorm = accnorm_ed / max(accnorm_len, 1)

    folder_data = {
        "folder": folder.name,
        "n_detected": len(boxes),
        "n_gt": len(gt_lines),
        "detection_method": stats["method"],
        "cer_trocr": round(cer_trocr, 4),
        "cer_rule": round(cer_rule, 4),
        "cer_t5": round(cer_t5, 4),
        "cer_accent_normalised": round(cer_accnorm, 4),
        "time_seconds": round(elapsed, 2),
        "lines": line_results,
    }
    all_results.append(folder_data)
    per_folder_metrics.append({
        "folder": folder.name,
        "cer_trocr": f"{cer_trocr:.2%}",
        "cer_rule": f"{cer_rule:.2%}",
        "cer_t5": f"{cer_t5:.2%}",
        "cer_accent_norm": f"{cer_accnorm:.2%}",
        "lines_detected": len(boxes),
        "lines_gt": len(gt_lines),
        "time_s": round(elapsed, 1),
    })
    print(f"  Folder {folder.name}: CER raw={cer_trocr:.2%}, rules={cer_rule:.2%}, t5={cer_t5:.2%}, accnorm={cer_accnorm:.2%} ({elapsed:.1f}s)")

# Save detailed results
OUT_DIR.mkdir(exist_ok=True)
with open(OUT_DIR / "pipeline_results.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)
print(f"\nSaved: {OUT_DIR / 'pipeline_results.json'}")

# Save per-folder summary CSV
with open(OUT_DIR / "per_folder_metrics.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=per_folder_metrics[0].keys())
    writer.writeheader()
    writer.writerows(per_folder_metrics)
print(f"Saved: {OUT_DIR / 'per_folder_metrics.csv'}")

# Compute and save aggregate summary
total_trocr_ed = sum(sum(editdistance.eval(l["trocr_prediction"], l["ground_truth"]) for l in r["lines"]) for r in all_results)
total_rule_ed = sum(sum(editdistance.eval(l["rule_corrected"], l["ground_truth"]) for l in r["lines"]) for r in all_results)
total_t5_ed = sum(sum(editdistance.eval(l["t5_corrected"], l["ground_truth"]) for l in r["lines"]) for r in all_results)
total_len = sum(sum(len(l["ground_truth"]) for l in r["lines"]) for r in all_results)
total_accnorm_ed = sum(sum(editdistance.eval(normalize_for_cer(l["rule_corrected"]), normalize_for_cer(l["ground_truth"])) for l in r["lines"]) for r in all_results)
total_accnorm_len = sum(sum(len(normalize_for_cer(l["ground_truth"])) for l in r["lines"]) for r in all_results)

summary = {
    "total_folders": len(all_results),
    "total_matched_lines": sum(len(r["lines"]) for r in all_results),
    "aggregate_cer_trocr": round(total_trocr_ed / max(total_len, 1), 4),
    "aggregate_cer_rule": round(total_rule_ed / max(total_len, 1), 4),
    "aggregate_cer_t5": round(total_t5_ed / max(total_len, 1), 4),
    "aggregate_cer_accent_normalised": round(total_accnorm_ed / max(total_accnorm_len, 1), 4),
}
with open(OUT_DIR / "aggregate_metrics.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"Saved: {OUT_DIR / 'aggregate_metrics.json'}")

print(f"\n{'='*50}")
print(f"AGGREGATE RESULTS ({summary['total_folders']} folders, {summary['total_matched_lines']} lines)")
print(f"{'='*50}")
print(f"  TrOCR (raw):          {summary['aggregate_cer_trocr']:.2%}")
print(f"  + Rule-based:         {summary['aggregate_cer_rule']:.2%}")
print(f"  + T5 corrector:       {summary['aggregate_cer_t5']:.2%}")
print(f"  Accent-normalised:    {summary['aggregate_cer_accent_normalised']:.2%}")
