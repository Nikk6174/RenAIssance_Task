"""Quick speed benchmark for TrOCR and T5 inference."""
import time, torch
from pathlib import Path
from PIL import Image
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# ── T5 Speed ─────────────────────────────────────────────────────────────────
print("Loading T5...")
from eval_t5 import load_t5
tokenizer, t5_model = load_t5("models/t5", device)

test_sentences = [
    "Compañia de los Indiaf orientales",
    "haçer la guerra a los enemigos",
    "el Rey mando que fe hizieffe",
    "en la ciudad de Sevilla",
    "los navios partieron del puerto",
] * 20  # 100 sentences

print(f"Running T5 on {len(test_sentences)} sentences...")
t0 = time.time()
total_words = 0
for s in test_sentences:
    enc = tokenizer(f"fix: {s}", return_tensors="pt", max_length=128, truncation=True).to(device)
    with torch.no_grad():
        out = t5_model.generate(**enc, max_new_tokens=128)
    tokenizer.decode(out[0], skip_special_tokens=True)
    total_words += len(s.split())
t5_elapsed = time.time() - t0
print(f"T5: {total_words} words in {t5_elapsed:.2f}s = {total_words/t5_elapsed:.0f} words/sec")

# ── TrOCR Speed ──────────────────────────────────────────────────────────────
print("\nLoading TrOCR...")
from run_ocr import load_trocr
processor, trocr_model = load_trocr("models/trocr", device)

# Get some real crop images
input_dir = Path("input")
crops = []
for folder in sorted(input_dir.iterdir()):
    if not folder.is_dir():
        continue
    for img_file in folder.glob("*.jpg"):
        crops.append(str(img_file))
    for img_file in folder.glob("*.png"):
        crops.append(str(img_file))
    if len(crops) >= 5:
        break

# Use the first source image to generate line crops
if crops:
    src = cv2.imread(crops[0])
    from pipeline3 import binarize, horizontal_projection, find_line_segments, crop_line, detect_text_lines
    boxes, stats = detect_text_lines(src)
    pad_v, pad_h = stats["pad_v"], stats["pad_h"]
    pil_crops = []
    for box in boxes:
        crop_bgr, _ = crop_line(src, box["bbox"], pad_v, pad_h)
        pil_crops.append(Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)))
    # Repeat to get ~50+ samples
    pil_crops = (pil_crops * 5)[:50]
    
    print(f"Running TrOCR on {len(pil_crops)} line crops...")
    t0 = time.time()
    total_words = 0
    for pil in pil_crops:
        px = processor(pil, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            ids = trocr_model.generate(px, max_new_tokens=128)
        text = processor.decode(ids[0], skip_special_tokens=True)
        total_words += len(text.split())
    trocr_elapsed = time.time() - t0
    print(f"TrOCR: {total_words} words in {trocr_elapsed:.2f}s = {total_words/trocr_elapsed:.0f} words/sec")
    print(f"TrOCR: {len(pil_crops)} lines in {trocr_elapsed:.2f}s = {len(pil_crops)/trocr_elapsed:.1f} lines/sec")
else:
    print("No input images found for TrOCR benchmark")
