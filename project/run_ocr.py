

import argparse
import csv
import json
import os
import types
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)
import evaluate

from gemini_corrector import GeminiCorrector, get_text_changes, categorize_changes

# ── Rule-based corrector ─────────────────────────────────────────────────────
from rule_corrector import SpanishDictionary, apply_rules, normalize_for_cer


TROCR_MODEL_DIR = "models/trocr"
VAL_CSV         = "data/val.csv"
MAX_LEN         = 128
BEAM_SIZE       = 4
GEMINI_MODEL    = "gemini-2.5-flash"


cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")




def _resolve_api_key(cli_key: str | None) -> str:
    """
    Priority:
      1. --api_key CLI argument
      2. GEMINI_API_KEY env var
      3. GOOGLE_API_KEY env var
      4. gemini.env file in the current directory
    """
    if cli_key:
        return cli_key

    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        val = os.getenv(var)
        if val:
            return val

    # Try loading from gemini.env
    env_file = Path("gemini.env")
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("GEMINI_API_KEY="):
                return line.split("=", 1)[1].strip()

    raise RuntimeError(
        "No Gemini API key found.\n"
        "  • Pass --api_key YOUR_KEY, or\n"
        "  • Set the GEMINI_API_KEY environment variable, or\n"
        "  • Create a gemini.env file with: GEMINI_API_KEY=your_key"
    )



def _patch_embed_positions(model: VisionEncoderDecoderModel) -> None:
    """
    Fix TrOCRLearnedPositionalEmbedding / TrOCRSinusoidalPositionalEmbedding
    device mismatch that occurs when the model is moved to CUDA.

    The plain `_float_tensor` attribute used for device detection in the
    original code is not a registered buffer, so .to() never moves it.
    The patched forward() reads the device from self.weight instead.
    """
    embed_pos = model.decoder.model.decoder.embed_positions

    if type(embed_pos).__name__ == "TrOCRLearnedPositionalEmbedding":
        def _forward_learned(self, input_ids_shape: torch.Size,
                             past_key_values_length: int = 0):
            bsz, seq_len = input_ids_shape
            positions = torch.arange(
                past_key_values_length,
                past_key_values_length + seq_len,
                dtype=torch.long,
                device=self.weight.device,
            )
            return torch.nn.Embedding.forward(self, positions)
        embed_pos.forward = types.MethodType(_forward_learned, embed_pos)

    elif type(embed_pos).__name__ == "TrOCRSinusoidalPositionalEmbedding":
        def _forward_sinusoidal(self, input_ids: torch.Tensor,
                                past_key_values_length: int = 0):
            bsz, seq_len = input_ids.size()
            position_ids = self.create_position_ids_from_input_ids(
                input_ids, self.padding_idx, past_key_values_length
            ).to(input_ids.device)
            max_pos = self.padding_idx + 1 + seq_len
            if (self.weights is None
                    or max_pos > self.weights.size(0)
                    or self.weights.device != input_ids.device):
                self.weights = self.get_embedding(
                    max_pos, self.embedding_dim, self.padding_idx
                ).to(input_ids.device)
            x = (self.weights
                     .index_select(0, position_ids.view(-1))
                     .view(bsz, seq_len, -1)
                     .detach())
            return x
        embed_pos.forward = types.MethodType(_forward_sinusoidal, embed_pos)


def load_trocr(model_dir: str, device):
    print(f"Loading TrOCR from {model_dir}…")
    processor = TrOCRProcessor.from_pretrained(model_dir)
    model     = VisionEncoderDecoderModel.from_pretrained(model_dir).to(device).eval()
    _patch_embed_positions(model)
    model.generation_config.max_length = MAX_LEN
    model.generation_config.num_beams  = BEAM_SIZE
    return processor, model




def trocr_predict(processor, model, image_path: str, device) -> str:
    image  = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        ids = model.generate(**inputs)
    return processor.tokenizer.decode(ids[0], skip_special_tokens=True).strip()


def compute_scores(preds: list[str], refs: list[str]) -> dict:
    cer = cer_metric.compute(predictions=preds, references=refs)
    wer = wer_metric.compute(predictions=preds, references=refs)
    return {"CER": round(cer, 4), "WER": round(wer, 4)}




def run_evaluate(device, api_key: str):
    print("\n" + "=" * 60)
    print("  EVALUATION MODE  (TrOCR → Gemini correction)")
    print("=" * 60)

    
    trocr_proc, trocr_model = load_trocr(TROCR_MODEL_DIR, device)
    corrector               = GeminiCorrector(api_key=api_key, model=GEMINI_MODEL)

    
    with open(VAL_CSV, newline="", encoding="utf-8") as f:
        val_rows = list(csv.DictReader(f))[:19]
    print(f"\nEvaluating on {len(val_rows)} val lines…")

    
    print("Loading Spanish dictionary for rule-based corrections…")
    dictionary = SpanishDictionary()

    raw_preds, rule_preds, corrected_preds, references = [], [], [], []
    results = []

    for i, row in enumerate(val_rows):
        gt  = row["transcription"]

        # Stage 1 — TrOCR
        raw = trocr_predict(trocr_proc, trocr_model, row["image_path"], device)

        # Stage 2 — Rule-based correction (deterministic)
        ruled = apply_rules(raw, dictionary)

        # Stage 3 — Gemini correction (on rule-corrected text)
        corr, status = corrector.correct(ruled)
        
        import time
        time.sleep(13)  # Respect the 5 RPM rate limit
        
        raw_preds.append(raw)
        rule_preds.append(ruled)
        corrected_preds.append(corr)
        references.append(gt)

        # Optional: track what Gemini changed
        changes  = get_text_changes(ruled, corr)
        cats     = categorize_changes(changes)

        results.append({
            "image"            : row["image_path"],
            "gt"               : gt,
            "trocr"            : raw,
            "after_rules"      : ruled,
            "corrected"        : corr,
            "gemini_status"    : status,
            "n_changes"        : len(changes),
            "n_ocr_fixes"      : len(cats["surface_forms"]),
            "n_semantic_fixes" : len(cats["orthographic_errors"]),
        })

        if (i + 1) % 10 == 0 or i == len(val_rows) - 1:
            print(f"  {i + 1}/{len(val_rows)}")

    
    raw_scores   = compute_scores(raw_preds,       references)
    rule_scores  = compute_scores(rule_preds,      references)
    corr_scores  = compute_scores(corrected_preds, references)

    
    norm_preds = [normalize_for_cer(p) for p in corrected_preds]
    norm_refs  = [normalize_for_cer(r) for r in references]
    norm_cer   = round(cer_metric.compute(predictions=norm_preds, references=norm_refs), 4)

    print("\n" + "─" * 60)
    print("  Results summary")
    print("─" * 60)
    print(f"  Stage                    │  CER    │  WER")
    print(f"  ─────────────────────────┼─────────┼─────────")
    print(f"  TrOCR (raw)              │ {raw_scores['CER']:.4f}  │ {raw_scores['WER']:.4f}")
    print(f"  + Rule-based fixes       │ {rule_scores['CER']:.4f}  │ {rule_scores['WER']:.4f}")
    print(f"  + Gemini correction      │ {corr_scores['CER']:.4f}  │ {corr_scores['WER']:.4f}")
    print(f"  (accent-normalised CER)  │ {norm_cer:.4f}  │  —")
    cer_delta = raw_scores["CER"] - corr_scores["CER"]
    wer_delta = raw_scores["WER"] - corr_scores["WER"]
    print(f"  Total improvement        │ {cer_delta:+.4f}  │ {wer_delta:+.4f}")
    print("─" * 60)
    print("  CER = Character Error Rate (lower is better)")
    print("  WER = Word Error Rate      (lower is better)")

   
    print("\n  Sample predictions (first 5):")
    for r in results[:5]:
        print(f"    GT        : {r['gt']}")
        print(f"    TrOCR     : {r['trocr']}")
        print(f"    Rules     : {r['after_rules']}")
        print(f"    Gemini    : {r['corrected']}")
        print(f"    Changes   : {r['n_changes']} "
              f"(OCR={r['n_ocr_fixes']}, semantic={r['n_semantic_fixes']})")
        print()

    
    out_path = Path("data/eval_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": {
                "trocr_raw"        : raw_scores,
                "after_rules"      : rule_scores,
                "after_gemini"     : corr_scores,
                "accent_norm_cer"  : norm_cer,
            },
            "samples": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"  Full results → {out_path}")




def run_predict(input_dir: str, device, api_key: str):
    print("\n" + "=" * 60)
    print(f"  PREDICT MODE  →  {input_dir}")
    print("  Pipeline: TrOCR → Rule-based → Gemini correction")
    print("=" * 60)

    trocr_proc, trocr_model = load_trocr(TROCR_MODEL_DIR, device)
    corrector               = GeminiCorrector(api_key=api_key, model=GEMINI_MODEL)

    # Load dictionary for rule-based correction
    print("Loading Spanish dictionary for rule-based corrections…")
    dictionary = SpanishDictionary()

    image_extensions = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    images = sorted(
        p for p in Path(input_dir).rglob("*")
        if p.suffix.lower() in image_extensions
    )
    print(f"Found {len(images)} images\n")

    lines    = []
    raw_all  = []
    meta     = []

    for i, img_path in enumerate(images):
        # Stage 1: TrOCR
        raw = trocr_predict(trocr_proc, trocr_model, str(img_path), device)

        # Stage 2: Rule-based correction
        ruled = apply_rules(raw, dictionary)

        # Stage 3: Gemini correction
        corr, status = corrector.correct(ruled)

        changes = get_text_changes(ruled, corr)
        cats    = categorize_changes(changes)

        print(f"  [{i + 1:>3}] {img_path.name:<35}")
        print(f"        TrOCR  : {raw}")
        print(f"        Rules  : {ruled}")
        print(f"        Gemini : {corr}  [{status}, {len(changes)} fix(es)]")

        lines.append(corr)
        raw_all.append(raw)
        meta.append({
            "file"         : str(img_path),
            "trocr"        : raw,
            "after_rules"  : ruled,
            "corrected"    : corr,
            "status"       : status,
            "n_changes"    : len(changes),
            "ocr_fixes"    : len(cats["surface_forms"]),
            "semantic_fixes": len(cats["orthographic_errors"]),
        })

    
    base     = Path(input_dir)
    out_txt  = base / "ocr_output.txt"
    out_raw  = base / "ocr_raw_trocr.txt"
    out_json = base / "ocr_output_meta.json"

    out_txt.write_text("\n".join(lines),   encoding="utf-8")
    out_raw.write_text("\n".join(raw_all), encoding="utf-8")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n  Corrected text   → {out_txt}")
    print(f"  Raw TrOCR text   → {out_raw}")
    print(f"  Per-image meta   → {out_json}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Spanish OCR pipeline: TrOCR → Gemini correction"
    )
    parser.add_argument(
        "--mode", choices=["evaluate", "predict"], default="evaluate"
    )
    parser.add_argument(
        "--input_dir", default=None,
        help="Folder of images (predict mode only)"
    )
    parser.add_argument(
        "--api_key", default=None,
        help="Gemini API key (overrides env var / gemini.env)"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    api_key = _resolve_api_key(args.api_key)
    print(f"Gemini : {GEMINI_MODEL}")

    if args.mode == "evaluate":
        run_evaluate(device, api_key)
    else:
        if not args.input_dir:
            print("ERROR: --input_dir is required for predict mode")
        else:
            run_predict(args.input_dir, device, api_key)