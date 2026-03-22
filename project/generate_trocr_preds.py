"""
generate_trocr_preds.py
=======================
Run TrOCR on all train + val line crops and save the real
(ocr_output, ground_truth) pairs for T5 training.

This gives T5 the EXACT error profile of TrOCR, which is far
more valuable than synthetic noise alone.

Also generates rule-corrected pairs so T5 can learn to improve
on rule-based output (its actual input in the pipeline).

Output: data/trocr_val_predictions.json
"""

import csv
import json
import types
from pathlib import Path

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from rule_corrector import SpanishDictionary, apply_rules

# ── Config ────────────────────────────────────────────────────────────────────
TROCR_MODEL_DIR = "models/trocr"
TRAIN_CSV       = "data/train.csv"
VAL_CSV         = "data/val.csv"
OUTPUT_FILE     = "data/trocr_val_predictions.json"
MAX_LEN_TROCR   = 128
BEAM_SIZE_TROCR = 4


def _patch_embed_positions(model):
    """Same patch as eval_t5.py to handle long sequences."""
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load TrOCR
    print("Loading TrOCR…")
    processor = TrOCRProcessor.from_pretrained(TROCR_MODEL_DIR)
    model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL_DIR).to(device).eval()
    _patch_embed_positions(model)
    model.generation_config.max_length = MAX_LEN_TROCR
    model.generation_config.num_beams = BEAM_SIZE_TROCR

    # Load dictionary for rule corrections
    print("Loading dictionary…")
    dictionary = SpanishDictionary()

    # Read all CSVs
    pairs = []
    project_dir = Path(__file__).resolve().parent

    def remap_path(p: str) -> Path:
        """Remap old absolute paths to current project directory."""
        p_path = Path(p)
        # Try the path as-is first
        if p_path.exists():
            return p_path
        # Extract the relative output3/... portion
        parts = p_path.parts
        for i, part in enumerate(parts):
            if part.startswith("output"):
                relative = Path(*parts[i:])
                local = project_dir / relative
                if local.exists():
                    return local
        return p_path  # fallback

    for csv_path in [TRAIN_CSV, VAL_CSV]:
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        print(f"\nProcessing {csv_path}: {len(rows)} lines")

        for i, row in enumerate(rows):
            img_path = remap_path(row["image_path"])
            gt = row["transcription"].strip()
            if not gt or not img_path.exists():
                continue

            try:
                image = Image.open(img_path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt").to(device)
                with torch.no_grad():
                    ids = model.generate(**inputs)
                ocr_output = processor.tokenizer.decode(ids[0], skip_special_tokens=True).strip()

                # Also generate rule-corrected version
                rule_corrected = apply_rules(ocr_output, dictionary)

                pairs.append({
                    "ocr_output": ocr_output,
                    "rule_corrected": rule_corrected,
                    "ground_truth": gt,
                    "image_path": str(img_path),
                })
            except Exception as e:
                print(f"  SKIP {img_path}: {e}")
                continue

            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(rows)}")

    print(f"\nTotal pairs: {len(pairs)}")

    # Save
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)
    print(f"Saved → {OUTPUT_FILE}")

    # Quick stats
    n_different = sum(1 for p in pairs if p["ocr_output"] != p["ground_truth"])
    n_rule_helped = sum(1 for p in pairs if p["rule_corrected"] != p["ocr_output"])
    print(f"  Lines where TrOCR ≠ GT   : {n_different}/{len(pairs)}")
    print(f"  Lines where rules changed : {n_rule_helped}/{len(pairs)}")


if __name__ == "__main__":
    main()
