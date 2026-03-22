"""
train_trocr.py
==============
Fine-tunes  qantev/trocr-large-spanish  on the 586 manually verified
line-crop → transcription pairs produced by pipeline3.py.

Optimised for: RTX 3050 Ti (4 GB VRAM), Ryzen 7 6000, 4–6 hour budget.

Key accuracy improvements vs the baseline config
-------------------------------------------------
1. Image augmentation  — critical for only ~470 train lines.
   Each epoch the model sees slightly different brightness, contrast,
   sharpness and noise versions of each crop, multiplying effective
   data and preventing overfitting.

2. 30 epochs instead of 15  — the dataset is tiny; the model needs
   more passes to converge. Early-stopping on CER prevents wasting
   time if it plateaus.

3. Cosine LR schedule  — gradually anneals from 3e-5 down to near
   zero, letting the model fine-tune delicate weights without
   overshooting in later epochs.

4. Label smoothing (0.1)  — mild regularisation that prevents the
   decoder from becoming overconfident on the small training set.

5. Gradient checkpointing  — trades ~30 % more compute for a ~35 %
   VRAM reduction, keeping TrOCR-Large safe on 4 GB.

6. Beam width 6 at inference  — wider search, better accuracy.

Architecture
------------
TrOCR = Vision Encoder (BEiT) + Text Decoder (RoBERTa)
Pre-trained on printed Spanish text, fine-tuned on your historical font.

Evaluation Metrics
------------------
CER (Character Error Rate) = Levenshtein / len(gt)   — primary, target < 5 %
WER (Word Error Rate)      = word-level edits         — secondary

Estimated training time on RTX 3050 Ti
---------------------------------------
  ~470 lines × 30 epochs / batch 2 × ~2.5 s/step  ≈  5 h 20 min
"""

import csv
import json
import random
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    default_data_collator,
)
import evaluate

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL     = "qantev/trocr-large-spanish"
TRAIN_CSV      = "data/train.csv"
VAL_CSV        = "data/val.csv"
OUTPUT_DIR     = "models/trocr"
OCR_PREDS_FILE = "data/trocr_val_predictions.json"

# Hardware-tuned for RTX 3050 Ti (4 GB VRAM)
BATCH_SIZE      = 1      # batch 2 still OOMs on 4GB; 1 is extremely safe
GRAD_ACCUM      = 16     # effective batch = 16
EPOCHS          = 30     # more passes for tiny dataset
LR              = 3e-5
WARMUP_STEPS    = 100
WEIGHT_DECAY    = 0.01
LABEL_SMOOTHING = 0.1    # regularisation against overconfidence
MAX_TARGET_LEN  = 128
BEAM_SIZE       = 6      # wider beam = better accuracy at inference
EARLY_STOP_PAT  = 6      # stop if CER flat for 6 consecutive epochs
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# 1. Augmentation
# ══════════════════════════════════════════════════════════════════════════════

def augment(image: Image.Image, is_train: bool) -> Image.Image:
    """
    Mild photometric augmentations applied only during training.

    Simulates real scan variations: uneven lighting, slight blur from
    page curl, scanner noise, different binarisation thresholds.
    Keeps text readable — augmentations are intentionally subtle.
    """
    if not is_train:
        return image

    # Brightness ±20 %
    if random.random() < 0.6:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.80, 1.20))

    # Contrast ±20 %
    if random.random() < 0.6:
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.80, 1.20))

    # Sharpness: soften or sharpen
    if random.random() < 0.4:
        image = ImageEnhance.Sharpness(image).enhance(random.uniform(0.7, 1.5))

    # Slight blur (simulates out-of-focus scan)
    if random.random() < 0.25:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))

    # Salt-and-pepper noise (simulates old paper speckle)
    if random.random() < 0.3:
        arr   = np.array(image).astype(np.float32)
        arr   = np.clip(arr + np.random.normal(0, 6, arr.shape), 0, 255).astype(np.uint8)
        image = Image.fromarray(arr)

    return image


# ══════════════════════════════════════════════════════════════════════════════
# 2. Dataset
# ══════════════════════════════════════════════════════════════════════════════

def read_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class LineDataset(Dataset):
    def __init__(self, rows: list[dict], processor: TrOCRProcessor,
                 is_train: bool = True):
        self.rows      = rows
        self.processor = processor
        self.is_train  = is_train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row   = self.rows[idx]
        image = augment(Image.open(row["image_path"]).convert("RGB"), self.is_train)

        pixel_values = self.processor(
            images=image, return_tensors="pt"
        ).pixel_values.squeeze(0)

        input_ids = self.processor.tokenizer(
            row["transcription"],
            max_length=MAX_TARGET_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        labels = input_ids.clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        decoder_input_ids = input_ids.clone()
        decoder_input_ids = torch.cat([
            torch.tensor([self.processor.tokenizer.cls_token_id], dtype=torch.long),
            decoder_input_ids[:-1]
        ])

        return {
            "pixel_values": pixel_values,
            "labels": labels,
            "decoder_input_ids": decoder_input_ids,
        }


# ══════════════════════════════════════════════════════════════════════════════
# 3. Metrics
# ══════════════════════════════════════════════════════════════════════════════

cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")


def compute_metrics(pred, processor):
    label_ids = pred.label_ids.copy()
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # Guard against negative sentinel values in beam-search output
    pred_ids = np.where(
        pred.predictions < 0,
        processor.tokenizer.pad_token_id,
        pred.predictions,
    )

    pred_str  = processor.tokenizer.batch_decode(pred_ids,   skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids,  skip_special_tokens=True)

    return {
        "cer": round(cer_metric.compute(predictions=pred_str, references=label_str), 4),
        "wer": round(wer_metric.compute(predictions=pred_str, references=label_str), 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. Positional-embedding device fix
# ══════════════════════════════════════════════════════════════════════════════

def _patch_embed_positions(model: VisionEncoderDecoderModel) -> None:
    """
    TrOCRLearnedPositionalEmbedding.forward() uses a plain Python attribute
    `_float_tensor = torch.zeros(1)` (set in __init__) purely to detect the
    current device.  Because it is NOT a buffer or parameter, it is invisible
    to .to() / .cuda() and never moves with the model.  This causes:

        RuntimeError: Tensor on device meta is not on the expected device cuda:0!

    The fix replaces the forward method with an identical one that reads the
    device from self.weight instead — a real nn.Embedding parameter that
    always lives on the correct device.
    
    For TrOCRSinusoidalPositionalEmbedding, it stores embeddings in a plain
    `self.weights` tensor that doesn't move with the module. We patch it to
    recreate embeddings on the correct device if needed.
    """
    embed_pos = model.decoder.model.decoder.embed_positions

    if type(embed_pos).__name__ == "TrOCRLearnedPositionalEmbedding":
        def _forward_learned(self, input_ids_shape: torch.Size, past_key_values_length: int = 0):
            bsz, seq_len = input_ids_shape
            positions = torch.arange(
                past_key_values_length,
                past_key_values_length + seq_len,
                dtype=torch.long,
                device=self.weight.device,   # ← always correct
            )
            return torch.nn.Embedding.forward(self, positions)
        embed_pos.forward = types.MethodType(_forward_learned, embed_pos)

    elif type(embed_pos).__name__ == "TrOCRSinusoidalPositionalEmbedding":
        def _forward_sinusoidal(self, input_ids: torch.Tensor, past_key_values_length: int = 0):
            bsz, seq_len = input_ids.size()
            position_ids = self.create_position_ids_from_input_ids(
                input_ids, self.padding_idx, past_key_values_length
            ).to(input_ids.device)

            max_pos = self.padding_idx + 1 + seq_len
            if self.weights is None or max_pos > self.weights.size(0) or self.weights.device != input_ids.device:
                self.weights = self.get_embedding(max_pos, self.embedding_dim, self.padding_idx).to(input_ids.device)

            x = self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, -1).detach()
            return x
        embed_pos.forward = types.MethodType(_forward_sinusoidal, embed_pos)



# ══════════════════════════════════════════════════════════════════════════════
# 5. Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  TrOCR Fine-tuning — High-accuracy config (4–6 h budget)")
    print("=" * 65)

    processor = TrOCRProcessor.from_pretrained(BASE_MODEL)
    model     = VisionEncoderDecoderModel.from_pretrained(BASE_MODEL)

    # ── Patch positional embedding before anything else ──────────────────────
    _patch_embed_positions(model)

    # ── Decoder config ───────────────────────────────────────────────────────
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id           = processor.tokenizer.pad_token_id
    model.config.vocab_size             = model.config.decoder.vocab_size
    model.config.eos_token_id           = processor.tokenizer.sep_token_id

    # ── Generation config ────────────────────────────────────────────────────
    model.generation_config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.generation_config.pad_token_id           = processor.tokenizer.pad_token_id
    model.generation_config.eos_token_id           = processor.tokenizer.sep_token_id
    model.generation_config.max_length             = MAX_TARGET_LEN
    model.generation_config.no_repeat_ngram_size   = 3
    model.generation_config.length_penalty         = 2.0
    model.generation_config.num_beams              = BEAM_SIZE

    print("  Gradient checkpointing : ON  (handled by Trainer)")

    train_rows = read_csv(TRAIN_CSV)
    val_rows   = read_csv(VAL_CSV)
    print(f"\n  Train : {len(train_rows)} lines (augmented per epoch)")
    print(f"  Val   : {len(val_rows)} lines (no augmentation)")

    steps_per_epoch = len(train_rows) // BATCH_SIZE
    total_steps     = steps_per_epoch * EPOCHS
    print(f"\n  Steps/epoch : {steps_per_epoch}")
    print(f"  Total steps : {total_steps}")
    print(f"  Estimated   : {total_steps * 2.5 / 3600:.1f} h  (RTX 3050 Ti)")

    train_dataset = LineDataset(train_rows, processor, is_train=True)
    val_dataset   = LineDataset(val_rows,   processor, is_train=False)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir                      = OUTPUT_DIR,
        num_train_epochs                = EPOCHS,
        per_device_train_batch_size     = BATCH_SIZE,
        per_device_eval_batch_size      = BATCH_SIZE,
        gradient_accumulation_steps     = GRAD_ACCUM,
        learning_rate                   = LR,
        lr_scheduler_type               = "cosine",
        warmup_steps                    = WARMUP_STEPS,
        weight_decay                    = WEIGHT_DECAY,
        label_smoothing_factor          = LABEL_SMOOTHING,
        eval_strategy                   = "epoch",
        save_strategy                   = "epoch",
        save_total_limit                = 3,
        load_best_model_at_end          = True,
        metric_for_best_model           = "cer",
        greater_is_better               = False,
        predict_with_generate           = True,
        generation_num_beams            = BEAM_SIZE,
        logging_steps                   = 10,
        report_to                       = [],
        fp16                            = torch.cuda.is_available(),
        optim                           = "adafactor",  # ultra memory-efficient optimizer
        gradient_checkpointing          = True,
        gradient_checkpointing_kwargs   = {"use_reentrant": True},
        dataloader_num_workers          = 0,
    )

    trainer = Seq2SeqTrainer(
        model            = model,
        args             = training_args,
        train_dataset    = train_dataset,
        eval_dataset     = val_dataset,
        data_collator    = default_data_collator,
        compute_metrics  = lambda pred: compute_metrics(pred, processor),
        processing_class = processor.image_processor,
        callbacks        = [EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PAT)],
    )

    print(f"\nTraining  (early stop if CER flat for {EARLY_STOP_PAT} epochs)…\n")
    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"\nBest model saved → {OUTPUT_DIR}")

    metrics = trainer.evaluate()
    cer = metrics.get("eval_cer", float("nan"))
    wer = metrics.get("eval_wer", float("nan"))
    print(f"\n  CER : {cer:.4f}  (target < 0.05)")
    print(f"  WER : {wer:.4f}")
    if   cer < 0.05: print("  ✓ Excellent — below 5 %")
    elif cer < 0.10: print("  ~ Good — T5 correction will push it lower")
    else:            print("  ! Above 10 % — check data quality or add epochs")

    print(f"\nSaving val predictions for T5 → {OCR_PREDS_FILE}")
    _save_predictions(model, processor, val_rows)
    print("Done. Next step: python train_t5.py")


def _save_predictions(model, processor, val_rows):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device).eval()
    preds  = []
    for row in val_rows:
        image  = Image.open(row["image_path"]).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            ids  = model.generate(**inputs, max_length=MAX_TARGET_LEN, num_beams=BEAM_SIZE)
            text = processor.tokenizer.decode(ids[0], skip_special_tokens=True)
        preds.append({"ocr_output": text, "ground_truth": row["transcription"]})

    Path(OCR_PREDS_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(OCR_PREDS_FILE, "w", encoding="utf-8") as f:
        json.dump(preds, f, indent=2, ensure_ascii=False)
    print(f"  {len(preds)} pairs saved")


if __name__ == "__main__":
    main()