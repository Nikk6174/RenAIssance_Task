"""
train_t5.py
===========
Fine-tunes a Spanish T5 spellchecker as the LLM late-stage correction
step in the OCR pipeline.

Optimised for: RTX 3050 Ti (4 GB VRAM), runs in ~15–20 min after TrOCR.

Accuracy improvements vs baseline
----------------------------------
1. More epochs (6 instead of 3)  — T5 is small and fast; 6 epochs costs
   only ~15 extra minutes but improves correction quality noticeably.

2. Heavier synthetic noise  — SYNTH_AUGMENT_N raised from 5 to 10,
   giving the model more diverse error patterns to learn from.

3. More noise types  — added ocr_specific_errors() which replicates
   common TrOCR mistakes on Spanish historical text:
     u/n confusion, long-s (ſ → f/s), v/b confusion, dropped accents,
     merged/split words, repeated characters.

4. Word-level targeted noise — introduce_errors_in_word() from the
   Kaggle notebook applies 1–2 substitutions per word (accent drops,
   visually-similar char swaps). Interleaved with sentence-level noise
   for broader coverage.

5. "correct: " input prefix  — following the notebook's preprocess_function,
   all source sequences are prefixed with "correct: " so the T5 encoder
   sees a clear task signal.

6. Cosine LR + warmup  — same schedule as TrOCR for consistency.

7. Label smoothing (0.1)  — same regularisation as TrOCR.

Architecture
------------
T5 Seq2Seq: "correct: <noisy OCR text>" → corrected text
Pre-trained on Spanish Wikipedia/medical spelling correction,
fine-tuned on (TrOCR-output → ground-truth) pairs from your own data.

Estimated time: ~15–20 min on RTX 3050 Ti.

Inference
---------
Two ready-to-use corrector classes are provided at the bottom:
  • WordLevelCorrector  — processes each word with a rolling context
                          window (good for very short MAX_LEN models).
  • UniversalSpellingCorrector — marks the target word with brackets
                          inside the context window (more precise).

Evaluation / Analysis
---------------------
After training, use the helper functions at the end of this file to:
  • compute_all_metrics()   — CER, WER, cosine, BLEU, Jaccard
  • get_text_changes()      — diff original vs corrected at token level
  • print_changes()         — colour-coded diff output
  • categorize_changes()    — split diffs into surface-form vs orthographic
"""

import csv
import difflib
import json
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
import evaluate

# ── Config ────────────────────────────────────────────────────────────────────
BASE_MODEL       = "jorgeortizfuentes/spanish-spellchecker-t5-base-wiki200000"
TROCR_PREDS_FILE = "data/trocr_val_predictions.json"
TRAIN_CSV        = "data/train.csv"
OUTPUT_DIR       = "models/t5"

EPOCHS           = 10     # more data + cleaner noise = more epochs beneficial
BATCH_SIZE       = 4
GRAD_ACCUM       = 8
LR               = 3e-5
WARMUP_STEPS     = 50
WEIGHT_DECAY     = 0.01
LABEL_SMOOTHING  = 0.1
MAX_LEN          = 128
SYNTH_AUGMENT_N  = 5      # reduced — real TrOCR data is now the primary source
RANDOM_SEED      = 42
EARLY_STOP_PAT   = 3

# Input prefix used in the notebook's preprocess_function.
# Prepended to every source sequence so the T5 encoder sees a task signal.
TASK_PREFIX      = "correct: "
# ─────────────────────────────────────────────────────────────────────────────

random.seed(RANDOM_SEED)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Noise functions
# ══════════════════════════════════════════════════════════════════════════════

def introduce_errors_in_word(word: str) -> str:
    """
    Word-level targeted noise from the Kaggle notebook.

    Introduces 1 or 2 errors into a single word by modifying random
    character positions.  Mirrors the kinds of mistakes a scanner /
    early OCR pass makes on historical Spanish typefaces:
      - Dropped/swapped accented vowels  (á → a, etc.)
      - Visually similar character pairs  (u → v, m → rn, d → cl, …)
      - Random letter substitution for everything else
    """
    WORD_SUBS = {
        "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ü": "u", "ñ": "n",
        "l": "1", "o": "0", "z": "2", "s": "5", "B": "8",
        "u": "v", "m": "rn", "n": "h", "c": "e", "d": "cl", "g": "q",
    }
    if not word:
        return word

    num_errors = random.choice([1, 2]) if len(word) >= 2 else 1
    indices    = random.sample(range(len(word)), min(num_errors, len(word)))
    chars      = list(word)

    for idx in indices:
        ch = chars[idx]
        if ch in WORD_SUBS:
            chars[idx] = WORD_SUBS[ch]
        else:
            letters  = "abcdefghijklmnopqrstuvwxyz"
            if ch.isupper():
                letters = letters.upper()
            choices = [c for c in letters if c != ch]
            chars[idx] = random.choice(choices)

    return "".join(chars)


def introduce_errors_in_sentence(text: str) -> str:
    """
    Apply introduce_errors_in_word to every token in a sentence,
    preserving whitespace structure.  Complements the sentence-level
    noise functions below.
    """
    words  = text.split(" ")
    noisy  = [introduce_errors_in_word(w) if w else w for w in words]
    return " ".join(noisy)


def ocr_specific_errors(text: str) -> str:
    """
    Simulate errors specific to TrOCR on 17th-century Spanish printed text:
      - Long-s (ſ) misread as f or s
      - u/n confusion (very common in this era's typefaces)
      - v/b confusion
      - Dropped or added diacritics (á → a, e → é)
      - Occasional word merging (space dropped)
      - Character repetition (doubling)
    """
    SUBS = [
        ("u", "n"), ("n", "u"),
        ("v", "b"), ("b", "v"),
        ("f", "s"), ("s", "f"),
        ("á", "a"), ("é", "e"), ("í", "i"), ("ó", "o"), ("ú", "u"),
        ("a", "á"), ("e", "é"),
        ("ñ", "n"), ("n", "ñ"),
        ("ll", "l"), ("l", "ll"),
        ("rr", "r"),
    ]
    out = []
    i   = 0
    while i < len(text):
        applied = False
        for src, tgt in SUBS:
            if text[i : i + len(src)] == src and random.random() < 0.12:
                out.append(tgt)
                i      += len(src)
                applied = True
                break
        if not applied:
            ch = text[i]
            if random.random() < 0.03:           # character doubling
                out.append(ch)
                out.append(ch)
            elif ch == " " and random.random() < 0.04:
                pass                             # drop space → merge words
            else:
                out.append(ch)
            i += 1
    return "".join(out)


def generic_perturb(text: str, rate: float = 0.03) -> str:
    """Generic character-level perturbation (insert / delete / substitute)."""
    CONFUSABLES = {
        "o": "0", "0": "o", "l": "1", "1": "l", "i": "1",
        "c": "e", "e": "c", "m": "rn",
    }
    result = []
    i      = 0
    chars  = list(text)
    while i < len(chars):
        c = chars[i]
        r = random.random()
        if r < rate / 4:
            pass                                               # deletion
        elif r < rate / 2:
            result.append(c)
            result.append(random.choice("abcdefghijklmnoprstuvwxyz"))
        elif r < rate * 0.75 and c.lower() in CONFUSABLES:
            sub = CONFUSABLES[c.lower()]
            result.append(sub if c.islower() else sub.upper())
        elif r < rate and i + 1 < len(chars):
            result.append(chars[i + 1])                        # transposition
            result.append(c)
            i += 2
            continue
        else:
            result.append(c)
        i += 1
    return "".join(result)


def make_noisy(text: str) -> str:
    """
    Apply ONE random noise function per sample (not all 3 stacked).
    This produces more realistic, moderate errors that match actual
    TrOCR output better than heavily-garbled text.
    """
    choice = random.random()
    if choice < 0.33:
        return introduce_errors_in_sentence(text)
    elif choice < 0.66:
        return ocr_specific_errors(text)
    else:
        return generic_perturb(text)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Build training pairs
# ══════════════════════════════════════════════════════════════════════════════

def read_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_training_pairs() -> list[dict]:
    """
    Merge two data sources:
      1. Real TrOCR errors on val set  (highest signal — exact error profile)
      2. Synthetic noise on train set  (broadens coverage)

    Every source string is prefixed with TASK_PREFIX ("correct: ") so the
    tokeniser and model both see the same task signal at training and
    inference time.
    """
    pairs = []

    # ── Real TrOCR prediction errors ─────────────────────────────────────────
    real_count = 0
    rule_count = 0
    if Path(TROCR_PREDS_FILE).exists():
        with open(TROCR_PREDS_FILE, encoding="utf-8") as f:
            real = json.load(f)
        for item in real:
            ocr  = item.get("ocr_output",   "").strip()
            gold = item.get("ground_truth", "").strip()
            rule = item.get("rule_corrected", ocr).strip()
            if ocr and gold and ocr != gold:
                # Pair 1: raw OCR → ground truth
                pairs.append({
                    "source": TASK_PREFIX + ocr,
                    "target": gold,
                })
                real_count += 1
            if rule and gold and rule != gold:
                # Pair 2: rule-corrected → ground truth
                # This is T5's actual input in the pipeline!
                pairs.append({
                    "source": TASK_PREFIX + rule,
                    "target": gold,
                })
                rule_count += 1
        print(f"  Real TrOCR error pairs     : {real_count}")
        print(f"  Rule-corrected → GT pairs  : {rule_count}")
    else:
        print(f"  WARNING: {TROCR_PREDS_FILE} not found.")
        print(f"           Run generate_trocr_preds.py first!")

    # ── Synthetic noisy pairs ────────────────────────────────────────────────
    train_rows  = read_csv(TRAIN_CSV)
    synth_count = 0
    for row in train_rows:
        gold = row["transcription"].strip()
        if not gold:
            continue
        for _ in range(SYNTH_AUGMENT_N):
            noisy = make_noisy(gold)
            if noisy != gold:
                pairs.append({
                    "source": TASK_PREFIX + noisy,
                    "target": gold,
                })
                synth_count += 1

    # ── Identity pairs (teach T5 not to break correct text) ──────────────────
    identity_count = 0
    for row in train_rows:
        gold = row["transcription"].strip()
        if gold:
            pairs.append({
                "source": TASK_PREFIX + gold,
                "target": gold,
            })
            identity_count += 1

    print(f"  Synthetic noise pairs      : {synth_count}")
    print(f"  Identity (don't break) pairs: {identity_count}")
    print(f"  Total pairs                : {len(pairs)}")
    return pairs


# ══════════════════════════════════════════════════════════════════════════════
# 3. Metrics
# ══════════════════════════════════════════════════════════════════════════════

cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")


def make_compute_metrics(tokenizer):
    """Returns the compute_metrics callback used by Seq2SeqTrainer."""
    def compute_metrics(pred):
        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        pred_ids = pred.predictions.copy()
        pred_ids = np.where(pred_ids < 0, tokenizer.pad_token_id, pred_ids)

        pred_str  = tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {
            "cer": round(cer_metric.compute(predictions=pred_str, references=label_str), 4),
            "wer": round(wer_metric.compute(predictions=pred_str, references=label_str), 4),
        }
    return compute_metrics


# ── Extended metrics (from CORRECTIONS_LLM notebook) ─────────────────────────

def _cer_editdistance(text1: str, text2: str) -> float:
    """Character Error Rate via edit distance (no external library)."""
    t1 = text1.replace(" ", "").replace("\n", "")
    t2 = text2.replace(" ", "").replace("\n", "")
    if not t1 and not t2:
        return 0.0
    # Simple DP edit distance
    dp = list(range(len(t2) + 1))
    for i, c1 in enumerate(t1):
        new_dp = [i + 1]
        for j, c2 in enumerate(t2):
            new_dp.append(min(dp[j] + (c1 != c2), dp[j + 1] + 1, new_dp[-1] + 1))
        dp = new_dp
    return dp[-1] / max(len(t1), len(t2))


def _wer_editdistance(text1: str, text2: str) -> float:
    """Word Error Rate via edit distance."""
    w1 = text1.split()
    w2 = text2.split()
    if not w1 and not w2:
        return 0.0
    dp = list(range(len(w2) + 1))
    for i, word1 in enumerate(w1):
        new_dp = [i + 1]
        for j, word2 in enumerate(w2):
            new_dp.append(min(dp[j] + (word1 != word2), dp[j + 1] + 1, new_dp[-1] + 1))
        dp = new_dp
    return dp[-1] / max(len(w1), len(w2))


def _cosine_similarity(text1: str, text2: str) -> float:
    """Cosine similarity over bag-of-words."""
    vocab  = list(set(text1.split()) | set(text2.split()))
    if not vocab:
        return 1.0
    idx    = {w: i for i, w in enumerate(vocab)}
    v1     = np.zeros(len(vocab))
    v2     = np.zeros(len(vocab))
    for w in text1.split():
        v1[idx[w]] += 1
    for w in text2.split():
        v2[idx[w]] += 1
    denom  = np.linalg.norm(v1) * np.linalg.norm(v2)
    return float(np.dot(v1, v2) / denom) if denom else 0.0


def _bleu_score(hypothesis: str, reference: str) -> float:
    """
    Sentence BLEU (1–4 gram) with add-1 smoothing.
    No external NLTK dependency needed.
    """
    from collections import Counter
    import math

    hyp  = hypothesis.split()
    ref  = reference.split()
    if not hyp or not ref:
        return 0.0

    score = 0.0
    for n in range(1, 5):
        hyp_ngrams = Counter(tuple(hyp[i : i + n]) for i in range(len(hyp) - n + 1))
        ref_ngrams = Counter(tuple(ref[i : i + n]) for i in range(len(ref) - n + 1))
        match      = sum(min(c, ref_ngrams[g]) for g, c in hyp_ngrams.items())
        total      = max(sum(hyp_ngrams.values()), 1)
        score     += math.log((match + 1) / (total + 1))

    bp    = min(1.0, math.exp(1 - len(ref) / max(len(hyp), 1)))
    return bp * math.exp(score / 4)


def _jaccard_similarity(text1: str, text2: str) -> float:
    s1  = set(text1.split())
    s2  = set(text2.split())
    uni = s1 | s2
    return len(s1 & s2) / len(uni) if uni else 0.0


def compute_all_metrics(original: str, corrected: str) -> dict:
    """
    Compute the full metric suite used in the CORRECTIONS_LLM notebook.

    Parameters
    ----------
    original  : text before T5 correction  (e.g. raw TrOCR output)
    corrected : text after  T5 correction  (model output or ground truth)

    Returns
    -------
    dict with keys: cer, wer, cosine_similarity, bleu, jaccard
    """
    return {
        "cer":               round(_cer_editdistance(original, corrected), 4),
        "wer":               round(_wer_editdistance(original, corrected), 4),
        "cosine_similarity": round(_cosine_similarity(original, corrected), 4),
        "bleu":              round(_bleu_score(original, corrected),        4),
        "jaccard":           round(_jaccard_similarity(original, corrected), 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. Text-change analysis  (from CORRECTIONS_LLM notebook)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TextChange:
    original:   str
    corrected:  str
    start:      int
    end:        int
    context:    str
    is_ocr_error: bool


def _simple_tokenize(text: str):
    """
    Tokenize text into (word, start, end) triples using a simple regex,
    avoiding the NLTK dependency from the notebook.
    """
    tokens = []
    spans  = []
    for m in re.finditer(r"\S+", text):
        tokens.append(m.group())
        spans.append((m.start(), m.end()))
    return tokens, spans


def get_text_changes(original: str, corrected: str) -> list[TextChange]:
    """
    Identify word-level changes between original and corrected text.

    Uses difflib.SequenceMatcher to find 'replace' opcodes and classifies
    each change as an OCR error (same letters, different accents/case) or
    a real orthographic correction.

    Ported from the CORRECTIONS_LLM notebook's get_text_changes().
    """
    orig_tokens, orig_spans = _simple_tokenize(original)
    corr_tokens, corr_spans = _simple_tokenize(corrected)

    sm      = difflib.SequenceMatcher(None, orig_tokens, corr_tokens, autojunk=False)
    changes = []

    for opcode, a0, a1, b0, b1 in sm.get_opcodes():
        if opcode != "replace":
            continue

        orig_start = orig_spans[a0][0]
        orig_end   = orig_spans[a1 - 1][1] if a1 > a0 else orig_spans[a0][1]
        orig_text  = original[orig_start:orig_end]

        corr_start = corr_spans[b0][0]
        corr_end   = corr_spans[b1 - 1][1] if b1 > b0 else corr_spans[b0][1]
        corr_text  = corrected[corr_start:corr_end]

        # OCR error: same letters once accents and punctuation are stripped
        clean_orig = re.sub(r"\W+", "", orig_text).lower()
        clean_corr = re.sub(r"\W+", "", corr_text).lower()
        is_ocr     = clean_orig == clean_corr

        ctx_start  = max(0, orig_start - 50)
        ctx_end    = min(len(original), orig_end + 50)
        context    = original[ctx_start:ctx_end].replace("\n", " ")

        changes.append(TextChange(
            original    = orig_text.strip(),
            corrected   = corr_text.strip(),
            start       = orig_start,
            end         = orig_end,
            context     = context,
            is_ocr_error= is_ocr,
        ))

    return changes


def print_changes(changes: list[TextChange]) -> None:
    """
    Colour-coded terminal output for a list of TextChange objects.

    Yellow  = OCR surface-form correction  (accent / case only)
    Red     = Real orthographic correction
    """
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    RESET  = "\033[0m"

    for ch in changes:
        color = YELLOW if ch.is_ocr_error else RED
        print(f"{color}{ch.original} → {ch.corrected}{RESET}")
        print(f"  Position : {ch.start}–{ch.end}")
        print(f"  Context  : {ch.context}\n")


def categorize_changes(
    changes:           list[TextChange],
    surface_forms_out: str = "surface_forms.json",
    ortho_errors_out:  str = "orthographic_errors.json",
) -> tuple[dict, list[dict]]:
    """
    Split a list of TextChange objects into:
      • surface_forms      — OCR noise / accent / case variants of the same word
      • orthographic_errors— genuine spelling mistakes

    Saves both to JSON and returns them as Python objects.

    Ported from the CORRECTIONS_LLM notebook's categorize_changes().
    """
    surface_forms      = defaultdict(lambda: defaultdict(int))
    orthographic_errors = []

    for ch in changes:
        normalized = re.sub(r"\W+", "", ch.corrected).lower()
        if ch.is_ocr_error or re.sub(r"\W+", "", ch.original).lower() == normalized:
            key = ch.corrected if ch.is_ocr_error else normalized
            surface_forms[key][ch.original] += 1
        else:
            orthographic_errors.append({
                "original":  ch.original,
                "corrected": ch.corrected,
                "position":  (ch.start, ch.end),
                "context":   ch.context,
            })

    # Serialise defaultdict → plain dict for JSON
    sf_serializable = {k: dict(v) for k, v in surface_forms.items()}

    with open(surface_forms_out, "w", encoding="utf-8") as f:
        json.dump(sf_serializable, f, indent=2, ensure_ascii=False)
    with open(ortho_errors_out, "w", encoding="utf-8") as f:
        json.dump(orthographic_errors, f, indent=2, ensure_ascii=False)

    print(
        f"  Surface forms       : {len(surface_forms)} entries  → {surface_forms_out}\n"
        f"  Orthographic errors : {len(orthographic_errors)} entries → {ortho_errors_out}"
    )
    return sf_serializable, orthographic_errors


# ══════════════════════════════════════════════════════════════════════════════
# 5. Inference classes  (from finetuning-attempt notebook)
# ══════════════════════════════════════════════════════════════════════════════

class WordLevelCorrector:
    """
    Context-aware word-by-word corrector from the Kaggle notebook.

    Processes text token by token, feeding each word together with a
    rolling window of surrounding words.  This lets the model exploit
    local context even though it was trained on full sentences.

    Parameters
    ----------
    model_path     : path to a saved T5 checkpoint (or HF model id)
    context_window : number of preceding words to include as context
    batch_size     : words processed per forward pass
    task_prefix    : must match the prefix used during fine-tuning
    """

    def __init__(
        self,
        model_path:     str = OUTPUT_DIR,
        context_window: int = 3,
        batch_size:     int = 8,
        task_prefix:    str = TASK_PREFIX,
    ):
        self.tokenizer      = AutoTokenizer.from_pretrained(model_path)
        self.model          = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.context_window = context_window
        self.batch_size     = batch_size
        self.task_prefix    = task_prefix

    def correct_text(self, text: str) -> str:
        """Return a corrected version of *text*."""
        original_words  = re.findall(r"\S+|\n", text)
        corrected_words = []

        for i in range(0, len(original_words), self.batch_size):
            batch   = original_words[i : i + self.batch_size]
            inputs  = []
            meta    = []

            for j, word in enumerate(batch):
                position     = i + j
                ctx_start    = max(0, position - self.context_window)
                context_raw  = original_words[ctx_start:position]
                context      = [w for w in context_raw if w != "\n"][-self.context_window:]

                inputs.append(f"{self.task_prefix}{' '.join(context)} {word}".strip())
                meta.append({"original": word, "is_newline": word == "\n"})

            encoded = self.tokenizer(
                inputs,
                padding   = True,
                truncation= True,
                max_length= MAX_LEN,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids      = encoded.input_ids,
                    attention_mask = encoded.attention_mask,
                    max_length     = MAX_LEN,
                    num_beams      = 3,
                    early_stopping = True,
                )

            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for result, m in zip(decoded, meta):
                if m["is_newline"]:
                    corrected_words.append("\n")
                else:
                    corrected_words.extend(result.split() or [m["original"]])

        # Reconstruct with correct spacing
        final = []
        for w in corrected_words:
            if not final or final[-1].endswith("\n") or w == "\n":
                final.append(w)
            else:
                final.append(" " + w)
        return "".join(final)


class UniversalSpellingCorrector:
    """
    Bracket-marking context corrector from the Kaggle notebook.

    Each target word is wrapped in square brackets inside its context
    window, giving the model an explicit pointer to what to fix:
        "correct: previous_words [target_word] following_words"

    This works well when the model has been fine-tuned on sentence-level
    pairs (MAX_LEN ≥ 64) and can exploit wider context.

    Parameters
    ----------
    model_name_or_path : HF model id or local checkpoint directory
    context_window     : words before *and* after the target word
    batch_size         : tokens processed per forward pass
    task_prefix        : set to "" if the model does not use a prefix
    """

    def __init__(
        self,
        model_name_or_path: str = OUTPUT_DIR,
        context_window:     int = 5,
        batch_size:         int = 16,
        task_prefix:        str = TASK_PREFIX,
    ):
        self.tokenizer      = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model          = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
        self.device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.context_window = context_window
        self.batch_size     = batch_size
        self.task_prefix    = task_prefix

    def _build_input(self, tokens: list[str], target_idx: int) -> str:
        start   = max(0, target_idx - self.context_window)
        end     = min(len(tokens), target_idx + self.context_window + 1)
        window  = [
            f"[{t}]" if j == target_idx - start else t
            for j, t in enumerate(tokens[start:end])
        ]
        ctx_str = " ".join(window)
        return f"{self.task_prefix}{ctx_str}" if self.task_prefix else ctx_str

    def correct_text(self, text: str) -> str:
        """Return a corrected version of *text*."""
        # Preserve whitespace tokens for reconstruction
        all_tokens = re.findall(r"\S+|\n+|\s+", text)
        corrected  = []

        for i, token in enumerate(all_tokens):
            if not re.match(r"\w+", token):
                corrected.append(token)
                continue

            word_tokens = [t for t in all_tokens if re.match(r"\S+", t)]
            word_idx    = len([t for t in all_tokens[:i] if re.match(r"\S+", t)])

            input_text = self._build_input(word_tokens, word_idx)

            enc = self.tokenizer(
                input_text,
                return_tensors="pt",
                max_length    = MAX_LEN,
                truncation    = True,
            ).to(self.device)

            with torch.no_grad():
                out = self.model.generate(
                    enc.input_ids,
                    attention_mask = enc.attention_mask,
                    max_length     = 32,
                    num_beams      = 3,
                    early_stopping = True,
                )

            decoded   = self.tokenizer.decode(out[0], skip_special_tokens=True)
            # Notebook extracts correction before the closing bracket
            correction = decoded.split("]")[0].strip()
            corrected.append(correction if correction else token)

        return "".join(corrected)


# ══════════════════════════════════════════════════════════════════════════════
# 6. Main training loop
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  T5 Spellchecker Fine-tuning — OCR post-correction")
    print("=" * 65)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model     = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
    model.train()

    print("\nBuilding training pairs…")
    pairs = build_training_pairs()

    random.shuffle(pairs)
    n_val   = max(1, int(len(pairs) * 0.10))
    val_p   = pairs[:n_val]
    train_p = pairs[n_val:]
    print(f"  Train: {len(train_p)}   Val: {len(val_p)}")

    def tokenize(examples):
        """
        Tokenise source/target pairs.
        Source strings already carry the TASK_PREFIX from build_training_pairs(),
        mirroring the notebook's `preprocess_function` which prepended "correct: ".
        """
        return tokenizer(
            examples["source"],
            text_target=examples["target"],
            max_length  = MAX_LEN,
            truncation  = True,
            padding     = False,
        )

    train_ds = Dataset.from_list(train_p).map(
        tokenize, batched=True, remove_columns=["source", "target"]
    )
    val_ds = Dataset.from_list(val_p).map(
        tokenize, batched=True, remove_columns=["source", "target"]
    )

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir                  = OUTPUT_DIR,
        num_train_epochs            = EPOCHS,
        per_device_train_batch_size = BATCH_SIZE,
        per_device_eval_batch_size  = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUM,
        learning_rate               = LR,
        lr_scheduler_type           = "cosine",
        warmup_steps                = WARMUP_STEPS,
        weight_decay                = WEIGHT_DECAY,
        label_smoothing_factor      = LABEL_SMOOTHING,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        save_total_limit            = 3,
        load_best_model_at_end      = True,
        metric_for_best_model       = "cer",
        greater_is_better           = False,
        predict_with_generate       = True,
        logging_steps               = 20,
        report_to                   = [],
        fp16                        = torch.cuda.is_available(),
        optim                       = "adafactor",
        dataloader_num_workers      = 0,
    )

    trainer = Seq2SeqTrainer(
        model            = model,
        args             = training_args,
        train_dataset    = train_ds,
        eval_dataset     = val_ds,
        data_collator    = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True),
        processing_class = tokenizer,
        compute_metrics  = make_compute_metrics(tokenizer),
        callbacks        = [EarlyStoppingCallback(early_stopping_patience=EARLY_STOP_PAT)],
    )

    print("\nTraining T5…")
    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nModel saved → {OUTPUT_DIR}")

    # ── Final evaluation ──────────────────────────────────────────────────────
    metrics = trainer.evaluate()
    print(f"\n  CER after T5 correction : {metrics.get('eval_cer', 'N/A'):.4f}")
    print(f"  WER after T5 correction : {metrics.get('eval_wer', 'N/A'):.4f}")

    # ── Quick sanity-check with extended metrics ──────────────────────────────
    # Use the first validation pair as a smoke-test.
    sample_noisy = val_p[0]["source"].removeprefix(TASK_PREFIX)
    sample_gold  = val_p[0]["target"]

    corrector        = WordLevelCorrector(model_path=OUTPUT_DIR)
    sample_corrected = corrector.correct_text(sample_noisy)

    print("\n── Sample correction ─────────────────────────────────────────")
    print(f"  Noisy   : {sample_noisy[:120]}")
    print(f"  Gold    : {sample_gold[:120]}")
    print(f"  Model   : {sample_corrected[:120]}")

    ext = compute_all_metrics(sample_corrected, sample_gold)
    print("\n── Extended metrics (model vs gold) ──────────────────────────")
    for k, v in ext.items():
        print(f"  {k:<22}: {v}")

    # ── Change analysis ───────────────────────────────────────────────────────
    changes = get_text_changes(sample_noisy, sample_corrected)
    if changes:
        print(f"\n── Text changes ({len(changes)} found) ───────────────────────────")
        print_changes(changes[:5])   # show first 5 to keep output manageable
        categorize_changes(changes)

    print("\nDone. Next step: python run_ocr.py")


if __name__ == "__main__":
    main()