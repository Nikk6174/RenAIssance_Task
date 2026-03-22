"""
gemini_corrector.py
===================
Gemini-powered post-OCR correction for historical Spanish text.

Drop-in replacement for the T5 correction stage used in run_ocr.py.
Also exposes helpers for change-tracking and error categorisation that
were developed in the CORRECTIONS_LLM notebook.

Public API
----------
  gemini_correct(text, api_key, model, max_retries)  →  str
  get_text_changes(original, corrected)              →  list[TextChange]
  print_changes(changes)
  categorize_changes(changes)                        →  dict
  compute_correction_metrics(original, corrected)    →  dict

Quick usage
-----------
  from gemini_corrector import GeminiCorrector

  corrector = GeminiCorrector(api_key="YOUR_KEY")
  fixed = corrector.correct("vueftra Benignidad profeguid")
"""

from __future__ import annotations

import difflib
import json
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

# ── Optional heavy deps (only needed for metric helpers) ──────────────────────
try:
    import editdistance as _editdistance
    _HAS_EDITDISTANCE = True
except ImportError:
    _HAS_EDITDISTANCE = False

try:
    from sklearn.feature_extraction.text import CountVectorizer as _CV
    import numpy as _np
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

try:
    from nltk.tokenize import WordPunctTokenizer as _WPT
    _HAS_NLTK = True
except ImportError:
    _HAS_NLTK = False

try:
    import google.generativeai as genai
    _HAS_GENAI = True
except ImportError as e:
    import traceback
    traceback.print_exc()
    print(f"ACTUAL IMPORT ERROR: {e}")
    _HAS_GENAI = False
    genai = None  # type: ignore


# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_MODEL      = "gemini-2.5-flash"
DEFAULT_MAX_RETRY  = 5
DEFAULT_BACKOFF    = 15  # seconds base to handle 5 RPM rate limits


# ══════════════════════════════════════════════════════════════════════════════
# Prompt
# ══════════════════════════════════════════════════════════════════════════════

def _build_prompt(text: str) -> str:
    """
    Ultra-conservative OCR post-correction prompt for 17th-century Spanish.

    Key design principles:
    - MINIMAL intervention: only fix clear single-character OCR errors
    - NEVER add, remove, or reorder words
    - NEVER complete fragments or expand abbreviations
    - NEVER change the number of words or tokens
    - Preserve all line-ending hyphens exactly
    - Explicit few-shot examples showing correct vs incorrect behaviour
    """
    return f"""You are a CONSERVATIVE OCR post-corrector for 17th-century Spanish printed text.

CRITICAL RULES — violating ANY of these is a failure:
1. The output MUST have EXACTLY the same number of words as the input.
2. NEVER add new words, remove words, or merge/split words.
3. NEVER complete word fragments (e.g. "tro" must stay "tro", NOT become "Maestro").
4. NEVER expand abbreviations (e.g. "Oc." stays "Oc.", NOT "etcétera").
5. NEVER change hyphens at line endings ("Fran-" stays "Fran-", NOT "Francisco").
6. Only fix CLEAR single-character OCR mistakes within individual words.
7. Preserve ALL original spelling, even if archaic (auia, auer, Escriuano, vuestra, etc.).
8. Return ONLY the corrected text. No explanations, no markdown fences, no quotes.

DOMAIN-SPECIFIC OCR ERRORS TO FIX:
- ç should always be z (old spelling convention: "cobraçen" → "cobrazen")
- Confused u/n: if OCR wrote "dorntu" and "dorniu" is a valid name, fix it
- Confused f/s ONLY when the result is clearly wrong (e.g. "profeguir" → "proseguir")
- Broken accents: "Señor" is correct, "Senor" may need an accent restored
- Single-character swaps: rn↔m, cl↔d, only if the result is a recognized word

EXAMPLES OF CORRECT BEHAVIOUR:
Input:  "Synodal de los Obisgados de Gerona, Ur-"
Output: "Synodal de los Obispados de Gerona, Ur-"
(Fixed: Obisgados→Obispados. Kept: "Ur-" fragment and comma.)

Input:  "de Gerona, del Consejo de su Magestad, ac-"
Output: "de Gerona, del Consejo de su Magestad, ac-"
(No changes needed. "Magestad" is archaic but correct. "ac-" is a line-end fragment.)

Input:  "no solo que nada contiene contra la Fé, y"
Output: "no solo que nada contiene contra la Fé, y"
(No changes. All words are valid archaic Spanish.)

EXAMPLES OF INCORRECT BEHAVIOUR (do NOT do this):
Input:  "tro que fue de Theologia, Examinador"
WRONG:  "Mro. que fue de Theología, Examinador"  ← added period, changed word
RIGHT:  "tro que fue de Theologia, Examinador"    ← no changes needed

Input:  "cisco Drechos, Canonigo, y Sacristan"
WRONG:  "Francisco Derechos, Canónigo, y Sacristán"  ← expanded fragment, modernised
RIGHT:  "cisco Drechos, Canonigo, y Sacristan"        ← all valid, no change

Input:  "Fran-"
WRONG:  "Francisco"  ← NEVER complete or expand fragments
RIGHT:  "Fran-"      ← keep as-is

Now correct this OCR output. Remember: same number of words, minimal changes only.

OCR text:
{text}

Corrected:"""


# ══════════════════════════════════════════════════════════════════════════════
# Core corrector class
# ══════════════════════════════════════════════════════════════════════════════

class GeminiCorrector:
    """
    Wraps the Gemini generative API for OCR post-correction.

    Parameters
    ----------
    api_key   : Gemini API key. If None, reads GEMINI_API_KEY env var.
    model     : Gemini model name (default: gemini-2.5-pro-exp-03-25).
    max_retries: Number of attempts before giving up on a request.
    """

    def __init__(
        self,
        api_key:     Optional[str] = None,
        model:       str           = DEFAULT_MODEL,
        max_retries: int           = DEFAULT_MAX_RETRY,
    ):
        if not _HAS_GENAI:
            raise ImportError(
                "google-generativeai is not installed.\n"
                "Run:  pip install google-generativeai"
            )

        import os
        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError(
                "No Gemini API key supplied. Pass api_key= or set "
                "the GEMINI_API_KEY environment variable."
            )

        genai.configure(api_key=key)
        self._model_name  = model
        self._max_retries = max_retries
        self._model       = genai.GenerativeModel(model)

    # ── Public interface ──────────────────────────────────────────────────────

    def correct(self, text: str) -> tuple[str, str]:
        """
        Correct *text* via Gemini.

        Returns
        -------
        (corrected_text, status)
        status is "success" or "max_retries_exceeded" or "empty_input" or "word_count_mismatch".
        """
        if not text.strip():
            return text, "empty_input"

        prompt = _build_prompt(text)

        # Use temperature=0 for deterministic, conservative output
        gen_config = genai.GenerationConfig(temperature=0.0, max_output_tokens=256)

        for attempt in range(self._max_retries):
            try:
                response = self._model.generate_content(
                    prompt,
                    generation_config=gen_config,
                )
                if response.candidates and response.text:
                    result = response.text.strip()
                    # Strip any markdown fences that Gemini might add
                    if result.startswith("```"):
                        result = re.sub(r"^```\w*\n?", "", result)
                        result = re.sub(r"\n?```$", "", result).strip()
                    # Remove surrounding quotes if Gemini wrapped the output
                    if result.startswith('"') and result.endswith('"'):
                        result = result[1:-1]
                    
                    # GUARD: reject if word count changed (sign of hallucination)
                    orig_words = text.split()
                    corr_words = result.split()
                    if len(corr_words) != len(orig_words):
                        print(f"  [GeminiCorrector] word count mismatch: "
                              f"{len(orig_words)} → {len(corr_words)}, "
                              f"keeping original.")
                        return text, "word_count_mismatch"
                    
                    # Add baseline throttling for the free tier (~15 RPM)
                    time.sleep(6) 
                    
                    return result, "success"
            
            except Exception as exc:
                # Default backoff
                wait = DEFAULT_BACKOFF * (attempt + 1)
                
                # Try to extract the exact wait time requested by the API
                error_str = str(exc)
                match = re.search(r"Please retry in (\d+\.?\d*)s", error_str)
                if match:
                    api_wait = float(match.group(1))
                    # Wait for the API's requested time + 1 second buffer
                    wait = max(wait, api_wait + 1.0)

                import traceback
                traceback.print_exc()
                print(f"  [GeminiCorrector] attempt {attempt + 1} failed: "
                      f"— retrying in {wait:.2f}s…")
                time.sleep(wait)

        # All retries exhausted — return original text unchanged
        print("  [GeminiCorrector] max retries exceeded; returning original text.")
        return text, "max_retries_exceeded"

    def correct_text(self, text: str) -> str:
        """Convenience wrapper that returns only the corrected string."""
        corrected, _ = self.correct(text)
        return corrected


# ══════════════════════════════════════════════════════════════════════════════
# Change-tracking (ported from notebook)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TextChange:
    original:    str
    corrected:   str
    start:       int
    end:         int
    context:     str
    is_ocr_error: bool   # True  → likely OCR glyph confusion (normalised forms match)
                          # False → semantic / orthographic change


def get_text_changes(original: str, corrected: str) -> list[TextChange]:
    """
    Diff *original* and *corrected* at token level and return a list of
    TextChange objects describing every substitution.

    Insertions and deletions are ignored (only replacements tracked).
    """
    if _HAS_NLTK:
        tokenizer   = _WPT()
        orig_tokens = tokenizer.tokenize(original)
        orig_spans  = list(tokenizer.span_tokenize(original))
        corr_tokens = tokenizer.tokenize(corrected)
        corr_spans  = list(tokenizer.span_tokenize(corrected))
    else:
        # Fallback: naive whitespace split (less accurate for punctuation)
        orig_tokens = original.split()
        orig_spans  = _whitespace_spans(original)
        corr_tokens = corrected.split()
        corr_spans  = _whitespace_spans(corrected)

    sm      = difflib.SequenceMatcher(None, orig_tokens, corr_tokens)
    changes = []

    for opcode, a0, a1, b0, b1 in sm.get_opcodes():
        if opcode != "replace":
            continue

        orig_start = orig_spans[a0][0]
        orig_end   = orig_spans[a1 - 1][1] if a1 > 0 else 0
        orig_text  = original[orig_start:orig_end]

        corr_start = corr_spans[b0][0]
        corr_end   = corr_spans[b1 - 1][1] if b1 > 0 else 0
        corr_text  = corrected[corr_start:corr_end]

        clean_orig = re.sub(r"\W+", "", orig_text).lower()
        clean_corr = re.sub(r"\W+", "", corr_text).lower()
        is_ocr     = clean_orig == clean_corr   # same alphanum content → OCR glyph swap

        ctx_start = max(0, orig_start - 50)
        ctx_end   = min(len(original), orig_end + 50)
        context   = original[ctx_start:ctx_end].replace("\n", " ")

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
    """ANSI colour-coded display of changes (yellow = OCR fix, red = semantic)."""
    for c in changes:
        color = "\033[93m" if c.is_ocr_error else "\033[91m"
        reset = "\033[0m"
        print(f"{color}{c.original} → {c.corrected}{reset}")
        print(f"  Position : {c.start}–{c.end}")
        print(f"  Context  : {c.context}\n")


def categorize_changes(changes: list[TextChange]) -> dict:
    """
    Split changes into two buckets:

    surface_forms      — same word, different encoding/glyph (OCR artefacts)
    orthographic_errors — genuinely different words / semantic fixes
    """
    surface_forms       = defaultdict(lambda: defaultdict(int))
    orthographic_errors = []

    for c in changes:
        if c.is_ocr_error:
            surface_forms[c.corrected][c.original] += 1
        else:
            orthographic_errors.append({
                "original" : c.original,
                "corrected": c.corrected,
                "position" : (c.start, c.end),
                "context"  : c.context,
            })

    return {
        "surface_forms":       dict(surface_forms),
        "orthographic_errors": orthographic_errors,
    }


def save_changes(changes: list[TextChange], path: str | Path = "text_changes.json") -> None:
    """Persist change list to JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(c) for c in changes], f, indent=2, ensure_ascii=False)
    print(f"  Changes saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# Metrics (optional — require editdistance / sklearn)
# ══════════════════════════════════════════════════════════════════════════════

def compute_correction_metrics(text1: str, text2: str) -> dict:
    """
    Compute CER, WER, cosine similarity between *text1* and *text2*.

    Returns a dict; keys that couldn't be computed are set to None.
    """
    result: dict = {}

    # CER
    if _HAS_EDITDISTANCE:
        t1 = text1.replace(" ", "").replace("\n", "")
        t2 = text2.replace(" ", "").replace("\n", "")
        dist       = _editdistance.eval(t1, t2)
        result["CER"] = round(dist / max(len(t1), len(t2), 1), 4)

        w1 = text1.split()
        w2 = text2.split()
        wdist      = _editdistance.eval(w1, w2)
        result["WER"] = round(wdist / max(len(w1), len(w2), 1), 4)
    else:
        result["CER"] = None
        result["WER"] = None

    # Cosine similarity
    if _HAS_SKLEARN:
        vecs = _CV().fit_transform([text1, text2]).toarray()
        denom = _np.linalg.norm(vecs[0]) * _np.linalg.norm(vecs[1])
        result["cosine_similarity"] = round(
            float(_np.dot(vecs[0], vecs[1]) / denom) if denom else 0.0, 4
        )
    else:
        result["cosine_similarity"] = None

    return result


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _whitespace_spans(text: str) -> list[tuple[int, int]]:
    """Return (start, end) tuples for each whitespace-split token."""
    spans = []
    i = 0
    for word in text.split():
        start = text.index(word, i)
        end   = start + len(word)
        spans.append((start, end))
        i = end
    return spans


# ══════════════════════════════════════════════════════════════════════════════
# CLI smoke-test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    import sys

    sample = (
        "vueftra  Benignidad,  profeguid\n"
        "o  Niño  tierno, y  Dios Eterno,\n"
        "profeguid en bendecirles, y favo-\n"
        "recerles.  Sean tan fervorofamen\n"
    )

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY environment variable to run the smoke-test.")
        sys.exit(1)

    corrector = GeminiCorrector(api_key=api_key)
    corrected, status = corrector.correct(sample)

    print("═" * 55)
    print("Original:\n", sample)
    print("═" * 55)
    print("Corrected:\n", corrected)
    print(f"\nStatus: {status}")

    changes = get_text_changes(sample, corrected)
    print(f"\n{len(changes)} change(s) detected:")
    print_changes(changes)

    cats = categorize_changes(changes)
    print(f"  Surface-form fixes  : {len(cats['surface_forms'])}")
    print(f"  Orthographic fixes  : {len(cats['orthographic_errors'])}")