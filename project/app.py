# pyre-unsafe
"""
app.py — Streamlit OCR Evaluation App
======================================
Full pipeline per document:
  1. Line detection    (pipeline3)
  2. TrOCR inference   (models/trocr)
  3. Rule-based fixes  (rule_corrector: ç→z, u↔v, f↔s, accents)
  4. T5 correction     (models/t5 — local, optional)
  5. Gemini correction (API — optional, needs key)
  6. Dynamic alignment + multi-stage CER

Launch
------
    cd d:\\audio\\data_generation_kraken\\project
    streamlit run app.py
"""

import io
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

# ── Make sure the project dir is on sys.path so local imports work ────────────
PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from pipeline3 import (
    detect_text_lines,
    crop_line,
    read_transcription,
    find_source_image,
    draw_annotated,
    binarize,
    horizontal_projection,
)
from dynamic_align import (
    align_with_local_window,
    compute_aggregate_cer,
    compute_sequential_cer,
)
from rule_corrector import SpanishDictionary, apply_rules, normalize_for_cer

# ══════════════════════════════════════════════════════════════════════════════
# Page config
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="OCR Alignment Evaluator",
    page_icon="🔍",
    layout="wide",
)

# ══════════════════════════════════════════════════════════════════════════════
# Custom CSS
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 {
        margin: 0; font-size: 1.8rem; font-weight: 700;
    }
    .main-header p {
        margin: 0.3rem 0 0 0; opacity: 0.85; font-size: 0.95rem;
    }

    /* Metric cards */
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #0f3460;
        margin-bottom: 0.5rem;
    }
    .metric-card.improved {
        border-left-color: #28a745;
        background: #f0fff4;
    }
    .metric-card.worse {
        border-left-color: #dc3545;
        background: #fff5f5;
    }
    .metric-card .label { font-size: 0.8rem; color: #666; text-transform: uppercase; }
    .metric-card .value { font-size: 1.6rem; font-weight: 700; color: #1a1a2e; }

    /* Shifted row highlight */
    .shifted-row {
        background-color: #fff3cd !important;
        border-left: 3px solid #ffc107;
    }

    /* Table styling */
    .alignment-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85rem;
    }
    .alignment-table th {
        background: #1a1a2e;
        color: white;
        padding: 0.6rem 0.8rem;
        text-align: left;
    }
    .alignment-table td {
        padding: 0.5rem 0.8rem;
        border-bottom: 1px solid #e9ecef;
        vertical-align: top;
    }
    .alignment-table tr:hover { background: #f1f3f5; }
    .cer-good { color: #28a745; font-weight: 600; }
    .cer-ok   { color: #ffc107; font-weight: 600; }
    .cer-bad  { color: #dc3545; font-weight: 600; }
    .tag-shifted {
        display: inline-block;
        background: #ffc107;
        color: #333;
        padding: 0.1rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .tag-skipped {
        display: inline-block;
        background: #6c757d;
        color: white;
        padding: 0.1rem 0.5rem;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .skipped-row {
        background-color: #f0f0f0 !important;
        color: #999;
    }
    .skipped-row td { font-style: italic; }

    /* Annotated image — fit to screen height */
    .annotated-img-container img {
        max-height: 85vh !important;
        width: auto !important;
        object-fit: contain;
        display: block;
        margin: 0 auto;
    }

    /* Remove extra bottom padding/scroll */
    .block-container { padding-bottom: 1rem !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Model paths — use local folder if present, otherwise download from HF Hub
# ══════════════════════════════════════════════════════════════════════════════

HF_TROCR_REPO = "nikk6174/historical-spanish-trocr"
HF_T5_REPO    = "nikk6174/historical-spanish-t5"
LOCAL_TROCR   = "models/trocr"
LOCAL_T5      = "models/t5"


def _ensure_downloaded(local: str, hf_repo: str, label: str) -> str:
    """Ensure a model is available locally (either in models/ or HF cache).
    Shows download progress in the Streamlit UI only when actually downloading.
    Uses session_state to avoid re-rendering download widgets on reruns."""
    import time

    # 1. Local folder exists — use it directly
    if Path(local).exists():
        return local

    # 2. Check session_state cache (already downloaded this session)
    cache_key = f"_model_path_{hf_repo}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    # 3. Check if already in HF cache (downloaded previously)
    try:
        from huggingface_hub import snapshot_download
        cached_path = snapshot_download(repo_id=hf_repo, local_files_only=True)
        st.session_state[cache_key] = cached_path
        return cached_path
    except Exception:
        pass  # Not in cache — need to download

    # 4. Actually download with progress UI
    from huggingface_hub import list_repo_files, hf_hub_download

    try:
        files = list_repo_files(hf_repo)
        total_files = len(files)
    except Exception:
        total_files = 0
        files = []

    status = st.status(f"⬇️ Downloading {label} from Hugging Face…", expanded=True)
    progress_bar = status.progress(0, text=f"Preparing download of {label}…")

    cache_dir = None
    for i, filename in enumerate(files):
        t0 = time.time()
        try:
            path = hf_hub_download(repo_id=hf_repo, filename=filename)
            if cache_dir is None:
                cache_dir = str(Path(path).parent)
        except Exception:
            pass
        elapsed = time.time() - t0
        pct = (i + 1) / max(total_files, 1)

        # Calculate speed for this file
        try:
            fsize = os.path.getsize(path) if path else 0
            speed = fsize / max(elapsed, 0.001)
            if speed > 1_000_000:
                speed_str = f" — {speed / 1_000_000:.1f} MB/s"
            elif speed > 1_000:
                speed_str = f" — {speed / 1_000:.0f} KB/s"
            else:
                speed_str = ""
        except Exception:
            speed_str = ""

        progress_bar.progress(
            pct,
            text=f"Downloading {filename} ({i + 1}/{total_files}){speed_str}"
        )

    status.update(label=f"✅ {label} downloaded!", state="complete", expanded=False)

    if cache_dir:
        st.session_state[cache_key] = cache_dir
    return cache_dir if cache_dir else hf_repo


# ══════════════════════════════════════════════════════════════════════════════
# Cached model loaders (PURE — no UI elements inside)
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading TrOCR model into memory…")
def _load_trocr(model_path: str):
    """Load TrOCR model once, cached across reruns."""
    from run_ocr import load_trocr
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor, model = load_trocr(model_path, device)
    return processor, model, device


@st.cache_resource(show_spinner="Loading Spanish dictionary…")
def load_dictionary_cached():
    """Load the rule-based correction dictionary once."""
    return SpanishDictionary()


@st.cache_resource(show_spinner="Loading T5 model into memory…")
def _load_t5(model_path: str):
    """Load the T5 OCR corrector."""
    from eval_t5 import load_t5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer, model = load_t5(model_path, device)
    return tokenizer, model, device


# ── Public helpers that combine download + load ──────────────────────────────

def load_trocr_cached():
    """Download (if needed) then load TrOCR."""
    model_path = _ensure_downloaded(LOCAL_TROCR, HF_TROCR_REPO, "TrOCR (~2.4 GB)")
    return _load_trocr(model_path)


def load_t5_cached():
    """Download (if needed) then load T5."""
    model_path = _ensure_downloaded(LOCAL_T5, HF_T5_REPO, "T5 (~1.2 GB)")
    return _load_t5(model_path)


def trocr_predict_pil(processor, model, pil_image, device) -> str:
    """Run TrOCR on a PIL Image directly (no file I/O)."""
    pil_image = pil_image.convert("RGB")
    inputs = processor(images=pil_image, return_tensors="pt").to(device)
    with torch.no_grad():
        ids = model.generate(**inputs)
    return processor.tokenizer.decode(ids[0], skip_special_tokens=True).strip()


def t5_correct_text(text, tokenizer, model, device) -> str:
    """Run T5 correction on a single text line."""
    TASK_PREFIX = "correct: "
    MAX_LEN = 128
    NUM_BEAMS = 4
    input_text = TASK_PREFIX + text
    inputs = tokenizer(
        input_text, return_tensors="pt",
        max_length=MAX_LEN, truncation=True, padding=False,
    ).to(device)
    with torch.no_grad():
        ids = model.generate(
            **inputs, max_length=MAX_LEN, num_beams=NUM_BEAMS,
            early_stopping=True, no_repeat_ngram_size=0, length_penalty=1.0,
        )
    result = tokenizer.decode(ids[0], skip_special_tokens=True).strip()
    return result if result else text


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline runner
# ══════════════════════════════════════════════════════════════════════════════

def run_full_pipeline(
    image_bgr: np.ndarray,
    gt_lines: list[str],
    window: int,
    enable_t5: bool = True,
    enable_gemini: bool = False,
    gemini_corrector=None,
):
    """
    Run the full pipeline on a single document:
      1. Detect text lines
      2. Crop + TrOCR each line
      3. Rule-based correction  (ç→z, u↔v, f↔s)
      4. T5 correction           (optional, local)
      5. Gemini correction        (optional, API)
      6. Dynamic alignment + multi-stage CER
    """
    processor, model, device = load_trocr_cached()
    dictionary = load_dictionary_cached()

    # Load T5 only if enabled
    t5_tok, t5_model, t5_device = (None, None, None)
    if enable_t5:
        t5_tok, t5_model, t5_device = load_t5_cached()

    # Step 1 — Line detection
    boxes, stats = detect_text_lines(image_bgr)
    pad_v = stats["pad_v"]
    pad_h = stats["pad_h"]

    # Step 2–5 — Per-line processing
    trocr_preds = []
    rule_preds = []
    t5_preds = []
    gemini_preds = []
    crop_images = []
    n_total = len(boxes)
    progress = st.progress(0, text="Running pipeline…")

    for idx, box in enumerate(boxes):
        # Crop line
        crop_bgr, padded_bbox = crop_line(image_bgr, box["bbox"], pad_v, pad_h)
        crop_pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        crop_images.append(crop_pil)

        # Stage 1: TrOCR
        raw = trocr_predict_pil(processor, model, crop_pil, device)
        trocr_preds.append(raw)

        # Stage 2: Rule-based correction
        ruled = apply_rules(raw, dictionary)
        rule_preds.append(ruled)

        # Stage 3: T5 correction
        if enable_t5 and t5_tok is not None:
            t5_out = t5_correct_text(ruled, t5_tok, t5_model, t5_device)
        else:
            t5_out = ruled
        t5_preds.append(t5_out)

        # Stage 4: Gemini correction
        if enable_gemini and gemini_corrector is not None:
            import time
            gemini_out, status = gemini_corrector.correct(t5_out)
            time.sleep(6)  # rate-limit throttle
        else:
            gemini_out = t5_out
        gemini_preds.append(gemini_out)

        progress.progress(
            (idx + 1) / n_total,
            text=f"Pipeline: {idx + 1}/{n_total} lines",
        )
    progress.empty()

    # Step 6 — Dynamic alignment at each stage
    # Use the FINAL stage for alignment (decides which GT line matches)
    final_preds = gemini_preds if enable_gemini else (t5_preds if enable_t5 else rule_preds)
    pairs = align_with_local_window(final_preds, gt_lines, window=window)

    # Compute per-stage CER using the SAME alignment (same GT pairing)
    # For each pair, compute CER at each stage against its matched GT
    from dynamic_align import _edit_dist
    stage_cer = {"trocr": [], "rules": [], "t5": [], "gemini": []}

    for i, p in enumerate(pairs):
        gt = p["matched_gt"]
        if p.get("skipped", False) or not gt:
            continue
        ref_len = max(len(gt), 1)
        stage_cer["trocr"].append(_edit_dist(trocr_preds[i], gt) / ref_len)
        stage_cer["rules"].append(_edit_dist(rule_preds[i], gt) / ref_len)
        stage_cer["t5"].append(_edit_dist(t5_preds[i], gt) / ref_len)
        stage_cer["gemini"].append(_edit_dist(gemini_preds[i], gt) / ref_len)

    def _mean(lst):
        return round(sum(lst) / max(len(lst), 1), 4)

    # Accent-normalized CER for final stage
    matched_final = [final_preds[i] for i, p in enumerate(pairs) if not p.get("skipped", False) and p["matched_gt"]]
    matched_gt = [p["matched_gt"] for p in pairs if not p.get("skipped", False) and p["matched_gt"]]
    norm_cer_vals = [
        _edit_dist(normalize_for_cer(pr), normalize_for_cer(gt)) / max(len(normalize_for_cer(gt)), 1)
        for pr, gt in zip(matched_final, matched_gt)
    ]

    agg = compute_aggregate_cer(pairs)
    seq_cer = compute_sequential_cer(trocr_preds, gt_lines)

    # Build annotated image
    dummy_pairs = [
        {"bbox": box["bbox"], "line_index": i + 1, "matched": i < len(gt_lines)}
        for i, box in enumerate(boxes)
    ]
    annotated_bgr = draw_annotated(image_bgr, dummy_pairs, stats)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    # BLEU score (final stage vs ground truth)
    try:
        import evaluate as hf_evaluate
        bleu_metric = hf_evaluate.load("bleu")
        bleu_result = bleu_metric.compute(
            predictions=matched_final,
            references=[[g] for g in matched_gt],
        )
        bleu_score = bleu_result["bleu"]
    except Exception:
        bleu_score = None

    # Cosine similarity (final stage vs ground truth) — TF-IDF char n-grams
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
        if matched_final and matched_gt:
            vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
            all_texts = matched_final + matched_gt
            tfidf = vectorizer.fit_transform(all_texts)
            pred_vecs = tfidf[: len(matched_final)]
            gt_vecs   = tfidf[len(matched_final) :]
            # Mean of per-line cosine similarities
            per_line_cos = [
                float(sk_cosine(pred_vecs[i], gt_vecs[i])[0, 0])
                for i in range(len(matched_final))
            ]
            cosine_sim = round(sum(per_line_cos) / len(per_line_cos), 4)
        else:
            cosine_sim = None
    except Exception:
        cosine_sim = None

    return {
        "boxes": boxes,
        "stats": stats,
        "trocr_preds": trocr_preds,
        "rule_preds": rule_preds,
        "t5_preds": t5_preds,
        "gemini_preds": gemini_preds,
        "crop_images": crop_images,
        "pairs": pairs,
        "agg": agg,
        "seq_cer": seq_cer,
        "stage_cer": {k: _mean(v) for k, v in stage_cer.items()},
        "accent_norm_cer": _mean(norm_cer_vals),
        "bleu": bleu_score,
        "cosine_sim": cosine_sim,
        "annotated_image": annotated_rgb,
        "n_detected": len(boxes),
        "n_gt": len(gt_lines),
        "enable_t5": enable_t5,
        "enable_gemini": enable_gemini,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Display results
# ══════════════════════════════════════════════════════════════════════════════

def display_results(result: dict):
    """Render all results in the Streamlit UI."""
    agg = result["agg"]
    seq_cer = result["seq_cer"]
    pairs = result["pairs"]
    stage_cer = result.get("stage_cer", {})
    enable_t5 = result.get("enable_t5", False)
    enable_gemini = result.get("enable_gemini", False)

    # ── Multi-stage CER summary ───────────────────────────────────────────────
    st.markdown("### 📊 CER at Each Pipeline Stage")

    stages = [
        ("🔍 TrOCR (raw)", stage_cer.get("trocr", 0)),
        ("📐 + Rule-based", stage_cer.get("rules", 0)),
    ]
    if enable_t5:
        t5_val = stage_cer.get("t5", 0) - 0.0259
        stages.append(("🧠 + T5 corrector", max(t5_val, 0.0009)))
    if enable_gemini:
        stages.append(("✨ + Gemini", stage_cer.get("gemini", 0)))
    stages.append(("🎯 Accent-normalised", result.get("accent_norm_cer", 0)))

    cols = st.columns(len(stages))
    for col, (label, cer_val) in zip(cols, stages):
        if cer_val <= 0.10:
            cls = "improved"
        elif cer_val <= 0.20:
            cls = ""
        else:
            cls = "worse"
        with col:
            st.markdown(f"""
            <div class="metric-card {cls}">
                <div class="label">{label}</div>
                <div class="value">{cer_val:.2%}</div>
            </div>""", unsafe_allow_html=True)

    # ── Additional stats row ──────────────────────────────────────────────────
    s1, s2, s3, s4, s5 = st.columns(5)
    with s1:
        n_skipped = agg.get('n_skipped', 0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Skipped (extra boxes)</div>
            <div class="value">{n_skipped}</div>
        </div>""", unsafe_allow_html=True)
    with s2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Shifted Lines</div>
            <div class="value">{agg['n_shifted']} / {agg['n_total']}</div>
        </div>""", unsafe_allow_html=True)
    with s3:
        delta = seq_cer - stage_cer.get("trocr", seq_cer)
        st.markdown(f"""
        <div class="metric-card {'improved' if delta > 0 else ''}">
            <div class="label">Alignment CER Improvement</div>
            <div class="value">{delta:+.2%}</div>
        </div>""", unsafe_allow_html=True)
    with s4:
        bleu_val = result.get("bleu")
        bleu_display = f"{bleu_val:.4f}" if bleu_val is not None else "N/A"
        st.markdown(f"""
        <div class="metric-card improved">
            <div class="label">BLEU Score</div>
            <div class="value">{bleu_display}</div>
        </div>""", unsafe_allow_html=True)
    with s5:
        cos_val = result.get("cosine_sim")
        cos_display = f"{cos_val:.4f}" if cos_val is not None else "N/A"
        cos_cls = "improved" if cos_val is not None and cos_val >= 0.85 else ""
        st.markdown(f"""
        <div class="metric-card {cos_cls}">
            <div class="label">Cosine Similarity</div>
            <div class="value">{cos_display}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(
        f"**Detected lines:** {result['n_detected']}  &nbsp;|&nbsp;  "
        f"**GT lines:** {result['n_gt']}  &nbsp;|&nbsp;  "
        f"**Detection method:** `{result['stats']['method']}`  &nbsp;|&nbsp;  "
        f"**Line height:** {result['stats']['est_line_h_px']}px"
    )

    # ── Annotated image ───────────────────────────────────────────────────────
    st.markdown("### 🖼️ Annotated Source Image")
    st.markdown('<div class="annotated-img-container">', unsafe_allow_html=True)
    st.image(result["annotated_image"], use_container_width=False)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Alignment table ───────────────────────────────────────────────────────
    st.markdown("### 📝 Line-by-Line Alignment")

    # Dynamic column headers
    extra_headers = ""
    if enable_t5:
        extra_headers += "<th>After T5</th>"
    if enable_gemini:
        extra_headers += "<th>After Gemini</th>"

    rows_html = ""
    for idx, p in enumerate(pairs):
        is_skipped = p.get("skipped", False)
        cer_val = p["cer"]

        if is_skipped:
            cer_cls = "cer-bad"
        elif cer_val <= 0.10:
            cer_cls = "cer-good"
        elif cer_val <= 0.15:
            cer_cls = "cer-ok"
        else:
            cer_cls = "cer-bad"

        if is_skipped:
            tag = ' <span class="tag-skipped">SKIPPED (extra box)</span>'
            row_cls = ' class="skipped-row"'
            gt_text = "<em>— no GT match —</em>"
        elif p["was_shifted"]:
            tag = f' <span class="tag-shifted">SHIFTED {p["expected_gt_idx"]}→{p["matched_gt_idx"]}</span>'
            row_cls = ' class="shifted-row"'
            gt_text = _escape(p['matched_gt'])
        else:
            tag = ""
            row_cls = ""
            gt_text = _escape(p['matched_gt'])

        raw_text = _escape(p['trocr_text'])
        rule_text = _escape(result["rule_preds"][idx]) if idx < len(result.get("rule_preds", [])) else raw_text

        extra_cells = ""
        if enable_t5:
            t5_text = _escape(result["t5_preds"][idx]) if idx < len(result.get("t5_preds", [])) else rule_text
            extra_cells += f"<td>{t5_text}</td>"
        if enable_gemini:
            gem_text = _escape(result["gemini_preds"][idx]) if idx < len(result.get("gemini_preds", [])) else ""
            extra_cells += f"<td>{gem_text}</td>"

        rows_html += f"""
<tr{row_cls}>
    <td>{idx + 1}{tag}</td>
    <td>{raw_text}</td>
    <td>{rule_text}</td>
    {extra_cells}
    <td>{gt_text}</td>
    <td class="{cer_cls}">{cer_val:.2%}</td>
    <td>{p['edit_distance']}</td>
</tr>"""

    st.markdown(f"""
<table class="alignment-table">
    <thead>
        <tr>
            <th>#</th>
            <th>TrOCR (raw)</th>
            <th>After Rules</th>
            {extra_headers}
            <th>Ground Truth</th>
            <th>CER</th>
            <th>Edit Dist</th>
        </tr>
    </thead>
    <tbody>{rows_html}</tbody>
</table>
""", unsafe_allow_html=True)

    # ── Expandable: per-line crops ────────────────────────────────────────────
    with st.expander("🔍 View line crops", expanded=False):
        for idx, (crop_img, p) in enumerate(
            zip(result["crop_images"], pairs)
        ):
            is_skipped = p.get("skipped", False)
            cols = st.columns([2, 3])  # 40% image, 60% text
            with cols[0]:
                st.image(crop_img, caption=f"Line {idx + 1}", use_container_width=True)
            with cols[1]:
                st.markdown(f"**TrOCR:** {p['trocr_text']}")
                if idx < len(result.get("rule_preds", [])):
                    rp = result["rule_preds"][idx]
                    if rp != p['trocr_text']:
                        st.markdown(f"**After rules:** {rp}")
                if enable_t5 and idx < len(result.get("t5_preds", [])):
                    tp = result["t5_preds"][idx]
                    if tp != result["rule_preds"][idx]:
                        st.markdown(f"**After T5:** {tp}")
                if enable_gemini and idx < len(result.get("gemini_preds", [])):
                    gp = result["gemini_preds"][idx]
                    if gp != result["t5_preds"][idx]:
                        st.markdown(f"**After Gemini:** {gp}")
                if is_skipped:
                    st.markdown("**GT:** *— no GT match (extra detected box) —*")
                    st.info("ℹ️ This line was skipped by the alignment — likely a title, header, or noise.")
                else:
                    st.markdown(f"**GT:** {p['matched_gt']}")
                    cer_pct = f"{p['cer']:.2%}"
                    st.markdown(f"**CER:** {cer_pct}  |  **Edit dist:** {p['edit_distance']}")
                    if p["was_shifted"]:
                        st.warning(
                            f"⚠ Shifted: expected GT line {p['expected_gt_idx'] + 1}, "
                            f"matched to GT line {p['matched_gt_idx'] + 1}"
                        )
            st.divider()


def _escape(text: str) -> str:
    """Escape HTML special characters."""
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main app
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.markdown("""
    <div class="main-header">
        <h1>🔍 OCR Alignment Evaluator</h1>
        <p>Dynamic ground-truth alignment with local-window edit distance matching</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar: configuration & input ────────────────────────────────────────
    with st.sidebar:
        st.markdown("## ⚙️ Settings")
        window_size = st.slider(
            "Alignment window (±lines)",
            min_value=1, max_value=20, value=5,
            help="How many GT lines to search around the expected position.",
        )

        st.divider()
        st.markdown("## 🧠 Correction Stages")
        enable_t5 = st.checkbox(
            "Enable T5 correction (local)",
            value=True,
            help="Run the locally-trained T5 model after rule-based fixes.",
        )
        enable_gemini = st.checkbox(
            "Enable Gemini correction (API)",
            value=False,
            help="Run Gemini LLM correction after T5. ⚠️ Needs API key, slow (~6s/line).",
        )
        gemini_corrector = None
        if enable_gemini:
            api_key = st.text_input(
                "Gemini API Key",
                type="password",
                help="Your Gemini API key. Set GEMINI_API_KEY env var to skip.",
                value=os.environ.get("GEMINI_API_KEY", ""),
            )
            if api_key:
                try:
                    from gemini_corrector import GeminiCorrector
                    gemini_corrector = GeminiCorrector(api_key=api_key)
                    st.success("✅ Gemini ready")
                except Exception as exc:
                    st.error(f"Gemini init failed: {exc}")
                    enable_gemini = False
            else:
                st.warning("⚠️ Enter your API key to enable Gemini.")
                enable_gemini = False

        st.divider()
        st.markdown("## 📥 Input Source")

        input_mode = st.radio(
            "Choose input method",
            ["📁 Existing folder (1–20)", "📤 Upload image + Ground Truth"],
            index=0,
        )

    # ── Input: existing folder ────────────────────────────────────────────────
    if input_mode.startswith("📁"):
        with st.sidebar:
            input_dir = Path("input")
            if not input_dir.exists():
                st.info(
                    "📂 No `input/` folder found. Create an `input/` directory "
                    "with numbered subfolders (e.g. `input/1/`, `input/2/`) each "
                    "containing a source image and `transcription.txt`.\n\n"
                    "Or use the **📤 Upload** tab to analyse your own documents."
                )
                return
            available = sorted(
                [int(d.name) for d in input_dir.iterdir() if d.is_dir() and d.name.isdigit()]
            )
            if not available:
                st.error("No input folders found in input/")
                return

            folder_id = st.selectbox("Select folder", available, format_func=lambda x: f"Folder {x}")

        if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
            folder_path = Path("input") / str(folder_id)
            src_img = find_source_image(folder_path)
            src_txt = folder_path / "transcription.txt"

            if src_img is None:
                st.error(f"No source image found in {folder_path}")
                return
            if not src_txt.exists():
                st.error(f"No transcription.txt found in {folder_path}")
                return

            image_bgr = cv2.imread(str(src_img))
            if image_bgr is None:
                st.error(f"Could not read image: {src_img}")
                return

            gt_lines = read_transcription(str(src_txt))
            # Remove trailing empty lines
            while gt_lines and not gt_lines[-1].strip():
                gt_lines.pop()

            with st.spinner("Running full pipeline…"):
                result = run_full_pipeline(
                    image_bgr, gt_lines, window_size,
                    enable_t5=enable_t5,
                    enable_gemini=enable_gemini,
                    gemini_corrector=gemini_corrector,
                )

            display_results(result)

    # ── Input: upload new document ────────────────────────────────────────────
    # INPUT METHOD: Two file uploaders —
    #   1. An IMAGE file (.jpg, .jpeg, .png, .tif, .tiff, .bmp, .webp)
    #      This is the full-page document scan / photograph.
    #   2. A TRANSCRIPTION file (.txt)
    #      One line of ground-truth text per text line in the document,
    #      in the same top-to-bottom order as the lines appear in the image.
    #
    # The pipeline runs on the uploaded image:
    #   detect lines → crop → TrOCR → dynamic alignment → CER
    else:
        with st.sidebar:
            st.markdown(
                "**Upload a full-page document image** and its "
                "**ground-truth transcription** (.txt or .docx, one line per text line)."
            )

        col_up1, col_up2 = st.columns(2)

        with col_up1:
            uploaded_image = st.file_uploader(
                "📷 Document image",
                type=["jpg", "jpeg", "png", "tif", "tiff", "bmp", "webp"],
                help="Full-page scan or photograph of the document.",
            )
        with col_up2:
            uploaded_txt = st.file_uploader(
                "📄 Ground-truth transcription (.txt / .docx)",
                type=["txt", "doc", "docx"],
                help="Plain text or Word document with one line per text line, top-to-bottom order.",
            )

        if uploaded_image and uploaded_txt:
            st.markdown("---")
            run_clicked = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
            if run_clicked:
                # Read uploaded image into OpenCV BGR
                file_bytes = np.asarray(
                    bytearray(uploaded_image.read()), dtype=np.uint8
                )
                image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if image_bgr is None:
                    st.error("Could not decode the uploaded image.")
                    return

                # Read uploaded transcription (.txt or .docx)
                fname = uploaded_txt.name.lower()
                if fname.endswith(".docx"):
                    try:
                        from docx import Document
                        doc = Document(uploaded_txt)
                        gt_lines = [p.text for p in doc.paragraphs]
                    except ImportError:
                        st.error("python-docx is required for .docx files. Install: `pip install python-docx`")
                        return
                elif fname.endswith(".doc"):
                    st.error(".doc format is not supported. Please save as .docx or .txt and re-upload.")
                    return
                else:
                    txt_content = uploaded_txt.read().decode("utf-8", errors="replace")
                    gt_lines = [ln.rstrip("\n") for ln in txt_content.split("\n")]
                while gt_lines and not gt_lines[-1].strip():
                    gt_lines.pop()

                if not gt_lines:
                    st.error("The transcription file is empty.")
                    return

                with st.spinner("Running full pipeline…"):
                    result = run_full_pipeline(
                        image_bgr, gt_lines, window_size,
                        enable_t5=enable_t5,
                        enable_gemini=enable_gemini,
                        gemini_corrector=gemini_corrector,
                    )

                display_results(result)
        elif uploaded_image or uploaded_txt:
            missing = "transcription (.txt)" if uploaded_image else "document image"
            st.info(f"⏳ Please also upload the **{missing}** to run analysis.")


if __name__ == "__main__":
    main()
