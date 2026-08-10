

import json
import cv2
import numpy as np
from pathlib import Path
import shutil
from scipy.signal import find_peaks


INPUT_DIR  = "input"
OUTPUT_DIR = "output"

# Padding as a FRACTION of the estimated line height (adaptive per page)
PAD_FRAC_V = 0.15   # vertical:   15 % of line height top & bottom
PAD_FRAC_H = 0.05   # horizontal:  5 % of image width left & right

# Lines smaller than this fraction of median area → noise / marginalia
MARGIN_FILTER_RATIO = 0.20

# Supported source image extensions (tried in order)
IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp']





def binarize(image_bgr):
    """Otsu binarization → ink=white, background=black."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bw


def horizontal_projection(bw):
    """Row-wise sum of foreground pixels."""
    return np.sum(bw, axis=1).astype(np.float64)


def _moving_avg(arr, window):
    w = max(1, int(window))
    return np.convolve(arr.astype(float), np.ones(w) / w, mode='same')




def estimate_line_period_acf(proj):
    """
    Estimate line period via autocorrelation of the horizontal projection.
    Returns (period_px, confidence) or (None, 0.0) if signal is too short/flat.
    """
    n = len(proj)
    if n < 20:
        return None, 0.0
    p = proj - proj.mean()
    if p.std() < 1e-6:
        return None, 0.0
    acf = np.correlate(p, p, mode='full')
    acf = acf[len(acf) // 2:]
    acf = acf / acf[0]
    min_lag = 4
    max_lag = max(min_lag + 2, n // 3)
    search  = acf[min_lag:max_lag]
    if len(search) < 2:
        return None, 0.0
    peaks, props = find_peaks(search, height=0.05, prominence=0.03)
    if len(peaks) == 0:
        return None, 0.0
    best       = int(np.argmax(props['prominences']))
    period     = int(peaks[best]) + min_lag
    confidence = float(acf[period])
    return period, confidence


def estimate_line_period_cc(image_bgr):
    """
    Estimate line period from connected-component heights.

    The 75th-percentile CC height approximates cap-letter height.
    Multiplying by 1.5 gives a good estimate of the full line period
    (letter + inter-line gap), confirmed to match ACF results on normal pages.

    Used as fallback when ACF cannot find a reliable period (too few lines,
    unusual ink patterns, strikethrough bars, etc.).
    """
    bw = binarize(image_bgr)
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(bw)
    if n_labels < 2:
        return None
    heights = stats[1:, cv2.CC_STAT_HEIGHT]
    areas   = stats[1:, cv2.CC_STAT_AREA]
    h_img   = image_bgr.shape[0]
    # Keep only plausible letter-sized components
    good = heights[(areas > 10) & (heights > 3) & (heights < h_img * 0.8)]
    if len(good) < 3:
        return None
    return max(8, int(np.percentile(good, 75) * 1.5))




def find_line_segments(proj, image_bgr):
    """
    Fully adaptive text-line detector.

    Period estimation (in priority order):
      1. Autocorrelation (ACF) — most accurate on pages with ≥ 4 lines
      2. Connected-component heights (CC) — works on shorter images
      3. Image-height heuristic — last resort

    Returns
    -------
    segments   : list of (start_row, end_row) tuples
    est_line_h : int   – estimated single-line height in pixels
    period     : int   – estimated line period in pixels
    confidence : float – ACF confidence (0 if CC / heuristic used)
    method     : str   – which estimation method was used
    """
    n          = len(proj)
    image_height = image_bgr.shape[0]

    if n == 0 or not np.any(proj > 0):
        return [], 0, 0, 0.0, 'empty'

    # ── Step 1: Determine line period ─────────────────────────────────────
    period, confidence = estimate_line_period_acf(proj)

    if period is not None and confidence >= 0.10:
        method = 'autocorrelation'
    else:
        # Try connected-component estimate
        cc_period = estimate_line_period_cc(image_bgr)
        if cc_period is not None:
            period, confidence = cc_period, 0.0
            method = 'cc_height'
        else:
            period, confidence = max(8, image_height // 50), 0.0
            method = 'heuristic'

    est_line_h      = max(4, int(period * 0.75))
    valley_min_dist = max(4, period // 2)

    # ── Step 2: Smooth projection ─────────────────────────────────────────
    smooth_win = max(3, est_line_h // 3)
    smoothed   = _moving_avg(proj, smooth_win)

    # ── Step 3: Find valleys ──────────────────────────────────────────────
    inv    = -smoothed
    active = smoothed[smoothed > np.percentile(smoothed, 10)]
    median_val     = np.median(active) if len(active) > 0 else 1.0
    prom_threshold = median_val * 0.06

    valleys, _ = find_peaks(
        inv,
        distance=valley_min_dist,
        prominence=prom_threshold
    )

    # ── Step 4: Build segments from spaces between valleys ────────────────
    cut_points = np.concatenate([[-1], valleys, [n]])
    min_seg_h  = max(4, est_line_h // 4)
    segments   = []
    for i in range(len(cut_points) - 1):
        s = int(cut_points[i]) + 1
        e = int(cut_points[i + 1]) - 1
        if s > e:
            continue
        if e - s + 1 >= min_seg_h and proj[s:e + 1].max() > 0:
            segments.append((s, e))

    # ── Step 5: Fallback if valley detection found nothing ────────────────
    if len(segments) <= 1 and np.any(proj > 0):
        segments = _fallback_threshold_split(proj, est_line_h)
        method  += '+fallback'

    # ── Step 6: Absorb fragment segments ─────────────────────────────────
    # Any segment shorter than 60 % of est_line_h is a fragment (strikethrough
    # bar, partial descender, noise band) — absorb it into the nearest neighbour.
    # This fixes "one line counted as two" without affecting full-size lines.
    segments = _absorb_fragments(segments, est_line_h)

    return segments, est_line_h, period, confidence, method


def _fallback_threshold_split(proj, est_line_h):
    """
    Split on rows whose ink density falls below a relative threshold.
    Used when valley detection finds no splits at all.
    """
    smoothed  = _moving_avg(proj, max(3, est_line_h // 2))
    pos_vals  = smoothed[smoothed > 0]
    threshold = np.percentile(pos_vals, 20) * 0.5 if len(pos_vals) > 0 else 1.0

    in_text  = False
    segments = []
    start    = 0
    gap_run  = 0
    min_gap  = max(2, est_line_h // 5)

    for i, val in enumerate(smoothed):
        if val >= threshold:
            if not in_text:
                in_text = True
                start   = i
            gap_run = 0
        else:
            if in_text:
                gap_run += 1
                if gap_run >= min_gap:
                    segments.append((start, i - gap_run))
                    in_text = False
                    gap_run = 0
    if in_text:
        segments.append((start, len(smoothed) - 1))
    return segments


def _absorb_fragments(segments, est_line_h):
    """
    Absorb "fragment" segments — those shorter than 60 % of the estimated
    line height — into the adjacent full-size segment.

    This is the correct fix for intra-line splits caused by:
      • strikethrough bars (thin horizontal spike above letter bodies)
      • descenders creating a local dip near the bottom of a line
      • partial ascenders at the very top or bottom of a scan crop

    Why it doesn't collapse normal pages:
      Valley detection with distance=valley_min_dist ensures real inter-line
      valleys are at least half a period apart.  Each resulting segment spans
      ~75 % of the period (≥ 60 % threshold) so no normal-page segment is
      ever flagged as a fragment.
    """
    if len(segments) < 2:
        return segments

    frag_threshold = est_line_h * 0.60

    # Iteratively absorb until no fragments remain
    for _ in range(len(segments)):          # at most len(segments) passes
        changed = False
        result  = []
        i       = 0
        while i < len(segments):
            s, e = segments[i]
            h    = e - s + 1
            if h < frag_threshold:
                # Fragment — absorb into nearest neighbour
                if result:
                    # Extend the previous segment
                    result[-1] = (result[-1][0], e)
                elif i + 1 < len(segments):
                    # No previous segment; extend the next one instead
                    ns, ne = segments[i + 1]
                    segments[i + 1] = (s, ne)
                    i += 1
                    continue
                changed = True
            else:
                result.append((s, e))
            i += 1
        segments = result
        if not changed:
            break

    return segments


def column_extent(bw, row_start, row_end):
    """Horizontal extent of ink within a row band."""
    band     = bw[row_start:row_end + 1, :]
    col_proj = np.sum(band, axis=0)
    nz       = np.where(col_proj > 0)[0]
    if len(nz) == 0:
        return 0, bw.shape[1]
    return int(nz[0]), int(nz[-1])


def detect_text_lines(image_bgr):
    """
    Main detection entry point.

    Returns
    -------
    boxes      : list of {'bbox': [x1,y1,x2,y2], 'area': int}
    stats      : dict with detected line metrics (for logging)
    """
    h, w = image_bgr.shape[:2]
    bw   = binarize(image_bgr)
    proj = horizontal_projection(bw)

    segments, est_line_h, period, confidence, method = find_line_segments(proj, image_bgr)

    # Compute adaptive padding from the measured line height
    pad_v = max(2, int(est_line_h * PAD_FRAC_V))
    pad_h = max(2, int(w * PAD_FRAC_H))

    boxes = []
    for y1, y2 in segments:
        x1, x2 = column_extent(bw, y1, y2)
        area    = (x2 - x1) * (y2 - y1)
        boxes.append({
            'bbox' : [x1, y1, x2, y2],
            'area' : area,
        })

    # Remove noise / marginalia
    if boxes:
        median = float(np.median([b['area'] for b in boxes]))
        boxes  = [b for b in boxes if b['area'] >= median * MARGIN_FILTER_RATIO]

    stats = {
        'image_h'       : h,
        'image_w'       : w,
        'period_px'     : period,
        'est_line_h_px' : est_line_h,
        'acf_confidence': round(confidence, 3),
        'pad_v'         : pad_v,
        'pad_h'         : pad_h,
        'method'        : method,
    }

    return boxes, stats




def crop_line(image_bgr, bbox, pad_v, pad_h):
    h, w = image_bgr.shape[:2]
    x1, y1, x2, y2 = bbox
    x1p = max(0, x1 - pad_h)
    y1p = max(0, y1 - pad_v)
    x2p = min(w, x2 + pad_h)
    y2p = min(h, y2 + pad_v)
    return image_bgr[y1p:y2p, x1p:x2p], [x1p, y1p, x2p, y2p]




def read_transcription(txt_path):
    for enc in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
        try:
            with open(txt_path, 'r', encoding=enc) as f:
                text = f.read()
            break
        except UnicodeDecodeError:
            continue
    else:
        text = ""
    return [ln.rstrip('\n') for ln in text.split('\n')]


def align_lines(detected_boxes, transcription_lines, verbose=True):
    n_det  = len(detected_boxes)
    n_text = len(transcription_lines)

    if verbose:
        print(f"  Detected image lines : {n_det}")
        print(f"  Transcription lines  : {n_text}")
        if n_det != n_text:
            print(f"  ⚠  Count mismatch – mapping {min(n_det, n_text)} pairs; "
                  f"remainder marked unmatched.")

    pairs = []
    for i, box in enumerate(detected_boxes):
        pairs.append({
            'line_index'   : i + 1,
            'bbox'         : box['bbox'],
            'transcription': transcription_lines[i] if i < n_text else "",
            'matched'      : i < n_text,
        })

    unmatched_text = [
        {'line_index': j + 1, 'transcription': transcription_lines[j]}
        for j in range(n_det, n_text)
    ] if n_text > n_det else []

    return pairs, unmatched_text




def draw_annotated(image_bgr, pairs, stats):
    vis = image_bgr.copy()
    lh  = stats['est_line_h_px']
    font_scale = max(0.35, min(0.8, lh / 30))
    thickness  = max(1, lh // 15)

    for p in pairs:
        x1, y1, x2, y2 = p['bbox']
        color = (0, 200, 0) if p['matched'] else (0, 0, 220)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(vis, str(p['line_index']),
                    (x1 + 2, max(y1 + lh - 2, lh)),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color,
                    max(1, thickness - 1), cv2.LINE_AA)

    # Stamp detection stats in the corner
    info = (f"period={stats['period_px']}px  "
            f"line_h={stats['est_line_h_px']}px  "
            f"conf={stats['acf_confidence']}  "
            f"[{stats['method']}]")
    cv2.putText(vis, info, (4, max(12, lh)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 0, 180), 1, cv2.LINE_AA)
    return vis




def find_source_image(folder_path):
    folder_path = Path(folder_path)
    for ext in IMAGE_EXTENSIONS:
        c = folder_path / f"source{ext}"
        if c.exists():
            return c
    for f in sorted(folder_path.iterdir()):
        if f.suffix.lower() in IMAGE_EXTENSIONS:
            return f
    return None




def process_folder(folder_id, input_root, output_root):
    folder_path = Path(input_root) / str(folder_id)
    src_txt     = folder_path / "transcription.txt"

    if not folder_path.exists():
        print(f"[{folder_id}] ⚠  folder not found – skipping")
        return None

    src_img = find_source_image(folder_path)
    if src_img is None:
        print(f"[{folder_id}] ⚠  no image file found – skipping")
        return None
    if not src_txt.exists():
        print(f"[{folder_id}] ⚠  transcription.txt not found – skipping")
        return None

    print(f"\n{'─'*60}")
    print(f"[{folder_id}]  image: {src_img.name}")

    image_bgr = cv2.imread(str(src_img))
    if image_bgr is None:
        print(f"[{folder_id}] ⚠  Could not read image – skipping")
        return None

    trans_lines = read_transcription(str(src_txt))

    # ── Detect lines adaptively ──────────────────────────────────────────
    boxes, stats = detect_text_lines(image_bgr)
    pad_v = stats['pad_v']
    pad_h = stats['pad_h']

    print(f"  Adaptive stats  → period={stats['period_px']}px  "
          f"line_h≈{stats['est_line_h_px']}px  "
          f"confidence={stats['acf_confidence']}  "
          f"method={stats['method']}")
    print(f"  Adaptive padding → vertical=±{pad_v}px  horizontal=±{pad_h}px")
    print(f"  → {len(boxes)} text-line regions detected")

   
    pairs, unmatched = align_lines(boxes, trans_lines, verbose=True)

    
    out_dir = Path(output_root) / str(folder_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    dest_img = out_dir / f"source{src_img.suffix}"
    shutil.copy(str(src_img), str(dest_img))

    
    mapping_records = []
    for p in pairs:
        crop, padded_bbox = crop_line(image_bgr, p['bbox'], pad_v, pad_h)
        fname = f"line_{p['line_index']:03d}.png"
        cv2.imwrite(str(out_dir / fname), crop)
        mapping_records.append({
            "line_index"    : p['line_index'],
            "crop_file"     : fname,
            "bbox_original" : p['bbox'],
            "bbox_padded"   : padded_bbox,
            "transcription" : p['transcription'],
            "matched"       : p['matched'],
        })

    
    annotated = draw_annotated(image_bgr, pairs, stats)
    cv2.imwrite(str(out_dir / "annotated.png"), annotated)

    
    result = {
        "folder_id"                    : folder_id,
        "source_image"                 : dest_img.name,
        "detection_stats"              : stats,
        "total_image_lines"            : len(boxes),
        "total_transcription_lines"    : len(trans_lines),
        "matched_pairs"                : len([p for p in pairs if p['matched']]),
        "unmatched_image_lines"        : len([p for p in pairs if not p['matched']]),
        "unmatched_text_lines"         : len(unmatched),
        "lines"                        : mapping_records,
        "unmatched_transcription_lines": unmatched,
    }
    with open(str(out_dir / "mapping.json"), 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"  ✓ Saved {len(mapping_records)} crops  →  {out_dir}")
    return result




def run_pipeline(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR,
                 folder_range=range(1, 21)):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    all_results = []

    for fid in folder_range:
        res = process_folder(fid, input_dir, output_dir)
        if res:
            all_results.append(res)

    summary = {
        "output_dir"             : output_dir,
        "total_folders_processed": len(all_results),
        "total_matched_pairs"    : sum(r['matched_pairs'] for r in all_results),
        "total_unmatched_image"  : sum(r['unmatched_image_lines'] for r in all_results),
        "total_unmatched_text"   : sum(r['unmatched_text_lines'] for r in all_results),
        "per_folder"             : [
            {
                "folder_id"    : r['folder_id'],
                "image_lines"  : r['total_image_lines'],
                "text_lines"   : r['total_transcription_lines'],
                "matched"      : r['matched_pairs'],
                "period_px"    : r['detection_stats']['period_px'],
                "est_line_h_px": r['detection_stats']['est_line_h_px'],
                "acf_conf"     : r['detection_stats']['acf_confidence'],
                "method"       : r['detection_stats']['method'],
            }
            for r in all_results
        ],
    }
    with open(str(Path(output_dir) / "pipeline_summary.json"), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Pretty summary table
    print(f"\n{'═'*70}")
    print("PIPELINE COMPLETE")
    print(f"{'═'*70}")
    print(f"{'Folder':<8} {'Img':>5} {'Txt':>5} {'Match':>6} {'Period':>8} {'LineH':>7} {'Conf':>6} {'Method'}")
    print(f"{'─'*70}")
    for r in summary['per_folder']:
        print(f"  {r['folder_id']:<6} {r['image_lines']:>5} {r['text_lines']:>5} "
              f"{r['matched']:>6} {r['period_px']:>7}px {r['est_line_h_px']:>6}px "
              f"{r['acf_conf']:>6.2f}  {r['method']}")
    print(f"{'─'*70}")
    print(f"  Total matched pairs  : {summary['total_matched_pairs']}")
    print(f"  Unmatched image lines: {summary['total_unmatched_image']}")
    print(f"  Unmatched text lines : {summary['total_unmatched_text']}")
    print(f"  Output → {Path(output_dir).resolve()}")
    print(f"{'═'*70}\n")

    return summary


if __name__ == "__main__":
    run_pipeline()