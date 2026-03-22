# pyre-unsafe
"""
dynamic_align.py
================
Order-preserving dynamic alignment for TrOCR predictions vs ground truth.

Uses dynamic programming (Needleman-Wunsch style) to find the optimal
alignment between detected-line predictions and GT transcription lines.

Key capabilities:
  • Handles EXTRA detected boxes (titles, noise) — they get skipped
  • Handles MISSING detections (GT lines with no box) — they get skipped
  • Preserves top-to-bottom order — no swapping or crossing
  • Minimises total edit distance across the whole document

Usage
-----
    from dynamic_align import align_with_local_window, compute_aggregate_cer

    pairs = align_with_local_window(trocr_preds, gt_lines, window=5)
    for p in pairs:
        print(p["trocr_text"], "→", p["matched_gt"], f"CER={p['cer']:.2%}")
"""

from __future__ import annotations

from typing import Any

import editdistance  # type: ignore[import-untyped]


def _edit_dist(a: str, b: str) -> int:
    """Wrapper around editdistance.eval to satisfy type checkers."""
    result: Any = editdistance.eval(a, b)
    return int(result)


# ══════════════════════════════════════════════════════════════════════════════
# Core alignment  (DP — order-preserving)
# ══════════════════════════════════════════════════════════════════════════════

_MATCH  = 0   # traceback: matched pred[i] ↔ gt[j]
_SKIP_P = 1   # traceback: skipped prediction i  (extra detected box)
_SKIP_G = 2   # traceback: skipped GT line j      (missing detection)


def align_with_local_window(
    trocr_preds: list[str],
    gt_lines: list[str],
    window: int = 5,
) -> list[dict]:
    """
    Align TrOCR predictions to GT lines using order-preserving DP.

    The `window` parameter is accepted for API compatibility but the DP
    considers all pairwise costs (documents rarely exceed 50 lines so
    this is fast).

    Parameters
    ----------
    trocr_preds : list[str]
        TrOCR output for each detected line.
    gt_lines : list[str]
        Ground truth transcription lines.
    window : int
        Kept for backward compatibility; not used in the DP version.

    Returns
    -------
    list[dict]
        One entry per TrOCR prediction with keys:
          trocr_text, matched_gt, expected_gt_idx, matched_gt_idx,
          edit_distance, cer, was_shifted, skipped.
    """
    n = len(trocr_preds)   # number of predictions (detected boxes)
    m = len(gt_lines)      # number of GT lines

    # Edge cases
    if n == 0:
        return []
    if m == 0:
        return [
            _make_pair(i, pred, "", i, -1, len(pred), True)
            for i, pred in enumerate(trocr_preds)
        ]

    # ── Build cost matrix (edit distance for every plausible pair) ─────────
    # For efficiency we could band this, but documents < 100 lines → trivial
    cost: list[list[int]] = []
    for i in range(n):
        row: list[int] = []
        for j in range(m):
            row.append(_edit_dist(trocr_preds[i].lower(), gt_lines[j].lower()))
        cost.append(row)

    # ── DP table ──────────────────────────────────────────────────────────────
    # dp[i][j] = minimum total edit-distance cost to align
    #            preds[0..i-1]  with  gt[0..j-1]
    #
    # Transitions:
    #   MATCH:  dp[i][j] = dp[i-1][j-1] + cost[i-1][j-1]
    #   SKIP_P: dp[i][j] = dp[i-1][j]   + skip_pred_penalty
    #   SKIP_G: dp[i][j] = dp[i][j-1]   + skip_gt_penalty
    #
    # skip_pred_penalty = len(pred)  → as if CER = 100% for the extra box
    # skip_gt_penalty   = len(gt)    → as if CER = 100% for the missing line

    INF = float("inf")
    dp = [[INF] * (m + 1) for _ in range(n + 1)]
    tb = [[_MATCH] * (m + 1) for _ in range(n + 1)]  # traceback

    dp[0][0] = 0.0

    # Base case: skip all GT lines from the start (no predictions used yet)
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + len(gt_lines[j - 1])
        tb[0][j] = _SKIP_G

    # Fill DP
    for i in range(1, n + 1):
        skip_p_cost = len(trocr_preds[i - 1])  # penalty for skipping this pred

        # Base case: skip all predictions (no GT lines used yet)
        dp[i][0] = dp[i - 1][0] + skip_p_cost
        tb[i][0] = _SKIP_P

        for j in range(1, m + 1):
            # Option 1: Match pred i ↔ gt j
            c_match = dp[i - 1][j - 1] + cost[i - 1][j - 1]

            # Option 2: Skip prediction i (extra detected box)
            c_skip_p = dp[i - 1][j] + skip_p_cost

            # Option 3: Skip GT line j (missing detection)
            c_skip_g = dp[i][j - 1] + len(gt_lines[j - 1])

            best = min(c_match, c_skip_p, c_skip_g)
            dp[i][j] = best

            if best == c_match:
                tb[i][j] = _MATCH
            elif best == c_skip_p:
                tb[i][j] = _SKIP_P
            else:
                tb[i][j] = _SKIP_G

    # ── Traceback ─────────────────────────────────────────────────────────────
    # Walk back from dp[n][m] to dp[0][0] to recover the alignment.
    alignment: list[tuple[int, int]] = []   # (pred_idx, gt_idx) pairs
    skipped_preds: set[int] = set()

    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and tb[i][j] == _MATCH:
            alignment.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and tb[i][j] == _SKIP_P:
            skipped_preds.add(i - 1)
            i -= 1
        else:  # _SKIP_G
            j -= 1

    alignment.reverse()

    # ── Build result list (one entry per prediction) ──────────────────────────
    # Create a mapping from pred_idx → gt_idx
    pred_to_gt: dict[int, int] = {pi: gj for pi, gj in alignment}

    results: list[dict] = []
    for i in range(n):
        pred = trocr_preds[i]
        if i in pred_to_gt:
            gj = pred_to_gt[i]
            gt = gt_lines[gj]
            ed = cost[i][gj]
            results.append(_make_pair(i, pred, gt, i, gj, ed, False))
        else:
            # This prediction was skipped (extra detected box / noise)
            results.append(_make_pair(i, pred, "", i, -1, len(pred), True))

    return results


def _make_pair(
    idx: int, pred: str, gt: str,
    expected_idx: int, matched_idx: int,
    edit_dist: int, skipped: bool,
) -> dict:
    """Build one alignment-pair dict."""
    ref_len = max(len(gt), 1)
    cer = float(edit_dist) / ref_len
    return {
        "trocr_text": pred,
        "matched_gt": gt,
        "expected_gt_idx": expected_idx,
        "matched_gt_idx": matched_idx,
        "edit_distance": edit_dist,
        "cer": round(cer, 4),
        "was_shifted": (matched_idx != expected_idx) and (matched_idx >= 0),
        "skipped": skipped,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Aggregate metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_aggregate_cer(pairs: list[dict]) -> dict:
    """
    Compute aggregate CER from aligned pairs (skipped predictions excluded).

    Returns dict with:
      - total_cer     : float – total edit distance / total reference length
      - mean_line_cer : float – average per-line CER (matched lines only)
      - n_shifted     : int   – number of lines that were shifted
      - n_skipped     : int   – number of predictions with no GT match
      - n_total       : int   – total number of predictions
    """
    matched = [p for p in pairs if not p.get("skipped", False)]
    total_edit: int = sum(int(p["edit_distance"]) for p in matched)
    total_ref: int = sum(max(len(str(p["matched_gt"])), 1) for p in matched)
    n_shifted: int = sum(1 for p in matched if p["was_shifted"])
    n_skipped: int = sum(1 for p in pairs if p.get("skipped", False))

    n_matched = max(len(matched), 1)
    return {
        "total_cer": round(float(total_edit) / max(total_ref, 1), 4),
        "mean_line_cer": round(
            float(sum(float(p["cer"]) for p in matched)) / n_matched, 4
        ),
        "n_shifted": n_shifted,
        "n_skipped": n_skipped,
        "n_total": len(pairs),
    }


def compute_sequential_cer(trocr_preds: list[str], gt_lines: list[str]) -> float:
    """
    Compute CER using naive sequential (1-to-1) mapping for comparison.
    """
    total_edit: int = 0
    total_ref: int = 0
    n: int = min(len(trocr_preds), len(gt_lines))
    for i in range(n):
        total_edit += _edit_dist(trocr_preds[i], gt_lines[i])
        total_ref += max(len(gt_lines[i]), 1)
    return round(float(total_edit) / max(total_ref, 1), 4)


# ══════════════════════════════════════════════════════════════════════════════
# Self-test
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("TEST 1: Extra GT line (noise in transcription)")
    print("=" * 65)
    preds = ["hello world", "this is a test", "final line"]
    gt = ["hello world", "EXTRA NOISE LINE", "this is a test", "final line"]

    print(f"  Predictions ({len(preds)}): {preds}")
    print(f"  GT lines    ({len(gt)}):    {gt}")

    pairs = align_with_local_window(preds, gt, window=5)
    for p in pairs:
        tag = ""
        if p["skipped"]:
            tag = " ← SKIPPED (extra box)"
        elif p["was_shifted"]:
            tag = f" ← SHIFTED {p['expected_gt_idx']}→{p['matched_gt_idx']}"
        print(
            f"    [{p['matched_gt_idx']:>2}] "
            f"'{p['trocr_text']}' → '{p['matched_gt']}' "
            f"(CER={p['cer']:.0%}){tag}"
        )

    agg = compute_aggregate_cer(pairs)
    seq_cer = compute_sequential_cer(preds, gt)
    print(f"  Sequential CER: {seq_cer:.2%}")
    print(f"  Aligned CER:    {agg['total_cer']:.2%}")
    print(f"  Skipped:        {agg['n_skipped']}")

    print()
    print("=" * 65)
    print("TEST 2: Extra detected box (header/title not in GT)")
    print("=" * 65)
    preds2 = [
        "TITLE HEADER NOT IN GT",
        "cion Señor por sus rectissimos",
        "tos juyzios de darme tiempo",
        "que en vuestra educacion lo mani",
    ]
    gt2 = [
        "tro Señor por sus rectisimos",
        "tos juyzios de darme tiempo",
        "que en vuestra educacion lo mani",
    ]

    print(f"  Predictions ({len(preds2)}): {preds2}")
    print(f"  GT lines    ({len(gt2)}):    {gt2}")

    pairs2 = align_with_local_window(preds2, gt2, window=5)
    for p in pairs2:
        tag = ""
        if p["skipped"]:
            tag = " ← SKIPPED (extra box)"
        elif p["was_shifted"]:
            tag = f" ← SHIFTED {p['expected_gt_idx']}→{p['matched_gt_idx']}"
        print(
            f"    [{p['matched_gt_idx']:>2}] "
            f"'{p['trocr_text'][:40]}' → '{p['matched_gt'][:40]}' "
            f"(CER={p['cer']:.0%}){tag}"
        )

    agg2 = compute_aggregate_cer(pairs2)
    seq_cer2 = compute_sequential_cer(preds2, gt2)
    print(f"  Sequential CER: {seq_cer2:.2%}")
    print(f"  Aligned CER:    {agg2['total_cer']:.2%}")
    print(f"  Skipped:        {agg2['n_skipped']}")
