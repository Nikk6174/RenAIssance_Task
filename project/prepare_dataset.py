"""
prepare_dataset.py
==================
Reads all mapping.json files from output3/{1..20}/ and builds:
  data/train.csv  — 80 % of pages (page-level split to avoid leakage)
  data/val.csv    — 20 % of pages

CSV columns:
  image_path    absolute path to the line crop PNG
  transcription ground-truth text for that line

Run once before training:
  python prepare_dataset.py
"""

import json
import csv
import random
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
OUTPUT_DIR   = "output3"      # where pipeline3.py saved crops + mapping.json
DATA_DIR     = "data"         # will be created
VAL_FRACTION = 0.20           # fraction of pages held out for validation
RANDOM_SEED  = 42
# ────────────────────────────────────────────────────────────────────────────


def load_all_pairs(output_dir: str) -> dict[int, list[dict]]:
    """
    Returns {folder_id: [{"image_path": ..., "transcription": ...}, ...]}
    Only keeps lines where matched=True (ground truth exists).
    """
    output_path = Path(output_dir)
    pages: dict[int, list[dict]] = {}

    for folder in sorted(output_path.iterdir()):
        mapping_file = folder / "mapping.json"
        if not mapping_file.exists():
            continue

        with open(mapping_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        folder_id = int(folder.name)
        pairs = []
        for line in data.get("lines", []):
            if not line.get("matched", False):
                continue
            transcription = line.get("transcription", "").strip()
            if not transcription:
                continue
            crop_file = folder / line["crop_file"]
            if not crop_file.exists():
                continue
            pairs.append({
                "image_path":    str(crop_file.resolve()),
                "transcription": transcription,
            })

        if pairs:
            pages[folder_id] = pairs
            print(f"  Page {folder_id:>2}: {len(pairs)} matched lines")

    return pages


def split_pages(pages: dict, val_fraction: float, seed: int):
    """Page-level train/val split — no line from the same page appears in both."""
    ids = sorted(pages.keys())
    random.seed(seed)
    random.shuffle(ids)
    n_val  = max(1, int(len(ids) * val_fraction))
    val_ids   = set(ids[:n_val])
    train_ids = set(ids[n_val:])
    return train_ids, val_ids


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "transcription"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {len(rows):>4} rows → {path}")


def main():
    print("=" * 55)
    print("  Dataset preparation")
    print("=" * 55)

    print(f"\nLoading pairs from {OUTPUT_DIR}/")
    pages = load_all_pairs(OUTPUT_DIR)

    if not pages:
        print("ERROR: no matched pairs found. Run pipeline3.py first.")
        return

    train_ids, val_ids = split_pages(pages, VAL_FRACTION, RANDOM_SEED)

    train_rows = [row for pid in sorted(train_ids) for row in pages[pid]]
    val_rows   = [row for pid in sorted(val_ids)   for row in pages[pid]]

    print(f"\nSplit: {len(train_ids)} train pages / {len(val_ids)} val pages")
    print(f"  train pages: {sorted(train_ids)}")
    print(f"  val   pages: {sorted(val_ids)}")

    print(f"\nWriting CSVs to {DATA_DIR}/")
    write_csv(Path(DATA_DIR) / "train.csv", train_rows)
    write_csv(Path(DATA_DIR) / "val.csv",   val_rows)

    print(f"\nTotal: {len(train_rows)} train lines, {len(val_rows)} val lines")
    print("Done. Next step: python train_trocr.py")


if __name__ == "__main__":
    main()