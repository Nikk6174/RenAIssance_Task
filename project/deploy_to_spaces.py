"""
Deploy the Streamlit OCR app to Hugging Face Spaces.
Usage:
    python deploy_to_spaces.py --token YOUR_HF_TOKEN
"""
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo

SPACE_REPO = "nikk6174/historical-spanish-ocr"

# Files to upload (relative to project/)
APP_FILES = [
    "app.py",
    "pipeline3.py",
    "dynamic_align.py",
    "rule_corrector.py",
    "run_ocr.py",
    "eval_t5.py",
    "gemini_corrector.py",
]

# Space-specific README (YAML frontmatter for HF Spaces)
SPACE_README = """---
title: Historical Spanish OCR Pipeline
emoji: 🔍
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.44.1"
app_file: app.py
pinned: true
---

# 🔍 Historical Spanish OCR Pipeline

An interactive OCR evaluation tool for 17th-century Spanish manuscripts.

## Features
- **TrOCR** fine-tuned on historical Spanish text (5–7% CER)
- **Rule-based correction** with archaic Spanish dictionary
- **T5 post-correction** trained on real TrOCR errors
- **Dynamic ground-truth alignment** (Needleman-Wunsch)
- **Accent-normalised CER** metric

## Usage
1. Upload a full-page document image
2. Upload the ground-truth transcription (.txt, one line per text line)
3. Click "Run Analysis"

Models are automatically downloaded from Hugging Face on first use.
"""

# Space-specific requirements (lighter than full project)
SPACE_REQUIREMENTS = """# Core
opencv-python-headless>=4.8.0
scipy>=1.11.0
Pillow>=10.0.0
numpy>=1.24.0

# Deep learning
torch>=2.0.0
torchvision>=0.15.0

# Hugging Face
transformers>=4.40.0
huggingface_hub>=0.20.0

# Metrics
editdistance>=0.6.0
jiwer>=3.0.3
evaluate>=0.4.1

# Web app
streamlit>=1.30.0
python-docx>=1.0.0

# Misc
scikit-learn>=1.3.0
"""


def main():
    parser = argparse.ArgumentParser(description="Deploy to HF Spaces")
    parser.add_argument("--token", required=True, help="HuggingFace token")
    args = parser.parse_args()

    api = HfApi(token=args.token)

    # 1. Create the Space repo (SDK is set via README YAML, not here)
    print(f"Creating Space: {SPACE_REPO}")
    try:
        create_repo(
            repo_id=SPACE_REPO,
            repo_type="space",
            space_sdk="static",  # placeholder — overridden by README YAML
            token=args.token,
            exist_ok=True,
        )
        print("  ✅ Space repo created/exists")
    except Exception as e:
        print(f"  ⚠️ {e} — trying to continue anyway...")

    # 2. Upload Space README (with YAML config — this sets SDK to streamlit)
    print("Uploading Space config (README.md)...")
    api.upload_file(
        path_or_fileobj=SPACE_README.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=SPACE_REPO,
        repo_type="space",
    )
    print("  ✅ README.md")

    # 3. Upload requirements.txt
    print("Uploading requirements.txt...")
    api.upload_file(
        path_or_fileobj=SPACE_REQUIREMENTS.encode("utf-8"),
        path_in_repo="requirements.txt",
        repo_id=SPACE_REPO,
        repo_type="space",
    )
    print("  ✅ requirements.txt")

    # 4. Upload app files
    for fname in APP_FILES:
        fpath = Path(fname)
        if not fpath.exists():
            print(f"  ⚠️ SKIP {fname} (not found)")
            continue
        print(f"Uploading {fname}...")
        api.upload_file(
            path_or_fileobj=str(fpath),
            path_in_repo=fname,
            repo_id=SPACE_REPO,
            repo_type="space",
        )
        print(f"  ✅ {fname}")

    # 5. Upload input/ folder (test data — ~382 MB)
    input_dir = Path("input")
    if input_dir.exists() and input_dir.is_dir():
        print(f"\nUploading input/ folder (~382 MB, this may take a few minutes)...")
        api.upload_folder(
            folder_path=str(input_dir),
            path_in_repo="input",
            repo_id=SPACE_REPO,
            repo_type="space",
        )
        print("  ✅ input/ folder uploaded")
    else:
        print("  ⚠️ input/ folder not found — skipping")

    print(f"\n{'='*60}")
    print(f"🚀 Deployment complete!")
    print(f"   URL: https://huggingface.co/spaces/{SPACE_REPO}")
    print(f"{'='*60}")
    print(f"\nThe Space will now build and start automatically.")
    print(f"First run will download models (~3.6 GB) from HF Hub.")
    print(f"This may take 5-10 minutes on the first launch.")


if __name__ == "__main__":
    main()
