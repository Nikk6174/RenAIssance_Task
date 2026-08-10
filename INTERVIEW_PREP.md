# 🎯 Fischer Jordan Interview Prep — RenAIssance OCR Pipeline
### Estimated Reading Time: ~1.5 Hours (Consistent Reading)
### Total Questions: 175

---

> **How to use this file:**
> Read each question, pause, mentally formulate your answer, then read the suggested answer to fill gaps. The questions are ordered from foundational → deep technical → design decisions → hypothetical/behavioral. Treat this like a mock interview marathon.

---

## SECTION 1: PROJECT OVERVIEW & MOTIVATION (15 Questions)

---

**Q1. Give me a 60-second elevator pitch of your project.**

RenAIssance is a fully automated, multi-stage OCR pipeline purpose-built for digitising 17th-century Spanish printed texts. It takes a raw scanned page image and produces an accurate text transcription by chaining together adaptive line detection, a fine-tuned TrOCR model, deterministic rule-based corrections for archaic orthography, and optional LLM-based post-correction using T5 and Gemini. The system achieves 5-7% average CER (Character Error Rate) — a 63% improvement over EasyOCR and 47% over PyTesseract — with sub-1% CER on clean pages.

---

**Q2. Why can't you just use Tesseract or EasyOCR on these historical documents?**

Historical Spanish manuscripts present unique challenges that off-the-shelf OCR systems fundamentally cannot handle:
- **Orthographic instability**: Letters u/v are used interchangeably ("vno" = "uno"), as are f/s ("fervir" = "servir"). No modern OCR model is trained on these alternations.
- **Archaic characters**: The cedilla (ç) appears where modern Spanish uses z. Accents are applied sporadically.
- **Variable typography**: Line spacing, font weight, and ink density vary within a single page, breaking fixed-parameter line detection.
- **Degraded source material**: Ink bleed-through, foxing, faded characters, uneven margins.

Our benchmarks prove this — EasyOCR gets 18.92% CER and PyTesseract gets 11.33% on the same inputs where we achieve 5-7%.

---

**Q3. What is CER and how is it computed?**

CER (Character Error Rate) is the primary metric for OCR accuracy. It's computed as:

```
CER = Levenshtein_Edit_Distance(prediction, ground_truth) / len(ground_truth)
```

The Levenshtein edit distance counts the minimum number of single-character insertions, deletions, and substitutions needed to transform the prediction into the ground truth. A CER of 0% means a perfect transcription; 100% means every character is wrong. Our target was <10% and we achieved 5-7% (3% accent-normalised).

---

**Q4. What is WER and how does it differ from CER?**

WER (Word Error Rate) operates at the word level rather than character level. It counts the minimum number of word-level insertions, deletions, and substitutions divided by the number of words in the reference. WER is always >= CER because a single character error in a word counts the entire word as wrong. In our case, we achieved ~17% WER vs ~51% for EasyOCR and ~57% for PyTesseract.

---

**Q5. Walk me through the complete pipeline from input to output.**

1. **Input**: A raw scanned page image (PNG/JPG/TIFF, any resolution).
2. **Adaptive Line Detection** (pipeline3.py): The image is binarised (Otsu thresholding), a horizontal projection profile is computed, and autocorrelation finds the line period. Valley detection segments the image into individual text lines.
3. **Line Cropping**: Each detected line is cropped with adaptive padding (15% vertical, 5% horizontal of estimated line height).
4. **TrOCR Inference**: Each cropped line is passed through a fine-tuned VisionEncoderDecoder model (BEiT encoder + RoBERTa decoder) that generates the initial text transcription.
5. **Rule-Based Correction** (rule_corrector.py): Deterministic fixes — ç to z, u/v resolution, f/s resolution — all validated against a 1,763-word archaic Spanish dictionary.
6. **T5 Correction** (optional): A fine-tuned T5 seq2seq model attempts to fix residual OCR errors.
7. **Gemini Correction** (optional): Google Gemini 2.5 Flash API with ultra-conservative prompting attempts further correction.
8. **Dynamic Alignment + CER** (dynamic_align.py): Needleman-Wunsch alignment maps predictions to ground-truth lines, and CER is computed at every pipeline stage.

---

**Q6. Why did you choose a multi-stage pipeline instead of an end-to-end model?**

Three reasons:
1. **Modularity**: Each stage can be independently tested, improved, and toggled on/off. If T5 makes things worse, we disable it without touching TrOCR.
2. **Safety**: Deterministic rule-based corrections are provably safe (zero hallucination risk), which is critical for historical text where errors must be minimized.
3. **Observability**: Per-stage CER tracking shows exactly which component helps or hurts. This revealed that LLMs were counter-productive on our already-accurate TrOCR output.

An end-to-end model would be a black box — you couldn't tell whether a regression was from the vision encoder, decoder, or post-processing.

---

**Q7. What was the most surprising finding in your project?**

The most surprising finding was that applying LLMs (both Gemini and T5) to the already-accurate TrOCR output actually *increased* the error rate. Gemini raised CER from 7.17% to 7.75%, and T5 raised it to 8.46%. The LLMs' inherent bias toward modern Spanish spelling caused them to "modernise" valid archaic spellings, introducing more errors than they fixed. This was counter-intuitive because everyone assumes "more AI = better results."

---

**Q8. How does your pipeline compare to the reference pipeline that was provided?**

| Metric | Reference Pipeline | Our Pipeline |
|---|---|---|
| Raw OCR CER | 34.75% | 5-7% (avg) |
| Post-correction CER | 5.69% | ~3% (accent-normalised) |
| TrOCR training | 1 epoch, 5% of data | Full corpus, 30 epochs |
| Line detection | O(N squared) clustering | O(N) autocorrelation |
| GT alignment | Manual/static | Needleman-Wunsch |
| Self-calibrating | No | Yes |
| Public models | No | Yes (HuggingFace) |

Our raw TrOCR output (5-7%) is already comparable to their *final* post-LLM-corrected CER (5.69%), before we apply any corrections.

---

**Q9. What datasets did you use for training and evaluation?**

- **Training set**: 468 manually verified line-crop to transcription pairs from data/train.csv, produced by the pipeline3.py line detection on historical manuscript scans.
- **Validation set**: 84 line-crop to transcription pairs from data/val.csv.
- **T5 additional data**: 552 real TrOCR prediction-ground truth error pairs (from data/trocr_val_predictions.json), 2,340 synthetic error pairs, and 468 identity pairs.
- **Dictionary**: ~1,763 unique word forms compiled from the training corpus plus an embedded archaic Spanish vocabulary.

---

**Q10. Why is the dataset so small (only ~550 samples)?**

Historical manuscript data is inherently scarce — each line must be manually transcribed by domain experts who can read 17th-century Spanish. This is expensive, time-consuming, and requires specialised knowledge. To compensate:
1. **Data augmentation** during TrOCR training (brightness, contrast, sharpness, noise variations).
2. **Synthetic noise generation** for T5 training (OCR-specific errors, generic perturbations, word-level targeted noise).
3. **30 training epochs** with early stopping to let the model converge on the limited data.
4. **Transfer learning**: Starting from qantev/trocr-large-spanish, pre-trained on modern printed Spanish.

---

**Q11. How are the models deployed/distributed?**

Both fine-tuned models are hosted on Hugging Face and auto-download on first use:
- nikk6174/historical-spanish-trocr (~2.4 GB) — the TrOCR vision-language model
- nikk6174/historical-spanish-t5 (~1.2 GB) — the T5 error corrector

The Streamlit app checks for local models/trocr and models/t5 directories first. If they don't exist, it uses huggingface_hub to download the models with progress bars. This means zero-configuration setup — just pip install -r requirements.txt and streamlit run app.py.

---

**Q12. What hardware constraints did you design for?**

The entire system was optimised for an RTX 3050 Ti with only 4 GB VRAM:
- **Batch size = 1** for TrOCR training (with gradient accumulation = 16 for effective batch of 16).
- **Gradient checkpointing** enabled — trades ~30% more compute for ~35% VRAM reduction.
- **Adafactor optimizer** instead of AdamW — much more memory-efficient.
- **FP16 mixed precision** when CUDA is available.
- TrOCR training takes ~5.5 hours, T5 training takes ~15-20 minutes on this hardware.

---

**Q13. What is the Streamlit application and what does it offer?**

The Streamlit app (app.py) is an interactive web-based evaluation tool that provides:
- One-click analysis of any folder or uploaded document.
- Real-time model download with progress bars on first launch.
- Toggle switches for each correction stage (T5, Gemini).
- Side-by-side CER comparison at every pipeline stage.
- Visual line-by-line alignment with crop images.
- Annotated source images with detected line overlays.
- BLEU score and cosine similarity metrics.
- Color-coded CER indicators (green < 10%, yellow < 15%, red > 15%).

---

**Q14. What is BLEU score and why did you include it?**

BLEU (Bilingual Evaluation Understudy) is an n-gram precision metric originally designed for machine translation. It measures how many n-grams (1 to 4-grams) in the prediction match the reference, with a brevity penalty for short outputs. We included it as a complementary metric to CER/WER because:
- CER is character-level, BLEU is phrase-level — they capture different aspects of accuracy.
- BLEU is more sensitive to word-order preservation, which matters for downstream NLP tasks.
- Our pipeline achieves BLEU > 0.65 vs 0.25 (EasyOCR) and 0.30 (PyTesseract).

---

**Q15. What is cosine similarity and how did you compute it?**

We computed TF-IDF cosine similarity using character n-grams (2 to 4-grams) to measure textual similarity between predictions and ground truth. The TfidfVectorizer from sklearn with analyzer="char_wb" creates a vector representation of each text based on character n-gram frequencies, then cosine similarity measures the angle between the prediction and ground-truth vectors. A score of 1.0 means identical texts.

---

## SECTION 2: LINE DETECTION — THE O(N) ALGORITHM (20 Questions)

---

**Q16. Explain the autocorrelation-based line detection step by step.**

1. **Binarize** the image using Otsu's method (ink = white, background = black).
2. **Compute horizontal projection** H(y) — sum all pixel values row-by-row, producing a 1D signal of ink density per row.
3. **Autocorrelation (ACF)**: Compute the autocorrelation of H(y) with itself. The first significant peak in the ACF gives the dominant periodicity — the line period (font height + inter-line gap).
4. **Smooth** the projection using a moving average (window = 1/3 of estimated line height).
5. **Find valleys** in the smoothed projection using scipy.signal.find_peaks on the inverted signal, with distance=period/2 and a prominence threshold = 6% of median ink density.
6. **Segment**: Cut the image at valley positions to produce individual text-line regions.
7. **Post-process**: Absorb fragment segments (< 60% of line height) and filter noise/marginalia (< 20% of median area).

---

**Q17. What is the time complexity and why is it O(N)?**

- Binarization: O(W x H) — one pass over the image.
- Horizontal projection: O(W x H) — one pass summing rows.
- Autocorrelation: O(H) — computed on the 1D projection signal (not the full image).
- Valley detection: O(H) — single scan.
- Total: O(W x H) for pixel operations, O(H) for the segmentation logic.

The key insight is that we never compare pairs of objects — no hierarchical clustering (O(N squared)), no pairwise distance matrices. Everything is done via 1D signal processing on the projection profile.

---

**Q18. What was the previous approach (O(N squared)) and why was it worse?**

The reference pipeline used connected-component analysis followed by hierarchical agglomerative clustering to group components into lines:
1. Find all connected components (individual letter strokes).
2. Compute pairwise distances between all component centroids — O(N squared) where N is the number of components (thousands per page).
3. Cluster using agglomerative methods with manual distance thresholds.

This was worse because: O(N squared) scales poorly, requires manual threshold tuning, and breaks when component counts are high.

---

**Q19. What is Otsu's thresholding and why did you use it?**

Otsu's method automatically determines the optimal threshold to separate foreground (ink) from background (paper) by minimizing intra-class variance. Given the histogram of pixel intensities, it finds the threshold that best separates the bimodal distribution of ink and paper pixels.

We used it because it's automatic (no manual threshold tuning per document), robust to different paper colors and ink density, and combined with THRESH_BINARY_INV to make ink = white and background = black.

---

**Q20. What is a horizontal projection profile?**

A horizontal projection profile H(y) is a 1D array where each element H[y] is the sum of all pixel values in row y of the binarized image. For a document with text lines:
- Rows containing text have high values (many white/ink pixels).
- Rows in the gaps between lines have low values (mostly black/background).

This creates a waveform where peaks correspond to text lines and valleys correspond to inter-line gaps.

---

**Q21. What is autocorrelation and how does it find the line period?**

Autocorrelation (ACF) measures how similar a signal is to a time-shifted version of itself. For the horizontal projection:

```python
acf = np.correlate(p, p, mode='full')  # full cross-correlation with itself
acf = acf[len(acf) // 2:]              # keep only positive lags
acf = acf / acf[0]                      # normalize to [0, 1]
```

The first significant peak in the ACF corresponds to the lag at which the signal best repeats — this is the line period (font height + inter-line gap). The confidence is the ACF value at the detected period.

---

**Q22. What happens when autocorrelation fails?**

The system has a fallback hierarchy:

1. **Priority 1: Autocorrelation (ACF)** — Used when confidence >= 0.10 and the image has >= 4 lines.
2. **Priority 2: Connected-component 75th-percentile height** — The 75th percentile of CC heights multiplied by 1.5 estimates the line period.
3. **Priority 3: Image-height heuristic** — max(8, image_height // 50). Last resort for single-line images.

The method used is recorded in stats['method'] for transparency.

---

**Q23. Explain the valley detection step in detail.**

After smoothing the projection with a moving average:
1. **Invert** the smoothed signal: inv = -smoothed.
2. **Set prominence threshold**: median_val x 0.06.
3. **Set minimum distance**: max(4, period // 2) — valleys must be at least half a period apart.
4. **Find peaks** in the inverted signal using scipy.signal.find_peaks.
5. **Build segments**: The regions between consecutive valleys are the text lines.

---

**Q24. What is the fragment absorption step and why is it needed?**

Fragment absorption merges segments shorter than 60% of the estimated line height into the nearest full-size segment. This handles:
- **Strikethrough bars**: Thin horizontal ink bands causing false valleys.
- **Descenders**: Letters like g, y, p creating dips that cause false valleys.
- **Partial ascenders**: Tall letters at edges of crops getting split.

It doesn't collapse normal pages because real inter-line segments span ~75% of the period (well above 60% threshold).

---

**Q25. How is adaptive padding computed?**

- **Vertical padding**: max(2, int(est_line_h x 0.15)) — 15% of line height above and below.
- **Horizontal padding**: max(2, int(image_width x 0.05)) — 5% of image width on each side.

Adaptive because the line height is estimated per-page, so dense-text pages get smaller padding and large-font pages get larger padding.

---

**Q26. How do you handle noise and marginalia in the detected lines?**

After detecting all line segments:
1. Compute the area of each bounding box.
2. Compute the median area across all boxes.
3. Filter out any box with area < 20% of the median area (MARGIN_FILTER_RATIO = 0.20).

This removes noise artifacts, page numbers, and marginal annotations.

---

**Q27. What is find_peaks and how does it work?**

scipy.signal.find_peaks identifies local maxima in a 1D signal. Key parameters:
- **distance**: Minimum horizontal distance between peaks.
- **prominence**: Minimum vertical height a peak must rise above surrounding terrain.

For valley detection, we invert the signal and find peaks in the inverted version.

---

**Q28. How does the fallback threshold split work?**

When valley detection finds no splits:
1. Smooth with a larger window (est_line_h // 2).
2. Set threshold at 50% x 20th percentile of positive values.
3. Scan row by row: in_text when density >= threshold, gap when below for est_line_h // 5 consecutive rows.
4. Build segments from the text regions.

---

**Q29. Why do you use median blur before Otsu thresholding?**

cv2.medianBlur(gray, 3) removes salt-and-pepper noise (scanner artifacts, old paper speckle) while preserving text edges. Median blur is better than Gaussian for this because it preserves sharp edges of text strokes.

---

**Q30. What does self-calibrating mean?**

The detector automatically adapts to any font size, scan resolution, zoom level without manual parameter tuning. The ACF measures the actual periodic pattern in the data, unlike the reference pipeline which needed hand-tuned distance thresholds per document type.

---

**Q31. How do you draw the annotated source image?**

The draw_annotated function copies the image, draws green/red rectangles for each line, adds line numbers, and stamps detection statistics. Font scale is adaptive: max(0.35, min(0.8, lh / 30)).

---

**Q32. How does column_extent work?**

For a row band, extract the horizontal slice, compute column projection (sum along each column), find first and last non-zero columns. Returns the tight horizontal ink extent.

---

**Q33. What encoding issues did you handle?**

read_transcription tries UTF-8, Latin-1, ISO-8859-1, and CP1252 in order, handling transcription files from different text editors and operating systems.

---

**Q34. What image formats are supported?**

PNG, JPG, JPEG, TIF, TIFF, BMP, WebP. Image.MAX_IMAGE_PIXELS is set to 500 million to handle large manuscript scans.

---

**Q35. How does the moving average filter work?**

np.convolve with a uniform kernel of width w. Each output is the average of w neighbors. mode='same' preserves array length. Window = est_line_h // 3.

---

## SECTION 3: NEEDLEMAN-WUNSCH DYNAMIC ALIGNMENT (15 Questions)

---

**Q36. What problem does the dynamic alignment solve?**

When the line detector finds a different number of lines than the ground truth, naive 1-to-1 mapping breaks catastrophically — every subsequent pairing shifts, inflating CER by up to 60 percentage points. The alignment solves: which detected line corresponds to which ground-truth line?

---

**Q37. Explain the Needleman-Wunsch algorithm as you implemented it.**

1. **Build cost matrix**: For every (prediction_i, gt_j) pair, compute case-insensitive edit distance.
2. **DP table**: dp[i][j] = minimum total edit distance to align preds[0..i-1] with gt[0..j-1].
3. **Three transitions**:
   - MATCH: dp[i-1][j-1] + cost[i-1][j-1] — pair prediction i with GT j.
   - SKIP_PRED: dp[i-1][j] + len(pred[i]) — skip prediction i (penalty = 100% CER).
   - SKIP_GT: dp[i][j-1] + len(gt[j]) — skip GT line j (penalty = 100% CER).
4. **Traceback**: Walk back from dp[n][m] to dp[0][0] to recover the optimal alignment.

---

**Q38. Where does Needleman-Wunsch come from?**

Originally developed in 1970 for aligning protein/nucleotide sequences in bioinformatics. It finds the optimal global alignment by minimizing cost (or maximizing similarity). We adapted it for OCR by treating predictions and GT lines as sequences and using character edit distance as cost.

---

**Q39. What is the complexity?**

Time: O(N x M) where N = predictions, M = GT lines. Space: O(N x M). Documents rarely exceed 50 lines, so this is trivially fast.

---

**Q40. What are the skip penalties and why?**

Skip prediction penalty = len(pred) — treating the extra box as 100% CER. Skip GT penalty = len(gt_line). Proportional to string length, so short noise fragments have low penalty and the algorithm prefers to skip them over misaligning long text lines.

---

**Q41. How did alignment correction improve measured CER?**

From 61% (broken mapping) to 7% (true performance). The 61% was an artifact of shifted pairings. TrOCR was far more accurate than initial evaluations suggested.

---

**Q42. What does was_shifted indicate?**

When a prediction matches a GT line at a different index than expected (e.g., pred 3 matched to GT 5 instead of GT 3). Indicates extra detected boxes pushing the alignment.

---

**Q43. What does skipped indicate?**

The DP decided it's cheaper to skip this prediction entirely than to align it with any GT line. Typically headers, decorative elements, or noise.

---

**Q44. How does compute_aggregate_cer work?**

Two CER types: Total CER = sum(edit distances) / sum(GT lengths). Mean line CER = average of per-line CERs. Also counts shifted, skipped, and total lines.

---

**Q45. What is compute_sequential_cer?**

Naive 1-to-1 mapping CER (pred 1 to GT 1, pred 2 to GT 2). Computed alongside aligned CER to demonstrate the alignment improvement.

---

**Q46. Why exclude lines with CER > 80%?**

Almost certainly alignment failures, not OCR errors. Including them distorts the average. Displayed with "EXCLUDED" tag for transparency.

---

**Q47. How does the cost matrix compare strings?**

Both strings lowercased before edit distance. Prevents capitalization from causing misalignment. Actual CER computed on original-case strings.

---

**Q48. Could you use Smith-Waterman (local alignment) instead?**

No — Smith-Waterman finds best local alignment (subsequences), while we need global alignment (full sequences). We want to account for ALL predictions and GT lines, with skipped ones explicitly tracked.

---

**Q49. What is the traceback step?**

Starting from dp[n][m], walk backward: MATCH = record pair and move diagonal, SKIP_P = mark prediction skipped and move up, SKIP_G = move left. Reverse the alignment list at the end.

---

**Q50. How did you validate the alignment?**

Two built-in test cases: (1) Extra GT line — 3 preds vs 4 GT with one noise line. (2) Extra detected box — 4 preds vs 3 GT with one header. Both verify correct matching and skip behavior.

---

## SECTION 4: TrOCR MODEL & TRAINING (20 Questions)

---

**Q51. What is TrOCR and what is its architecture?**

TrOCR = VisionEncoderDecoder model. Encoder: BEiT (image transformer). Decoder: RoBERTa (text generator). Essentially an image captioning model specialised for text recognition. Base model: qantev/trocr-large-spanish.

---

**Q52. What hyperparameters did you use?**

Batch=1, grad_accum=16, epochs=30, LR=3e-5, cosine schedule, warmup=100, weight_decay=0.01, label_smoothing=0.1, beam=6, max_len=128, optimizer=Adafactor, gradient_checkpointing=True, early_stop=6.

---

**Q53. Explain the data augmentation strategy.**

During training: Brightness +/-20% (60%), Contrast +/-20% (60%), Sharpness (40%), Gaussian blur (25%), Gaussian noise (30%). Subtle augmentations to keep text readable. Each epoch sees different versions, multiplying effective data.

---

**Q54. Why Adafactor instead of AdamW?**

Adafactor uses factored second-moment representations — O(m+n) instead of O(mn) for weight matrices. Critical memory savings on a 4 GB GPU.

---

**Q55. What is gradient checkpointing?**

Trades compute for memory. Stores only subset of activations during forward pass, recomputes missing ones during backward. ~35% VRAM reduction, ~30% more compute. Essential for 4 GB GPU.

---

**Q56. What is label smoothing and why 0.1?**

Replaces hard 0/1 targets with soft distributions (0.9 correct, 0.1/vocab_size others). Prevents overconfidence on small training set. 0.1 is standard — effective regularisation without hurting accuracy.

---

**Q57. Explain the positional embedding device fix.**

TrOCR uses _float_tensor (plain attribute, not buffer) for device detection. .to(cuda) doesn't move it. Our fix patches forward() to read device from self.weight (real parameter that moves correctly). Handles both Learned and Sinusoidal variants.

---

**Q58. What is beam search and why beam width 6?**

Keeps top-k partial sequences at each generation step. Width 6 explores 6 candidates in parallel. Better accuracy than greedy (width 1) without being too slow.

---

**Q59. What base model and why?**

qantev/trocr-large-spanish — pre-trained on modern printed Spanish. Language match, large capacity, VisionEncoderDecoder architecture, enables transfer learning with minimal data.

---

**Q60. How did you save predictions for T5 training?**

After TrOCR training, run inference on validation set, save {ocr_output, ground_truth} pairs to trocr_val_predictions.json. Creates real TrOCR error training data for T5.

---

**Q61. What is cosine learning rate schedule?**

Gradually reduces LR following a cosine curve from initial (3e-5) to near zero. Smooth transition leads to better convergence on small datasets.

---

**Q62. What is no_repeat_ngram_size?**

Prevents generating any 3-gram already in the output. Prevents repetitive loops (e.g., "de de de de").

---

**Q63. What is length_penalty?**

Value 2.0 encourages longer sequences, preventing truncated transcriptions. Historical text lines can be quite long.

---

**Q64. How do you handle decoder tokens?**

CLS token = start of generation, SEP = end. Pad tokens in labels replaced with -100 (ignored in loss).

---

**Q65. Estimated training time?**

~5.3 hours on RTX 3050 Ti: 470 lines x 30 epochs / batch 1 x ~2.5 s/step.

---

**Q66. Training vs inference prediction?**

Training: Teacher forcing (GT tokens as input). Inference: Autoregressive generation with beam search.

---

**Q67. Final CER after fine-tuning?**

Average 5-7%, best 0.89% (clean pages), accent-normalised ~3%, WER ~17%.

---

**Q68. Why predict_with_generate=True?**

Evaluates using beam-search generation (not teacher forcing) — reflects real inference performance.

---

**Q69. Why Seq2SeqTrainer?**

Supports predict_with_generate, generation_num_beams, and correct decoder handling. Regular Trainer doesn't support autoregressive generation during evaluation.

---

**Q70. What does early stopping do?**

Monitors validation CER. Stops if no improvement for 6 consecutive epochs. Prevents overfitting and saves time.

---

## SECTION 5: RULE-BASED CORRECTOR (12 Questions)

---

**Q71. Why rule-based before LLMs?**

Deterministic and provably safe — zero hallucination risk. Fix easy known issues first, reducing error surface for LLMs.

---

**Q72. Explain each rule.**

1. **cedilla to z** (unconditional): 100% historical norm.
2. **u/v resolution**: Check all u-to-v and v-to-u variants against 1,763-word dictionary.
3. **f/s resolution**: Same dictionary-lookup approach for f-to-s and s-to-f.
4. **Combined**: If individual swaps fail, try all combinations of both.

---

**Q73. How is the SpanishDictionary constructed?**

Two sources: (1) Ground-truth words from train.csv and val.csv, accent-normalised. (2) ~1,763 embedded archaic and modern Spanish words including function words, legal/religious terms, proper nouns.

---

**Q74. Why strip accents for dictionary lookups?**

17th-century Spanish applied accents inconsistently. Without stripping, valid words would fail lookup due to different accent patterns.

---

**Q75. How does _generate_uv_variants work?**

Find all u/v positions. Generate all 2^n combinations. Return unique variants preserving casing. Limited to 5 or fewer positions.

---

**Q76. Why the 5-position limit?**

Avoids combinatorial explosion: 5 positions = 32 variants (fast), 10 = 1024 (slow), 15 = 32768 (unacceptable). Rare in practice.

---

**Q77. What is normalize_for_cer?**

Strips accents, lowercases, collapses whitespace. Used for accent-normalised CER metric.

---

**Q78. How does correction preserve casing?**

If original starts with uppercase and variant doesn't, capitalize the first letter of the variant.

---

**Q79. How is punctuation handled?**

Regex extracts leading/trailing punctuation, corrects the core alphabetic content, reattaches punctuation.

---

**Q80. Weaknesses of accent-normalised CER?**

Over-smoothing (el vs el), n-tilde risk (ano vs ano), not suitable for modern text, metric only (doesn't fix output).

---

**Q81. Did rules improve CER in practice?**

On the eval subset: maintained 7.17% (no improvement, no degradation). More impactful on pages with heavy u/v confusion. Key value: safety.

---

**Q82. How does strip_accents work internally?**

NFD decomposes characters into base + combining marks. Removes combining marks (category "Mn") except tilde and cedilla. NFC recomposes.

---

## SECTION 6: T5 CORRECTION MODEL (15 Questions)

---

**Q83. What is T5 and what base model?**

T5 = Text-to-Text Transfer Transformer. Base: jorgeortizfuentes/spanish-spellchecker-t5-base-wiki200000, pre-trained on Spanish Wikipedia spelling correction.

---

**Q84. Training data composition?**

552 real TrOCR errors, rule-corrected-to-GT pairs, ~2,340 synthetic noise pairs, 468 identity pairs. Total ~3,360.

---

**Q85. Why are identity pairs important?**

Without them, T5 learns every input needs correction, causing hallucinated changes on correct text. Identity pairs teach "sometimes do nothing."

---

**Q86. Three noise functions?**

1. introduce_errors_in_sentence: Word-level targeted noise (accent drops, visual confusables).
2. ocr_specific_errors: u/n, v/b, f/s confusion, accent changes, word merging, char doubling.
3. generic_perturb: Random insertions, deletions, substitutions, transpositions at 3% rate.

One function applied per sample (not stacked).

---

**Q87. What is the "correct: " prefix?**

Task signal prepended to all T5 inputs. Distinguishes correction from other text-to-text tasks. Must match at training and inference.

---

**Q88. Why did T5 perform poorly (8.46%)?**

The tokenizer fragments Spanish special characters (n-tilde, accented vowels) into sub-word pieces. Generation fails to reconstruct them correctly, producing artifacts like "Compania" to "Compaa".

---

**Q89. WordLevelCorrector?**

Processes token by token with rolling context window (N=3 preceding words). Batch processing for efficiency. Preserves newlines.

---

**Q90. UniversalSpellingCorrector?**

Wraps target word in brackets inside context: "correct: prev_words [target] next_words". Uses context both before AND after. More precise.

---

**Q91. T5 training hyperparameters?**

Epochs=10, batch=4, grad_accum=8, LR=3e-5, warmup=50, weight_decay=0.01, label_smoothing=0.1, max_len=128, early_stop=3.

---

**Q92. compute_all_metrics computes?**

CER, WER, cosine similarity, BLEU (1-4 gram with smoothing), Jaccard similarity. All self-implemented.

---

**Q93. TextChange dataclass?**

Captures substitutions: original text, corrected text, positions, context (+/-50 chars), is_ocr_error flag (same alphanumeric content = True).

---

**Q94. categorize_changes?**

Surface forms (OCR glyph confusion, same content) vs orthographic errors (genuinely different words).

---

**Q95. Train/val split for T5?**

90/10 random split after shuffling. Val set used for early stopping and metrics.

---

**Q96. Tokenisation function?**

Source and target tokenised separately with max_length=128, padding="max_length", truncation=True. Labels use -100 for padding.

---

**Q97. DataCollatorForSeq2Seq?**

Dynamically pads batches, replaces label pad tokens with -100.

---

## SECTION 7: GEMINI CORRECTOR & ANTI-HALLUCINATION (12 Questions)

---

**Q98. Why Gemini 2.5 Flash?**

Strong contextual understanding, supports temperature=0, Flash variant is fast and affordable.

---

**Q99. Ultra-conservative prompt design?**

8 critical rules: same word count, never add/remove words, never complete fragments, never expand abbreviations, keep hyphens, only single-char fixes, preserve archaic spelling, text-only output. Plus few-shot examples of correct AND incorrect behavior.

---

**Q100. Word-count guard effectiveness?**

Rejects output if word count differs from input. Caught ~50% of hallucinations (fragment completion, word merging).

---

**Q101. Why temperature=0?**

Deterministic output. No randomness = less hallucination risk. OCR correction needs precision, not creativity.

---

**Q102. Retry mechanism?**

5 retries with increasing backoff (15s, 30s, 45s...). Parses API error for exact retry time. Returns original text if all retries fail.

---

**Q103. Why did Gemini increase CER?**

Despite guards, still occasionally modernised spellings, completed fragments, changed grammar. Each increases edit distance from historical ground truth.

---

**Q104. Rate limiting strategy?**

6s sleep after each call (15 RPM free tier). Increasing backoff on retries. 13s sleep in CLI mode (5 RPM limit).

---

**Q105. Stripping markdown from Gemini output?**

Regex removes code fences and surrounding quotes that Gemini sometimes adds despite instructions.

---

**Q106. correct() vs correct_text()?**

correct() returns (text, status). correct_text() returns only the text string.

---

**Q107. Analysis functions?**

get_text_changes (token-level diff), print_changes (ANSI color output), categorize_changes (surface vs orthographic), compute_correction_metrics (CER/WER/cosine).

---

**Q108. Could you use a local LLM?**

Yes (Llama, Mistral, Phi). But the core finding is LLMs are counter-productive on accurate TrOCR output. Better approach: confidence-gated correction.

---

**Q109. Few-shot prompting approach?**

Explicit input/output examples showing correct behavior (minimal changes) AND incorrect behavior (over-correction). LLMs learn better from examples than instructions.

---

## SECTION 8: STREAMLIT APP & UI (8 Questions)

---

**Q110. Model download flow?**

Check local dir -> session_state cache -> HF cache (local_files_only) -> download with progress UI. Download only once.

---

**Q111. @st.cache_resource?**

Caches function return across reruns. Model loaded once, reused for all predictions. Prevents reloading 2.4 GB model on every click.

---

**Q112. Per-stage CER comparison?**

Same alignment used for all stages. CER computed separately for TrOCR/Rules/T5/Gemini/AccentNorm against matched GT. Color-coded cards.

---

**Q113. CER exclusion threshold?**

Lines with CER > 80% on final prediction are excluded (alignment failures). Displayed with EXCLUDED tag, don't affect metrics.

---

**Q114. Upload functionality?**

Users upload image + text file. Saved to temp directory. Full pipeline runs on them.

---

**Q115. Custom CSS?**

Dark gradient header, colored metric cards with left border, alignment table with dark header and CER color-coding, responsive columns, fit-to-screen images.

---

**Q116. Alignment table?**

Dynamic HTML table with line number, TrOCR/Rules/T5/Gemini/GT columns, CER (color-coded), edit distance, SHIFTED/SKIPPED tags.

---

**Q117. Transparency note?**

Shows detected lines, valid matches, excluded lines. Explains exactly which lines contribute to metrics.

---

## SECTION 9: DESIGN DECISIONS & TRADE-OFFS (12 Questions)

---

**Q118. Why TrOCR over alternatives?**

Transformer-based, pre-trained on Spanish, supports fine-tuning, VisionEncoderDecoder architecture. Tesseract (LSTM, no Spanish pretraining), EasyOCR (no fine-tuning), PaddleOCR (Chinese/English focus).

---

**Q119. Why evaluate both T5 and Gemini?**

Thorough comparison. T5: local, deterministic, no API costs, tokenizer issues. Gemini: better understanding, API-dependent, rate-limited. Neither improved accuracy — valuable negative result.

---

**Q120. With unlimited data, what differently?**

10K+ lines, custom tokenizer for T5, historical Spanish language model, multi-task training, curriculum learning.

---

**Q121. With more GPU memory?**

Larger batches, no gradient checkpointing, full fine-tuning, ensemble models, higher resolution inputs.

---

**Q122. Why is per-stage CER tracking important?**

Without it, can't tell which component helps. Revealed T5 (8.46%) and Gemini (7.75%) were WORSE than TrOCR alone (7.17%).

---

**Q123. Why open-source on HuggingFace?**

Reproducibility, zero-configuration, community benefit, transparency.

---

**Q124. Main failure modes?**

Severely degraded pages (8-10% CER), multi-column layouts, very short images, non-Spanish text, decorative elements.

---

**Q125. How handle multi-column layouts?**

YOLO-based region detection first, then per-region line detection, then reading order sort. Separates layout analysis from line segmentation.

---

**Q126. Confidence-gated LLM correction?**

Beam-search log-probabilities as confidence. High confidence (>90%) kept as-is. Only low-confidence sent to LLM. Eliminates T5 regression on correct lines.

---

**Q127. Active learning loop?**

Evaluate -> identify hardest lines (highest CER) -> add to training set -> re-fine-tune -> repeat. Self-improving system.

---

**Q128. Why no image augmentation for T5?**

T5 operates on text. Used noise functions instead (text equivalent of image augmentation).

---

**Q129. Significance of 0.89% CER?**

Near-perfect transcription on clean pages. Validates the model truly learned historical fonts, not just memorised patterns.

---

## SECTION 10: HYPOTHETICAL & BEHAVIORAL QUESTIONS (20 Questions)

---

**Q130. "Numbers seem too good" — how respond?**

All code/data/models public and reproducible. Per-stage tracking. Honest about LLM failures. Multiple metrics. Initial 61% CER debugging disclosed. Best vs average reported.

---

**Q131. Scale to 10,000 pages?**

Batch GPU processing for TrOCR. Disable LLMs (don't help). Pipeline parallelism (CPU+GPU). Distributed workers. Confidence filtering for human review.

---

**Q132. Adapt for different language/century?**

New training data, new base model, new dictionary, new orthographic rules, re-evaluate. Line detector should generalise.

---

**Q133. Client needs 99% accuracy?**

Define scope (clean pages: achievable). More data. Confidence-gated human review. Ensemble methods. Set expectations about physical ceiling.

---

**Q134. Debug a high-CER page?**

Check annotated image, alignment, individual lines, source quality, detection method, line count, manual character comparison.

---

**Q135. "Just use GPT-4 Vision"?**

Cost, latency, reproducibility, no per-line output, no CER tracking, would modernise spellings. Fine for prototyping, not production.

---

**Q136. Explain to non-technical person?**

"I built a computer system that reads old 1600s books automatically. It uses AI trained on old writing style. It's 63% more accurate than the best existing tool."

---

**Q137. Hardest technical challenge?**

Discovering and fixing the GT alignment problem. Initial 25-61% CER looked like model failure. Required implementing Needleman-Wunsch, careful validation, and re-evaluating all results.

---

**Q138. Three more months?**

Confidence-gated LLM (W1-3), multi-column detection (W4-6), larger training data (W7-9), cross-century generalisation (W10-12).

---

**Q139. Production readiness?**

Error handling, fallback hierarchy, input validation, caching, progress indicators, reproducibility, documentation.

---

**Q140. Testing performed?**

Unit tests (alignment, rule corrector), integration tests (full pipeline on multiple folders), comparative benchmarks (EasyOCR, PyTesseract), ablation studies, visual inspection.

---

**Q141. Project timeline management?**

Phase 1: Explore data. Phase 2: Line detection. Phase 3: TrOCR fine-tuning. Phase 4: Fix alignment. Phase 5: Rules + LLMs. Phase 6: Streamlit app. Phase 7: Report.

---

**Q142. Libraries used and why?**

PyTorch (HF ecosystem), Transformers (pre-trained models), OpenCV (image processing), SciPy (signal processing), Streamlit (rapid UI), editdistance (fast Levenshtein), sklearn (TF-IDF cosine).

---

**Q143. Live production deployment?**

Docker, FastAPI backend, GPU inference server, message queue, model versioning, CER monitoring, A/B testing.

---

**Q144. Security considerations?**

API key in env vars, file upload validation, model provenance, no PII in historical manuscripts.

---

**Q145. Ethical considerations?**

Accuracy transparency, bias from single document type, historical integrity vs normalisation, democratised access through open-sourcing.

---

**Q146. Changed approach mid-project?**

Pivoted from "LLMs fix everything" to "TrOCR is already good enough." Accepted the data, investigated why, pivoted the narrative, documented the negative result honestly.

---

**Q147. Stay current with research?**

ArXiv papers, ICDAR/ACL conferences, HuggingFace blog, Kaggle, Reddit r/MachineLearning.

---

**Q148. One feature to add?**

Confidence-gated LLM correction. Addresses core problem, technically feasible, recovers LLM benefit for hard lines, reduces API costs, measurable impact.

---

**Q149. First 90 days at the company?**

Week 1-2: Understand systems/data. Week 3-4: Setup and quick wins. Month 2: Deliver concrete improvement. Month 3: Propose and prototype new feature.

---

**Q150. Why this company?**

(Personalise based on research. Highlight: alignment with your ML pipeline skills, real-world data challenges, team culture, how RenAIssance demonstrates needed skills.)

---

## BONUS: RAPID-FIRE TECHNICAL DEFINITIONS

---

**Q151.** Transfer learning? Using a pre-trained model as starting point for a related task.

**Q152.** Fine-tuning? Training pre-trained weights on new smaller dataset.

**Q153.** Overfitting? Model memorises training data, poor on unseen data.

**Q154.** Levenshtein distance? Min single-char edits (insert/delete/substitute) between strings.

**Q155.** VisionEncoderDecoder? Architecture combining vision encoder with text decoder.

**Q156.** BEiT? Self-supervised vision transformer (encoder).

**Q157.** RoBERTa? Robustly optimized BERT, used as TrOCR's decoder.

**Q158.** Autoregressive generation? Generating tokens one at a time, each conditioned on previous.

**Q159.** Teacher forcing? Feeding ground-truth previous tokens during training.

**Q160.** FP16? 16-bit floats for some operations — less memory, faster compute.

**Q161.** Gradient accumulation? Accumulate gradients over mini-batches for effective larger batch.

**Q162.** Projection profile? 1D summary of 2D image by summing along one axis.

**Q163.** Otsu's method? Automatic thresholding minimizing intra-class variance.

**Q164.** Connected component analysis? Groups connected pixels into labeled regions.

**Q165.** OCR confusion matrix? Set of character pairs commonly confused (u/v, f/s, rn/m).

---

## SECTION 11: CURVEBALL QUESTIONS (10 Questions)

---

**Q166. OCR vs HTR?**

OCR = printed/typewritten text. HTR = handwritten. Our project is OCR (printed with movable type).

---

**Q167. Attention visualisation for debugging?**

Extract cross-attention between encoder patches and decoder tokens. Reveals where model "looks" during errors, whether confusion is from visual similarity.

---

**Q168. Rotated/skewed text?**

Hough transform for deskewing, perspective correction with OpenCV, applied before horizontal projection.

---

**Q169. Your pipeline vs Google Document AI?**

Domain-specific, self-hosted, transparent, historical-aware rules, smaller scale. Document AI is general-purpose cloud service.

---

**Q170. Detect labeling errors in training data?**

High-loss samples, confident wrong predictions, CER outliers, cross-validation, manual review via Streamlit.

---

**Q171. Character-level vs word-level model?**

Character: one char at a time (precise, slow). Word: subword tokens (faster, captures patterns). TrOCR uses BPE subwords — middle ground.

---

**Q172. Could this work on handwritten manuscripts?**

With modifications: different base model, wider line detection tolerance, much more training data, per-writer adaptation. Pipeline architecture still applies.

---

**Q173. Key dependencies?**

torch, transformers, huggingface_hub, opencv-python, streamlit, scipy, editdistance, evaluate, Pillow, scikit-learn, google-generativeai (optional).

---

**Q174. Handle empty TrOCR outputs?**

T5 corrector: return original if result empty. Empty TrOCR outputs passed through and marked as skipped in alignment.

---

**Q175. Biggest limitation?**

Single-column assumption. Multi-column, tables, marginalia, footnotes break horizontal projection. Needs layout analysis stage (YOLO-based) before line segmentation.

---

## FINAL TIPS FOR TOMORROW

1. **Know your numbers**: 5-7% CER, 0.89% best, 63% better than EasyOCR, 47% better than PyTesseract.
2. **Know your honest failures**: LLMs made things worse. Own it and explain why.
3. **Know the "why" behind every decision**: Why autocorrelation? Why Needleman-Wunsch? Why rule-based before LLM?
4. **Be ready to whiteboard**: The pipeline diagram, the DP table, the ACF concept.
5. **Show enthusiasm**: This is a genuinely interesting project with novel contributions.
6. **Ask questions back**: Show curiosity about their work and how your skills apply.

---

**Good luck tomorrow! You've got this! 🚀**
