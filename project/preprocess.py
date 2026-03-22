import cv2
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from kraken.binarization import nlbin


def deskew(img_array: np.ndarray) -> np.ndarray:
    """
    Detect and correct page skew using Hough line detection.
    Only corrects if skew is meaningful (>0.5 degrees) to avoid
    introducing artifacts on already-straight pages.
    """
    gray = (
        cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        if len(img_array.shape) == 3
        else img_array
    )

    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180,
        threshold=100, minLineLength=100, maxLineGap=10
    )

    if lines is None:
        return img_array

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        if dx != 0:
            angle = np.arctan2(y2 - y1, dx) * 180 / np.pi
            if -45 < angle < 45:
                angles.append(angle)

    if not angles:
        return img_array

    median_angle = float(np.median(angles))

    # Skip correction if angle is negligible
    if abs(median_angle) < 0.5:
        return img_array

    h, w = img_array.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(
        img_array, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


def normalize_contrast(img_array: np.ndarray) -> np.ndarray:
    """
    Per-image adaptive contrast stretch (CLAHE).
    Works on each page independently so faded pages are
    brought up to the same ink/background ratio as clean ones.
    """
    if len(img_array.shape) == 3:
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img_array)


def suppress_bleedthrough(img_array: np.ndarray) -> np.ndarray:
    """
    Light morphological operation to suppress faint bleed-through
    from the reverse side of photographed historical pages.
    """
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array

    # Estimate background using a large morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    # Subtract bleed-through estimate
    suppressed = cv2.subtract(background, gray)
    suppressed = cv2.bitwise_not(suppressed)

    if len(img_array.shape) == 3:
        return cv2.cvtColor(suppressed, cv2.COLOR_GRAY2RGB)
    return suppressed


def preprocess(image_path: str) -> Image.Image:
    """
    Full adaptive preprocessing pipeline for a single page image.

    Steps:
        1. Load (handles both PNG and JPG)
        2. Deskew
        3. Adaptive contrast normalization (CLAHE)
        4. Bleed-through suppression
        5. Kraken nlbin adaptive binarization

    Returns:
        PIL Image in '1' (binary) mode, ready for Kraken segmentation.
    """
    img = Image.open(image_path).convert('RGB')
    arr = np.array(img)

    arr = deskew(arr)
    arr = normalize_contrast(arr)
    arr = suppress_bleedthrough(arr)

    pil = Image.fromarray(arr).convert('L')   # grayscale for nlbin
    binary = nlbin(pil)                        # Kraken adaptive binarization
    return binary