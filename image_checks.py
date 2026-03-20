"""
image_checks.py — Image quality and fundal image detection.
All functions take a PIL.Image and return (ok: bool, reason: str).
"""
import cv2
import numpy as np
from PIL import Image


# ── Thresholds (very lenient — only reject truly unusable images) ─────────────
BLUR_THRESHOLD        = 5.0    # Laplacian variance below this → essentially featureless image
MIN_DIM               = 100    # Minimum width/height in pixels
BRIGHTNESS_LOW        = 5     # Mean pixel value below this → nearly pure black
BRIGHTNESS_HIGH       = 252    # Mean pixel value above this → nearly pure white
DARK_BORDER_MIN_RATIO = 0.03   # ≥3% near-black pixels expected in fundal images
DARK_PIXEL_THRESH     = 20     # Pixel value (0-255) considered "near-black"


def check_quality(pil_image: Image.Image) -> tuple[bool, str]:
    """
    Check that the image has sufficient quality for DR analysis.
    Returns (True, "") if OK, or (False, rejection_message) if not.
    """
    img = np.array(pil_image.convert("RGB"))
    h, w = img.shape[:2]

    # 1. Minimum size
    if h < MIN_DIM or w < MIN_DIM:
        return False, (
            f"❌ Image is too small ({w}×{h} px). "
            f"Please upload an image of at least {MIN_DIM}×{MIN_DIM} pixels."
        )

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2. Brightness — only reject pure black or pure white images
    mean_brightness = float(gray.mean())
    if mean_brightness < BRIGHTNESS_LOW:
        return False, "❌ Image is completely black. Please upload a valid fundal photograph."
    if mean_brightness > BRIGHTNESS_HIGH:
        return False, "❌ Image appears to be blank/all-white. Please upload a valid fundal photograph."

    # 3. Blur — only reject completely featureless images (solid colour, etc.)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if lap_var < BLUR_THRESHOLD:
        return False, (
            "❌ Image appears to contain no visual information (featureless). "
            "Please upload a real fundal photograph."
        )

    return True, ""


def is_fundal(pil_image: Image.Image) -> tuple[bool, str]:
    """
    Heuristic check: does this look like a fundal/retinal photograph?

    Fundal images from ophthalmoscopy equipment have a characteristic
    feature: a small percentage of pixels are near-black (the circular
    vignette / equipment border). Normal fully-white images or pure
    gradients won't have this.

    This check is intentionally very lenient — only rejects images
    with virtually no dark pixels at all (e.g. solid white, screenshots
    of plain documents).

    Returns (True, "") if likely fundal, (False, rejection_message) if not.
    """
    img  = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    total_pixels      = gray.size
    dark_pixels       = int(np.sum(gray < DARK_PIXEL_THRESH))
    dark_border_ratio = dark_pixels / total_pixels

    if dark_border_ratio < DARK_BORDER_MIN_RATIO:
        return False, (
            "❌ This does not appear to be a fundal (retinal) image — "
            "it looks like a plain document, screenshot, or photo without the "
            "characteristic dark border of fundal photography equipment. "
            "Please upload a photograph taken with an ophthalmoscope or fundal camera."
        )

    return True, ""
