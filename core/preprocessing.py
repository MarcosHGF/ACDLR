from __future__ import annotations
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PreprocessResult:
    gray: np.ndarray
    enhanced: np.ndarray
    denoised: np.ndarray
    sharpened: np.ndarray
    edge_hint: np.ndarray
    blurred: np.ndarray


def run(
    image: np.ndarray,
    clahe_clip: float = 2.2,
    clahe_grid: int = 8,
    blur_ksize: int = 5,
) -> PreprocessResult:
    """
    Pré-processamento com normalização de iluminação local.

    Ajuste importante:
    - antes do CLAHE, removemos variação lenta de iluminação/relevo com um
      blur largo. Isso reduz falsos positivos causados por encostas escuras
      e sombras amplas.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=12, sigmaY=12)
    flattened = cv2.addWeighted(gray, 1.35, background, -0.35, 0)
    flattened = cv2.normalize(flattened, None, 0, 255, cv2.NORM_MINMAX)

    clahe = cv2.createCLAHE(
        clipLimit=clahe_clip,
        tileGridSize=(clahe_grid, clahe_grid),
    )
    enhanced = clahe.apply(flattened)

    denoised = cv2.bilateralFilter(enhanced, d=7, sigmaColor=35, sigmaSpace=35)

    ksize = _ensure_odd(blur_ksize)
    blurred = cv2.GaussianBlur(denoised, (ksize, ksize), sigmaX=0)
    sharpened = cv2.addWeighted(denoised, 1.40, blurred, -0.40, 0)

    low, high = _auto_canny_thresholds(sharpened)
    edge_hint = cv2.Canny(sharpened, low, high)

    return PreprocessResult(
        gray=gray,
        enhanced=enhanced,
        denoised=denoised,
        sharpened=sharpened,
        edge_hint=edge_hint,
        blurred=blurred,
    )


def _ensure_odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


def _auto_canny_thresholds(image: np.ndarray) -> tuple[int, int]:
    median = float(np.median(image))
    low = int(max(0, 0.66 * median))
    high = int(min(255, 1.33 * median))
    if high <= low:
        high = min(low + 40, 255)
    return low, high
