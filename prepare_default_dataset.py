"""Prepare tiles for the default LROC dataset gallery.

This script is designed for very large TIFF/GeoTIFF files such as
LROC NAC ROI mosaics. It avoids `cv2.imread`, which often fails on
huge rasters because of OpenCV's pixel-count safety limit.

Examples
--------
python prepare_default_dataset.py --input ./NAC_ROI_TORICELILOA_E047S0284.tiff
python prepare_default_dataset.py --input ./NAC_ROI_TORICELILOA_E047S0284.tiff --tile-size 1024 --overlap 64
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tifffile import memmap


OUTPUT_DIR = Path("data/lroc_nac_roi_toriceliloa_tiles")
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def split_positions(length: int, tile_size: int, overlap: int) -> list[int]:
    step = max(tile_size - overlap, 1)
    positions = list(range(0, max(length - tile_size + 1, 1), step))
    if length > tile_size and (length - tile_size) not in positions:
        positions.append(length - tile_size)
    return positions


def normalize_tile_to_uint8(tile: np.ndarray) -> np.ndarray:
    """Convert one tile to uint8 for gallery display and Streamlit input.

    Strategy:
    - if already uint8: keep as-is
    - if multi-channel: convert to grayscale first
    - robustly rescale using 1st/99th percentiles to preserve crater contrast
    """
    if tile.ndim == 3:
        # If TIFF has bands/channels, reduce to first channel or grayscale-like mean.
        if tile.shape[2] == 1:
            tile = tile[:, :, 0]
        else:
            tile = tile.mean(axis=2)

    tile = np.asarray(tile)

    if tile.dtype == np.uint8:
        return tile

    tile = tile.astype(np.float32)
    p1, p99 = np.percentile(tile, [1, 99])
    if p99 <= p1:
        p1 = float(tile.min())
        p99 = float(tile.max())
    if p99 <= p1:
        return np.zeros(tile.shape, dtype=np.uint8)

    tile = np.clip((tile - p1) * 255.0 / (p99 - p1), 0, 255)
    return tile.astype(np.uint8)


def is_useful_tile(tile_u8: np.ndarray, std_threshold: float) -> bool:
    return float(tile_u8.std()) >= std_threshold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to the large source image")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--prefix", default="torricelli")
    parser.add_argument(
        "--std-threshold",
        type=float,
        default=5.0,
        help="Discard tiles with very low contrast/variation",
    )
    args = parser.parse_args()

    image_path = Path(args.input)
    if not image_path.exists() or image_path.suffix.lower() not in SUPPORTED_EXTS:
        raise SystemExit("Input image not found or unsupported format")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Memory-mapped access keeps the TIFF on disk and slices tiles lazily.
    image = memmap(str(image_path))
    if image is None:
        raise SystemExit("Could not memory-map input image")

    h, w = image.shape[:2]
    print(f"Loaded raster shape: {w} x {h} px")
    print(f"Saving tiles to: {OUTPUT_DIR.resolve()}")

    y_positions = split_positions(h, args.tile_size, args.overlap)
    x_positions = split_positions(w, args.tile_size, args.overlap)

    saved = 0
    skipped = 0

    for y in y_positions:
        for x in x_positions:
            tile = image[y:y + args.tile_size, x:x + args.tile_size]
            if tile.shape[0] != args.tile_size or tile.shape[1] != args.tile_size:
                continue

            tile_u8 = normalize_tile_to_uint8(tile)
            if not is_useful_tile(tile_u8, args.std_threshold):
                skipped += 1
                continue

            out_path = OUTPUT_DIR / f"{args.prefix}_y{y:05d}_x{x:05d}.png"
            ok = cv2.imwrite(str(out_path), tile_u8)
            if not ok:
                raise SystemExit(f"Failed to write tile: {out_path}")
            saved += 1

    print(f"Saved {saved} tile(s) to {OUTPUT_DIR}")
    print(f"Skipped {skipped} low-information tile(s)")


if __name__ == "__main__":
    main()
