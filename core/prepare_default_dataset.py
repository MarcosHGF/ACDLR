"""Prepare tiles for the default LROC dataset gallery.

Usage:
    python prepare_default_dataset.py --input path/to/torricelli.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


OUTPUT_DIR = Path("data/lroc_nac_roi_toriceliloa_tiles")
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def split_into_tiles(image, tile_size: int, overlap: int):
    step = max(tile_size - overlap, 1)
    h, w = image.shape[:2]

    y_positions = list(range(0, max(h - tile_size + 1, 1), step))
    x_positions = list(range(0, max(w - tile_size + 1, 1), step))

    if h > tile_size and (h - tile_size) not in y_positions:
        y_positions.append(h - tile_size)
    if w > tile_size and (w - tile_size) not in x_positions:
        x_positions.append(w - tile_size)

    for y in y_positions:
        for x in x_positions:
            tile = image[y:y + tile_size, x:x + tile_size]
            if tile.shape[0] == tile_size and tile.shape[1] == tile_size:
                yield tile, y, x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to the large source image")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--prefix", default="torricelli")
    args = parser.parse_args()

    image_path = Path(args.input)
    if not image_path.exists() or image_path.suffix.lower() not in SUPPORTED_EXTS:
        raise SystemExit("Input image not found or unsupported format")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit("Could not load input image")

    count = 0
    for tile, y, x in split_into_tiles(image, args.tile_size, args.overlap):
        out_path = OUTPUT_DIR / f"{args.prefix}_y{y:05d}_x{x:05d}.png"
        cv2.imwrite(str(out_path), tile)
        count += 1

    print(f"Saved {count} tile(s) to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
