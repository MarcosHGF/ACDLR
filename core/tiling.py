from __future__ import annotations
from dataclasses import dataclass

import numpy as np


@dataclass
class Tile:
    image: np.ndarray
    row: int
    col: int
    x1: int
    y1: int
    x2: int
    y2: int


def split(
    image: np.ndarray,
    grid_rows: int = 3,
    grid_cols: int = 3,
    overlap: int = 80,
    tile_size: int | None = None,
) -> list[Tile]:
    """
    Divide a imagem em tiles.

    Modos suportados:
    1) grade clássica:
       split(image, grid_rows=3, grid_cols=3, overlap=40)

    2) tiles fixos para processamento:
       split(image, tile_size=1024, overlap=80)
    """
    h, w = image.shape[:2]

    if tile_size is not None:
        stride = max(tile_size - overlap, 1)

        xs = list(range(0, max(w - tile_size + 1, 1), stride))
        ys = list(range(0, max(h - tile_size + 1, 1), stride))

        if w > tile_size and (w - tile_size) not in xs:
            xs.append(w - tile_size)
        if h > tile_size and (h - tile_size) not in ys:
            ys.append(h - tile_size)

        tiles: list[Tile] = []
        for r, y1 in enumerate(ys):
            for c, x1 in enumerate(xs):
                x2 = min(x1 + tile_size, w)
                y2 = min(y1 + tile_size, h)

                tiles.append(
                    Tile(
                        image=image[y1:y2, x1:x2],
                        row=r,
                        col=c,
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                    )
                )
        return tiles

    base_h = h // grid_rows
    base_w = w // grid_cols

    tiles: list[Tile] = []

    for r in range(grid_rows):
        for c in range(grid_cols):
            cx1 = c * base_w
            cy1 = r * base_h
            cx2 = w if c == grid_cols - 1 else (c + 1) * base_w
            cy2 = h if r == grid_rows - 1 else (r + 1) * base_h

            x1 = max(cx1 - overlap, 0)
            y1 = max(cy1 - overlap, 0)
            x2 = min(cx2 + overlap, w)
            y2 = min(cy2 + overlap, h)

            tiles.append(
                Tile(
                    image=image[y1:y2, x1:x2],
                    row=r,
                    col=c,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )

    return tiles


def to_global(circles_local: np.ndarray, tile: Tile) -> np.ndarray:
    if circles_local.size == 0:
        return circles_local

    shifted = circles_local.copy().astype(float)
    shifted[:, 0] += tile.x1
    shifted[:, 1] += tile.y1
    return shifted.astype(int)


def deduplicate(circles: np.ndarray, min_dist: float | None = None) -> np.ndarray:
    if circles.size == 0:
        return circles

    circles = np.asarray(circles, dtype=float)
    order = np.argsort(-circles[:, 2])
    circles = circles[order]

    kept: list[np.ndarray] = []

    for cand in circles:
        x, y, r = cand
        duplicate = False

        for prev in kept:
            px, py, pr = prev
            dist = float(np.hypot(x - px, y - py))
            threshold = min_dist if min_dist is not None else float(0.65 * min(r, pr))
            radius_ratio = abs(r - pr) / max(r, pr)

            if dist < threshold and radius_ratio < 0.35:
                duplicate = True
                break

        if not duplicate:
            kept.append(cand)

    return np.round(np.array(kept)).astype(int)