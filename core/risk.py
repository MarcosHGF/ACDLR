from __future__ import annotations
from dataclasses import dataclass

import cv2
import numpy as np

from .measurement import Crater


@dataclass
class RegionStats:
    row: int
    col: int
    crater_count: int
    density: float
    mean_radius_px: float
    largest_radius_px: float
    coverage_ratio: float
    raw_score: float
    risk_score: float
    risk_label: str


@dataclass
class LandingPoint:
    x: int
    y: int
    row: int
    col: int
    clearance_px: float
    clearance_m: float


def analyse(
    craters: list[Crater],
    image_shape: tuple[int, int] | tuple[int, int, int],
    grid_rows: int = 3,
    grid_cols: int = 3,
) -> tuple[np.ndarray, list[list[RegionStats]]]:
    """
    Calcula o risco por célula da grade.

    Retorna:
    - score_matrix: matriz [rows, cols] com risco normalizado de 0 a 100
    - stats_grid: matriz [rows][cols] com estatísticas detalhadas
    """
    H, W = image_shape[:2]
    cell_h = H / grid_rows
    cell_w = W / grid_cols
    cell_area = cell_h * cell_w if cell_h > 0 and cell_w > 0 else 1.0

    grid_craters: list[list[list[Crater]]] = [
        [[] for _ in range(grid_cols)] for _ in range(grid_rows)
    ]

    for crater in craters:
        row = min(int(crater.cy / cell_h), grid_rows - 1)
        col = min(int(crater.cx / cell_w), grid_cols - 1)
        grid_craters[row][col].append(crater)

    raw = np.zeros((grid_rows, grid_cols), dtype=float)

    for r in range(grid_rows):
        for c in range(grid_cols):
            cell = grid_craters[r][c]
            n = len(cell)

            if n == 0:
                raw[r, c] = 0.0
                continue

            density = n / cell_area * 10_000.0
            mean_r = float(np.mean([cr.radius_px for cr in cell]))
            largest_r = float(np.max([cr.radius_px for cr in cell]))
            coverage = float(np.sum([cr.area_px for cr in cell])) / cell_area

            raw[r, c] = (
                density * 0.30
                + mean_r * 0.20
                + largest_r * 0.35
                + coverage * 100.0 * 0.15
            )

    raw_min = float(raw.min())
    raw_max = float(raw.max())

    if raw_max > raw_min:
        norm = (raw - raw_min) / (raw_max - raw_min) * 100.0
    else:
        norm = np.zeros_like(raw)

    stats_grid: list[list[RegionStats]] = []

    for r in range(grid_rows):
        row_stats: list[RegionStats] = []
        for c in range(grid_cols):
            cell = grid_craters[r][c]
            n = len(cell)

            density = float(n / cell_area * 10_000.0) if n > 0 else 0.0
            mean_radius_px = float(np.mean([cr.radius_px for cr in cell])) if cell else 0.0
            largest_radius_px = float(np.max([cr.radius_px for cr in cell])) if cell else 0.0
            coverage_ratio = float(np.sum([cr.area_px for cr in cell]) / cell_area) if cell else 0.0
            score = float(norm[r, c])

            row_stats.append(
                RegionStats(
                    row=r,
                    col=c,
                    crater_count=n,
                    density=density,
                    mean_radius_px=mean_radius_px,
                    largest_radius_px=largest_radius_px,
                    coverage_ratio=coverage_ratio,
                    raw_score=float(raw[r, c]),
                    risk_score=score,
                    risk_label=_label(score),
                )
            )
        stats_grid.append(row_stats)

    return norm, stats_grid


def best_landing_cell(score_matrix: np.ndarray) -> tuple[int, int]:
    """
    Retorna (row, col) da menor pontuação de risco.
    """
    idx = int(np.argmin(score_matrix))
    return divmod(idx, score_matrix.shape[1])


def suggest_landing_point(
    craters: list[Crater],
    image_shape: tuple[int, int] | tuple[int, int, int],
    best_row: int,
    best_col: int,
    grid_rows: int = 3,
    grid_cols: int = 3,
    scale_m_per_px: float = 1.10,
    safety_factor: float = 1.25,
    border_padding_px: int = 12,
) -> LandingPoint:
    """
    Encontra o ponto mais livre de crateras dentro da melhor célula.
    Usa distance transform sobre uma máscara de segurança.
    """
    H, W = image_shape[:2]

    x_edges = np.linspace(0, W, grid_cols + 1, dtype=int)
    y_edges = np.linspace(0, H, grid_rows + 1, dtype=int)

    x1, x2 = x_edges[best_col], x_edges[best_col + 1]
    y1, y2 = y_edges[best_row], y_edges[best_row + 1]

    cell_w = max(x2 - x1, 1)
    cell_h = max(y2 - y1, 1)

    safe_mask = np.full((cell_h, cell_w), 255, dtype=np.uint8)

    pad = min(border_padding_px, max(min(cell_h, cell_w) // 4, 1))
    safe_mask[:pad, :] = 0
    safe_mask[-pad:, :] = 0
    safe_mask[:, :pad] = 0
    safe_mask[:, -pad:] = 0

    for crater in craters:
        expanded_r = int(crater.radius_px * safety_factor + 4)

        if crater.cx + expanded_r < x1 or crater.cx - expanded_r >= x2:
            continue
        if crater.cy + expanded_r < y1 or crater.cy - expanded_r >= y2:
            continue

        local_x = int(crater.cx - x1)
        local_y = int(crater.cy - y1)

        cv2.circle(safe_mask, (local_x, local_y), expanded_r, 0, -1)

    dist = cv2.distanceTransform(safe_mask, cv2.DIST_L2, 5)
    max_val = float(dist.max())

    if max_val <= 0:
        fallback_x = x1 + cell_w // 2
        fallback_y = y1 + cell_h // 2
        return LandingPoint(
            x=fallback_x,
            y=fallback_y,
            row=best_row,
            col=best_col,
            clearance_px=0.0,
            clearance_m=0.0,
        )

    local_y, local_x = np.unravel_index(np.argmax(dist), dist.shape)

    return LandingPoint(
        x=int(x1 + local_x),
        y=int(y1 + local_y),
        row=best_row,
        col=best_col,
        clearance_px=max_val,
        clearance_m=max_val * scale_m_per_px,
    )


def _label(score: float) -> str:
    if score < 33:
        return "LOW"
    if score < 66:
        return "MEDIUM"
    return "HIGH"