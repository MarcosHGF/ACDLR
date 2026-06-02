from __future__ import annotations
from dataclasses import dataclass

import cv2
import numpy as np

from .measurement import Crater


DENSITY_WEIGHT = 30.0
MEAN_SIZE_WEIGHT = 20.0
LARGEST_SIZE_WEIGHT = 35.0
COVERAGE_WEIGHT = 15.0

DENSITY_SATURATION_PER_KM2 = 120.0
MEAN_DIAMETER_SATURATION_M = 80.0
LARGEST_DIAMETER_SATURATION_M = 140.0
COVERAGE_SATURATION_PERCENT = 8.0


@dataclass(frozen=True)
class RiskComponents:
    density: float
    mean_size: float
    largest_size: float
    coverage: float

    @property
    def total(self) -> float:
        return float(self.density + self.mean_size + self.largest_size + self.coverage)


@dataclass
class RegionStats:
    row: int
    col: int
    crater_count: int
    density: float
    density_per_km2: float
    mean_radius_px: float
    mean_diameter_m: float
    largest_radius_px: float
    largest_diameter_m: float
    coverage_ratio: float
    density_component: float
    mean_size_component: float
    largest_size_component: float
    coverage_component: float
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
    scale_m_per_px: float = 1.10,
) -> tuple[np.ndarray, list[list[RegionStats]]]:
    """
    Calcula o risco por célula da grade.

    Retorna:
    - score_matrix: matriz [rows, cols] com risco visual de 0 a 100
    - stats_grid: matriz [rows][cols] com estatísticas detalhadas

    O score usa limites fisicos fixos e interpretaveis em vez de normalizar
    sempre pelo pior tile da imagem. Isso torna resultados de imagens
    diferentes mais comparaveis sem recorrer a aprendizado profundo.
    """
    H, W = image_shape[:2]
    cell_h = H / grid_rows
    cell_w = W / grid_cols
    cell_area = cell_h * cell_w if cell_h > 0 and cell_w > 0 else 1.0
    cell_area_km2 = max(cell_area * (scale_m_per_px ** 2) / 1_000_000.0, 1e-9)

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
            density_per_km2 = n / cell_area_km2
            mean_diameter_m = float(np.mean([cr.diameter_m for cr in cell]))
            largest_diameter_m = float(np.max([cr.diameter_m for cr in cell]))
            coverage = float(np.sum([cr.area_px for cr in cell])) / cell_area

            components = _risk_components(
                density_per_km2=density_per_km2,
                mean_diameter_m=mean_diameter_m,
                largest_diameter_m=largest_diameter_m,
                coverage_ratio=coverage,
            )
            raw[r, c] = components.total

    score_matrix = np.clip(raw, 0.0, 100.0)

    stats_grid: list[list[RegionStats]] = []

    for r in range(grid_rows):
        row_stats: list[RegionStats] = []
        for c in range(grid_cols):
            cell = grid_craters[r][c]
            n = len(cell)

            density = float(n / cell_area * 10_000.0) if n > 0 else 0.0
            density_per_km2 = float(n / cell_area_km2) if n > 0 else 0.0
            mean_radius_px = float(np.mean([cr.radius_px for cr in cell])) if cell else 0.0
            mean_diameter_m = float(np.mean([cr.diameter_m for cr in cell])) if cell else 0.0
            largest_radius_px = float(np.max([cr.radius_px for cr in cell])) if cell else 0.0
            largest_diameter_m = float(np.max([cr.diameter_m for cr in cell])) if cell else 0.0
            coverage_ratio = float(np.sum([cr.area_px for cr in cell]) / cell_area) if cell else 0.0
            components = _risk_components(
                density_per_km2=density_per_km2,
                mean_diameter_m=mean_diameter_m,
                largest_diameter_m=largest_diameter_m,
                coverage_ratio=coverage_ratio,
            )
            score = float(score_matrix[r, c])

            row_stats.append(
                RegionStats(
                    row=r,
                    col=c,
                    crater_count=n,
                    density=density,
                    density_per_km2=density_per_km2,
                    mean_radius_px=mean_radius_px,
                    mean_diameter_m=mean_diameter_m,
                    largest_radius_px=largest_radius_px,
                    largest_diameter_m=largest_diameter_m,
                    coverage_ratio=coverage_ratio,
                    density_component=components.density,
                    mean_size_component=components.mean_size,
                    largest_size_component=components.largest_size,
                    coverage_component=components.coverage,
                    raw_score=float(raw[r, c]),
                    risk_score=score,
                    risk_label=_label(score),
                )
            )
        stats_grid.append(row_stats)

    return score_matrix, stats_grid


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


def _risk_components(
    density_per_km2: float,
    mean_diameter_m: float,
    largest_diameter_m: float,
    coverage_ratio: float,
) -> RiskComponents:
    """Heuristic 0-100 risk components based on physical quantities.

    The denominators are conservative defaults for the current LROC tile scale
    and can be tuned later with benchmark annotations.
    """
    density_component = (
        min(density_per_km2 / DENSITY_SATURATION_PER_KM2, 1.0)
        * DENSITY_WEIGHT
    )
    mean_size_component = (
        min(mean_diameter_m / MEAN_DIAMETER_SATURATION_M, 1.0)
        * MEAN_SIZE_WEIGHT
    )
    largest_size_component = (
        min(largest_diameter_m / LARGEST_DIAMETER_SATURATION_M, 1.0)
        * LARGEST_SIZE_WEIGHT
    )
    coverage_component = (
        min((coverage_ratio * 100.0) / COVERAGE_SATURATION_PERCENT, 1.0)
        * COVERAGE_WEIGHT
    )

    return RiskComponents(
        density=float(density_component),
        mean_size=float(mean_size_component),
        largest_size=float(largest_size_component),
        coverage=float(coverage_component),
    )
