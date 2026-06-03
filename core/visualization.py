from __future__ import annotations

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from .measurement import Crater
from .risk import RegionStats, LandingPoint


CRATER_RING = (90, 235, 110)          # verde
CRATER_CENTER = (255, 255, 0)         # amarelo/ciano
GRID_LINE = (200, 200, 200)           # cinza claro
BEST_BORDER = (0, 255, 0)             # verde forte
WARN_BORDER = (0, 165, 255)           # laranja
DANGER_BORDER = (0, 0, 220)           # vermelho
LANDING_POINT_COLOR = (255, 255, 255) # branco

_RISK_CMAP = LinearSegmentedColormap.from_list(
    "risk", ["#2ecc71", "#f1c40f", "#e74c3c"]
)


def draw_craters(
    image: np.ndarray,
    craters: list[Crater],
    ring_thickness: int = 2,
    show_legend: bool = True,
) -> np.ndarray:
    vis = _ensure_bgr(image)

    for crater in craters:
        cv2.circle(vis, (crater.cx, crater.cy), crater.radius_px, CRATER_RING, ring_thickness)
        cv2.circle(vis, (crater.cx, crater.cy), 2, CRATER_CENTER, -1)

    if show_legend:
        vis = _draw_legend(vis, include_landing_point=False)

    return vis


def draw_risk_grid(
    image: np.ndarray,
    score_matrix: np.ndarray,
    stats_grid: list[list[RegionStats]],
    grid_rows: int = 3,
    grid_cols: int = 3,
    alpha: float = 0.18,
) -> np.ndarray:
    vis = _ensure_bgr(image)
    h, w = vis.shape[:2]

    y_edges = np.linspace(0, h, grid_rows + 1, dtype=int)
    x_edges = np.linspace(0, w, grid_cols + 1, dtype=int)

    best_r, best_c = _best_cell(score_matrix)

    for r in range(grid_rows):
        for c in range(grid_cols):
            score = float(score_matrix[r, c])
            stats = stats_grid[r][c]

            x1, x2 = x_edges[c], x_edges[c + 1]
            y1, y2 = y_edges[r], y_edges[r + 1]

            fill_bgr = _score_to_bgr(score)
            overlay = vis.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), fill_bgr, -1)
            cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)

            is_best = (r == best_r and c == best_c)
            border_color = (
                BEST_BORDER if is_best
                else WARN_BORDER if stats.risk_label == "MEDIUM"
                else DANGER_BORDER if stats.risk_label == "HIGH"
                else GRID_LINE
            )
            thickness = 3 if is_best else 1
            cv2.rectangle(vis, (x1, y1), (x2, y2), border_color, thickness)

            _put_text(vis, f"Risk {score:.0f}", (x1 + 8, y1 + 24), scale=0.58)
            _put_text(vis, f"{stats.crater_count} craters", (x1 + 8, y1 + 46), scale=0.48)

            if is_best:
                _put_text(
                    vis,
                    "BEST ZONE",
                    (x1 + 8, y2 - 10),
                    scale=0.58,
                    color=(0, 255, 0),
                    bold=True,
                )

    return vis


def draw_final(
    image: np.ndarray,
    craters: list[Crater],
    score_matrix: np.ndarray,
    stats_grid: list[list[RegionStats]],
    grid_rows: int = 3,
    grid_cols: int = 3,
    landing_point: LandingPoint | None = None,
) -> np.ndarray:
    vis = draw_craters(image, craters, show_legend=False)
    vis = draw_risk_grid(vis, score_matrix, stats_grid, grid_rows, grid_cols)

    if landing_point is not None:
        vis = draw_landing_point(vis, landing_point)

    vis = _draw_legend(vis, include_landing_point=landing_point is not None)
    return vis


def draw_landing_point(
    image: np.ndarray,
    landing_point: LandingPoint,
) -> np.ndarray:
    vis = image.copy()
    x, y = landing_point.x, landing_point.y

    cv2.drawMarker(
        vis,
        (x, y),
        LANDING_POINT_COLOR,
        markerType=cv2.MARKER_CROSS,
        markerSize=28,
        thickness=2,
    )
    cv2.circle(vis, (x, y), 10, LANDING_POINT_COLOR, 2)
    cv2.circle(vis, (x, y), 3, LANDING_POINT_COLOR, -1)
    return vis


def risk_heatmap_figure(
    score_matrix: np.ndarray,
    stats_grid: list[list[RegionStats]],
) -> plt.Figure:
    rows, cols = score_matrix.shape
    best_r, best_c = _best_cell(score_matrix)

    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    im = ax.imshow(
        score_matrix,
        cmap=_RISK_CMAP,
        vmin=0,
        vmax=100,
        aspect="equal",
        interpolation="nearest",
    )

    for r in range(rows):
        for c in range(cols):
            score = score_matrix[r, c]
            stats = stats_grid[r][c]
            star = " ★" if (r == best_r and c == best_c) else ""
            label = f"{score:.0f}{star}\n{stats.crater_count} cr"
            ax.text(
                c,
                r,
                label,
                ha="center",
                va="center",
                color="white",
                fontsize=8.5,
                fontweight="bold",
            )

    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_xticklabels([f"Col {i+1}" for i in range(cols)], color="white")
    ax.set_yticklabels([f"Row {i+1}" for i in range(rows)], color="white")
    ax.tick_params(colors="white")

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("Risk Score", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_title("Landing Risk Heat-Map", color="white", fontsize=11, pad=10)
    fig.tight_layout()
    return fig


def _draw_legend(
    image: np.ndarray,
    include_landing_point: bool = True,
) -> np.ndarray:
    vis = image.copy()
    _, w = vis.shape[:2]

    header_h = 92 if include_landing_point else 68
    header = np.full((header_h, w, 3), 15, dtype=np.uint8)
    cv2.line(header, (0, header_h - 1), (w, header_h - 1), (70, 70, 70), 1)

    x1 = 14
    row_y = 25
    _put_text(header, "Legend", (x1, row_y), scale=0.55, bold=True)

    item_x = x1 + 92
    cv2.circle(header, (item_x, row_y - 4), 9, CRATER_RING, 2)
    cv2.circle(header, (item_x, row_y - 4), 2, CRATER_CENTER, -1)
    _put_text(header, "detected crater", (item_x + 18, row_y), scale=0.48)

    item_x = min(item_x + 190, max(14, w - 260))
    cv2.rectangle(header, (item_x, row_y - 14), (item_x + 22, row_y + 6), BEST_BORDER, 2)
    _put_text(header, "safest region", (item_x + 30, row_y), scale=0.48)

    row_y += 30
    item_x = x1
    cv2.rectangle(header, (item_x, row_y - 14), (item_x + 22, row_y + 6), WARN_BORDER, 2)
    cv2.rectangle(header, (item_x + 28, row_y - 14), (item_x + 50, row_y + 6), DANGER_BORDER, 2)
    _put_text(header, "higher risk", (item_x + 60, row_y), scale=0.48)

    if include_landing_point:
        item_x = min(x1 + 190, max(14, w - 280))
        cv2.drawMarker(
            header,
            (item_x + 10, row_y - 4),
            LANDING_POINT_COLOR,
            markerType=cv2.MARKER_CROSS,
            markerSize=18,
            thickness=2,
        )
        _put_text(header, "landing point", (item_x + 30, row_y), scale=0.48)

    return np.vstack([header, vis])


def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image.copy()


def _score_to_bgr(score: float) -> tuple[int, int, int]:
    t = float(score) / 100.0
    r = int(255 * t)
    g = int(255 * (1 - t))
    return (0, g, r)


def _best_cell(score_matrix: np.ndarray) -> tuple[int, int]:
    idx = int(np.argmin(score_matrix))
    return divmod(idx, score_matrix.shape[1])


def _put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float = 0.6,
    color: tuple[int, int, int] = (255, 255, 255),
    bold: bool = False,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2 if bold else 1
    shadow = (0, 0, 0)
    ox, oy = origin

    cv2.putText(image, text, (ox + 1, oy + 1), font, scale, shadow, thickness + 1)
    cv2.putText(image, text, (ox, oy), font, scale, color, thickness)
