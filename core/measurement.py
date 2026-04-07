"""
measurement.py
--------------
Converts raw Hough detections (x, y, r) into structured crater
measurements, optionally converting pixels to physical distances.

Dataset used: LROC NAC ROI_TORICELILOA — scale ≈ 1.10 m/pixel.
"""

from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np


# Default scale for the LROC NAC dataset
SCALE_M_PER_PX: float = 1.10


# ---------------------------------------------------------------------------
# Data type
# ---------------------------------------------------------------------------

@dataclass
class Crater:
    """Physical and geometric properties of one detected crater."""
    cx:           int     # centre x (global image pixels)
    cy:           int     # centre y (global image pixels)
    radius_px:    int     # radius in pixels
    diameter_px:  int     # = radius_px * 2
    diameter_m:   float   # estimated physical diameter in metres
    area_px:      float   # approximate area in pixels²


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def measure(circles: np.ndarray,
            scale_m_per_px: float = SCALE_M_PER_PX) -> list[Crater]:
    """
    Convert a raw circles array into a list of Crater measurements.

    Parameters
    ----------
    circles        : Nx3 int array of (x, y, radius)
    scale_m_per_px : metres per pixel for the image dataset

    Returns
    -------
    List of Crater dataclass instances.
    """
    if circles.size == 0:
        return []

    craters: list[Crater] = []
    for cx, cy, r in circles:
        craters.append(Crater(
            cx=int(cx),
            cy=int(cy),
            radius_px=int(r),
            diameter_px=int(r * 2),
            diameter_m=float(r * 2 * scale_m_per_px),
            area_px=float(np.pi * r ** 2),
        ))

    return craters


def summary_stats(craters: list[Crater]) -> dict:
    """
    Compute aggregate statistics for a list of craters.

    Returns a dict with keys:
      count, mean_diameter_m, max_diameter_m, min_diameter_m,
      mean_radius_px, total_area_px
    """
    if not craters:
        return {
            "count": 0,
            "mean_diameter_m": 0.0,
            "max_diameter_m":  0.0,
            "min_diameter_m":  0.0,
            "mean_radius_px":  0.0,
            "total_area_px":   0.0,
        }

    diameters = [c.diameter_m  for c in craters]
    radii     = [c.radius_px   for c in craters]
    areas     = [c.area_px     for c in craters]

    return {
        "count":           len(craters),
        "mean_diameter_m": float(np.mean(diameters)),
        "max_diameter_m":  float(np.max(diameters)),
        "min_diameter_m":  float(np.min(diameters)),
        "mean_radius_px":  float(np.mean(radii)),
        "total_area_px":   float(np.sum(areas)),
    }
