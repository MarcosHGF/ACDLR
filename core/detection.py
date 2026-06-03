from __future__ import annotations

"""
detection.py
------------
Detector clássico multi-escala para crateras lunares.

Objetivo
--------
Melhorar muito a precisão em relação ao fluxo anterior baseado em threshold
local + contornos. O detector agora usa quatro etapas mais estáveis:

1. filtro casado multi-escala (matched filter) com um template de cratera;
2. geração de candidatos por máximos locais na resposta do filtro;
3. refinamento local do centro e validação fotométrica/geométrica;
4. refinamento final com Hough local + non-max suppression mais forte.

Esse desenho não depende de rede neural e mantém o código legível, mas é
consideravelmente mais robusto para imagens lunares reais com relevo,
sombreamento e textura fina.
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .preprocessing import PreprocessResult


@dataclass(frozen=True)
class Candidate:
    x: float
    y: float
    radius: float
    response: float


@dataclass(frozen=True)
class Detection:
    x: float
    y: float
    radius: float
    score: float


def detect(
    prep: PreprocessResult,
    min_radius: int = 8,
    max_radius: int = 40,
    param1: int = 70,
    param2: int = 28,
    dp: float = 1.2,
) -> np.ndarray:
    """
    Retorna array Nx3 de (x, y, r).

    Parâmetros mantidos por compatibilidade com o app já existente.
    - param1 controla a agressividade das bordas do Canny.
    - param2 atua como "strictness": maior => menos crateras e mais confiança.
    - dp é mantido apenas por compatibilidade de interface.
    """
    del dp

    min_radius = int(max(4, min_radius))
    max_radius = int(max(min_radius + 2, max_radius))

    image_u8 = prep.sharpened
    image = image_u8.astype(np.float32) / 255.0

    edge_low = max(10, int(param1 * 0.50))
    edge_high = max(edge_low + 20, int(param1))
    edges = cv2.Canny(image_u8, edge_low, edge_high)
    edge_float = edges.astype(np.float32) / 255.0

    grad_base = cv2.GaussianBlur(image, (0, 0), sigmaX=1.0, sigmaY=1.0)
    grad_x = cv2.Sobel(grad_base, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(grad_base, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x * grad_x + grad_y * grad_y)

    strictness = float(np.clip((param2 - 28) / 20.0, -0.8, 2.5))

    candidates = _generate_candidates(
        image=image,
        grad_mag=grad_mag,
        min_radius=min_radius,
        max_radius=max_radius,
        strictness=strictness,
    )

    detections: list[Detection] = []
    for cand in candidates:
        refined = _refine_and_validate(
            image=image,
            image_u8=image_u8,
            grad_mag=grad_mag,
            edges=edges,
            edge_float=edge_float,
            candidate=cand,
            min_radius=min_radius,
            max_radius=max_radius,
            strictness=strictness,
        )
        if refined is not None:
            detections.append(refined)

    detections = _deduplicate(detections)

    if not detections:
        return np.empty((0, 3), dtype=int)

    circles = np.array(
        [[round(det.x), round(det.y), round(det.radius)] for det in detections],
        dtype=int,
    )
    return circles


def _generate_candidates(
    image: np.ndarray,
    grad_mag: np.ndarray,
    min_radius: int,
    max_radius: int,
    strictness: float,
) -> list[Candidate]:
    """
    Gera candidatos usando matched filter multi-escala.

    A ideia é procurar uma assinatura típica de cratera:
    centro escuro + aro mais brilhante + região externa levemente mais clara.
    """
    radii = _radius_schedule(min_radius, max_radius)
    candidates: list[Candidate] = []

    image_percentile = 99.84 + strictness * 0.13
    image_percentile = float(np.clip(image_percentile, 99.42, 99.95))

    edge_image = _normalise01(cv2.GaussianBlur(grad_mag, (0, 0), sigmaX=1.0, sigmaY=1.0))
    edge_percentile = 99.92 + strictness * 0.10
    edge_percentile = float(np.clip(edge_percentile, 99.55, 99.98))

    for radius in radii:
        kernel = _crater_kernel(radius)
        response = cv2.filter2D(image, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT)
        threshold = max(float(np.percentile(response, image_percentile)), 0.35)
        _append_response_candidates(
            candidates=candidates,
            response=response,
            radius=float(radius),
            threshold=threshold,
            width=image.shape[1],
            height=image.shape[0],
            response_scale=1.00,
        )

        edge_kernel = _ring_energy_kernel(radius)
        edge_response = cv2.filter2D(edge_image, cv2.CV_32F, edge_kernel, borderType=cv2.BORDER_REFLECT)
        edge_threshold = max(float(np.percentile(edge_response, edge_percentile)), 0.30)
        _append_response_candidates(
            candidates=candidates,
            response=edge_response,
            radius=float(radius),
            threshold=edge_threshold,
            width=image.shape[1],
            height=image.shape[0],
            response_scale=0.42,
        )

    # Ordena por resposta e já elimina excesso bruto de candidatos muito próximos.
    candidates.sort(key=lambda c: c.response, reverse=True)

    pruned: list[Candidate] = []
    for cand in candidates:
        duplicate = False
        for prev in pruned:
            dist = float(np.hypot(cand.x - prev.x, cand.y - prev.y))
            if dist < 0.55 * min(cand.radius, prev.radius):
                duplicate = True
                break
        if not duplicate:
            pruned.append(cand)
        if len(pruned) >= _candidate_limit(image.shape, strictness):
            break

    return pruned


def _append_response_candidates(
    candidates: list[Candidate],
    response: np.ndarray,
    radius: float,
    threshold: float,
    width: int,
    height: int,
    response_scale: float,
) -> None:
    local_max = response == cv2.dilate(response, np.ones((5, 5), np.float32))
    ys, xs = np.where(local_max & (response >= threshold))

    for y, x in zip(ys.tolist(), xs.tolist()):
        xf = float(x)
        yf = float(y)
        if _touches_border(xf, yf, radius, width, height, margin=4):
            continue
        candidates.append(
            Candidate(
                x=xf,
                y=yf,
                radius=radius,
                response=float(response[y, x]) * response_scale,
            )
        )


def _refine_and_validate(
    image: np.ndarray,
    image_u8: np.ndarray,
    grad_mag: np.ndarray,
    edges: np.ndarray,
    edge_float: np.ndarray,
    candidate: Candidate,
    min_radius: int,
    max_radius: int,
    strictness: float,
) -> Detection | None:
    """
    Faz pequena busca local no centro e depois valida a assinatura circular.
    """
    best: Detection | None = None

    radius_values = _local_radius_schedule(candidate.radius, min_radius, max_radius)
    shift = max(1, int(round(candidate.radius * 0.18)))
    limit = max(1, int(round(candidate.radius * 0.30)))

    for dy in range(-limit, limit + 1, shift):
        for dx in range(-limit, limit + 1, shift):
            x = candidate.x + dx
            y = candidate.y + dy

            for radius in radius_values:
                score = _score_circle(
                    image=image,
                    grad_mag=grad_mag,
                    edges=edges,
                    edge_float=edge_float,
                    x=x,
                    y=y,
                    radius=radius,
                    strictness=strictness,
                )
                if score is None:
                    continue

                det = Detection(x=x, y=y, radius=radius, score=score)
                if best is None or det.score > best.score:
                    best = det

    if best is None:
        return None

    hough_refined = _local_hough_refinement(image_u8, best, strictness)
    if hough_refined is not None:
        hough_score = _score_circle(
            image=image,
            grad_mag=grad_mag,
            edges=edges,
            edge_float=edge_float,
            x=hough_refined.x,
            y=hough_refined.y,
            radius=hough_refined.radius,
            strictness=strictness,
        )
        if hough_score is not None and hough_score >= best.score * 0.92:
            best = Detection(
                x=hough_refined.x,
                y=hough_refined.y,
                radius=hough_refined.radius,
                score=hough_score,
            )

    # Revalida o melhor candidato encontrado.
    final_score = _score_circle(
        image=image,
        grad_mag=grad_mag,
        edges=edges,
        edge_float=edge_float,
        x=best.x,
        y=best.y,
        radius=best.radius,
        strictness=strictness,
    )
    if final_score is None:
        return None

    # Penaliza círculos muito grandes e pouco confiáveis, que costumam ser encostas
    # ou sombras alongadas confundidas com crateras.
    large_penalty = 0.0
    if best.radius > 0.55 * max_radius:
        large_penalty = 0.10 * (best.radius - 0.55 * max_radius)

    score = final_score - large_penalty
    min_final_score = 2.48 + strictness * 0.32
    if score < min_final_score:
        return None

    return Detection(best.x, best.y, best.radius, score)


def _score_circle(
    image: np.ndarray,
    grad_mag: np.ndarray,
    edges: np.ndarray,
    edge_float: np.ndarray,
    x: float,
    y: float,
    radius: float,
    strictness: float,
) -> float | None:
    """
    Mede quão bem um círculo representa uma cratera.

    Critérios usados:
    - centro mais escuro que o aro
    - exterior também mais claro que o centro
    - suporte de gradiente/borda ao longo do aro
    - consistência angular do aro
    """
    h, w = image.shape[:2]
    if _touches_border(x, y, radius, w, h, margin=4):
        return None

    pad = int(math.ceil(1.8 * radius)) + 2
    x1 = max(int(x) - pad, 0)
    y1 = max(int(y) - pad, 0)
    x2 = min(int(x) + pad + 1, w)
    y2 = min(int(y) + pad + 1, h)

    patch = image[y1:y2, x1:x2]
    patch_grad = grad_mag[y1:y2, x1:x2]
    patch_edges = edges[y1:y2, x1:x2]

    if _looks_like_no_data_patch(patch):
        return None

    yy, xx = np.ogrid[y1:y2, x1:x2]
    dist = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)

    inner = dist <= 0.45 * radius
    rim = (dist >= 0.72 * radius) & (dist <= 1.10 * radius)
    outer = (dist >= 1.18 * radius) & (dist <= 1.60 * radius)

    min_inner_pixels = max(8, int(0.20 * math.pi * radius * radius))
    min_ring_pixels = max(14, int(0.35 * math.pi * radius * radius))
    if inner.sum() < min_inner_pixels or rim.sum() < min_ring_pixels or outer.sum() < min_ring_pixels:
        return None

    inner_mean = float(patch[inner].mean())
    rim_mean = float(patch[rim].mean())
    outer_mean = float(patch[outer].mean())

    inner_q35 = float(np.percentile(patch[inner], 35))
    inner_q50 = float(np.percentile(patch[inner], 50))
    rim_q75 = float(np.percentile(patch[rim], 75))

    contrast = rim_mean - inner_mean
    rim_highlight = rim_q75 - inner_q35
    outer_contrast = outer_mean - inner_mean
    edge_support = float(np.count_nonzero(patch_edges[rim])) / float(rim.sum())
    rim_gradient = float(patch_grad[rim].mean())
    inner_gradient = float(patch_grad[inner].mean())
    outer_gradient = float(patch_grad[outer].mean())
    rim_prominence = rim_gradient - 0.5 * (inner_gradient + outer_gradient)

    angle_count = 48
    angles = np.linspace(0.0, 2.0 * math.pi, angle_count, endpoint=False)
    ring_x = x + radius * np.cos(angles)
    ring_y = y + radius * np.sin(angles)

    ring_values = _bilinear_sample(image, ring_x, ring_y)
    ring_grad = _bilinear_sample(grad_mag, ring_x, ring_y)

    brightness_floor = inner_mean + 0.55 * max(contrast, 0.0)
    bright_fraction = float(np.mean(ring_values > brightness_floor))

    grad_threshold = float(np.percentile(patch_grad, 72))
    grad_fraction = float(np.mean(ring_grad >= grad_threshold))

    sectors = 12
    sector_values = ring_values.reshape(sectors, angle_count // sectors).mean(axis=1)
    bright_sector_fraction = float(
        np.mean((sector_values - inner_q50) > max(0.020, 0.42 * max(contrast, rim_highlight, 0.0)))
    )
    sector_coverage, edge_angle_fraction, grad_angle_fraction = _angular_support(
        grad_mag=grad_mag,
        edge_float=edge_float,
        x=x,
        y=y,
        radius=radius,
        grad_threshold=grad_threshold,
    )

    min_contrast = 0.070 + strictness * 0.018
    min_outer_contrast = 0.040 + strictness * 0.012
    min_bright_fraction = 0.52 + strictness * 0.050
    min_edge_support = 0.12 + strictness * 0.025
    min_sector_coverage = 0.50 + strictness * 0.060
    min_rim_prominence = 0.006 + strictness * 0.004

    min_contrast = float(np.clip(min_contrast, 0.040, 0.105))
    min_outer_contrast = float(np.clip(min_outer_contrast, 0.015, 0.070))
    min_bright_fraction = float(np.clip(min_bright_fraction, 0.38, 0.65))
    min_edge_support = float(np.clip(min_edge_support, 0.055, 0.18))
    min_sector_coverage = float(np.clip(min_sector_coverage, 0.32, 0.64))
    min_rim_prominence = float(np.clip(min_rim_prominence, 0.001, 0.014))

    if contrast < min_contrast and rim_highlight < min_contrast * 1.40:
        return None
    if outer_contrast < min_outer_contrast:
        return None
    if bright_fraction < min_bright_fraction and bright_sector_fraction < 0.36:
        return None
    if rim_prominence < min_rim_prominence and contrast < min_contrast * 1.25:
        return None
    if (
        edge_support < min_edge_support
        and rim_gradient < 0.10
        and sector_coverage < min_sector_coverage
    ):
        return None
    if sector_coverage < min_sector_coverage and bright_sector_fraction < 0.42:
        return None

    score = (
        contrast * 2.6
        + rim_highlight * 1.2
        + outer_contrast * 1.05
        + rim_gradient * 1.7
        + max(rim_prominence, 0.0) * 4.0
        + edge_support * 0.9
        + bright_fraction * 0.95
        + grad_fraction * 0.6
        + sector_coverage * 1.35
        + edge_angle_fraction * 0.45
        + grad_angle_fraction * 0.45
        + bright_sector_fraction * 0.75
        + radius * 0.012
    )
    return float(score)


def _local_hough_refinement(
    image_u8: np.ndarray,
    detection: Detection,
    strictness: float,
) -> Detection | None:
    """
    Hough local, em patch pequeno, apenas para alinhar centro/raio.
    Nao e usado como detector principal.
    """
    x = detection.x
    y = detection.y
    radius = detection.radius

    h, w = image_u8.shape[:2]
    pad = int(max(20, 1.6 * radius))
    x1 = max(int(x) - pad, 0)
    y1 = max(int(y) - pad, 0)
    x2 = min(int(x) + pad + 1, w)
    y2 = min(int(y) + pad + 1, h)
    patch = image_u8[y1:y2, x1:x2]

    acc_threshold = int(max(10, 12 + radius * 0.05 + max(strictness, 0.0) * 2.0))

    circles = cv2.HoughCircles(
        patch,
        cv2.HOUGH_GRADIENT,
        dp=1.1,
        minDist=max(8, int(radius * 0.70)),
        param1=80,
        param2=acc_threshold,
        minRadius=max(4, int(0.78 * radius)),
        maxRadius=max(int(0.78 * radius) + 2, int(1.20 * radius)),
    )
    if circles is None:
        return detection

    best_circle: tuple[float, float, float] | None = None
    best_penalty: float | None = None

    for cx, cy, rr in circles[0]:
        gx = float(x1 + cx)
        gy = float(y1 + cy)
        gr = float(rr)
        center_shift = float(np.hypot(gx - x, gy - y))

        if center_shift > 0.45 * radius:
            continue

        penalty = center_shift * 0.08 + abs(gr - radius) * 0.06
        if best_penalty is None or penalty < best_penalty:
            best_penalty = penalty
            best_circle = (gx, gy, gr)

    if best_circle is None:
        return detection

    gx, gy, gr = best_circle
    return Detection(gx, gy, gr, detection.score)


def _deduplicate(detections: list[Detection]) -> list[Detection]:
    if not detections:
        return []

    kept: list[Detection] = []
    for det in sorted(detections, key=lambda item: item.score, reverse=True):
        duplicate = False
        for prev in kept:
            dist = float(np.hypot(det.x - prev.x, det.y - prev.y))
            radius_ratio = abs(det.radius - prev.radius) / max(det.radius, prev.radius)

            # mesmo centro/mesma cratera em escalas diferentes
            if dist < 0.75 * min(det.radius, prev.radius):
                duplicate = True
                break

            # sobreposição muito forte com raios parecidos
            if dist < 0.48 * (det.radius + prev.radius) and radius_ratio < 0.72:
                duplicate = True
                break

        if not duplicate:
            kept.append(det)

    return kept


def _radius_schedule(min_radius: int, max_radius: int) -> np.ndarray:
    count = int(np.clip(round((max_radius - min_radius) / 8) + 6, 6, 14))
    radii = np.unique(np.round(np.geomspace(min_radius, max_radius, count))).astype(int)
    return radii


def _local_radius_schedule(radius: float, min_radius: int, max_radius: int) -> np.ndarray:
    values = np.array([0.82, 0.92, 1.00, 1.10, 1.22], dtype=np.float32) * float(radius)
    values = np.clip(np.round(values), min_radius, max_radius).astype(int)
    return np.unique(values)


def _crater_kernel(radius: int) -> np.ndarray:
    """
    Template simples e interpretável:
    - disco interno negativo (depressão)
    - anel positivo (aro)
    - anel externo levemente positivo para contexto local
    """
    pad = int(math.ceil(1.7 * radius))
    yy, xx = np.mgrid[-pad : pad + 1, -pad : pad + 1]
    dist = np.sqrt(xx.astype(np.float32) ** 2 + yy.astype(np.float32) ** 2)

    kernel = np.zeros_like(dist, dtype=np.float32)
    kernel[dist <= 0.45 * radius] = -1.0
    kernel[(dist >= 0.75 * radius) & (dist <= 1.08 * radius)] = 1.40
    kernel[(dist >= 1.18 * radius) & (dist <= 1.55 * radius)] = 0.35

    mask = kernel != 0.0
    kernel[mask] -= float(kernel[mask].mean())

    norm = float(np.sqrt(np.sum(kernel[mask] ** 2)))
    if norm > 0:
        kernel /= norm

    return kernel


def _ring_energy_kernel(radius: int) -> np.ndarray:
    """
    Annular kernel used over gradient magnitude.

    It does not care about bright/dark polarity, so it complements the crater
    template for partially lit or degraded rims.
    """
    pad = int(math.ceil(1.45 * radius))
    yy, xx = np.mgrid[-pad : pad + 1, -pad : pad + 1]
    dist = np.sqrt(xx.astype(np.float32) ** 2 + yy.astype(np.float32) ** 2)

    kernel = np.zeros_like(dist, dtype=np.float32)
    kernel[(dist >= 0.86 * radius) & (dist <= 1.16 * radius)] = 1.0
    kernel[(dist >= 0.48 * radius) & (dist <= 0.68 * radius)] = -0.25
    kernel[(dist >= 1.28 * radius) & (dist <= 1.45 * radius)] = -0.20

    mask = kernel != 0.0
    kernel[mask] -= float(kernel[mask].mean())

    norm = float(np.sqrt(np.sum(kernel[mask] ** 2)))
    if norm > 0:
        kernel /= norm

    return kernel


def _angular_support(
    grad_mag: np.ndarray,
    edge_float: np.ndarray,
    x: float,
    y: float,
    radius: float,
    grad_threshold: float,
) -> tuple[float, float, float]:
    angle_count = 72
    sectors = 12
    angles = np.linspace(0.0, 2.0 * math.pi, angle_count, endpoint=False)
    radial_factors = np.array([0.84, 0.94, 1.00, 1.08, 1.18], dtype=np.float32)

    grad_samples = []
    edge_samples = []
    for factor in radial_factors:
        xs = x + radius * float(factor) * np.cos(angles)
        ys = y + radius * float(factor) * np.sin(angles)
        grad_samples.append(_bilinear_sample(grad_mag, xs, ys))
        edge_samples.append(_bilinear_sample(edge_float, xs, ys))

    grad_stack = np.vstack(grad_samples)
    edge_stack = np.vstack(edge_samples)

    grad_by_angle = np.max(grad_stack, axis=0)
    edge_by_angle = np.max(edge_stack, axis=0)
    supported = (grad_by_angle >= grad_threshold) | (edge_by_angle >= 0.20)

    sector_support = supported.reshape(sectors, angle_count // sectors).mean(axis=1)
    sector_coverage = float(np.mean(sector_support >= 0.30))
    edge_angle_fraction = float(np.mean(edge_by_angle >= 0.20))
    grad_angle_fraction = float(np.mean(grad_by_angle >= grad_threshold))
    return sector_coverage, edge_angle_fraction, grad_angle_fraction


def _normalise01(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    lo = float(np.min(image))
    hi = float(np.max(image))
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    return (image - lo) / (hi - lo)


def _looks_like_no_data_patch(patch: np.ndarray) -> bool:
    dark_fraction = float(np.mean(patch <= 0.015))
    flat_dark = float(np.percentile(patch, 5)) <= 0.020
    return dark_fraction > 0.035 and flat_dark


def _candidate_limit(shape: tuple[int, ...], strictness: float) -> int:
    h, w = shape[:2]
    base = float(np.clip((h * w) / 8200.0, 90, 280))
    if strictness < 0:
        base *= 1.0 + min(abs(strictness), 0.8) * 2.0
    return int(np.clip(base, 90, 620))


def _bilinear_sample(image: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]

    xs = np.clip(xs.astype(np.float32), 0, w - 1)
    ys = np.clip(ys.astype(np.float32), 0, h - 1)

    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)

    wx = xs - x0
    wy = ys - y0

    return (
        (1.0 - wx) * (1.0 - wy) * image[y0, x0]
        + wx * (1.0 - wy) * image[y0, x1]
        + (1.0 - wx) * wy * image[y1, x0]
        + wx * wy * image[y1, x1]
    )


def _touches_border(
    x: float,
    y: float,
    radius: float,
    width: int,
    height: int,
    margin: int = 4,
) -> bool:
    return (
        x - radius - margin < 0
        or y - radius - margin < 0
        or x + radius + margin >= width
        or y + radius + margin >= height
    )
