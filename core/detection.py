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

    grad_base = cv2.GaussianBlur(image, (0, 0), sigmaX=1.0, sigmaY=1.0)
    grad_x = cv2.Sobel(grad_base, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(grad_base, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x * grad_x + grad_y * grad_y)

    strictness = float(np.clip((param2 - 28) / 20.0, -0.8, 2.5))

    candidates = _generate_candidates(
        image=image,
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

    percentile = 99.88 + min(max(strictness, 0.0) * 0.04, 0.07)
    percentile = float(np.clip(percentile, 99.75, 99.95))

    for radius in radii:
        kernel = _crater_kernel(radius)
        response = cv2.filter2D(image, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT)

        local_max = response == cv2.dilate(response, np.ones((5, 5), np.float32))
        threshold = max(float(np.percentile(response, percentile)), 0.35)
        ys, xs = np.where(local_max & (response >= threshold))

        for y, x in zip(ys.tolist(), xs.tolist()):
            if _touches_border(float(x), float(y), float(radius), image.shape[1], image.shape[0], margin=4):
                continue
            candidates.append(Candidate(float(x), float(y), float(radius), float(response[y, x])))

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

    return pruned


def _refine_and_validate(
    image: np.ndarray,
    image_u8: np.ndarray,
    grad_mag: np.ndarray,
    edges: np.ndarray,
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

    # Revalida o melhor candidato encontrado.
    final_score = _score_circle(
        image=image,
        grad_mag=grad_mag,
        edges=edges,
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
    min_final_score = 2.60 + max(strictness, 0.0) * 0.35
    if score < min_final_score:
        return None

    return Detection(best.x, best.y, best.radius, score)


def _score_circle(
    image: np.ndarray,
    grad_mag: np.ndarray,
    edges: np.ndarray,
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

    yy, xx = np.ogrid[y1:y2, x1:x2]
    dist = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)

    inner = dist <= 0.45 * radius
    rim = (dist >= 0.72 * radius) & (dist <= 1.10 * radius)
    outer = (dist >= 1.18 * radius) & (dist <= 1.60 * radius)

    if inner.sum() < 30 or rim.sum() < 40 or outer.sum() < 40:
        return None

    inner_mean = float(patch[inner].mean())
    rim_mean = float(patch[rim].mean())
    outer_mean = float(patch[outer].mean())

    contrast = rim_mean - inner_mean
    outer_contrast = outer_mean - inner_mean
    edge_support = float(np.count_nonzero(patch_edges[rim])) / float(rim.sum())
    rim_gradient = float(patch_grad[rim].mean())

    angle_count = 48
    angles = np.linspace(0.0, 2.0 * math.pi, angle_count, endpoint=False)
    ring_x = x + radius * np.cos(angles)
    ring_y = y + radius * np.sin(angles)

    ring_values = _bilinear_sample(image, ring_x, ring_y)
    ring_grad = _bilinear_sample(grad_mag, ring_x, ring_y)

    brightness_floor = inner_mean + 0.55 * max(contrast, 0.0)
    bright_fraction = float(np.mean(ring_values > brightness_floor))

    grad_threshold = float(np.percentile(ring_grad, 60))
    grad_fraction = float(np.mean(ring_grad >= grad_threshold))

    sectors = 12
    sector_values = ring_values.reshape(sectors, angle_count // sectors).mean(axis=1)
    sector_spread = float(np.std(sector_values))
    sector_score = 1.0 / (1.0 + 8.0 * sector_spread)

    min_contrast = 0.070 + 0.015 * max(strictness, 0.0)
    min_outer_contrast = 0.040 + 0.010 * max(strictness, 0.0)
    min_bright_fraction = 0.52 + 0.05 * max(strictness, 0.0)
    min_edge_support = 0.12 + 0.02 * max(strictness, 0.0)
    min_sector_score = 0.38

    if contrast < min_contrast:
        return None
    if outer_contrast < min_outer_contrast:
        return None
    if bright_fraction < min_bright_fraction:
        return None
    if edge_support < min_edge_support and rim_gradient < 0.11:
        return None
    if sector_score < min_sector_score:
        return None

    score = (
        contrast * 3.2
        + outer_contrast * 1.35
        + rim_gradient * 1.8
        + edge_support * 1.2
        + bright_fraction * 1.3
        + grad_fraction * 0.6
        + sector_score * 0.8
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
    Não é usado como detector principal.
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
            if dist < 0.35 * (det.radius + prev.radius) and radius_ratio < 0.60:
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
