"""
app.py
------
ACDLR — Automated Crater Detection and Landing Risk
Streamlit front-end.

Modes
-----
1. Dataset padrão (LROC NAC ROI_TORICELILOA tiles salvos no repositório)
2. Upload de imagem do usuário

Pipeline
--------
image
  → tiling (split into N×N tiles with overlap)
  → preprocessing (greyscale → CLAHE → blur)
  → detection (multi-scale matched filter + validation + local Hough refinement)
  → merge + deduplicate (global coordinate space)
  → measurement (physical sizes from scale factor)
  → risk scoring (per-region visual risk, 0–100)
  → visualisation (craters + grid + best zone)
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from core import tiling, preprocessing, detection, measurement, risk, visualization, evaluation


# ============================================================
# Constants
# ============================================================

DEFAULT_SCALE_MPX = 5.00
STUDY_TARGET_SIZE_PX = 416
SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
DATASET_DIR_CANDIDATES = [
    Path("data/LU3M6TGT_yolo_format/valid/images"),
    Path("data/lroc_nac_roi_toriceliloa_tiles"),
    Path("data/dataset_tiles"),
    Path("dataset_tiles"),
]
STUDY_DATASET_DIR_CANDIDATES = [
    Path("data/potholes"),
    Path("data/pothole"),
    Path("data/buracos_rua"),
    Path("data/buracos"),
    Path("data/road_potholes"),
    Path("data/study_datasets"),
    Path("study_datasets"),
]
STUDY_DATASET_IDEAS = [
    {
        "Dataset": "Buracos em rua / potholes",
        "Why": "Tem bordas circulares/irregulares, sombra interna e textura parecida com depressao.",
        "Expected": "Bom para estudar falsos positivos, sombras e bordas quebradas.",
    },
    {
        "Dataset": "Crateras terrestres ou impactos em solo",
        "Why": "Mantem a geometria circular do problema lunar, mas muda textura e iluminacao.",
        "Expected": "Provavelmente o melhor teste fora da Lua.",
    },
    {
        "Dataset": "Microscopia de celulas ou bolhas",
        "Why": "Objetos aproximadamente circulares com bordas claras.",
        "Expected": "Bom para estudar limite de escala e excesso de deteccoes.",
    },
    {
        "Dataset": "Copas de arvores em imagem aerea",
        "Why": "Regioes circulares repetidas em cenas naturais.",
        "Expected": "Pode funcionar em copas isoladas, mas textura interna pode confundir.",
    },
    {
        "Dataset": "Pivos centrais / circulos agricolas em satelite",
        "Why": "Formas circulares grandes e bem definidas.",
        "Expected": "Bom para testar raios maiores e deduplicacao.",
    },
    {
        "Dataset": "Manchas solares ou estruturas planetarias circulares",
        "Why": "Tambem dependem de contraste e borda em imagens astronomicas.",
        "Expected": "Interessante quando ha bordas/contraste fortes.",
    },
]
STUDY_DETECTION_PRESETS = {
    "Crateras / depressões": {
        "min_radius": 4,
        "max_radius": 70,
        "canny": 45,
        "strictness": 16,
        "clahe": 2.5,
        "blur": 5,
    },
    "Buracos em rua": {
        "min_radius": 10,
        "max_radius": 95,
        "canny": 40,
        "strictness": 28,
        "clahe": 2.8,
        "blur": 5,
    },
    "Objetos circulares preenchidos": {
        "min_radius": 10,
        "max_radius": 55,
        "canny": 35,
        "strictness": 34,
        "clahe": 2.0,
        "blur": 5,
    },
    "Exploratório sensível": {
        "min_radius": 4,
        "max_radius": 80,
        "canny": 30,
        "strictness": 10,
        "clahe": 3.2,
        "blur": 3,
    },
}
ELLIPSE_MODEL_DIR = Path("artifacts/ellipse_rcnn_pretrained/crater-rcnn")
ELLIPSE_MODEL_FILE = ELLIPSE_MODEL_DIR / "model.safetensors"
ELLIPSE_SCORE_THRESHOLD = 0.60
ELLIPSE_MAX_DET = 150
ELLIPSE_DEVICE = "cpu"
AI_REFERENCE_METRICS = [
    {
        "label": "ACDLR F1",
        "value": "0.564",
        "caption": "local YOLO valid smoke",
    },
    {
        "label": "Ellipse R-CNN F1",
        "value": "pending",
        "caption": "run fair visual benchmark",
    },
    {
        "label": "AI baseline",
        "value": "Ellipse R-CNN",
        "caption": "visual pretrained crater model",
    },
    {
        "label": "ACDLR AI use",
        "value": "0",
        "caption": "classical image processing only",
    },
]
AI_COMPARISON_ROWS = [
    {
        "Criterion": "Input data",
        "ACDLR": "Lunar surface images or local tiles",
        "Ellipse R-CNN": "Same visual tiles with YOLO labels",
    },
    {
        "Criterion": "Core method",
        "ACDLR": "Classical CV: CLAHE, matched filters, edges and geometric validation",
        "Ellipse R-CNN": "Faster/Ellipse R-CNN detector predicting crater ellipses",
    },
    {
        "Criterion": "Training",
        "ACDLR": "No training; parameters are explicit and inspectable",
        "Ellipse R-CNN": "Pretrained crater weights from Hugging Face",
    },
    {
        "Criterion": "Crater extraction",
        "ACDLR": "Direct circle candidates validated by visual/geometric criteria",
        "Ellipse R-CNN": "Predicted ellipses are converted to circles",
    },
    {
        "Criterion": "Metrics",
        "ACDLR": "Precision, recall, F1, center error and radius error on annotated tiles",
        "Ellipse R-CNN": "The same metrics on the same local annotations",
    },
    {
        "Criterion": "Project role",
        "ACDLR": "Explainable academic demo for landing-risk visualization",
        "Ellipse R-CNN": "Neural competitor for fair same-dataset comparison",
    },
]
AI_PIPELINE_ROWS = [
    {
        "Stage": "1. Data source",
        "ACDLR": "LROC visual tile or uploaded lunar image",
        "Ellipse R-CNN": "YOLO image split from LU3M6TGT_yolo_format",
    },
    {
        "Stage": "2. Representation",
        "ACDLR": "Enhanced grayscale image with edges and circular signatures",
        "Ellipse R-CNN": "Grayscale visual tile",
    },
    {
        "Stage": "3. Detection",
        "ACDLR": "Matched filter proposes circles; validators reject weak candidates",
        "Ellipse R-CNN": "CNN predicts crater ellipses",
    },
    {
        "Stage": "4. Extraction",
        "ACDLR": "Circle center and radius are produced directly",
        "Ellipse R-CNN": "Ellipse center and semi-axes are converted to circle",
    },
    {
        "Stage": "5. Decision layer",
        "ACDLR": "Risk grid and landing point are computed from detected craters",
        "Ellipse R-CNN": "Only used for benchmark comparison",
    },
]
AI_METRIC_ALIGNMENT_ROWS = [
    {
        "Shared metric": "Recall",
        "ACDLR benchmark field": "recall",
        "Ellipse benchmark field": "recall",
        "Meaning": "Annotated craters recovered by each detector",
    },
    {
        "Shared metric": "Precision",
        "ACDLR benchmark field": "precision",
        "Ellipse benchmark field": "precision",
        "Meaning": "Detected craters that match annotations",
    },
    {
        "Shared metric": "F1",
        "ACDLR benchmark field": "f1",
        "Ellipse benchmark field": "f1",
        "Meaning": "Single score balancing precision and recall",
    },
    {
        "Shared metric": "Center error ratio",
        "ACDLR benchmark field": "mean_center_error_ratio",
        "Ellipse benchmark field": "mean_center_error_ratio",
        "Meaning": "Center error divided by annotated radius",
    },
    {
        "Shared metric": "Radius error ratio",
        "ACDLR benchmark field": "mean_radius_error_ratio",
        "Ellipse benchmark field": "mean_radius_error_ratio",
        "Meaning": "Radius error divided by annotated radius",
    },
]
VALIDITY_LIMITATION_ROWS = [
    {
        "Area": "Ellipse R-CNN domain",
        "Limitation": "The pretrained model was trained on synthetic/mission-like visual crater imagery, not this exact dataset.",
        "Impact": "It is a fair visual CNN baseline, but still has domain shift.",
        "Mitigation": "Report it as pretrained zero-shot/frozen inference on the local dataset.",
    },
    {
        "Area": "Benchmark",
        "Limitation": "Local manual annotations are still required.",
        "Impact": "Current comparison is methodological until annotations exist.",
        "Mitigation": "Annotate tiles and report precision, recall, F1 and errors.",
    },
    {
        "Area": "Detection",
        "Limitation": "Shadows, illumination direction and degraded rims can mimic craters.",
        "Impact": "False positives and false negatives remain possible.",
        "Mitigation": "Tune parameters on annotated validation tiles.",
    },
    {
        "Area": "Scale",
        "Limitation": "Physical measurements depend on the metres-per-pixel setting.",
        "Impact": "Risk components can shift if the scale is wrong.",
        "Mitigation": "Use the correct image scale before comparing tiles.",
    },
    {
        "Area": "Risk score",
        "Limitation": "The landing-risk score is didactic, not a flight-safety metric.",
        "Impact": "It cannot certify real landing safety.",
        "Mitigation": "Present it as visual decision support only.",
    },
    {
        "Area": "Validation",
        "Limitation": "Tuning and evaluation on the same small set can overfit.",
        "Impact": "Reported results may look better than real generalization.",
        "Mitigation": "Separate annotated tiles into tuning and test groups.",
    },
    {
        "Area": "Terrain coverage",
        "Limitation": "Rocks, true slope, regolith properties and operational lighting are not modelled.",
        "Impact": "A suggested region can ignore risks outside the crater detector.",
        "Mitigation": "Declare the scope as visual support, not geological or flight certification.",
    },
]
CLASSICAL_EVOLUTION_STEPS = [
    "Use the YOLO annotated dataset as the primary benchmark.",
    "Report precision, recall, F1, center error and radius error before tuning.",
    "Tune the classical detector with benchmark evidence instead of visual guesswork.",
    "Score landing risk with physical components that remain comparable across tiles.",
    "Present ACDLR and Ellipse R-CNN side by side in the interface.",
    "Document limitations and validity threats explicitly.",
    "Keep ACDLR classical; use Ellipse R-CNN only as external AI competitor.",
]


# ============================================================
# Page config
# ============================================================

st.set_page_config(
    page_title="ACDLR — Crater Detection & Landing Risk",
    page_icon="🌕",
    layout="wide",
)


# ============================================================
# Cached helpers
# ============================================================

@st.cache_data(show_spinner=False)
def decode_image_bytes(file_bytes: bytes) -> np.ndarray | None:
    """Decode image bytes into a BGR NumPy array."""
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


@st.cache_data(show_spinner=False)
def load_local_image(image_path: str) -> np.ndarray | None:
    """Load an image from disk into a BGR NumPy array."""
    path = Path(image_path)
    if not path.exists():
        return None
    return decode_image_bytes(path.read_bytes())


@st.cache_data(show_spinner=False, ttl=5)
def discover_dataset_images() -> tuple[str | None, list[str]]:
    """Discover local dataset tiles in the expected repository folders."""
    for directory in DATASET_DIR_CANDIDATES:
        if directory.exists():
            files = sorted(
                str(path)
                for path in directory.rglob("*")
                if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTS
            )
            if files:
                return str(directory), files
    return None, []


@st.cache_data(show_spinner=False, ttl=5)
def discover_images_in_directory(directory_text: str) -> list[str]:
    """Discover image files in a user-selected study directory."""
    directory = Path(directory_text).expanduser()
    if not directory.exists() or not directory.is_dir():
        return []
    return sorted(
        str(path)
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTS
    )


@st.cache_data(show_spinner=False, ttl=5)
def discover_data_study_datasets() -> list[dict[str, int | str]]:
    """Discover datasets placed under data/ for the cross-domain study tab."""
    data_dir = Path("data")
    if not data_dir.exists() or not data_dir.is_dir():
        return []

    datasets: list[dict[str, int | str]] = []
    for directory in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        image_count = sum(
            1
            for path in directory.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTS
        )
        if image_count > 0:
            datasets.append(
                {
                    "name": directory.name,
                    "path": str(directory),
                    "images": image_count,
                }
            )
    return datasets


@st.cache_resource(show_spinner=False)
def load_ellipse_detector(model_dir: str):
    """Load the pretrained Ellipse R-CNN crater model."""
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    model_path = Path(model_dir)
    if (model_path / "config.json").exists() and (model_path / "model.safetensors").exists():
        from ellipse_rcnn import EllipseRCNN
        from safetensors.torch import load_file

        config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
        config["weights"] = None
        model = EllipseRCNN(**config)
        state_dict = load_file(str(model_path / "model.safetensors"))
        model.load_state_dict(state_dict)
    else:
        from ellipse_rcnn.hf import EllipseRCNN

        model = EllipseRCNN.from_pretrained(model_dir, weights=None)
    model.eval()
    model.to(ELLIPSE_DEVICE)
    return model


def predict_ellipse_circles(image_bgr: np.ndarray) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    """Run Ellipse R-CNN and convert predicted ellipses to crater circles."""
    if not ELLIPSE_MODEL_FILE.exists():
        return (
            None,
            None,
            f"Peso Ellipse R-CNN nao encontrado: `{ELLIPSE_MODEL_FILE}`",
        )

    try:
        import torch
        from PIL import Image
        from torchvision.transforms.functional import to_tensor

        model = load_ellipse_detector(str(ELLIPSE_MODEL_DIR.resolve()))
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        pil_image = Image.fromarray(gray)
        tensor = to_tensor(pil_image).to(ELLIPSE_DEVICE)
        with torch.no_grad():
            pred = model([tensor])[0]
    except Exception as exc:
        return None, None, f"Nao foi possivel executar Ellipse R-CNN: {exc}"

    scores = pred["scores"].detach().cpu()
    ellipses = pred["ellipse_params"].detach().cpu()
    keep = scores >= ELLIPSE_SCORE_THRESHOLD
    scores = scores[keep]
    ellipses = ellipses[keep]
    if scores.numel() > 0:
        order = torch.argsort(scores, descending=True)[:ELLIPSE_MAX_DET]
        ellipses = ellipses[order]

    if ellipses.numel() == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 5), dtype=np.float32), None

    arr = ellipses.numpy().astype(np.float32)
    radius = (arr[:, 0] + arr[:, 1]) / 2.0
    circles = np.column_stack([arr[:, 2], arr[:, 3], radius]).astype(np.float32)
    return circles, arr, None


# ============================================================
# Sidebar — parameters
# ============================================================

with st.sidebar:
    st.title("🌕 ACDLR")
    st.caption("Automated Crater Detection and Landing Risk")
    st.divider()

    st.subheader("Risk grid")
    grid_rows = st.slider("Grid rows", min_value=2, max_value=5, value=3)
    grid_cols = st.slider("Grid columns", min_value=2, max_value=5, value=3)

    st.divider()
    st.subheader("Processing tiles")
    tile_size = st.slider(
        "Tile size (px)",
        min_value=512,
        max_value=2048,
        value=1024,
        step=128,
    )
    overlap = st.slider(
        "Tile overlap (px)",
        min_value=32,
        max_value=256,
        value=96,
        step=16,
        help="Overlap between processing tiles",
    )

    st.divider()
    st.subheader("Pre-processing")
    clahe_clip = st.slider(
        "CLAHE clip limit",
        1.0,
        5.0,
        2.5,
        0.1,
        help="Higher → stronger local contrast enhancement",
    )
    blur_ksize = st.slider("Blur kernel size (odd)", 3, 15, 5, 2)

    st.divider()
    st.subheader("Crater Detection")
    min_radius = st.slider("Min radius (px)", 4, 50, 4)
    max_radius = st.slider("Max radius (px)", 20, 200, 70)
    param1 = st.slider(
        "Canny threshold",
        20,
        150,
        45,
        help="Canny edge upper threshold — higher = fewer edges",
    )
    param2 = st.slider(
        "Detector strictness",
        10,
        80,
        16,
        help="Higher = detector mais seletivo, com menos falsos positivos",
    )

    st.divider()
    st.subheader("Scale")
    scale_mpx = st.number_input(
        "Metres per pixel",
        min_value=0.1,
        max_value=100.0,
        value=DEFAULT_SCALE_MPX,
        step=0.1,
        help="Adjust according to the selected dataset scale.",
    )
    st.caption(
        "_O novo dataset YOLO local usa imagens 416 x 416 com anotações em labels/. "
        "A escala em metros é usada apenas no fluxo lunar/risco, não na aba de estudo._"
    )


# ============================================================
# Shared UI helpers
# ============================================================

def show_image_header(image_bgr: np.ndarray, scale_m_per_px: float) -> None:
    h, w = image_bgr.shape[:2]
    st.success(
        f"Image loaded — {w} × {h} px  "
        f"({w * scale_m_per_px / 1000:.2f} × {h * scale_m_per_px / 1000:.2f} km at {scale_m_per_px:.2f} m/px)"
    )


def show_pixel_image_header(image_bgr: np.ndarray, label: str = "Image") -> None:
    """Show image dimensions without inventing physical scale for study datasets."""
    h, w = image_bgr.shape[:2]
    st.success(f"{label} loaded - {w} x {h} px")


def normalize_study_image(
    image_bgr: np.ndarray,
    target_size: int,
    mode: str,
) -> tuple[np.ndarray, str]:
    """Prepare non-lunar study images for the existing ACDLR detector."""
    target_size = max(64, int(target_size))
    h, w = image_bgr.shape[:2]

    if mode == "Manter original":
        return image_bgr.copy(), f"mantida em {w} x {h}px"

    if mode == "Recortar centro + redimensionar":
        side = min(h, w)
        y1 = max(0, (h - side) // 2)
        x1 = max(0, (w - side) // 2)
        cropped = image_bgr[y1 : y1 + side, x1 : x1 + side]
        resized = cv2.resize(
            cropped,
            (target_size, target_size),
            interpolation=_resize_interpolation(side, target_size),
        )
        return resized, f"crop central {side} x {side}px -> {target_size} x {target_size}px"

    if mode == "Adaptar com bordas":
        scale = min(target_size / max(w, 1), target_size / max(h, 1))
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(
            image_bgr,
            (new_w, new_h),
            interpolation=_resize_interpolation(max(h, w), target_size),
        )
        canvas = np.zeros((target_size, target_size, 3), dtype=image_bgr.dtype)
        y1 = (target_size - new_h) // 2
        x1 = (target_size - new_w) // 2
        canvas[y1 : y1 + new_h, x1 : x1 + new_w] = resized
        return canvas, f"imagem inteira preservada {w} x {h}px -> {new_w} x {new_h}px com bordas"

    resized = cv2.resize(
        image_bgr,
        (target_size, target_size),
        interpolation=_resize_interpolation(max(h, w), target_size),
    )
    return resized, f"redimensionada diretamente {w} x {h}px -> {target_size} x {target_size}px"


def _resize_interpolation(source_size: int, target_size: int) -> int:
    return cv2.INTER_AREA if source_size > target_size else cv2.INTER_CUBIC


def draw_detection_overlay(
    image_bgr: np.ndarray,
    circles: np.ndarray,
    title: str,
    color: tuple[int, int, int],
) -> np.ndarray:
    """Draw detection circles with a header band outside the image area."""
    vis = image_bgr.copy()
    rows = np.asarray(circles, dtype=float) if circles.size else np.empty((0, 3), dtype=float)
    for x, y, radius in rows[:, :3]:
        cv2.circle(vis, (int(round(x)), int(round(y))), int(round(radius)), color, 1)
        cv2.circle(vis, (int(round(x)), int(round(y))), 2, color, -1)

    header_h = 54
    header = np.zeros((header_h, vis.shape[1], 3), dtype=np.uint8)
    _put_overlay_text(header, title, (8, 22), scale=0.58, bold=True)
    _put_overlay_text(header, f"{len(rows)} detections", (8, 43), scale=0.46)
    return np.vstack([header, vis])


def draw_ellipse_overlay(
    image_bgr: np.ndarray,
    ellipses: np.ndarray,
    title: str,
    color: tuple[int, int, int],
) -> np.ndarray:
    """Draw Ellipse R-CNN detections with a header band outside the image area."""
    vis = image_bgr.copy()
    rows = np.asarray(ellipses, dtype=float) if ellipses.size else np.empty((0, 5), dtype=float)
    for a, b, cx, cy, theta in rows[:, :5]:
        center = (int(round(cx)), int(round(cy)))
        axes = (max(1, int(round(a))), max(1, int(round(b))))
        cv2.ellipse(vis, center, axes, float(np.degrees(theta)), 0, 360, color, 1)
        cv2.circle(vis, center, 2, color, -1)

    header_h = 54
    header = np.zeros((header_h, vis.shape[1], 3), dtype=np.uint8)
    _put_overlay_text(header, title, (8, 22), scale=0.56, bold=True)
    _put_overlay_text(header, f"{len(rows)} ellipses", (8, 43), scale=0.46)
    return np.vstack([header, vis])


def show_fit_image(
    image: np.ndarray,
    caption: str,
    *,
    max_width: int = 720,
    clamp: bool = False,
) -> None:
    """Show images without enlarging small tiles until they become hard to inspect."""
    width = min(max_width, int(image.shape[1]))
    st.image(image, caption=caption, width=width, clamp=clamp, output_format="PNG")


def render_selected_image_ellipse_comparison(
    image_bgr: np.ndarray,
    acdlr_circles: np.ndarray,
    image_path: str | None,
) -> None:
    st.divider()
    st.subheader("Comparacao final: ACDLR x Ellipse R-CNN")

    with st.spinner("Executando Ellipse R-CNN na mesma imagem..."):
        ellipse_circles, ellipses, error = predict_ellipse_circles(image_bgr)

    if error is not None:
        st.warning(error)
        st.caption("Baixe o peso pre-treinado e rode novamente para ver a comparacao lado a lado.")
        st.code(
            "python scripts/setup_ellipse_rcnn_pretrained.py\n"
            "# se o download falhar, coloque model.safetensors em:\n"
            "artifacts/ellipse_rcnn_pretrained/crater-rcnn/model.safetensors",
            language="bash",
        )
        return

    if ellipse_circles is None:
        ellipse_circles = np.empty((0, 3), dtype=np.float32)
    if ellipses is None:
        ellipses = np.empty((0, 5), dtype=np.float32)

    acdlr_overlay = cv2.cvtColor(
        draw_detection_overlay(image_bgr, acdlr_circles, "ACDLR", (80, 255, 100)),
        cv2.COLOR_BGR2RGB,
    )
    ellipse_overlay = cv2.cvtColor(
        draw_ellipse_overlay(image_bgr, ellipses, "Ellipse R-CNN", (70, 70, 255)),
        cv2.COLOR_BGR2RGB,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        show_fit_image(acdlr_overlay, "ACDLR - processamento classico", max_width=520)
    with col_b:
        show_fit_image(ellipse_overlay, "Ellipse R-CNN - CNN visual pre-treinada", max_width=520)

    label_path = find_yolo_label_for_image(image_path)
    if label_path is None:
        st.caption("Sem label YOLO encontrado para esta imagem; exibindo apenas comparacao visual.")
        return

    truth = load_yolo_ground_truth(label_path, image_bgr.shape)
    acdlr_eval = evaluation.evaluate_circles(
        acdlr_circles,
        truth,
        center_tolerance_ratio=1.34,
        radius_tolerance_ratio=1.0,
    )
    ellipse_eval = evaluation.evaluate_circles(
        ellipse_circles,
        truth,
        center_tolerance_ratio=1.34,
        radius_tolerance_ratio=1.0,
    )

    st.markdown("**Metricas nesta imagem selecionada**")
    st.dataframe(
        [
            metric_row("ACDLR", acdlr_eval),
            metric_row("Ellipse R-CNN", ellipse_eval),
        ],
        width="stretch",
        hide_index=True,
    )



def find_yolo_label_for_image(image_path: str | None) -> Path | None:
    if not image_path:
        return None
    path = Path(image_path)
    if path.parent.name != "images":
        return None
    label_path = path.parent.parent / "labels" / f"{path.stem}.txt"
    return label_path if label_path.exists() else None


def load_yolo_ground_truth(
    label_path: Path,
    image_shape: tuple[int, int] | tuple[int, int, int],
) -> list[evaluation.GroundTruthCrater]:
    h, w = image_shape[:2]
    craters: list[evaluation.GroundTruthCrater] = []
    with label_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, cx, cy, bw, bh = parts[:5]
            radius_px = ((float(bw) * w) + (float(bh) * h)) / 4.0
            if radius_px <= 0:
                continue
            craters.append(
                evaluation.GroundTruthCrater(
                    cx=float(cx) * w,
                    cy=float(cy) * h,
                    radius_px=radius_px,
                )
            )
    return craters


def metric_row(method: str, result: evaluation.EvaluationResult) -> dict[str, int | str]:
    return {
        "Método": method,
        "Detecções": result.detections,
        "GT": result.ground_truth,
        "TP": result.true_positive,
        "FP": result.false_positive,
        "FN": result.false_negative,
        "Precision": f"{result.precision:.3f}",
        "Recall": f"{result.recall:.3f}",
        "F1": f"{result.f1:.3f}",
    }


def _put_overlay_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float = 0.5,
    color: tuple[int, int, int] = (255, 255, 255),
    bold: bool = False,
) -> None:
    x, y = origin
    thickness = 2 if bold else 1
    cv2.putText(image, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 1)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def render_dataset_gallery(dataset_files: list[str], selected_path: str) -> None:
    st.subheader("Dataset padrão local")
    st.caption(
        "Tiles locais do dataset padrão do projeto. "
        "Selecione um tile para análise ou navegue pela galeria abaixo."
    )

    preview = load_local_image(selected_path)
    if preview is not None:
        show_fit_image(
            cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
            caption=f"Preview — {Path(selected_path).name}",
            max_width=520,
        )

    with st.expander("Ver galeria do dataset", expanded=True):
        cols = st.columns(3)
        for idx, image_path in enumerate(dataset_files[:12]):
            img = load_local_image(image_path)
            if img is None:
                continue
            with cols[idx % 3]:
                show_fit_image(
                    cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                    caption=Path(image_path).name,
                    max_width=260,
                )
        if len(dataset_files) > 12:
            st.caption(f"Mostrando 12 de {len(dataset_files)} tiles disponíveis.")


def render_method_comparison() -> None:
    st.subheader("ACDLR x Ellipse R-CNN")
    st.markdown(
        "A comparação neural principal agora usa **Ellipse R-CNN** com o modelo "
        "`wdoppenberg/crater-rcnn`. Ele foi escolhido porque roda em imagens visuais "
        "lunares, prediz elipses de crateras e pode ser avaliado no mesmo dataset YOLO "
        "do ACDLR. O ACDLR continua sendo processamento de imagem clássico, sem treino "
        "neural e sem IA no método principal."
    )

    metric_cards = _latest_comparison_metrics() or AI_REFERENCE_METRICS
    metric_cols = st.columns(len(metric_cards))
    for col, metric in zip(metric_cols, metric_cards):
        with col:
            st.metric(metric["label"], metric["value"])
            st.caption(metric["caption"])

    comparison_visual = Path("artifacts/acdlr_vs_ellipse_rcnn/visual_comparison.png")
    comparison_report = Path("artifacts/acdlr_vs_ellipse_rcnn/comparison_report.md")
    if comparison_visual.exists():
        comparison_img = cv2.imread(str(comparison_visual), cv2.IMREAD_COLOR)
        if comparison_img is not None:
            show_fit_image(
                cv2.cvtColor(comparison_img, cv2.COLOR_BGR2RGB),
                "Comparacao executada: ACDLR x Ellipse R-CNN",
                max_width=960,
            )
    if comparison_report.exists():
        st.caption(f"Relatório gerado: `{comparison_report}`")

    tab_summary, tab_pipeline, tab_metrics, tab_limits = st.tabs([
        "Resumo",
        "Pipeline",
        "Métricas",
        "Limitações",
    ])

    with tab_summary:
        st.dataframe(AI_COMPARISON_ROWS, width="stretch", hide_index=True)

    with tab_pipeline:
        st.dataframe(AI_PIPELINE_ROWS, width="stretch", hide_index=True)

    with tab_metrics:
        st.dataframe(AI_METRIC_ALIGNMENT_ROWS, width="stretch", hide_index=True)
        st.caption(
            "Os dois metodos sao avaliados no mesmo split visual YOLO, com as "
            "mesmas labels convertidas para circulos e as mesmas tolerancias."
        )

    with tab_limits:
        st.dataframe(VALIDITY_LIMITATION_ROWS, width="stretch", hide_index=True)
        st.caption(
            "Essas limitações delimitam o que o ACDLR demonstra: uma solução "
            "clássica, explicável e didática, não um sistema real de navegação."
        )

    st.markdown("**Roteiro atual da comparação**")
    for idx, step in enumerate(CLASSICAL_EVOLUTION_STEPS, start=1):
        st.markdown(f"{idx}. {step}")


def _latest_comparison_metrics() -> list[dict[str, str]] | None:
    acdlr_path = Path("artifacts/acdlr_vs_ellipse_rcnn/acdlr/acdlr_yolo_summary.json")
    cnn_path = Path("artifacts/acdlr_vs_ellipse_rcnn/ellipse_rcnn/ellipse_rcnn_yolo_summary.json")
    if not acdlr_path.exists() or not cnn_path.exists():
        return None
    try:
        acdlr = json.loads(acdlr_path.read_text(encoding="utf-8"))
        cnn = json.loads(cnn_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    images = str(acdlr.get("images_processed", "?"))
    return [
        {
            "label": "ACDLR F1",
            "value": f"{float(acdlr.get('f1', 0.0)):.3f}",
            "caption": f"{images} valid images",
        },
        {
            "label": "Ellipse R-CNN F1",
            "value": f"{float(cnn.get('f1', 0.0)):.3f}",
            "caption": "same visual labels",
        },
        {
            "label": "ACDLR precision",
            "value": f"{float(acdlr.get('precision', 0.0)):.3f}",
            "caption": "same labels",
        },
        {
            "label": "Ellipse R-CNN precision",
            "value": f"{float(cnn.get('precision', 0.0)):.3f}",
            "caption": "same labels",
        },
    ]


def render_results(
    image_bgr: np.ndarray,
    prep_full,
    circles: np.ndarray,
    craters,
    stats: dict,
    score_matrix: np.ndarray,
    stats_grid,
    best_r: int,
    best_c: int,
    grid_rows: int,
    grid_cols: int,
    landing_point,
    image_path: str | None = None,
) -> None:
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_craters = cv2.cvtColor(
        visualization.draw_craters(image_bgr, craters),
        cv2.COLOR_BGR2RGB,
    )
    img_grid = cv2.cvtColor(
        visualization.draw_risk_grid(image_bgr, score_matrix, stats_grid, grid_rows, grid_cols),
        cv2.COLOR_BGR2RGB,
    )
    img_final = cv2.cvtColor(
        visualization.draw_final(
            image_bgr,
            craters,
            score_matrix,
            stats_grid,
            grid_rows,
            grid_cols,
            landing_point=landing_point,
        ),
        cv2.COLOR_BGR2RGB,
    )
    fig_heatmap = visualization.risk_heatmap_figure(score_matrix, stats_grid)

    st.divider()
    st.subheader("📊 Summary")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Craters detected", stats["count"])
    m2.metric("Mean diameter", f"{stats['mean_diameter_m']:.1f} m")
    m3.metric("Largest crater", f"{stats['max_diameter_m']:.1f} m")
    m4.metric("Best zone", f"Row {best_r+1} · Col {best_c+1}")
    m5.metric("Landing clearance", f"{landing_point.clearance_m:.1f} m")

    st.caption(
        f"Suggested landing point: x={landing_point.x}px, y={landing_point.y}px "
        f"· clearance ≈ {landing_point.clearance_m:.1f} m"
    )

    st.divider()
    st.subheader("🔬 Pipeline Steps")

    tabs = st.tabs([
        "① Original",
        "② Pre-processed",
        "③ Craters",
        "④ Risk Grid",
        "⑤ Final Result",
        "⑥ Heat-Map",
    ])

    with tabs[0]:
        show_fit_image(img_rgb, "Original image — no modifications", max_width=720)

    with tabs[1]:
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            show_fit_image(prep_full.gray, "Greyscale", max_width=260, clamp=True)
        with col_b:
            show_fit_image(prep_full.enhanced, "CLAHE", max_width=260, clamp=True)
        with col_c:
            show_fit_image(prep_full.sharpened, "Sharpened", max_width=260, clamp=True)

        show_fit_image(prep_full.edge_hint, "Edge hint", max_width=720, clamp=True)

    with tabs[2]:
        show_fit_image(
            img_craters,
            f"{stats['count']} craters detected (green ring = crater boundary)",
            max_width=720,
        )

    with tabs[3]:
        show_fit_image(
            img_grid,
            "Risk grid overlay — green=safe, red=dangerous",
            max_width=720,
        )

    with tabs[4]:
        show_fit_image(
            img_final,
            f"Final result — Best landing zone: Row {best_r + 1}, Col {best_c + 1}",
            max_width=720,
        )

    with tabs[5]:
        col_heat, col_table = st.columns([1, 1])
        with col_heat:
            st.pyplot(fig_heatmap, width="stretch")
        with col_table:
            st.markdown("**Region statistics**")
            st.caption(
                "Risk is scored from fixed physical components: density/km², "
                "diameter in metres and crater coverage."
            )
            rows_data = []
            for r in range(grid_rows):
                for c in range(grid_cols):
                    s = stats_grid[r][c]
                    best_flag = "★" if (r == best_r and c == best_c) else ""
                    rows_data.append(
                        {
                            "Region": f"{best_flag} R{r + 1}·C{c + 1}",
                            "Craters": s.crater_count,
                            "Density /km²": f"{s.density_per_km2:.1f}",
                            "Mean diam m": f"{s.mean_diameter_m:.1f}",
                            "Largest diam m": f"{s.largest_diameter_m:.1f}",
                            "Coverage %": f"{s.coverage_ratio * 100:.2f}",
                            "D/M/L/C": (
                                f"{s.density_component:.0f}/"
                                f"{s.mean_size_component:.0f}/"
                                f"{s.largest_size_component:.0f}/"
                                f"{s.coverage_component:.0f}"
                            ),
                            "Risk score": f"{s.risk_score:.1f}",
                            "Label": s.risk_label,
                        }
                    )
            st.dataframe(rows_data, width="stretch", hide_index=True)


    render_selected_image_ellipse_comparison(image_bgr, circles, image_path)

    st.divider()
    st.subheader("💾 Export")

    col_dl1, col_dl2 = st.columns(2)

    def _encode_png(rgb_img: np.ndarray) -> bytes:
        bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".png", bgr)
        return buf.tobytes() if ok else b""

    with col_dl1:
        st.download_button(
            "Download — Craters annotated",
            data=_encode_png(img_craters),
            file_name="acdlr_craters.png",
            mime="image/png",
            width="stretch",
        )

    with col_dl2:
        st.download_button(
            "Download — Final risk analysis",
            data=_encode_png(img_final),
            file_name="acdlr_risk_analysis.png",
            mime="image/png",
            width="stretch",
        )


def run_analysis(image_bgr: np.ndarray, image_path: str | None = None) -> None:
    progress = st.progress(0, text="Starting pipeline…")
    t_start = time.perf_counter()

    def _step(pct: int, msg: str) -> None:
        progress.progress(pct, text=msg)

    _step(5, "Step 1/6 — Splitting image into tiles…")
    tiles = tiling.split(
        image_bgr,
        tile_size=tile_size,
        overlap=overlap,
    )

    _step(15, "Step 2/6 — Pre-processing tiles…")
    prep_results: dict[tuple[int, int], preprocessing.PreprocessResult] = {}
    for tile in tiles:
        prep_results[(tile.row, tile.col)] = preprocessing.run(
            tile.image,
            clahe_clip=clahe_clip,
            blur_ksize=blur_ksize,
        )

    prep_full = preprocessing.run(image_bgr, clahe_clip=clahe_clip, blur_ksize=blur_ksize)

    _step(35, "Step 3/6 — Detecting craters…")
    all_circles: list[np.ndarray] = []
    for tile in tiles:
        local = detection.detect(
            prep_results[(tile.row, tile.col)],
            min_radius=min_radius,
            max_radius=max_radius,
            param1=param1,
            param2=param2,
        )
        if local.size > 0:
            all_circles.append(tiling.to_global(local, tile))

    if all_circles:
        merged = np.vstack(all_circles)
        circles = tiling.deduplicate(merged)
    else:
        circles = np.empty((0, 3), dtype=int)

    _step(55, "Step 4/6 — Measuring craters…")
    craters = measurement.measure(circles, scale_m_per_px=scale_mpx)
    stats = measurement.summary_stats(craters)

    _step(70, "Step 5/6 — Calculating landing risk…")
    h, w = image_bgr.shape[:2]
    score_matrix, stats_grid = risk.analyse(
        craters,
        image_shape=(h, w),
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        scale_m_per_px=scale_mpx,
    )
    best_r, best_c = risk.best_landing_cell(score_matrix)

    landing_point = risk.suggest_landing_point(
        craters,
        image_shape=(h, w),
        best_row=best_r,
        best_col=best_c,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        scale_m_per_px=scale_mpx,
    )

    elapsed = time.perf_counter() - t_start
    _step(100, f"Done — {elapsed:.1f}s")

    render_results(
        image_bgr=image_bgr,
        prep_full=prep_full,
        circles=circles,
        craters=craters,
        stats=stats,
        score_matrix=score_matrix,
        stats_grid=stats_grid,
        best_r=best_r,
        best_c=best_c,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        landing_point=landing_point,
        image_path=image_path,
    )


def run_detection_study(
    image_bgr: np.ndarray,
    source_label: str,
    *,
    study_clahe_clip: float,
    study_blur_ksize: int,
    study_min_radius: int,
    study_max_radius: int,
    study_canny_threshold: int,
    study_strictness: int,
) -> None:
    """Run the same ACDLR detection stage for qualitative cross-dataset study."""
    progress = st.progress(0, text="Starting ACDLR detection study...")
    t_start = time.perf_counter()

    def _step(pct: int, msg: str) -> None:
        progress.progress(pct, text=msg)

    _step(10, "Step 1/4 - Splitting image into tiles...")
    tiles = tiling.split(
        image_bgr,
        tile_size=tile_size,
        overlap=overlap,
    )

    _step(30, "Step 2/4 - Running the same pre-processing...")
    prep_results: dict[tuple[int, int], preprocessing.PreprocessResult] = {}
    for tile in tiles:
        prep_results[(tile.row, tile.col)] = preprocessing.run(
            tile.image,
            clahe_clip=study_clahe_clip,
            blur_ksize=study_blur_ksize,
        )
    prep_full = preprocessing.run(image_bgr, clahe_clip=study_clahe_clip, blur_ksize=study_blur_ksize)

    _step(60, "Step 3/4 - Detecting circular depression candidates...")
    all_circles: list[np.ndarray] = []
    for tile in tiles:
        local = detection.detect(
            prep_results[(tile.row, tile.col)],
            min_radius=study_min_radius,
            max_radius=study_max_radius,
            param1=study_canny_threshold,
            param2=study_strictness,
        )
        if local.size > 0:
            all_circles.append(tiling.to_global(local, tile))

    if all_circles:
        merged = np.vstack(all_circles)
        circles = tiling.deduplicate(merged)
    else:
        circles = np.empty((0, 3), dtype=int)

    elapsed = time.perf_counter() - t_start
    _step(100, f"Done - {elapsed:.1f}s")

    h, w = image_bgr.shape[:2]
    overlay = cv2.cvtColor(
        draw_detection_overlay(image_bgr, circles, "ACDLR study detections", (80, 255, 100)),
        cv2.COLOR_BGR2RGB,
    )

    st.divider()
    st.subheader("Resultado do estudo")
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Detections", int(len(circles)))
    col_b.metric("Image width", f"{w}px")
    col_c.metric("Image height", f"{h}px")
    col_d.metric("Runtime", f"{elapsed:.1f}s")
    st.caption(
        "Esta aba reaproveita o mesmo detector ACDLR. Os resultados aqui sao "
        "qualitativos e servem para estudar comportamento fora do dominio lunar."
    )
    st.caption(
        "Parametros usados: "
        f"radius={study_min_radius}-{study_max_radius}px, "
        f"canny={study_canny_threshold}, strictness={study_strictness}, "
        f"CLAHE={study_clahe_clip:.1f}, blur={study_blur_ksize}."
    )

    tabs = st.tabs(["Original", "Pre-processamento", "Deteccoes"])
    with tabs[0]:
        show_fit_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), source_label, max_width=720)

    with tabs[1]:
        col_1, col_2, col_3 = st.columns(3)
        with col_1:
            show_fit_image(prep_full.gray, "Greyscale", max_width=260, clamp=True)
        with col_2:
            show_fit_image(prep_full.enhanced, "CLAHE", max_width=260, clamp=True)
        with col_3:
            show_fit_image(prep_full.sharpened, "Sharpened", max_width=260, clamp=True)
        show_fit_image(prep_full.edge_hint, "Edge hint", max_width=720, clamp=True)

    with tabs[2]:
        show_fit_image(overlay, "ACDLR detections on study image", max_width=720)


def render_study_tab() -> None:
    st.subheader("Estudo do ACDLR em outros datasets")
    st.caption(
        "Esta area e separada do fluxo lunar principal. Ela nao altera o algoritmo: "
        "usa os mesmos parametros e a mesma etapa de deteccao do ACDLR para estudo qualitativo."
    )

    data_datasets = discover_data_study_datasets()
    dataset_paths = [str(dataset["path"]) for dataset in data_datasets]
    dataset_counts = {str(dataset["path"]): int(dataset["images"]) for dataset in data_datasets}
    preferred_names = ("holes", "pothole", "potholes", "buracos", "buracos_rua", "road_potholes")
    default_dataset_index = 0
    for idx, path_text in enumerate(dataset_paths):
        if Path(path_text).name.lower() in preferred_names:
            default_dataset_index = idx
            break

    study_mode = st.radio(
        "Entrada para estudo",
        options=["Dataset em data/", "Pasta personalizada", "Upload"],
        horizontal=True,
        index=0,
        key="study_input_mode",
    )

    study_image: np.ndarray | None = None
    study_label = "Study image"

    with st.expander("Normalizacao de entrada para estudo", expanded=True):
        st.caption(
            "Esta etapa prepara imagens comuns para o detector sem atribuir escala fisica. "
            "O algoritmo ACDLR continua igual; apenas a imagem de entrada e normalizada."
        )
        normalization_mode = st.selectbox(
            "Como adaptar a imagem",
            options=[
                "Recortar centro + redimensionar",
                "Adaptar com bordas",
                "Redimensionar direto",
                "Manter original",
            ],
            index=0,
            help=(
                "Recortar centro preserva proporcao e entrega uma imagem quadrada. "
                "Adaptar com bordas preserva a imagem inteira. Redimensionar direto pode distorcer."
            ),
        )
        target_size = st.number_input(
            "Tamanho de entrada para estudo (px)",
            min_value=128,
            max_value=2048,
            value=STUDY_TARGET_SIZE_PX,
            step=32,
            help="416 px combina com o dataset visual anotado usado no benchmark.",
        )

    with st.expander("Configuracoes de deteccao para estudo", expanded=True):
        st.caption(
            "Estas configuracoes afetam apenas esta aba. O fluxo lunar principal continua "
            "usando os controles da sidebar e o mesmo algoritmo ACDLR."
        )
        preset_name = st.selectbox(
            "Preset de estudo",
            options=list(STUDY_DETECTION_PRESETS.keys()),
            index=0,
            help=(
                "Use 'Crateras / depressoes' para manter o comportamento mais proximo "
                "do objetivo principal. Para celulas/bolhas, aumente o raio minimo e a strictness."
            ),
        )
        preset = STUDY_DETECTION_PRESETS[preset_name]
        col_a, col_b = st.columns(2)
        with col_a:
            study_min_radius = st.slider(
                "Study min radius (px)",
                4,
                100,
                int(preset["min_radius"]),
                key=f"study_min_radius_{preset_name}",
            )
            study_canny_threshold = st.slider(
                "Study Canny threshold",
                20,
                150,
                int(preset["canny"]),
                key=f"study_canny_{preset_name}",
            )
            study_clahe_clip = st.slider(
                "Study CLAHE clip",
                1.0,
                5.0,
                float(preset["clahe"]),
                0.1,
                key=f"study_clahe_{preset_name}",
            )
        with col_b:
            study_max_radius = st.slider(
                "Study max radius (px)",
                20,
                220,
                int(preset["max_radius"]),
                key=f"study_max_radius_{preset_name}",
            )
            study_strictness = st.slider(
                "Study detector strictness",
                10,
                80,
                int(preset["strictness"]),
                key=f"study_strictness_{preset_name}",
            )
            study_blur_ksize = st.slider(
                "Study blur kernel size (odd)",
                3,
                15,
                int(preset["blur"]),
                2,
                key=f"study_blur_{preset_name}",
            )
        if study_max_radius <= study_min_radius:
            st.warning("O raio maximo precisa ser maior que o raio minimo.")

    if study_mode == "Dataset em data/":
        st.caption(
            "Coloque seus datasets como subpastas de `data/`. O app procura imagens "
            "recursivamente, entao estruturas como `data/holes/train/images` tambem funcionam."
        )
        if not dataset_paths:
            st.info("Nenhum dataset com imagens foi encontrado dentro de `data/`.")
            st.code(
                "data/\n"
                "  holes/\n"
                "    imagem_001.jpg\n"
                "  potholes/\n"
                "    train/images/*.jpg",
                language="text",
            )
        else:
            selected_dataset = st.selectbox(
                "Selecione um dataset em data/",
                options=dataset_paths,
                index=default_dataset_index,
                format_func=lambda path: f"{Path(path).name} ({dataset_counts.get(path, 0)} imagens)",
                key="study_dataset_in_data",
            )
            study_files = discover_images_in_directory(selected_dataset)
            st.caption(f"Pasta selecionada: `{selected_dataset}`")
            if not study_files:
                st.info("Esse dataset foi encontrado, mas nenhuma imagem carregavel apareceu agora.")
            else:
                selected_path = st.selectbox(
                    "Selecione uma imagem para estudo",
                    options=study_files,
                    format_func=lambda path: Path(path).name,
                    key="study_selected_data_image",
                )
                study_image = load_local_image(selected_path)
                study_label = f"{Path(selected_dataset).name} / {Path(selected_path).name}"
                if study_image is not None:
                    show_pixel_image_header(study_image, "Original study image")
                    show_fit_image(
                        cv2.cvtColor(study_image, cv2.COLOR_BGR2RGB),
                        f"Original preview - {study_label}",
                        max_width=640,
                    )

    if study_mode == "Pasta personalizada":
        existing_candidate = next((path for path in STUDY_DATASET_DIR_CANDIDATES if path.exists()), None)
        default_dir = str(existing_candidate) if existing_candidate is not None else "data/holes"
        study_dir = st.text_input(
            "Pasta com imagens de estudo",
            value=default_dir,
            help="Exemplo: data/potholes, data/buracos_rua ou qualquer pasta local com imagens.",
        )
        study_files = discover_images_in_directory(study_dir)
        if not study_files:
            st.info("Nenhuma imagem encontrada nessa pasta. Ajuste o caminho ou use Upload.")
        else:
            selected_path = st.selectbox(
                "Selecione uma imagem para estudo",
                options=study_files,
                format_func=lambda path: Path(path).name,
                key="study_selected_image",
            )
            study_image = load_local_image(selected_path)
            study_label = Path(selected_path).name
            if study_image is not None:
                show_pixel_image_header(study_image, "Original study image")
                show_fit_image(
                    cv2.cvtColor(study_image, cv2.COLOR_BGR2RGB),
                    f"Original preview - {study_label}",
                    max_width=640,
                )

    if study_mode == "Upload":
        uploaded = st.file_uploader(
            "Envie uma imagem para estudo",
            type=["png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"],
            key="study_upload",
        )
        if uploaded is not None:
            study_image = decode_image_bytes(uploaded.read())
            study_label = uploaded.name
            if study_image is None:
                st.error("Could not decode the uploaded image.")
                return
            show_pixel_image_header(study_image, "Original study image")
            show_fit_image(
                cv2.cvtColor(study_image, cv2.COLOR_BGR2RGB),
                f"Original preview - {study_label}",
                max_width=640,
            )
        else:
            st.info("Envie uma imagem ou use uma pasta local para iniciar o estudo.")

    if study_image is not None:
        normalized_image, normalization_note = normalize_study_image(
            study_image,
            int(target_size),
            normalization_mode,
        )
        st.markdown("**Imagem que sera enviada ao ACDLR**")
        show_pixel_image_header(normalized_image, "Normalized study image")
        st.caption(normalization_note)
        show_fit_image(
            cv2.cvtColor(normalized_image, cv2.COLOR_BGR2RGB),
            f"Normalized preview - {study_label}",
            max_width=480,
        )
        can_run_study = study_max_radius > study_min_radius
        if st.button("Run ACDLR detection study", type="primary", width="stretch", disabled=not can_run_study):
            run_detection_study(
                normalized_image,
                f"{study_label} ({normalization_note})",
                study_clahe_clip=float(study_clahe_clip),
                study_blur_ksize=int(study_blur_ksize),
                study_min_radius=int(study_min_radius),
                study_max_radius=int(study_max_radius),
                study_canny_threshold=int(study_canny_threshold),
                study_strictness=int(study_strictness),
            )


# ============================================================
# Header
# ============================================================

st.title("🌕 ACDLR")
main_tab, study_tab = st.tabs(["Crateras lunares", "Estudo em outros datasets"])

with main_tab:
    st.markdown(
        "**Automated Crater Detection and Landing Risk** — "
        "analise tiles dos datasets locais "
        "ou envie uma nova imagem lunar para detectar crateras, avaliar o risco "
        "de pouso por região e destacar a zona mais segura."
    )
    st.divider()

    with st.expander("Comparação ACDLR x Ellipse R-CNN", expanded=True):
        render_method_comparison()

    mode = st.radio(
        "Modo de entrada",
        options=["Dataset padrão", "Enviar imagem"],
        horizontal=True,
        index=0,
        key="main_input_mode",
    )

    analysis_image: np.ndarray | None = None
    analysis_image_path: str | None = None

    if mode == "Dataset padrão":
        dataset_dir, dataset_files = discover_dataset_images()

        if not dataset_files:
            st.warning(
                "Nenhum tile local do dataset padrão foi encontrado no repositório. "
                "Adicione arquivos de imagem em `data/LU3M6TGT_yolo_format/valid/images/` "
                "ou use a aba de upload para analisar uma imagem manualmente."
            )
            st.code("data/LU3M6TGT_yolo_format/valid/images/")
        else:
            default_index = 0
            selected_path = st.selectbox(
                "Selecione um tile do dataset",
                options=dataset_files,
                index=default_index,
                format_func=lambda path: Path(path).name,
                key="main_selected_image",
            )
            st.caption(f"Pasta detectada: `{dataset_dir}` · {len(dataset_files)} tile(s) encontrados")

            selected_image = load_local_image(selected_path)
            if selected_image is None:
                st.error("Não foi possível carregar o tile selecionado.")
                st.stop()

            show_image_header(selected_image, scale_mpx)
            render_dataset_gallery(dataset_files, selected_path)

            if st.button("▶ Run Analysis on selected dataset tile", type="primary", width="stretch"):
                analysis_image = selected_image
                analysis_image_path = selected_path

    if mode == "Enviar imagem":
        st.subheader("Upload de imagem")
        uploaded = st.file_uploader(
            "Envie uma imagem lunar",
            type=["png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"],
            help="Works best with high-contrast greyscale images. Large images are tiled automatically.",
            key="main_upload",
        )

        if uploaded is None:
            st.info("Envie uma imagem para começar a análise.")
        else:
            image_bgr = decode_image_bytes(uploaded.read())
            if image_bgr is None:
                st.error("Could not decode the image. Please upload a valid image file.")
                st.stop()

            show_image_header(image_bgr, scale_mpx)
            show_fit_image(
                cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
                "Uploaded image (preview)",
                max_width=720,
            )

            if st.button("▶ Run Analysis on uploaded image", type="primary", width="stretch"):
                analysis_image = image_bgr
                analysis_image_path = None

    if analysis_image is not None:
        run_analysis(analysis_image, image_path=analysis_image_path)

with study_tab:
    render_study_tab()
