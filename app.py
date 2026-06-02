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

import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from core import tiling, preprocessing, detection, measurement, risk, visualization


# ============================================================
# Constants
# ============================================================

DEFAULT_SCALE_MPX = 5.00
SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
DATASET_DIR_CANDIDATES = [
    Path("data/lroc_nac_roi_toriceliloa_tiles"),
    Path("data/dataset_tiles"),
    Path("dataset_tiles"),
]
DEEPMOON_REFERENCE_METRICS = [
    {
        "label": "Recall",
        "value": "92%",
        "caption": "post-processed test set",
    },
    {
        "label": "Precision",
        "value": "56%",
        "caption": "post-processed test set",
    },
    {
        "label": "Median radius error",
        "value": "7%",
        "caption": "fractional error",
    },
    {
        "label": "False positives",
        "value": "11%",
        "caption": "manual inspection estimate",
    },
]
DEEPMOON_COMPARISON_ROWS = [
    {
        "Criterion": "Input data",
        "ACDLR": "Lunar surface images or local tiles",
        "DeepMoon": "Global lunar DEM crops derived from LRO/Kaguya data",
    },
    {
        "Criterion": "Core method",
        "ACDLR": "Classical CV: CLAHE, matched filters, edges and geometric validation",
        "DeepMoon": "CNN based on U-Net that predicts binary crater-rim masks",
    },
    {
        "Criterion": "Training",
        "ACDLR": "No training; parameters are explicit and inspectable",
        "DeepMoon": "Requires labelled catalogues, generated targets and GPU training",
    },
    {
        "Criterion": "Crater extraction",
        "ACDLR": "Direct circle candidates validated by visual/geometric criteria",
        "DeepMoon": "Post-CNN template matching extracts x, y and radius from predicted rims",
    },
    {
        "Criterion": "Metrics to mirror",
        "ACDLR": "Precision, recall, F1, center error and radius error on annotated tiles",
        "DeepMoon": "Precision, recall, F1 and fractional longitude/latitude/radius errors",
    },
    {
        "Criterion": "Project role",
        "ACDLR": "Explainable academic demo for landing-risk visualization",
        "DeepMoon": "Scientific crater-catalogue extraction pipeline",
    },
]
DEEPMOON_PIPELINE_ROWS = [
    {
        "Stage": "1. Data source",
        "ACDLR": "LROC visual tile or uploaded lunar image",
        "DeepMoon": "Global lunar DEM crops",
    },
    {
        "Stage": "2. Representation",
        "ACDLR": "Enhanced grayscale image with edges and circular signatures",
        "DeepMoon": "DEM input paired with binary crater-rim target mask",
    },
    {
        "Stage": "3. Detection",
        "ACDLR": "Matched filter proposes circles; validators reject weak candidates",
        "DeepMoon": "CNN predicts rim mask pixels",
    },
    {
        "Stage": "4. Extraction",
        "ACDLR": "Circle center and radius are produced directly",
        "DeepMoon": "Template matching converts mask into center and radius",
    },
    {
        "Stage": "5. Decision layer",
        "ACDLR": "Risk grid and landing point are computed from detected craters",
        "DeepMoon": "Crater catalogue is aggregated and deduplicated globally",
    },
]
DEEPMOON_METRIC_ALIGNMENT_ROWS = [
    {
        "DeepMoon metric": "Recall",
        "ACDLR benchmark field": "recall",
        "Meaning": "Annotated craters recovered by the detector",
    },
    {
        "DeepMoon metric": "Precision",
        "ACDLR benchmark field": "precision",
        "Meaning": "Detected craters that match annotations",
    },
    {
        "DeepMoon metric": "F1",
        "ACDLR benchmark field": "f1",
        "Meaning": "Single score balancing precision and recall",
    },
    {
        "DeepMoon metric": "Fractional coordinate error",
        "ACDLR benchmark field": "mean_center_error_ratio",
        "Meaning": "Center error divided by annotated radius",
    },
    {
        "DeepMoon metric": "Fractional radius error",
        "ACDLR benchmark field": "mean_radius_error_ratio",
        "Meaning": "Radius error divided by annotated radius",
    },
]
VALIDITY_LIMITATION_ROWS = [
    {
        "Area": "DeepMoon comparison",
        "Limitation": "DeepMoon uses DEM crops; ACDLR uses visual lunar images/tiles.",
        "Impact": "Scores are not directly interchangeable.",
        "Mitigation": "Compare method, metrics and protocol; avoid claiming parity.",
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
    "Build a small annotated LROC tile benchmark using DeepMoon-like metrics.",
    "Report precision, recall, F1, center error and radius error before tuning.",
    "Tune the classical detector with benchmark evidence instead of visual guesswork.",
    "Score landing risk with physical components that remain comparable across tiles.",
    "Present ACDLR and DeepMoon side by side in the interface.",
    "Document limitations and validity threats explicitly.",
    "Keep DeepMoon as the neural reference, not as a dependency of the project.",
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
    min_radius = st.slider("Min radius (px)", 5, 50, 10)
    max_radius = st.slider("Max radius (px)", 20, 200, 40)
    param1 = st.slider(
        "Canny threshold",
        20,
        150,
        60,
        help="Canny edge upper threshold — higher = fewer edges",
    )
    param2 = st.slider(
        "Accumulator threshold",
        10,
        80,
        34,
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
        help="Local validation tiles use the official LROC 5M product: 5.00 m/px",
    )
    st.caption("_Local validation dataset: LROC NAC ROI_TORICELILOA 5M (5.00 m/px)_")


# ============================================================
# Shared UI helpers
# ============================================================

def show_image_header(image_bgr: np.ndarray, scale_m_per_px: float) -> None:
    h, w = image_bgr.shape[:2]
    st.success(
        f"Image loaded — {w} × {h} px  "
        f"({w * scale_m_per_px / 1000:.2f} × {h * scale_m_per_px / 1000:.2f} km at {scale_m_per_px:.2f} m/px)"
    )


def render_dataset_gallery(dataset_files: list[str], selected_path: str) -> None:
    st.subheader("Dataset padrão — LROC NAC ROI_TORICELILOA")
    st.caption(
        "Tiles locais do dataset padrão do projeto. "
        "Selecione um tile para análise ou navegue pela galeria abaixo."
    )

    preview = load_local_image(selected_path)
    if preview is not None:
        st.image(
            cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
            caption=f"Preview — {Path(selected_path).name}",
            use_container_width=True,
        )

    with st.expander("Ver galeria do dataset", expanded=True):
        cols = st.columns(3)
        for idx, image_path in enumerate(dataset_files[:12]):
            img = load_local_image(image_path)
            if img is None:
                continue
            with cols[idx % 3]:
                st.image(
                    cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                    caption=Path(image_path).name,
                    use_container_width=True,
                )
        if len(dataset_files) > 12:
            st.caption(f"Mostrando 12 de {len(dataset_files)} tiles disponíveis.")


def render_method_comparison() -> None:
    st.subheader("ACDLR x DeepMoon")
    st.markdown(
        "DeepMoon é a referência neural usada para comparação: uma CNN baseada "
        "em U-Net para extrair crateras de DEMs lunares. O ACDLR mantém uma "
        "abordagem clássica e usa métricas semelhantes para avaliação."
    )

    metric_cols = st.columns(len(DEEPMOON_REFERENCE_METRICS))
    for col, metric in zip(metric_cols, DEEPMOON_REFERENCE_METRICS):
        with col:
            st.metric(metric["label"], metric["value"])
            st.caption(metric["caption"])

    tab_summary, tab_pipeline, tab_metrics, tab_limits = st.tabs([
        "Resumo",
        "Pipeline",
        "Métricas",
        "Limitações",
    ])

    with tab_summary:
        st.dataframe(DEEPMOON_COMPARISON_ROWS, use_container_width=True, hide_index=True)

    with tab_pipeline:
        st.dataframe(DEEPMOON_PIPELINE_ROWS, use_container_width=True, hide_index=True)

    with tab_metrics:
        st.dataframe(DEEPMOON_METRIC_ALIGNMENT_ROWS, use_container_width=True, hide_index=True)
        st.caption(
            "O ACDLR adapta os erros fracionários do DeepMoon para coordenadas "
            "em pixels: erro dividido pelo raio anotado."
        )

    with tab_limits:
        st.dataframe(VALIDITY_LIMITATION_ROWS, use_container_width=True, hide_index=True)
        st.caption(
            "Essas limitações delimitam o que o ACDLR demonstra: uma solução "
            "clássica, explicável e didática, não um sistema real de navegação."
        )

    st.markdown("**Roteiro atual inspirado no DeepMoon**")
    for idx, step in enumerate(CLASSICAL_EVOLUTION_STEPS, start=1):
        st.markdown(f"{idx}. {step}")


def render_results(
    image_bgr: np.ndarray,
    prep_full,
    craters,
    stats: dict,
    score_matrix: np.ndarray,
    stats_grid,
    best_r: int,
    best_c: int,
    grid_rows: int,
    grid_cols: int,
    landing_point,
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
        st.image(img_rgb, caption="Original image — no modifications", use_container_width=True)

    with tabs[1]:
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.image(prep_full.gray, caption="Greyscale", use_container_width=True, clamp=True)
        with col_b:
            st.image(prep_full.enhanced, caption="CLAHE", use_container_width=True, clamp=True)
        with col_c:
            st.image(prep_full.sharpened, caption="Sharpened", use_container_width=True, clamp=True)

        st.image(prep_full.edge_hint, caption="Edge hint", use_container_width=True, clamp=True)

    with tabs[2]:
        st.image(
            img_craters,
            caption=f"{stats['count']} craters detected (green ring = crater boundary)",
            use_container_width=True,
        )

    with tabs[3]:
        st.image(
            img_grid,
            caption="Risk grid overlay — green=safe, red=dangerous",
            use_container_width=True,
        )

    with tabs[4]:
        st.image(
            img_final,
            caption=f"Final result — Best landing zone: Row {best_r + 1}, Col {best_c + 1}",
            use_container_width=True,
        )

    with tabs[5]:
        col_heat, col_table = st.columns([1, 1])
        with col_heat:
            st.pyplot(fig_heatmap, use_container_width=True)
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
            st.dataframe(rows_data, use_container_width=True, hide_index=True)

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
            use_container_width=True,
        )

    with col_dl2:
        st.download_button(
            "Download — Final risk analysis",
            data=_encode_png(img_final),
            file_name="acdlr_risk_analysis.png",
            mime="image/png",
            use_container_width=True,
        )


def run_analysis(image_bgr: np.ndarray) -> None:
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
        craters=craters,
        stats=stats,
        score_matrix=score_matrix,
        stats_grid=stats_grid,
        best_r=best_r,
        best_c=best_c,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        landing_point=landing_point,
    )


# ============================================================
# Header
# ============================================================

st.title("🌕 ACDLR")
st.markdown(
    "**Automated Crater Detection and Landing Risk** — "
    "analise tiles padrão do dataset **LROC NAC ROI_TORICELILOA** "
    "ou envie uma nova imagem lunar para detectar crateras, avaliar o risco "
    "de pouso por região e destacar a zona mais segura."
)
st.divider()

with st.expander("Comparação com DeepMoon", expanded=True):
    render_method_comparison()

mode = st.radio(
    "Modo de entrada",
    options=["Dataset padrão", "Enviar imagem"],
    horizontal=True,
    index=0,
)

analysis_image: np.ndarray | None = None


# ============================================================
# Mode 1 — default local dataset
# ============================================================

if mode == "Dataset padrão":
    dataset_dir, dataset_files = discover_dataset_images()

    if not dataset_files:
        st.warning(
            "Nenhum tile local do dataset padrão foi encontrado no repositório. "
            "Adicione arquivos de imagem em `data/lroc_nac_roi_toriceliloa_tiles/` "
            "ou use a aba de upload para analisar uma imagem manualmente."
        )
        st.code("data/lroc_nac_roi_toriceliloa_tiles/")
    else:
        default_index = 0
        selected_path = st.selectbox(
            "Selecione um tile do dataset",
            options=dataset_files,
            index=default_index,
            format_func=lambda path: Path(path).name,
        )
        st.caption(f"Pasta detectada: `{dataset_dir}` · {len(dataset_files)} tile(s) encontrados")

        selected_image = load_local_image(selected_path)
        if selected_image is None:
            st.error("Não foi possível carregar o tile selecionado.")
            st.stop()

        show_image_header(selected_image, scale_mpx)
        render_dataset_gallery(dataset_files, selected_path)

        if st.button("▶ Run Analysis on selected dataset tile", type="primary", use_container_width=True):
            analysis_image = selected_image


# ============================================================
# Mode 2 — uploaded image
# ============================================================

if mode == "Enviar imagem":
    st.subheader("Upload de imagem")
    uploaded = st.file_uploader(
        "Envie uma imagem lunar",
        type=["png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"],
        help="Works best with high-contrast greyscale images. Large images are tiled automatically.",
    )

    if uploaded is None:
        st.info("Envie uma imagem para começar a análise.")
    else:
        image_bgr = decode_image_bytes(uploaded.read())
        if image_bgr is None:
            st.error("Could not decode the image. Please upload a valid image file.")
            st.stop()

        show_image_header(image_bgr, scale_mpx)
        st.image(
            cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
            caption="Uploaded image (preview)",
            use_container_width=True,
        )

        if st.button("▶ Run Analysis on uploaded image", type="primary", use_container_width=True):
            analysis_image = image_bgr


# ============================================================
# Run selected analysis
# ============================================================

if analysis_image is not None:
    run_analysis(analysis_image)
