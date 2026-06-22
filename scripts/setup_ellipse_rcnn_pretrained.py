from __future__ import annotations

"""
Prepare the visual-image CNN baseline used for the fair ACDLR comparison.

The selected baseline is wdoppenberg/ellipse-rcnn with the Hugging Face crater
weights wdoppenberg/crater-rcnn. It accepts visual camera-style lunar images
and predicts crater ellipses, which can be converted to circles for the
existing ACDLR metrics. The related wdoppenberg/crater-detection repository is
the larger lunar-navigation/TRN project; this script uses the standalone model
package and pretrained crater weights.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ELLIPSE_REPO_URL = "https://github.com/wdoppenberg/ellipse-rcnn.git"
CRATER_DETECTION_REPO_URL = "https://github.com/wdoppenberg/crater-detection"
HF_MODEL_ID = "wdoppenberg/crater-rcnn"


def main() -> None:
    parser = argparse.ArgumentParser(description="Clone Ellipse R-CNN and download crater-rcnn weights.")
    parser.add_argument("--ellipse-dir", default="external/ellipse-rcnn")
    parser.add_argument("--weights-dir", default="artifacts/ellipse_rcnn_pretrained/crater-rcnn")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    ellipse_dir = _resolve(args.ellipse_dir)
    weights_dir = _resolve(args.weights_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)

    if ellipse_dir.exists():
        repo_status = "already_exists"
    else:
        ellipse_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", ELLIPSE_REPO_URL, str(ellipse_dir)], cwd=REPO_ROOT, check=True)
        repo_status = "cloned"

    install_status = "skipped"
    if not args.skip_install:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", f"{ellipse_dir}[hf]"],
            cwd=REPO_ROOT,
            check=True,
        )
        install_status = "installed"

    download_status = "skipped"
    download_error = ""
    model_path = weights_dir / "model.safetensors"
    if not args.skip_download and not model_path.exists():
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(HF_MODEL_ID, local_dir=weights_dir)
            download_status = "downloaded" if model_path.exists() else "incomplete"
            if not model_path.exists():
                download_error = f"snapshot_download finished but {model_path.name} is missing"
        except Exception as exc:  # pragma: no cover - depends on network/DNS.
            download_status = "failed"
            download_error = str(exc)
    elif model_path.exists():
        download_status = "already_exists"

    manifest = {
        "baseline": "Ellipse R-CNN",
        "model_id": HF_MODEL_ID,
        "paper": "Ellipse R-CNN: Learning to Infer Elliptical Object from Clustering and Occlusion",
        "paper_url": "https://arxiv.org/abs/2001.11584",
        "standalone_model_repo_url": ELLIPSE_REPO_URL,
        "source_project_repo_url": CRATER_DETECTION_REPO_URL,
        "hf_model": f"https://huggingface.co/{HF_MODEL_ID}",
        "ellipse_dir": _display_path(ellipse_dir),
        "weights_dir": _display_path(weights_dir),
        "model_path": _display_path(model_path),
        "repo_status": repo_status,
        "install_status": install_status,
        "download_status": download_status,
        "download_error": download_error,
        "note": (
            "This is the selected fair visual-image CNN baseline. It predicts "
            "ellipses on visual lunar images; the benchmark converts each ellipse "
            "to a circle using radius=(a+b)/2."
        ),
    }
    (weights_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("Ellipse R-CNN setup complete")
    print(f"repository: {_display_path(ellipse_dir)} ({repo_status})")
    print(f"install: {install_status}")
    print(f"weights: {_display_path(model_path)} ({download_status})")
    if download_error:
        print(f"download_error: {download_error}")
        print("If Hugging Face Xet DNS is blocked, download model.safetensors manually from:")
        print(f"https://huggingface.co/{HF_MODEL_ID}/tree/main")
        print(f"and place it in: {_display_path(weights_dir)}")
    print(f"manifest: {_display_path(weights_dir / 'manifest.json')}")


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _display_path(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
