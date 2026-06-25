from __future__ import annotations

"""
Download and prepare the LU3M6TGT YOLO crater dataset from Kaggle.

Dataset page:
https://www.kaggle.com/datasets/riccardolagrassa/lu3m6tgt

The app and benchmark scripts expect:
data/LU3M6TGT_yolo_format/
  train/images/
  train/labels/
  valid/images/
  valid/labels/
"""

import argparse
import shutil
import zipfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
KAGGLE_DATASET = "riccardolagrassa/lu3m6tgt"
DEFAULT_TARGET = REPO_ROOT / "data" / "LU3M6TGT_yolo_format"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download LU3M6TGT from Kaggle and place it in data/.")
    parser.add_argument("--target", default=str(DEFAULT_TARGET), help="Final dataset directory.")
    parser.add_argument("--force", action="store_true", help="Overwrite the target directory if it exists.")
    args = parser.parse_args()

    target = _resolve(args.target)
    if target.exists() and _is_expected_yolo_root(target) and not args.force:
        _write_data_yaml(target)
        _print_summary(target, "Dataset already present")
        return
    if target.exists() and not args.force:
        raise SystemExit(
            f"Target already exists but is not a complete YOLO dataset: {target}\n"
            "Use --force to overwrite it, or move it manually."
        )

    try:
        import kagglehub
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: kagglehub\n"
            "Install it with: python -m pip install kagglehub\n"
            "or run: python -m pip install -r requirements.txt"
        ) from exc

    print(f"Downloading Kaggle dataset: {KAGGLE_DATASET}")
    try:
        downloaded_path = Path(kagglehub.dataset_download(KAGGLE_DATASET)).resolve()
    except Exception as exc:  # pragma: no cover - depends on network/auth.
        raise SystemExit(
            "Could not download the Kaggle dataset automatically.\n"
            "Dataset page: https://www.kaggle.com/datasets/riccardolagrassa/lu3m6tgt\n"
            "Check your internet connection and Kaggle credentials, or download the zip "
            "manually and extract it to data/LU3M6TGT_yolo_format."
        ) from exc
    print(f"Kaggle cache path: {downloaded_path}")

    dataset_root = _find_yolo_root(downloaded_path)
    if dataset_root is None:
        dataset_root = _extract_and_find(downloaded_path)
    if dataset_root is None:
        raise SystemExit(
            "Could not find YOLO folders in the downloaded dataset.\n"
            "Expected train/images, train/labels, valid/images and valid/labels."
        )

    if target.exists():
        _safe_rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    if dataset_root.resolve() == target.resolve():
        status = "Dataset already in target path"
    else:
        shutil.copytree(dataset_root, target)
        status = f"Copied dataset from {dataset_root}"

    _normalise_valid_split(target)
    _write_data_yaml(target)
    _print_summary(target, status)


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _is_expected_yolo_root(path: Path) -> bool:
    return all(
        (path / rel).exists()
        for rel in (
            "train/images",
            "train/labels",
            "valid/images",
            "valid/labels",
        )
    )


def _is_yolo_root_with_valid_or_val(path: Path) -> bool:
    has_train = (path / "train/images").exists() and (path / "train/labels").exists()
    has_valid = (path / "valid/images").exists() and (path / "valid/labels").exists()
    has_val = (path / "val/images").exists() and (path / "val/labels").exists()
    return has_train and (has_valid or has_val)


def _find_yolo_root(base: Path) -> Path | None:
    if _is_yolo_root_with_valid_or_val(base):
        return base
    for path in base.rglob("*"):
        if path.is_dir() and _is_yolo_root_with_valid_or_val(path):
            return path
    return None


def _extract_and_find(downloaded_path: Path) -> Path | None:
    zip_files = list(downloaded_path.rglob("*.zip"))
    if not zip_files:
        return None

    extract_base = REPO_ROOT / "artifacts" / "dataset_downloads" / "lu3m6tgt_extracted"
    extract_base.mkdir(parents=True, exist_ok=True)

    for zip_path in zip_files:
        out_dir = extract_base / zip_path.stem
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(out_dir)

    return _find_yolo_root(extract_base)


def _normalise_valid_split(target: Path) -> None:
    valid_dir = target / "valid"
    val_dir = target / "val"
    if valid_dir.exists() or not val_dir.exists():
        return
    shutil.copytree(val_dir, valid_dir)


def _write_data_yaml(target: Path) -> None:
    data_yaml = target / "data.yaml"
    data_yaml.write_text(
        "path: .\n"
        "train: train/images\n"
        "val: valid/images\n"
        "nc: 1\n"
        "names:\n"
        "  - crater\n",
        encoding="utf-8",
    )


def _safe_rmtree(path: Path) -> None:
    resolved = path.resolve()
    repo = REPO_ROOT.resolve()
    if not _is_within(resolved, repo):
        raise SystemExit(f"Refusing to delete path outside repository: {resolved}")
    shutil.rmtree(resolved)


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.iterdir() if item.is_file())


def _print_summary(target: Path, status: str) -> None:
    print(status)
    print(f"Target: {target}")
    print(f"Train images: {_count_files(target / 'train/images')}")
    print(f"Train labels: {_count_files(target / 'train/labels')}")
    print(f"Valid images: {_count_files(target / 'valid/images')}")
    print(f"Valid labels: {_count_files(target / 'valid/labels')}")
    print("Ready for:")
    print("  streamlit run app.py")
    print("  python scripts/benchmark_yolo_dataset.py --split valid --max-images 25")


if __name__ == "__main__":
    main()
