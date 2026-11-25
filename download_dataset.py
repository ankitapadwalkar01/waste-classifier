from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

from src.config import DATASET_URL, RAW_DIR
from src.data import find_class_directory


def safe_extract(zip_path: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if destination_resolved not in target.parents and target != destination_resolved:
                raise RuntimeError(f"Unsafe archive path detected: {member.filename}")
        archive.extractall(destination)


def main():
    parser = argparse.ArgumentParser(description="Download and extract the UCI RealWaste dataset.")
    parser.add_argument("--url", default=DATASET_URL)
    parser.add_argument("--force", action="store_true", help="Download again even if an archive exists.")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = RAW_DIR / "realwaste.zip"
    extracted_dir = RAW_DIR / "extracted"
    if not archive_path.exists() or args.force:
        print("Downloading RealWaste from the UCI Machine Learning Repository (~657 MB)...")
        response = requests.get(args.url, stream=True, timeout=(30, 120))
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        temporary_path = archive_path.with_suffix(".part")
        with open(temporary_path, "wb") as handle, tqdm(
            total=total, unit="B", unit_scale=True, desc="RealWaste"
        ) as progress:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    progress.update(len(chunk))
        temporary_path.replace(archive_path)
    else:
        print(f"Using existing archive: {archive_path}")

    if extracted_dir.exists() and args.force:
        shutil.rmtree(extracted_dir)
    if not extracted_dir.exists():
        extracted_dir.mkdir(parents=True)
        print("Extracting dataset...")
        safe_extract(archive_path, extracted_dir)

    class_root = find_class_directory(extracted_dir)
    print(f"\nDataset is ready. Class directory: {class_root}")
    print("Next command: python train.py")


if __name__ == "__main__":
    main()
