#!/usr/bin/env python3
"""
Download, extract, and organize the AISD dataset for ADN training/testing.

Usage:
    # Default: organize under ./data/AISD_data_resample
    python download_aisd.py

    # Custom output directory
    python download_aisd.py --data-dir /data/StrokeCT/AISD_data_resample

    # Skip download if zips already exist
    python download_aisd.py --cache-dir ./zips --data-dir ./data

This downloads the AISD (Acute Ischemic Stroke Dataset) used by the ADN paper.
The dataset contains 397 NCCT scans of acute ischemic stroke patients.

Test set (52 patients): hardcoded from the official AISD repository.
Train set (remaining 345 patients): inferred by subtracting test IDs from all downloaded patients.
"""

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Google Drive file IDs from https://github.com/GriffinLiang/AISD
# ---------------------------------------------------------------------------
GDRIVE_IMAGE_ID = "157f9aE3ZhRSdIuIbP2PRG8ub9JJWvMGk"
GDRIVE_MASK_ID = "1d08fFpEvK4D6YTKfRlNuv_OlIxigZxl6"

# ---------------------------------------------------------------------------
# 52 test-set patient IDs — from the official AISD README
# ---------------------------------------------------------------------------
TEST_PATIENT_IDS = {
    "0073410", "0072723", "0226290", "0537908", "0538058",
    "0091415", "0538780", "0073540", "0226188", "0226258",
    "0226314", "0091507", "0226298", "0538975", "0226257",
    "0226142", "0072681", "0091538", "0538983", "0537961",
    "0091646", "0072765", "0226137", "0091621", "0091458",
    "0021822", "0538319", "0226133", "0091657", "0537925",
    "0073489", "0538502", "0091476", "0226136", "0538532",
    "0073312", "0539025", "0226309", "0226307", "0091383",
    "0021092", "0537990", "0226299", "0073060", "0538505",
    "0073424", "0091534", "0226125", "0072691", "0538425",
    "0226199", "0226261",
}

# ---------------------------------------------------------------------------
# Patient ID pattern — matches AISD naming convention
# ---------------------------------------------------------------------------
PATIENT_ID_PATTERN = re.compile(r"(\d{7})")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_gdown():
    """Check gdown is installed; try to install if missing."""
    try:
        import gdown  # noqa: F401
    except ImportError:
        print("[INFO] gdown not found. Installing...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "gdown"],
            stdout=subprocess.DEVNULL,
        )
        print("[INFO] gdown installed.")


def download_file(file_id: str, output_path: str, desc: str = ""):
    """
    Download a file from Google Drive using gdown.
    gdown handles the large-file confirmation dialog automatically.
    """
    import gdown

    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"[INFO] Downloading {desc}...")
    print(f"       URL: {url}")
    print(f"       -> {output_path}")
    try:
        gdown.download(url, output_path, quiet=False, resume=True)
    except Exception as e:
        print(f"[ERROR] Download failed: {e}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
        print(f"[ERROR] Downloaded file is empty or missing: {output_path}", file=sys.stderr)
        sys.exit(1)
    print(f"[INFO] {desc} downloaded ({os.path.getsize(output_path) / 1024**3:.2f} GB).")


def extract_zip(zip_path: str, extract_dir: str, desc: str = ""):
    """Extract a zip file, preserving its internal directory structure."""
    print(f"[INFO] Extracting {desc}...")
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    print(f"[INFO] {desc} extracted to {extract_dir}")


def find_nii_files(directory: str) -> list[tuple[str, str]]:
    """
    Walk *directory* and return a list of ``(patient_id, nii_path)`` tuples
    for every ``.nii`` file found.

    Patient IDs are discovered in order of priority:
      1. The parent directory name if it looks like an ID.
      2. The stem of the filename if it looks like an ID.
      3. The first 7-digit number found anywhere in the relative path.

    Files containing "mask" in their stem are excluded (they belong to the
    masks zip, not the images zip).
    """
    results: list[tuple[str, str]] = []
    for root, _dirs, files in os.walk(directory):
        for fname in sorted(files):
            if not fname.endswith(".nii"):
                continue
            fpath = os.path.join(root, fname)
            stem = Path(fname).stem.lower()

            # Skip mask files when scanning the image directory
            # (this is a safety check; masks shouldn't be in image.zip anyway)
            if "mask" in stem:
                continue

            rel_path = os.path.relpath(fpath, directory)
            parent_dir = os.path.basename(os.path.dirname(rel_path))

            # Priority 1: parent directory looks like a patient ID
            m = PATIENT_ID_PATTERN.match(parent_dir)
            if m:
                results.append((m.group(1), fpath))
                continue

            # Priority 2: filename stem looks like a patient ID
            m = PATIENT_ID_PATTERN.match(stem)
            if m:
                results.append((m.group(1), fpath))
                continue

            # Priority 3: first 7-digit number anywhere in the relative path
            m = PATIENT_ID_PATTERN.search(rel_path)
            if m:
                results.append((m.group(1), fpath))
                continue

            print(f"  [WARN] Could not identify patient ID for: {rel_path}")
    return results


def organize(
    image_extract_dir: str,
    mask_extract_dir: str,
    data_dir: str,
):
    """
    Detect the structure of the extracted zip contents and copy files into
    ``{data_dir}/{patient_id}/CT.nii`` and ``{data_dir}/{patient_id}/mask.nii``.
    """
    print("[INFO] Scanning extracted image files...")
    image_files = find_nii_files(image_extract_dir)
    print(f"       Found {len(image_files)} CT scans.")

    print("[INFO] Scanning extracted mask files...")
    # For masks we use a relaxed matcher (no "mask" exclusion)
    mask_files: list[tuple[str, str]] = []
    for root, _dirs, files in os.walk(mask_extract_dir):
        for fname in sorted(files):
            if not fname.endswith(".nii"):
                continue
            fpath = os.path.join(root, fname)
            stem = Path(fname).stem
            rel_path = os.path.relpath(fpath, mask_extract_dir)
            parent_dir = os.path.basename(os.path.dirname(rel_path))

            m = PATIENT_ID_PATTERN.match(parent_dir)
            if m:
                mask_files.append((m.group(1), fpath))
                continue
            m = PATIENT_ID_PATTERN.match(stem)
            if m:
                mask_files.append((m.group(1), fpath))
                continue
            m = PATIENT_ID_PATTERN.search(rel_path)
            if m:
                mask_files.append((m.group(1), fpath))
                continue
            print(f"  [WARN] Could not identify patient ID for mask: {rel_path}")
    print(f"       Found {len(mask_files)} masks.")

    # Build lookup dicts
    image_map: dict[str, str] = dict(image_files)
    mask_map: dict[str, str] = dict(mask_files)

    all_ids = sorted(set(image_map.keys()) | set(mask_map.keys()))
    print(f"[INFO] Total unique patient IDs found: {len(all_ids)}")

    copied = 0
    missing_ct = 0
    missing_mask = 0

    for pid in all_ids:
        patient_dir = os.path.join(data_dir, pid)
        os.makedirs(patient_dir, exist_ok=True)

        ct_src = image_map.get(pid)
        mask_src = mask_map.get(pid)

        if ct_src:
            shutil.copy2(ct_src, os.path.join(patient_dir, "CT.nii"))
            copied += 1
        else:
            missing_ct += 1
            print(f"  [WARN] Missing CT.nii for patient {pid}")

        if mask_src:
            shutil.copy2(mask_src, os.path.join(patient_dir, "mask.nii"))
        else:
            missing_mask += 1
            print(f"  [WARN] Missing mask.nii for patient {pid}")

    print(f"[INFO] Organized {copied} patients into {data_dir}")
    if missing_ct:
        print(f"  [WARN] {missing_ct} patients missing CT scans")
    if missing_mask:
        print(f"  [WARN] {missing_mask} patients missing masks")

    return all_ids


def create_splits(data_dir: str, all_ids: list[str]):
    """Create train.txt and test.txt in data_dir."""
    all_ids_set = set(all_ids)
    test_ids_present = all_ids_set & TEST_PATIENT_IDS
    train_ids_present = sorted(all_ids_set - TEST_PATIENT_IDS)

    # Write test.txt — only IDs that are actually present in the downloaded data
    test_txt = os.path.join(data_dir, "aisd_test.txt")
    with open(test_txt, "w") as f:
        for pid in sorted(test_ids_present):
            f.write(pid + "\n")
    print(f"[INFO] Created {test_txt} ({len(test_ids_present)} patients)")

    # Write train.txt
    train_txt = os.path.join(data_dir, "aisd_train.txt")
    with open(train_txt, "w") as f:
        for pid in train_ids_present:
            f.write(pid + "\n")
    print(f"[INFO] Created {train_txt} ({len(train_ids_present)} patients)")

    # Sanity-check test IDs that were NOT found (allow partial downloads)
    missing_test_ids = sorted(TEST_PATIENT_IDS - all_ids_set)
    if missing_test_ids:
        print(f"  [WARN] {len(missing_test_ids)} test-set patients not found in downloaded data:")
        print(f"         {', '.join(missing_test_ids)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download, extract, and organize the AISD dataset for ADN.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.join(os.getcwd(), "data", "AISD_data_resample"),
        help="Output directory for the organized dataset "
             "(default: ./data/AISD_data_resample)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=os.path.join(os.getcwd(), "data", "downloads"),
        help="Where to store the downloaded zip files (default: ./data/downloads)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading; only organize from existing zips in cache-dir",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    data_dir = os.path.abspath(args.data_dir)
    cache_dir = os.path.abspath(args.cache_dir)

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    image_zip = os.path.join(cache_dir, "image.zip")
    mask_zip = os.path.join(cache_dir, "mask.zip")

    # ---- Download -----------------------------------------------------------
    if not args.skip_download:
        ensure_gdown()
        download_file(GDRIVE_IMAGE_ID, image_zip, desc="image.zip (CT scans)")
        download_file(GDRIVE_MASK_ID, mask_zip, desc="mask.zip (segmentation)")
    else:
        for fp in (image_zip, mask_zip):
            if not os.path.isfile(fp):
                print(f"[ERROR] --skip-download but file not found: {fp}", file=sys.stderr)
                sys.exit(1)

    # ---- Extract ------------------------------------------------------------
    with tempfile.TemporaryDirectory(prefix="aisd-img-") as img_tmp:
        extract_zip(image_zip, img_tmp, "image.zip")

        with tempfile.TemporaryDirectory(prefix="aisd-msk-") as msk_tmp:
            extract_zip(mask_zip, msk_tmp, "mask.zip")

            # ---- Organize ---------------------------------------------------
            all_ids = organize(img_tmp, msk_tmp, data_dir)

    # ---- Create train / test splits -----------------------------------------
    create_splits(data_dir, all_ids)

    # ---- Summary ------------------------------------------------------------
    print()
    print("=" * 60)
    print("  AISD dataset ready for ADN!")
    print(f"  Data directory: {data_dir}")
    print(f"  Total patients: {len(all_ids)}")
    print(f"  Train: {len(set(all_ids) - TEST_PATIENT_IDS)}")
    print(f"  Test:  {len(set(all_ids) & TEST_PATIENT_IDS)}")
    print()
    print("  To train the transformation network T:")
    print(f"    python train_align_model.py --data-dir {data_dir} \\")
    print(f"        --train-txt {data_dir}/aisd_train.txt")
    print()
    print("  To train D+F (after training T):")
    print(f"    python train.py --data-dir {data_dir} \\")
    print(f"        --train-txt {data_dir}/aisd_train.txt")
    print()
    print("  To test:")
    print(f"    python test.py --data-dir {data_dir} \\")
    print(f"        --test-txt {data_dir}/aisd_test.txt")
    print("=" * 60)


if __name__ == "__main__":
    main()
