#!/usr/bin/env python3
"""
Download, extract, and organize the AISD dataset for ADN training/testing.

The AISD dataset is distributed as per-slice PNG images inside zip files:
    image/{patient_id}/{slice:03d}.png    (CT slices)
    mask/{patient_id}/{slice:03d}.png      (segmentation masks)

This script stacks the 2D PNG slices into 3D NIfTI volumes expected by ADN:
    {data_dir}/{patient_id}/CT.nii
    {data_dir}/{patient_id}/mask.nii

Usage:
    python download_aisd.py
    python download_aisd.py --data-dir /path/to/AISD_data_resample
    python download_aisd.py --skip-download   # re-run from cached zips
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

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

PATIENT_ID_RE = re.compile(r"(\d{7})")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_gdown():
    try:
        import gdown  # noqa: F401
    except ImportError:
        print("[INFO] gdown not found. Installing...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "gdown"],
            stdout=subprocess.DEVNULL,
        )


def ensure_nibabel():
    try:
        import nibabel  # noqa: F401
    except ImportError:
        print("[INFO] nibabel not found. Installing...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "nibabel"],
            stdout=subprocess.DEVNULL,
        )


def download_file(file_id: str, output_path: str, desc: str = ""):
    import gdown
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"[INFO] Downloading {desc}...")
    print(f"       -> {output_path}")
    gdown.download(url, output_path, quiet=False, resume=True)
    if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
        print(f"[ERROR] Downloaded file is empty or missing: {output_path}", file=sys.stderr)
        sys.exit(1)
    gb = os.path.getsize(output_path) / 1024**3
    print(f"[INFO] {desc} downloaded ({gb:.2f} GB).")


def extract_zip(zip_path: str, extract_dir: str, desc: str = ""):
    print(f"[INFO] Extracting {desc}...")
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)


def find_patients(extract_dir: str) -> list[str]:
    """
    Scan extracted zip and return sorted list of patient IDs.

    Expected structure:
        {extract_dir}/image/{patient_id}/{slice:03d}.png
    or  {extract_dir}/mask/{patient_id}/{slice:03d}.png

    Returns patient IDs sorted numerically.
    """
    # The zip root has a single top-level folder: "image" or "mask"
    top = os.path.join(extract_dir, os.listdir(extract_dir)[0])
    pids = []
    for entry in sorted(os.listdir(top)):
        entry_path = os.path.join(top, entry)
        if os.path.isdir(entry_path):
            m = PATIENT_ID_RE.match(entry)
            if m:
                pids.append(m.group(1))
    return pids


def count_slices(patient_dir: str) -> int:
    """Return number of PNG slices in *patient_dir*."""
    return len(sorted(Path(patient_dir).glob("*.png")))


def stack_and_pad(png_dir: str, target_depth: int) -> np.ndarray:
    """
    Read PNG slices from *png_dir*, stack into 3D array (D, H, W),
    then pad along axis-0 with zeros to reach *target_depth*.

    If the volume already has *target_depth* slices, no padding is applied.
    """
    pngs = sorted(Path(png_dir).glob("*.png"))
    if not pngs:
        return None
    slices = []
    for p in pngs:
        im = Image.open(p)
        arr = np.array(im, dtype=np.float32)
        slices.append(arr)
    vol = np.stack(slices, axis=0)  # (D, H, W)
    d = vol.shape[0]
    if d < target_depth:
        pad_shape = (target_depth - d, vol.shape[1], vol.shape[2])
        vol = np.concatenate([vol, np.zeros(pad_shape, dtype=np.float32)], axis=0)
    elif d > target_depth:
        vol = vol[:target_depth, ...]
    return vol


def organize(
    image_extract_dir: str,
    mask_extract_dir: str,
    data_dir: str,
) -> list[str]:
    """
    For each patient, stack PNG slices into 3D NIfTI volumes and
    save as CT.nii / mask.nii under {data_dir}/{patient_id}/.

    All volumes are padded/cropped to a uniform depth (the maximum
    number of slices found across the dataset) so that the ADN
    DataLoader can form batches of consistent shape.
    """
    import nibabel as nib

    img_root = os.path.join(image_extract_dir, os.listdir(image_extract_dir)[0])
    msk_root = os.path.join(mask_extract_dir, os.listdir(mask_extract_dir)[0])

    # Build union of all patient IDs found in both zips
    img_pids = set()
    for d in os.listdir(img_root):
        m = PATIENT_ID_RE.match(d)
        if m:
            img_pids.add(m.group(1))

    msk_pids = set()
    for d in os.listdir(msk_root):
        m = PATIENT_ID_RE.match(d)
        if m:
            msk_pids.add(m.group(1))

    all_ids = sorted(img_pids | msk_pids)
    print(f"[INFO] Found {len(all_ids)} unique patients "
          f"({len(img_pids)} with CT, {len(msk_pids)} with masks).")

    if not all_ids:
        return []

    # ---- First pass: determine target depth --------------------------------
    depths = []
    for pid in all_ids:
        img_patient_dir = os.path.join(img_root, pid)
        if os.path.isdir(img_patient_dir):
            depths.append(count_slices(img_patient_dir))
    target_depth = max(depths) if depths else 40
    depth_range = (min(depths), max(depths)) if depths else (0, 0)
    print(f"[INFO] Slice count range: {depth_range[0]}-{depth_range[1]}. "
          f"Padding/cropping all volumes to {target_depth} slices.")

    # ---- Second pass: stack, pad, save ------------------------------------
    converted = 0
    skipped_ct = 0
    skipped_mask = 0

    for pid in all_ids:
        patient_dir = os.path.join(data_dir, pid)
        os.makedirs(patient_dir, exist_ok=True)

        # CT slices
        img_patient_dir = os.path.join(img_root, pid)
        if os.path.isdir(img_patient_dir):
            vol = stack_and_pad(img_patient_dir, target_depth)
            if vol is not None:
                nii = nib.Nifti1Image(vol, affine=np.eye(4))
                nib.save(nii, os.path.join(patient_dir, "CT.nii"))
                converted += 1
            else:
                skipped_ct += 1
                print(f"  [WARN] No PNGs found for CT: {pid}")
        else:
            skipped_ct += 1
            print(f"  [WARN] CT directory not found: {pid}")

        # Mask slices
        msk_patient_dir = os.path.join(msk_root, pid)
        if os.path.isdir(msk_patient_dir):
            vol = stack_and_pad(msk_patient_dir, target_depth)
            if vol is not None:
                nii = nib.Nifti1Image(vol, affine=np.eye(4))
                nib.save(nii, os.path.join(patient_dir, "mask.nii"))
            else:
                skipped_mask += 1
                print(f"  [WARN] No PNGs found for mask: {pid}")
        else:
            skipped_mask += 1
            print(f"  [WARN] Mask directory not found: {pid}")

    print(f"[INFO] Converted {converted} patients to NIfTI "
          f"(target depth = {target_depth} slices).")
    if skipped_ct:
        print(f"  [WARN] {skipped_ct} patients missing CT")
    if skipped_mask:
        print(f"  [WARN] {skipped_mask} patients missing masks")

    # Verify a few NIfTIs
    sample_ids = all_ids[:3]
    for pid in sample_ids:
        ct_path = os.path.join(data_dir, pid, "CT.nii")
        msk_path = os.path.join(data_dir, pid, "mask.nii")
        if os.path.isfile(ct_path):
            img = nib.load(ct_path)
            print(f"  [INFO] {pid}: CT shape={img.shape}, dtype={img.get_data_dtype()}")
        if os.path.isfile(msk_path):
            img = nib.load(msk_path)
            print(f"  [INFO] {pid}: mask shape={img.shape}, dtype={img.get_data_dtype()}")

    return all_ids


def create_splits(data_dir: str, all_ids: list[str]):
    all_set = set(all_ids)
    test_present = sorted(all_set & TEST_PATIENT_IDS)
    train_present = sorted(all_set - TEST_PATIENT_IDS)

    test_path = os.path.join(data_dir, "aisd_test.txt")
    with open(test_path, "w") as f:
        for pid in test_present:
            f.write(pid + "\n")
    print(f"[INFO] Created {test_path} ({len(test_present)} patients)")

    train_path = os.path.join(data_dir, "aisd_train.txt")
    with open(train_path, "w") as f:
        for pid in train_present:
            f.write(pid + "\n")
    print(f"[INFO] Created {train_path} ({len(train_present)} patients)")

    missing = sorted(TEST_PATIENT_IDS - all_set)
    if missing:
        print(f"  [WARN] {len(missing)} test IDs missing from dataset:")
        print(f"         {', '.join(missing)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Download, extract, and organize the AISD dataset for ADN.",
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join(os.getcwd(), "data", "AISD_data_resample"),
        help="Output directory for organized dataset (default: ./data/AISD_data_resample)",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.path.join(os.getcwd(), "data", "downloads"),
        help="Where to store downloaded zip files (default: ./data/downloads)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading; only organize from existing zips",
    )
    return parser.parse_args()


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
        download_file(GDRIVE_IMAGE_ID, image_zip, "image.zip (CT scans)")
        download_file(GDRIVE_MASK_ID, mask_zip, "mask.zip (segmentation)")
    else:
        for fp in (image_zip, mask_zip):
            if not os.path.isfile(fp):
                print(f"[ERROR] --skip-download but file not found: {fp}", file=sys.stderr)
                sys.exit(1)

    # ---- Extract + convert to NIfTI ----------------------------------------
    ensure_nibabel()

    with tempfile.TemporaryDirectory(prefix="aisd-img-") as img_tmp:
        extract_zip(image_zip, img_tmp, "image.zip")
        with tempfile.TemporaryDirectory(prefix="aisd-msk-") as msk_tmp:
            extract_zip(mask_zip, msk_tmp, "mask.zip")
            all_ids = organize(img_tmp, msk_tmp, data_dir)

    # ---- Train / test splits -----------------------------------------------
    create_splits(data_dir, all_ids)

    print()
    print("=" * 60)
    print("  AISD dataset ready for ADN!")
    print(f"  Data directory: {data_dir}")
    print(f"  Total patients: {len(all_ids)}")
    print(f"  Train: {len(set(all_ids) - TEST_PATIENT_IDS)}")
    print(f"  Test:  {len(set(all_ids) & TEST_PATIENT_IDS)}")
    print()
    print("  To train T:")
    print(f"    python train_align_model.py --data-dir {data_dir} \\")
    print(f"        --train-txt {os.path.join(data_dir, 'aisd_train.txt')}")
    print()
    print("  To train D+F (after T is trained):")
    print(f"    python train.py --data-dir {data_dir} \\")
    print(f"        --train-txt {os.path.join(data_dir, 'aisd_train.txt')}")
    print()
    print("  To test:")
    print(f"    python test.py --data-dir {data_dir} \\")
    print(f"        --test-txt {os.path.join(data_dir, 'aisd_test.txt')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
