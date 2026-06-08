# AI Use Disclosure
#   Student estimate: 40% student-designed, 60% AI-assisted implementation
#   Claude assisted with: nibabel zip-reading (FileHolder + BytesIO), logging, portal vein fallback logic
#   See: "Documentation/AI Use Disclosure.md" for full details

"""
stages/load.py — Stage 1: Load all NIfTI segmentation files for one patient.

Returns a standardized dict so every downstream stage works with the same
data shape regardless of how files are named or laid out on disk.

Supports two sources — checked in this order:
  1. Data/segmentations.zip  — {patient_id}/{filename}.nii.gz
  2. Data/{patient_id}/      — loose files on disk (fallback)
"""

import io
import gzip
import logging
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np

log = logging.getLogger(__name__)

# Candidate filenames for the portal vein (TotalSegmentator names differ by version)
_PORTAL_CANDIDATES = [
    "portal_vein.nii.gz",
    "portal_vein_and_splenic_vein.nii.gz",
]


def _load_nifti(path: Path):
    """Load a NIfTI from disk. Returns (array, affine) or raises FileNotFoundError."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    img     = nib.load(str(path))
    data    = img.get_fdata(dtype=np.float32)
    affine  = img.affine
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0)).round(2)
    log.info(f"    Loaded {path.name}  shape={data.shape}  spacing={spacing} mm")
    return data, affine


def _load_nifti_from_bytes(data: bytes):
    """Load a NIfTI from raw bytes (used when reading from zip)."""
    # Zip entries are typically .nii.gz; from_file_map expects decompressed
    # NIfTI bytes, so gunzip first when needed.
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)

    fh  = nib.FileHolder(fileobj=io.BytesIO(data))
    img = nib.Nifti1Image.from_file_map({"header": fh, "image": fh})
    return img.get_fdata(dtype=np.float32), img.affine


def load_patient(data_dir: Path, patient_id: str) -> dict:
    """
    Load all segmentation structures for one patient.

    Checks Data/segmentations.zip first, falls back to Data/{patient_id}/
    on disk. Required files (liver) raise on failure. Optional files log a
    warning and return None if missing.

    Args:
        data_dir:   Root data directory (e.g. Path("Data"))
        patient_id: Patient folder name (e.g. "0004")

    Returns:
        dict with keys:
            "patient_id"    : str
            "liver"         : np.ndarray, shape (X, Y, Z)
            "affine"        : np.ndarray, shape (4, 4)
            "portal_vein"   : np.ndarray or None
            "hepatic_vein"  : np.ndarray or None
            "segs"          : dict {1: array, ..., 8: array}  (may be empty)
    """
    data_dir = Path(data_dir)
    seg_zip  = data_dir / "segmentations.zip"
    use_zip  = seg_zip.exists()

    raw_pid = str(patient_id).strip()

    # Support both "0004" and "s0004" style IDs.
    digits_pid = raw_pid[1:] if (raw_pid.lower().startswith("s") and raw_pid[1:].isdigit()) else raw_pid
    pid_variants = []
    for pid in [raw_pid, digits_pid]:
        if pid and pid not in pid_variants:
            pid_variants.append(pid)
    if digits_pid.isdigit():
        zpid = digits_pid.zfill(4)
        for pid in [zpid, f"s{zpid}"]:
            if pid not in pid_variants:
                pid_variants.append(pid)

    log.info(f"Loading patient {patient_id} "
             f"({'zip' if use_zip else 'disk'})")

    def _load(filename):
        """Load one file from zip or disk. Returns (array, affine) or (None, None)."""
        if use_zip:
            with zipfile.ZipFile(seg_zip, "r") as zf:
                all_entries = set(zf.namelist())
                entry = None
                for pid in pid_variants:
                    for prefix in ("", "Data/"):
                        candidate = f"{prefix}{pid}/{filename}"
                        if candidate in all_entries:
                            entry = candidate
                            break
                    if entry is not None:
                        break

                if entry is None:
                    return None, None

                raw = zf.read(entry)

            data, affine = _load_nifti_from_bytes(raw)
            spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0)).round(2)
            log.info(f"    Loaded {filename}  shape={data.shape}  spacing={spacing} mm")
            return data, affine
        else:
            for pid in pid_variants:
                path = data_dir / pid / filename
                if path.exists():
                    return _load_nifti(path)
            return None, None

    # --- Required: liver mask ---
    liver, affine = _load("liver.nii.gz")
    if liver is None:
        raise FileNotFoundError(
            f"liver.nii.gz not found for patient {patient_id} "
            f"({'in ' + str(seg_zip) if use_zip else 'at ' + str(data_dir / patient_id)})"
        )

    # --- Optional: portal vein (try multiple filenames) ---
    portal_vein = None
    for name in _PORTAL_CANDIDATES:
        data, _ = _load(name)
        if data is not None:
            portal_vein = data
            break
    if portal_vein is None:
        log.warning(f"    Portal vein not found for {patient_id} — skipping")

    # --- Optional: hepatic vessels ---
    hepatic_vein, _ = _load("liver_vessels.nii.gz")
    if hepatic_vein is None:
        log.warning(f"    Hepatic vessels not found for {patient_id} — skipping")

    # --- Optional: Couinaud segments 1–8 ---
    segs = {}
    for i in range(1, 9):
        seg_data, _ = _load(f"liver_segment_{i}.nii.gz")
        if seg_data is not None:
            segs[i] = seg_data
        else:
            log.debug(f"    Segment {i} not found for {patient_id}")

    if segs:
        log.info(f"    Loaded {len(segs)}/8 Couinaud segments")
    else:
        log.warning(f"    No Couinaud segments found for {patient_id}")

    return {
        "patient_id":   patient_id,
        "liver":        liver,
        "affine":       affine,
        "portal_vein":  portal_vein,
        "hepatic_vein": hepatic_vein,
        "segs":         segs,
    }