# Tristan Jones
# Dataset Validator — Totalsegmentator_dataset_v201
# Spring 2026 Capstone
#
# AI Use Disclosure
#   Student estimate: 60% student-designed, 40% AI-assisted implementation
#   Claude assisted with: zip scanning logic, per-patient validation checks
#   See: "Documentation/AI Use Disclosure.md" for full details
#
# Scans every patient folder in the dataset (either from the zip directly
# or from an already-extracted Data/ directory) and checks for all the
# issues that would cause the registration pipeline to fail or produce
# bad results:
#
#   1. Missing required files
#   2. Empty masks (file exists but zero nonzero voxels)
#   3. Implausibly small liver (likely a failed/partial segmentation)
#   4. Broken affine (zero or non-finite voxel spacing)
#   5. Missing or incomplete Couinaud segments (causes landmark count mismatch)
#   6. Non-finite voxel values (NaN / Inf in the volume data)
#
# Outputs:
#   - Live console progress with per-patient pass/fail
#   - Summary table at the end
#   - CSV report saved to disk so you can sort/filter in Excel
#
# Two modes:
#   --source zip   : reads directly from Totalsegmentator_dataset_v201.zip
#                    without extracting to disk (fast, no extra storage needed)
#   --source disk  : reads from an already-extracted Data/ directory
#
# Usage examples:
#   python validate_dataset.py
#   python validate_dataset.py --source zip --zip-path Totalsegmentator_dataset_v201.zip
#   python validate_dataset.py --source disk --data-dir Data
#   python validate_dataset.py --source zip --max-patients 100
#   python validate_dataset.py --source disk --output-csv my_report.csv

from __future__ import annotations

import argparse
import csv
import gzip
import io
import os
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np


# ---------------------------------------------------------------------------
# Configuration — minimum acceptable values
# ---------------------------------------------------------------------------

# Liver voxel count thresholds.  A healthy adult liver is roughly 1000–1800 cc.
# At 1.5mm isotropic spacing that's ~300k–800k voxels.  We flag anything below
# MIN_LIVER_VOXELS as suspiciously small (likely a failed segmentation).
MIN_LIVER_VOXELS     = 50_000   # very conservative lower bound
MIN_PORTAL_VOXELS    = 500      # portal vein can be small but not this small
MIN_HEPATIC_VOXELS   = 500      # same for hepatic vessels
MIN_SEGMENT_VOXELS   = 200      # minimum voxels for a single Couinaud segment

# Minimum number of Couinaud segments required for registration.
# register() uses however many segments exist — fewer than this and the
# Procrustes pre-alignment will be poorly constrained.
# MIN_SEGMENTS_REQUIRED = 8

# Required files per patient folder (after TotalSegmentator has run)
REQUIRED_FILES = [
    "liver.nii.gz",
    "portal_vein.nii.gz",
    "liver_vessels.nii.gz",
]

SEGMENT_FILES = [f"liver_segment_{i}.nii.gz" for i in range(1, 9)]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PatientReport:
    patient_id: str
    passed: bool = True
    issues: list[str] = field(default_factory=list)

    # Per-file stats (filled in during checks)
    liver_voxels:   int = 0
    portal_voxels:  int = 0
    hepatic_voxels: int = 0
    n_segments:     int = 0   # how many of the 8 segments are present + non-empty
    liver_spacing:  str = ""  # e.g. "1.50 x 1.50 x 1.50 mm"

    def fail(self, reason: str) -> None:
        """Execute fail."""
        self.passed = False
        self.issues.append(reason)

    def warn(self, reason: str) -> None:
        """Non-fatal issue — patient still usable but worth noting."""
        self.issues.append(f"[WARN] {reason}")

    @property
    def status(self) -> str:
        """Execute status."""
        if self.passed and not self.issues:
            return "PASS"
        if self.passed:
            return "PASS*"   # passed but has warnings
        return "FAIL"


# ---------------------------------------------------------------------------
# NIfTI loading helpers — work for both on-disk and in-memory (zip) files
# ---------------------------------------------------------------------------

def _load_nifti_from_bytes(data: bytes) -> nib.Nifti1Image:
    """Load a NIfTI image from raw bytes (used when reading from zip)."""
    # Zip members are typically .nii.gz; nibabel expects uncompressed NIfTI
    # bytes when parsing from an in-memory file-like object.
    if data[:2] == b"\x1f\x8b":
        data = gzip.decompress(data)
    fh = nib.FileHolder(fileobj=io.BytesIO(data))
    img = nib.Nifti1Image.from_file_map({"header": fh, "image": fh})
    return img


def _load_nifti_from_path(path: Path) -> nib.Nifti1Image:
    """Helper for load nifti from path."""
    return nib.load(str(path))


def _get_voxel_data(img: nib.Nifti1Image) -> np.ndarray:
    """Return float32 voxel data, catching memory issues gracefully."""
    return np.asarray(img.dataobj, dtype=np.float32)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_affine(img: nib.Nifti1Image, label: str, report: PatientReport) -> bool:
    """
    Check that the affine is well-formed:
      - No zero-length axes (would break mm conversion)
      - No NaN / Inf values
    Returns True if affine is usable.
    """
    affine = img.affine
    if not np.all(np.isfinite(affine)):
        report.fail(f"{label}: affine contains NaN or Inf")
        return False

    zooms = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    if np.any(zooms == 0):
        report.fail(f"{label}: affine has zero-length axis (zooms={zooms.round(3)})")
        return False

    if np.any(zooms > 10):
        report.warn(f"{label}: unusually large voxel spacing {zooms.round(2)} mm "
                    f"— check orientation")

    return True


def _check_volume(img: nib.Nifti1Image,
                  label: str,
                  report: PatientReport,
                  min_voxels: int) -> int:
    """
    Check that the volume has nonzero voxels and meets the minimum count.
    Returns the nonzero voxel count (0 on failure).
    """
    data = _get_voxel_data(img)

    if not np.all(np.isfinite(data)):
        report.fail(f"{label}: volume contains NaN or Inf values")
        return 0

    n_nonzero = int((data > 0).sum())

    if n_nonzero == 0:
        report.fail(f"{label}: mask is completely empty (0 nonzero voxels)")
        return 0

    if n_nonzero < min_voxels:
        report.fail(f"{label}: only {n_nonzero:,} nonzero voxels "
                    f"(minimum {min_voxels:,}) — likely failed segmentation")
        return n_nonzero

    return n_nonzero


# ---------------------------------------------------------------------------
# Per-patient validation
# ---------------------------------------------------------------------------

def validate_patient_disk(patient_dir: Path) -> PatientReport:
    """
    Validate one patient from an on-disk directory structure:
        Data/{patient_id}/liver.nii.gz
        Data/{patient_id}/portal_vein.nii.gz
        ...
    """
    pid    = patient_dir.name
    report = PatientReport(patient_id=pid)

    # ---- Required files ----
    for fname in REQUIRED_FILES:
        fpath = patient_dir / fname
        if not fpath.exists():
            report.fail(f"Missing required file: {fname}")

    # If any required file is missing we can't load them — stop here
    if not report.passed:
        return report

    # ---- Liver ----
    try:
        liver_img = _load_nifti_from_path(patient_dir / "liver.nii.gz")
        if _check_affine(liver_img, "liver", report):
            zooms = np.sqrt((liver_img.affine[:3, :3] ** 2).sum(axis=0))
            report.liver_spacing = " x ".join(f"{z:.2f}" for z in zooms) + " mm"
        report.liver_voxels = _check_volume(
            liver_img, "liver", report, MIN_LIVER_VOXELS)
    except Exception as e:
        report.fail(f"liver.nii.gz failed to load: {e}")

    # ---- Portal vein ----
    try:
        portal_img = _load_nifti_from_path(patient_dir / "portal_vein.nii.gz")
        _check_affine(portal_img, "portal_vein", report)
        report.portal_voxels = _check_volume(
            portal_img, "portal_vein", report, MIN_PORTAL_VOXELS)
    except Exception as e:
        report.fail(f"portal_vein.nii.gz failed to load: {e}")

    # ---- Hepatic vessels ----
    try:
        hepatic_img = _load_nifti_from_path(patient_dir / "liver_vessels.nii.gz")
        _check_affine(hepatic_img, "liver_vessels", report)
        report.hepatic_voxels = _check_volume(
            hepatic_img, "liver_vessels", report, MIN_HEPATIC_VOXELS)
    except Exception as e:
        report.fail(f"liver_vessels.nii.gz failed to load: {e}")

    # ---- Couinaud segments ----
    n_good_segments = 0
    for fname in SEGMENT_FILES:
        fpath = patient_dir / fname
        if not fpath.exists():
            continue
        try:
            seg_img  = _load_nifti_from_path(fpath)
            seg_data = _get_voxel_data(seg_img)
            n_nonzero = int((seg_data > 0).sum())
            if n_nonzero >= MIN_SEGMENT_VOXELS:
                n_good_segments += 1
            else:
                report.warn(f"{fname}: only {n_nonzero} nonzero voxels — "
                            f"segment may be empty")
        except Exception as e:
            report.warn(f"{fname}: failed to load ({e})")

    report.n_segments = n_good_segments
    if n_good_segments < MIN_SEGMENTS_REQUIRED:
        report.fail(f"Only {n_good_segments}/8 usable Couinaud segments "
                    f"(need ≥{MIN_SEGMENTS_REQUIRED} for registration)")

    return report


def validate_patient_zip(zf: zipfile.ZipFile,
                          patient_id: str,
                          all_entries: set[str],
                          zip_layout: str) -> PatientReport:
    """
    Validate one patient reading directly from the zip.
    Zip internal structure: s{patient_id:04d}/{filename}
    After TotalSegmentator has run the outputs live in a Data/ sub-folder,
    but in the raw dataset zip the structure is s{id}/.
    
    This validator handles BOTH layouts:
      Layout A (raw zip):    s0004/liver.nii.gz
      Layout B (processed):  Data/0004/liver.nii.gz  (zipped after processing)
    """
    pid_z  = patient_id.zfill(4)
    report = PatientReport(patient_id=patient_id)

    # Raw TotalSegmentator zip has labels under sXXXX/segmentations/ and
    # uses portal_vein_and_splenic_vein instead of portal_vein.
    is_raw_ts_layout = zip_layout == "raw_totalseg"

    name_aliases = {
        "liver.nii.gz": ["liver.nii.gz"],
        "portal_vein.nii.gz": [
            "portal_vein.nii.gz",
            "portal_vein_and_splenic_vein.nii.gz",
        ],
        "liver_vessels.nii.gz": ["liver_vessels.nii.gz"],
    }

    required_files = ["liver.nii.gz", "portal_vein.nii.gz"]
    if not is_raw_ts_layout:
        required_files.append("liver_vessels.nii.gz")

    def _zip_path(fname: str) -> Optional[str]:
        """Helper for zip path."""
        fname_candidates = name_aliases.get(fname, [fname])
        candidates: list[str] = []

        for current_name in fname_candidates:
            candidates.extend([
                f"{pid_z}/{current_name}",              # 0004/liver.nii.gz
                f"{patient_id}/{current_name}",         # 1366/liver.nii.gz
                f"s{pid_z}/{current_name}",             # s0004/liver.nii.gz
                f"Data/{patient_id}/{current_name}",
                f"Data/{pid_z}/{current_name}",
                f"s{pid_z}/segmentations/{current_name}",
                f"{pid_z}/segmentations/{current_name}",
                f"Data/{patient_id}/segmentations/{current_name}",
                f"Data/{pid_z}/segmentations/{current_name}",
            ])

        for candidate in candidates:
            if candidate in all_entries:
                return candidate
        return None

    def _read(fname: str) -> Optional[bytes]:
        """Helper for read."""
        zpath = _zip_path(fname)
        if zpath is None:
            return None
        with zf.open(zpath) as f:
            return f.read()

    def _load(fname: str) -> Optional[nib.Nifti1Image]:
        """Helper for load."""
        raw = _read(fname)
        if raw is None:
            return None
        try:
            return _load_nifti_from_bytes(raw)
        except Exception as e:
            report.fail(f"{fname}: failed to parse NIfTI ({e})")
            return None

    # ---- Required files ----
    for fname in required_files:
        if _zip_path(fname) is None:
            report.fail(f"Missing required file: {fname}")

    if not report.passed:
        return report

    # ---- Liver ----
    liver_img = _load("liver.nii.gz")
    if liver_img is not None:
        if _check_affine(liver_img, "liver", report):
            zooms = np.sqrt((liver_img.affine[:3, :3] ** 2).sum(axis=0))
            report.liver_spacing = " x ".join(f"{z:.2f}" for z in zooms) + " mm"
        report.liver_voxels = _check_volume(
            liver_img, "liver", report, MIN_LIVER_VOXELS)

    # ---- Portal vein ----
    portal_img = _load("portal_vein.nii.gz")
    if portal_img is not None:
        _check_affine(portal_img, "portal_vein", report)
        report.portal_voxels = _check_volume(
            portal_img, "portal_vein", report, MIN_PORTAL_VOXELS)

    # ---- Hepatic vessels ----
    hepatic_img = _load("liver_vessels.nii.gz")
    if hepatic_img is not None:
        _check_affine(hepatic_img, "liver_vessels", report)
        report.hepatic_voxels = _check_volume(
            hepatic_img, "liver_vessels", report, MIN_HEPATIC_VOXELS)
    elif not is_raw_ts_layout:
        report.fail("Missing required file: liver_vessels.nii.gz")

    # ---- Couinaud segments ----
    n_good_segments = 0
    for fname in SEGMENT_FILES:
        seg_img = _load(fname)
        if seg_img is None:
            continue
        seg_data  = _get_voxel_data(seg_img)
        n_nonzero = int((seg_data > 0).sum())
        if n_nonzero >= MIN_SEGMENT_VOXELS:
            n_good_segments += 1
        else:
            report.warn(f"{fname}: only {n_nonzero} nonzero voxels")

    report.n_segments = n_good_segments
    if (not is_raw_ts_layout) and n_good_segments < MIN_SEGMENTS_REQUIRED:
        report.fail(f"Only {n_good_segments}/8 usable Couinaud segments "
                    f"(need ≥{MIN_SEGMENTS_REQUIRED} for registration)")

    return report


# ---------------------------------------------------------------------------
# Discover patient IDs
# ---------------------------------------------------------------------------

def discover_ids_from_zip(zf: zipfile.ZipFile) -> list[str]:
    """Execute discover ids from zip."""
    ids = set()
    for entry in zf.namelist():
        parts = entry.split("/")
        for token in parts:
            if not token or token.lower() in {"data", "segmentations"}:
                continue
            pid = token.lstrip("s")
            if pid.isdigit():
                ids.add(pid.zfill(4))
                break
    return sorted(ids)


def detect_zip_layout(all_entries: set[str]) -> str:
    """Detect whether the zip is raw TotalSegmentator or processed outputs."""
    if any(e.endswith("/segmentations/liver.nii.gz") for e in all_entries):
        return "raw_totalseg"
    return "processed"


def discover_ids_from_disk(data_dir: Path) -> list[str]:
    """
    Find all patient subdirectories under data_dir that contain at least
    one expected file.
    """
    ids = []
    for p in sorted(data_dir.iterdir()):
        if p.is_dir() and (p / "liver.nii.gz").exists():
            ids.append(p.name)
    return sorted(ids)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(reports: list[PatientReport]) -> None:
    """Execute print summary."""
    passed  = [r for r in reports if r.passed]
    failed  = [r for r in reports if not r.passed]
    warned  = [r for r in passed  if r.issues]

    print(f"\n{'='*70}")
    print(f"  VALIDATION SUMMARY — {len(reports)} patients scanned")
    print(f"{'='*70}")
    print(f"  PASS (clean):   {len(passed) - len(warned):>5}")
    print(f"  PASS (warned):  {len(warned):>5}  (usable but has minor issues)")
    print(f"  FAIL:           {len(failed):>5}  (should be excluded)")
    print(f"{'='*70}\n")

    if failed:
        print(f"  Failed patients ({len(failed)}):")
        # Group by failure reason for a clean summary
        from collections import Counter
        all_issues = []
        for r in failed:
            for issue in r.issues:
                if not issue.startswith("[WARN]"):
                    all_issues.append(issue)
        counts = Counter(all_issues)
        for issue, count in counts.most_common():
            print(f"    {count:>4}x  {issue}")
        print()

    if warned:
        print(f"  Patients with warnings ({len(warned)}):")
        for r in warned[:20]:   # cap at 20 to keep output manageable
            print(f"    {r.patient_id}: "
                  + " | ".join(i for i in r.issues if i.startswith("[WARN]")))
        if len(warned) > 20:
            print(f"    ... and {len(warned) - 20} more (see CSV for full list)")
        print()

    # Print usable patient IDs — this is the list to feed into liver_atlas.py
    usable_ids = sorted(r.patient_id for r in passed)
    print(f"  Usable patient IDs ({len(usable_ids)}):")
    # Print in rows of 10 for readability
    for i in range(0, len(usable_ids), 10):
        print("    " + "  ".join(usable_ids[i:i+10]))
    print()


def save_csv(reports: list[PatientReport], output_path: Path) -> None:
    """Save full per-patient report as a CSV for sorting/filtering in Excel."""
    fieldnames = [
        "patient_id", "status", "liver_voxels", "portal_voxels",
        "hepatic_voxels", "n_segments", "liver_spacing", "issues",
    ]
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in reports:
            writer.writerow({
                "patient_id":     r.patient_id,
                "status":         r.status,
                "liver_voxels":   r.liver_voxels,
                "portal_voxels":  r.portal_voxels,
                "hepatic_voxels": r.hepatic_voxels,
                "n_segments":     r.n_segments,
                "liver_spacing":  r.liver_spacing,
                "issues":         " | ".join(r.issues),
            })
    print(f"  Full report saved → {output_path}")


def save_usable_ids(reports: list[PatientReport], output_path: Path) -> None:
    """Save just the passing patient IDs as a plain text file, one per line."""
    usable = sorted(r.patient_id for r in reports if r.passed)
    with open(output_path, "w", encoding="utf-8") as f:
        for pid in usable:
            f.write(pid + "\n")
    print(f"  Usable IDs saved  → {output_path}  ({len(usable)} patients)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Execute parse args."""
    parser = argparse.ArgumentParser(
        description="Validate TotalSegmentator dataset for liver atlas pipeline."
    )
    parser.add_argument(
        "--source",
        choices=["zip", "disk"],
        default="zip",
        help="Where to read patient data from (default: zip)",
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=Path(__file__).resolve().parent / "Totalsegmentator_dataset_v201.zip",
        help="Path to the dataset zip file",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "Data",
        help="Path to the extracted Data/ directory (used when --source disk)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(__file__).resolve().parent / "validation_report.csv",
        help="Where to save the CSV report",
    )
    parser.add_argument(
        "--output-ids",
        type=Path,
        default=Path(__file__).resolve().parent / "usable_patient_ids.txt",
        help="Where to save the list of passing patient IDs",
    )
    parser.add_argument(
        "--max-patients",
        type=int,
        default=None,
        help="Stop after this many patients (useful for a quick test run)",
    )
    parser.add_argument(
        "--min-liver-voxels",
        type=int,
        default=MIN_LIVER_VOXELS,
        help=f"Minimum liver voxel count (default {MIN_LIVER_VOXELS:,})",
    )
    parser.add_argument(
        "--min-segments",
        type=int,
        default=MIN_SEGMENTS_REQUIRED,
        help=f"Minimum Couinaud segments required (default {MIN_SEGMENTS_REQUIRED})",
    )
    return parser.parse_args()


def main() -> int:
    """Execute main."""
    args = parse_args()

    # Allow overriding thresholds from CLI
    global MIN_LIVER_VOXELS, MIN_SEGMENTS_REQUIRED
    MIN_LIVER_VOXELS      = args.min_liver_voxels
    MIN_SEGMENTS_REQUIRED = args.min_segments

    reports: list[PatientReport] = []

    if args.source == "zip":
        if not args.zip_path.exists():
            print(f"[ERROR] Zip not found: {args.zip_path}")
            return 1

        print(f"Reading from zip: {args.zip_path}")
        with zipfile.ZipFile(args.zip_path, "r") as zf:
            all_entries = set(zf.namelist())
            zip_layout = detect_zip_layout(all_entries)
            patient_ids = discover_ids_from_zip(zf)

            print(f"Detected zip layout: {zip_layout}")

            if args.max_patients:
                patient_ids = patient_ids[:args.max_patients]

            print(f"Found {len(patient_ids)} patient IDs in zip.\n")

            for i, pid in enumerate(patient_ids, start=1):
                report = validate_patient_zip(zf, pid, all_entries, zip_layout)
                reports.append(report)

                # Live progress line
                status = f"{'✓' if report.passed else '✗'}"
                issues = (f"  → {report.issues[0]}"
                          if report.issues and not report.passed else "")
                print(f"  [{i:>4}/{len(patient_ids)}]  {status}  "
                      f"s{pid}  liver={report.liver_voxels:>8,}  "
                      f"segs={report.n_segments}/8{issues}")

    else:  # disk
        if not args.data_dir.exists():
            print(f"[ERROR] Data directory not found: {args.data_dir}")
            return 1

        print(f"Reading from disk: {args.data_dir}")
        patient_ids = discover_ids_from_disk(args.data_dir)

        if args.max_patients:
            patient_ids = patient_ids[:args.max_patients]

        print(f"Found {len(patient_ids)} patient directories.\n")

        for i, pid in enumerate(patient_ids, start=1):
            patient_dir = args.data_dir / pid
            report      = validate_patient_disk(patient_dir)
            reports.append(report)

            status = f"{'✓' if report.passed else '✗'}"
            issues = (f"  → {report.issues[0]}"
                      if report.issues and not report.passed else "")
            print(f"  [{i:>4}/{len(patient_ids)}]  {status}  "
                  f"{pid}  liver={report.liver_voxels:>8,}  "
                  f"segs={report.n_segments}/8{issues}")

    print_summary(reports)
    save_csv(reports, args.output_csv)
    save_usable_ids(reports, args.output_ids)

    n_failed = sum(1 for r in reports if not r.passed)
    return 1 if n_failed > 0 else 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
