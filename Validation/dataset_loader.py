# Validation/dataset_loader.py
# Tristan Jones — Spring 2026 Capstone
#
# AI Use Disclosure
#   Student estimate: 60% student-designed, 40% AI-assisted implementation
#   Claude assisted with: full function implementation
#   See: "Documentation/AI Use Disclosure.md" for full details
#
# Reads the manually-reviewed patient CSV files from Data/Dataset_Reviewed/
# and returns filtered lists of patient IDs ready to pass into the Atlas.
#
# Expected CSV format (one file per cohort):
#   patient_id,gender,voxel_count
#   0004,M,412847
#   0010,F,387234
#   ...
#
# Files expected in Data/Dataset_Reviewed/:
#   reviewed_all.csv     — full reviewed cohort
#   reviewed_male.csv    — male patients only
#   reviewed_female.csv  — female patients only
#
# If your filenames differ just pass them explicitly to load_patient_ids().

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Default filenames — change here if yours differ
_DEFAULT_FILES = {
    "all":    "reviewed_all.csv",
    "male":   "reviewed_male.csv",
    "female": "reviewed_female.csv",
}

# Column names — change here if your CSV uses different headers
_COL_ID     = "patient_id"
_COL_GENDER = "gender"
_COL_VOXELS = "voxel_count"


def load_patient_ids(
    csv_path: Path,
    gender: Optional[str] = None,
    min_voxels: Optional[int] = None,
    max_voxels: Optional[int] = None,
    exclude_ids: Optional[list[str]] = None,
) -> list[str]:
    """
    Read a reviewed CSV and return a filtered list of patient ID strings.

    Args:
        csv_path    : Path to the CSV file.
        gender      : If given, keep only "M" or "F" rows (case-insensitive).
        min_voxels  : Drop patients with fewer voxels than this.
        max_voxels  : Drop patients with more voxels than this.
        exclude_ids : List of patient IDs to explicitly exclude.

    Returns:
        Sorted list of zero-padded patient ID strings, e.g. ["0004", "0010"].
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset CSV not found: {csv_path}")

    exclude = set(exclude_ids or [])
    rows    = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)

        # Be flexible about column name casing
        fieldnames_lower = {fn.strip().lower(): fn.strip()
                            for fn in (reader.fieldnames or [])}

        def _get(row, col):
            """Case-insensitive column lookup."""
            actual = fieldnames_lower.get(col.lower())
            return row.get(actual, "").strip() if actual else ""

        for row in reader:
            raw_pid = _get(row, _COL_ID)
            pid = raw_pid.lstrip("sS").zfill(4)   # normalize s0004/0004 -> 0004

            if pid in exclude:
                continue

            # Gender filter
            if gender is not None:
                row_gender = _get(row, _COL_GENDER).upper()
                if row_gender != gender.upper():
                    continue

            # Voxel count filters
            voxel_str = _get(row, _COL_VOXELS)
            if voxel_str and (min_voxels is not None or max_voxels is not None):
                try:
                    n = int(voxel_str)
                    if min_voxels is not None and n < min_voxels:
                        continue
                    if max_voxels is not None and n > max_voxels:
                        continue
                except ValueError:
                    log.warning(f"  Could not parse voxel_count for {pid}: '{voxel_str}'")

            rows.append(pid)

    result = sorted(set(rows))
    log.info(f"  Loaded {len(result)} patient IDs from {csv_path.name}"
             + (f" (gender={gender})" if gender else "")
             + (f" (min_voxels={min_voxels})" if min_voxels else ""))
    return result


def load_cohort(
    data_dir: Path,
    cohort: str = "all",
    gender: Optional[str] = None,
    min_voxels: Optional[int] = None,
    exclude_ids: Optional[list[str]] = None,
    use_validated_ids: bool = True,
    validated_ids_path: Optional[Path] = None,
) -> list[str]:
    """
    Convenience wrapper — loads from Data/Dataset_Reviewed/<cohort>.csv.

    Args:
        data_dir    : Root data directory (e.g. Path("Data")).
        cohort      : "all", "male", or "female"  (maps to default filenames).
                      Or pass any filename stem to use a custom file.
        gender      : Optional additional gender filter on top of cohort file.
        min_voxels  : Optional minimum voxel count filter.
        exclude_ids : Patient IDs to skip.
        use_validated_ids : If True, intersect with usable_patient_ids.txt.
        validated_ids_path: Optional override path for usable IDs text file.

    Returns:
        Sorted list of patient ID strings.

    Example:
        ids = load_cohort(Path("Data"), cohort="all")
        ids = load_cohort(Path("Data"), cohort="all", gender="F")
        ids = load_cohort(Path("Data"), cohort="female", min_voxels=100_000)
    """
    reviewed_dir = Path(data_dir) / "Dataset_Reviewed"

    # Try the cohort name as a key in defaults first, then as a bare filename
    filename = _DEFAULT_FILES.get(cohort.lower(), f"{cohort}.csv")
    csv_path = reviewed_dir / filename

    ids = load_patient_ids(
        csv_path    = csv_path,
        gender      = gender,
        min_voxels  = min_voxels,
        exclude_ids = exclude_ids,
    )

    if not use_validated_ids:
        return ids

    ids_file = (Path(validated_ids_path) if validated_ids_path is not None
                else Path(__file__).resolve().parent / "usable_patient_ids.txt")

    if not ids_file.exists():
        log.warning(f"  Usable IDs file not found: {ids_file} — using reviewed cohort only")
        return ids

    validated_ids = set()
    with open(ids_file, "r", encoding="utf-8") as f:
        for line in f:
            pid = line.strip().lstrip("sS")
            if pid:
                validated_ids.add(pid.zfill(4))

    filtered = [pid for pid in ids if pid in validated_ids]
    log.info(f"  Applied validated ID filter from {ids_file.name}: "
             f"{len(ids)} -> {len(filtered)}")
    return filtered


def print_cohort_summary(data_dir: Path) -> None:
    """
    Print a quick summary of all three reviewed CSV files — total counts,
    gender split, voxel count stats.  Useful to run before building the atlas.
    """
    reviewed_dir = Path(data_dir) / "Dataset_Reviewed"
    print(f"\n{'='*55}")
    print(f"  Dataset_Reviewed summary — {reviewed_dir}")
    print(f"{'='*55}")

    for label, filename in _DEFAULT_FILES.items():
        csv_path = reviewed_dir / filename
        if not csv_path.exists():
            print(f"  {label:8s}: not found ({filename})")
            continue

        ids     = load_patient_ids(csv_path)
        male    = load_patient_ids(csv_path, gender="M")
        female  = load_patient_ids(csv_path, gender="F")

        # Voxel stats
        voxels = []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                v = row.get("voxel_count", "").strip()
                if v:
                    try:
                        voxels.append(int(v))
                    except ValueError:
                        pass

        if voxels:
            import numpy as np
            v = np.array(voxels)
            vox_str = (f"voxels  mean={v.mean():,.0f}  "
                       f"min={v.min():,}  max={v.max():,}")
        else:
            vox_str = "voxel counts not available"

        print(f"\n  {label.upper()} ({filename})")
        print(f"    Total patients : {len(ids)}")
        print(f"    Male           : {len(male)}")
        print(f"    Female         : {len(female)}")
        print(f"    {vox_str}")

    print()


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    DATA_DIR = Path("Data")
    print_cohort_summary(DATA_DIR)

    # Example: get all reviewed IDs
    all_ids = load_cohort(DATA_DIR, cohort="all")
    print(f"All usable IDs ({len(all_ids)}): {all_ids[:10]} ...")

    # Example: female only, at least 100k voxels
    female_ids = load_cohort(DATA_DIR, cohort="all", gender="F", min_voxels=100_000)
    print(f"Female ≥100k voxels ({len(female_ids)}): {female_ids[:10]} ...")
