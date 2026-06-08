# This was made by Claude I asked Claude if it was possible to create a way to automaticall parse through different files and then 
# give me the segmentations as a wrapper for total segmentator I was really initally just messing around with it wasnt intending on this
# to become a main staple of whats happening here. Prompt: can you help me to create a way to just grab these certains
# segmentations [list of ones in question] output them to files from data that I have currently? Something along these lines
# I made some alterations but definetly a lot made by Claude.
#
# AI Use Disclosure
#   Student estimate: 50% student-designed, 50% AI-assisted implementation
#   Claude assisted with: subprocess wrapper, zip I/O, temp dir management, error handling
#   See: "Documentation/AI Use Disclosure.md" for full details 
#
# Updated to process one patient at a time — extract CT from zip, run TS, write
# segmentations into segmentations.zip, delete the CT. Never keeps more than one
# CT on disk at a time. Safe to interrupt and resume.

"""
Run TotalSegmentator on every patient in the reviewed CSV and save all
segmentations into a single zip file:

  Data/segmentations.zip
    {id}/liver.nii.gz
    {id}/portal_vein.nii.gz
    {id}/liver_vessels.nii.gz
    {id}/liver_tumor.nii.gz
    {id}/liver_segment_{1-8}.nii.gz

One patient is processed at a time:
  1. Extract ct.nii.gz from Totalsegmentator_dataset_v201.zip into a temp dir
  2. Run TotalSegmentator (three subtasks: ROI subset, liver segments, liver vessels)
  3. Write segmentation outputs into segmentations.zip
  4. Delete the CT — only segmentations are kept

Aorta removed — not used anywhere downstream.
The 8 liver segments come from a separate model:
  https://link.springer.com/chapter/10.1007/978-3-030-32692-0_32
The hepatic vessels + tumor come from:
  https://arxiv.org/abs/1902.09063
"""

from __future__ import annotations

import csv
import importlib.util
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
import time


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Maps TotalSegmentator output filenames → output keys written to the zip.
# Aorta removed — not needed by atlas or registration pipeline.
# Note: If you run on CPU use --roi_subset to greatly improve runtime (from
# the TotalSegmentator github) — that's exactly what this dict drives.
_ROI_SUBSET_STRUCTURES = {
    "liver":                        "liver",
    "portal_vein_and_splenic_vein": "portal_vein",
}

# similarly this is not a part of the main TS model
# https://arxiv.org/abs/1902.09063
_LIVER_VESSELS_OUTPUTS = ["liver_vessels", "liver_tumor"]

# same things as above — segments come from a different model
# https://link.springer.com/chapter/10.1007/978-3-030-32692-0_32
NUM_SEGMENTS = 8


# ---------------------------------------------------------------------------
# Find TotalSegmentator executable
# ---------------------------------------------------------------------------

# I feel like this is relatively self explanatory but this just makes sure
# that TotalSegmentator is on the PATH then grabs it to call as an executable
def find_totalsegmentator_command(custom_path: str | None = None) -> list[str]:
    """Return a runnable TotalSegmentator command prefix."""
    if custom_path:
        return [custom_path]
    for candidate in ["TotalSegmentator", "totalsegmentator"]:
        found = shutil.which(candidate)
        if found:
            return [found]
    if importlib.util.find_spec("totalsegmentator.bin.TotalSegmentator") is not None:
        return [sys.executable, "-m", "totalsegmentator.bin.TotalSegmentator"]
    raise FileNotFoundError(
        "Could not find TotalSegmentator CLI on PATH and module fallback was unavailable. "
        "Install it with: python -m pip install TotalSegmentator"
    )


# ---------------------------------------------------------------------------
# Check which patients are already done in the output zip
# ---------------------------------------------------------------------------

# This checks the output zip so we can skip patients that are already done
# and safely resume if the script gets interrupted partway through
def get_completed_ids(out_zip: Path) -> set[str]:
    """
    Return set of patient IDs already fully segmented in the output zip.
    A patient is considered complete if their liver.nii.gz is present.
    """
    if not out_zip.exists():
        return set()
    completed = set()
    with zipfile.ZipFile(out_zip, "r") as zf:
        for name in zf.namelist():
            parts = name.split("/")
            if len(parts) == 2 and parts[1] == "liver.nii.gz":
                completed.add(parts[0])
    return completed


# ---------------------------------------------------------------------------
# ROI subset task: liver + portal vein in one call
# ---------------------------------------------------------------------------

# this is the function that takes in the list of desired substructures and
# creates the command for TotalSegmentator.
# Note: If you run on CPU use the option --fast or --roi_subset to greatly
# improve runtime — from the TotalSegmentator github. This is running on CPU
# which is the point of adding --roi_subset here.
def _run_roi_subset(cmd: list[str], ct_path: Path, tmp_dir: Path) -> None:
    """Run TotalSegmentator with an explicit --roi_subset list."""
    full_cmd = [
        *cmd, "-i", str(ct_path), "-o", str(tmp_dir),
        "--roi_subset", *list(_ROI_SUBSET_STRUCTURES.keys()),
    ]
    print(f"\n  [1/3] ROI subset (liver + portal vein)...")
    subprocess.run(full_cmd, check=True)


# ---------------------------------------------------------------------------
# 8 Couinaud liver segment masks (liver_segments subtask)
# ---------------------------------------------------------------------------

# same things as above — the issue is that for these segments these come
# from a different model:
# https://link.springer.com/chapter/10.1007/978-3-030-32692-0_32
def _run_liver_segments(cmd: list[str], ct_path: Path, tmp_dir: Path) -> None:
    """Run TotalSegmentator with the liver_segments subtask."""
    full_cmd = [*cmd, "-i", str(ct_path), "-o", str(tmp_dir), "-ta", "liver_segments"]
    print(f"\n  [2/3] Liver segments (Couinaud 1-8)...")
    subprocess.run(full_cmd, check=True)


# ---------------------------------------------------------------------------
# liver_vessels subtask → liver_vessels.nii.gz + liver_tumor.nii.gz
# ---------------------------------------------------------------------------

# similarly this is not a part of the main TS model
# https://arxiv.org/abs/1902.09063
# comes from the work here — essentially same as other functions though
def _run_liver_vessels(cmd: list[str], ct_path: Path, tmp_dir: Path) -> None:
    """Run TotalSegmentator with the liver_vessels subtask."""
    full_cmd = [*cmd, "-i", str(ct_path), "-o", str(tmp_dir), "-ta", "liver_vessels"]
    print(f"\n  [3/3] Liver vessels + tumor...")
    subprocess.run(full_cmd, check=True)


# ---------------------------------------------------------------------------
# Collect output files from a temp directory
# ---------------------------------------------------------------------------

# this command essentially takes all the outputs from the three TS calls and
# combines them into a dictionary mapping zip filename → local path,
# which is then used to write everything into the output zip
def _collect_outputs(tmp_dir: Path) -> dict[str, Path]:
    """
    Return {zip_filename: local_path} for all segmentation files in tmp_dir.
    Handles both .nii.gz and .nii extensions.
    """
    found = {}

    for ts_name, key in _ROI_SUBSET_STRUCTURES.items():
        for suffix in [".nii.gz", ".nii"]:
            p = tmp_dir / f"{ts_name}{suffix}"
            if p.exists():
                found[f"{key}.nii.gz"] = p
                break

    for name in _LIVER_VESSELS_OUTPUTS:
        for suffix in [".nii.gz", ".nii"]:
            p = tmp_dir / f"{name}{suffix}"
            if p.exists():
                found[f"{name}.nii.gz"] = p
                break

    for i in range(1, NUM_SEGMENTS + 1):
        for suffix in [".nii.gz", ".nii"]:
            p = tmp_dir / f"liver_segment_{i}{suffix}"
            if p.exists():
                found[f"liver_segment_{i}.nii.gz"] = p
                break

    return found


# ---------------------------------------------------------------------------
# Per-case orchestration
# ---------------------------------------------------------------------------

# this ties all of the above together — extracts one CT, runs all three TS
# tasks, writes results to the output zip, then deletes the CT.
# The CT is always deleted in the finally block even if something fails.
def process_one_patient(
    patient_id: str,
    ct_zip: Path,
    out_zip: Path,
    cmd: list[str],
    work_dir: Path,
) -> bool:
    """
    Extract CT → run TotalSegmentator → write to output zip → delete CT.

    Args:
        patient_id : Zero-padded 4-digit string e.g. "0004"
        ct_zip     : Source Totalsegmentator_dataset_v201.zip
        out_zip    : Output segmentations.zip (appended to if it exists)
        cmd        : TotalSegmentator command prefix
        work_dir   : Temp working directory (one per patient, cleaned after)

    Returns:
        True on success, False on failure.
    """
    ct_entry = f"s{patient_id}/ct.nii.gz"

    with zipfile.ZipFile(ct_zip, "r") as zf:
        if ct_entry not in set(zf.namelist()):
            print(f"  WARNING: {ct_entry} not found in dataset zip — skipping")
            return False

    # Extract CT to temp dir
    ct_path = work_dir / f"ct{patient_id}.nii.gz"
    print(f"  Extracting CT...")
    with zipfile.ZipFile(ct_zip, "r") as zf:
        with zf.open(ct_entry) as src, open(ct_path, "wb") as dst:
            shutil.copyfileobj(src, dst)

    try:
        roi_dir = work_dir / "roi"
        seg_dir = work_dir / "segments"
        ves_dir = work_dir / "vessels"
        roi_dir.mkdir()
        seg_dir.mkdir()
        ves_dir.mkdir()

        _run_roi_subset(cmd, ct_path, roi_dir)
        _run_liver_segments(cmd, ct_path, seg_dir)
        _run_liver_vessels(cmd, ct_path, ves_dir)

        # Collect all outputs from all three temp dirs
        all_outputs = {}
        for d in [roi_dir, seg_dir, ves_dir]:
            all_outputs.update(_collect_outputs(d))

        if not all_outputs:
            print(f"  ERROR: No output files found for {patient_id}")
            return False

        # Write segmentations into output zip as {patient_id}/{filename}
        mode = "a" if out_zip.exists() else "w"
        with zipfile.ZipFile(out_zip, mode,
                             compression=zipfile.ZIP_DEFLATED) as zf:
            for zip_name, local_path in all_outputs.items():
                arcname = f"{patient_id}/{zip_name}"
                zf.write(local_path, arcname)
                print(f"  → {arcname}")

        print(f"  Patient {patient_id} complete ({len(all_outputs)} files)")
        return True

    finally:
        # Always delete the CT regardless of success or failure —
        # we only want to keep the segmentations, not the raw CTs
        if ct_path.exists():
            ct_path.unlink()
            print(f"  Deleted CT: {ct_path.name}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# This reads the reviewed CSV, checks what's already done, and processes
# the rest one patient at a time.
if __name__ == "__main__":
    import argparse as _argparse
    _ap = _argparse.ArgumentParser(description="Batch TotalSegmentator pipeline")
    _ap.add_argument("--gpu", action="store_true",
                     help="Run TotalSegmentator on GPU (passes --device gpu to TS)")
    _ap.add_argument("--fast", action="store_true",
                     help="Run TotalSegmentator in fast/low-res mode (CPU speedup)")
    _cli = _ap.parse_args()

    _ROOT         = Path(__file__).resolve().parent.parent
    _DATA_DIR     = _ROOT / "Data"
    _CT_ZIP       = _DATA_DIR / "Totalsegmentator_dataset_v201.zip"
    _OUT_ZIP      = _DATA_DIR / "segmentations.zip"
    _REVIEWED_CSV = _DATA_DIR / "Dataset_Reviewed" / "reviewed_all.csv"

    # Read patient IDs from the reviewed CSV — same list the atlas uses
    with open(_REVIEWED_CSV) as f:
        _PATIENT_IDS = [
            row["patient_id"].lstrip("s").zfill(4)
            for row in csv.DictReader(f)
        ]

    print(f"=== TotalSegmentator Batch Pipeline ===")
    print(f"  Dataset zip : {_CT_ZIP}")
    print(f"  Output zip  : {_OUT_ZIP}")
    print(f"  Patients    : {len(_PATIENT_IDS)}")

    try:
        _CMD = find_totalsegmentator_command()
        if _cli.gpu:
            _CMD += ["--device", "gpu"]
            print("  Mode        : GPU")
        elif _cli.fast:
            _CMD += ["--fast"]
            print("  Mode        : CPU fast (low-res)")
        else:
            print("  Mode        : CPU (default)")
        print(f"  Command     : {' '.join(_CMD)}")
    except FileNotFoundError as e:
        print(str(e))
        sys.exit(1)

    # Check which patients are already in the output zip so we can skip them
    completed = get_completed_ids(_OUT_ZIP)
    remaining = [pid for pid in _PATIENT_IDS if pid not in completed]
    print(f"  Already done: {len(completed)}")
    print(f"  To process  : {len(remaining)}")

    if not remaining:
        print("\nAll patients already processed.")
        sys.exit(0)

    successes = 0
    failures  = []

    for i, pid in enumerate(remaining, start=1):
        print(f"\n{'='*55}")
        print(f"  Patient {pid}  [{i}/{len(remaining)}]  "f"({successes} done, {len(failures)} failed so far)")
        print(f"{'='*55}")
        # Fresh temp dir per patient — automatically cleaned up after
        t0 = time.time()
        with tempfile.TemporaryDirectory() as tmp:
            ok = process_one_patient(
                patient_id = pid,
                ct_zip     = _CT_ZIP,
                out_zip    = _OUT_ZIP,
                cmd        = _CMD,
                work_dir   = Path(tmp),
            )
        elapsed = time.time() - t0

        if ok:
            successes += 1
            print(f"  ✓ Patient {pid} complete in {elapsed/60:.1f} min")
        else:
            failures.append(pid)
            print(f"  ✗ Patient {pid} failed after {elapsed/60:.1f} min")

    # will be changing a lot of this stuff in the coming weeks just the
    # structure of the pipeline and what not
    print(f"\n{'='*55}")
    print(f"  DONE")
    print(f"  Successful : {successes}")
    print(f"  Failed     : {len(failures)}")
    if failures:
        print(f"  Failed IDs : {failures}")
    print(f"  Output zip : {_OUT_ZIP}")
    print(f"{'='*55}")

    sys.exit(1 if failures else 0)