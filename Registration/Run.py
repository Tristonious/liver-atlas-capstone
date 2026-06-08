"""
run.py — Single entry point for the CT-to-CT liver registration pipeline.

Usage:
    python run.py --ref 0004 --src 0010
    python run.py --ref 0004 --src 0010 --alpha 0.05 --data-dir Data
    python run.py --ref 0004 --src 0010 --skip-if-done   # skip stages with saved outputs

Pipeline stages (in order):
    1. load       — Load NIfTI segmentations for both patients
    2. landmarks  — Extract anatomical landmark clusters
    3. align      — Procrustes pre-alignment in mm world space
    4. tps        — Thin Plate Spline non-rigid registration
    5. evaluate   — Apply transform + compute Dice coefficient
"""

import argparse
import sys
import time
from pathlib import Path

from Registration.stages.load      import load_patient
from Registration.stages.landmarks import extract_landmarks
from Registration.stages.align     import prealign
from Registration.stages.tps       import fit_tps
from Registration.stages.evaluate  import transform_and_evaluate
from Registration.utils.Checkpoint import Checkpoint


def parse_args():
    """Execute parse args."""
    p = argparse.ArgumentParser(description="Liver CT-to-CT registration pipeline")
    p.add_argument("--ref",          required=True,  help="Reference patient ID, e.g. 0004")
    p.add_argument("--src",          required=True,  help="Source patient ID, e.g. 0010")
    p.add_argument("--data-dir",     default="Data", help="Root data directory (default: Data)")
    p.add_argument("--out-dir",      default="outputs", help="Where to save results")
    p.add_argument("--alpha",        type=float, default=0.05, help="TPS regularization (default: 0.05)")
    p.add_argument("--organ-label",  type=int,   default=1,    help="Voxel label for liver (default: 1)")
    p.add_argument("--skip-if-done", action="store_true",
                   help="Skip a stage if its output file already exists")
    return p.parse_args()


def run_stage(name, fn, checkpoint, skip_if_done, *args, **kwargs):
    """
    Run one pipeline stage with timing, logging, and optional checkpointing.

    Args:
        name:         Human-readable stage name for logging
        fn:           The stage function to call
        checkpoint:   Checkpoint object for save/load
        skip_if_done: If True and checkpoint exists, skip computation
        *args/**kwargs: Forwarded to fn

    Returns:
        Whatever fn returns (or the loaded checkpoint value)
    """
    print("=" * 55)
    print(f"STAGE: {name}")
    print("=" * 55)

    if skip_if_done and checkpoint.exists(name):
        print(f"  Checkpoint found - loading saved result for '{name}'")
        return checkpoint.load(name)

    t0 = time.time()
    result = fn(*args, **kwargs)
    elapsed = time.time() - t0

    checkpoint.save(name, result)
    print(f"  Done in {elapsed:.1f}s  - saved checkpoint")
    return result


def main():
    """Execute main."""
    args = parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir) / f"{args.src}_to_{args.ref}"
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = Checkpoint(out_dir)

    print("=" * 55)
    print("LIVER REGISTRATION PIPELINE")
    print(f"  Reference : {args.ref}")
    print(f"  Source    : {args.src}")
    print(f"  Data dir  : {data_dir}")
    print(f"  Output dir: {out_dir}")
    print(f"  Alpha     : {args.alpha}")
    print("=" * 55)

    # ------------------------------------------------------------------
    # Stage 1 — Load
    # ------------------------------------------------------------------
    ref_data = run_stage("load_ref", load_patient, checkpoint,
                         args.skip_if_done,
                         data_dir=data_dir, patient_id=args.ref)

    src_data = run_stage("load_src", load_patient, checkpoint,
                         args.skip_if_done,
                         data_dir=data_dir, patient_id=args.src)

    # ------------------------------------------------------------------
    # Stage 2 — Landmark extraction
    # ------------------------------------------------------------------
    ref_landmarks = run_stage("landmarks_ref", extract_landmarks, checkpoint,
                              args.skip_if_done,
                              patient_data=ref_data)

    src_landmarks = run_stage("landmarks_src", extract_landmarks, checkpoint,
                              args.skip_if_done,
                              patient_data=src_data)

    # ------------------------------------------------------------------
    # Stage 3 — Pre-alignment
    # ------------------------------------------------------------------
    alignment = run_stage("align", prealign, checkpoint,
                          args.skip_if_done,
                          src_landmarks=src_landmarks,
                          ref_landmarks=ref_landmarks,
                          src_affine=src_data["affine"],
                          ref_affine=ref_data["affine"],
                          n_segment_landmarks=len(ref_data.get("segs", {})))

    # ------------------------------------------------------------------
    # Stage 4 — TPS registration
    # ------------------------------------------------------------------
    coefficients = run_stage("tps", fit_tps, checkpoint,
                             args.skip_if_done,
                             src_landmarks=alignment["src_landmarks_aligned"],
                             ref_landmarks=ref_landmarks,
                             alpha=args.alpha)

    # ------------------------------------------------------------------
    # Stage 5 — Transform + evaluate
    # ------------------------------------------------------------------
    metrics = run_stage("evaluate", transform_and_evaluate, checkpoint,
                        args.skip_if_done,
                        src_data=src_data,
                        ref_data=ref_data,
                        coefficients=coefficients,
                        alignment=alignment,
                        out_dir=out_dir,
                        organ_label=args.organ_label)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 55)
    print("PIPELINE COMPLETE")
    print(f"  Dice before registration : {metrics['dice_before']:.4f}")
    print(f"  Dice after  registration : {metrics['dice_after']:.4f}")
    print(f"  Improvement              : {metrics['dice_after'] - metrics['dice_before']:+.4f}")
    print(f"  Output saved to          : {out_dir}")
    print("=" * 55)

    return 0


if __name__ == "__main__":
    sys.exit(main())