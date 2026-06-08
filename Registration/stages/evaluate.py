# AI Use Disclosure — LEGACY FILE (not used in current pipeline)
#   Student estimate: 65% student-designed, 35% AI-assisted implementation
#   Claude assisted with: pre-alignment transform application, TPS + Dice evaluation integration
#   See: "Documentation/AI Use Disclosure.md" for full details

"""
stages/evaluate.py — Stage 5: Apply transformation + compute Dice coefficient.

This stage:
  1. Takes all non-zero voxels from the source liver segmentation
  2. Applies pre-alignment (Procrustes rotation in mm space)
  3. Applies TPS non-rigid transformation
  4. Writes the transformed segmentation as a NIfTI file
  5. Computes Dice coefficients before and after registration

Dice Similarity Coefficient (DSC)
----------------------------------
  DSC = 2 * |A ∩ B| / (|A| + |B|)

  DSC = 0.0 → no overlap
  DSC = 1.0 → perfect overlap

  Scores above 0.7 are generally considered acceptable for organ registration.
  Clinical-grade registration typically targets > 0.85.
"""

import json
import logging
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom as nd_zoom

from Registration.stages.tps import transform_points

log = logging.getLogger(__name__)


def _apply_prealignment(voxels: np.ndarray, alignment: dict) -> np.ndarray:
    """
    Apply the Procrustes pre-alignment (mm-space rotation) to voxel coordinates.

    Args:
        voxels:    (N, 3) source voxel coordinates
        alignment: Dict returned by stages/align.py::prealign

    Returns:
        (N, 3) coordinates in reference voxel space
    """
    R         = alignment["R"]
    t_src     = alignment["t_src"]
    t_ref     = alignment["t_ref"]
    src_aff   = alignment["src_affine"]
    ref_aff   = alignment["ref_affine"]

    # Voxel → mm (source space)
    ones    = np.ones((len(voxels), 1))
    mm_pts  = (src_aff @ np.hstack([voxels, ones]).T).T[:, :3]

    # Rotate in mm space
    mm_rot  = (R @ (mm_pts - t_src).T).T + t_ref

    # mm → voxel (reference space)
    inv_ref = np.linalg.inv(ref_aff)
    ones2   = np.ones((len(mm_rot), 1))
    vox_ref = (inv_ref @ np.hstack([mm_rot, ones2]).T).T[:, :3]
    return vox_ref


def _dice(seg_a: np.ndarray, seg_b: np.ndarray, label: int = 1) -> float:
    """
    Dice similarity coefficient between two binary masks.

    Pads both arrays to the same bounding box before comparison so that
    different output grid sizes don't cause errors.

    Args:
        seg_a, seg_b: 3D segmentation arrays
        label:        Voxel value to treat as foreground

    Returns:
        Dice score in [0, 1]
    """
    out_shape = tuple(max(a, b) for a, b in zip(seg_a.shape, seg_b.shape))

    def pad_to(arr, shape):
        """Execute pad to."""
        pad = [(0, shape[i] - arr.shape[i]) for i in range(arr.ndim)]
        return np.pad(arr, pad)

    a = pad_to(seg_a, out_shape) == label
    b = pad_to(seg_b, out_shape) == label

    intersection = np.logical_and(a, b).sum()
    denom        = a.sum() + b.sum()
    return float(2 * intersection / denom) if denom > 0 else 0.0


def transform_and_evaluate(src_data: dict,
                           ref_data: dict,
                           coefficients: dict,
                           alignment: dict,
                           out_dir: Path,
                           organ_label: int = 1) -> dict:
    """
    Apply the full registration to the source liver and evaluate quality.

    Args:
        src_data:     Dict from stages/load.py for the source patient
        ref_data:     Dict from stages/load.py for the reference patient
        coefficients: Dict from stages/tps.py::fit_tps
        alignment:    Dict from stages/align.py::prealign
        out_dir:      Directory to save the transformed NIfTI
        organ_label:  Foreground label in the liver masks

    Returns:
        dict with keys:
            "dice_before"         : Dice before registration
            "dice_after"          : Dice after registration
            "n_voxels_transformed": Number of non-zero voxels in output
            "output_path"         : Path to saved NIfTI
    """
    src_liver = src_data["liver"]
    ref_liver = ref_data["liver"]
    ref_affine = ref_data["affine"]
    ref_shape  = ref_liver.shape

    # --- Dice BEFORE registration (resample src to ref grid for fair comparison) ---
    zoom = tuple(r / s for r, s in zip(ref_shape, src_liver.shape))
    src_resampled = (
        nd_zoom(src_liver.astype(np.float32), zoom, order=0) > 0.5
    ).astype(src_liver.dtype)
    dice_before = _dice(ref_liver, src_resampled, label=organ_label)
    log.info(f"  Dice before registration: {dice_before:.4f}")

    # --- Gather source voxels to transform ---
    src_voxels = np.argwhere(src_liver > 0)
    if len(src_voxels) == 0:
        log.error("  Source liver mask is empty — cannot transform")
        return {"dice_before": dice_before, "dice_after": 0.0,
                "n_voxels_transformed": 0, "output_path": None}

    log.info(f"  Transforming {len(src_voxels):,} source voxels...")

    # --- Apply pre-alignment (mm-space Procrustes) ---
    voxels_aligned = _apply_prealignment(src_voxels.astype(np.float32), alignment)

    # --- Apply TPS non-rigid transform ---
    voxels_tps = transform_points(voxels_aligned, coefficients)

    # --- Round and filter to valid grid positions ---
    voxels_int = np.round(voxels_tps).astype(np.int32)
    valid_mask = np.all(voxels_int >= 0, axis=1)
    src_voxels_v = src_voxels[valid_mask]
    voxels_int_v = voxels_int[valid_mask]

    # Output grid at least as large as reference
    out_shape = tuple(
        max(int(ref_shape[i]), int(voxels_int_v[:, i].max()) + 1)
        for i in range(3)
    )

    # --- Build output volume ---
    transformed = np.zeros(out_shape, dtype=src_liver.dtype)
    for orig_idx, new_idx in zip(src_voxels_v, voxels_int_v):
        transformed[tuple(new_idx)] = src_liver[tuple(orig_idx)]

    n_out = int(np.sum(transformed > 0))
    log.info(f"  Retained {n_out:,} voxels after transform")

    # --- Dice AFTER registration ---
    dice_after = _dice(ref_liver, transformed, label=organ_label)
    log.info(f"  Dice after  registration: {dice_after:.4f}")
    log.info(f"  Improvement: {dice_after - dice_before:+.4f}")

    # --- Save transformed NIfTI ---
    out_path = out_dir / f"transformed_{src_data['patient_id']}_to_{ref_data['patient_id']}.nii.gz"
    nib.save(nib.Nifti1Image(transformed, ref_affine), str(out_path))
    log.info(f"  Saved: {out_path}")

    # --- Save metrics JSON ---
    metrics = {
        "reference_id":        ref_data["patient_id"],
        "source_id":           src_data["patient_id"],
        "dice_before":         round(dice_before, 6),
        "dice_after":          round(dice_after,  6),
        "improvement":         round(dice_after - dice_before, 6),
        "n_voxels_src":        int(len(src_voxels)),
        "n_voxels_transformed": n_out,
        "tps_alpha":           coefficients.get("alpha"),
    }
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"  Metrics saved: {metrics_path}")

    return metrics