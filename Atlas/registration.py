# Atlas/registration.py
# Tristan Jones — Spring 2026 Capstone
#
# AI Use Disclosure
#   Student estimate: 55% student-designed, 45% AI-assisted implementation
#   Claude assisted with: npz save/load, forward_warp_mask, apply_rigid_to_volume, compute_warp_extents
#   See: "Documentation/AI Use Disclosure.md" for full details
#
# Thin wrapper around Registration.stages.* that the Atlas package uses.
#
# The Atlas only needs:
#   1. load_patient  — load liver + portal + hepatic + segments
#   2. extract_landmarks — Couinaud + vascular landmark clusters
#   3. prealign — Procrustes rigid alignment in mm space
#   4. transform_segmentation — apply the rigid warp to a mask volume
#
# TPS (stages/tps.py) is intentionally NOT called here.
# Distance maps are computed in native patient space then rigidly warped,
# which is more accurate than warping the vessel masks first.
#
# Caching: each patient's rigid alignment result is saved as a .npz so
# it is computed only once regardless of how many Atlas files use it.

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates

from Registration.stages.load      import load_patient
from Registration.stages.landmarks import extract_landmarks
from Registration.stages.align     import prealign
from Registration.utils.Nifti      import voxels_to_mm, mm_to_voxels

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_path(cache_dir: Path, atlas_id: str, patient_id: str) -> Path:
    """Helper for cache path."""
    return cache_dir / f"rigid_{atlas_id}_{patient_id}.npz"


def save_alignment(alignment: dict, path: Path) -> None:
    """Persist a prealign() result dict to .npz."""
    np.savez(
        path,
        R           = alignment["R"],
        t_src       = alignment["t_src"],
        t_ref       = alignment["t_ref"],
        src_affine  = alignment["src_affine"],
        ref_affine  = alignment["ref_affine"],
        residuals_mm= alignment["residuals_mm"],
    )
    log.info(f"  Rigid alignment saved → {path}")


def load_alignment(path: Path) -> dict:
    """Load a saved prealign() result dict from .npz."""
    d = np.load(path)
    alignment = {k: d[k] for k in d.files}
    # residuals may be missing in old cache files — provide a default
    if "residuals_mm" not in alignment:
        alignment["residuals_mm"] = np.array([])
    log.info(f"  Rigid alignment loaded ← {path}")
    return alignment


# ---------------------------------------------------------------------------
# Apply rigid alignment to a volume (scalar field interpolation)
# ---------------------------------------------------------------------------

def apply_rigid_to_volume(vol: np.ndarray,
                           alignment: dict,
                           ref_shape: tuple,
                           order: int = 1) -> np.ndarray:
    """
    Warp a source volume into reference voxel space using the rigid alignment.

    Uses INVERSE mapping — for each reference voxel, finds where it came from
    in the source. Good for continuous scalar fields (distance maps).
    For binary masks going into the atlas use forward_warp_mask() instead,
    which preserves the full liver shape without clipping at grid boundaries.

    For binary masks (label volumes) use order=0 (nearest-neighbour).
    For continuous scalar fields (distance maps) use order=1 (trilinear).

    Args:
        vol       : Source volume (X, Y, Z).
        alignment : Dict from prealign() or load_alignment().
        ref_shape : Shape of the reference grid to warp into.
        order     : Interpolation order (0=NN, 1=linear).

    Returns:
        Warped volume with shape ref_shape.
    """
    R       = alignment["R"]
    t_src   = alignment["t_src"]
    t_ref   = alignment["t_ref"]
    src_aff = alignment["src_affine"]
    ref_aff = alignment["ref_affine"]

    # Build a grid of all reference voxel coordinates
    xi, yi, zi = np.meshgrid(
        np.arange(ref_shape[0]),
        np.arange(ref_shape[1]),
        np.arange(ref_shape[2]),
        indexing="ij",
    )
    ref_vox = np.stack([xi.ravel(), yi.ravel(), zi.ravel()], axis=1).astype(np.float32)

    # ref voxels → ref mm → undo rotation → src mm → src voxels
    ref_mm  = voxels_to_mm(ref_vox, ref_aff)
    src_mm  = (np.linalg.inv(R) @ (ref_mm - t_ref).T).T + t_src
    src_vox = mm_to_voxels(src_mm, src_aff)   # float coords in source grid

    coords = [src_vox[:, i].reshape(ref_shape) for i in range(3)]
    warped = map_coordinates(vol, coords, order=order, mode="constant", cval=0.0)
    return warped.astype(vol.dtype)


def compute_warp_extents(src_vol: np.ndarray,
                          alignment: dict) -> dict:
    """
    Compute the min/max voxel coordinates that this patient's liver would
    occupy in reference space after forward warping — WITHOUT actually
    building the output volume.

    Used in the first pass of the two-pass atlas build to determine the
    global bounding box that fits all patients.

    Returns:
        dict with keys "min" and "max" — each a list [x, y, z]
    """
    R       = alignment["R"]
    t_src   = alignment["t_src"]
    t_ref   = alignment["t_ref"]
    src_aff = alignment["src_affine"]
    ref_aff = alignment["ref_affine"]

    src_voxels = np.argwhere(src_vol > 0).astype(np.float32)
    if len(src_voxels) == 0:
        return {"min": [0, 0, 0], "max": [0, 0, 0]}

    src_mm      = voxels_to_mm(src_voxels, src_aff)
    src_mm_c    = src_mm - t_src
    ref_mm      = (R @ src_mm_c.T).T + t_ref
    ref_vox     = mm_to_voxels(ref_mm, ref_aff)
    ref_vox_int = np.round(ref_vox).astype(np.int32)

    return {
        "min": ref_vox_int.min(axis=0).tolist(),
        "max": ref_vox_int.max(axis=0).tolist(),
    }


def forward_warp_mask(src_vol: np.ndarray,
                       alignment: dict,
                       ref_shape: tuple,
                       global_offset: np.ndarray = None) -> np.ndarray:
    """
    Warp a binary source mask into reference voxel space using FORWARD mapping.

    Unlike apply_rigid_to_volume (inverse mapping), this moves each source
    voxel forward to its new location. No voxels are dropped — negative
    coordinates are shifted by global_offset so everything fits in the grid.

    Args:
        src_vol       : Source binary mask (X, Y, Z).
        alignment     : Dict from prealign() or load_alignment().
        ref_shape     : Minimum output shape.
        global_offset : (3,) int array — shift applied to all voxel coords
                        so negative coords become positive. Computed from
                        the global bounding box across all patients.
                        If None, negative voxels are dropped (old behaviour).

    Returns:
        Warped binary mask in the shared global grid.
    """
    R       = alignment["R"]
    t_src   = alignment["t_src"]
    t_ref   = alignment["t_ref"]
    src_aff = alignment["src_affine"]
    ref_aff = alignment["ref_affine"]

    src_voxels = np.argwhere(src_vol > 0).astype(np.float32)
    if len(src_voxels) == 0:
        return np.zeros(ref_shape, dtype=src_vol.dtype)

    # Forward: src voxels → src mm → rotate → ref mm → ref voxels
    src_mm      = voxels_to_mm(src_voxels, src_aff)
    src_mm_c    = src_mm - t_src
    ref_mm      = (R @ src_mm_c.T).T + t_ref
    ref_vox     = mm_to_voxels(ref_mm, ref_aff)
    ref_vox_int = np.round(ref_vox).astype(np.int32)

    if global_offset is not None:
        # Shift all coordinates so nothing is negative
        ref_vox_int = ref_vox_int - global_offset
        src_voxels_v = src_voxels.astype(np.int32)
        ref_vox_v    = ref_vox_int
    else:
        # Old behaviour — drop negative voxels
        valid        = np.all(ref_vox_int >= 0, axis=1)
        src_voxels_v = src_voxels[valid].astype(np.int32)
        ref_vox_v    = ref_vox_int[valid]

    # Output grid — at least ref_shape, grows to fit all warped voxels
    out_shape = tuple(
        max(int(ref_shape[i]), int(ref_vox_v[:, i].max()) + 1)
        for i in range(3)
    )

    out = np.zeros(out_shape, dtype=src_vol.dtype)
    out[ref_vox_v[:, 0], ref_vox_v[:, 1], ref_vox_v[:, 2]] = \
        src_vol[src_voxels_v[:, 0], src_voxels_v[:, 1], src_voxels_v[:, 2]]

    log.info(f"  Forward warp: {len(src_voxels):,} → {int((out > 0).sum()):,} voxels  "
             f"output shape={out_shape}")
    return out

# ---------------------------------------------------------------------------
# High-level: align one patient into atlas space (with caching)
# ---------------------------------------------------------------------------

def align_patient(patient_id: str,
                   atlas_id: str,
                   data_dir: Path,
                   cache_dir: Path,
                   median_volume: float = None,
                   patient_volume: float = None,
                   canonical_direction: np.ndarray = None) -> Optional[dict]:
    """
    Compute or load the rigid alignment for one patient → atlas.

    Rotation is derived from the patient's affine direction matrix relative
    to the canonical (mean population) orientation — no landmark correspondence
    needed, no sign ambiguity.

    Scale is normalized to the median population liver volume.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = _cache_path(cache_dir, atlas_id, patient_id)

    if cache.exists():
        log.info(f"  Cache hit — loading rigid alignment for {patient_id}")
        alignment = load_alignment(cache)
    else:
        log.info(f"  Cache miss — computing rigid alignment for {patient_id}")
        try:
            ref_data = load_patient(data_dir, atlas_id)
            src_data = load_patient(data_dir, patient_id)
        except FileNotFoundError as e:
            log.error(f"  Could not load patient data: {e}")
            return None

        ref_lm = extract_landmarks(ref_data)
        src_lm = extract_landmarks(src_data)

        if len(ref_lm) != len(src_lm):
            log.error(f"  Landmark count mismatch: {len(ref_lm)} ref vs {len(src_lm)} src")
            return None

        n_segs = len(ref_data.get("segs", {}))
        alignment = prealign(
            src_landmarks        = src_lm,
            ref_landmarks        = ref_lm,
            src_affine           = src_data["affine"],
            ref_affine           = ref_data["affine"],
            n_segment_landmarks  = n_segs,
            src_liver_vol        = src_data["liver"],
            ref_liver_vol        = ref_data["liver"],
        )
        save_alignment(alignment, cache)

    # Override rotation using affine direction matrices → canonical orientation
    # This replaces any landmark-based rotation with one derived purely from
    # the scanner geometry, which is consistent across all patients.
    if canonical_direction is not None:
        def _dir(affine):
            """Helper for dir."""
            rot   = affine[:3, :3]
            zooms = np.sqrt((rot**2).sum(axis=0))
            return rot / zooms

        src_dir  = _dir(alignment["src_affine"])
        R_affine = canonical_direction @ src_dir.T

        # Orthonormalize to ensure proper rotation matrix
        U, _, Vt = np.linalg.svd(R_affine)
        R_clean  = U @ Vt
        if np.linalg.det(R_clean) < 0:
            U[:, -1] *= -1
            R_clean = U @ Vt

        alignment = dict(alignment)   # don't mutate cached dict
        alignment["R"] = R_clean
        log.info(f"  Affine-based rotation applied (canonical orientation)")

    # Override scale with median-volume-based scale
    if median_volume is not None and patient_volume is not None and patient_volume > 0:
        volume_scale = float(np.cbrt(median_volume / patient_volume))
        log.info(f"  Volume scale: {volume_scale:.4f}  "
                 f"(patient={patient_volume:,.0f}  median={median_volume:,.0f})")
        alignment = dict(alignment)
        alignment["scale"] = volume_scale

    return alignment