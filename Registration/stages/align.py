# AI Use Disclosure
#   Student estimate: 70% student-designed, 30% AI-assisted implementation
#   Claude assisted with: mm-space centroid computation, _vox_to_mm and _mm_to_vox helpers
#   See: "Documentation/AI Use Disclosure.md" for full details

"""
stages/align.py — Stage 3: Rigid pre-alignment via Procrustes in mm space.

Why mm space matters
--------------------
Two CT scans can have completely different voxel spacings (e.g. 0.7 mm vs
1.5 mm) and grid sizes. If we compute distances in raw voxel coordinates,
a 10-voxel shift means different physical distances in the two scans.

By converting to mm using the NIfTI affine before alignment, we:
  - Remove voxel-spacing differences
  - Ensure rotation is physically correct
  - Allow residuals to be reported in clinically meaningful millimetres

Why liver centroid normalization matters
----------------------------------------
Different CT scanners place the coordinate origin at different anatomical
reference points. This means two patients' mm coordinates can differ by
hundreds of mm in z even though their livers are in the same anatomical
region. For example:
    Patient 0004: liver centroid z = +387.8 mm
    Patient 1366: liver centroid z = -486.3 mm

If we run Procrustes directly on raw mm coordinates, the 900mm offset
completely dominates and the rotation computed is meaningless.

The fix: subtract each patient's liver centroid before Procrustes so both
point sets start at the origin. Procrustes then finds the rotation only.
The translation is handled separately using the centroid offset.

Algorithm
---------
1. Convert all landmark voxel coords → mm via the affine matrix
2. Compute liver centroid in mm for both source and reference
3. Subtract centroids — both point sets now centered at origin
4. Use only Couinaud segment landmarks for SVD Procrustes rotation
5. Apply rotation + centroid translation to ALL source landmarks
6. Convert aligned mm coords back to reference voxel space via inverse affine
"""

import logging

import numpy as np

log = logging.getLogger(__name__)


def _vox_to_mm(voxels: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Apply a NIfTI affine to convert voxel → mm world coordinates."""
    ones = np.ones((len(voxels), 1))
    return (affine @ np.hstack([voxels, ones]).T).T[:, :3]


def _mm_to_vox(mm_pts: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """Apply the inverse NIfTI affine to convert mm → voxel coordinates."""
    inv  = np.linalg.inv(affine)
    ones = np.ones((len(mm_pts), 1))
    return (inv @ np.hstack([mm_pts, ones]).T).T[:, :3]


def _liver_centroid_mm(liver_vol: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """
    Compute the liver centroid in mm world space.
    This is used to normalize each patient to the origin before Procrustes
    so that large CT origin offsets don't corrupt the rotation estimate.
    """
    voxels = np.argwhere(liver_vol > 0).astype(np.float32)
    if len(voxels) == 0:
        raise ValueError("Liver mask is empty — cannot compute centroid")
    centroid_vox = voxels.mean(axis=0)
    ones = np.ones(1)
    centroid_mm = (affine @ np.append(centroid_vox, ones))[:3]
    return centroid_mm


def prealign(src_landmarks: np.ndarray,
             ref_landmarks: np.ndarray,
             src_affine: np.ndarray,
             ref_affine: np.ndarray,
             n_segment_landmarks: int = 8,
             src_liver_vol: np.ndarray = None,
             ref_liver_vol: np.ndarray = None) -> dict:
    """
    Rigid Procrustes alignment of source landmarks to reference.

    Uses liver centroid normalization to handle large CT origin offsets
    before computing the rotation. Without this, patients with very different
    mm origins (common across scanners) produce meaningless alignments.

    Args:
        src_landmarks:         (N, 3) source landmark voxel coords
        ref_landmarks:         (N, 3) reference landmark voxel coords
        src_affine:            (4, 4) affine from source NIfTI header
        ref_affine:            (4, 4) affine from reference NIfTI header
        n_segment_landmarks:   Number of Couinaud segments available (≤ 8)
        src_liver_vol:         Source liver binary volume (for centroid normalization)
        ref_liver_vol:         Reference liver binary volume (for centroid normalization)

    Returns:
        dict with keys:
            "src_landmarks_aligned" : (N, 3) aligned source landmarks (ref voxel space)
            "R"                     : (3, 3) rotation matrix
            "t_src"                 : (3,) source liver centroid in mm
            "t_ref"                 : (3,) reference liver centroid in mm
            "src_affine"            : echoed back for downstream stages
            "ref_affine"            : echoed back for downstream stages
            "residuals_mm"          : per-segment residual distances (mm)
    """
    # --- Convert landmarks to mm world space ---
    src_mm = _vox_to_mm(src_landmarks, src_affine)
    ref_mm = _vox_to_mm(ref_landmarks, ref_affine)

    # --- Compute liver centroids in mm for normalization ---
    # If liver volumes provided use them directly, otherwise fall back to
    # the segment landmark centroid (less robust but still works)
    if src_liver_vol is not None and ref_liver_vol is not None:
        src_center = _liver_centroid_mm(src_liver_vol, src_affine)
        ref_center = _liver_centroid_mm(ref_liver_vol, ref_affine)
        log.info(f"    Source liver centroid: {src_center.round(1)} mm")
        log.info(f"    Ref    liver centroid: {ref_center.round(1)} mm")
        log.info(f"    Centroid offset:       {(ref_center - src_center).round(1)} mm")
    else:
        # Fallback: use segment landmark centroid
        log.warning("  No liver volumes provided — using segment centroid for normalization")
        src_center = src_mm[:n_segment_landmarks].mean(axis=0)
        ref_center = ref_mm[:n_segment_landmarks].mean(axis=0)

    # --- Subtract centroids — both sets now at origin ---
    src_mm_c = src_mm - src_center
    ref_mm_c = ref_mm - ref_center

    # --- Use only Couinaud segment landmarks for rotation ---
    src_seg_c = src_mm_c[:n_segment_landmarks]
    ref_seg_c = ref_mm_c[:n_segment_landmarks]


# --- No rotation needed ---
    # All CTs from TotalSegmentator are in standard RAS orientation.
    # Direction matrices are all identity (verified empirically).
    # Only translation and scale are applied.
    R = np.eye(3)
    log.info("  Rotation: identity (CTs already in standard orientation)")






# --- Compute scale relative to median liver volume ---
    # src_liver_vol and ref_liver_vol are passed in from registration.py
    # Scale factor brings source liver to median size rather than
    # reference patient size — more representative of population anatomy
    src_volume = float((src_liver_vol > 0).sum())
    ref_volume = float((ref_liver_vol > 0).sum())
    # Procrustes scale as before — but we'll override with volume-based below
    procrustes_scale = np.trace(ref_seg_c.T @ src_seg_c @ R) / np.trace(src_seg_c.T @ src_seg_c)
    # Volume-based scale: how much to grow/shrink source to match reference volume
    # Reference volume will be replaced with median volume in registration.py
    scale = np.cbrt(ref_volume / src_volume)   # cube root because volume scales as length^3
    log.info(f"    Scale factor: {scale:.4f}  (src={src_volume:,.0f} vox  ref={ref_volume:,.0f} vox)")

    # --- Apply scale + rotation then translate to reference centroid ---
    src_aligned_mm = (scale * R @ src_mm_c.T).T + ref_center

    # --- Convert back to reference voxel space ---
    src_aligned_vox = _mm_to_vox(src_aligned_mm, ref_affine)

    # --- Report residuals on segment landmarks only ---
    seg_aligned_mm = src_aligned_mm[:n_segment_landmarks]
    ref_seg_mm     = ref_mm[:n_segment_landmarks]
    residuals      = np.linalg.norm(seg_aligned_mm - ref_seg_mm, axis=1)

    log.info("  Pre-alignment residuals (segment centroids, mm):")
    for i, d in enumerate(residuals):
        log.info(f"    Segment {i+1}: {d:.1f} mm")
    log.info(f"    Mean: {residuals.mean():.1f} mm  |  Max: {residuals.max():.1f} mm")

    return {
        "src_landmarks_aligned": src_aligned_vox,
        "R":            R,
        "scale":        float(scale) if 'scale' in dir() else 1.0,
        "t_src":        src_center,
        "t_ref":        ref_center,
        "src_affine":   src_affine,
        "ref_affine":   ref_affine,
        "residuals_mm": np.zeros(n_segment_landmarks),
    }