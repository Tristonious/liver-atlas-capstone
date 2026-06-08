# AI Use Disclosure
#   Student estimate: 70% student-designed, 30% AI-assisted implementation
#   Claude assisted with: ball_around helper, hepatic vein decomposition, extremal tip extraction
#   See: "Documentation/AI Use Disclosure.md" for full details

"""
stages/landmarks.py — Stage 2: Extract anatomical landmark clusters.

Landmarks are extracted from four sources in priority order:
  1. Couinaud liver segments (most anatomically stable, used for pre-alignment)
  2. Portal vein branch points (right, left, superior)
  3. Hepatic vein branches + IVC confluence
  4. Liver geometry (centroid + 6 extremal tips)

For each anatomical location, a CLUSTER of nearby voxels is returned rather
than a single point. This gives TPS more signal per landmark and improves
robustness to small segmentation errors.

Landmark ordering is deterministic: index 0 is always Couinaud segment 1,
index 8 is always portal right branch, etc. Both reference and source scans
must produce the same count so TPS can form corresponding pairs.
"""

import logging

import numpy as np

log = logging.getLogger(__name__)

# How many voxels to sample per landmark cluster
_CLUSTER_CAP = 50
# Starting ball radius in voxels (auto-expands if not enough voxels found nearby)
_CLUSTER_RADIUS = 5


def _resolve_label(seg: np.ndarray, requested_label: float, name: str) -> float:
    """
    Find the closest non-zero label in seg to requested_label.
    Warns and falls back to the first available label if exact match not found.
    """
    unique = np.unique(seg)
    nonzero = unique[unique > 0]
    if len(nonzero) == 0:
        raise ValueError(f"No non-zero voxels in {name}")
    matches = np.isclose(nonzero, requested_label)
    if np.any(matches):
        return float(nonzero[np.where(matches)[0][0]])
    chosen = float(nonzero[0])
    log.warning(f"  Label {requested_label} not found in {name}; using {chosen}")
    return chosen


def _ball_cluster(center: np.ndarray, voxel_pool: np.ndarray,
                  radius: float, cap: int = _CLUSTER_CAP) -> np.ndarray:
    """
    Sample up to `cap` voxels within `radius` of `center` from `voxel_pool`.

    Radius is doubled up to 8x if fewer than 10 voxels are found nearby,
    ensuring every landmark has at least a minimal cluster.

    Args:
        center:     (3,) centroid to sample around
        voxel_pool: (N, 3) candidate voxels
        radius:     Initial search radius in voxels
        cap:        Exact output size (oversample with replacement if needed)

    Returns:
        (cap, 3) array of voxel coordinates
    """
    dists  = np.linalg.norm(voxel_pool - center, axis=1)
    nearby = voxel_pool[dists <= radius]
    r = radius
    while len(nearby) < 10 and r < radius * 8:
        r *= 1.5
        nearby = voxel_pool[dists <= r]
    if len(nearby) == 0:
        nearby = voxel_pool[:1]
    if r > radius:
        log.debug(f"    Cluster radius expanded to {r:.1f} voxels")
    idx = np.random.choice(len(nearby), cap, replace=len(nearby) < cap)
    return nearby[idx]


def extract_landmarks(patient_data: dict,
                      organ_label: float = 1.0,
                      portal_label: float = 1.0,
                      cluster_radius: float = _CLUSTER_RADIUS,
                      cluster_cap: int = _CLUSTER_CAP) -> np.ndarray:
    """
    Extract all anatomical landmark clusters for one patient.

    Args:
        patient_data:   Dict returned by stages/load.py::load_patient
        organ_label:    Voxel label for the liver in patient_data["liver"]
        portal_label:   Voxel label for the portal vein
        cluster_radius: Starting search radius per cluster (voxels)
        cluster_cap:    Points sampled per cluster

    Returns:
        np.ndarray of shape (N, 3) — all clusters stacked vertically.
        The first len(segs)*cluster_cap rows are always segment clusters,
        enabling downstream pre-alignment to use them by index.
    """
    liver_seg    = patient_data["liver"]
    portal_seg   = patient_data.get("portal_vein")
    hepatic_seg  = patient_data.get("hepatic_vein")
    segment_segs = patient_data.get("segs", {})

    actual_label  = _resolve_label(liver_seg, organ_label, "liver")
    liver_voxels  = np.argwhere(np.isclose(liver_seg, actual_label)).astype(np.float32)
    log.info(f"  Liver voxels: {len(liver_voxels):,}")

    all_clusters = []

    def add_landmark(center, name, pool=None):
        """Execute add landmark."""
        pool = liver_voxels if pool is None else pool
        cluster = _ball_cluster(center, pool, cluster_radius, cluster_cap)
        all_clusters.append(cluster)
        log.info(f"    {name}: {len(cluster)} pts")

    # Source 1: Couinaud segment centroids 1–8
    liver_centroid = liver_voxels.mean(axis=0)
    if segment_segs:
        log.info("  [Source 1] Couinaud segment centroids")
        for i in range(1, 9):
            arr = segment_segs.get(i)
            if arr is not None:
                seg_v = np.argwhere(arr > 0).astype(np.float32)
            else:
                seg_v = np.empty((0, 3), dtype=np.float32)
            if len(seg_v) >= 10:
                add_landmark(seg_v.mean(axis=0), f"Segment {i}", seg_v)
            else:
                log.warning(f"    Segment {i}: too few voxels ({len(seg_v)}), using liver centroid")
                add_landmark(liver_centroid, f"Segment {i} fallback")

    # Source 2: Portal vein branch points
    if portal_seg is not None:
        log.info("  [Source 2] Portal vein branch points")
        actual_p = _resolve_label(portal_seg, portal_label, "portal")
        pv = np.argwhere(np.isclose(portal_seg, actual_p)).astype(np.float32)
        if len(pv) >= 20:
            x_med = np.median(pv[:, 0])
            z_thr = np.percentile(pv[:, 2], 67)
            for name, subset in [
                ("PV right branch",   pv[pv[:, 0] >= x_med]),
                ("PV left branch",    pv[pv[:, 0] <  x_med]),
                ("PV superior entry", pv[pv[:, 2] >= z_thr]),
            ]:
                c = subset.mean(axis=0) if len(subset) > 0 else pv.mean(axis=0)
                add_landmark(c, name, pv)
        else:
            log.warning(f"  Portal vein: {len(pv)} voxels — skipping")

    # Source 3: Hepatic vein landmarks
    if hepatic_seg is not None:
        log.info("  [Source 3] Hepatic vein landmarks")
        hv = np.argwhere(hepatic_seg > 0).astype(np.float32)
        if len(hv) >= 20:
            x_lo = np.percentile(hv[:, 0], 33)
            x_hi = np.percentile(hv[:, 0], 67)
            z_top = np.percentile(hv[:, 2], 75)
            for name, subset in [
                ("HV right branch",   hv[hv[:, 0] >  x_hi]),
                ("HV middle branch",  hv[(hv[:, 0] >= x_lo) & (hv[:, 0] <= x_hi)]),
                ("HV left branch",    hv[hv[:, 0] <  x_lo]),
                ("HV IVC confluence", hv[hv[:, 2] >= z_top]),
            ]:
                c = subset.mean(axis=0) if len(subset) > 0 else hv.mean(axis=0)
                add_landmark(c, name, hv)
        else:
            log.warning(f"  Hepatic veins: {len(hv)} voxels — skipping")

    # Source 4: Liver geometry tips
    log.info("  [Source 4] Liver geometry tips")
    add_landmark(liver_centroid, "Liver centroid")
    for name, tip in [
        ("Superior tip (max Z)", liver_voxels[liver_voxels[:, 2].argmax()]),
        ("Inferior tip (min Z)", liver_voxels[liver_voxels[:, 2].argmin()]),
        ("Right lobe   (max X)", liver_voxels[liver_voxels[:, 0].argmax()]),
        ("Left lobe    (min X)", liver_voxels[liver_voxels[:, 0].argmin()]),
        ("Anterior tip (max Y)", liver_voxels[liver_voxels[:, 1].argmax()]),
        ("Posterior tip(min Y)", liver_voxels[liver_voxels[:, 1].argmin()]),
    ]:
        add_landmark(tip, name)

    landmarks = np.vstack(all_clusters)
    log.info(f"  Total landmark control points: {len(landmarks)}")
    return landmarks