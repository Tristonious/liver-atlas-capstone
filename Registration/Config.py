"""
config.py — All tunable pipeline parameters in one place.

Change values here rather than hunting through individual stage files.
"""

# ── Data ────────────────────────────────────────────────────
DATA_DIR    = "Data"
OUTPUT_DIR  = "outputs"

# Voxel label value for the liver in TotalSegmentator masks
ORGAN_LABEL = 1

# Portal vein filename candidates (TotalSegmentator name changed across versions)
PORTAL_VEIN_CANDIDATES = [
    "portal_vein.nii.gz",
    "portal_vein_and_splenic_vein.nii.gz",
]

# ── Landmarks ───────────────────────────────────────────────
# Starting search radius around each anatomical centroid (voxels)
CLUSTER_RADIUS = 5

# Number of voxels sampled per landmark cluster
# Reduce to speed up TPS; increase for more control point density
CLUSTER_CAP = 50

# ── Pre-alignment ────────────────────────────────────────────
# Voxel label for the portal vein in portal_vein masks
PORTAL_LABEL = 1

# ── TPS Registration ─────────────────────────────────────────
# Regularization weight. Higher = smoother warp, less exact fit.
# Typical range: 0.01 (tight) to 0.5 (very smooth)
ALPHA = 0.05

# ── Transform (evaluate stage) ───────────────────────────────
# Points processed per batch during voxel transformation.
# Reduce if you get a MemoryError on large liver masks.
TRANSFORM_BATCH_SIZE = 5000