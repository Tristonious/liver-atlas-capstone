# AI Use Disclosure — LEGACY FILE (not used in current pipeline)
#   Student estimate: 45% student-designed, 55% AI-assisted implementation
#   Claude assisted with: TPS kernel matrix formulation, linear system assembly, batch processing
#   See: "Documentation/AI Use Disclosure.md" for full details

"""
stages/tps.py — Stage 4: Fit a Thin Plate Spline transformation.

Thin Plate Splines (TPS) are a classical method for non-rigid registration.
Given N pairs of corresponding control points, TPS finds a smooth mapping
that exactly interpolates them (when alpha=0) or approximates them with
controlled smoothness (alpha > 0).

The 3D kernel used here is U(r) = r (biharmonic), which minimises a bending
energy integral over R^3.

Mathematical formulation
------------------------
Given source points P ∈ R^{N×3} and targets Y ∈ R^{N×3}:

  L * [w; a] = [Y; 0]

where L is the (N+4) × (N+4) matrix:

  L = [ K + αI    P_h ]
      [ P_h^T      0  ]

K_ij = U(||p_i - p_j||),  P_h = [1 | P],  α = regularization weight

The solution gives weights w (N×3) and affine coefficients a (4×3).
To map a new point x:  f(x) = Σ w_i * U(||x - p_i||) + a_0 + a_1*x + a_2*y + a_3*z
"""

import logging

import numpy as np

log = logging.getLogger(__name__)


def _tps_kernel(r: np.ndarray) -> np.ndarray:
    """
    3D TPS kernel: U(r) = r  (biharmonic in 3D).

    Zero distances are left as zero (the kernel is zero at zero distance).
    """
    return r  # already 0 where r == 0


def _compute_kernel_matrix(points: np.ndarray) -> np.ndarray:
    """Compute the N×N TPS kernel matrix K for a set of N control points."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    r    = np.sqrt((diff ** 2).sum(axis=2))
    return _tps_kernel(r)


def fit_tps(src_landmarks: np.ndarray,
            ref_landmarks: np.ndarray,
            alpha: float = 0.05) -> dict:
    """
    Solve for TPS coefficients mapping src_landmarks → ref_landmarks.

    Args:
        src_landmarks: (N, 3) source control points (pre-aligned, ref voxel space)
        ref_landmarks: (N, 3) reference control points (ref voxel space)
        alpha:         Regularization weight. Larger = smoother but less accurate.
                       Typical range: 0.01 (tight) to 0.5 (very smooth)

    Returns:
        dict with keys:
            "w"              : (N, 3) radial basis weights
            "a"              : (4, 3) affine coefficients [intercept, dx, dy, dz]
            "src_landmarks"  : echoed back (needed for transform_points)
            "alpha"          : echoed back for logging
    """
    if len(src_landmarks) != len(ref_landmarks):
        raise ValueError(
            f"Landmark count mismatch: {len(src_landmarks)} src vs {len(ref_landmarks)} ref"
        )

    n = len(src_landmarks)
    log.info(f"  Fitting TPS with {n} control points, alpha={alpha}")

    K = _compute_kernel_matrix(src_landmarks)
    P = np.column_stack([np.ones(n), src_landmarks])   # shape (N, 4)

    L = np.block([
        [K + alpha * np.eye(n),  P              ],
        [P.T,                    np.zeros((4, 4))],
    ])
    Y = np.vstack([ref_landmarks, np.zeros((4, 3))])

    try:
        coeffs = np.linalg.solve(L, Y)
    except np.linalg.LinAlgError:
        log.warning("  Singular matrix — falling back to least-squares solve")
        coeffs = np.linalg.lstsq(L, Y, rcond=None)[0]

    w = coeffs[:n]      # radial basis weights
    a = coeffs[n:]      # affine coefficients

    log.info(f"  TPS fit complete  |  weight norm: {np.linalg.norm(w):.2f}")

    return {
        "w":             w,
        "a":             a,
        "src_landmarks": src_landmarks,
        "alpha":         alpha,
    }


def transform_points(points: np.ndarray,
                     coefficients: dict,
                     batch_size: int = 5000) -> np.ndarray:
    """
    Apply the fitted TPS transformation to a set of query points.

    Processes points in batches to limit peak memory usage. A batch of 5000
    points against N=750 control points requires ~11 MB, which is safe.

    Args:
        points:       (M, 3) query points to transform
        coefficients: Dict returned by fit_tps
        batch_size:   Points processed per batch (reduce if OOM)

    Returns:
        (M, 3) transformed points
    """
    w   = coefficients["w"]
    a   = coefficients["a"]
    src = coefficients["src_landmarks"]

    n_points  = len(points)
    n_batches = (n_points + batch_size - 1) // batch_size
    transformed = np.zeros_like(points, dtype=np.float64)

    for idx, i in enumerate(range(0, n_points, batch_size)):
        batch = points[i : i + batch_size]
        diff  = batch[:, np.newaxis, :] - src[np.newaxis, :, :]
        r     = np.sqrt((diff ** 2).sum(axis=2))
        K_b   = _tps_kernel(r)
        P_b   = np.column_stack([np.ones(len(batch)), batch])
        transformed[i : i + batch_size] = K_b @ w + P_b @ a

        if (idx + 1) % 10 == 0 or (idx + 1) == n_batches:
            log.info(f"    Batch {idx+1}/{n_batches}")

    return transformed