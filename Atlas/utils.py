# Atlas/utils.py
# Tristan Jones — Spring 2026 Capstone
#
# AI Use Disclosure
#   Student estimate: 60% student-designed, 40% AI-assisted implementation
#   Claude assisted with: pipeline implementation of shared helper functions
#   See: "Documentation/AI Use Disclosure.md" for full details
#
# Shared geometry helpers and Plotly building blocks used by both
# liver_atlas.py and vascular_distance.py.
#
# Importing from Registration.utils.Nifti where possible so there is one
# canonical implementation of voxel ↔ mm conversion.

from __future__ import annotations

from typing import Optional

import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import sobel
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes

from Registration.utils.Nifti import voxels_to_mm, mm_to_voxels   # canonical helpers


# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

# this is just for plotting things it sets what the colors are very customizable and is really not necessary if you remove
# this it would just make them random 

PATIENT_COLORS = [
    "#e6194b",  # red
    "#4363d8",  # blue
    "#f58231",  # orange
    "#911eb4",  # purple
    "#42d4f4",  # cyan
    "#f032e6",  # magenta
    "#bfef45",  # lime
    "#fabed4",  # pink
]


# ---------------------------------------------------------------------------
# Array padding
# ---------------------------------------------------------------------------

# Needed to add this because essentially all of the patients have different bounding box sizes
# so in order to compare Organ A to Organ B we pad the array with zeors. This ensures that they are 
# all the same size 

# Idealy we track the size of every array in the pipeline and then figure out the max size and pad everything to that 
# rather than constantly just pad -> go next pad -> oh wait go back -> pad again. This ensures that 
#   1. Theres no chopping/data loss
#   2. all of the different voxel spaces for each patient/organ is all within the same bounding box/space

# I think actually implementing this could be maybe made more efficient like making sure we track on json or somehting 
# as we kind of gather data/create data so O(n) or it could maybe be made faster but I think the actual padding itself may 
# be forced to be O(n) regardless so may be hard to make it much faster than this. 
def pad_to_common(a: np.ndarray, b: np.ndarray):
    """Zero-pad two arrays so they share the same shape along every axis."""
    out = tuple(max(a.shape[i], b.shape[i]) for i in range(a.ndim)) # out is the larger of the two sizes 
    def _pad(arr):
        """Helper for pad."""
        return np.pad(arr, [(0, out[i] - arr.shape[i]) for i in range(arr.ndim)])
    return _pad(a), _pad(b) # both gauranteed to be the same shape could add a check to ensure if you wanted tho returns tuple

#same thing but to target shape as I said previously 
def pad_vol_to(vol: np.ndarray, shape: tuple) -> np.ndarray:
    """Execute pad vol to."""
    return np.pad(vol, [(0, shape[i] - vol.shape[i]) for i in range(vol.ndim)])


# ---------------------------------------------------------------------------
# Dice coefficient
# ---------------------------------------------------------------------------


# essentially just calculates the dice similarity coef
# makes sure that they are padded to the same space so there's no grid related error
def dice(seg_a: np.ndarray, seg_b: np.ndarray, label: int = 1) -> float:
    """
    Dice Similarity Coefficient between two segmentation volumes.
    Pads both to a common bounding box so grid-size differences don't matter.
    """
    a, b = pad_to_common(seg_a, seg_b)
    a = a == label
    b = b == label
    intersection = np.logical_and(a, b).sum()
    denom = a.sum() + b.sum()
    return float(2.0 * intersection / denom) if denom > 0 else 0.0
# Returns Dice corealtion which is essentially 2* intersection vs sum


# ---------------------------------------------------------------------------
# Point cloud extraction
# ---------------------------------------------------------------------------


# So here we're just essentially applying the sobel filter which essentially is a gradient operator
# used for edge detection so it finds where the intensity changes rapidly. In this fonction we essentially 
# set it to be true if the liver exists in that voxel and false if it doesnt exist in the voxel.

# this function is really just used for getting the surface of whatever volume it is given in this case largely we are doing the liver
# however this could also be used to for instance get information on the vasculature surface space for instance if it was beneficial. 
def extract_surface_mm(liver_vol: np.ndarray,
                        affine: np.ndarray,
                        downsample: int = 8) -> np.ndarray:
    """
    Sobel-edge surface voxels → mm world coords.
    Used by LiverAtlas for the common-basis diagnostic scatter plot.

    Returns (N, 3) float32 array in mm.
    """
    binary = (liver_vol > 0).astype(np.float32)
    edges  = sobel(binary)
    vox    = np.argwhere(edges > 0)[::downsample].astype(np.float32)
    return voxels_to_mm(vox, affine).astype(np.float32)




# this esentially just finds the coordinates of every nonzero liver voxel and stores them as [x,y,z] in 32 bit 
# normally this would be stored in 64 bit so this halves the memory needed probably not necessary for higher end computers
# Gets every voxel where the liver exists basically  
def extract_liver_voxels(vol: np.ndarray) -> np.ndarray:
    """
    All nonzero liver voxel indices — full resolution, no downsampling.
    Used by VascularDistanceCloud for accumulation and the slice browser.

    Returns (N, 3) int32 array.
    """
    return np.argwhere(vol > 0).astype(np.int32)


# This is notably only used for visualization purposes for the visualization of what these look like without crashing my computer
# but essentialy because the livers can be many multiple thousands of voxels this essentially downsamples the voxel space to 
# only take the nth voxel. Its attempting to downsample down to 50k and generally the models (atlas models) landed around 400k so 
# it should end up downsampling about 8 times meaning that instead of showing every voxel it shows every 8th voxel or about every 12mm or so
# again this is only for visualization
def downsample_voxels_to_mm(voxels: np.ndarray,
                              affine: np.ndarray,
                              target: int = 50_000) -> tuple[np.ndarray, int]:
    """
    Stride-downsample voxels to ~target points and convert to mm.
    Used for 3-D scatter visualization only — not for accumulation.

    Returns (mm_pts, stride).
    """
    n      = len(voxels)
    stride = max(1, n // target) # floor division so how many times does it fit into this. 
    subset = voxels[::stride].astype(np.float32) # essentially stride-th element of the whole array (probably 8)
    print(f"    Viz downsample: {n:,} → {len(subset):,} pts (stride={stride})")
    return voxels_to_mm(subset, affine).astype(np.float32), stride # return but in mm 

# this is very similar to the above function except instead of just returning the voxel indecies 
# it returns these in the mm space by using the nifti affine. 
def extract_vessel_mm(vol: np.ndarray,
                       affine: np.ndarray) -> np.ndarray:
    """
    All nonzero vessel voxels → mm coords.
    Returns (N, 3) float32. Returns empty array if mask is empty.
    """
    vox = np.argwhere(vol > 0).astype(np.float32)
    if len(vox) == 0:
        return np.empty((0, 3), dtype=np.float32)
    return voxels_to_mm(vox, affine).astype(np.float32)


# ---------------------------------------------------------------------------
# KNN distance
# ---------------------------------------------------------------------------

# Takes liver query points and vessel points (both in mm space), builds one
# KD-tree over vessel points, then queries nearest neighbors for every liver
# point in a vectorized call. Returns the mean distance to the k nearest
# vessel points for each query point (k defaults to 5).

def knn_mean_distance(query_pts: np.ndarray,
                       vessel_pts: np.ndarray,
                       k: int = 5) -> np.ndarray:
    """
    For every point in query_pts return the mean distance to the k nearest
    points in vessel_pts.

    Using k>1 makes the estimate robust to single misregistered vessel voxels.
    Returns NaN for all query points if vessel_pts is empty.

    Returns (N,) float32 array.
    """
    # this is more of a safety mechanism but it may also induce some error it skips if 
    # the patient had segmentation issues/errors so good to manually check and run 
    # validation 
    if len(vessel_pts) == 0:
        return np.full(len(query_pts), np.nan, dtype=np.float32)

    k_actual = min(k, len(vessel_pts)) # also a safety mechanism if k is increased beyond 500 
    tree     = cKDTree(vessel_pts)
    dists, _ = tree.query(query_pts, k=k_actual, workers=-1)

    if dists.ndim == 1:
        return dists.astype(np.float32)
    return dists.mean(axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Plotly mesh from density volume (marching cubes)
# ---------------------------------------------------------------------------



# the poitnt of this is to implement the marching cubes method to essentially take the voxels and turn them into a 
# hollow shape that we can see as the liver like we see in plot 2. 
# Really just for visualization purposes 
def density_to_mesh(vol: np.ndarray,
                     level: float,
                     color: str,
                     opacity: float,
                     name: str,
                     affine: np.ndarray) -> Optional[go.Mesh3d]:
    """
    Run marching cubes on a density volume and return a Plotly Mesh3d trace.
    Vertex coordinates are converted to mm using the atlas affine.

    Returns None if the volume max is below `level` or marching cubes fails.
    """
    # essentiall if the volume never reaches the set threshold (75% for instance) print that it cant/skip
    if vol.max() < level:
        print(f"  [skip] {name}: max={vol.max():.3f} < level={level}")
        return None


    # zooms  = np.sqrt((affine[:3, :3] ** 2).sum(axis=0)) not used anymore changed to something else

    # needs to pad to ensure that the bounding box is closed so pads 1 layer otherwise theres a hole
    padded = np.pad(vol, 1, mode="constant", constant_values=0)
    try:
        # finds all triangles on the isosurface at the given probability threshold level 
        verts, faces, _, _ = marching_cubes(padded, level=level) #returns vertexs and faces, drops normals and curvature
    except Exception as e:
        print(f"  [skip] {name}: marching cubes failed — {e}")
        return None

    verts -= 1       # undo padding offset
    # Apply full affine (rotation + scaling + translation) so the mesh sits
    # in the same mm coordinate space as the surface point clouds.
    # Previously only zooms (voxel spacing) was applied, which ignored the
    # affine origin and caused the mesh to float away from the point clouds.
    ones  = np.ones((len(verts), 1))
    verts = (affine @ np.hstack([verts, ones]).T).T[:, :3]

    # returns the terecies and faces where the faces are the triangels 
    return go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color=color, opacity=opacity, name=name, showlegend=True,
    )


# ---------------------------------------------------------------------------
# Plotly slider layout helper
# ---------------------------------------------------------------------------


# This helps to make the plotly slider that we see in figure 3 or output 3.
# this is pretty nice visualization 
# called by vascular distance and atlas
def make_slider_layout(n_frames: int,
                        mid: int,
                        prefix: str = "Axial slice z=") -> dict:
    """Return a Plotly sliders + updatemenus layout block for an axial browser."""
    return dict(
        sliders=[dict(
            active=mid, # start at middle slice
            currentvalue=dict(prefix=prefix, visible=True), # shows current axial slice 
            pad=dict(t=50), # add some top padding so it doesnt overlap the top
            steps=[dict(    #one tick per slice 
                method="animate",
                args=[[str(z)], dict(
                    mode="immediate",
                    frame=dict(duration=0, redraw=True), #makes it snap instantly 
                    transition=dict(duration=0),
                )],
                label=str(z),
            ) for z in range(n_frames)],
        )],

        # manages updating the menus
        updatemenus=[dict(
            type="buttons", showactive=False,
            y=0, x=0.5, xanchor="center", yanchor="top",
            buttons=[dict(
                label="▶ Play",
                method="animate",
                args=[None, dict(
                    frame=dict(duration=80, redraw=True), #80ms per frame 
                    fromcurrent=True, transition=dict(duration=0),
                )],
            )],
        )],
    )