# Tristan Jones 
# Srping 2026 Capstone 

# This file is a collection of helper functions that assist most other files
# with loading or saving these nifti type files. Additionally this file helps 
# with calculating the mm to voxel transformation from the nifti afffine 
# which is created by the imagine machine itself. 


import logging
from pathlib import Path

import nibabel as nib
import numpy as np

log = logging.getLogger(__name__)


def load_nifti(path: Path) -> tuple:
    """
    Load a NIfTI file and return (data array, affine matrix).

    Args:
        path: Path to a .nii or .nii.gz file

    Returns:
        Tuple of (np.ndarray float32, 4x4 affine np.ndarray)

    Raises:
        FileNotFoundError: if the file does not exist
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NIfTI file not found: {path}")

    img     = nib.load(str(path))
    data    = img.get_fdata(dtype=np.float32)
    affine  = img.affine
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0)).round(2)

    log.info(f"    Loaded {path.name}  shape={data.shape}  spacing={spacing} mm")
    return data, affine


def save_nifti(data: np.ndarray, affine: np.ndarray, path: Path) -> None:
    """
    Save a numpy array as a NIfTI file.

    Args:
        data:   3D numpy array to save
        affine: 4x4 affine matrix (use the reference patient's affine
                so the output sits correctly in physical space)
        path:   Destination path, e.g. Path("outputs/transformed.nii.gz")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine), str(path))
    log.info(f"    Saved {path.name}  shape={data.shape}")


def get_spacing(affine: np.ndarray) -> np.ndarray:
    """
    Extract voxel spacing (mm) from a NIfTI affine matrix.

    Args:
        affine: 4x4 affine matrix from a NIfTI header

    Returns:
        (3,) array of voxel sizes in mm [x, y, z]
    """
    return np.sqrt((affine[:3, :3] ** 2).sum(axis=0))


def voxels_to_mm(voxels: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """
    Convert voxel coordinates to mm world coordinates using the affine.

    Args:
        voxels: (N, 3) array of voxel coordinates
        affine: 4x4 NIfTI affine matrix

    Returns:
        (N, 3) array of mm world coordinates
    """
    ones = np.ones((len(voxels), 1))
    return (affine @ np.hstack([voxels, ones]).T).T[:, :3]


def mm_to_voxels(mm_pts: np.ndarray, affine: np.ndarray) -> np.ndarray:
    """
    Convert mm world coordinates back to voxel coordinates.

    Args:
        mm_pts: (N, 3) array of mm world coordinates
        affine: 4x4 NIfTI affine matrix of the target grid

    Returns:
        (N, 3) array of voxel coordinates (float, not rounded)
    """
    inv  = np.linalg.inv(affine)
    ones = np.ones((len(mm_pts), 1))
    return (inv @ np.hstack([mm_pts, ones]).T).T[:, :3]