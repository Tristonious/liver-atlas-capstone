# This file was made with Claude entirely based on the code from another testing file. Essentially heres what it does:
# the file loads the NifTi Files for the souce and refrence liver shapes as well as the new post transformation liver. 

#then it extacts the 3d surfaces using an algorithm that changes voxels into a triangle mesh
# this helps us to plot this and we use plotly as a means of actually making these 3d mesh plots

# those are the main "algorithmic" components this is maing just plotting code though and not transformation 
# the only transformation that happens here is to transform from voxel -> mesh for plotting.


#again though I didn't write this code claude did I just asked claude to make this code based off another file
# prompt: Take the code from Testing.py and copy it over to this new file but for testing the specific output of the 
# transformed 10-> 4 then it created this.



"""
Visualize the registration result:
  Left panel  — Reference liver (0004) vs Source liver (0010) before transform
  Right panel — Reference liver (0004) vs Transformed liver (0010 → 0004)

Opens an interactive 3D plot in your browser via plotly.
"""

from pathlib import Path
import io
import zipfile
import numpy as np
import nibabel as nib
import nibabel.filebasedimages
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from skimage.measure import marching_cubes

PROJECT_DIR = Path(__file__).resolve().parent.parent
SEG_ZIP     = PROJECT_DIR / "Data" / "segmentations.zip"

REF_ID  = "0004"
SRC_ID  = "0010"
# If you have a pre-warped transformed file saved somewhere, point to it here.
# If not, set to None and only the before-panel will render.
TRANSFORMED_PATH = None  # e.g. Path("...") / "transformed_0010_to_0004.nii.gz"

# Loads a NIfTI binary mask and converts it to a 3D triangle mesh using marching cubes.
# The padding ensures the mesh closes cleanly at the edges of the volume.

def _load_nifti_from_zip(zf: zipfile.ZipFile, patient_id: str, filename: str = "liver.nii.gz"):
    """Read a single NIfTI file for a patient straight out of the zip, no extraction needed."""
    import gzip
    entry = f"{patient_id}/{filename}"
    if entry not in zf.namelist():
        raise FileNotFoundError(f"[ERROR] {entry} not found in {SEG_ZIP}")
    raw_gz = zf.read(entry)
    # .nii.gz files are gzip-compressed - decompress before nibabel reads them
    raw_nii = gzip.decompress(raw_gz)
    fh = nib.FileHolder(fileobj=io.BytesIO(raw_nii))
    img = nib.Nifti1Image.from_file_map({"header": fh, "image": fh})
    return img


def load_surface_from_zip(zf: zipfile.ZipFile, patient_id: str):
    """Load liver mask from zip and return (verts, faces) mesh, or None if empty."""
    img = _load_nifti_from_zip(zf, patient_id)
    data = np.asarray(img.dataobj) > 0
    if not data.any():
        return None
    data_padded = np.pad(data, pad_width=1, mode="constant", constant_values=0)
    verts, faces, _, _ = marching_cubes(data_padded, level=0.5)
    verts -= 1
    verts *= np.array(img.header.get_zooms()[:3])
    return verts, faces


def load_surface_from_file(seg_path: Path):
    """Load liver mask from a loose .nii.gz file and return (verts, faces) mesh, or None."""
    img = nib.load(str(seg_path))
    data = np.asarray(img.dataobj) > 0
    if not data.any():
        return None
    data_padded = np.pad(data, pad_width=1, mode="constant", constant_values=0)
    verts, faces, _, _ = marching_cubes(data_padded, level=0.5)
    verts -= 1
    verts *= np.array(img.header.get_zooms()[:3])
    return verts, faces


# Wraps a mesh (verts + faces) into a Plotly Mesh3d trace with a given color and label.
def mesh(verts, faces, color, opacity, name):
    """Execute mesh."""
    return go.Mesh3d(
        x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        color=color, opacity=opacity, name=name, showlegend=True,
    )


# Imma be so for real I had making plots either way because the syntax of making plots is such a hassle so either 
# way this would have largely been  writen by AI because I just find this process annoying 

# Essentiallyt this loads the refrence and source and transformed masks builds the mesh for each of them and 
# then it makes a two panel 3d plot showing before/after registration

# Prompt: Make a 3d Plot that shows the before and after of the shape based on the code from testing.py I created
#testing.py as a way of looking at the shapes in 3d because I wasnt able to before lots of issue with 3d slicer

def main() -> None:
    """Execute main."""
    if not SEG_ZIP.exists():
        print(f"[ERROR] Could not find segmentations zip: {SEG_ZIP}")
        return

    print(f"Loading from {SEG_ZIP} ...")
    with zipfile.ZipFile(SEG_ZIP, "r") as zf:
        ref_result = load_surface_from_zip(zf, REF_ID)
        src_result = load_surface_from_zip(zf, SRC_ID)

    if ref_result is None:
        print(f"[ERROR] Liver mask for {REF_ID} is empty.")
        return
    if src_result is None:
        print(f"[ERROR] Liver mask for {SRC_ID} is empty.")
        return

    rv, rf = ref_result
    sv, sf = src_result
    print(f"  Reference  ({REF_ID}): {len(rf):,} triangles")
    print(f"  Source     ({SRC_ID}): {len(sf):,} triangles")

    xfm_result = None
    if TRANSFORMED_PATH and Path(TRANSFORMED_PATH).exists():
        xfm_result = load_surface_from_file(Path(TRANSFORMED_PATH))
        if xfm_result:
            print(f"  Transformed:           {len(xfm_result[1]):,} triangles")

    # Build figure - one or two panels depending on whether transformed exists
    n_cols = 2 if xfm_result else 1
    subplot_titles = [f"Before: Reference ({REF_ID}) vs Source ({SRC_ID})"]
    if xfm_result:
        subplot_titles.append(f"After: Reference ({REF_ID}) vs Transformed ({SRC_ID}->{REF_ID})")

    specs = [[{"type": "scene"}] * n_cols]
    fig = make_subplots(
        rows=1, cols=n_cols,
        specs=specs,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.02,
    )

    fig.add_trace(mesh(rv, rf, "#3cb44b", 0.45, f"Reference ({REF_ID})"), row=1, col=1)
    fig.add_trace(mesh(sv, sf, "#e6194b", 0.45, f"Source ({SRC_ID})"),    row=1, col=1)

    if xfm_result:
        xv, xf = xfm_result
        fig.add_trace(mesh(rv, rf, "#3cb44b", 0.45, f"Reference ({REF_ID})"), row=1, col=2)
        fig.add_trace(mesh(xv, xf, "#4363d8", 0.45, f"Transformed ({SRC_ID}->{REF_ID})"), row=1, col=2)

    scene_opts = dict(
        xaxis_title="x (mm)", yaxis_title="y (mm)", zaxis_title="z (mm)",
        aspectmode="data",
    )
    layout_kwargs = dict(
        title="Registration Visualization",
        scene=scene_opts,
        legend=dict(groupclick="toggleitem"),
        margin=dict(l=0, r=0, t=60, b=0),
    )
    if xfm_result:
        layout_kwargs["scene2"] = scene_opts
    fig.update_layout(**layout_kwargs)

    fig.show()


if __name__ == "__main__":
    main()
