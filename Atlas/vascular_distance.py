# Atlas/vascular_distance.py
# Tristan Jones — Spring 2026 Capstone
#
# AI Use Disclosure
#   Student estimate: 55% student-designed, 45% AI-assisted implementation
#   Claude assisted with: VascularDistanceCloud class, NaN-safe accumulation, Plotly visualizations
#   See: "Documentation/AI Use Disclosure.md" for full details
#
# Distance-to-Vasculature Electron Cloud
#
# Key design choice:
#   Distances are computed in each patient's OWN native space first,
#   then the resulting distance map is rigidly warped into "atlas space".
#   This avoids introducing registration error into the distance values —
#   the distances are exact (computed before any warping), and only the
#   rigid alignment is needed to bring them into a common coordinate frame.
#
# Pipeline (per patient):
#   1. Load liver + portal vein + hepatic vessels in native space.
#   2. For every liver voxel, compute mean distance to the k nearest
#      vessel voxels using a KD-tree.  → native-space distance map.
#   3. Rigidly warp the distance map into atlas space (trilinear interp).
#   4. Accumulate warped distance maps across patients.
#   5. Divide by N → mean expected distance map.
#   6. Visualise as 3-D colored point cloud + axial slice browser.


from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
import sys
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Allow running this file directly: python Atlas/vascular_distance.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Registration.stages.load import load_patient
from Registration.utils.Nifti import voxels_to_mm

from Atlas.utils import (
    extract_liver_voxels,
    downsample_voxels_to_mm,
    extract_vessel_mm,
    knn_mean_distance,
    make_slider_layout,
)
from Atlas.registration import align_patient, apply_rigid_to_volume

log = logging.getLogger(__name__)


def _accum(total: np.ndarray, count: np.ndarray, values: np.ndarray) -> None:
    """Accumulate valid values into running sum and observation count.

    Args:
        total: Running sum array updated in place.
        count: Running valid-observation count array updated in place.
        values: New values to accumulate. NaN entries are ignored.
    """
    valid = ~np.isnan(values)
    total[valid] += values[valid]
    count[valid] += 1


class VascularDistanceCloud:
    """
    Per-voxel mean distance to vasculature across a patient population.

    Distances are computed in native patient space then rigidly warped into
    atlas space — more accurate than warping vessel masks first.

    Color convention: red = close to vessel, blue = far from vessel.

    Usage
    -----
    vdc = VascularDistanceCloud(atlas_id="0004", data_dir=Path("Data"))
    vdc.build(["0010", "0011", "0012", "0013"]) full set of desired subjects
    vdc.save(Path("outputs/atlas"))

    vdc.visualize()                    # 3-D colored point cloud
    vdc.visualize_all_modes()          # portal / hepatic / combined side by side
    vdc.visualize_distance_slices()    # axial slice browser from utils.py
    vdc.visualize_distance_histogram() # check distribution 
    """

    def __init__(self,
                 atlas_id: str,
                 data_dir: Path = Path("Data"),
                 cache_dir: Path = Path("outputs/reg_cache"),
                 atlas_dir: Path = Path("outputs/atlas_male"),
                 density_threshold: float = 0.5,
                 k_neighbors: int = 5,
                 cohort_label: Optional[str] = None):
        """Initialize vascular-distance atlas builder and visualization state.

        Args:
            atlas_id: Patient ID used as rigid-registration reference.
            data_dir: Root directory containing patient segmentations.
            cache_dir: Directory with cached registration artifacts and extents.
            atlas_dir: Output atlas directory containing density and saved arrays.
            density_threshold: Liver consensus threshold on atlas density map.
            k_neighbors: Number of nearest vessel voxels used for mean distance.
        """
        self.atlas_id          = atlas_id
        self.data_dir          = Path(data_dir)
        self.cache_dir         = Path(cache_dir)
        self.atlas_dir         = Path(atlas_dir)
        self.density_threshold = density_threshold
        self.k                 = k_neighbors
        self.cohort_label      = cohort_label

        # Full-resolution distance arrays (one value per liver voxel)
        # Used by slice browser and histogram
        self.all_voxel_idx      : Optional[np.ndarray] = None
        self.full_dist_portal   : Optional[np.ndarray] = None
        self.full_dist_hepatic  : Optional[np.ndarray] = None
        self.full_dist_combined : Optional[np.ndarray] = None

        # Downsampled (~50k points) for 3-D scatter visualization
        self.surface_pts_mm     : Optional[np.ndarray] = None
        self.mean_dist_portal   : Optional[np.ndarray] = None
        self.mean_dist_hepatic  : Optional[np.ndarray] = None
        self.mean_dist_combined : Optional[np.ndarray] = None

        self.n_patients   = 0
        self.atlas_affine : Optional[np.ndarray] = None
        self.atlas_shape  : Optional[tuple] = None

    def _cohort_subject_text(self) -> str:
        """Return cohort-aware subject count text for titles/logs."""
        if self.cohort_label:
            return f"{self.n_patients} {self.cohort_label} subjects"
        return f"{self.n_patients} subjects"

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, patient_ids: list[str]) -> None:
        """
        For each patient: compute native-space distance map, rigidly warp
        into atlas space, accumulate.

        Args:
            patient_ids : Source patient IDs, e.g. ["0010", "0011"].
        """
        log.info(f"\n{'='*60}")
        log.info(f"  Building VascularDistanceCloud")
        log.info(f"  Atlas reference : {self.atlas_id}")
        log.info(f"  k neighbors     : {self.k}")
        log.info(f"{'='*60}")

        import json

        ref_data   = load_patient(self.data_dir, self.atlas_id)
        ref_liver  = ref_data["liver"]
        ref_affine = ref_data["affine"]
        self.atlas_affine = ref_affine

        # Load extents metadata from liver_atlas.py so we use the exact same
        # global atlas grid (including negative-coordinate offset handling).
        extents_path = self.cache_dir / f"extents_{self.atlas_id}.json"
        all_extents = None
        canonical_direction = None
        median_volume = None
        global_offset = None

        if extents_path.exists():
            print(f"  Loading extents from {extents_path}")
            with open(extents_path, "r", encoding="utf-8") as f:
                all_extents = json.load(f)

            # Compute global grid shape from extents — must match liver_atlas.py
            all_mins = np.array([e["min"] for e in all_extents.values()])
            all_maxs = np.array([e["max"] for e in all_extents.values()])
            global_min    = all_mins.min(axis=0)
            global_offset = global_min.clip(max=0)
            global_shape  = tuple(
                int(all_maxs.max(axis=0)[i] - global_offset[i]) + 1 for i in range(3))
            self.atlas_shape = global_shape
            print(f"  Global atlas grid: shape={self.atlas_shape}, "
                  f"offset={global_offset.tolist()}")

            all_voxel_counts = [e["voxel_count"] for e in all_extents.values()
                                if "voxel_count" in e]
            if all_voxel_counts:
                median_volume = float(np.median(all_voxel_counts))

            all_directions = np.array([e["direction"] for e in all_extents.values()
                                       if "direction" in e])
            if len(all_directions) > 0:
                mean_dir = all_directions.mean(axis=0)
                U, _, Vt = np.linalg.svd(mean_dir)
                canonical_direction = U @ Vt
                if np.linalg.det(canonical_direction) < 0:
                    U[:, -1] *= -1
                    canonical_direction = U @ Vt
        else:
            print(f"  WARNING: Extents JSON not found at {extents_path}")
            print("  Falling back to density/reference shape only")

        # Load atlas density volume — use consensus liver region as sampling grid
        # instead of patient 0004's specific liver boundary
        atlas_density_path = self.atlas_dir / "atlas_liver_density.nii.gz"
        if atlas_density_path.exists():
            print(f"  Loading atlas density from {atlas_density_path}")
            density_img     = nib.load(str(atlas_density_path))
            atlas_density   = density_img.get_fdata(dtype=np.float32)
            # Use voxels where >50% of patients had liver — the consensus region
            consensus_liver = (atlas_density >= self.density_threshold).astype(np.uint8)
            ref_vox_idx     = extract_liver_voxels(consensus_liver)
            if self.atlas_shape is None:
                self.atlas_shape = consensus_liver.shape
            elif tuple(consensus_liver.shape) != tuple(self.atlas_shape):
                print(f"  WARNING: Density shape {consensus_liver.shape} does not match "
                      f"extents grid {self.atlas_shape}")
            print(f"  Atlas consensus liver: {len(ref_vox_idx):,} voxels "
                  f"(threshold={self.density_threshold})")
        else:
            print(f"  WARNING: Atlas density not found at {atlas_density_path}")
            print(f"  Falling back to reference patient liver — run liver_atlas.py first")
            ref_vox_idx = extract_liver_voxels(ref_liver)
            if self.atlas_shape is None:
                self.atlas_shape = ref_liver.shape

        # Downsampled mm cloud for 3-D scatter (stored as surface_pts_mm)
        ref_pts_mm, stride = downsample_voxels_to_mm(ref_vox_idx, ref_affine)
        self.all_voxel_idx  = ref_vox_idx
        self.surface_pts_mm = ref_pts_mm

        n_pts          = len(ref_vox_idx)
        portal_sum     = np.zeros(n_pts, dtype=np.float64)
        hepatic_sum    = np.zeros(n_pts, dtype=np.float64)
        combined_sum   = np.zeros(n_pts, dtype=np.float64)
        portal_count   = np.zeros(n_pts, dtype=np.int32)
        hepatic_count  = np.zeros(n_pts, dtype=np.int32)
        combined_count = np.zeros(n_pts, dtype=np.int32)

        # ---- Reference patient — native space, no warp needed ----
        log.info(f"\n  Computing reference patient distances ({self.atlas_id})...")
        ref_pts_full = voxels_to_mm(ref_vox_idx.astype(np.float32), ref_affine)

        def _mask_or_empty(mask):
            """Return a valid binary volume even when a vessel mask is missing."""
            return mask if mask is not None else np.zeros((1, 1, 1), dtype=np.uint8)

        ref_portal_mm  = extract_vessel_mm(_mask_or_empty(ref_data.get("portal_vein")), ref_affine)
        ref_hepatic_mm = extract_vessel_mm(_mask_or_empty(ref_data.get("hepatic_vein")), ref_affine)

        ref_d_p = knn_mean_distance(ref_pts_full, ref_portal_mm,  self.k)
        ref_d_h = knn_mean_distance(ref_pts_full, ref_hepatic_mm, self.k)
        ref_d_c = np.fmin(ref_d_p, ref_d_h)

        _accum(portal_sum,   portal_count,   ref_d_p)
        _accum(hepatic_sum,  hepatic_count,  ref_d_h)
        _accum(combined_sum, combined_count, ref_d_c)

        # ---- Source patients ----
        for pid in patient_ids:
            if pid == self.atlas_id:
                continue

            log.info(f"\n{'='*60}")
            log.info(f"  Processing patient {pid}")
            log.info(f"{'='*60}")

            try:
                src_data = load_patient(self.data_dir, pid)
                src_liver = src_data["liver"]

                # Step 1: compute distance map in native patient space
                log.info(f"  Computing native-space distances...")
                src_vox_idx    = extract_liver_voxels(src_liver)
                src_pts_full   = voxels_to_mm(src_vox_idx.astype(np.float32),
                                              src_data["affine"])

                src_portal_mm  = extract_vessel_mm(
                    _mask_or_empty(src_data.get("portal_vein")),
                    src_data["affine"])
                src_hepatic_mm = extract_vessel_mm(
                    _mask_or_empty(src_data.get("hepatic_vein")),
                    src_data["affine"])

                # Native-space distance volumes (sparse — only liver voxels)
                nat_d_p = knn_mean_distance(src_pts_full, src_portal_mm,  self.k)
                nat_d_h = knn_mean_distance(src_pts_full, src_hepatic_mm, self.k)
                nat_d_c = np.fmin(nat_d_p, nat_d_h)

                # Scatter into dense native-space volumes for warping
                src_shape = src_liver.shape
                def _to_vol(distances):
                    """Scatter sparse liver distances into a dense NaN-padded volume."""
                    vol = np.full(src_shape, np.nan, dtype=np.float32)
                    valid = ~np.isnan(distances)
                    xi, yi, zi = (src_vox_idx[valid, 0],
                                  src_vox_idx[valid, 1],
                                  src_vox_idx[valid, 2])
                    vol[xi, yi, zi] = distances[valid]
                    return vol

                vol_p = _to_vol(nat_d_p)
                vol_h = _to_vol(nat_d_h)
                vol_c = _to_vol(nat_d_c)

                # Step 2: load rigid alignment (from cache or compute)
                patient_voxels = None
                if all_extents is not None:
                    patient_voxels = all_extents.get(pid, {}).get("voxel_count", None)

                alignment = align_patient(
                    patient_id          = pid,
                    atlas_id            = self.atlas_id,
                    data_dir            = self.data_dir,
                    cache_dir           = self.cache_dir,
                    median_volume       = median_volume,
                    patient_volume      = patient_voxels,
                    canonical_direction = canonical_direction,
                )
                if alignment is None:
                    log.warning(f"  Skipping {pid} — alignment failed.")
                    continue

                # Step 3: warp distance maps into atlas space (trilinear)
                log.info(f"  Warping distance maps into atlas space...")
                warp_p = apply_rigid_to_volume(vol_p, alignment,
                                               self.atlas_shape, order=1)
                warp_h = apply_rigid_to_volume(vol_h, alignment,
                                               self.atlas_shape, order=1)
                warp_c = apply_rigid_to_volume(vol_c, alignment,
                                               self.atlas_shape, order=1)

                # Step 4: sample the warped distance volumes at the reference
                # liver voxel locations
                xi, yi, zi = (ref_vox_idx[:, 0],
                              ref_vox_idx[:, 1],
                              ref_vox_idx[:, 2])

                def _safe_sample(warped_vol):
                    """Sample warped volume at reference voxels and mark missing data NaN."""
                    out = np.full(n_pts, np.nan, dtype=np.float32)
                    valid = ((xi < warped_vol.shape[0]) &
                             (yi < warped_vol.shape[1]) &
                             (zi < warped_vol.shape[2]))
                    sampled = warped_vol[xi[valid], yi[valid], zi[valid]]
                    # NaN in warped volume = no data at that location
                    sampled[sampled == 0] = np.nan   # 0 from map_coordinates OOB
                    out[valid] = sampled
                    return out

                d_p = _safe_sample(warp_p)
                d_h = _safe_sample(warp_h)
                d_c = _safe_sample(warp_c)

                _accum(portal_sum,   portal_count,   d_p)
                _accum(hepatic_sum,  hepatic_count,  d_h)
                _accum(combined_sum, combined_count, d_c)
                log.info(f"  Patient {pid} done.")

            except Exception as exc:
                log.warning(f"  Patient {pid} failed: {exc}", exc_info=True)

        # ---- Normalise ----
        # Cohort size is reported as source patient count (validated IDs,
        # excluding atlas reference), while the reference still contributes
        # to the accumulated averages above.
        self.n_patients = len(patient_ids)

        full_p = np.where(portal_count   > 0, portal_sum   / portal_count,   np.nan).astype(np.float32)
        full_h = np.where(hepatic_count  > 0, hepatic_sum  / hepatic_count,  np.nan).astype(np.float32)
        full_c = np.where(combined_count > 0, combined_sum / combined_count, np.nan).astype(np.float32)

        self.full_dist_portal   = full_p
        self.full_dist_hepatic  = full_h
        self.full_dist_combined = full_c

        # Downsampled versions for 3-D scatter
        self.mean_dist_portal   = full_p[::stride]
        self.mean_dist_hepatic  = full_h[::stride]
        self.mean_dist_combined = full_c[::stride]

        self._print_stats()

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, out_dir: Path) -> None:
        """Persist computed voxel grids and distance arrays to disk.

        Args:
            out_dir: Destination directory for .npy outputs.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "vdc_all_voxel_idx.npy",      self.all_voxel_idx)
        np.save(out_dir / "vdc_full_dist_portal.npy",   self.full_dist_portal)
        np.save(out_dir / "vdc_full_dist_hepatic.npy",  self.full_dist_hepatic)
        np.save(out_dir / "vdc_full_dist_combined.npy", self.full_dist_combined)
        np.save(out_dir / "vdc_surface_pts_mm.npy",     self.surface_pts_mm)
        np.save(out_dir / "vdc_dist_portal.npy",        self.mean_dist_portal)
        np.save(out_dir / "vdc_dist_hepatic.npy",       self.mean_dist_hepatic)
        np.save(out_dir / "vdc_dist_combined.npy",      self.mean_dist_combined)
        np.save(out_dir / "vdc_atlas_affine.npy",       self.atlas_affine)
        np.save(out_dir / "vdc_n_patients.npy",            np.array(self.n_patients))
        log.info(f"  Saved VascularDistanceCloud to {out_dir}")

    def load(self, out_dir: Path) -> None:
        """Load previously saved vascular-distance arrays from disk.

        Args:
            out_dir: Directory containing files written by save().
        """
        out_dir = Path(out_dir)
        self.all_voxel_idx      = np.load(out_dir / "vdc_all_voxel_idx.npy")
        self.full_dist_portal   = np.load(out_dir / "vdc_full_dist_portal.npy")
        self.full_dist_hepatic  = np.load(out_dir / "vdc_full_dist_hepatic.npy")
        self.full_dist_combined = np.load(out_dir / "vdc_full_dist_combined.npy")
        self.surface_pts_mm     = np.load(out_dir / "vdc_surface_pts_mm.npy")
        self.mean_dist_portal   = np.load(out_dir / "vdc_dist_portal.npy")
        self.mean_dist_hepatic  = np.load(out_dir / "vdc_dist_hepatic.npy")
        self.mean_dist_combined = np.load(out_dir / "vdc_dist_combined.npy")
        self.atlas_affine       = np.load(out_dir / "vdc_atlas_affine.npy")
        n_path = out_dir / "vdc_n_patients.npy"
        self.n_patients = int(np.load(n_path)) if n_path.exists() else len(self.mean_dist_combined)
        log.info(f"  Loaded VascularDistanceCloud from {out_dir}")
        if self.all_voxel_idx is not None and self.all_voxel_idx.max() < 10:
            log.warning("  all_voxel_idx looks wrong — delete the output folder and re-run to rebuild")
        self._print_stats()

    # ------------------------------------------------------------------
    # Visualise — 3-D scatter
    # ------------------------------------------------------------------

    def visualize(self,
                  mode: str = "combined",
                  point_cap: int = 50_000,
                  output_html: Optional[str] = None) -> None:
        """Render a 3-D liver point cloud colored by mean vessel distance.

        Args:
            mode: Distance channel to display: portal, hepatic, or combined.
            point_cap: Maximum number of points plotted for interactivity.
            output_html: Optional path to save the Plotly figure as HTML.
        """
        if self.surface_pts_mm is None:
            raise RuntimeError("No data — run build() or load() first.")

        dist_map = {
            "portal":   (self.mean_dist_portal,   "Portal vein"),
            "hepatic":  (self.mean_dist_hepatic,  "Hepatic vessels"),
            "combined": (self.mean_dist_combined, "Nearest vessel"),
        }
        if mode not in dist_map:
            raise ValueError(f"mode must be one of {list(dist_map)}")

        distances, label = dist_map[mode]
        valid  = ~np.isnan(distances)
        pts    = self.surface_pts_mm[valid]
        dists  = distances[valid]

        if len(pts) > point_cap:
            idx   = np.random.choice(len(pts), point_cap, replace=False)
            pts   = pts[idx]
            dists = dists[idx]

        print(f"  Plotting {len(pts):,} points  (mode={mode}, "
              f"{self._cohort_subject_text()})")
        print(f"  Distance range: {dists.min():.1f}–{dists.max():.1f} mm  "
              f"mean={dists.mean():.1f} mm")

        fig = go.Figure(data=go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers",
            marker=dict(size=2, color=dists, colorscale="Jet",
                        reversescale=True,
                        colorbar=dict(title=dict(
                            text="Mean dist<br>to vessel (mm)", side="right")),
                        opacity=0.75, showscale=True),
            text=[f"{d:.1f} mm" for d in dists],
            hovertemplate="x=%{x:.1f}  y=%{y:.1f}  z=%{z:.1f}<br>"
                          "Mean dist: %{text}<extra></extra>",
            name="Liver volume",
        ))
        fig.update_layout(
            title=dict(
                text=(f"Distance-to-Vasculature — {self._cohort_subject_text()}<br>"
                      f"<sup>{label}  (k={self.k})  red=close  blue=far</sup>"),
                x=0.5),
            scene=dict(xaxis_title="x (mm)", yaxis_title="y (mm)",
                       zaxis_title="z (mm)", aspectmode="data"),
            margin=dict(l=0, r=0, t=90, b=0),
        )

        if output_html:
            fig.write_html(output_html)
            print(f"  Saved: {output_html}")
        fig.show()

    def visualize_all_modes(self, output_html: Optional[str] = None) -> None:
        """Render portal, hepatic, and combined distance clouds side by side.

        Args:
            output_html: Optional path to save the Plotly figure as HTML.
        """
        if self.surface_pts_mm is None:
            raise RuntimeError("No data — run build() or load() first.")

        fig = make_subplots(
            rows=1, cols=3,
            specs=[[{"type": "scene"}]*3],
            subplot_titles=["Portal vein", "Hepatic vessels", "Combined"],
            horizontal_spacing=0.02,
        )
        d_min = float(np.nanmin(self.mean_dist_combined))
        d_max = float(np.nanmax(self.mean_dist_combined))

        for col, (distances, name) in enumerate([
            (self.mean_dist_portal,   "Portal"),
            (self.mean_dist_hepatic,  "Hepatic"),
            (self.mean_dist_combined, "Combined"),
        ], start=1):
            valid = ~np.isnan(distances)
            pts   = self.surface_pts_mm[valid]
            d     = distances[valid]
            fig.add_trace(go.Scatter3d(
                x=pts[:,0], y=pts[:,1], z=pts[:,2],
                mode="markers",
                marker=dict(size=1.8, color=d, colorscale="Jet",
                            reversescale=True, cmin=d_min, cmax=d_max,
                            showscale=(col==3),
                            colorbar=dict(title=dict(text="mm"), thickness=15, x=1.01)
                            if col==3 else None,
                            opacity=0.7),
                name=name, showlegend=False,
            ), row=1, col=col)

        scene = dict(xaxis_title="x (mm)", yaxis_title="y (mm)",
                     zaxis_title="z (mm)", aspectmode="data")
        fig.update_layout(
            title=dict(text=f"Distance-to-Vasculature — {self._cohort_subject_text()}  "
                            f"red=close  blue=far", x=0.5),
            scene=scene, scene2=scene, scene3=scene,
            margin=dict(l=0, r=60, t=80, b=0),
        )

        if output_html:
            fig.write_html(output_html)
            print(f"  Saved: {output_html}")
        fig.show()

    # ------------------------------------------------------------------
    # Visualise — axial slice browser
    # ------------------------------------------------------------------

    def visualize_distance_slices(self,
                                   mode: str = "combined",
                                   output_html: Optional[str] = None) -> None:
        """Render an axial slice browser over full-resolution distance voxels.

        Args:
            mode: Distance channel to display: portal, hepatic, or combined.
            output_html: Optional path to save the Plotly figure as HTML.
        """
        if self.all_voxel_idx is None:
            raise RuntimeError("No data — run build() or load() first.")

        dist_map = {
            "portal":   (self.full_dist_portal,   "Portal vein"),
            "hepatic":  (self.full_dist_hepatic,  "Hepatic vessels"),
            "combined": (self.full_dist_combined, "Nearest vessel"),
        }
        if mode not in dist_map:
            raise ValueError(f"mode must be one of {list(dist_map)}")

        distances, structure_label = dist_map[mode]
        valid   = ~np.isnan(distances)
        vox_idx = self.all_voxel_idx[valid]
        dists   = distances[valid]

        print(f"Building axial slice browser ({mode}, {len(vox_idx):,} voxels)...")

        # Clip to actual liver extent
        z_min, z_max = int(vox_idx[:,2].min()), int(vox_idx[:,2].max())
        x_min, x_max = int(vox_idx[:,0].min()), int(vox_idx[:,0].max())
        y_min, y_max = int(vox_idx[:,1].min()), int(vox_idx[:,1].max())

        print(f"  Extent: x={x_min}–{x_max}  y={y_min}–{y_max}  "
              f"z={z_min}–{z_max}  ({z_max-z_min+1} slices)")

        xi = vox_idx[:,0] - x_min
        yi = vox_idx[:,1] - y_min
        zi = vox_idx[:,2] - z_min

        vol = np.full((x_max-x_min+1, y_max-y_min+1, z_max-z_min+1),
                      np.nan, dtype=np.float32)
        vol[xi, yi, zi] = dists

        n_z   = vol.shape[2]
        mid   = n_z // 2
        d_min = float(np.nanmin(vol))
        d_max = float(np.nanmax(vol))

        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            z=vol[:,:,mid].T, colorscale="Jet", reversescale=True,
            zmin=d_min, zmax=d_max,
            colorbar=dict(title=dict(text="Mean dist<br>to vessel (mm)", side="right"))))

        fig.frames = [
            go.Frame(data=[go.Heatmap(
                z=vol[:,:,z].T, colorscale="Jet", reversescale=True,
                zmin=d_min, zmax=d_max)], name=str(z))
            for z in range(n_z)
        ]

        layout = make_slider_layout(n_z, mid, prefix="Axial slice z=")
        fig.update_layout(
            title=dict(
                text=(f"Distance-to-Vasculature Axial Slices — "
                      f"{self._cohort_subject_text()}<br>"
                      f"<sup>Colored by mean distance to {structure_label} — "
                      f"red=close  blue=far</sup>"),
                x=0.5),
            xaxis_title="x (voxels)", yaxis_title="y (voxels)",
            height=600, margin=dict(l=60, r=80, t=90, b=80),
            **layout,
        )

        if output_html:
            fig.write_html(output_html)
            print(f"  Saved: {output_html}")
        fig.show()

    # ------------------------------------------------------------------
    # Visualise — histogram
    # ------------------------------------------------------------------

    def visualize_distance_histogram(self,
                                      output_html: Optional[str] = None) -> None:
        """Render histogram of combined distance values across liver voxels.

        Args:
            output_html: Optional path to save the Plotly figure as HTML.
        """
        if self.full_dist_combined is None:
            raise RuntimeError("No data — run build() or load() first.")

        valid = self.full_dist_combined[~np.isnan(self.full_dist_combined)]
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=valid, nbinsx=60, marker_color="#4363d8",
                                   opacity=0.75, name="Combined"))
        fig.add_vline(x=float(np.mean(valid)),   line_dash="dash",
                      line_color="red",    annotation_text=f"mean {np.mean(valid):.1f}mm")
        fig.add_vline(x=float(np.median(valid)), line_dash="dot",
                      line_color="orange", annotation_text=f"median {np.median(valid):.1f}mm")
        fig.update_layout(
            title=dict(text=f"Distance to Vasculature Distribution — "
                            f"{self._cohort_subject_text()}", x=0.5),
            xaxis_title="Mean distance to nearest vessel (mm)",
            yaxis_title="Number of liver voxels",
            bargap=0.05, margin=dict(l=60, r=20, t=80, b=60),
        )

        if output_html:
            fig.write_html(output_html)
            print(f"  Saved: {output_html}")
        fig.show()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _print_stats(self) -> None:
        """Print summary statistics for currently loaded full-resolution maps."""
        print(f"\n  === VascularDistanceCloud ({self._cohort_subject_text()}) ===")
        for name, arr in [("Portal",   self.full_dist_portal),
                          ("Hepatic",  self.full_dist_hepatic),
                          ("Combined", self.full_dist_combined)]:
            if arr is None:
                continue
            v = arr[~np.isnan(arr)]
            if len(v) == 0:
                print(f"  {name:10s}: no valid data")
            else:
                print(f"  {name:10s}: mean={v.mean():.1f}mm  "
                      f"median={np.median(v):.1f}mm  "
                      f"min={v.min():.1f}mm  max={v.max():.1f}mm")
        print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from Validation.dataset_loader import load_cohort

    DATA_DIR  = Path("Data")
    CACHE_DIR = Path("outputs/reg_cache")   # shared with liver_atlas.py
    USE_VALIDATED_IDS = True  # intersect reviewed cohorts with Validation/usable_patient_ids.txt

    # Must match the ATLASES config in liver_atlas.py
    ATLASES = [
        {
            "label":    "male",
            "atlas_id": "0004",
            "cohort":   "male",
            "out_dir":  Path("outputs/atlas_male"),
        },
        {
            "label":    "female",
            "atlas_id": "0011",     
            "cohort":   "female",
            "out_dir":  Path("outputs/atlas_female"),
        },
    ]

    for cfg in ATLASES:
        label    = cfg["label"]
        atlas_id = cfg["atlas_id"]
        out_dir  = cfg["out_dir"]

        print(f"\n{'#'*60}")
        print(f"  Building {label.upper()} vascular distance cloud  "
              f"(reference={atlas_id})")
        print(f"{'#'*60}")

        source_ids = load_cohort(
            DATA_DIR,
            cohort      = cfg["cohort"],
            exclude_ids = [atlas_id],
            use_validated_ids = USE_VALIDATED_IDS,
        )
        print(f"  Source patients: {len(source_ids)}")

        vdc = VascularDistanceCloud(
            atlas_id          = atlas_id,
            data_dir          = DATA_DIR,
            cache_dir         = CACHE_DIR,
            atlas_dir         = out_dir,
            density_threshold = 0.5,
            k_neighbors       = 5,
            cohort_label      = label,
        )

        load_existing = (out_dir / "vdc_dist_combined.npy").exists()
        if load_existing:
            expected_n = len(source_ids)
            n_path = out_dir / "vdc_n_patients.npy"
            cached_n = int(np.load(n_path)) if n_path.exists() else None
            if cached_n != expected_n:
                print(
                    f"  Cached VDC patient count mismatch: "
                    f"saved={cached_n}, expected={expected_n}. Rebuilding..."
                )
                vdc.build(source_ids)
                vdc.save(out_dir)
            else:
                vdc.load(out_dir)
        else:
            vdc.build(source_ids)
            vdc.save(out_dir)

        vdc.visualize(
            output_html=str(out_dir / "vdc_combined.html"))
        vdc.visualize_all_modes(
            output_html=str(out_dir / "vdc_all_modes.html"))
        vdc.visualize_distance_slices(
            output_html=str(out_dir / "vdc_slices.html"))
        vdc.visualize_distance_histogram(
            output_html=str(out_dir / "vdc_histogram.html"))

