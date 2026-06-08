# Atlas/liver_atlas.py
# Tristan Jones — Spring 2026 Capstone
#
# AI Use Disclosure
#   Student estimate: 80% student-designed, 20% AI-assisted implementation
#   Claude assisted with: two-pass build loop implementation, global offset computation
#   See: "Documentation/AI Use Disclosure.md" for full details
#
# Probabilistic Liver Atlas — "Electron Cloud" Model
#
# Dr. Hale's framing (the question that must come first):
#   "Can you overlay many livers in 3D space using a common basis,
#    compute their surface geometry, and identify the 'average liver'
#    as the densest region of the point distribution across the dataset?"
#
# Pipeline:
#   Stage 1 — COMMON BASIS DIAGNOSTIC
#     Rigid-align every patient's liver surface into atlas space.
#     Plot all surface point clouds in one figure, colored by patient.
#     Tight clustering = common basis works. Scattered = it doesn't.
#
#   Stage 2 — AVERAGE LIVER
#     Accumulate rigidly-warped liver masks → voxel-wise density volume.
#     Visualise as nested probability shells (25 / 50 / 75 %).
#
#   Stage 3 — AXIAL SLICE BROWSER
#     Scrollable 2-D heatmap of the liver density volume.

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import nibabel as nib
import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter, zoom as nd_zoom

from Registration.stages.load import load_patient
from Registration.utils.Nifti import voxels_to_mm

from Atlas.utils import (
    PATIENT_COLORS,
    pad_to_common,
    dice,
    extract_surface_mm,
    density_to_mesh,
    make_slider_layout,
)
from Atlas.registration import align_patient, apply_rigid_to_volume, forward_warp_mask, compute_warp_extents

log = logging.getLogger(__name__)


class LiverAtlas:
    """
    Probabilistic liver atlas — geometric stage only.

    Answers Dr. Hale's question: can you overlay many livers in a common
    3D basis and find the average?

    Uses rigid (Procrustes) alignment only — no TPS.
    Vascular distance analysis lives in vascular_distance.py.

    Usage
    -----
    atlas = LiverAtlas(atlas_id="0004", data_dir=Path("Data"))
    atlas.build(["0010", "0011", "0012", "0013"])
    atlas.save(Path("outputs/atlas"))

    atlas.visualize_common_basis()   # Stage 1 — show Dr. Hale this first
    atlas.visualize_average_liver()  # Stage 2
    atlas.visualize_density_slices() # Stage 3
    """

    def __init__(self,
                 atlas_id: str,
                 data_dir: Path = Path("Data"),
                 cache_dir: Path = Path("outputs/reg_cache"),
                 cohort_label: Optional[str] = None):
        """Helper for init."""
        self.atlas_id  = atlas_id
        self.data_dir  = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.cohort_label = cohort_label

        self.liver_density     : Optional[np.ndarray] = None
        self.atlas_affine      : Optional[np.ndarray] = None
        self.atlas_shape       : Optional[tuple]       = None
        self.ref_surface_mm    : Optional[np.ndarray]  = None
        self.surface_clouds_mm : dict[str, np.ndarray] = {}
        self.n_patients        = 0
        self.display_n_patients = 0
        self.registration_dice : dict = {}
        self._liver_acc        : Optional[np.ndarray] = None

    def _cohort_subject_text(self) -> str:
        """Return cohort-aware subject count text for figure titles."""
        count = self.display_n_patients or self.n_patients
        if self.cohort_label:
            return f"{count} {self.cohort_label} subjects"
        return f"{count} subjects"

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, patient_ids: list[str], organ_label: int = 1) -> None:
        """
        Two-pass atlas build that guarantees no voxels are lost to clipping.

        Pass 1 — compute warp extents for every patient and find the global
                  bounding box that fits all warped livers. Saved to a JSON
                  file so it doesn't need to be recomputed on re-runs.

        Pass 2 — warp every patient into the shared global grid using the
                  global offset so negative coordinates become positive.
                  Accumulate into the density volume.

        Args:
            patient_ids : Source patient IDs, e.g. ["0010", "0011"].
            organ_label : Voxel label for liver in liver.nii.gz (usually 1).
        """
        import json

        log.info(f"\n{'='*60}")
        log.info(f"  Loading ATLAS reference: {self.atlas_id}")
        log.info(f"{'='*60}")

        ref_data    = load_patient(self.data_dir, self.atlas_id)
        ref_liver   = ref_data["liver"]
        ref_affine  = ref_data["affine"]

        self.atlas_affine = ref_affine
        self.atlas_shape  = ref_liver.shape
        self.ref_surface_mm = extract_surface_mm(ref_liver, ref_affine)

        # Path for the bounding box JSON
        extents_path = self.cache_dir / f"extents_{self.atlas_id}.json"

        # ------------------------------------------------------------------
        # Pass 1 — compute or load warp extents + voxel counts + directions
        # ------------------------------------------------------------------

        def _direction(affine):
            """Extract normalized direction matrix from NIfTI affine."""
            rot   = affine[:3, :3]
            zooms = np.sqrt((rot**2).sum(axis=0))
            return (rot / zooms).tolist()

        if extents_path.exists():
            print(f"\n  Loading cached extents from {extents_path}")
            with open(extents_path) as f:
                all_extents = json.load(f)
        else:
            print(f"\n{'='*60}")
            print(f"  Pass 1 — Computing warp extents + voxel counts + directions")
            print(f"{'='*60}")
            all_extents = {}

            # Reference patient
            ref_voxels = int((ref_liver > 0).sum())
            all_extents[self.atlas_id] = {
                "min":         [0, 0, 0],
                "max":         list(ref_liver.shape),
                "voxel_count": ref_voxels,
                "direction":   _direction(ref_affine),
            }

            for i, pid in enumerate(patient_ids, start=1):
                if pid == self.atlas_id:
                    continue
                print(f"  [{i}/{len(patient_ids)}] {pid} — computing extents...")
                try:
                    alignment = align_patient(
                        patient_id = pid,
                        atlas_id   = self.atlas_id,
                        data_dir   = self.data_dir,
                        cache_dir  = self.cache_dir,
                    )
                    if alignment is None:
                        continue

                    src_data    = load_patient(self.data_dir, pid)
                    src_liver_p = src_data["liver"]
                    extents     = compute_warp_extents(src_liver_p, alignment)
                    extents["voxel_count"] = int((src_liver_p > 0).sum())
                    extents["direction"]   = _direction(src_data["affine"])
                    all_extents[pid] = extents
                    print(f"    voxels={extents['voxel_count']:,}  "
                          f"min={extents['min']}  max={extents['max']}")

                except Exception as exc:
                    log.warning(f"  {pid} extents failed: {exc}")

            with open(extents_path, "w") as f:
                json.dump(all_extents, f, indent=2)
            print(f"  Saved extents → {extents_path}")

        # Compute global bounding box across all patients
        all_mins = np.array([e["min"] for e in all_extents.values()])
        all_maxs = np.array([e["max"] for e in all_extents.values()])
        global_min = all_mins.min(axis=0)
        global_max = all_maxs.max(axis=0)

        # Offset shifts all coordinates so global_min maps to (0,0,0)
        global_offset = global_min.clip(max=0)
        global_shape  = tuple(
            int(global_max[i] - global_offset[i]) + 1 for i in range(3))

        # Median liver volume across all patients — used for scale normalization
        all_voxel_counts = [e["voxel_count"] for e in all_extents.values()
                            if "voxel_count" in e]
        median_volume = float(np.median(all_voxel_counts))
        print(f"\n  Liver volume stats across {len(all_voxel_counts)} patients:")
        print(f"    Median : {median_volume:,.0f} voxels")
        print(f"    Mean   : {np.mean(all_voxel_counts):,.0f} voxels")
        print(f"    Min    : {np.min(all_voxel_counts):,.0f} voxels")
        print(f"    Max    : {np.max(all_voxel_counts):,.0f} voxels")

        # Canonical direction — mean orientation across all patients
        # orthonormalized via SVD to be a proper rotation matrix
        all_directions = np.array([e["direction"] for e in all_extents.values()
                                   if "direction" in e])
        mean_dir = all_directions.mean(axis=0)
        U, _, Vt = np.linalg.svd(mean_dir)
        canonical_direction = U @ Vt
        if np.linalg.det(canonical_direction) < 0:
            U[:, -1] *= -1
            canonical_direction = U @ Vt
        print(f"\n  Canonical direction matrix (mean population orientation):")
        print(f"{canonical_direction.round(4)}")

        print(f"\n  Global bounding box:")
        print(f"    Min coords : {global_min.tolist()}")
        print(f"    Max coords : {global_max.tolist()}")
        print(f"    Offset     : {global_offset.tolist()}")
        print(f"    Grid shape : {global_shape}")

        # ------------------------------------------------------------------
        # Pass 2 — warp all patients into the shared global grid
        # ------------------------------------------------------------------
        print(f"\n{'='*60}")
        print(f"  Pass 2 — Warping all patients into shared global grid")
        print(f"{'='*60}")

        # Reference patient — shift into global grid
        ref_shifted = np.zeros(global_shape, dtype=ref_liver.dtype)
        ox, oy, oz  = [-int(global_offset[i]) for i in range(3)]
        rx, ry, rz  = ref_liver.shape
        ref_shifted[ox:ox+rx, oy:oy+ry, oz:oz+rz] = ref_liver

        self._liver_acc = (ref_shifted > 0).astype(np.float32)
        self.ref_surface_mm = extract_surface_mm(ref_shifted, ref_affine)
        n_accumulated = 1

        for i, pid in enumerate(patient_ids, start=1):
            if pid == self.atlas_id:
                continue

            print(f"\n  [{i}/{len(patient_ids)}] Patient {pid}")

            try:
                # Get this patient's voxel count from the JSON
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

                src_data  = load_patient(self.data_dir, pid)
                src_liver = src_data["liver"]

                # Dice BEFORE — resample src to ref grid
                zf = tuple(r / s for r, s in zip(ref_liver.shape, src_liver.shape))
                src_rs = (nd_zoom(src_liver.astype(np.float32), zf, order=0) > 0.5
                          ).astype(src_liver.dtype)
                dice_before = dice(ref_liver, src_rs, label=organ_label)

                # Warp into shared global grid — no voxels dropped
                xfm_liver = forward_warp_mask(
                    src_liver, alignment, global_shape,
                    global_offset=global_offset)

                dice_after = dice(ref_shifted, xfm_liver, label=organ_label)
                self.registration_dice[pid] = {
                    "liver_before": dice_before,
                    "liver_after":  dice_after,
                }
                print(f"  Dice before: {dice_before:.4f}  |  after: {dice_after:.4f}")

                # Warped surface for diagnostic plot
                self.surface_clouds_mm[pid] = extract_surface_mm(
                    xfm_liver, ref_affine)

                # Accumulate
                xfm_l = (xfm_liver > 0).astype(np.float32)
                self._liver_acc, xfm_l = pad_to_common(self._liver_acc, xfm_l)
                self._liver_acc += xfm_l

                n_accumulated += 1
                print(f"  Patient {pid} accumulated.")

            except Exception as exc:
                log.warning(f"  Patient {pid} failed: {exc}", exc_info=True)

        self.n_patients    = n_accumulated
        self.display_n_patients = len(self.surface_clouds_mm)
        self.liver_density = self._liver_acc / n_accumulated
        nz = self.liver_density[self.liver_density > 0]
        print(f"\n  Atlas built from {n_accumulated} patient(s).")
        print(f"  Grid shape: {global_shape}")
        print(f"  Liver density — max: {self.liver_density.max():.3f}  "
              f"mean(nonzero): {nz.mean():.3f}")

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, out_dir: Path) -> None:
        """Execute save."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        path = out_dir / "atlas_liver_density.nii.gz"
        nib.save(nib.Nifti1Image(
            self.liver_density.astype(np.float32), self.atlas_affine), str(path))
        log.info(f"  Saved {path}")

        np.save(out_dir / "ref_surface_mm.npy", self.ref_surface_mm)
        np.save(out_dir / "n_patients.npy", np.array(self.n_patients))
        for pid, cloud in self.surface_clouds_mm.items():
            np.save(out_dir / f"surface_{pid}_mm.npy", cloud)
        log.info(f"  Saved surface clouds for {len(self.surface_clouds_mm)} patient(s).")

    def load(self, out_dir: Path) -> None:
        """Execute load."""
        out_dir = Path(out_dir)
        img = nib.load(str(out_dir / "atlas_liver_density.nii.gz"))
        self.atlas_affine  = img.affine
        self.liver_density = img.get_fdata().astype(np.float32)

        ref_path = out_dir / "ref_surface_mm.npy"
        if ref_path.exists():
            self.ref_surface_mm = np.load(ref_path)

        import glob
        for cp in sorted(out_dir.glob("surface_*_mm.npy")):
            pid = cp.stem.replace("surface_", "").replace("_mm", "")
            self.surface_clouds_mm[pid] = np.load(cp)

        n_path = out_dir / "n_patients.npy"
        self.n_patients = int(np.load(n_path)) if n_path.exists() else len(self.surface_clouds_mm) + 1
        self.display_n_patients = len(self.surface_clouds_mm) or max(self.n_patients - 1, 0)

        log.info(f"  Atlas loaded from {out_dir}  "
                 f"({len(self.surface_clouds_mm)} source clouds)")

    # ------------------------------------------------------------------
    # Stage 1 — COMMON BASIS DIAGNOSTIC
    # ------------------------------------------------------------------

    def visualize_common_basis(self,
                                point_cap: int = 3000,
                                mesh_offset_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
                                side_by_side_mesh: bool = False,
                                side_by_side_gap_mm: float = 25.0,
                                output_html: Optional[str] = None) -> None:
        """
        Overlay all liver surface clouds in atlas mm space.
        Green = reference patient. Each source patient gets its own color.
        Tight overlap = common basis is working.

        Args:
            point_cap: Maximum points shown per cloud.
            mesh_offset_mm: Optional (x, y, z) offset in mm applied to the
                Stage-1 marching-cubes shell for visual separation.
            side_by_side_mesh: If True, ignores mesh_offset_mm and computes an
                automatic x-offset that places the shell entirely to the right
                of all point clouds.
            side_by_side_gap_mm: Gap (mm) between the right edge of the clouds
                and the left edge of the shifted shell when side_by_side_mesh
                is enabled.
            output_html: Optional file path to save the interactive figure.
        """
        if self.ref_surface_mm is None:
            raise RuntimeError("No surface data — run build() or load() first.")

        print("Building common-basis diagnostic figure...")
        traces = []

        ref_pts = self.ref_surface_mm
        if len(ref_pts) > point_cap:
            ref_pts = ref_pts[np.random.choice(len(ref_pts), point_cap, replace=False)]
        traces.append(go.Scatter3d(
            x=ref_pts[:, 0], y=ref_pts[:, 1], z=ref_pts[:, 2],
            mode="markers",
            marker=dict(size=1.5, color="#3cb44b", opacity=0.6),
            name=f"Reference ({self.atlas_id})",
        ))

        for i, (pid, cloud) in enumerate(sorted(self.surface_clouds_mm.items())):
            pts = cloud
            if len(pts) > point_cap:
                pts = pts[np.random.choice(len(pts), point_cap, replace=False)]
            d = self.registration_dice.get(pid, {})
            label = (f"Patient {pid}  "
                     f"(Dice before={d.get('liver_before', 0):.3f}  "
                     f"after={d.get('liver_after', 0):.3f})")
            traces.append(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode="markers",
                marker=dict(size=1.5,
                            color=PATIENT_COLORS[i % len(PATIENT_COLORS)],
                            opacity=0.5),
                name=label,
            ))

        if self.liver_density is not None:
            mesh = density_to_mesh(
                self.liver_density, 0.5, "#3cb44b", 0.08,
                "Average liver (≥50%)", self.atlas_affine)
            if mesh:
                dx, dy, dz = mesh_offset_mm
                if side_by_side_mesh:
                    cloud_max_x = float(np.max(ref_pts[:, 0]))
                    for cloud in self.surface_clouds_mm.values():
                        if len(cloud) > 0:
                            cloud_max_x = max(cloud_max_x, float(np.max(cloud[:, 0])))

                    mesh_min_x = float(np.min(np.asarray(mesh.x)))
                    dx = (cloud_max_x + side_by_side_gap_mm) - mesh_min_x
                    dy = 0.0
                    dz = 0.0

                if dx != 0.0 or dy != 0.0 or dz != 0.0:
                    mesh.x = np.asarray(mesh.x) + dx
                    mesh.y = np.asarray(mesh.y) + dy
                    mesh.z = np.asarray(mesh.z) + dz
                    mesh.name = f"{mesh.name} (offset ({dx:.1f}, {dy:.1f}, {dz:.1f}) mm)"
                traces.append(mesh)

        fig = go.Figure(data=traces)
        fig.update_layout(
            title=dict(
                text=(f"Stage 1 — Common Basis Diagnostic — "
                      f"{self._cohort_subject_text()}<br>"
                      f"<sup>Colored clouds should overlap the green reference</sup>"),
                x=0.5,
            ),
            scene=dict(xaxis_title="x (mm)", yaxis_title="y (mm)",
                       zaxis_title="z (mm)", aspectmode="data"),
            legend=dict(itemsizing="constant"),
            margin=dict(l=0, r=0, t=90, b=0),
        )

        if output_html:
            fig.write_html(output_html)
            print(f"  Saved: {output_html}")
        fig.show()
        self._print_spread_stats()

    # ------------------------------------------------------------------
    # Stage 2 — AVERAGE LIVER
    # ------------------------------------------------------------------

    def visualize_average_liver(self,
                                 thresholds: list[float] = [0.25, 0.5, 0.75],
                                 output_html: Optional[str] = None) -> None:
        """
        Nested isosurface shells of the liver density volume.
        Each shell = fraction of patients with liver at that location.
        """
        if self.liver_density is None:
            raise RuntimeError("No density data — run build() or load() first.")

        print("Building average liver figure...")
        shell_styles = {0.25: ("#a8e6a3", 0.12),
                        0.50: ("#3cb44b", 0.30),
                        0.75: ("#1a5e26", 0.60)}

        smooth  = gaussian_filter(self.liver_density, sigma=1.5)
        traces  = []
        for level in sorted(thresholds):
            color, opacity = shell_styles.get(level, ("#3cb44b", 0.30))
            m = density_to_mesh(smooth, level, color, opacity,
                                f"Liver in ≥{int(level*100)}% of patients",
                                self.atlas_affine)
            if m:
                traces.append(m)

        if not traces:
            print("  No isosurfaces found — try lowering thresholds.")
            return

        fig = go.Figure(data=traces)
        fig.update_layout(
            title=dict(
                text=(f"Stage 2 — Average Liver — {self._cohort_subject_text()}<br>"
                      f"<sup>Nested shells show cross-patient consistency</sup>"),
                x=0.5,
                y=0.96,
                xanchor="center",
                yanchor="top"),
            scene=dict(xaxis_title="x (mm)", yaxis_title="y (mm)",
                       zaxis_title="z (mm)", aspectmode="data",
                       domain=dict(x=[0.0, 0.83], y=[0.0, 1.0])),
            legend=dict(
                x=0.85,
                y=0.94,
                xanchor="left",
                yanchor="top",
                bgcolor="rgba(255,255,255,0.75)",
                bordercolor="rgba(0,0,0,0.15)",
                borderwidth=1,
            ),
            margin=dict(l=20, r=20, t=70, b=20),
            width=1250,
            height=780,
        )

        if output_html:
            fig.write_html(output_html)
            print(f"  Saved: {output_html}")
        fig.show()

    # ------------------------------------------------------------------
    # Stage 3 — AXIAL SLICE BROWSER
    # ------------------------------------------------------------------

    def visualize_density_slices(self,
                                  output_html: Optional[str] = None) -> None:
        """Scrollable axial heatmap cropped to the liver-density extent."""
        if self.liver_density is None:
            raise RuntimeError("No density data — run build() or load() first.")

        print("Building axial slice browser...")
        vol_full = self.liver_density

        present = vol_full > 0
        if np.any(present):
            vox_idx = np.argwhere(present)
            x_min, y_min, z_min = vox_idx.min(axis=0)
            x_max, y_max, z_max = vox_idx.max(axis=0)
            vol = vol_full[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
            print(f"  Cropped extent: x={x_min}–{x_max}  y={y_min}–{y_max}  "
                  f"z={z_min}–{z_max}  ({z_max-z_min+1} slices)")
        else:
            # Defensive fallback for empty volumes.
            x_min, y_min, z_min = 0, 0, 0
            vol = vol_full
            print("  Warning: no nonzero density voxels found; using full volume extent")

        n_z = vol.shape[2]
        mid = n_z // 2

        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            z=vol[:, :, mid].T, colorscale="Greens", zmin=0, zmax=1,
            colorbar=dict(title="Probability")))

        fig.frames = [
            go.Frame(data=[go.Heatmap(z=vol[:, :, z].T, colorscale="Greens",
                                      zmin=0, zmax=1)], name=str(z))
            for z in range(n_z)
        ]

        layout = make_slider_layout(n_z, mid, prefix="Axial slice z=")
        fig.update_layout(
            title=dict(text=f"Stage 3 — Liver Density Slices — "
                            f"{self._cohort_subject_text()}<br>"
                            f"<sup>Cropped to liver extent; global z start={z_min}</sup>", x=0.5),
            xaxis_title="x (voxels)", yaxis_title="y (voxels)",
            height=550, margin=dict(l=60, r=60, t=80, b=80),
            **layout,
        )

        if output_html:
            fig.write_html(output_html)
            print(f"  Saved: {output_html}")
        fig.show()

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def print_registration_summary(self) -> None:
        """Execute print registration summary."""
        if not self.registration_dice:
            print("No registration results yet.")
            return
        print(f"\n{'Patient':>10}  {'Dice before':>12}  {'Dice after':>10}  {'Delta':>8}")
        print("-" * 50)
        for pid, d in sorted(self.registration_dice.items()):
            b, a  = d["liver_before"], d["liver_after"]
            delta = a - b
            arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "=")
            print(f"  {pid:>8}  {b:>12.4f}  {a:>10.4f}  {arrow}{abs(delta):.4f}")
        print()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _print_spread_stats(self) -> None:
        """Helper for print spread stats."""
        if not self.surface_clouds_mm:
            return
        centroids = np.array([c.mean(axis=0) for c in self.surface_clouds_mm.values()])
        ref_c     = self.ref_surface_mm.mean(axis=0)
        all_c     = np.vstack([ref_c[np.newaxis], centroids])
        std       = all_c.std(axis=0)
        dists     = np.linalg.norm(centroids - ref_c, axis=1)
        print(f"\n  === Common Basis Spread (mm) ===")
        print(f"  Centroid std dev:  x={std[0]:.1f}  y={std[1]:.1f}  z={std[2]:.1f}")
        print(f"  Mean centroid dist from reference: {dists.mean():.1f} mm")
        print(f"  Max  centroid dist from reference: {dists.max():.1f} mm")
        print(f"\n  Interpretation:")
        print(f"    < 10 mm  → common basis looks good")
        print(f"    10–20 mm → acceptable, check 3D figure")
        print(f"    > 20 mm  → alignment insufficient")
        print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from Validation.dataset_loader import load_cohort, print_cohort_summary

    DATA_DIR  = Path("Data")
    CACHE_DIR = Path("outputs/reg_cache")   # shared across all atlases
    USE_VALIDATED_IDS = True  # intersect reviewed cohorts with Validation/usable_patient_ids.txt

    print_cohort_summary(DATA_DIR)

    # ------------------------------------------------------------------
    # Choose which atlases to build — comment out any you don't need
    # ------------------------------------------------------------------
    ATLASES = [
        {
            "label":    "male",
            "atlas_id": "0004",           # reference patient for male atlas
            "cohort":   "male",           # reads reviewed_male.csv
            "out_dir":  Path("outputs/atlas_male"),
        },
        {
            "label":    "female",
            "atlas_id": "0011",           # reference patient for female atlas
                                          # ← change to a known female patient ID
            "cohort":   "female",         # reads reviewed_female.csv
            "out_dir":  Path("outputs/atlas_female"),
        },
    ]
    # ------------------------------------------------------------------

    for cfg in ATLASES:
        label    = cfg["label"]
        atlas_id = cfg["atlas_id"]
        out_dir  = cfg["out_dir"]

        print(f"\n{'#'*60}")
        print(f"  Building {label.upper()} atlas  (reference={atlas_id})")
        print(f"{'#'*60}")

        source_ids = load_cohort(
            DATA_DIR,
            cohort      = cfg["cohort"],
            exclude_ids = [atlas_id],     # don't register reference to itself
            use_validated_ids = USE_VALIDATED_IDS,
        )
        print(f"  Source patients: {len(source_ids)}")

        atlas = LiverAtlas(
            atlas_id  = atlas_id,
            data_dir  = DATA_DIR,
            cache_dir = CACHE_DIR,        # shared cache — rigid alignments reused
            cohort_label = label,
        )

        load_existing = (out_dir / "atlas_liver_density.nii.gz").exists()
        if load_existing:
            atlas.load(out_dir)
            # Atlas n_patients includes the reference patient (+1).
            expected_n = len(source_ids) + 1
            if atlas.n_patients != expected_n:
                print(
                    f"  Cached atlas patient count mismatch: "
                    f"saved={atlas.n_patients}, expected={expected_n}. Rebuilding..."
                )
                atlas.build(source_ids)
                atlas.save(out_dir)
        else:
            atlas.build(source_ids)
            atlas.save(out_dir)

        atlas.print_registration_summary()
        atlas.visualize_common_basis(
            side_by_side_mesh=True,
            side_by_side_gap_mm=25.0,
            output_html=str(out_dir / "01_common_basis.html"))
        atlas.visualize_average_liver(
            output_html=str(out_dir / "02_average_liver.html"))
        atlas.visualize_density_slices(
            output_html=str(out_dir / "03_density_slices.html"))