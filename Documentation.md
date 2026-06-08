# Documentation - Liver Atlas Capstone
## CIST 8950 - Spring 2026 - Tristan Jones

This is a file-by-file walkthrough of the project.
I keep it simple: what each function does, what goes in, what comes out,
and (for math-heavy parts) what the calculation means in plain language.

AI Usage disclosure:
I used Claude to help me to make this file, I basically provided it both 
my pipeline visualization that I created and additonally a write up on the 
important functions here and how they work, then asked Claude to help me to 
format this file. Then VS Code helped to kind of refine this even more and I 
had it add the file dependency map. 

---

## How the System Works - Pipeline Overview

The project has five main stages. Each stage feeds into the next.
Here is how they connect:

    [1] Segmentation
        Segmentation/run_totalsegmentator.py
        - Runs TotalSegmentator on raw CT scans
        - Produces liver, vessel, and segment masks for each patient
        - Saves everything into a segmentations.zip file
        |
        v
    [2] Validation
        Validation/validate_dataset.py
        Validation/dataset_loader.py
        - Checks the zip for bad/empty segmentations before anything else runs
        - Loads the reviewed male/female CSV files to decide which patients to use
        - Produces usable_patient_ids.txt (one ID per line, patients that passed)
        - This file is the gate for everything downstream:
          dataset_loader.load_cohort() intersects the reviewed CSV with this list
          so any patient that failed validation is silently dropped before
          liver_atlas.py or vascular_distance.py ever sees them
        |
        v
    [3] Liver Atlas (Model Setup Stage)
        Atlas/liver_atlas.py
        Atlas/registration.py
        Registration/stages/align.py
        Registration/stages/load.py
        Registration/utils/Nifti.py
        - Reads the patient ID list from validation
        - Loads each patient's segmentation from the zip
        - Computes rigid alignment (rotation + translation) to a reference patient
        - Warps all liver masks into a shared atlas space
        - Accumulates them into a density volume (fraction of patients with liver at each voxel)
        - Produces liver_density.nii.gz and surface clouds for visualization
        |
        v
    [4] Vasculature Distance Calculations (Model Setup Stage)
        Atlas/vascular_distance.py
        Atlas/registration.py
        Atlas/utils.py
        - Loads segmentations again for each patient
        - Loads the pre-computed alignment from stage 3 (no recomputing)
        - Computes distance from every liver voxel to nearest portal and hepatic vessels
        - Does this in native patient space, then warps results to atlas space
        - Accumulates mean distance maps across all patients
        - Saves portal, hepatic, and combined distance models
        - Can visualize as 3D scatter, slices, or histogram
        |
        v
    [5] Applications (Model Testing Stage - future work)
        - Load segmentation of a new patient
        - Compute transformation parameters using the same alignment pipeline
        - Look up vascular distance values from the atlas model
        - Predict distance to vasculature for that new patient

---

## File Dependency Map

Active files (actually used in the pipeline):

    run_totalsegmentator.py     -- standalone, no project deps
    validate_dataset.py         -- reads output of run_totalsegmentator.py
    dataset_loader.py           -- reads reviewed CSV, used by liver_atlas.py and vascular_distance.py
    Nifti.py                    -- used by load.py, align.py, registration.py, utils.py
    load.py                     -- used by align.py and atlas/registration.py
    align.py                    -- used by atlas/registration.py
    atlas/registration.py       -- used by liver_atlas.py and vascular_distance.py
    atlas/utils.py              -- used by liver_atlas.py and vascular_distance.py
    liver_atlas.py              -- core atlas, run first in model setup
    vascular_distance.py        -- run after liver_atlas.py, depends on alignment cache

Legacy files (still in the repo but not called by anything):

    Checkpoint.py               -- old pickle-based checkpoint store, replaced by .npz cache
    tps.py                      -- Thin Plate Spline warp code, replaced by rigid-only pipeline
    evaluate.py                 -- TPS evaluation helpers, no longer called

---

## Registration/utils/Nifti.py

This file handles voxel-to-mm coordinate conversion.
That matters because different CT scans can use different voxel spacing.

### load_nifti(path)
Loads a NIfTI file from disk.
- In: path to .nii or .nii.gz
- Out: (data, affine)

### save_nifti(data, affine, path)
Saves a numpy array as a NIfTI file.
- In: data array, 4x4 affine, output path
- Out: none (writes file)

### get_spacing(affine)
Gets voxel spacing in mm from the affine.
- In: 4x4 affine
- Out: spacing array with 3 values

### voxels_to_mm(voxels, affine)
Converts voxel indices to physical mm coordinates.

How it works:
1. Add a 1 to each [i, j, k] so matrix math works with translation.
2. Multiply by the 4x4 affine.
3. Result is [x, y, z] in mm.

- In: (N, 3) voxel points, 4x4 affine
- Out: (N, 3) mm points

### mm_to_voxels(mm_pts, affine)
Inverse of voxels_to_mm.
- In: (N, 3) mm points, 4x4 affine
- Out: (N, 3) voxel-space points (can be fractional)

---

## Registration/stages/load.py

### _load_nifti(path)
Loads one NIfTI file.
- Out: (data, affine)

### _load_nifti_from_bytes(data)
Loads NIfTI from raw bytes (for zip reads).

How it works:
- Wrap bytes in io.BytesIO (in-memory file)
- Let nibabel read from that object
- No need to extract file to disk first

### load_patient(data_dir, patient_id)
Loads all segmentation data for one patient.
Checks zip first, then disk fallback.

- Out: dict with:
  - liver: binary liver mask
  - affine: 4x4 affine from liver file
  - portal_vein: portal/splenic mask or None
  - hepatic_vein: hepatic vessel mask or None
  - segs: dict of Couinaud segment masks 1..8

Note:
- Liver volume is the big memory item.
- Vessel and segment masks are sparse, so they are lighter in practice.

---

## Registration/stages/landmarks.py

### _resolve_label(seg, requested_label, name)
Handles label mismatch issues by finding the best matching nonzero label.

### _ball_cluster(center, voxel_pool, radius, cap)
Gets up to cap voxels inside a radius around center.
If too few are found, radius is expanded.

### extract_landmarks(patient_data, cluster_radius, return_representative_points)
Builds landmark clusters from segments, vessels, and liver boundary.

What this function does exactly:
1. Reads each available structure mask (Couinaud segments, portal vein, hepatic vein, liver).
2. Finds representative center points for each structure.
3. Around each center point, samples a local cluster of voxels.
4. Concatenates all clusters into one landmark array.
5. Optionally also returns representative single points for plotting/debugging.

- Out: landmark array, often about (1100, 3)

Why this matters now:
- In the current rigid pipeline, landmarks mainly support centroid alignment.
- In the old TPS pipeline, dense landmark correspondence mattered much more.

---

## Registration/stages/align.py

### _vox_to_mm(voxels, affine) / _mm_to_vox(mm_pts, affine)
Local wrappers for coordinate conversion.

### _liver_centroid_mm(liver_vol, affine)
Computes liver centroid in mm.

How:
1. Find all liver voxels where value > 0
2. Average coordinates
3. Convert that centroid to mm with affine

Why:
- Different scans can have very different origin offsets.
- Centering by centroid removes that scanner-origin shift.

### prealign(src_landmarks, ref_landmarks, src_affine, ref_affine, ...)
Computes rigid prealignment terms.

What this function does exactly:
1. Converts source landmarks from voxel space to mm space with src_affine.
2. Converts reference landmarks from voxel space to mm space with ref_affine.
3. Computes source liver centroid and reference liver centroid in mm.
4. Re-centers both point sets by subtracting their own centroid.
5. Initializes rotation as identity (R = I).
6. Computes two scale ideas:
   - procrustes_scale from centered landmarks
   - volume_scale from cbrt(ref_volume / src_volume)
7. Stores translation anchors t_src and t_ref.
8. Returns an alignment dict that later warp code can use.

Why this is done this way:
- Centering removes scanner-origin offsets so patients can be compared fairly.
- Rotation is handled later by affine-direction override in align_patient.
- Scale is stored for analysis/experiments, but not applied by current warp functions.

Important notes:
- procrustes_scale is computed but not used.
- alignment["scale"] is stored but not used by warp functions.
- In practice current warps are rigid: translation + rotation.

---

## Atlas/utils.py

Shared helper functions for geometry, metrics, and visualization.

### extract_liver_voxels(vol)
Returns all voxel indices where liver mask > 0.

### extract_surface_mm(liver_vol, affine, downsample)
Finds boundary voxels with Sobel edges, downsamples, converts to mm.

### extract_vessel_mm(vol, affine)
Gets vessel voxel points in mm.

### downsample_voxels_to_mm(voxels, affine, target_points)
Downsamples large voxel sets for plotting speed.

Rule used:
- stride = max(1, N // target_points)

### knn_mean_distance(query_pts, vessel_pts, k)
Core vessel distance function.

What it computes exactly:
- Input query_pts: liver points in mm
- Input vessel_pts: vessel points in mm
- For each liver point, find k closest vessel points
- Return one value per liver point: average distance to those k neighbors

How it computes:
1. Builds cKDTree from vessel_pts so nearest-neighbor queries are fast.
2. Runs tree.query for all liver points in one vectorized call.
3. Handles shape differences for k=1 vs k>1.
4. Takes mean along the neighbor axis.
5. Returns a distance vector aligned with query_pts order.

Why k=5 helps:
- Less sensitive to isolated vessel outliers than k=1
- Better reflects local vessel neighborhood, not a single noisy voxel

### pad_to_common(a, b)
Zero-pads two arrays to same shape so they can be compared safely.

### dice(seg_a, seg_b, label)
Dice overlap metric:
- Formula: 2 * |A intersect B| / (|A| + |B|)
- Range: 0 to 1

### density_to_mesh(vol, level, color, opacity, name, affine)
Builds isosurface mesh from volume using marching cubes,
then maps vertices into mm space with affine.

What this function does exactly:
1. Checks if the volume has values above the selected level.
2. Runs marching cubes to extract a triangle surface at that level.
3. Gets vertices and triangle indices from that surface.
4. Converts vertex coordinates from voxel space to mm space.
5. Packages the result into a Plotly Mesh3d object.

Why this is useful:
- It turns a dense 3D array into a clean surface you can inspect visually.
- Different levels show different confidence shells in the atlas.

### make_slider_layout(n_frames, initial_frame, prefix)
Creates Plotly slider config for slice browsing.

---

## Atlas/registration.py

Handles alignment caching and atlas-space warps.

### _cache_path(cache_dir, atlas_id, patient_id)
Builds path for rigid alignment cache file (.npz).

### save_alignment(alignment, path) / load_alignment(path)
Saves/loads alignment dict fields (R, centroids, affines).

### compute_warp_extents(src_vol, alignment)
Finds min/max voxel bounds after warping source liver into ref space.
Used to size global atlas grid before accumulation.

### forward_warp_mask(src_vol, alignment, ref_shape, global_offset)
Forward maps source liver voxels into atlas grid.

What this function does exactly:
1. Finds all source liver voxels where src_vol > 0.
2. Converts those voxel coordinates to source mm using src_affine.
3. Re-centers points by subtracting t_src.
4. Rotates points with R.
5. Moves points into reference frame by adding t_ref.
6. Converts mm points back to reference voxel coordinates.
7. Rounds to integer voxel indices.
8. Applies global_offset so negative coordinates become valid array indices.
9. Creates output volume and scatters warped voxels into it.
10. Clips indices to valid bounds before writing.

Why this matters:
- This is the core mask warp used for atlas accumulation.
- global_offset is what allows one shared grid across all patients.

### apply_rigid_to_volume(vol, alignment, ref_shape, order)
Inverse-map interpolation warp.

What this function does exactly:
1. Creates full reference voxel grid for ref_shape.
2. Converts reference voxels to mm with ref_affine.
3. Applies inverse rigid transform to map ref mm back to source mm.
4. Converts source mm to source voxel coordinates.
5. Interpolates source volume values with scipy map_coordinates.
6. Reshapes sampled values back to ref_shape.

When to use each order:
- order=0: nearest neighbor (binary masks)
- order=1: trilinear (continuous maps like distances)

Why inverse mapping is used here:
- Every reference voxel gets a sampled value.
- This avoids holes that can happen with naive forward resampling of scalar fields.

### align_patient(patient_id, atlas_id, ...)
Main alignment entry point.

What this function does exactly:
1. Builds patient-specific cache path.
2. If cache exists, loads saved alignment dict.
3. If cache does not exist:
   - loads patient segmentation data
   - extracts landmarks
   - runs prealign
   - saves new alignment cache
4. Computes direction-based rotation using patient affine and canonical direction.
5. Orthonormalizes rotation with SVD so it stays a clean rotation matrix.
6. Stores/update scale from median-volume ratio.
7. Returns final alignment dict.

Important behavior:
- Rotation from affine direction is the effective rotation used.
- Stored scale is still not consumed by current warp functions.

---

## Atlas/liver_atlas.py

### LiverAtlas.build(patient_ids, organ_label)
Builds probabilistic liver atlas in two passes.

Why two passes:
- Need all warped extents first to know global grid size.
- Then can allocate accumulator once and fill it.

What this function does exactly:
1. Initializes reference patient info and runtime containers.
2. Pass 1 loops over patients and computes warp extents only.
3. Saves/reuses extents cache json to avoid recomputing on reruns.
4. Aggregates all extents to get one global min/max bound.
5. Derives global_offset from min bounds so all indices are non-negative.
6. Computes median liver volume for consistent scale normalization metadata.
7. Builds canonical direction matrix from patient direction matrices.
8. Allocates atlas accumulator with final global shape.
9. Pass 2 loops over patients and runs forward_warp_mask.
10. Adds warped mask into accumulator.
11. Computes Dice before and after registration per patient.
12. Stores per-patient surfaces for visualization.
13. Divides accumulator by patient count to create liver_density.

Final output meaning:
- liver_density[x, y, z] is the fraction of patients with liver at that voxel.
- Values are in [0, 1], where higher means more consistent overlap.

### LiverAtlas.save(out_dir) / LiverAtlas.load(out_dir)
Saves/loads atlas density, ref surface, patient count, and point clouds.

### LiverAtlas.visualize_common_basis(point_cap, output_html)
Stage 1 check: overlay warped liver surfaces.
Tighter clustering means better alignment.

### LiverAtlas.visualize_average_liver(thresholds, output_html)
Stage 2: nested isosurfaces (for example 25, 50, 75 percent).

### LiverAtlas.visualize_density_slices(output_html)
Stage 3: axial slice browser of density heatmap.

### LiverAtlas.print_registration_summary()
Prints per-patient Dice before and after registration.

---

## Atlas/vascular_distance.py

### _accum(total, count, values)
NaN-safe in-place accumulation helper.
Only valid values are added.

### VascularDistanceCloud.build(patient_ids)
Builds mean vessel-distance maps in atlas space.

Design choice:
- Compute distances in each patient's native space first.
- Then warp distance volumes to atlas space.
This avoids mixing registration error into distance calculation itself.

What this function does exactly:
1. Initializes running sum and count arrays for portal, hepatic, and combined maps.
2. For each patient, loads liver and vessel masks.
3. Extracts liver points and vessel points in mm.
4. Runs knn_mean_distance for portal and hepatic vessels.
5. Creates combined distance as pointwise min(portal, hepatic).
6. Scatters sparse point distances into dense NaN-filled volumes.
7. Loads rigid alignment for that patient.
8. Warps distance volumes into atlas space with apply_rigid_to_volume(order=1).
9. Samples warped values at consensus liver voxels.
10. Accumulates valid values using _accum (NaN-safe).
11. Repeats for all patients.
12. Divides total by count to get mean maps.

After all patients:
- Mean map = sum / count at each voxel
- Voxels with no valid data remain NaN

### VascularDistanceCloud.visualize(mode, point_cap, output_html)
3D scatter colored by distance values.

### VascularDistanceCloud.visualize_all_modes(output_html)
Side-by-side portal, hepatic, and combined plots.

### VascularDistanceCloud.visualize_distance_slices(mode, output_html)
Axial slice browser for distance map.

### VascularDistanceCloud.visualize_distance_histogram(output_html)
Histogram of distance values across consensus liver voxels.

---

## Segmentation/run_totalsegmentator.py

Runs TotalSegmentator tasks and manages input/output flow.

### find_totalsegmentator_command(custom_path)
Finds executable command form for TotalSegmentator.

### get_completed_ids(out_zip)
Finds patient IDs already completed in output zip.
Uses set lookups for fast resume behavior.

### _run_roi_subset / _run_liver_segments / _run_liver_vessels
Runs the three required TotalSegmentator task groups.

### _collect_outputs(tmp_dir)
Finds expected output files in temp folder.

### process_one_patient(patient_id, ct_zip, out_zip, cmd, work_dir)
Per-patient pipeline:
1. Extracts one CT from input zip to temp workspace.
2. Runs roi_subset model for liver + portal/splenic structures.
3. Runs liver_segments model for Couinaud segments.
4. Runs liver_vessels model for hepatic vessel structures.
5. Collects expected output files.
6. Writes outputs to final zip under patient folder.
7. Cleans temporary files in finally block.

Why this design works:
- Only one CT is unpacked at a time, so disk usage stays controlled.
- finally cleanup makes crash/restart behavior safe.
- get_completed_ids allows resume without reprocessing finished patients.

---

## Validation/dataset_loader.py

### load_patient_ids(csv_path, gender, min_voxels, max_voxels, exclude_ids)
Core function. Reads one reviewed CSV and returns a filtered sorted list of patient ID strings.
- gender filter: "M" or "F" (case-insensitive)
- min/max voxel filters drop patients with liver too small or too large
- exclude_ids: explicit skip list
- normalizes IDs like s0004 -> 0004 automatically

### load_cohort(data_dir, cohort, gender, min_voxels, exclude_ids, use_validated_ids)
Convenience wrapper. Points at Data/Dataset_Reviewed/ and picks the right CSV by name
("all", "male", or "female"). When use_validated_ids=True (the default), it intersects
the CSV results with usable_patient_ids.txt so any patient that failed validation is
silently dropped. This is the function liver_atlas.py and vascular_distance.py call.

### print_cohort_summary(data_dir)
Prints a quick table: counts, gender split, and voxel stats for all three CSV files.
Useful sanity check before building an atlas.

---

## Validation/validate_dataset.py

This is the script that produces usable_patient_ids.txt and validation_report.csv.
Run it once before building any atlas. It can read straight from the zip (default)
or from an extracted disk directory.

Checks it runs on every patient:
1. Required files present (liver, portal_vein, liver_vessels)
2. Masks are non-empty (nonzero voxel count above threshold)
3. Liver is not implausibly small (below MIN_LIVER_VOXELS = 50,000)
4. Affine is well-formed (no NaN/Inf, no zero-length axes)
5. At least 6 of 8 Couinaud segments are present and non-empty
6. No NaN or Inf values in the voxel data itself

### class PatientReport
Dataclass that holds all results for one patient: pass/fail status, issue list,
voxel counts, segment count, liver spacing. Has a .fail() method that marks
the patient as unusable and records the reason.

### validate_patient_disk(patient_dir)
Validates one patient from a folder on disk. Runs all checks above.

### validate_patient_zip(zf, patient_id, all_entries, zip_layout)
Validates one patient reading directly from the zip without extracting.
Handles two zip layouts: raw TotalSegmentator zip (s0004/segmentations/...)
and processed output zip (0004/liver.nii.gz).

### discover_ids_from_zip(zf)
Scans zip entries to find all patient IDs. Returns a sorted list.

### detect_zip_layout(all_entries)
Checks zip entry paths to determine which layout is in use.

### discover_ids_from_disk(data_dir)
Finds all patient subdirectories on disk that have at least a liver.nii.gz.

### print_summary(reports)
Prints console summary: pass/fail counts, failure reasons grouped by type,
warning list, and all usable IDs printed in rows of 10.

### save_csv(reports, output_path)
Writes the full per-patient report to validation_report.csv with columns:
patient_id, status, liver_voxels, portal_voxels, hepatic_voxels, n_segments,
liver_spacing, issues.

### save_usable_ids(reports, output_path)
Writes usable_patient_ids.txt: one passing patient ID per line.
This is the file that dataset_loader.py reads downstream.

---

## Validation/visualize_registration.py

Interactive 3D plot showing the before/after of rigid registration for two patients.
Reads directly from Data/segmentations.zip so nothing needs to be extracted first.

Left panel: reference liver vs source liver before alignment (unregistered).
Right panel: reference liver vs transformed source liver after alignment.

REF_ID and SRC_ID at the top of the file control which patients are shown.
TRANSFORMED_PATH can be set to a pre-saved warped NIfTI if you have one;
if left as None, only the before panel renders.

### _load_nifti_from_zip(zf, patient_id, filename)
Reads a .nii.gz entry from the zip, gzip-decompresses it in memory,
and returns a nibabel image without ever touching disk.

### load_surface_from_zip(zf, patient_id)
Loads the liver mask from zip, pads it, runs marching cubes to get a triangle mesh,
scales vertices to mm using voxel zooms. Returns (verts, faces) or None if empty.

### load_surface_from_file(seg_path)
Same as above but reads from a loose file on disk.
Used for the transformed output if TRANSFORMED_PATH is set.

### mesh(verts, faces, color, opacity, name)
Wraps verts and faces into a Plotly Mesh3d trace.

### main()
Opens the zip, loads ref and src, optionally loads transformed,
builds 1 or 2 panel Plotly figure, calls fig.show().

---

## Registration/Config.py

All tunable pipeline parameters in one place.
If you need to change a threshold or filename, change it here rather than
hunting through individual stage files.

Key settings:
- DATA_DIR, OUTPUT_DIR: where data and outputs live
- ORGAN_LABEL: voxel label for liver in TotalSegmentator masks (default 1)
- PORTAL_VEIN_CANDIDATES: filename list to handle TotalSegmentator naming changes
- CLUSTER_RADIUS, CLUSTER_CAP: landmark sampling parameters
- ALPHA: TPS regularization weight (legacy, not used in rigid pipeline)
- TRANSFORM_BATCH_SIZE: memory control for TPS transform (legacy)

---

## Registration/Run.py

Legacy entry point for the old TPS registration pipeline.
Runs the full original pipeline: load -> landmarks -> align -> TPS -> evaluate.
Not used in the current atlas build. The atlas now calls align_patient() directly.

### run_stage(name, fn, checkpoint, skip_if_done, *args)
Helper that runs one stage with timing, logging, and checkpointing.
If skip_if_done=True and a checkpoint exists, loads saved result instead of recomputing.

---

---
## Legacy / Unused Code

These files are still in the repo but are not called by the current pipeline.
Left here for reference in case the approach ever comes back.

---

## Registration/utils/Checkpoint.py

Old pickle-based checkpoint store. Was used before the .npz alignment cache existed.
Not used in current pipeline.

### class Checkpoint
Simple key-value store using pickle files on disk.

### Checkpoint.__init__(directory)
Creates checkpoint directory if needed.

### Checkpoint.exists(name)
Checks if a named checkpoint file exists.

### Checkpoint.save(name, value)
Writes value to disk via pickle.

### Checkpoint.load(name)
Loads value from pickle file.

### Checkpoint.clear(name) / Checkpoint.clear_all()
Deletes one checkpoint or all checkpoints.

---

## Registration/stages/tps.py

Thin Plate Spline warp code. The pipeline originally used TPS for non-rigid alignment.
Replaced by the current rigid-only (rotation + translation) approach.
Not used in current pipeline.

### _tps_kernel(r)
TPS radial basis function in 3D: U(r) = r.

### _compute_kernel_matrix(points)
Builds dense NxN pairwise-distance kernel matrix.
This is one of the expensive parts of TPS.

### fit_tps(src_landmarks, ref_landmarks, alpha)
Fits TPS by solving linear system L * coeffs = Y.

### transform_points(points, coefficients, source_points, batch_size)
Applies TPS warp to points in batches to control memory.

---

## Registration/stages/evaluate.py

Evaluation helpers written for the TPS pipeline. Not called anymore now that
rigid alignment handles everything.
Not used in current pipeline.

### _apply_prealignment(voxels, alignment)
Applies rotation + translation to voxel coordinates.

### _dice(seg_a, seg_b, label)
Computes Dice overlap after padding arrays to a common shape.

### transform_and_evaluate(src_data, alignment, tps_coefficients, ref_data)
Applies TPS transform and evaluates overlap with reference.

---

---
## Research Paper Scripts

These scripts live in research_paper/scripts/ and are separate from the main pipeline.
They all read from already-built outputs (atlas files, alignment cache, stats CSV)
and produce figures and tables for the paper. You run them after the atlas is built.

---

## research_paper/scripts/alignment_statistics.py

Computes the Section 4 evaluation statistics used in the paper.
Reads from the outputs/reg_cache/ folder (the .npz and .json files written by the atlas).
Does not re-run any registration, just analyzes what is already cached.

Outputs two files:
- outputs/reg_cache/alignment_stats_summary.json: aggregate stats (mean, median, std, percentiles)
- outputs/reg_cache/alignment_stats_rows.csv: one row per patient per metric

Three metrics it computes:

1. Centroid displacement before alignment (mm)
   How far apart are the liver centroids of source and reference before any alignment?
   Loaded from rigid_*.npz cache files (t_src, t_ref fields).

2. Effective rotation angle (degrees)
   What rotation angle does the canonical->patient affine transform produce?
   Computed from extents_*.json direction matrices. Uses SVD to ensure clean rotation.

3. Dice similarity before/after alignment
   Compares source liver to reference liver before alignment (resized to match),
   then re-runs forward_warp_mask and computes Dice on the result.
   Skippable with --skip-dice flag since it takes a long time.

### _rotation_angle_deg(R)
Gets the principal rotation angle from a 3x3 rotation matrix using arccos((trace(R)-1)/2).

### _orthonormalize(R)
Projects a near-rotation matrix onto SO(3) via SVD so it stays a valid rotation.

### _describe(values)
Returns a dict of summary stats for a 1D array: n, mean, median, std, min, max, p90, p95, p99.

### _load_usable_ids(path)
Loads usable_patient_ids.txt into a set for fast membership checks.

### compute_centroid_before(cache_dir, usable_ids)
Scan all rigid_*.npz files, compute ||t_src - t_ref|| for usable patient pairs.
Deduplicates by normalized (atlas_id, patient_id) pair to avoid double counting.

### compute_effective_rotation(cache_dir, usable_ids)
For each extents_*.json, rebuild the canonical direction matrix,
then compute rotation angle for each patient relative to canonical.

### compute_dice_statistics(cache_dir, data_dir, usable_ids)
Loads each patient's liver, resizes source to ref space for before-Dice,
then runs forward_warp_mask and computes after-Dice.
This is the slow one; use --skip-dice to skip it.

### collect_qualitative_artifacts(project_root)
Just scans the outputs folder for existing .html files and returns their paths
for Section 4.4 figure references in the paper.

### write_outputs(cache_dir, summary, rows)
Writes the JSON summary and CSV rows files.

---

## research_paper/scripts/generate_research_paper_figures.py

Produces all the static figure files for the paper:
- Dice boxplot (before vs after alignment)
- Centroid displacement histogram
- Effective rotation histogram
- Compact versions of the atlas visualizations (average liver, density slices, distance scatter, histogram)

Reads from alignment_stats_rows.csv for the metric plots.
Reads from atlas output folders for the visualization plots.
Saves PNGs to research_paper/outputs/figures/ and HTMLs alongside the atlas files.

### compact_stage2_average_liver(atlas_dir, cohort_label)
Loads atlas_liver_density.nii.gz, applies Gaussian smoothing,
renders nested isosurfaces at 25/50/75% with a cleaner layout than the default.

### compact_stage3_density_slices(atlas_dir, cohort_label)
Loads density volume and builds an axial slice browser cropped to the liver extent.

### compact_vdc_combined(atlas_dir, cohort_label)
Loads vascular distance point cloud and renders a 3D scatter colored by distance.
Downsamples to 50k points for browser performance.

### compact_vdc_histogram(atlas_dir, cohort_label)
Histogram of all valid combined distance values with mean/median lines.

### make_dice_boxplot()
Side-by-side boxplot of Dice before vs after alignment for the paper.
Loads from alignment_stats_rows.csv.

### make_alignment_histograms()
Two-panel histogram: centroid displacement (mm) and effective rotation (degrees).
Clips centroid axis at p99 so tail outliers do not squash the main distribution.

---

## research_paper/scripts/generate_slice_gifs.py

Exports the axial slice browsers into four animated GIFs for the paper:
- Male atlas density slices
- Female atlas density slices
- Male vascular-distance slices
- Female vascular-distance slices

Reads atlas_liver_density.nii.gz for the density GIFs.
Reads vdc_all_voxel_idx.npy and vdc_full_dist_combined.npy for the VDC GIFs.
Saves the GIFs into research_paper/outputs/figures/.

### _crop_to_nonzero_bounds(vol)
Finds the nonzero bounding box of a 3D volume and crops to that extent.
Used so density GIFs do not include large empty margins.

### _make_density_volume(atlas_dir)
Loads atlas_liver_density.nii.gz and returns the cropped density volume.

### _make_vdc_volume(atlas_dir)
Rebuilds a dense cropped 3D volume from sparse VDC voxel indices and values.
Only valid combined-distance entries are inserted into the volume.

### _render_slice_frame(arr2d, cmap_name, vmin, vmax, colorbar_label, title)
Renders one axial slice with Matplotlib using origin="lower" so the GIF orientation
matches the intended anatomical view. Includes a side colorbar in every frame.

### _write_gif(vol, out_path, cmap_name, colorbar_label, title_prefix, fps, reverse_z=False)
Loops through the axial dimension, renders each slice frame, then saves a GIF.
Uses a fixed min/max scale across all frames so the color meaning stays consistent.

### generate_all_gifs(male_dir, female_dir, out_dir, fps)
Top-level orchestration function. Builds and writes all four GIFs in one call.

---

## research_paper/scripts/visualize_vessel_comparison.py

Quick diagnostic script. Shows portal vein vs hepatic vessels side by side for a single patient
using marching cubes so you can visually compare the two segmentation sources.

Left panel: portal_vein_and_splenic_vein (from roi_subset task) + liver surface.
Right panel: liver_vessels (from liver_vessels task) + liver surface.

Useful for checking segmentation quality or explaining the difference in the paper.

### _mesh_from_mask(mask, affine, color, opacity, name)
Runs marching cubes on a binary mask and maps vertices to mm space using the affine.
Returns a Plotly Mesh3d trace or None if the mask is empty.

### visualize_vessel_comparison(patient_id, data_dir)
Loads one patient from the zip, builds all four meshes (two livers, portal, hepatic),
assembles a two-panel Plotly figure, and calls fig.show().
