# AI Use Disclosure 
## CIST 8950 — Spring 2026 Capstone — Tristan Jones

---

## 1. AI Tools Used

| Tool | Version / Interface | Purpose |
|------|-------------------|---------|
| Claude (Anthropic) | claude.ai (Claude Sonnet) | Primary coding assistant throughout project |
| GitHub Copilot (Microsoft) | VS Code extension | Inline code suggestions during editing |

---

## 2. Summary of AI Assistance by File

### Active Pipeline & Research Code (In Use)

#### `Segmentation/run_totalsegmentator.py`
**Original prompt (paraphrased):** *"Can you help me create a way to just grab these certain segmentations liver, portal vein, hepatic vessels, Couinaud segments and output them to files from my CT data? Something that wraps TotalSegmentator automatically."*

**What the student designed:**
- Decision to use TotalSegmentator as the segmentation backbone
- Selection of which anatomical structures to extract (liver, portal vein, hepatic vessels, Couinaud segments 1–8)
- Decision to process one patient at a time and write to a zip file to save disk space
- Decision to skip patients already in the output zip (resume logic)


**What Claude assisted with:**
- Implementation of the subprocess wrapper calling TotalSegmentator CLI
- Zip file read/write logic
- Temp directory management and CT deletion after segmentation
- Error handling and progress printing

**Student estimate:** 50% student-designed, 50% AI-assisted implementation

---

#### `Registration/stages/load.py`
**Original prompt (paraphrased):** *"Help me write a function that loads all the segmentation files for one patient liver, portal vein, hepatic vessels, and all 8 Couinaud segments from a zip file."*

**What the student designed:**
- The data structure (dict with patient_id, liver, affine, portal_vein, hepatic_vein, segs keys)
- Decision to support both zip and disk sources transparently
- Decision to make vascular structures optional (warn but don't crash if missing)

**What Claude assisted with:**
- nibabel zip-reading implementation (FileHolder + BytesIO pattern)
- Logging structure
- Portal vein candidate filename fallback logic

**Student estimate:** 40% student-designed, 60% AI-assisted implementation

---

#### `Registration/stages/landmarks.py`
**Original prompt (paraphrased):** *"I want to extract anatomical landmark clusters from the liver Couinaud segment centroids, portal vein branch points, hepatic vein branches, and liver geometry tips. Instead of single points I want clusters of 50 points around each location."*

**What the student designed:**
- The four-source landmark strategy (segments, portal vein, hepatic veins, geometry tips)
- The ball_around cluster approach (50 points per landmark rather than single point)
- The fallback strategy when segments have too few voxels
- The portal vein branch decomposition (right/left/superior)
- Designed functions outlined how they should look

**What Claude assisted with:**
- Implementation of the ball_around helper function
- Hepatic vein decomposition into right/middle/left/IVC confluence
- Extremal tip extraction code

**Student estimate:** 70% student-designed, 30% AI-assisted implementation

---

#### `Registration/stages/align.py`
**Original prompt (paraphrased):** *"The Procrustes alignment is giving wrong rotations because the segment centroids are unreliable. I want to align using the affine direction matrices instead, since all the CTs are in standard RAS orientation. Use the mean direction across all patients as the canonical orientation."*

**What the student designed:**
- The key insight that all TotalSegmentator CTs are in standard RAS orientation
- The decision to use affine direction matrices instead of landmark-based Procrustes
- The centroid normalization approach to handle large z-offset differences between patients
- The canonical direction concept (mean population orientation)
- Designed functions outlined how they should look

**What Claude assisted with:**
- The mm-space centroid computation from liver volumes
- Implementation of _vox_to_mm and _mm_to_vox helpers

**Student estimate:** 70% student-designed, 30% AI-assisted implementation

---

#### `Atlas/registration.py`
**Original prompt (paraphrased):** *"I need a caching wrapper around the alignment compute it once, save to .npz, reload on subsequent runs. It also needs to apply median volume scaling and canonical direction rotation on top of the cached values."*

**What the student designed:**
- The two-level caching strategy (base alignment cached, scale + canonical direction applied at runtime)
- The decision to compute scale as cube root of median/patient volume ratio
- The decision to override rotation with affine-based canonical direction rather than using cached Procrustes rotation
- Designed functions outlined how they should look

**What Claude assisted with:**
- npz save/load implementation
- forward_warp_mask and apply_rigid_to_volume functions
- compute_warp_extents function

**Student estimate:** 55% student-designed, 45% AI-assisted implementation

---

#### `Atlas/liver_atlas.py`
**Original prompt (paraphrased):** *"I want to build a probabilistic liver atlas overlay many livers in a common space, find the average. Two-pass approach: first pass compute extents and save to JSON, second pass warp all patients into a shared global grid so no voxels get clipped."*

**What the student designed:**
- The two-pass atlas build concept
- The JSON extents file approach for tracking per-patient bounding boxes
- The global bounding box computation to avoid voxel clipping
- The decision to use forward warping (not inverse) for binary masks
- The consensus liver threshold (≥50%) for the average liver definition
- The volume scale normalization to median population volume
- Designed functions outlined how they should look

**What Claude assisted with:**
- Implementation of the two-pass build loop
- Global offset computation

**Algorithm source (Procrustes/similarity transform):** Dryden, I.L. & Mardia, K.V. (1998). *Statistical Shape Analysis*. Wiley, Chichester.

**Student estimate:** 80% student-designed, 20% AI-assisted implementation

---

#### `Atlas/vascular_distance.py`
**Original prompt (paraphrased, from vascular_distance_cloud.py comment):**
*"Basically I designed the algorithm and then prompted Claude to help me flesh this out more. I really primarily need it for creating the visualizations but the core kind of backbone of finding the nearest point through KNN methods was my own."*

Later updated with: *"I want the distances computed in each patient's native space first, then warped into atlas space, sampled on the atlas consensus liver."*

**What the student designed:**
- The core algorithmic concept: compute distances in native space, warp to atlas, accumulate
- The KNN mean distance approach (robustness over single nearest neighbor)
- The decision to use the atlas consensus liver as the sampling grid
- The electron cloud visualization concept
- The clinical motivation (tumor proximity to vasculature)
- Designed functions outlined how they should look

**What Claude assisted with:**
- Implementation of the VascularDistanceCloud class
- NaN-safe accumulation helper
- Plotly 3D scatter, slice browser, histogram visualizations
- Integration with extents JSON for global grid consistency

**Algorithm source (KD-tree / KNN):** scipy.spatial.cKDTree — implemented via SciPy. Virtanen, P. et al. (2020). SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. *Nature Methods*, 17, 261–272.

**Student estimate:** 55% student-designed, 45% AI-assisted implementation

---

#### `Atlas/utils.py`
**Original prompt (paraphrased):** *"Refactor the shared helper functions out of liver_atlas and vascular_distance into a shared utils file."*

**What the student designed:**
- Decision to separate utilities from pipeline logic
- Selection of which functions belong in shared utils vs staying in their respective files
- Already previously worked to of these functions before this was just bringing it over in the pipeline some are very old functions
- Designed functions outlined how they should look

**What Claude assisted with:**
- Pipeline implementation of extract_liver_voxels, extract_surface_mm, extract_vessel_mm, knn_mean_distance, downsample_voxels_to_mm, pad_to_common, dice, density_to_mesh, make_slider_layout

**Student estimate:** 60% student-designed, 40% AI-assisted implementation

---

#### `Validation/validate_dataset.py`
**Original prompt (paraphrased):** *"Scan the segmentations zip for missing or bad files empty masks, broken affines, insufficient Couinaud segments — and output a report CSV and a list of usable patient IDs."*

**What the student designed:**
- The validation criteria (what constitutes a bad segmentation)
- The output format (CSV report + usable ID text file)
- CSV writing
- My idea to ensure that there should be validation steps 

**What Claude assisted with:**
- Implementation of the zip scanning logic
- Per-patient validation checks


**Student estimate:** 60% student-designed, 40% AI-assisted implementation

---

#### `Validation/dataset_loader.py`
**Original prompt (paraphrased):** *"Write a function that reads the reviewed CSV files and returns a filtered list of patient IDs by cohort."*

**What the student designed:**
- The reviewed CSV structure and patient review process (done manually by student)
- The cohort filtering requirements

**What Claude assisted with:**
- implementation of function 

**Student estimate:** 60% student-designed, 40% AI-assisted implementation

---

### Research Paper Scripts

**Files:** `alignment_statistics.py`, `generate_research_paper_figures.py`, `generate_slice_gifs.py`, `visualize_vessel_comparison.py`

These scripts live in `research_paper/scripts/` and are separate from the main pipeline. They read from already-built outputs and produce figures and statistics for the paper.

**Original prompt (paraphrased):** *"Need to make this plotly plot into something that is readible with the ACM format in one column currently its too big can you help me to take what we already have from previous works (files) and then implement that here for resizing. Also need to just output statistics here."*

**Original prompt (paraphrased):** *"Can you turn this slider plot into a Gif for the powerpoint? Just use the same logic/parameters as before."*

**What the student designed:**
- Which metrics and figures are needed for the paper and what they should show
- The decision to read from cache rather than re-run registration
- The output formats (JSON summary, CSV rows, PNGs, HTMLs, GIFs)
- The decision to use GIFs for showing 3D volumetric data in presentaiton
- The diagnostic purpose of the vessel comparison script

**What Claude assisted with:**
- Implementation of scanning and parsing .npz cache files
- Matplotlib figure layout, styling, and GIF assembly
- Compact HTML figure generation for atlas density and VDC

**Student estimate:** 70% student-designed, 30% AI-assisted implementation

---

### Legacy / Unused (Kept for Reference)

These files are still in the repo but are not called by the current pipeline.
Left here for reference in case the approach ever comes back.

---

#### `Registration/utils/Checkpoint.py`
**Original prompt (paraphrased):** *"Create a checkpoint/caching mechanism to save intermediate stage results during registration."*

**What the student designed:**
- The concept of a persistent key-value store for checkpointing

**What Claude assisted with:**
- Full pickle-based implementation of save/load/exists/clear methods

**Student estimate:** 30% student-designed, 70% AI-assisted implementation

---

#### `Registration/stages/tps.py`
**Original prompt (paraphrased):** *"Help me implement a 3D Thin Plate Spline registration. I need the kernel function, the system matrix, and batched point transformation."*

**What the student designed:**
- Decision to use TPS for non-rigid registration
- The batched transformation approach to handle memory constraints (documented in code comments with date 3/15/26)
- Regularization parameter alpha

**What Claude assisted with:**
- TPS kernel matrix formulation
- Linear system assembly and solve
- Batch processing loop

**Algorithm source:** Bookstein, F.L. (1989). Principal Warps: Thin-Plate Splines and the Decomposition of Deformations. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 11(6), 567–585.

**Student estimate:** 45% student-designed, 55% AI-assisted implementation

---

#### `Registration/stages/evaluate.py`
**Original prompt (paraphrased):** *"Create evaluation functions to assess registration quality compute Dice overlap, apply transforms, measure residuals."*

**What the student designed:**
- The concept of post-registration evaluation metrics
- Which metrics to compute (Dice, residuals)
- Implementation of Dice computation with padding

**What Claude assisted with:**
- Pre-alignment transform application
- TPS transform + Dice evaluation integration

**Student estimate:** 65% student-designed, 35% AI-assisted implementation

---

## 3. What Was Entirely Student Work

- **Clinical motivation and problem framing** — the idea of mapping tumor proximity to vasculature using a population atlas came from discussions with Dr. Ghersi and the student's own research
- **Data curation** — manually reviewing 303 patients, creating the reviewed CSV files, flagging problematic cases in liver_flagged.csv
- **Algorithm selection decisions** — choice of TPS → rigid → affine-direction alignment progression as understanding of the problem deepened
- **Debugging and iteration** — identifying that 90–150 degree rotations were wrong, diagnosing the centroid z-offset problem, identifying the forward vs inverse warp distinction
- **All code comments and inline documentation** in student's voice (marked distinctly from docstrings)

---

## 4. Key External References

### TotalSegmentator
Wasserthal, J., Breit, H.-C., Meyer, M.T., Pradella, M., Hinck, D., Sauter, A.W., Heye, T., Boll, D., Cyriac, J., Yang, S., Bach, M., & Segeroth, M. (2023). TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images. *Radiology: Artificial Intelligence*, 5(5), e230024. https://doi.org/10.1148/ryai.230024

### Couinaud Liver Segments
Couinaud, C. (1957). *Le Foie: Études Anatomiques et Chirurgicales*. Masson, Paris.

### Thin Plate Splines
Bookstein, F.L. (1989). Principal Warps: Thin-Plate Splines and the Decomposition of Deformations. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 11(6), 567–585.

### Procrustes / Similarity Transform
Dryden, I.L. & Mardia, K.V. (1998). *Statistical Shape Analysis*. Wiley, Chichester.

### NIfTI File Handling
Brett, M., Hanke, M., et al. (2020). NiBabel: Access a cacophony of neuro-imaging file formats. https://nipy.org/nibabel/

### KD-Tree / KNN Distance
Virtanen, P. et al. (2020). SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. *Nature Methods*, 17, 261–272. https://doi.org/10.1038/s41592-019-0686-2

### Marching Cubes (surface mesh generation)
Lewiner, T., Lopes, H., Vieira, A.W., & Tavares, G. (2003). Efficient Implementation of Marching Cubes' Cases with Topological Guarantees. *Journal of Graphics Tools*, 8(2), 1–15. (Implemented via scikit-image `marching_cubes`)

### Liver Segment Segmentation Model (TotalSegmentator subtask)
Referenced in TotalSegmentator codebase: https://link.springer.com/chapter/10.1007/978-3-030-32692-0_32

### Hepatic Vessels Segmentation Model (TotalSegmentator subtask)
Referenced in TotalSegmentator codebase: https://arxiv.org/abs/1902.09063

---

## 5. Note on Collaboration Style

The interaction with Claude was iterative and conversational over many sessions. The student directed all high-level decisions, caught errors in AI-generated code, and frequently overrode or redirected Claude's suggestions. The AI served as an implementation accelerator and debugging partner, not as an autonomous agent. All submitted code has been read, understood, and is explainable by the student.