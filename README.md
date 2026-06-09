# Probabilistic CT Liver Atlas — Tristan Jones
**Biomedical Informatics Capstone · University of Nebraska Omaha · Spring 2026**

> **Note:** This is a public-facing version of the capstone repository, cleaned and packaged for GitHub. It reflects the final state of the project but is not the exact working directory used during development.

An end-to-end pipeline for constructing a sex-stratified probabilistic liver atlas and population-level distance-to-vasculature maps from clinical CT volumes.  Rigid alignment improved median Dice from **0.146 → 0.724** across 261 subjects (154 male, 107 female).

---

## Paper

Covers the full methodology, alignment pipeline design, atlas construction, distance-to-vasculature mapping, and evaluation across 261 subjects.

[`Final_Research_Report_Tristan_Jones.pdf`](https://github.com/Tristonious/liver-atlas-capstone/blob/main/research_paper/Final_Research_Report_Tristan_Jones.pdf)

---

## Quick Start

```bash
# 1 — Segment CT scans (CPU; add --gpu for CUDA)
python -m Segmentation.run_totalsegmentator

# 2 — Validate dataset
python -m Validation.validate_dataset --source zip --zip-path Data/segmentations.zip

# 3 — Build probabilistic atlas (male + female)
python -m Atlas.liver_atlas

# 4 — Compute vascular distance cloud
python -m Atlas.vascular_distance

# 5 — Alignment statistics (for paper)
python -m Validation.alignment_statistics

# 6 — (Optional) pairwise TPS registration
python -m Registration.Run --ref 0004 --src 0010 --skip-if-done
```

Run all commands from the project root (`liver-atlas-capstone/`).

---

## Repository Structure

```
liver-atlas-capstone/
├── Registration/               # CT-to-CT registration pipeline
│   ├── Config.py               # All tunable parameters
│   ├── Run.py                  # CLI entry point
│   ├── stages/
│   │   ├── load.py             # Stage 1 — load NIfTI segmentations
│   │   ├── landmarks.py        # Stage 2 — anatomical landmark clusters
│   │   ├── align.py            # Stage 3 — Procrustes rigid alignment (mm space)
│   │   ├── tps.py              # Stage 4 — Thin Plate Spline (legacy, not used by Atlas)
│   │   └── evaluate.py         # Stage 5 — apply transform + Dice (legacy)
│   └── utils/
│       ├── Checkpoint.py       # Pickle-based stage checkpointing
│       └── Nifti.py            # Shared NIfTI helpers (voxels_to_mm, mm_to_voxels)
│
├── Atlas/                      # Probabilistic liver atlas
│   ├── liver_atlas.py          # Average liver + density accumulation
│   ├── vascular_distance.py    # Distance-to-vasculature electron cloud
│   ├── registration.py         # Rigid alignment wrapper + .npz cache
│   └── utils.py                # Geometry helpers + Plotly building blocks
│
├── Segmentation/
│   └── run_totalsegmentator.py # Batch TotalSegmentator wrapper
│
├── Validation/
│   ├── validate_dataset.py     # Scan segmentations.zip for silent failures
│   ├── dataset_loader.py       # Load reviewed patient CSVs with filters
│   ├── visualize_registration.py # Before/after 3-D mesh viewer (Plotly)
│   ├── usable_patient_ids.txt  # 263 validated patient IDs
│   └── validation_report.csv   # Per-patient QC report
│
├── Data(without CT)/                       # ⚠ Not tracked — place data here
│   └── Dataset_Reviewed/       # reviewed_all/male/female.csv go here
│
├── outputs/
│   ├── atlas/                  # Atlas density volumes + HTML visualizations
│   └── reg_cache/              # Cached .npz alignments + alignment stats
│
├── research_paper/
│   ├── Final_Research_Report_Tristan_Jones.pdf
│   └── figures/                # All paper figures and GIFs
│
└── docs/
    ├── PIPELINE_FILE_BREAKDOWN.md   # File-by-file technical documentation
    └── AI_Use_Disclosure.md         # AI use disclosure for all files
```

---

## Module Summary

### Registration

| File | Role | AI Assist |
|------|------|-----------|
| `Config.py` | All tunable parameters in one place | 0% |
| `Run.py` | CLI pipeline orchestrator (stages 1–5) | 0% |
| `stages/load.py` | Load NIfTI masks from zip or disk; portal vein fallback | 60% |
| `stages/landmarks.py` | Couinaud + vascular + geometry landmark clusters | 30% |
| `stages/align.py` | Procrustes rigid alignment in mm space; centroid normalization | 30% |
| `stages/tps.py` | Thin Plate Spline fit + transform *(legacy)* | 55% |
| `stages/evaluate.py` | Apply transform; Dice before/after *(legacy)* | 35% |
| `utils/Checkpoint.py` | Pickle-based save/load/exists for stage outputs *(legacy)* | 70% |
| `utils/Nifti.py` | load/save NIfTI, voxels↔mm conversion | 0% |

### Atlas

| File | Role | AI Assist |
|------|------|-----------|
| `liver_atlas.py` | Two-pass atlas build; density accumulation; HTML outputs | — |
| `vascular_distance.py` | Native-space kNN distance maps warped to atlas | — |
| `registration.py` | Rigid alignment wrapper; .npz cache; forward warp | 45% |
| `utils.py` | Padding, Dice, surface extraction, KNN, Plotly helpers | 40% |

### Segmentation

| File | Role | AI Assist |
|------|------|-----------|
| `run_totalsegmentator.py` | Batch TS wrapper; three subtasks; one CT on disk at a time | 50% |

### Validation

| File | Role | AI Assist |
|------|------|-----------|
| `validate_dataset.py` | QC scan — empty masks, voxel counts, missing files | — |
| `dataset_loader.py` | Read reviewed CSVs; gender/voxel filters; validated ID intersection | 40% |
| `visualize_registration.py` | Marching-cubes 3-D before/after viewer | 100% |

---

## Key Results

| Metric | Value |
|--------|-------|
| Analytic cohort | 261 subjects (154 M, 107 F) |
| Median Dice — pre-alignment | 0.146 |
| Median Dice — post-alignment | 0.724 |
| Centroid displacement — pre | 296.98 mm mean (131.18 mm median) |
| Centroid displacement — post | 0 mm (by construction) |
| Median effective rotation | 0.097° |
| Male atlas consensus voxels | 467,894 |
| Female atlas consensus voxels | 389,109 |
| Male mean dist-to-vessel | 18.29 mm (median 17.50 mm) |
| Female mean dist-to-vessel | 16.48 mm (median 15.80 mm) |

---

## Key Design Decisions

**Rigid-only alignment** — Distances are computed in each patient's native space then rigidly warped into atlas space. This avoids compounding TPS registration error into the distance values. TPS is preserved in `Registration/stages/tps.py` for evaluation but is not called by the atlas pipeline.

**No isotropic scaling** — A similarity transform (Procrustes + scale) was evaluated and intentionally omitted to preserve natural liver volume variation across the population as a meaningful anatomical signal.

**Shared .npz cache** — `outputs/reg_cache/rigid_{atlas_id}_{patient_id}.npz` stores the Procrustes alignment for each patient. Both `liver_atlas.py` and `vascular_distance.py` read from the same cache, so registration is computed only once per patient.

**Forward warp with global offset** — A two-pass build computes a global bounding box across all patients before warping any masks, preventing any patient's liver from being clipped at grid boundaries.

**`LOAD_EXISTING` flag** — Set to `True` in each Atlas script after the first run to reload saved outputs without re-running anything.

---

## Data Setup

Place the following in `Data/` before running (not tracked by git):

```
Data(without CT)/
  Totalsegmentator_dataset_v201.zip   # raw CT dataset (not tracked)
  segmentations.zip                   # generated by Segmentation step (not tracked)
  Dataset_Reviewed/
    reviewed_all.csv                  # full reviewed cohort (patient_id, gender, voxel_count)
    reviewed_male.csv                 # male patients
    reviewed_female.csv               # female patients
    liver_flagged.csv                 # excluded cases (truncation, tumors, etc.)
```

---

## Dependencies

```bash
pip install nibabel numpy scipy scikit-image plotly totalsegmentator
```

Python 3.10+ recommended. GPU strongly recommended for TotalSegmentator (`--gpu`).

---

## Citations and Licenses

This pipeline is built on data and models from the following works. Please cite all that apply.

---

**TotalSegmentator** — Wasserthal, J. et al., "TotalSegmentator: Robust segmentation of 104 anatomic structures in CT images," *Radiology: Artificial Intelligence*, vol. 5, no. 5, e230024, 2023.

This paper describes both the dataset and the segmentation toolkit used in this project.

Dataset license: **Creative Commons Attribution 4.0 International (CC BY 4.0)**
Toolkit license: **Apache 2.0**
Dataset source: https://zenodo.org/records/10047292

---

**nnU-Net** *(backbone architecture underlying TotalSegmentator)*
> Isensee, F. et al., "nnU-Net: A self-configuring method for deep learning-based biomedical image segmentation," *Nature Methods*, vol. 18, no. 2, pp. 203–211, 2021.

---

**liver_vessels subtask** *(hepatic vessel + tumor segmentation model)*
> Simpson, A. L. et al., "A large annotated medical image dataset for the development and evaluation of segmentation algorithms," arXiv:1902.09063, 2019.

---

**liver_segments subtask** *(Couinaud segment model)*
> Tian, J. et al., "Automatic Couinaud segmentation from CT volumes on liver using GLC-UNet," in *Proc. MLMI*, LNCS vol. 11861, Springer, 2019, pp. 274–282.

---

## AI Use

Per-file AI contribution estimates are listed in the Module Summary tables above and documented in full in `AI_Use_Disclosure.md`. The research paper used Grammarly for grammar and Claude for reference formatting and outline assistance.

---

*Capstone project — Biomedical Informatics MS, University of Nebraska Omaha, Spring 2026.*
