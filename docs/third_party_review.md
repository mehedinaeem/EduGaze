# Third-party review

Historical Phase 1 record. For the current real dataset integration, see
[Phase 2](coco_freeview.md).

Review scope: files supplied inside the two local ZIP archives. No external dataset
was downloaded and no third-party code or model checkpoint was executed.

| Archive | License evidence | Reuse decision |
| --- | --- | --- |
| Scanpath_Prediction-master.zip | Top-level MIT License, copyright 2020 CVLab@StonyBrook | Preserve full extraction including LICENSE; no code imported into the core package. Future reuse must retain the notice. |
| GazeTrack-Education-Starter.zip | No LICENSE file or explicit code-license grant in README | Inspection/reference only; do not copy its code or demo assets into the package without permission or a license grant. |

Extracted paths are `third_party/Scanpath_Prediction-master/Scanpath_Prediction-master/`
and `third_party/GazeTrack-Education-Starter/GazeTrack-Education/`.
Extra upstream directory levels are retained to preserve the archives' structure.

## Research candidates

The MIT scanpath project provides GAIL/inverse reinforcement learning, PPO training,
dynamic contextual belief maps, model/environment code, scanpath plotting, sequence
matching, target-search efficiency and MultiMatch evaluation. Its training code expects
COCO-Search18 fixation JSON, bounding-box annotations and precomputed belief maps,
which are not bundled. Checkpoints are bundled but were not deserialized. Natural-image
target search is a different task from educational diagram viewing; transfer needs validation.
`irl_dcb/multimatch.py` credits another upstream project in its first line; check that
upstream's license separately before extracting or redistributing this component. The
archive's top-level MIT file alone is insufficient evidence for every embedded work,
checkpoint or dataset's permissions.

The education starter demonstrates CSV/COCO JSON reading, duration-weighted Gaussian
heatmaps, individual-participant scanpaths, rectangular AOI counts/dwell shares and
first-fixation ordinal metrics. It includes 72 synthetic fixations, a diagram and demo
results, not measured participant data. Its fixed display assumptions, first-image
selection and row filtering need review before any future licensed adaptation.
No implementation was copied. EduGaze's small CSV validator and fixture were newly written.

The supplied `Dataset/readme.txt` documents COCO-FreeView JSON, 1680x1050 aspect-preserving
padded stimuli and the Chen et al. (2022) citation. It is documentation, not actual
fixation data or a redistribution license. The original archives and extracted material
are excluded from new Git additions pending redistribution review. Keep their notices
and follow separate terms for any future dataset acquisition. No project-wide license
has been invented for the user's existing work.
