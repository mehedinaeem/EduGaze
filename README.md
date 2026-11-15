# EduGaze

**EduGaze: An Eye-Tracking-Based Visual Attention Analysis Framework for Improving Educational Diagrams.**

This repository provides a modular Python package, strict CSV validation, a recorded
COCO-FreeView JSON loader, exact photograph matching, explicit coordinate registration,
normalized CSV export, recorded human-gaze scanpaths and heatmaps, gaze/AOI statistics, provenance records,
and automated tests. A local Streamlit research dashboard integrates these modules.
Prediction and educational evaluation remain reserved for later phases. Gaze measurements alone do not establish comprehension.

```text
EduGaze/
├── src/edugaze/          # Validation, dataset loaders, registration, export and CLI
├── tests/               # unittest: parser, alignment and real local data
├── tools/               # Official acquisition and real-data verification
├── configs/             # Future experiment configurations
├── data/
│   ├── raw/             # Unmodified measurements and stimuli (local only)
│   ├── processed/       # Reproducible derivatives (local only)
│   └── synthetic/       # Clearly labeled artificial software fixture
├── Dataset/             # Original ZIP archives and supplied dataset readme
├── third_party/         # Both extracted archives, unchanged (local only)
├── legacy/house_price_eda/ # Preserved original README, dependencies, CSV, notebook
├── notebooks/           # Future research exploration
├── scripts/             # Existing Git submodule pointer (uninitialized)
├── outputs/figures/     # Generated results
├── docs/                # Workspace audit, license review and source inventories
├── .vscode/             # Interpreter and unittest configuration
├── .venv/               # Linux runtime plus preserved Windows environment files
├── pyproject.toml
├── requirements.txt
├── requirements-research.txt
└── requirements-tested.txt
```

## Setup and checks

Python 3.10+ is required. The existing environment was configured with the available
Linux Python; its original Windows files were retained. Use the Linux interpreter:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m edugaze validate data/synthetic/fixations.csv --width 100 --height 100
```

The Phase 1 bootstrap used the bundled setuptools wheel. With Phase 2 core
dependencies already installed, the local package can also be reinstalled offline:

```bash
python -m pip install --no-index /usr/share/python-wheels/setuptools-78.1.1-py3-none-any.whl
python -m pip install --no-index --no-build-isolation -r requirements.txt
```

NumPy, Pandas, headless OpenCV and Matplotlib are core dependencies installed by
`requirements.txt`. Optional additional scientific and notebook packages can be installed with
`python -m pip install -r requirements-research.txt`. The old Windows dependency freeze is preserved in `legacy/`.

CSV columns are `image_name,participant_id,x,y,duration_ms`. Coordinates must use
the actual stimulus canvas in pixels, origin at top left; duration is milliseconds.
Rows retain recorded order; invalid records raise an error with a row number.
Keep separate trials in separate files in this initial schema. No implicit resizing,
filtering, or coordinate conversion occurs. The included three-row fixture is synthetic
and has no corresponding image or scientific validity.

See [the workspace audit](docs/workspace_audit.md) and
[third-party review](docs/third_party_review.md) before using reference code or data.
Original archives remain unchanged and are fingerprinted in `docs/archive_manifest.json`.

The existing Git history tracks Windows `.venv` files. They were deliberately left
tracked to avoid destructive index changes; `.gitignore` prevents new environment files
from being added but cannot untrack old files. No commits or staging were performed.

## Phase 2: real COCO-FreeView data

See [the dataset integration guide](docs/coco_freeview.md) for verified official
links, actual schema, destinations, alignment rules and acquisition details.

```bash
python -m edugaze coco data/raw/coco_freeview/fixations/COCOFreeView_fixations_trainval.json \
  --images data/raw/coco_freeview/images --image 000000128398.jpg --participant 101
python tools/verify_coco_freeview.py --image 000000128398.jpg --out outputs/coco_freeview_new_check
```

The verification tool exports all recorded fixations for the selected photograph,
plus an overlay of one participant's single recorded scanpath. Use a new output
directory on each run to preserve existing results. The canonical CSV command
(`edugaze validate`) is separate from the richer dataset export, which includes
source indexes, display/image/normalized coordinates and validity flags.

## Phase 3: recorded human gaze visualization

Two reusable modules under `src/edugaze/visualization/` create numbered single-trial
scanpaths with duration tables and individual/pooled Gaussian heatmaps. Heatmaps
support count and duration weighting, configurable smoothing and transparency.
They describe measured fixations; they are not model predictions.

```bash
python tools/visualize_coco_freeview.py --image 000000128398.jpg --participant 101 \
  --sigma 32 --alpha .65 --dpi 300 --out outputs/my_new_visualization_run
```

The verified 300-DPI examples are in `outputs/human_gaze_visualization_publication/`.
See [visualization methods and usage](docs/human_gaze_visualization.md) for weighting,
normalization, coordinate conventions, parameters, output provenance and tests.

## Phase 4: gaze statistics and rectangular AOIs

`src/edugaze/analysis/` provides fixation/duration statistics, within-trial endpoint
distances, spatial grid distributions, participant summaries, CSV/JSON AOI loading,
AOI counts/durations/viewed flags and direct transitions. TTFF is computed only from
actual onset timestamps and trial-start times; it is unavailable for COCO-FreeView.

```bash
python tools/analyze_coco_freeview.py --aois configs/aois/coco_000000128398.csv \
  --out outputs/my_new_aoi_run --dpi 300
```

The real-data example, tables and charts are saved in `outputs/gaze_aoi_analysis/`.
See [analysis methods and usage](docs/gaze_statistics_aoi.md) for boundary rules,
missing-value handling, timing requirements, diagram reuse and result interpretation.

## Phase 5: interactive research dashboard

```bash
source .venv/bin/activate
python -m streamlit run streamlit_app.py
```

Open **http://127.0.0.1:8501**. Use the sidebar to select an image and participant,
then explore recorded fixations, generate interactive scanpaths and heatmaps,
inspect statistics, edit/upload rectangular AOIs, and download CSV/JSON/PNG/ZIP
results. All scientific calculations use the existing validated modules.

See [dashboard instructions](docs/research_dashboard.md) for configuration,
missing-data behavior, research interpretation and the complete tested workflow.
No machine learning has been added.
