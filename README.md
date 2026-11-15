# EduGaze

**EduGaze: An Eye-Tracking-Based Visual Attention Analysis Framework for Improving Educational Diagrams.**

This repository currently provides the research foundation: a modular Python package,
strict fixation CSV validation, separated data storage, provenance records, and automated
tests. Full attention analysis, visualization, prediction, and educational evaluation
are reserved for later phases. Gaze measurements alone do not establish comprehension.

```text
EduGaze/
├── src/edugaze/          # Fixation schema, CSV validation, paths and CLI
├── tests/               # Standard-library automated tests
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
└── requirements-research.txt
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

Offline setup on this host used the bundled setuptools wheel:

```bash
python -m pip install --no-index /usr/share/python-wheels/setuptools-78.1.1-py3-none-any.whl
python -m pip install --no-index --no-build-isolation -r requirements.txt
```

Optional scientific and notebook packages can be installed with
`python -m pip install -r requirements-research.txt`. These are not required by the
foundation. The old Windows dependency freeze is preserved in `legacy/`.

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
