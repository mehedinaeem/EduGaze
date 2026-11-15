# Workspace audit — 2026-11-15

Historical Phase 1 record. For the current real dataset integration, see
[Phase 2](coco_freeview.md).

Initial Git state: only `Dataset/` was untracked; no existing tracked edits were present.
The repository already tracked a Windows Python 3.11.9 virtual environment (about 514 MB),
a King County housing CSV, an EDA notebook, a generic housing README, a large frozen
requirements list, an empty scripts directory and an outputs/figures placeholder.
No EduGaze analysis implementation or tests existed. No AGENTS.md applied in this repository.

The EDA notebook loads the housing CSV and plots price against living area with seaborn;
it supplies no eye-tracking algorithm. The original CSV, notebook, README and requirements
were preserved under `legacy/house_price_eda/`. Their relative notebook data path still
works there. The old pyvenv.cfg was backed up there before configuring the Linux runtime.
Windows Lib/, Scripts/ and share/ files remain in .venv; Linux uses lib/ and bin/.
The original Git index/history was left untouched.

Both ZIP archives were enumerated, checked for traversal/symlink entries and extracted
without executing code. Archive SHA-256 hashes and sizes are recorded in
`archive_manifest.json`. `source_inventory.json` records all 17 Python files' top-level
definitions, imports and line counts. READMEs, requirements, analysis/demo code,
evaluation metrics, license and MultiMatch attribution were inspected. See the separate
license/reuse review for limitations. The original dataset notes and archive locations
were retained. Demo outputs stay inside the unchanged third-party extraction.

Existing folder conventions (data, notebooks, scripts, outputs) were retained and given
clear roles. The foundation uses no third-party runtime dependencies. Scientific tooling
is optional; old unrelated EDA and Windows-specific packages are not installed in the
Linux runtime. New synthetic fixture data is isolated from raw and processed directories.
Tests cover invalid values, bounds, missing data, order preservation, CLI behavior,
path separation and checksums of locally available original archives.

This phase does not implement heatmaps, AOI analysis, prediction, data collection,
training or educational evaluation. No real eye-tracking data is currently available.

Final checks: all four archived legacy files match their original Git blobs byte for
byte. `scripts` is recorded in Git as a gitlink (mode 160000), with no checked-out
source and no .gitmodules present; the pointer was preserved and not repurposed.
The Linux environment is Python 3.14.4. Network package installation was unavailable;
setuptools 78.1.1 was installed from the host's bundled wheel, then EduGaze was
installed editable offline. All eight unittest tests passed, the synthetic validation
CLI reported three fixations/two participants, and pip check reported no broken
requirements. Optional research packages were not installed in the Linux environment.
