# Interactive research dashboard — Phase 5

EduGaze now includes a local Streamlit application built around the Phase 2 loader,
Phase 3 visualization functions and Phase 4 statistics/AOI functions. The interface
is presentation and orchestration code; it does not reimplement gaze metrics,
coordinate registration, AOI matching or Gaussian-density calculations. No machine
learning, inference or training is included.

## Launch on Ubuntu

From the existing workspace:

```bash
cd /home/mehedinaeem/Desktop/Code/EduGaze
source .venv/bin/activate
python -m streamlit run streamlit_app.py
```

Open **http://127.0.0.1:8501** in a local browser. Stop the app with Ctrl+C.
The project configuration binds Streamlit to loopback, uses headless server mode
and disables usage-statistics collection. It does not require an X server for PNG
rendering and does not upload datasets or publish a hosted site.

For a new environment, install `python -m pip install -r requirements.txt` first.
The existing `.venv` has been updated and tested with Python 3.14.4, Streamlit 1.67.0
and Plotly 6.9.0. `requirements-tested.txt` records the exact tested core dependency
versions. Plotly and Streamlit are declared in `pyproject.toml` and installed by
`requirements.txt`. Optional browser tests use `requirements-dev.txt`.

## Navigation and workflow

| Page | Function |
| --- | --- |
| Overview | Dataset-wide image, participant, scanpath and fixation counts; photograph availability and recorded-data quality |
| Dataset explorer | Search identifiers, inspect availability and recorded rows, view the original photograph |
| Gaze workspace | Interactive recorded fixation positions, single-trial scanpaths and attention-density generation |
| Statistics | Existing gaze summaries, participant/trial tables, count charts, spatial grid counts and endpoint distances |
| AOI analysis | Bundled example, editable rectangle table, CSV/JSON upload or pasted definitions; statistics and charts |
| Exports | Current-selection normalized CSV, statistics CSV/JSON, PNG figures and a complete ZIP bundle |

1. Choose an **Image** in the sidebar (the selector supports typing/search).
2. Choose **Analysis participant**. “All participants” pools that photograph's
   recorded fixations for statistics, heatmaps, AOIs and normalized CSV export.
3. A scanpath always uses one participant and source trial. For pooled analysis,
   **Scanpath participant** chooses that viewer; **Recorded trial** selects its
   source record. This does not silently filter pooled statistics to that trial.
4. On **Gaze workspace → Scanpath**, click **Generate scanpath**. Hover, pan, zoom,
   use the fixation-order slider or **First fixation / Full trial** buttons. These
   are ordinal controls, not elapsed-time playback. Consecutive-line topology,
   original order and invalid-coordinate gap policy come from `plot_scanpath`.
5. On **Attention heatmap**, click **Generate heatmap**. Sidebar settings control
   duration/count weighting, Gaussian sigma (image pixels), maximum opacity,
   150/300/600-DPI PNG export, and the scanpath's recorded-duration table.
6. On **AOI analysis**, select/edit/upload rectangles and click **Analyze AOIs**.
   The illustrative bundled regions belong only to `000000128398.jpg`; other
   photographs start with an empty editor or require their own CSV/JSON.
7. Open **Exports → Prepare exports**. Download individual CSV, JSON and PNG files,
   a chosen result table, or the complete ZIP. Statistics exports work without AOIs;
   generating AOI analysis adds the full Phase 4 report and its five charts.

PNG exports contain the full selected scanpath, even if the interactive ordinal
slider currently shows a prefix. All scientific plots are labeled recorded human
gaze. Density colors are normalized to each map's own peak, not directly comparable
absolute dwell/count values. Duration heatmaps use measured positive milliseconds;
count heatmaps count coordinate-valid recorded positions. See
[visualization methods](human_gaze_visualization.md) for boundary/smoothing details.

AOIs use registered image pixels and the existing nonoverlap/half-open boundary rules.
“Viewed” means an observed valid fixation hit, not proof of conscious perception.
COCO-FreeView has no onset/trial-start timestamps; TTFF remains unavailable rather
than being estimated from summed durations. See [analysis methods](gaze_statistics_aoi.md).

## Input handling and result consistency

The default data directory is `data/raw/coco_freeview/`, containing:

```text
fixations/COCOFreeView_fixations_trainval.json
images/                    # Exact matching photograph filenames, possibly nested
```

Change **Dataset location** in the sidebar or set `EDUGAZE_DATA_ROOT` before launch.
**Refresh dataset** clears cached source objects and generated results after local
file changes. The cache is bounded and source files are fingerprinted by path, size
and modification time. Shared cached loader objects are used read-only; analysis
works on copies of selected records.

Missing, empty or corrupt JSON, unavailable/ambiguous photographs, unsupported image
alignment, empty searches, invalid AOI geometry and unusable measurements produce
messages instead of fabricated fallback data. Differently sized source photographs
must first have their explicit registration configured through the Phase 2 loader;
the dashboard never guesses a resize or letterbox transform.

Changing the dataset, image, participant, trial or visualization settings invalidates
old generated artifacts. Editing AOI definitions invalidates AOI results and export
bundles. Regenerate plots/analysis and prepare exports after a change; stale outputs
are never silently downloaded under a new selection. Downloads do not rerun the app
or interrupt interactive chart state.

User AOIs are loaded through the existing CSV/JSON parser. Uploaded definitions are
limited to 2 MB. Temporary files are scoped to the request and removed afterward;
raw datasets and previous result directories are untouched. Downloads remain local.
The ZIP includes original normalized-record provenance, gaze statistics, generated
figures, their settings and available AOI definitions/reports. Source/image hashes
are carried into the analysis provenance.

## Implementation

- `streamlit_app.py`: application entrypoint.
- `src/edugaze/dashboard/app.py`: navigation, widget state and presentation.
- `src/edugaze/dashboard/service.py`: adapters calling existing loader, plotting,
  statistics, AOI parsing and export functions; interactive Plotly views and ZIP assembly.
- `.streamlit/config.toml`: local academic theme, loopback serving and usage settings.
- `tests/test_dashboard.py`: service and Streamlit AppTest workflows/error cases.
- `tools/test_dashboard_browser.py`: optional real Chrome workflow and actual downloads.

The Plotly view adapts the existing scanpath's plotted coordinates and line segments;
it does not invent new positions or connect participants, trials or invalid gaps.
Fixation-order controls use direct trace updates, avoiding cancellable animation
promises and avoiding any claim to reconstruct missing elapsed timing.

## Verification

```bash
python -m unittest discover -s tests -v
# Optional actual-browser check; uses an already installed Chrome/Chromium:
python -m pip install -r requirements-dev.txt
python tools/test_dashboard_browser.py --start-server \
  --out outputs/my_new_dashboard_browser_check
```

The browser harness starts its own loopback server and shuts it down afterward.
Without `--start-server`, start the app separately first. Choose a fresh `--out`
directory. It verifies HTTP health, browser JavaScript errors, real image/participant
selection, first/full fixation-order controls, heatmap generation, AOIs, and actual
CSV/JSON/PNG/ZIP downloads. No browser download is needed if Chrome is already installed.

The verified real workflow used `000000128398.jpg`, participant `101`, 15 fixations,
4,238 ms of measured duration, four AOIs and a 25-file ZIP. TTFF stayed null. Browser
screenshots, downloaded files and the passing validation report are saved under
`outputs/dashboard_complete_workflow/`. The full unittest suite passes 66 tests,
including ten dashboard tests and the real-data AppTest workflow. Tests additionally
cover missing/empty/corrupt datasets, missing photographs, empty search results,
invalid/uploaded AOIs, statistics-only exports, geometry, gap policy, PNG output and
selection/settings invalidation. Earlier data, code and outputs are preserved.
