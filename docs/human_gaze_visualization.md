# Human gaze visualization — Phase 3

The two modules operate on `MatchedImage` returned by the Phase 2
`CocoFreeView.load_image` adapter. They use the already registered `x_image` and
`y_image` values; they do not resize, flip or remap fixation coordinates. The
photograph and all overlays share the exact extent `(0, width, height, 0)` with
top-left origin. OpenCV BGR photographs are converted to RGB for Matplotlib.

These figures visualize **recorded human gaze**. The heatmaps are smoothed
empirical fixation densities, not model-predicted attention or evidence of learning.
No third-party starter code or trained model is used.

## Scanpaths

`src/edugaze/visualization/scanpath.py` provides `plot_scanpath`.

- Select a participant; if several source trials remain, also select a
  `record_index`. Separate participants/trials are never connected.
- Fixations are sorted by their recorded within-trial index. Point numbers show
  the original index plus one, including gaps when coordinates are unusable.
- Consecutive valid original indices are joined with straight lines. Lines are
  descriptive connections, not measured continuous saccade trajectories.
- Durations appear in a table keyed to point numbers, keeping dense clusters
  legible. Missing/nonpositive/nonfinite durations are shown as unavailable;
  their valid coordinates can still be plotted. `show_durations=False` hides
  the table. Coordinates outside the image or rejected by the loader are omitted
  with an explicit exclusion count. Removed coordinates never create a bridging line.
- Ordering comes from the recorded sequence; no onset timestamps are inferred.

## Heatmaps

`src/edugaze/visualization/heatmap.py` provides `compute_density` and `plot_heatmap`.
`compute_density` returns a `DensityMap` with the raw smoothed array and metadata.
The array shape is exactly `(image_height, image_width)`.

| Setting | Behavior |
| --- | --- |
| `weighting='duration'` (default) | Add each positive, finite observed duration in milliseconds; exclude missing/invalid durations |
| `weighting='count'` | Add one for each coordinate-valid recorded fixation, even if duration is unavailable |
| `participant='101'` | Use only that original participant identifier |
| `participant=None` | Pool all participants/trials present in the supplied matched image |
| `sigma=32` (default) | Gaussian standard deviation in registered **image pixels**, both axes |
| `sigma=0` | Disable smoothing |
| `alpha=.65` (default) | Maximum heat overlay opacity; supported range 0–1 |

Subpixel coordinates enter pixel cells at `floor(x), floor(y)`; repeat visits
accumulate using NumPy. OpenCV applies a Gaussian with radius `ceil(4*sigma)` and
a zero-valued exterior. Gaussian mass extending outside the photograph is cropped,
not reflected or wrapped. Metadata reports input weight and weight retained inside
the canvas. Aggregation sums observations; it does not balance each participant's
contribution. Participants with more fixations/longer recorded dwell contribute more.

The raw count density has units of fixations per pixel; raw duration density has
units of milliseconds per pixel. For display, each map is divided by its own peak.
The colorbar explicitly labels this relative scale. It is **not a probability map**;
colors across separate figures do not establish equal absolute dwell/count values.
The inferno overlay opacity equals `alpha * relative_density`, leaving zero-density
pixels fully transparent. No hidden threshold discards low-density areas.

## Python usage

```python
from edugaze.datasets import CocoFreeView
from edugaze.visualization import compute_density, plot_scanpath, plot_heatmap, save_png

data = CocoFreeView(
    'data/raw/coco_freeview/fixations/COCOFreeView_fixations_trainval.json',
    'data/raw/coco_freeview/images',
)
matched = data.load_image('000000128398.jpg')
scan = plot_scanpath(matched, participant='101', show_durations=True)
save_png(scan, 'outputs/my_new_figures/scanpath.png', dpi=300)

heat = plot_heatmap(matched, participant=None, weighting='duration', sigma=32, alpha=.65)
save_png(heat, 'outputs/my_new_figures/heatmap.png', dpi=300)
raw_density = compute_density(matched, participant='101', weighting='count', sigma=16)
print(raw_density.values.shape, raw_density.metadata)
# Figures use the Agg canvas without changing the global Matplotlib backend.
# Release figures after use when generating a batch:
scan.figure.clear()
heat.figure.clear()
```

`save_png` defaults to 300 DPI and a nine-inch-wide figure (2,700 pixels wide).
Text/lines are rendered at export resolution; axes, labels and duration tables are
included without automatic cropping. Figure height adapts to the photograph and
duration table. PNG Description metadata identifies recorded gaze, settings,
selections and exclusion counts. Existing outputs are never overwritten.

## Reproduce real examples

Install the project dependencies with `python -m pip install -r requirements.txt`.
Matplotlib is now a core dependency. Exact tested versions are recorded in
`requirements-tested.txt`.

```bash
python tools/visualize_coco_freeview.py \
  --image 000000128398.jpg --participant 101 \
  --sigma 32 --alpha .65 --dpi 300 \
  --out outputs/my_new_visualization_run
python -m unittest discover -s tests -v
```

Use a fresh output directory per run. `--record-index` chooses one source trial for
the scanpath when needed. Individual heatmaps pool all selected participant trials
for the photograph; pooled maps use all available viewers. The runner exports five
PNGs plus a manifest with source JSON/image hashes, alignment geometry, plot
parameters, selections, counts and PNG hashes. Raw input datasets are unchanged.
Matplotlib's cache can optionally be directed to a writable folder using
`MPLCONFIGDIR` when running in a restricted environment.

Verified examples are in `outputs/human_gaze_visualization_publication/`:

| File | Recorded observations |
| --- | --- |
| `scanpath_participant.png` | Participant 101, source record 0; 15 fixations, 14 connections, 15 measured durations |
| `heatmap_duration_participant.png` | Participant 101; 15 fixations, 4,238 ms input weight |
| `heatmap_count_participant.png` | Participant 101; 15 counted fixations |
| `heatmap_duration_all.png` | Ten viewers; 148 fixations, 42,023 ms input weight |
| `heatmap_count_all.png` | Ten viewers; 148 counted fixations |

All figures use the real `000000128398.jpg` stimulus and verified Phase 2 alignment.
Heatmap examples use sigma 32 pixels and maximum alpha .65. Each is 2,700 pixels
wide with 300 DPI metadata; no exclusions occurred for this image.

The complete test suite passes 38 tests, including 12 visualization tests. Tests
verify asymmetric coordinate placement, RGB conversion, line segments, trial
isolation, missing-duration handling, repeated-fixation weights, Gaussian shape and
boundary behavior, participant aggregation, transparency, explicit loader transforms,
PNG dimensions/DPI/provenance, output preservation and actual local COCO-FreeView
plot generation. No subsequent phase has been started.
