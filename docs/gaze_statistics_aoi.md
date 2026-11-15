# Gaze statistics and AOI analysis — Phase 4

The reusable `edugaze.analysis` package operates on validated aligned fixation
DataFrames. It is independent of COCO image content and can also analyze educational
diagrams once their measurements use the same explicit schema and registration.
All results describe recorded human fixations; no model predictions or learning
outcomes are inferred. No raw input datasets are changed.

## Inputs and statistics

Supply one image's records, its `(width, height)` in pixels, and these columns:

- `image_name`, `participant_id`, `record_index`: observed image/participant and source
  trial identifiers. The Phase 2 loader's record_index is its source-list index.
- `fixation_index`: original zero-based within-trial ordinal; duplicates are rejected.
- `x_image`, `y_image`: registered image coordinates, top-left origin.
- Optional `duration_ms`: positive measured fixation duration, never synthesized.
- Optional `fixation_onset_ms` and `trial_start_ms`: observed times in the **same clock**.
- Loader-provided `inside_image`, `image_width`, `image_height` are honored/validated
  when present. Dimensions must agree with the supplied image size.

`analyze_gaze` returns an overall summary plus participant, trial, saccade and spatial
tables. All recorded rows count toward fixation_count. Spatial measures exclude
nonfinite/out-of-canvas coordinates and loader-rejected positions, with quality counts
reported explicitly. Numeric missing/invalid measurements remain unavailable; booleans
are not accepted as coordinates or durations. Input DataFrames are copied, not modified.

| Output | Definition |
| --- | --- |
| Fixation count | Number of recorded fixation rows |
| Mean fixation duration | Mean of available positive finite measured durations; coverage counts accompany it |
| Observed fixation duration | Sum of available positive measured durations; null when all durations are unknown |
| Total fixation duration | Reported only when all relevant fixation durations are available; otherwise null |
| Saccade distance | Euclidean distance between consecutive valid fixation endpoints in the same source trial, in pixels |
| Observed scanpath length | Sum of available endpoint distances; zero when no valid pair was observed |
| Total scanpath length | Sum only when coordinates and the complete zero-based sequence are valid; otherwise null |
| Spatial distribution | Unsmoothed grid-cell counts and percentages of coordinate-valid fixations; configurable grid, default 6×4 |

Distances never connect participants, trials, invalid coordinates or gaps in original
fixation indices. They are displacement proxies, not measured continuous saccade
trajectories, velocities or visual-angle amplitudes. Aggregate path length sums
separate trial lengths; it is not one trajectory across participants. Empty observed
sequences of AOI hits have count/duration sum zero, while mean duration is unavailable.

## Timing and time to first fixation (TTFF)

TTFF requires both observed fixation onset timestamps and an observed trial-start
origin. Each trial must have finite times, a constant origin, nondecreasing onsets
no earlier than the origin, valid coordinates, and a complete ordinal sequence.
The latency is the first relevant onset minus the trial origin. Invalid/incomplete
trials return null with a reason. A measured zero latency remains zero.

Durations are **never accumulated to estimate onsets**, because they omit saccades
and gaps. COCO-FreeView supplies durations but not these timestamps/origins, so all
TTFF values in the real example are unavailable (CSV blank / JSON null).

Trial tables contain individual TTFFs and statuses. For multiple trials the single
TTFF field is null; explicitly named mean_trial_time_to_first_fixation_ms averages
only available trial latencies, with timed_trial_count / timed_viewed_trial_count
showing coverage. AOI trials without a recorded hit have no AOI latency. This avoids
combining independent trial clocks or treating an unseen AOI as a zero-latency hit.

## AOI definitions and boundaries

Canonical CSV fields: `aoi_id,name,x,y,width,height`. IDs/names are nonempty strings;
IDs must be unique. CSV leading zeros are preserved. JSON accepts a list of objects
with those same six fields, with numeric geometry. The paired files in `configs/aois/`
demonstrate equivalent definitions for `000000128398.jpg`.

```json
[
  {"aoi_id": "region_1", "name": "Diagram label", "x": 50, "y": 50, "width": 150, "height": 100}
]
```

Rectangles use **image coordinates**, with left/top edges included and right/bottom
edges excluded: `[x,x+width) × [y,y+height)`. Shared edges belong to just one region.
Invalid sizes, nonfinite geometry, out-of-canvas boxes, duplicate IDs and positive-area
overlaps raise errors. Overlapping/nested AOIs are deliberately rejected in this phase
rather than silently assigned a priority. Transform display-space rectangles explicitly
when using a cropped/rescaled original photograph; never reuse boxes solely because
two images have the same dimensions.

`analyze_aois` returns these tables:

- AOI summary: count, observed/complete duration totals and coverage, percentages,
  participants with observed hits, trial coverage and latency summaries.
- Participant/AOI and trial/AOI tables, including `viewed` booleans. Viewed means at
  least one coordinate-valid recorded hit; false means no recorded hit, not proof that
  the participant never perceived the region. Interpret alongside coordinate quality.
- Fixation-level assignments: blank aoi_id for outside/invalid coordinates; validation
  flags distinguish the two. Metadata reports outside and invalid counts separately.
- Direct transitions: different AOIs at consecutive valid source fixation indices
  within one participant/trial, plus directional aggregate counts and participant counts.
  Repeated same-AOI visits do not form a change. Outside/invalid/gapped records
  interrupt transitions; sequences are not compressed to bridge these interruptions.

`percentage_of_fixations` uses **all recorded fixations** as its denominator.
`percentage_of_valid_fixations` separately uses coordinate-valid records and is null
if that denominator is zero. Percentages can sum to less than 100 because fixations
outside the defined AOIs remain unassigned. AOI durations count only recorded hits;
unknown duration values are not replaced by zeros.

## Python API

```python
from edugaze.datasets import CocoFreeView
from edugaze.analysis import analyze_gaze, analyze_aois, load_aois
from edugaze.analysis.reports import export_analysis

data = CocoFreeView(
    'data/raw/coco_freeview/fixations/COCOFreeView_fixations_trainval.json',
    'data/raw/coco_freeview/images',
)
matched = data.load_image('000000128398.jpg')
size = (matched.pixels_bgr.shape[1], matched.pixels_bgr.shape[0])
aois = load_aois('configs/aois/coco_000000128398.csv', image_size=size)
stats = analyze_gaze(matched.fixations, image_size=size, grid=(6, 4))
regions = analyze_aois(matched.fixations, aois, image_size=size)
print(stats.summary)
print(regions.participants)
export_analysis(matched, aois, 'outputs/my_new_analysis', dpi=300)
```

For an educational diagram, pass its own registered DataFrame, image size and AOIs to
these same functions. Real timestamps may be supplied explicitly when recorded by that
experiment; the analysis does not assume them from the COCO-FreeView schema.

## Run examples and tests

```bash
source .venv/bin/activate
python tools/analyze_coco_freeview.py \
  --image 000000128398.jpg --aois configs/aois/coco_000000128398.csv \
  --out outputs/my_new_aoi_run --dpi 300
# Participant selection and JSON definitions:
python tools/analyze_coco_freeview.py \
  --participant 101 --aois configs/aois/coco_000000128398.json \
  --out outputs/my_new_individual_aoi_run
python -m unittest discover -s tests -v
```

Choose a fresh output directory on every run. An existing directory is never overwritten.
Using a different photograph requires an explicit `--aois` file to avoid reusing the
bundled demonstration rectangles accidentally. Optional MPLCONFIGDIR can direct the
Matplotlib cache to a writable directory in a restricted environment.

The report directory contains 11 CSV tables, a complete `report.json`, and five PNG
charts: AOI overlay, fixation-count bars, observed-duration bars with coverage labels,
spatial grid distribution and directional AOI-transition matrix. PNGs default to 300 DPI.
CSV/JSON preserve missing metrics as blanks/null. Report metadata describes boundaries,
denominators, exclusions, units and timing policies; the runner records source JSON,
photograph and AOI-file SHA-256 fingerprints and coordinate geometry.

## Verified real-data example

`outputs/gaze_aoi_analysis/` analyzes 148 fixations from ten viewers on
`000000128398.jpg`. Mean duration is 283.94 ms; total measured duration is 42,023 ms
across those viewers. There are 138 consecutive-fixation displacements with mean
233.15 pixels and aggregate within-trial length 32,174.35 pixels. All 148 coordinates
and durations are valid; all ten trials lack onset timing and therefore have no TTFF.

The four illustrative hand-defined regions are not semantic ground truth or model
segmentation:

| AOI | Fixations | Recorded duration (ms) | Share of all fixations | Viewers with hits |
| --- | ---: | ---: | ---: | ---: |
| Foreground preparation | 106 | 30,633 | 71.62% | 10 |
| Background preparation | 20 | 5,052 | 13.51% | 10 |
| Central bottle | 6 | 2,018 | 4.05% | 5 |
| Left context | 3 | 843 | 2.03% | 2 |

Thirteen valid fixations are outside these rectangles. Thirty direct changes between
different AOIs were observed. The JSON-AOI participant example is separately saved in
`outputs/gaze_aoi_analysis_participant101_json/`.

Testing includes 18 analysis tests plus the existing suite. It covers known arithmetic,
partial/missing durations, finite and invalid positions, trial/participant isolation,
measured timing versus missing/invalid clocks, AOI edges/overlaps, CSV/JSON equivalence,
explicit Phase 2 coordinate transforms, transition gaps, strict JSON null serialization,
chart/CSV generation and a real-data integration test. No subsequent phase is started.
