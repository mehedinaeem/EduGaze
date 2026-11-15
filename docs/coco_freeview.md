# COCO-FreeView integration — Phase 2

## Sources and acquisition

Official pages inspected on 2026-11-15:

- [COCO-FreeView](https://sites.google.com/view/cocosearch/coco-freeview)
- [Shared image stimuli and usage terms](https://sites.google.com/view/cocosearch/)
- [Official linked README](https://drive.google.com/file/d/1Hj_jyK8Ml27Ge_5sogEtyI7XOjyad4aj/view?usp=sharing)

The website identifies ten viewers and five-second free-viewing trials and links the
train/validation data. Its test fixation set is withheld. The local `Dataset/readme.txt`
contains the supplied v1.0 schema and display-preparation documentation; the linked
Drive preview did not expose its text through the browser tool. The actual downloaded
JSON was inspected in full and takes precedence over absent fields in that README.

These are the exact publisher links followed from the two official pages:

| File | Official URL | Local destination |
| --- | --- | --- |
| Train/validation fixation JSON | https://vision.cs.stonybrook.edu/~cvlab_download/COCOFreeView_fixations_trainval.json | `data/raw/coco_freeview/fixations/COCOFreeView_fixations_trainval.json` |
| Target-present stimulus ZIP | https://vision.cs.stonybrook.edu/~cvlab_download/COCOSearch18-images-TP.zip | `data/raw/coco_freeview/archives/COCOSearch18-images-TP.zip` |
| Target-absent stimulus ZIP | https://vision.cs.stonybrook.edu/~cvlab_download/COCOSearch18-images-TA.zip | `data/raw/coco_freeview/archives/COCOSearch18-images-TA.zip` |

No manual interaction was required. HTTPS certificate validation failed at the publisher;
acquisition used the same published host and paths over HTTP, without disabling TLS
verification. This transport is recorded in `data/raw/coco_freeview/download_manifest.json`.
SHA-256 fingerprints establish local reproducibility, not publisher authentication.
Full archive acquisition uses bounded parallel ranges and checkpoints completed chunks
with the archive ETag and size, allowing a retry to resume without losing completed work.
Images are extracted by exact names observed in the real JSON and ZIP directories;
ZIP CRC checks validate extraction. The official category folders repeat some image
filenames: acquisition compares all copies byte for byte, stores one only when they are
identical, and records all member paths. Conflicting copies fail instead of being guessed. Data and generated overlays remain Git-ignored.
The shared image page limits redistribution of images and fixation data; keep them local.
The code license does not substitute for dataset permissions.

If browser download becomes necessary, download the three exact files above into their
listed folders. Extract JPEG photographs from both archives into
`data/raw/coco_freeview/images/`, retaining exact 12-digit filenames. Category subfolders
are supported; duplicate basenames raise an ambiguity error. Do not download COCO-Search18
**search fixation JSON** as a substitute for the FreeView file.

Acquisition commands, from the repository root:

```bash
# All train/validation-referenced photographs; saves the two full stimulus archives.
python tools/download_coco_freeview.py --all-images
# Publisher certificate-error fallback used in this environment:
python tools/download_coco_freeview.py --transport http --all-images
# Smaller verification-only collection, if a full download is unnecessary:
python tools/download_coco_freeview.py --transport http --sample-images 3
```

## Observed file and schema

The downloaded JSON is 50,110,884 bytes, SHA-256
`771167f4b5de5b9b58fb07c40d6e285d38b3b2ab9e4c6267df8a454a909da730`.
Direct local inspection found 43,048 scanpath objects, 665,005 fixation samples,
4,317 unique photograph names and ten participant identifiers:
`101, 102, 103, 104, 105, 107, 108, 109, 110, 116`.
Train has 37,038 scanpaths; valid has 6,010. These are counts of this downloaded
train/validation file, not the website's total dataset count.

| Actual key | Meaning / treatment |
| --- | --- |
| `name` | Exact photograph basename; preserve leading zeros and extension |
| `subject` | Original participant ID; exposed as a string without renumbering |
| `condition` | `freeview` in all observed records; not a search-target label |
| `X`, `Y` | Equal-length ordered pixel-coordinate arrays on the display canvas |
| `T` | Ordered fixation durations, milliseconds; not absolute onset timestamps |
| `length` | Number of entries; checked against all three arrays |
| `split` | `train` or `valid` in this file |
| `fixOnTarget`, `correct` | Present but not interpreted; retained as unsupported metadata |

The supplied README mentions `trial_ind` and `task`, but neither exists in this download.
They remain missing in normalized records. `record_index` is the zero-based source-list
position, not an observed trial number; `fixation_index` is the zero-based array index.
No onset timestamp, task, trial number, duration cap or participant observation is invented.
Do not derive an observed time-to-first-fixation by summing durations: saccade/gap timing
is absent. The actual duration range is 1–5,270 ms; measurements are not clipped to a
nominal five-second trial length. The initial center fixation is retained.

## Coordinates and image registration

The supplied README specifies aspect-preserving stimulus preparation with zero padding
on a 1680×1050 display. Coordinates use the top-left origin with x rightwards and y
downwards. The official shared stimuli already have that canvas size, so loaded gaze
coordinates map identically to those photographs, including any display padding.
OpenCV decodes without applying EXIF rotation, avoiding a silent coordinate change.

For a differently sized original photograph, the loader refuses to guess alignment.
Supply the actual rendered content rectangle `(left, top, width, height)` in display
pixels, obtained from stimulus-preparation metadata. If the original photograph has
size `(W, H)`, the inverse mapping is:

```text
x_image = (X - left) * W / width
y_image = (Y - top)  * H / height
x_normalized = x_image / W
y_normalized = y_image / H
```

Bounds are half-open. Display outliers, missing/nonfinite values, nonpositive durations
and fixations in padding outside the photograph are flagged. No clipping, interpolation,
imputation, rescaling based only on aspect ratio or unreported row removal occurs.

## Python API and outputs

```python
from edugaze.datasets import CocoFreeView, ImageGeometry

data = CocoFreeView(
    'data/raw/coco_freeview/fixations/COCOFreeView_fixations_trainval.json',
    'data/raw/coco_freeview/images',
)
print(data.inventory())
records = data.select(image='000000128398.jpg', participant=101)
matched = data.load_image('000000128398.jpg', participant=101)
# matched.pixels_bgr: OpenCV BGR array; matched.fixations: aligned pandas DataFrame
# For a non-canvas photograph only, supply its actual ImageGeometry(...).
data.export_csv('data/processed/coco_freeview/image_128398.csv',
                image='000000128398.jpg', participant=101)
```

Structural errors and explicit non-freeview condition labels raise with their source
scanpath index. `strict=False` skips malformed
scanpaths and reports them in `issues`; individual invalid measurements remain with
validity flags. Unknown keys and unprovided optional metadata remain unaltered in
`metadata` and the CSV's provenance sidecar. Missing or ambiguous photographs fail
alignment/export explicitly while remaining visible in the inventory.

Normalized exports contain source/within-trial indexes, original identifiers and split,
display coordinates, image pixels, unit coordinates, dimensions, relative image path,
durations, and validity flags. All rows are retained. The JSON sidecar records the source
checksum, filters, transformations, skipped scanpaths and original non-array metadata.
Existing exports are never overwritten. No CSV or image is written into raw input folders
by the loader.

```bash
python -m unittest discover -s tests -v
python tools/verify_coco_freeview.py --image 000000128398.jpg --audit-all-images --out outputs/coco_freeview_new_check
```

`requirements-tested.txt` records the exact core dependency versions verified on Linux
Python 3.14.4. The broader compatibility ranges remain in `pyproject.toml`.

Unit fixtures are explicitly artificial parser/geometry inputs, never claimed as dataset
records. A separate integration test reads the actual local JSON and an official
photograph; it skips only when those local downloads are unavailable. Verification
outputs include a full image inventory, normalized real fixations, provenance, summary,
and one actual participant scanpath drawn on its photograph for alignment inspection.
This phase implements data integration only; attention metrics and educational evaluation
remain outside scope.

Dataset citation: Chen et al. (2022), *Characterizing Target-Absent Human Attention*,
CVPR Workshops, pp. 5031–5040. [Paper](https://openaccess.thecvf.com/content/CVPR2022W/GAZE/papers/Chen_Characterizing_Target-Absent_Human_Attention_CVPRW_2022_paper.pdf).

## Verified local result

| Split | Scanpaths | Fixations |
| --- | ---: | ---: |
| train | 37,038 | 572,184 |
| valid | 6,010 | 92,821 |
| Total | 43,048 | 665,005 |

Both official stimulus archives are retained locally. All 4,317 referenced photographs
were extracted, decoded with OpenCV, checked at 1680×1050, and compared with their
acquisition SHA-256 fingerprints. No missing/ambiguous matches, invalid display
coordinates or invalid durations were found. Twenty-two referenced filenames occur in
both archives; their copies were also compared byte for byte and are identical.

- Full normalized CSV (665,005 rows, 110,486,036 bytes):
  `data/processed/coco_freeview/fixations_trainval.normalized.csv`
- Export provenance: the same path plus `.metadata.json`.
- Full available-image inventory, real-image overlay, sample CSV and audit summary:
  `outputs/coco_freeview_complete_verification/`.
- Full-export row/order/measurement comparison: `export_validation.json` in that folder.
- Cross-archive alias check: `archive_alias_validation.json` in that folder.

The example photograph `000000128398.jpg` has 148 recorded fixations across ten
participants. Its overlay shows participant 101's first recorded scanpath (15 fixations)
with an identity display-to-image mapping. The first measured sample has X=821.5,
Y=559.2 and duration 316 ms; these are read from the real source, not generated data.
The full export was reloaded and compared with every parsed source row: identifiers,
source ordering, coordinates and durations matched, and unobserved trial/task fields
remained missing. All 26 automated tests passed, including the actual local-data test;
`pip check` found no broken requirements. Phase 3 has not been started.
