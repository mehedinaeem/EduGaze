"""COCO-FreeView parsing and explicit display-to-photograph registration.

Original measurements are retained, including invalid values. Validity flags are
computed without clipping or interpolation. Durations are not onset timestamps.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

DISPLAY_WIDTH, DISPLAY_HEIGHT = 1680, 1050
FIELDS = {'trial_ind', 'name', 'subject', 'task', 'condition', 'X', 'Y', 'T', 'length', 'split'}
COLUMNS = ['record_index', 'trial_ind', 'image_name', 'participant_id', 'task',
           'condition', 'split', 'fixation_index', 'x_display', 'y_display',
           'duration_ms']


def _identifier(value):
    if isinstance(value, bool) or value is None:
        raise ValueError('subject must be a nonempty string or finite integer')
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, int):
        return str(value)
    raise ValueError('subject must be a nonempty string or finite integer')


def _number(value):
    # Numeric strings are unsupported JSON measurements, not silently coerced.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return np.nan
    try:
        return float(value)
    except OverflowError:
        return np.nan


@dataclass(frozen=True)
class ImageGeometry:
    """Actual stimulus content rectangle in the 1680x1050 display canvas.

    Supply measured rendering metadata for original photographs. A rectangle is
    never inferred merely from the photograph's aspect ratio. The full rectangle
    maps to the full decoded image. Padding coordinates are flagged, not clipped.
    """
    left: float
    top: float
    width: float
    height: float

    def __post_init__(self):
        values = (self.left, self.top, self.width, self.height)
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not math.isfinite(v) for v in values):
            raise ValueError('Content rectangle must contain four finite numbers')
        if (self.left < 0 or self.top < 0 or self.width <= 0 or self.height <= 0
                or self.left + self.width > DISPLAY_WIDTH
                or self.top + self.height > DISPLAY_HEIGHT):
            raise ValueError('Content rectangle must lie within the display canvas')


@dataclass
class MatchedImage:
    path: Path
    pixels_bgr: np.ndarray
    fixations: pd.DataFrame
    geometry: ImageGeometry


class CocoFreeView:
    """One official JSON list, preserving scanpath/trial boundaries and row order.

    ``strict=True`` rejects structurally malformed scanpaths; ``strict=False``
    skips them and records each rejection in ``issues``. Missing/non-numeric
    individual measurements remain NaN with validity flags in either mode.
    Unsupported fields are retained in ``metadata`` and never interpreted.
    """
    def __init__(self, json_path, image_root, *, strict=True):
        self.json_path = Path(json_path).resolve()
        self.image_root = Path(image_root).resolve()
        # Stream hashing avoids an extra full in-memory copy of the source JSON.
        digest = hashlib.sha256()
        with self.json_path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        self.source_sha256 = digest.hexdigest()
        with self.json_path.open(encoding='utf-8-sig') as stream:
            items = json.load(stream)
        if not isinstance(items, list):
            raise ValueError('COCO-FreeView JSON must be a list of scanpath objects')
        self.issues = []
        self.metadata = []
        rows = []
        for record_index, item in enumerate(items):
            try:
                self._validate_trial(item)
            except ValueError as exc:
                issue = {'record_index': record_index, 'error': str(exc)}
                if strict:
                    raise ValueError(f'Scanpath {record_index}: {exc}') from exc
                self.issues.append(issue)
                continue
            participant = _identifier(item['subject'])
            self.metadata.append({'record_index': record_index,
                                  'fields': {k: v for k, v in item.items() if k not in {'X', 'Y', 'T'}},
                                  'unsupported_fields': sorted(set(item) - FIELDS)})
            for index, (x, y, duration) in enumerate(zip(item['X'], item['Y'], item['T'])):
                rows.append((record_index, item.get('trial_ind'), item['name'], participant,
                             item.get('task'), item.get('condition'), item.get('split'),
                             index, _number(x), _number(y), _number(duration)))
        self.fixations = pd.DataFrame.from_records(rows, columns=COLUMNS)
        self.fixations[['x_display', 'y_display', 'duration_ms']] = self.fixations[
            ['x_display', 'y_display', 'duration_ms']].astype(float)
        df = self.fixations
        df['valid_display'] = (np.isfinite(df.x_display) & np.isfinite(df.y_display)
                               & df.x_display.ge(0) & df.x_display.lt(DISPLAY_WIDTH)
                               & df.y_display.ge(0) & df.y_display.lt(DISPLAY_HEIGHT))
        df['valid_duration'] = np.isfinite(df.duration_ms) & df.duration_ms.gt(0)
        df['valid_fixation'] = df.valid_display & df.valid_duration
        self._image_rows = self.fixations.groupby('image_name', sort=False).indices
        self._images = {}
        if self.image_root.is_dir():
            for path in sorted(self.image_root.rglob('*')):
                if path.is_file() and path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}:
                    self._images.setdefault(path.name, []).append(path)

    @staticmethod
    def _validate_trial(item):
        if not isinstance(item, dict):
            raise ValueError('Expected a scanpath object')
        missing = {'name', 'subject', 'X', 'Y', 'T'} - set(item)
        if missing:
            raise ValueError(f'Missing required fields: {sorted(missing)}')
        name = item['name']
        if (not isinstance(name, str) or not name.strip() or '/' in name
                or '\\' in name or name in {'.', '..'}):
            raise ValueError('name must be an exact photograph basename')
        _identifier(item['subject'])
        if not all(isinstance(item[k], list) for k in ('X', 'Y', 'T')):
            raise ValueError('X, Y and T must be arrays')
        length = len(item['X'])
        if len(item['Y']) != length or len(item['T']) != length:
            raise ValueError('X/Y/T array lengths differ')
        if length == 0:
            raise ValueError('Empty scanpath')
        if 'length' in item and (type(item['length']) is not int or item['length'] != length):
            raise ValueError('length does not match fixation arrays')
        if 'trial_ind' in item and (type(item['trial_ind']) is not int or item['trial_ind'] < 0):
            raise ValueError('trial_ind must be a nonnegative integer when supplied')
        if item.get('condition') not in (None, 'freeview'):
            raise ValueError('Expected condition=freeview; search-task records are not COCO-FreeView')
        for field in ('task', 'condition', 'split'):
            if field in item and item[field] is not None and not isinstance(item[field], str):
                raise ValueError(f'{field} must be a string or null')

    def select(self, *, image=None, participant=None):
        """Return a copy; preserve source record and within-trial fixation indexes."""
        df = self.fixations
        if image is not None:
            df = df.iloc[self._image_rows.get(image, np.array([], dtype=int))]
        if participant is not None:
            df = df[df.participant_id.eq(_identifier(participant))]
        return df.copy()

    def image_path(self, name):
        candidates = self._images.get(name, [])
        if not candidates:
            raise FileNotFoundError(f'No exact photograph match for {name!r} under {self.image_root}')
        if len(candidates) != 1:
            raise ValueError(f'Ambiguous photograph {name!r}: {len(candidates)} exact basename matches')
        path = candidates[0].resolve()
        if not path.is_relative_to(self.image_root):
            raise ValueError('Photograph symlink points outside image_root')
        return path

    def inventory(self):
        """One row per recorded image, including missing/ambiguous image status."""
        if self.fixations.empty:
            return pd.DataFrame(columns=['image_name', 'scanpaths', 'participants', 'fixations',
                                         'valid_fixations', 'image_matches', 'image_status'])
        df = self.fixations.groupby('image_name', sort=True).agg(
            scanpaths=('record_index', 'nunique'), participants=('participant_id', 'nunique'),
            fixations=('fixation_index', 'size'), valid_fixations=('valid_fixation', 'sum')).reset_index()
        df['image_matches'] = df.image_name.map(lambda n: len(self._images.get(n, [])))
        df['image_status'] = df.image_matches.map(lambda n: 'missing' if n == 0 else 'matched' if n == 1 else 'ambiguous')
        return df

    def load_image(self, name, *, participant=None, geometry=None):
        df = self.select(image=name, participant=participant)
        if df.empty:
            raise ValueError('No fixation records match the selected image/participant')
        path = self.image_path(name)
        pixels = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if pixels is None:
            raise ValueError(f'OpenCV cannot decode photograph: {path}')
        height, width = pixels.shape[:2]
        if geometry is None:
            if (width, height) != (DISPLAY_WIDTH, DISPLAY_HEIGHT):
                raise ValueError(f'Image is {width}x{height}, not the official 1680x1050 canvas; '
                                 'supply the actual display content rectangle as ImageGeometry')
            geometry = ImageGeometry(0, 0, DISPLAY_WIDTH, DISPLAY_HEIGHT)
        if not isinstance(geometry, ImageGeometry):
            raise TypeError('geometry must be an ImageGeometry')
        if ((width, height) == (DISPLAY_WIDTH, DISPLAY_HEIGHT)
                and geometry == ImageGeometry(0, 0, DISPLAY_WIDTH, DISPLAY_HEIGHT)):
            # Preserve exact recorded floats for an identity registration.
            df['x_image'] = df.x_display
            df['y_image'] = df.y_display
        else:
            df['x_image'] = (df.x_display - geometry.left) * width / geometry.width
            df['y_image'] = (df.y_display - geometry.top) * height / geometry.height
        df['x_normalized'] = df.x_image / width
        df['y_normalized'] = df.y_image / height
        df['inside_image'] = (df.valid_display & df.x_image.ge(0) & df.x_image.lt(width)
                              & df.y_image.ge(0) & df.y_image.lt(height))
        df['valid_for_image'] = df.inside_image & df.valid_duration
        df['image_width'] = width
        df['image_height'] = height
        df['image_path'] = str(path.relative_to(self.image_root))
        return MatchedImage(path, pixels, df, geometry)

    def export_csv(self, destination, *, image=None, participant=None, geometries=None):
        """Export aligned rows and JSON provenance. Missing images fail explicitly.

        All rows, including invalid/padding points, are exported with flags. An
        output is never silently overwritten. ``geometries`` maps image names to
        actual display rectangles for non-canvas photographs.
        """
        destination = Path(destination)
        sidecar = destination.with_suffix(destination.suffix + '.metadata.json')
        if destination.exists() or sidecar.exists():
            raise FileExistsError(f'Export already exists: {destination} or its metadata')
        selected = self.select(image=image, participant=participant)
        if selected.empty:
            raise ValueError('No fixation records match the requested export')
        parts, transforms = [], {}
        for name in selected.image_name.unique():
            matched = self.load_image(name, participant=participant,
                                      geometry=(geometries or {}).get(name))
            parts.append(matched.fixations)
            transforms[name] = {'content_rectangle': vars(matched.geometry),
                                'image_path': str(matched.path.relative_to(self.image_root))}
        result = pd.concat(parts).sort_values(['record_index', 'fixation_index'], kind='stable')
        exported_record_indexes = set(result.record_index.unique())
        provenance = {'source_json': str(self.json_path), 'source_sha256': self.source_sha256,
                      'display_size': [DISPLAY_WIDTH, DISPLAY_HEIGHT],
                      'duration_unit': 'ms', 'coordinate_origin': 'top-left',
                      'fixation_index_base': 0, 'timing': 'durations only; no observed onset timestamps',
                      'filters': {'image': image, 'participant': participant},
                      'rows': len(result), 'transforms': transforms, 'issues': self.issues,
                      'scanpath_metadata': [m for m in self.metadata
                                            if m['record_index'] in exported_record_indexes]}
        metadata_json = json.dumps(provenance, indent=2, allow_nan=False) + '\n'
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('x', encoding='utf-8', newline='') as stream:
            result.to_csv(stream, index=False, na_rep='')
        with sidecar.open('x', encoding='utf-8') as stream:
            stream.write(metadata_json)
        return result
