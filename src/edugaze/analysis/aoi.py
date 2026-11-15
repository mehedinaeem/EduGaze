"""Rectangular, nonoverlapping areas of interest in registered image pixels."""
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ._records import TRIAL_KEYS, duration_statistics, first_fixation_latency, prepare_records
from .statistics import _saccades


@dataclass(frozen=True)
class AOI:
    aoi_id: str
    name: str
    x: float
    y: float
    width: float
    height: float

    def __post_init__(self):
        for field in ('aoi_id', 'name'):
            if not isinstance(getattr(self, field), str) or not getattr(self, field).strip():
                raise ValueError(f'{field} must be a nonempty string')
        for field in ('x', 'y', 'width', 'height'):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError('AOI coordinates and size must be finite numbers')
        if self.x < 0 or self.y < 0 or self.width <= 0 or self.height <= 0:
            raise ValueError('AOI coordinates must be nonnegative and size positive')


def validate_aois(aois, *, image_size):
    if not aois:
        raise ValueError('At least one AOI is required')
    if len(image_size) != 2 or any(type(v) is not int or v <= 0 for v in image_size):
        raise ValueError('image_size must contain positive integer width and height')
    identifiers = [a.aoi_id for a in aois]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError('AOI IDs must be unique')
    width, height = image_size
    for i, a in enumerate(aois):
        if a.x+a.width > width or a.y+a.height > height:
            raise ValueError(f'AOI {a.aoi_id} lies outside the image canvas')
        for b in aois[i+1:]:
            if (max(a.x, b.x) < min(a.x+a.width, b.x+b.width)
                    and max(a.y, b.y) < min(a.y+a.height, b.y+b.height)):
                raise ValueError(f'Overlapping AOIs are ambiguous: {a.aoi_id}, {b.aoi_id}')
    return list(aois)


def load_aois(path, *, image_size):
    """Read canonical CSV or JSON list: aoi_id,name,x,y,width,height.

    IDs stay strings (CSV leading zeros are preserved). Rectangles are in the
    photograph's registered coordinates; display-space boxes must be transformed
    explicitly before use with a differently sized original photograph.
    """
    path = Path(path)
    if path.suffix.lower() == '.csv':
        with path.open(encoding='utf-8-sig', newline='') as stream:
            items = list(csv.DictReader(stream))
        for item in items:
            for key in ('x', 'y', 'width', 'height'):
                if key in item:
                    try:
                        item[key] = float(item[key])
                    except (ValueError, TypeError) as exc:
                        raise ValueError(f'Invalid numeric AOI field: {key}') from exc
    elif path.suffix.lower() == '.json':
        with path.open(encoding='utf-8-sig') as stream:
            items = json.load(stream)
    else:
        raise ValueError('AOI definitions must be CSV or JSON')
    if not isinstance(items, list):
        raise ValueError('AOI JSON must be a list of rectangle objects')
    fields = {'aoi_id', 'name', 'x', 'y', 'width', 'height'}
    aois = []
    for index, item in enumerate(items):
        if not isinstance(item, dict) or set(item) != fields:
            raise ValueError(f'AOI {index} requires exactly {sorted(fields)}')
        aois.append(AOI(**item))
    return validate_aois(aois, image_size=image_size)


@dataclass
class AOIAnalysis:
    summary: pd.DataFrame
    participants: pd.DataFrame
    trials: pd.DataFrame
    assignments: pd.DataFrame
    transitions: pd.DataFrame
    transition_counts: pd.DataFrame


def _latency_summary(trials):
    values = trials.time_to_first_fixation_ms.dropna()
    return {'time_to_first_fixation_ms': float(values.iloc[0]) if len(trials) == 1 and len(values) else None,
            'ttff_status': str(trials.ttff_status.iloc[0]) if len(trials) == 1 else 'multiple_trials_see_trial_table',
            'mean_trial_time_to_first_fixation_ms': float(values.mean()) if len(values) else None,
            'timed_viewed_trial_count': len(values)}


def analyze_aois(records, aois, *, image_size):
    """Count hits/dwell and direct consecutive-fixation AOI transitions.

    Rectangle bounds are [x, x+width), [y, y+height). Outside/invalid fixations
    interrupt transitions; they are never skipped to invent an AOI-to-AOI edge.
    viewed means an observed valid fixation hit, not proof of conscious perception.
    """
    aois = validate_aois(aois, image_size=image_size)
    df = prepare_records(records, image_size)
    df['aoi_id'] = pd.Series([None]*len(df), dtype=object)
    for a in aois:
        hits = (df.coordinate_valid & df.x_image.ge(a.x) & df.x_image.lt(a.x+a.width)
                & df.y_image.ge(a.y) & df.y_image.lt(a.y+a.height))
        df.loc[hits, 'aoi_id'] = a.aoi_id
    trial_rows = []
    for keys, trial in df.groupby(TRIAL_KEYS, sort=False):
        for a in aois:
            hit = trial.aoi_id.eq(a.aoi_id)
            ttff, status = first_fixation_latency(trial, hit)
            trial_rows.append(dict(zip(TRIAL_KEYS, keys), aoi_id=a.aoi_id, aoi_name=a.name,
                                   **duration_statistics(trial[hit]), viewed=bool(hit.any()),
                                   time_to_first_fixation_ms=ttff, ttff_status=status))
    trials = pd.DataFrame(trial_rows)
    summary, participants = [], []
    for a in aois:
        hit = df.aoi_id.eq(a.aoi_id)
        aoi_trials = trials[trials.aoi_id.eq(a.aoi_id)]
        summary.append(dict(asdict(a), **duration_statistics(df[hit]),
                            percentage_of_fixations=100*int(hit.sum())/len(df),
                            percentage_of_valid_fixations=100*int(hit.sum())/int(df.coordinate_valid.sum()) if df.coordinate_valid.any() else None,
                            participants_viewing=int(df[hit].participant_id.nunique()),
                            participant_count=int(df.participant_id.nunique()),
                            viewed_trial_count=int(aoi_trials.viewed.sum()),
                            **_latency_summary(aoi_trials)))
        for participant, selected in df.groupby('participant_id', sort=False):
            inside = selected.aoi_id.eq(a.aoi_id)
            p_trials = aoi_trials[aoi_trials.participant_id.eq(participant)]
            participants.append(dict(participant_id=participant, aoi_id=a.aoi_id, aoi_name=a.name,
                                     **duration_statistics(selected[inside]), viewed=bool(inside.any()),
                                     percentage_of_fixations=100*int(inside.sum())/len(selected),
                                     **_latency_summary(p_trials)))
    distances = _saccades(df)
    lookup = {tuple(row[k] for k in TRIAL_KEYS)+ (int(row['fixation_index']),): row['aoi_id']
              for row in df.to_dict('records')}
    transitions = []
    for row in distances.to_dict('records'):
        key = tuple(row[k] for k in TRIAL_KEYS)
        before = lookup[key + (row['from_fixation_index'],)]
        after = lookup[key + (row['to_fixation_index'],)]
        if before is not None and after is not None and before != after:
            transitions.append(dict(row, from_aoi_id=before, to_aoi_id=after))
    transitions = pd.DataFrame(transitions, columns=list(distances.columns)+['from_aoi_id', 'to_aoi_id'])
    if len(transitions):
        counts = transitions.groupby(['from_aoi_id', 'to_aoi_id'], sort=True).agg(
            transition_count=('distance_px', 'size'), participants=('participant_id', 'nunique')).reset_index()
    else:
        counts = pd.DataFrame(columns=['from_aoi_id', 'to_aoi_id', 'transition_count', 'participants'])
    return AOIAnalysis(pd.DataFrame(summary), pd.DataFrame(participants), trials, df,
                       transitions, counts)
