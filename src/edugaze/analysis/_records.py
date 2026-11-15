"""Canonical aligned-record validation and explicit observed timing support."""
import numpy as np
import pandas as pd

TRIAL_KEYS = ['image_name', 'participant_id', 'record_index']


def prepare_records(records, image_size):
    """Copy one image's aligned records; duration and timing are optional.

    image_size is (width, height). Coordinates are top-left image pixels. Trial
    identity plus fixation_index is required so trajectories are never invented.
    """
    if (len(image_size) != 2 or any(type(v) is not int or v <= 0 for v in image_size)):
        raise ValueError('image_size must contain positive integer width and height')
    required = set(TRIAL_KEYS + ['fixation_index', 'x_image', 'y_image'])
    if not required.issubset(records):
        raise ValueError(f'Missing aligned-record columns: {sorted(required - set(records))}')
    if records.empty:
        raise ValueError('No fixation records to analyze')
    df = records.copy()
    if df[TRIAL_KEYS].isna().any().any() or df.image_name.nunique() != 1:
        raise ValueError('Analyze one image with nonmissing participant and source trial identifiers')
    for column in ['image_name', 'participant_id']:
        df[column] = df[column].astype(str)
        if df[column].str.strip().eq('').any():
            raise ValueError(f'{column} must be nonempty')
    indices = pd.to_numeric(df.fixation_index, errors='coerce')
    if (not np.isfinite(indices).all() or indices.lt(0).any()
            or indices.ne(np.floor(indices)).any() or df.fixation_index.map(lambda x: isinstance(x, (bool, np.bool_))).any()):
        raise ValueError('fixation_index must contain nonnegative integers')
    df['fixation_index'] = indices.astype(int)
    if df.duplicated(TRIAL_KEYS + ['fixation_index']).any():
        raise ValueError('Duplicate within-trial fixation index')
    for column in ['x_image', 'y_image', 'duration_ms', 'fixation_onset_ms', 'trial_start_ms']:
        if column not in df:
            df[column] = np.nan
        else:
            values = pd.to_numeric(df[column], errors='coerce')
            df[column] = values.mask(df[column].map(lambda x: isinstance(x, (bool, np.bool_)))).astype(float)
    width, height = image_size
    for column, expected in [('image_width', width), ('image_height', height)]:
        if column in df and not df[column].eq(expected).all():
            raise ValueError('Aligned record dimensions disagree with image_size')
    df['coordinate_valid'] = (np.isfinite(df.x_image) & np.isfinite(df.y_image)
                              & df.x_image.ge(0) & df.x_image.lt(width)
                              & df.y_image.ge(0) & df.y_image.lt(height))
    if 'inside_image' in df:
        df['coordinate_valid'] &= df.inside_image.eq(True)
    df['duration_valid'] = np.isfinite(df.duration_ms) & df.duration_ms.gt(0)
    return df.sort_values(TRIAL_KEYS + ['fixation_index'], kind='stable').reset_index(drop=True)


def timing_status(trial):
    """Require observed onsets and trial origin, complete ordering and valid coordinates.

    Times must share a clock and be in milliseconds. Durations are never summed
    to estimate onsets. Incomplete/invalid trials conservatively have no TTFF.
    """
    if trial.fixation_onset_ms.isna().all() or trial.trial_start_ms.isna().all():
        return 'timing_unavailable'
    if (not np.isfinite(trial.fixation_onset_ms).all()
            or not np.isfinite(trial.trial_start_ms).all()
            or trial.trial_start_ms.nunique() != 1):
        return 'invalid_timing'
    onsets = trial.fixation_onset_ms.to_numpy()
    start = trial.trial_start_ms.iloc[0]
    if (onsets < start).any() or (np.diff(onsets) < 0).any():
        return 'invalid_timing'
    if not np.array_equal(trial.fixation_index.to_numpy(), np.arange(len(trial))):
        return 'incomplete_sequence'
    if not trial.coordinate_valid.all():
        return 'invalid_coordinates'
    return 'available'


def first_fixation_latency(trial, hit_mask=None):
    status = timing_status(trial)
    if status != 'available':
        return None, status
    selected = trial if hit_mask is None else trial[hit_mask]
    if selected.empty:
        return None, 'not_viewed'
    return float(selected.fixation_onset_ms.iloc[0] - trial.trial_start_ms.iloc[0]), 'available'


def duration_statistics(df):
    available = df[df.duration_valid].duration_ms
    n, measured = len(df), len(available)
    observed = float(available.sum()) if measured else (0.0 if n == 0 else None)
    return {'fixation_count': n, 'duration_available_count': measured,
            'duration_unavailable_count': n-measured,
            'duration_status': 'complete' if measured == n else 'partial' if measured else 'unavailable',
            'mean_fixation_duration_ms': float(available.mean()) if measured else None,
            'observed_fixation_duration_ms': observed,
            'total_fixation_duration_ms': observed if measured == n else None}
