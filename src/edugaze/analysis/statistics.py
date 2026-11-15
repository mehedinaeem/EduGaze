"""Descriptive fixation and consecutive-fixation displacement statistics."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._records import TRIAL_KEYS, duration_statistics, first_fixation_latency, prepare_records

SACCADE_COLUMNS = TRIAL_KEYS + ['from_fixation_index', 'to_fixation_index', 'distance_px']


@dataclass
class GazeStatistics:
    summary: dict
    participants: pd.DataFrame
    trials: pd.DataFrame
    saccades: pd.DataFrame
    spatial: pd.DataFrame


def _saccades(df):
    rows = []
    for _, trial in df.groupby(TRIAL_KEYS, sort=False):
        records = list(trial.itertuples(index=False))
        for before, after in zip(records, records[1:]):
            if (before.coordinate_valid and after.coordinate_valid
                    and after.fixation_index == before.fixation_index+1):
                rows.append({**{key: getattr(before, key) for key in TRIAL_KEYS},
                             'from_fixation_index': before.fixation_index,
                             'to_fixation_index': after.fixation_index,
                             'distance_px': float(np.hypot(after.x_image-before.x_image, after.y_image-before.y_image))})
    return pd.DataFrame(rows, columns=SACCADE_COLUMNS)


def saccade_distances(records, *, image_size):
    """Euclidean endpoint displacements in pixels, never across trials or gaps.

    These are fixation-to-fixation distances, not measured continuous saccade
    trajectories, velocities, or visual-angle amplitudes.
    """
    return _saccades(prepare_records(records, image_size))


def _spatial(df, image_size, grid):
    if len(grid) != 2 or any(type(v) is not int or v <= 0 for v in grid):
        raise ValueError('grid must contain positive integer x and y bin counts')
    width, height = image_size
    nx, ny = grid
    valid = df[df.coordinate_valid]
    counts, yedges, xedges = np.histogram2d(valid.y_image, valid.x_image,
                                          bins=(ny, nx), range=((0, height), (0, width)))
    return pd.DataFrame([{'column': x, 'row': y, 'x': float(xedges[x]), 'y': float(yedges[y]),
                          'width': float(xedges[x+1]-xedges[x]), 'height': float(yedges[y+1]-yedges[y]),
                          'fixation_count': int(counts[y, x]),
                          'percentage_of_valid_fixations': 100*float(counts[y, x])/len(valid) if len(valid) else None}
                         for y in range(ny) for x in range(nx)])


def spatial_distribution(records, *, image_size, grid=(6, 4)):
    return _spatial(prepare_records(records, image_size), image_size, grid)


def _summary(df, saccades, trial_stats):
    coords = df[df.coordinate_valid]
    latencies = trial_stats.time_to_first_fixation_ms.dropna()
    # A zero length is known for a singleton, but not for missing transitions.
    expected_pairs = sum(max(len(t)-1, 0) for _, t in df.groupby(TRIAL_KEYS, sort=False))
    observed_length = float(saccades.distance_px.sum())
    sequence_complete = all(np.array_equal(t.fixation_index.to_numpy(), np.arange(len(t)))
                            for _, t in df.groupby(TRIAL_KEYS, sort=False))
    complete = len(saccades) == expected_pairs and df.coordinate_valid.all() and sequence_complete
    return {**duration_statistics(df),
            'valid_coordinate_count': len(coords), 'invalid_coordinate_count': len(df)-len(coords),
            'participant_count': int(df.participant_id.nunique()), 'trial_count': len(trial_stats),
            'saccade_count': len(saccades), 'unavailable_adjacent_pair_count': expected_pairs-len(saccades),
            'mean_saccade_distance_px': float(saccades.distance_px.mean()) if len(saccades) else None,
            'observed_scanpath_length_px': observed_length,
            'total_scanpath_length_px': observed_length if complete else None,
            'scanpath_length_status': 'complete' if complete else 'partial_or_unavailable',
            'mean_x_px': float(coords.x_image.mean()) if len(coords) else None,
            'mean_y_px': float(coords.y_image.mean()) if len(coords) else None,
            'time_to_first_fixation_ms': float(latencies.iloc[0]) if len(trial_stats) == 1 and len(latencies) else None,
            'ttff_status': str(trial_stats.ttff_status.iloc[0]) if len(trial_stats) == 1 else 'multiple_trials_see_trial_table',
            'mean_trial_time_to_first_fixation_ms': float(latencies.mean()) if len(latencies) else None,
            'timed_trial_count': len(latencies)}


def analyze_gaze(records, *, image_size, grid=(6, 4)):
    """Summarize a single image without inferring missing durations or timestamps."""
    df = prepare_records(records, image_size)
    distances = _saccades(df)
    trials = []
    for keys, trial in df.groupby(TRIAL_KEYS, sort=False):
        ttff, status = first_fixation_latency(trial)
        trials.append(dict(zip(TRIAL_KEYS, keys), **duration_statistics(trial),
                           time_to_first_fixation_ms=ttff, ttff_status=status))
    trials = pd.DataFrame(trials)
    participants = []
    for participant, selected in df.groupby('participant_id', sort=False):
        stats = _summary(selected, distances[distances.participant_id.eq(participant)],
                         trials[trials.participant_id.eq(participant)])
        participants.append(dict(participant_id=participant, **stats))
    return GazeStatistics(_summary(df, distances, trials), pd.DataFrame(participants), trials,
                          distances, _spatial(df, image_size, grid))
