"""CSV/JSON reports and publication charts for recorded gaze/AOI statistics."""
import json
from pathlib import Path

import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from edugaze.visualization import PlotResult, save_png
from .aoi import analyze_aois
from .statistics import analyze_gaze


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    return value


def _table_records(table):
    return table.astype(object).where(pd.notna(table), None).to_dict('records')


def _figure():
    figure = Figure(figsize=(9, 6), layout='constrained', facecolor='white')
    FigureCanvasAgg(figure)
    return figure, figure.add_subplot()


def _save(figure, axes, path, kind, dpi):
    save_png(PlotResult(figure, axes, {'kind': kind, 'data_kind': 'recorded_human_gaze'}), path, dpi=dpi)
    figure.clear()


def export_analysis(matched, aois, destination, *, grid=(6, 4), dpi=300, provenance=None):
    """Create one new report directory; existing results are never overwritten.

    AOIs and coordinates must both use the matched image's pixel canvas. Empty
    timing values are CSV blanks / JSON null, never zero-filled or inferred.
    """
    if type(dpi) is not int or dpi <= 0:
        raise ValueError('dpi must be a positive integer')
    height, width = matched.pixels_bgr.shape[:2]
    if not matched.fixations.image_name.eq(matched.path.name).all():
        raise ValueError('Fixation image identifiers do not match the photograph')
    stats = analyze_gaze(matched.fixations, image_size=(width, height), grid=grid)
    aoi = analyze_aois(matched.fixations, aois, image_size=(width, height))
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    tables = {'gaze_summary': pd.DataFrame([stats.summary]), 'participant_statistics': stats.participants,
              'trial_statistics': stats.trials, 'saccade_distances': stats.saccades,
              'spatial_distribution': stats.spatial, 'aoi_summary': aoi.summary,
              'participant_aoi_statistics': aoi.participants, 'trial_aoi_statistics': aoi.trials,
              'fixation_aoi_assignments': aoi.assignments, 'aoi_transitions': aoi.transitions,
              'aoi_transition_counts': aoi.transition_counts}
    metadata = {'image_name': matched.path.name, 'image_size': [width, height],
                'data_kind': 'recorded_human_gaze', 'coordinate_unit': 'image pixels',
                'aoi_boundary_policy': 'left/top included; right/bottom excluded; overlaps rejected',
                'duration_policy': 'positive finite observed milliseconds; unknown totals null; observed subtotal explicit',
                'ttff_policy': 'observed fixation_onset_ms minus observed trial_start_ms; complete valid trials only',
                'transition_policy': 'different AOIs at consecutive valid source indices within one trial; outside interrupts',
                'aoi_percentage_denominator': 'all recorded fixations; valid-coordinate percentage also provided',
                'assigned_fixations': int(aoi.assignments.aoi_id.notna().sum()),
                'valid_fixations_outside_aois': int((aoi.assignments.coordinate_valid & aoi.assignments.aoi_id.isna()).sum()),
                'invalid_coordinate_fixations': int((~aoi.assignments.coordinate_valid).sum()),
                'scanpath_length_policy': 'Euclidean fixation endpoint distances in pixels; no gaps or trial joins',
                'provenance': provenance or {}}
    for name, table in tables.items():
        table.to_csv(destination/f'{name}.csv', index=False, na_rep='')
    report = {'metadata': metadata, 'summary': stats.summary,
              'tables': {key: _table_records(value) for key, value in tables.items()}}
    (destination/'report.json').write_text(json.dumps(_json_safe(report), indent=2, allow_nan=False)+'\n')

    figure, axes = _figure()
    axes.imshow(cv2.cvtColor(matched.pixels_bgr, cv2.COLOR_BGR2RGB), origin='upper',
                extent=(0, width, height, 0))
    for i, a in enumerate(aois):
        color = ['#00c7be', '#ffcc00', '#ff6482', '#64d2ff'][i % 4]
        axes.add_patch(Rectangle((a.x, a.y), a.width, a.height, facecolor='none', edgecolor=color, linewidth=2))
        axes.text(a.x+5, a.y+15, a.aoi_id, va='top', fontsize=9, color=color,
                  bbox={'facecolor':'black','alpha':.7,'edgecolor':'none','pad':2})
    axes.set(xlim=(0,width), ylim=(height,0), xlabel='Image x (pixels)', ylabel='Image y (pixels)',
             title=f'User-defined rectangular AOIs · {matched.path.name}')
    _save(figure, axes, destination/'aoi_overlay.png', 'aoi_definitions', dpi)

    figure, axes = _figure()
    axes.barh(aoi.summary.aoi_id, aoi.summary.fixation_count, color='#173f5f')
    axes.set(xlabel='Recorded fixation count', ylabel='AOI ID', title='Recorded fixations per AOI')
    for i, count in enumerate(aoi.summary.fixation_count):
        axes.text(count, i, f' {count}', va='center', fontsize=9)
    _save(figure, axes, destination/'aoi_fixation_counts.png', 'aoi_fixation_counts', dpi)

    figure, axes = _figure()
    observed = aoi.summary.observed_fixation_duration_ms
    present = observed.notna()
    positions = np.arange(len(observed))
    axes.barh(positions[present], observed[present], color='#267f82')
    axes.set_yticks(positions, labels=aoi.summary.aoi_id)
    for i, row in aoi.summary.iterrows():
        label = 'unavailable' if pd.isna(row.observed_fixation_duration_ms) else f'{row.observed_fixation_duration_ms:g} ms ({row.duration_status})'
        axes.text(0 if pd.isna(row.observed_fixation_duration_ms) else row.observed_fixation_duration_ms,
                  i, f' {label}', va='center', fontsize=8)
    axes.margins(x=.35)
    axes.set(xlabel='Sum of available positive measured durations (ms)', ylabel='AOI ID',
             title='Observed fixation duration per AOI · coverage shown in labels')
    _save(figure, axes, destination/'aoi_durations.png', 'observed_aoi_durations', dpi)

    figure, axes = _figure()
    values = stats.spatial.fixation_count.to_numpy().reshape(grid[1], grid[0])
    heat = axes.imshow(values, origin='upper', extent=(0,width,height,0), interpolation='nearest', cmap='viridis')
    figure.colorbar(heat, ax=axes, label='Recorded fixation count per grid cell')
    axes.set(xlabel='Image x (pixels)', ylabel='Image y (pixels)', title='Spatial fixation distribution · unsmoothed counts')
    _save(figure, axes, destination/'spatial_distribution.png', 'spatial_fixation_counts', dpi)

    figure, axes = _figure()
    identifiers = [a.aoi_id for a in aois]
    positions = {name:i for i, name in enumerate(identifiers)}
    matrix = np.zeros((len(aois),len(aois)), dtype=int)
    for row in aoi.transition_counts.itertuples():
        matrix[positions[row.from_aoi_id], positions[row.to_aoi_id]] = row.transition_count
    heat = axes.imshow(matrix, cmap='Blues', vmin=0, vmax=max(int(matrix.max()),1))
    for y in range(len(aois)):
        for x in range(len(aois)):
            axes.text(x,y,str(matrix[y,x]),ha='center',va='center',color='white' if matrix[y,x] > matrix.max()/2 else 'black')
    axes.set_xticks(range(len(aois)), labels=identifiers, rotation=35, ha='right')
    axes.set_yticks(range(len(aois)), labels=identifiers)
    axes.set(xlabel='To AOI', ylabel='From AOI', title='Direct transitions between different AOIs')
    figure.colorbar(heat, ax=axes, label='Recorded consecutive-fixation transition count')
    _save(figure, axes, destination/'aoi_transitions.png', 'direct_aoi_transitions', dpi)
    return report
