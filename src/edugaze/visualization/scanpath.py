"""Single-trial scanpaths of recorded human fixations, never model predictions."""
from matplotlib.collections import LineCollection
import matplotlib.patheffects as effects
import numpy as np

from ._common import PlotResult, coordinate_mask, new_canvas, select_rows


def plot_scanpath(matched, *, participant=None, record_index=None, show_durations=True):
    """Number original fixation indices (one-based) and join adjacent valid points.

    One participant and one source scanpath must be selected. Gaps left by invalid
    coordinates are not bridged, and missing durations are not imputed. Displayed
    order is recorded fixation order; this dataset contains no onset timestamps.
    """
    df = select_rows(matched, participant)
    if df.participant_id.nunique() != 1:
        raise ValueError('Scanpaths require one participant; select participant explicitly')
    if record_index is not None:
        df = df[df.record_index.eq(record_index)]
    if df.empty:
        raise ValueError('No recorded scanpath matches record_index')
    if df.record_index.nunique() != 1:
        raise ValueError('Multiple trials selected; specify one source record_index')
    indexes = df.fixation_index.to_numpy(dtype=float)
    if (not np.isfinite(indexes).all() or (indexes < 0).any()
            or (indexes != np.floor(indexes)).any() or df.fixation_index.duplicated().any()):
        raise ValueError('Within-trial fixation indices must be unique nonnegative integers')
    df = df.sort_values('fixation_index', kind='stable')
    valid = coordinate_mask(df, matched.pixels_bgr.shape)
    points = df[valid]
    if points.empty:
        raise ValueError('No usable recorded fixation coordinates')
    who = str(df.participant_id.iloc[0])
    record = int(df.record_index.iloc[0])
    table_rows = (len(points) + 2) // 3
    bottom = 1.2 + (.2 + .16 * table_rows if show_durations else 0)
    figure, axes = new_canvas(matched, 'Recorded human gaze · Scanpath',
                             f'{matched.path.name}  |  participant {who}  |  source record {record}',
                             bottom_inches=bottom)
    xy = points[['x_image', 'y_image']].to_numpy()
    order = points.fixation_index.to_numpy(dtype=int)
    segments = [xy[i:i+2] for i in range(len(xy)-1) if order[i+1] == order[i]+1]
    lines = LineCollection(segments, colors='#fff1a8', linewidths=1.8, zorder=2)
    lines.set_path_effects([effects.Stroke(linewidth=3.2, foreground='#17202a'), effects.Normal()])
    axes.add_collection(lines)
    axes.scatter(xy[:, 0], xy[:, 1], s=90, facecolor='#173f5f', edgecolor='white',
                 linewidth=1.0, alpha=.94, zorder=3)
    available_durations = 0
    if show_durations:
        figure.text(.09, (bottom-.8)/figure.get_figheight(), 'Fixation duration (recorded milliseconds)',
                    fontsize=8, fontweight='bold', color='#173f5f')
    for position, row in enumerate(points.itertuples()):
        number = int(row.fixation_index) + 1
        axes.annotate(str(number), (row.x_image, row.y_image),
                      color='white', fontsize=5.5, fontweight='bold', ha='center', va='center', zorder=4)
        if show_durations:
            has_duration = np.isfinite(row.duration_ms) and row.duration_ms > 0
            available_durations += int(has_duration)
            duration = f'{row.duration_ms:g} ms' if has_duration else 'unavailable'
            column, table_row = divmod(position, table_rows)
            figure.text(.09 + .29*column, (bottom-1.0-.16*table_row)/figure.get_figheight(),
                        f'{number}: {duration}', fontsize=8, color='#173f5f')
    excluded = int((~valid).sum())
    figure.text(.09, .12/figure.get_figheight(), f'{len(points)} recorded fixations · {excluded} excluded coordinates · '
                'numbers follow source order; gaps are retained', fontsize=8, color='#394756')
    metadata = {'kind': 'scanpath', 'data_kind': 'recorded_human_gaze',
                'image_name': matched.path.name, 'participant_id': who, 'record_index': record,
                'selected_fixations': len(df), 'plotted_fixations': len(points),
                'excluded_coordinates': excluded, 'line_segments': len(segments),
                'duration_labels': available_durations, 'duration_unit': 'ms',
                'numbering': 'one-based original fixation index; no timestamps inferred',
                'image_size': list(matched.pixels_bgr.shape[1::-1])}
    return PlotResult(figure, axes, metadata)
