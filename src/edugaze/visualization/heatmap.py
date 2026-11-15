"""Spatial densities calculated from recorded human gaze, not predicted attention."""
from dataclasses import dataclass
import math

import cv2
import numpy as np

from ._common import PlotResult, coordinate_mask, new_canvas, select_rows


@dataclass
class DensityMap:
    values: np.ndarray
    metadata: dict

    @property
    def relative(self):
        """Per-map peak normalization for display only, not a probability map."""
        return self.values / self.values.max()


def compute_density(matched, *, participant=None, weighting='duration', sigma=32.0):
    """Bin registered coordinates, then smooth with a Gaussian in image pixels.

    ``duration`` adds positive measured milliseconds; ``count`` adds one per
    coordinate-valid fixation regardless of duration availability. Subpixel
    coordinates are assigned to floor(x), floor(y) pixel cells. Repeated visits
    accumulate. sigma=0 disables smoothing. A four-sigma kernel and zero exterior
    are used; tails outside the canvas are lost, not reflected or wrapped.

    participant=None pools all selected participants/trials without participant
    balancing. No coordinates, missing durations or extra observations are made up.
    """
    if weighting not in {'duration', 'count'}:
        raise ValueError('weighting must be duration or count')
    if (isinstance(sigma, bool) or not isinstance(sigma, (int, float))
            or not math.isfinite(sigma) or sigma < 0):
        raise ValueError('sigma must be a finite nonnegative number of image pixels')
    df = select_rows(matched, participant)
    valid = coordinate_mask(df, matched.pixels_bgr.shape)
    valid_duration = np.isfinite(df.duration_ms) & df.duration_ms.gt(0)
    used = valid & valid_duration if weighting == 'duration' else valid
    points = df[used]
    if points.empty:
        raise ValueError('No usable recorded fixations for this weighting')
    height, width = matched.pixels_bgr.shape[:2]
    bins = np.zeros((height, width), dtype=np.float64)
    weights = points.duration_ms.to_numpy(dtype=float) if weighting == 'duration' else np.ones(len(points))
    np.add.at(bins, (np.floor(points.y_image).astype(int), np.floor(points.x_image).astype(int)), weights)
    if sigma:
        radius = math.ceil(4 * sigma)
        bins = cv2.GaussianBlur(bins, (2 * radius + 1, 2 * radius + 1),
                                sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_CONSTANT)
    if not np.isfinite(bins).all() or bins.max() <= 0:
        raise ValueError('Density weights overflowed or produced an empty map')
    metadata = {'kind': 'heatmap', 'data_kind': 'recorded_human_gaze',
                'image_name': matched.path.name, 'weighting': weighting,
                'weight_unit': 'ms' if weighting == 'duration' else 'fixations',
                'sigma_pixels': float(sigma), 'kernel_radius_pixels': math.ceil(4 * sigma) if sigma else 0,
                'border_policy': 'zero exterior; Gaussian tails outside canvas are discarded',
                'rasterization': 'floor coordinates to pixel cells',
                'aggregation': 'pooled fixations; no equal-participant normalization',
                'participant_ids': sorted(points.participant_id.astype(str).unique().tolist()),
                'record_indices': sorted(int(v) for v in points.record_index.unique()),
                'selected_fixations': len(df), 'used_fixations': len(points),
                'excluded_coordinates': int((~valid).sum()),
                'excluded_durations': int((valid & ~valid_duration).sum()) if weighting == 'duration' else 0,
                'input_weight': float(weights.sum()), 'smoothed_weight_in_canvas': float(bins.sum()),
                'image_size': [width, height]}
    return DensityMap(bins, metadata)


def plot_heatmap(matched, *, participant=None, weighting='duration', sigma=32.0, alpha=.65):
    """Overlay an inferno-colored density; alpha scales its peak opacity.

    Opacity is alpha * relative_density, so a zero-density pixel leaves the
    photograph unchanged. Every display is normalized to its own peak (0–1),
    hence colors alone cannot compare absolute dwell time across images/groups.
    """
    if (isinstance(alpha, bool) or not isinstance(alpha, (int, float))
            or not math.isfinite(alpha) or not 0 <= alpha <= 1):
        raise ValueError('alpha must lie between 0 and 1')
    density = compute_density(matched, participant=participant, weighting=weighting, sigma=sigma)
    who = f'participant {participant}' if participant is not None else f'{len(density.metadata["participant_ids"])} participants pooled'
    figure, axes = new_canvas(matched, 'Recorded human gaze · Visual attention density',
                             f'{matched.path.name}  |  {who}  |  {weighting}-weighted  |  σ = {sigma:g} px')
    height, width = density.values.shape
    relative = density.relative
    heat = axes.imshow(relative, cmap='inferno', origin='upper', interpolation='nearest',
                       extent=(0, width, height, 0), vmin=0, vmax=1, alpha=alpha * relative, zorder=2)
    # Use a separate opaque scalar mappable so the colorbar is independent of image alpha.
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    bar_axes = figure.add_axes((.22, .82 / figure.get_figheight(), .63, .12 / figure.get_figheight()))
    colorbar = figure.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap='inferno'),
                              cax=bar_axes, orientation='horizontal')
    colorbar.set_label(f'Relative {weighting}-weighted fixation density (per-map peak = 1)', fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    figure.text(.09, .12/figure.get_figheight(), f'{density.metadata["used_fixations"]} recorded fixations · '
                'pooled observations · no model-predicted attention', fontsize=8, color='#394756')
    metadata = dict(density.metadata, alpha=float(alpha), display_normalization='per-map peak = 1')
    return PlotResult(figure, axes, metadata)
