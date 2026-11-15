"""Shared plotting and export utilities for registered, recorded human gaze."""
from dataclasses import dataclass
import json
from pathlib import Path

import cv2
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np

from edugaze.datasets import MatchedImage


@dataclass
class PlotResult:
    figure: Figure
    axes: object
    metadata: dict


def select_rows(matched: MatchedImage, participant=None):
    image = matched.pixels_bgr
    if (image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3
            or min(image.shape[:2]) <= 0):
        raise ValueError('Expected a nonempty uint8 OpenCV BGR photograph')
    df = matched.fixations
    required = {'image_name', 'participant_id', 'record_index', 'fixation_index',
                'x_image', 'y_image', 'duration_ms', 'inside_image',
                'image_width', 'image_height'}
    if not required.issubset(df):
        raise ValueError(f'Use the Phase 2 aligned loader; missing: {sorted(required - set(df))}')
    height, width = image.shape[:2]
    if (not df.image_width.eq(width).all() or not df.image_height.eq(height).all()
            or not df.image_name.eq(matched.path.name).all()):
        raise ValueError('Fixation metadata does not match this photograph')
    if participant is not None:
        df = df[df.participant_id.eq(str(participant))]
    if df.empty:
        raise ValueError('No recorded fixations match the selected participant')
    return df.copy()


def coordinate_mask(df, image_shape):
    height, width = image_shape[:2]
    return (df.inside_image.eq(True) & np.isfinite(df.x_image) & np.isfinite(df.y_image)
            & df.x_image.ge(0) & df.x_image.lt(width)
            & df.y_image.ge(0) & df.y_image.lt(height))


def new_canvas(matched, title, subtitle, *, bottom_inches=1.55):
    height, width = matched.pixels_bgr.shape[:2]
    image_inches = 9 * .86 * height / width
    figure_height = image_inches + .7 + bottom_inches
    figure = Figure(figsize=(9, figure_height), facecolor='white')
    FigureCanvasAgg(figure)  # Headless rendering without changing the global backend.
    axes = figure.add_axes((.09, bottom_inches / figure_height, .86, image_inches / figure_height))
    axes.imshow(cv2.cvtColor(matched.pixels_bgr, cv2.COLOR_BGR2RGB),
                origin='upper', extent=(0, width, height, 0), interpolation='nearest')
    axes.set(xlim=(0, width), ylim=(height, 0), xlabel='Image x (pixels)', ylabel='Image y (pixels)')
    axes.tick_params(labelsize=8)
    axes.set_title(subtitle, fontsize=9, pad=9)
    figure.suptitle(title, fontsize=14, fontweight='bold', y=.98)
    return figure, axes


def save_png(plot: PlotResult, destination, *, dpi=300):
    """Save an uncropped PNG at the requested DPI, with embedded provenance.

    The default 9-inch-wide figure exports at 2700 pixels / 300 DPI. Matplotlib
    renders text and line art at export resolution. Existing files are preserved.
    """
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi <= 0:
        raise ValueError('dpi must be a positive integer')
    destination = Path(destination)
    if destination.suffix.lower() != '.png':
        raise ValueError('Visualization destination must end in .png')
    description = json.dumps(plot.metadata, sort_keys=True, allow_nan=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as stream:
        plot.figure.savefig(stream, format='png', dpi=dpi, facecolor='white',
                            metadata={'Software': 'EduGaze', 'Description': description})
    return destination
