"""Publication figures derived only from aligned recorded human gaze."""
from ._common import PlotResult, save_png
from .heatmap import DensityMap, compute_density, plot_heatmap
from .scanpath import plot_scanpath

__all__ = ['PlotResult', 'DensityMap', 'compute_density', 'plot_scanpath', 'plot_heatmap', 'save_png']
