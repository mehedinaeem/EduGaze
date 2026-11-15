"""Geometry/weighting fixtures are artificial; the integration test uses real data."""
import json
from pathlib import Path
import tempfile
import unittest

import cv2
from matplotlib.collections import LineCollection
import numpy as np
from PIL import Image

from edugaze.datasets import CocoFreeView, ImageGeometry
from edugaze.visualization import compute_density, plot_heatmap, plot_scanpath, save_png


class VisualizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.json = self.root/'artificial.json'
        self.image = self.root/'artificial.png'
        self.plots = []
        self.addCleanup(lambda: [plot.figure.clear() for plot in self.plots])

    def matched(self, trials=None, *, width=1680, height=1050, geometry=None):
        # Asymmetric BGR image verifies color conversion and y orientation.
        pixels = np.full((height, width, 3), [10, 20, 30], dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(self.image), pixels))
        default = dict(name=self.image.name, subject=1, X=[7.2, 17.8, 27.3],
                       Y=[3.4, 13.1, 23.7], T=[100, 200, 300], length=3, condition='freeview')
        self.json.write_text(json.dumps(trials if trials is not None else [default]))
        data = CocoFreeView(self.json, self.root)
        return data.load_image(self.image.name, geometry=geometry)

    def track(self, plot):
        self.plots.append(plot)
        return plot

    def test_scanpath_original_coordinates_order_lines_and_rgb(self):
        matched = self.matched()
        matched.fixations = matched.fixations.iloc[[2, 0, 1]].copy()
        plot = self.track(plot_scanpath(matched, participant=1))
        np.testing.assert_array_equal(plot.axes.collections[1].get_offsets(), [[7.2, 3.4], [17.8, 13.1], [27.3, 23.7]])
        lines = plot.axes.collections[0]
        self.assertIsInstance(lines, LineCollection)
        np.testing.assert_array_equal(lines.get_segments(), [[[7.2, 3.4], [17.8, 13.1]], [[17.8, 13.1], [27.3, 23.7]]])
        self.assertEqual([t.get_text() for t in plot.axes.texts], ['1','2','3'])
        self.assertTrue({'1: 100 ms', '2: 200 ms', '3: 300 ms'}.issubset({t.get_text() for t in plot.figure.texts}))
        self.assertEqual(tuple(plot.axes.images[0].get_extent()), (0, 1680, 1050, 0))
        np.testing.assert_array_equal(plot.axes.images[0].get_array()[0, 0], [30, 20, 10])
        self.assertEqual(plot.axes.get_ylim(), (1050, 0))

    def test_no_bridge_over_invalid_fixation_and_no_duration_imputation(self):
        matched = self.matched()
        matched.fixations.loc[1, 'x_image'] = np.nan
        matched.fixations.loc[0, 'duration_ms'] = np.nan
        plot = self.track(plot_scanpath(matched))
        self.assertEqual(plot.metadata['plotted_fixations'], 2)
        self.assertEqual(plot.metadata['line_segments'], 0)
        self.assertEqual(plot.metadata['duration_labels'], 1)
        self.assertEqual([t.get_text() for t in plot.axes.texts], ['1','3'])
        self.assertIn('1: unavailable', [t.get_text() for t in plot.figure.texts])

    def test_participant_and_trial_selection_prevent_merged_scanpaths(self):
        base = dict(name='artificial.png', subject=1, X=[7.2], Y=[3.4], T=[100], length=1)
        matched = self.matched([base, dict(base, subject=2), dict(base, trial_ind=1)])
        with self.assertRaisesRegex(ValueError, 'one participant'):
            plot_scanpath(matched)
        with self.assertRaisesRegex(ValueError, 'Multiple trials'):
            plot_scanpath(matched, participant=1)
        plot = self.track(plot_scanpath(matched, participant=1, record_index=2))
        self.assertEqual(plot.metadata['record_index'], 2)
        self.assertEqual(plot.metadata['plotted_fixations'], 1)
        with self.assertRaises(ValueError):
            plot_scanpath(matched, participant=99)

    def test_count_duration_accumulation_and_pixel_mapping(self):
        matched = self.matched()
        matched.fixations.loc[1, ['x_image', 'y_image']] = [7.2, 3.4]
        count = compute_density(matched, weighting='count', sigma=0)
        duration = compute_density(matched, weighting='duration', sigma=0)
        self.assertEqual(count.values.shape, (1050, 1680))
        self.assertEqual(count.values[3, 7], 2)
        self.assertEqual(count.values[23, 27], 1)
        self.assertEqual(count.values.sum(), 3)
        self.assertEqual(duration.values[3, 7], 300)
        self.assertEqual(duration.values.sum(), 600)
        self.assertEqual(duration.relative.max(), 1)

    def test_missing_duration_count_includes_position_duration_excludes_it(self):
        matched = self.matched()
        matched.fixations.loc[0, 'duration_ms'] = np.nan
        matched.fixations.loc[1, 'duration_ms'] = -10
        self.assertEqual(compute_density(matched, weighting='count', sigma=0).values.sum(), 3)
        duration = compute_density(matched, sigma=0)
        self.assertEqual(duration.values.sum(), 300)
        self.assertEqual(duration.metadata['excluded_durations'], 2)
        plot = self.track(plot_scanpath(matched, show_durations=False))
        self.assertEqual(plot.metadata['duration_labels'], 0)

    def test_aggregation_is_sum_of_individual_maps(self):
        base = dict(name='artificial.png', subject=1, X=[7.2], Y=[3.4], T=[100], length=1)
        matched = self.matched([base, dict(base, subject=2, X=[70.3], Y=[30.5], T=[200])])
        for weighting in ['count', 'duration']:
            combined = compute_density(matched, weighting=weighting, sigma=2)
            one = compute_density(matched, participant=1, weighting=weighting, sigma=2)
            two = compute_density(matched, participant=2, weighting=weighting, sigma=2)
            np.testing.assert_allclose(combined.values, one.values+two.values, rtol=1e-13, atol=1e-14)
            self.assertEqual(combined.metadata['participant_ids'], ['1','2'])

    def test_gaussian_peak_symmetry_and_zero_boundary(self):
        base = dict(name='artificial.png', subject=1, X=[20.2], Y=[15.3], T=[100], length=1)
        matched = self.matched([base], width=40, height=30, geometry=ImageGeometry(0, 0, 40, 30))
        density = compute_density(matched, weighting='count', sigma=2)
        self.assertEqual(np.unravel_index(density.values.argmax(), density.values.shape), (15, 20))
        self.assertGreater(density.values[15, 21], 0)
        self.assertAlmostEqual(density.values[15, 21], density.values[15, 19])
        self.assertAlmostEqual(density.values.sum(), 1, places=12)
        matched.fixations.loc[0, ['x_image','y_image']] = [0, 0]
        edge = compute_density(matched, weighting='count', sigma=2)
        self.assertLess(edge.values.sum(), 1)
        self.assertEqual(edge.values[-1, -1], 0)

    def test_heatmap_registration_alpha_and_no_mutation(self):
        matched = self.matched()
        original = matched.pixels_bgr.copy()
        plot = self.track(plot_heatmap(matched, weighting='count', sigma=0, alpha=.4))
        self.assertEqual(tuple(plot.axes.images[0].get_extent()), tuple(plot.axes.images[1].get_extent()))
        opacity = plot.axes.images[1].get_alpha()
        self.assertEqual(opacity.shape, (1050,1680))
        self.assertEqual(opacity[3,7], .4)
        self.assertEqual(opacity[0,0], 0)
        np.testing.assert_array_equal(matched.pixels_bgr, original)
        invisible = self.track(plot_heatmap(matched, alpha=0))
        self.assertFalse(invisible.axes.images[1].get_alpha().any())

    def test_loader_transformed_coordinates_used_without_rescaling(self):
        base = dict(name='artificial.png', subject=1, X=[840], Y=[525], T=[100], length=1)
        matched = self.matched([base], width=800, height=600, geometry=ImageGeometry(140, 0, 1400, 1050))
        density = compute_density(matched, sigma=0, weighting='count')
        self.assertEqual(density.values[300,400], 1)
        plot = self.track(plot_scanpath(matched))
        np.testing.assert_array_equal(plot.axes.collections[1].get_offsets(), [[400,300]])

    def test_png_generation_dimensions_dpi_metadata_and_preservation(self):
        matched = self.matched()
        for label, plot in [('scan', self.track(plot_scanpath(matched))),
                            ('heat', self.track(plot_heatmap(matched)))]:
            path = self.root/f'{label}.png'
            save_png(plot, path, dpi=120)
            with Image.open(path) as image:
                expected = tuple(int(v*120) for v in plot.figure.get_size_inches())
                self.assertEqual(image.size, expected)
                self.assertAlmostEqual(image.info['dpi'][0], 120, places=1)
                metadata = json.loads(image.info['Description'])
                self.assertEqual(metadata['data_kind'], 'recorded_human_gaze')
            self.assertIsNotNone(cv2.imread(str(path)))
            with self.assertRaises(FileExistsError):
                save_png(plot, path)

    def test_invalid_parameters_empty_points_and_alignment(self):
        matched = self.matched()
        for sigma in [-1, float('nan'), float('inf'), True]:
            with self.subTest(sigma=sigma), self.assertRaises(ValueError):
                compute_density(matched, sigma=sigma)
        for alpha in [-.1, 1.1, float('nan'), True]:
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                plot_heatmap(matched, alpha=alpha)
        with self.assertRaises(ValueError):
            compute_density(matched, weighting='predicted')
        matched.fixations['inside_image'] = False
        with self.assertRaises(ValueError):
            compute_density(matched)
        with self.assertRaises(ValueError):
            plot_scanpath(matched)
        matched.fixations['image_width'] = 1
        with self.assertRaisesRegex(ValueError, 'does not match'):
            compute_density(matched)


class RealVisualizationTests(unittest.TestCase):
    def test_real_coco_sample_and_pngs(self):
        root = Path(__file__).resolve().parents[1]/'data/raw/coco_freeview'
        source = root/'fixations/COCOFreeView_fixations_trainval.json'
        if not source.exists() or not (root/'images/000000128398.jpg').exists():
            self.skipTest('Official local COCO-FreeView example unavailable')
        matched = CocoFreeView(source, root/'images').load_image('000000128398.jpg')
        scan = plot_scanpath(matched, participant='101')
        self.addCleanup(scan.figure.clear)
        original = matched.fixations[matched.fixations.participant_id.eq('101')]
        np.testing.assert_array_equal(scan.axes.collections[1].get_offsets(), original[['x_image','y_image']].to_numpy())
        self.assertEqual(scan.metadata['line_segments'], len(original)-1)
        count = compute_density(matched, weighting='count', sigma=0)
        self.assertEqual(count.values.sum(), len(matched.fixations))
        duration = compute_density(matched, participant='101', sigma=0)
        self.assertAlmostEqual(duration.values.sum(), original.duration_ms.sum())
        heat = plot_heatmap(matched, sigma=16)
        self.addCleanup(heat.figure.clear)
        with tempfile.TemporaryDirectory() as folder:
            for name, plot in [('scan',scan), ('heat',heat)]:
                path = save_png(plot, Path(folder)/f'{name}.png', dpi=72)
                self.assertIsNotNone(cv2.imread(str(path)))
