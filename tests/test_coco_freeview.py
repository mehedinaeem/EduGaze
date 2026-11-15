"""Artificial parser/geometry fixtures only: no invented human observations."""
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from edugaze.datasets import CocoFreeView, ImageGeometry


class CocoFreeViewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images = self.root / 'images'
        self.images.mkdir()
        self.path = self.root / 'artificial_test_scanpaths.json'

    def trial(self, **changes):
        item = dict(name='artificial.png', subject=1, X=[840, 850], Y=[525, 530],
                    T=[100, 200], length=2, trial_ind=0, task='test_only', split='train')
        item.update(changes)
        return item

    def dataset(self, trials=None, **kwargs):
        self.path.write_text(json.dumps(trials if trials is not None else [self.trial()]))
        return CocoFreeView(self.path, self.images, **kwargs)

    def image(self, width=1680, height=1050, name='artificial.png'):
        self.assertTrue(cv2.imwrite(str(self.images / name), np.zeros((height, width, 3), dtype=np.uint8)))

    def test_filter_and_preserve_trials(self):
        data = self.dataset([self.trial(), self.trial(subject=2), self.trial(trial_ind=1)])
        subset = data.select(image='artificial.png', participant=1)
        self.assertEqual(subset.record_index.tolist(), [0, 0, 2, 2])
        self.assertEqual(subset.fixation_index.tolist(), [0, 1, 0, 1])
        self.assertTrue(data.select(image='missing.png').empty)
        self.assertEqual(data.inventory().iloc[0].scanpaths, 3)

    def test_official_canvas_identity(self):
        self.image()
        matched = self.dataset().load_image('artificial.png')
        np.testing.assert_array_equal(matched.fixations.x_image, [840, 850])
        self.assertEqual(matched.fixations.x_normalized.iloc[0], 0.5)
        self.assertEqual(matched.pixels_bgr.shape, (1050, 1680, 3))
        fractional = self.dataset([self.trial(X=[0.1, 957.1], Y=[525.6, 530.3])]).load_image('artificial.png')
        np.testing.assert_array_equal(fractional.fixations.x_display, fractional.fixations.x_image)
        np.testing.assert_array_equal(fractional.fixations.y_display, fractional.fixations.y_image)

    def test_explicit_letterbox_inverse_and_padding(self):
        self.image(800, 600)
        data = self.dataset([self.trial(X=[840, 140, 139, 1540], Y=[525]*4,
                                        T=[100]*4, length=4)])
        with self.assertRaisesRegex(ValueError, 'actual display content rectangle'):
            data.load_image('artificial.png')
        matched = data.load_image('artificial.png', geometry=ImageGeometry(140, 0, 1400, 1050))
        np.testing.assert_allclose(matched.fixations.x_image, [400, 0, -1/1.75, 800])
        np.testing.assert_allclose(matched.fixations.y_image, [300]*4)
        self.assertEqual(matched.fixations.inside_image.tolist(), [True, True, False, False])

    def test_missing_values_flagged_and_not_imputed(self):
        data = self.dataset([self.trial(X=[None, -1, 1680, 20, '12', True],
            Y=[1]*6, T=[10, 10, 10, None, 10, 10], length=6)])
        self.assertEqual(len(data.fixations), 6)
        self.assertEqual(data.fixations.valid_fixation.tolist(), [False]*6)
        self.assertTrue(np.isnan(data.fixations.x_display.iloc[0]))
        self.assertEqual(data.fixations.x_display.iloc[1], -1)
        self.assertTrue(np.isnan(data.fixations.duration_ms.iloc[3]))

    def test_nonfinite_values_are_invalid(self):
        data = self.dataset([self.trial(X=[float('inf'), 5], T=[10, float('nan')])])
        self.assertFalse(data.fixations.valid_fixation.any())

    def test_bad_schema_raises_or_reports(self):
        bad = [None, {}, self.trial(Y=[1]), self.trial(length=10),
               self.trial(subject=None), self.trial(name='../escape.jpg'),
               self.trial(X='bad'), self.trial(condition='present'), self.trial(trial_ind=1.2), self.trial(X=[], Y=[], T=[], length=0)]
        for trial in bad:
            with self.subTest(trial=trial), self.assertRaises(ValueError):
                self.dataset([trial])
        data = self.dataset(bad + [self.trial()], strict=False)
        self.assertEqual(len(data.issues), len(bad))
        self.assertEqual(len(data.fixations), 2)
        self.assertEqual(data.fixations.record_index.tolist(), [len(bad)]*2)

    def test_non_list_json_and_empty_dataset(self):
        self.path.write_text('{}')
        with self.assertRaises(ValueError):
            CocoFreeView(self.path, self.images)
        data = self.dataset([])
        self.assertTrue(data.inventory().empty)
        with self.assertRaises(ValueError):
            data.export_csv(self.root/'empty.csv')

    def test_missing_ambiguous_and_corrupt_images(self):
        data = self.dataset()
        self.assertEqual(data.inventory().image_status.iloc[0], 'missing')
        with self.assertRaises(FileNotFoundError):
            data.load_image('artificial.png')
        self.image()
        folder = self.images / 'duplicate'
        folder.mkdir()
        (folder/'artificial.png').write_bytes((self.images/'artificial.png').read_bytes())
        data = self.dataset()
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            data.load_image('artificial.png')
        (folder/'artificial.png').unlink()
        (self.images/'artificial.png').write_text('not an image')
        with self.assertRaisesRegex(ValueError, 'decode'):
            self.dataset().load_image('artificial.png')

    def test_export_preserves_unknown_fields_and_missing_metadata(self):
        self.image()
        trial = self.trial(custom_sensor={'calibration': None}, X=[None, 850])
        del trial['trial_ind']
        data = self.dataset([trial])
        destination = self.root/'output'/'fixations.csv'
        data.export_csv(destination)
        exported = pd.read_csv(destination)
        self.assertTrue(pd.isna(exported.x_display.iloc[0]))
        self.assertTrue(exported.trial_ind.isna().all())
        metadata = json.loads(destination.with_suffix('.csv.metadata.json').read_text())
        self.assertEqual(metadata['scanpath_metadata'][0]['unsupported_fields'], ['custom_sensor'])
        self.assertIn('custom_sensor', metadata['scanpath_metadata'][0]['fields'])
        with self.assertRaises(FileExistsError):
            data.export_csv(destination)

    def test_multi_image_export_order_and_missing_image_failure(self):
        self.image()
        trials = [self.trial(subject=2), self.trial(name='artificial_second.png'),
                  self.trial(subject=2, trial_ind=1)]
        data = self.dataset(trials)
        failed_export = self.root/'should_not_exist.csv'
        with self.assertRaises(FileNotFoundError):
            data.export_csv(failed_export)
        self.assertFalse(failed_export.exists())
        self.image(name='artificial_second.png')
        data = self.dataset(trials)
        exported = data.export_csv(self.root/'complete.csv')
        self.assertEqual(exported.record_index.tolist(), [0, 0, 1, 1, 2, 2])
        self.assertEqual(exported.fixation_index.tolist(), [0, 1, 0, 1, 0, 1])
        filtered = data.export_csv(self.root/'participant.csv', participant=2)
        self.assertEqual(filtered.record_index.tolist(), [0, 0, 2, 2])

    def test_invalid_geometries(self):
        for rectangle in [(0, 0, 0, 10), (-1, 0, 10, 10), (0, 0, 1681, 1050),
                          (0, 0, float('nan'), 10), (True, 0, 10, 10)]:
            with self.subTest(rectangle=rectangle), self.assertRaises(ValueError):
                ImageGeometry(*rectangle)


class ActualLocalDataTests(unittest.TestCase):
    def test_official_local_sample(self):
        root = Path(__file__).resolve().parents[1] / 'data/raw/coco_freeview'
        path = root/'fixations/COCOFreeView_fixations_trainval.json'
        if not path.exists():
            self.skipTest('Official local JSON not downloaded')
        data = CocoFreeView(path, root/'images')
        inventory = data.inventory()
        matched = inventory[inventory.image_status.eq('matched')]
        if matched.empty:
            self.skipTest('Official photographs not downloaded')
        manifest_path = root/'download_manifest.json'
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(data.source_sha256, manifest['fixations_sha256'])
            if manifest.get('image_collection_complete'):
                self.assertTrue(inventory.image_status.eq('matched').all())
        name = matched.image_name.iloc[0]
        image = data.load_image(name)
        self.assertGreater(len(image.fixations), 0)
        self.assertEqual(image.pixels_bgr.shape[:2], (1050, 1680))
        np.testing.assert_array_equal(image.fixations.x_display, image.fixations.x_image)
        self.assertEqual(len(image.fixations), int(matched.iloc[0].fixations))
