import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from edugaze.cli import main
from edugaze.data import read_fixations
from edugaze.paths import ProjectPaths

ROOT = Path(__file__).resolve().parents[1]
HEADER = 'image_name,participant_id,x,y,duration_ms\n'


class FixationTests(unittest.TestCase):
    def load(self, text, **kwargs):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'fixations.csv'
            path.write_text(text)
            return read_fixations(path, **({'width': 100, 'height': 100} | kwargs))

    def test_preserves_recorded_order_and_subjects(self):
        rows = self.load(HEADER + 'a,p2,70,30,120\na,p1,20,40,150\n')
        self.assertEqual([(r.participant_id, r.x) for r in rows], [('p2', 70), ('p1', 20)])

    def test_rejects_bad_records_with_row_number(self):
        for fields in ['a,p,100,0,10', 'a,p,0,100,10', 'a,p,-1,0,10',
                       'a,p,nan,0,10', 'a,p,inf,0,10', 'a,p,0,0,0',
                       'a,p,0,0,-1', 'a,,0,0,10', 'a,p,no,0,10', 'a,p,0']:
            with self.subTest(fields=fields), self.assertRaisesRegex(ValueError, 'row 2'):
                self.load(HEADER + fields + '\n')

    def test_missing_columns_and_empty_input(self):
        for text in ['', 'x,y\n0,0\n', HEADER]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.load(text)

    def test_invalid_canvas(self):
        for width in [0, -1, True, 1.5]:
            with self.subTest(width=width), self.assertRaises(ValueError):
                self.load(HEADER, width=width)

    def test_cli_synthetic_fixture(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            main(['validate', str(ROOT / 'data/synthetic/fixations.csv'), '--width', '100', '--height', '100'])
        self.assertEqual(json.loads(output.getvalue())['fixations'], 3)

    def test_cli_rejects_invalid_data(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            main(['validate', str(ROOT / 'data/synthetic/fixations.csv'), '--width', '1', '--height', '1'])
        self.assertEqual(caught.exception.code, 2)

    def test_data_paths_are_distinct(self):
        paths = ProjectPaths(ROOT)
        self.assertEqual(len({paths.raw, paths.processed, paths.synthetic}), 3)
        self.assertEqual(paths.raw, ROOT / 'data/raw')

    def test_available_original_archives_unchanged(self):
        # Original research artifacts are local-only; CI need not have them.
        for entry in json.loads((ROOT / 'docs/archive_manifest.json').read_text()):
            path = ROOT / entry['path']
            if path.exists():
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), entry['sha256'])


if __name__ == '__main__':
    unittest.main()
