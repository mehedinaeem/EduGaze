"""Offline tests of bounded official-archive acquisition, no human data fixtures."""
import importlib.util
import io
from pathlib import Path
import re
import unittest
from unittest.mock import patch
import zipfile

spec = importlib.util.spec_from_file_location('edugaze_acquisition', Path(__file__).resolve().parents[1]/'tools/download_coco_freeview.py')
acquisition = importlib.util.module_from_spec(spec)
spec.loader.exec_module(acquisition)


class RangeReaderTests(unittest.TestCase):
    def setUp(self):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('artificial/test.txt', 'range-test content')
        self.content = stream.getvalue()
        self.requests = []

    def response(self, request, timeout):
        match = re.fullmatch(r'bytes=(\d+)-(\d+)', request.get_header('Range'))
        first, last = map(int, match.groups())
        self.requests.append((first, last))
        response = io.BytesIO(self.content[first:last+1])
        response.status = 206
        response.headers = {'Content-Range': f'bytes {first}-{last}/{len(self.content)}', 'ETag': 'stable-test'}
        return response

    def test_zip_reads_validate_member_crc(self):
        with patch.object(acquisition.urllib.request, 'urlopen', side_effect=self.response):
            reader = acquisition.RangeFile('https://example.invalid/offline-test.zip')
            with zipfile.ZipFile(reader) as archive:
                self.assertEqual(archive.read('artificial/test.txt'), b'range-test content')
        self.assertTrue(all(0 <= first <= last < len(self.content) for first, last in self.requests))

    def test_rejects_server_ignoring_ranges(self):
        def ignores(request, timeout):
            response = io.BytesIO(self.content)
            response.status = 200
            response.headers = {}
            return response
        with patch.object(acquisition.urllib.request, 'urlopen', side_effect=ignores), patch.object(acquisition.time, 'sleep'):
            with self.assertRaisesRegex(ValueError, 'honor bounded range'):
                acquisition.RangeFile('https://example.invalid/offline-test.zip')

    def test_rejects_changed_archive_etag(self):
        def changes(request, timeout):
            response = self.response(request, timeout)
            response.headers['ETag'] = 'changed-test' if len(self.requests) > 1 else 'stable-test'
            return response
        with patch.object(acquisition.urllib.request, 'urlopen', side_effect=changes), patch.object(acquisition.time, 'sleep'):
            reader = acquisition.RangeFile('https://example.invalid/offline-test.zip')
            with self.assertRaisesRegex(ValueError, 'ETag changed'):
                reader.read(1)

    def test_rejects_truncated_range(self):
        def truncated(request, timeout):
            response = self.response(request, timeout)
            response.seek(0)
            response.truncate(0)
            return response
        with patch.object(acquisition.urllib.request, 'urlopen', side_effect=truncated), patch.object(acquisition.time, 'sleep'):
            with self.assertRaisesRegex(ValueError, 'Incomplete range'):
                acquisition.RangeFile('https://example.invalid/offline-test.zip')


class ArchiveAliasTests(unittest.TestCase):
    def archive(self, second):
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, 'w') as archive:
            archive.writestr('category_one/artificial.jpg', b'artificial image bytes')
            archive.writestr('category_two/artificial.jpg', second)
        stream.seek(0)
        return zipfile.ZipFile(stream)

    def test_identical_category_copies_are_deduplicated(self):
        with self.archive(b'artificial image bytes') as archive:
            self.assertEqual(acquisition.read_unique_photo(archive, archive.infolist()), b'artificial image bytes')

    def test_conflicting_category_copies_are_rejected(self):
        with self.archive(b'different bytes') as archive:
            with self.assertRaisesRegex(ValueError, 'Conflicting ZIP photographs'):
                acquisition.read_unique_photo(archive, archive.infolist())
