#!/usr/bin/env python3
"""Acquire official COCO-FreeView data; default is three matched photographs.

Links were followed from https://sites.google.com/view/cocosearch/coco-freeview
on 2026-11-15. HTTP is an explicit fallback for the publisher's certificate error,
not an integrity guarantee. ZIP members are validated by zipfile's CRC checks.
No records or photograph URLs are generated from guessed dataset identifiers.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import time
import urllib.request
import zipfile

BASE = 'https://vision.cs.stonybrook.edu/~cvlab_download/'
JSON_FILE = 'COCOFreeView_fixations_trainval.json'
ARCHIVES = ['COCOSearch18-images-TP.zip', 'COCOSearch18-images-TA.zip']


class RangeFile(io.RawIOBase):
    """Seekable, bounded HTTP range reader for ZIP central directories/members."""
    def __init__(self, url):
        self.url, self.position, self.size = url, 0, None
        self.etag = None
        self._fetch(0, 0)

    def _fetch(self, first, last):
        headers = {'Range': f'bytes={first}-{last}', 'Accept-Encoding': 'identity'}
        if self.etag:
            headers['If-Range'] = self.etag
        for attempt in range(3):
            try:
                with urllib.request.urlopen(urllib.request.Request(self.url, headers=headers), timeout=30) as response:
                    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
                    if response.status != 206 or not match:
                        raise ValueError('Server did not honor bounded range request')
                    start, end, size = map(int, match.groups())
                    if (start, end) != (first, last) or (self.size is not None and size != self.size):
                        raise ValueError('Archive range/size changed during download')
                    etag = response.headers.get('ETag')
                    if self.etag and etag != self.etag:
                        raise ValueError('Archive ETag changed during download')
                    data = response.read(last - first + 2)
                    if len(data) != last - first + 1:
                        raise ValueError('Incomplete range response')
                    self.size, self.etag = size, etag
                    return data
            except (OSError, ValueError):
                if attempt == 2:
                    raise
                time.sleep(1)

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if position < 0:
            raise ValueError('Negative seek')
        self.position = position
        return position

    def read(self, size=-1):
        size = self.size - self.position if size < 0 else min(size, self.size - self.position)
        if size <= 0:
            return b''
        if size > 32 * 1024 * 1024:
            raise ValueError('Unexpected ZIP range larger than 32 MiB')
        data = self._fetch(self.position, self.position + size - 1)
        self.position += len(data)
        return data


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()



def read_unique_photo(archive, members):
    """Resolve category aliases only after verifying byte-for-byte identity."""
    content = None
    for member in members:
        if member.file_size > 32 * 1024 * 1024:
            raise ValueError(f'Unexpectedly large photograph: {member.filename}')
        candidate = archive.read(member)  # zipfile verifies member CRC
        if content is not None and candidate != content:
            raise ValueError(f'Conflicting ZIP photographs share basename: {member.filename}')
        content = candidate
    if content is None:
        raise ValueError('No ZIP photograph members supplied')
    return content


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/raw/coco_freeview'))
    parser.add_argument('--transport', choices=['https', 'http'], default='https')
    parser.add_argument('--sample-images', type=int, default=3)
    parser.add_argument('--image', action='append', help='Exact name from downloaded JSON; repeatable')
    parser.add_argument('--workers', type=int, default=12, help='Parallel archive range requests (1–16)')
    parser.add_argument('--all-images', action='store_true', help='Download all referenced images, roughly 2 GB')
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error('--workers must be between 1 and 16')
    if args.sample_images <= 0:
        parser.error('--sample-images must be positive')
    base = BASE.replace('https:', 'http:') if args.transport == 'http' else BASE
    args.root.mkdir(parents=True, exist_ok=True)
    fixation_path = args.root / 'fixations' / JSON_FILE
    fixation_path.parent.mkdir(parents=True, exist_ok=True)
    if not fixation_path.exists():
        partial = fixation_path.with_suffix('.json.part')
        # Incomplete data is never installed as a valid dataset.
        with urllib.request.urlopen(base + JSON_FILE, timeout=30) as response, partial.open('wb') as stream:
            for chunk in iter(lambda: response.read(1024 * 1024), b''):
                stream.write(chunk)
        with partial.open() as stream:
            json.load(stream)
        partial.rename(fixation_path)
    with fixation_path.open() as stream:
        scanpaths = json.load(stream)
    available = list(dict.fromkeys(row['name'] for row in scanpaths))
    requested = args.image or (available if args.all_images else available[:args.sample_images])
    unknown = set(requested) - set(available)
    if unknown:
        parser.error(f'Images not found in official fixation JSON: {sorted(unknown)}')
    output = args.root / 'images'
    output.mkdir(exist_ok=True)
    manifest_path = args.root / 'download_manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {'images': {}}
    manifest.update({'official_page': 'https://sites.google.com/view/cocosearch/coco-freeview',
                     'fixations_url': base + JSON_FILE, 'fixations_sha256': sha256(fixation_path),
                     'transport': args.transport, 'downloaded_on': time.strftime('%Y-%m-%d'),
                     'image_collection_complete': False})
    remaining = set(requested)
    for archive in ARCHIVES:
        if not remaining:
            break
        remote = RangeFile(base + archive)
        if args.all_images:
            archive_dir = args.root / 'archives'
            archive_dir.mkdir(exist_ok=True)
            archive_path = archive_dir / archive
            if not archive_path.exists():
                partial = archive_path.with_suffix('.zip.part')
                chunk_size = 4 * 1024 * 1024
                progress_path = partial.with_suffix('.progress.json')
                progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
                completed_prefix = progress.get('completed_prefix', 0)
                if (progress.get('size') != remote.size or progress.get('etag') != remote.etag
                        or not partial.exists() or not 0 <= completed_prefix <= remote.size):
                    completed_prefix = 0
                ranges = [(first, min(first + chunk_size, remote.size) - 1)
                          for first in range(completed_prefix, remote.size, chunk_size)]
                def fetch_range(bounds):
                    return bounds[0], remote._fetch(*bounds)
                # Bounded batches avoid retaining a whole archive in memory.
                with partial.open('r+b' if partial.exists() else 'w+b') as stream, ThreadPoolExecutor(max_workers=args.workers) as pool:
                    stream.truncate(remote.size)
                    for batch_start in range(0, len(ranges), args.workers):
                        for offset, content in pool.map(fetch_range, ranges[batch_start:batch_start+args.workers]):
                            stream.seek(offset)
                            stream.write(content)
                        stream.flush()
                        completed = ranges[min(batch_start + args.workers, len(ranges)) - 1][1] + 1
                        progress_path.write_text(json.dumps({'completed_prefix': completed,
                            'size': remote.size, 'etag': remote.etag}) + '\n')
                        print(f'{archive}: {completed}/{remote.size} bytes', flush=True)
                with zipfile.ZipFile(partial) as check:
                    check.infolist()
                partial.rename(archive_path)
            manifest.setdefault('archives', {})[archive] = {
                'url': base + archive, 'bytes': archive_path.stat().st_size,
                'sha256': sha256(archive_path), 'etag': remote.etag}
            archive_source = archive_path
            print(f'Official archive available: {archive_path}', flush=True)
        else:
            archive_source = remote
        with zipfile.ZipFile(archive_source) as z:
            index = {}
            for member in z.infolist():
                if member.is_dir():
                    continue
                name = PurePosixPath(member.filename).name
                if name in remaining:
                    index.setdefault(name, []).append(member)
            for name in sorted(remaining & index.keys()):
                members = index[name]
                member = members[0]
                content = read_unique_photo(z, members)
                target = output / name
                if target.exists() and target.read_bytes() != content:
                    raise FileExistsError(f'Different photograph already exists: {target}')
                if not target.exists():
                    target.write_bytes(content)
                manifest['images'][name] = {'archive_url': base + archive,
                    'member': member.filename, 'identical_members': [m.filename for m in members],
                    'sha256': sha256(target), 'zip_crc32': member.CRC}
                remaining.remove(name)
                manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
                if not args.all_images or len(manifest['images']) % 250 == 0:
                    print(f'Downloaded official photograph: {name} ({len(manifest["images"])} total)', flush=True)
    manifest['missing_requested_images'] = sorted(remaining)
    manifest['image_collection_complete'] = set(available).issubset(manifest['images'])
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    if remaining:
        raise FileNotFoundError(f'Referenced images absent from official archives: {sorted(remaining)}')
    print(f'JSON: {len(scanpaths)} scanpaths; photographs downloaded: {len(manifest["images"])}')


if __name__ == '__main__':
    main()
