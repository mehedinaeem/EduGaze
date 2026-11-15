#!/usr/bin/env python3
"""Analyze recorded gaze and illustrative manual AOIs on one real photograph."""
import argparse
import hashlib
import json
from pathlib import Path

from edugaze.analysis import load_aois
from edugaze.analysis.reports import export_analysis
from edugaze.datasets import CocoFreeView


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/raw/coco_freeview'))
    parser.add_argument('--image', default='000000128398.jpg')
    parser.add_argument('--participant')
    parser.add_argument('--aois', type=Path, help='CSV/JSON rectangles in this image coordinate system')
    parser.add_argument('--out', type=Path, default=Path('outputs/gaze_aoi_analysis'))
    parser.add_argument('--dpi', type=int, default=300)
    args = parser.parse_args()
    if args.aois is None:
        if args.image != '000000128398.jpg':
            parser.error('Provide --aois for the selected image; bundled AOIs belong only to 000000128398.jpg')
        args.aois = Path('configs/aois/coco_000000128398.csv')
    dataset = CocoFreeView(args.root/'fixations/COCOFreeView_fixations_trainval.json', args.root/'images')
    matched = dataset.load_image(args.image, participant=args.participant)
    height, width = matched.pixels_bgr.shape[:2]
    aois = load_aois(args.aois, image_size=(width,height))
    provenance = {'source_json': str(dataset.json_path), 'source_sha256':dataset.source_sha256,
                  'image_sha256':hashlib.sha256(matched.path.read_bytes()).hexdigest(),
                  'aoi_file':str(args.aois), 'aoi_sha256':hashlib.sha256(args.aois.read_bytes()).hexdigest(),
                  'geometry':vars(matched.geometry), 'participant_filter':args.participant,
                  'note':'AOIs are illustrative manually defined rectangles, not semantic ground truth or model output.'}
    report = export_analysis(matched, aois, args.out, dpi=args.dpi, provenance=provenance)
    print(json.dumps({'output_directory':str(args.out), 'image':args.image, 'summary':report['summary']},indent=2))
    print('\nAOI summary:')
    for a in report['tables']['aoi_summary']:
        print(f'{a["aoi_id"]}: {a["fixation_count"]} fixations; {a["observed_fixation_duration_ms"]} observed ms; {a["participants_viewing"]} viewers')


if __name__ == '__main__':
    main()
