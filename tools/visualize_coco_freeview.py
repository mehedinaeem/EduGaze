#!/usr/bin/env python3
"""Generate real human-gaze scanpath and heatmap examples using the Phase 2 loader."""
import argparse
import hashlib
import json
from pathlib import Path

from edugaze.datasets import CocoFreeView
from edugaze.visualization import plot_heatmap, plot_scanpath, save_png


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/raw/coco_freeview'))
    parser.add_argument('--image', default='000000128398.jpg')
    parser.add_argument('--participant', default='101')
    parser.add_argument('--record-index', type=int)
    parser.add_argument('--sigma', type=float, default=32)
    parser.add_argument('--alpha', type=float, default=.65)
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--out', type=Path, default=Path('outputs/human_gaze_visualization'))
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Output directory already exists; choose a new --out to preserve previous results')
    dataset = CocoFreeView(args.root/'fixations/COCOFreeView_fixations_trainval.json', args.root/'images')
    matched = dataset.load_image(args.image)
    # Selection/parameter errors are caught before creating the output directory.
    scanpath = plot_scanpath(matched, participant=args.participant, record_index=args.record_index)
    figures = {'scanpath_participant.png': scanpath}
    for weighting in ('duration', 'count'):
        for scope, participant in [('participant', args.participant), ('all', None)]:
            name = f'heatmap_{weighting}_{scope}.png'
            figures[name] = plot_heatmap(matched, participant=participant, weighting=weighting,
                                         sigma=args.sigma, alpha=args.alpha)
    manifest = {'dataset': 'COCO-FreeView train/validation', 'data_kind': 'recorded_human_gaze',
                'source_json': str(dataset.json_path), 'source_sha256': dataset.source_sha256,
                'image_path': str(matched.path),
                'image_sha256': hashlib.sha256(matched.path.read_bytes()).hexdigest(),
                'geometry': vars(matched.geometry), 'dpi': args.dpi,
                'note': 'Descriptive recorded-gaze visualization; no model prediction or learning inference.',
                'figures': {}}
    for name, plot in figures.items():
        path = save_png(plot, args.out/name, dpi=args.dpi)
        manifest['figures'][name] = dict(plot.metadata, png_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        plot.figure.clear()
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'output_directory': str(args.out), 'image': args.image,
                      'participant': args.participant, 'dpi': args.dpi,
                      'files': list(figures),
                      'image_fixations': len(matched.fixations),
                      'image_participants': int(matched.fixations.participant_id.nunique())}, indent=2))


if __name__ == '__main__':
    main()
