#!/usr/bin/env python3
"""Verify real local data, export one image and draw a registration check.

The photograph overlay is a coordinate-integration check, not an attention or
learning analysis. Only one actual scanpath is drawn. Raw inputs are unchanged.
"""
import argparse
import json
import hashlib
from collections import Counter
from pathlib import Path

import cv2

from edugaze.datasets import CocoFreeView


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('data/raw/coco_freeview'))
    parser.add_argument('--image')
    parser.add_argument('--participant')
    parser.add_argument('--audit-all-images', action='store_true',
                        help='Decode/check dimensions and recorded checksums of every referenced photograph')
    parser.add_argument('--out', type=Path, default=Path('outputs/coco_freeview_verification'))
    args = parser.parse_args()
    data = CocoFreeView(args.root/'fixations/COCOFreeView_fixations_trainval.json', args.root/'images')
    inventory = data.inventory()
    candidates = inventory[inventory.image_status.eq('matched')]
    if candidates.empty:
        parser.error('No matched photographs. Run tools/download_coco_freeview.py first.')
    name = args.image or candidates.iloc[0].image_name
    selected = data.select(image=name, participant=args.participant)
    if selected.empty:
        parser.error('No scanpaths match the selected image/participant')
    # First recorded trial, not a combined trajectory across participants/trials.
    first = selected.iloc[0]
    participant = first.participant_id
    matched = data.load_image(name, participant=participant)
    trial = matched.fixations[matched.fixations.record_index.eq(first.record_index)]
    args.out.mkdir(parents=True, exist_ok=True)
    targets = [args.out / n for n in ['inventory.csv', 'normalized_fixations.csv',
                                     'normalized_fixations.csv.metadata.json', 'alignment_check.png', 'summary.json']]
    if any(path.exists() for path in targets):
        parser.error('Output already exists; use a new --out directory to preserve prior results')
    exported = data.export_csv(args.out/'normalized_fixations.csv', image=name)
    inventory.to_csv(args.out/'inventory.csv', index=False)
    overlay = matched.pixels_bgr.copy()
    for row in trial[trial.valid_for_image].itertuples():
        point = (int(row.x_image), int(row.y_image))
        cv2.circle(overlay, point, 9, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(overlay, str(row.fixation_index + 1), (point[0]+10, point[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(overlay, f'COCO-FreeView | {name} | participant {participant}',
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    if not cv2.imwrite(str(args.out/'alignment_check.png'), overlay):
        raise OSError('OpenCV failed to save the alignment check')
    df = data.fixations
    audit = None
    if args.audit_all_images:
        manifest_path = args.root/'download_manifest.json'
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        if manifest.get('fixations_sha256') != data.source_sha256:
            raise ValueError('Source JSON checksum does not match acquisition manifest')
        errors, dimensions = [], Counter()
        for number, image_name in enumerate(inventory.image_name, start=1):
            try:
                path = data.image_path(image_name)
                pixels = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                if pixels is None:
                    raise ValueError('OpenCV decode failed')
                height, width = pixels.shape[:2]
                dimensions[f'{width}x{height}'] += 1
                if (width, height) != (1680, 1050):
                    raise ValueError('Not an official display-sized stimulus')
                expected = manifest.get('images', {}).get(image_name, {}).get('sha256')
                if not expected or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                    raise ValueError('Missing acquisition checksum or photograph checksum mismatch')
            except (OSError, ValueError) as exc:
                errors.append({'image_name': image_name, 'error': str(exc)})
            if number % 1000 == 0:
                print(f'Audited {number}/{len(inventory)} photographs', flush=True)
        audit = {'photographs_checked': len(inventory), 'dimensions': dict(dimensions),
                 'errors': errors, 'passed': not errors}
        if errors:
            (args.out/'image_audit_errors.json').write_text(json.dumps(audit, indent=2)+'\n')
            raise ValueError(f'Image audit failed for {len(errors)} photographs')
    summary = {'dataset': 'COCO-FreeView official train/validation',
               'source_sha256': data.source_sha256,
               'scanpaths': len(data.metadata), 'fixations': len(df),
               'images': int(df.image_name.nunique()),
               'participant_ids': sorted(df.participant_id.unique().tolist()),
               'scanpaths_by_split': df.drop_duplicates('record_index').groupby('split').size().to_dict(),
               'invalid_display_fixations': int((~df.valid_display).sum()),
               'invalid_durations': int((~df.valid_duration).sum()),
               'photograph_status': inventory.image_status.value_counts().to_dict(),
               'example_image': name, 'example_image_size': list(matched.pixels_bgr.shape[1::-1]),
               'example_participant': participant, 'example_record_index': int(first.record_index),
               'example_scanpath_fixations': len(trial),
               'example_scanpath_valid_fixations': int(trial.valid_for_image.sum()),
               'exported_image_fixations': len(exported),
               'all_image_audit': audit,
               'coordinate_mapping': 'Identity: official display-mapped 1680x1050 stimulus',
               'note': 'Registration check only; no comprehension or educational outcome is inferred.'}
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))
    print('\nAvailable matched images:')
    print(candidates.head(10).to_string(index=False))
    print(f'Full inventory: {args.out / "inventory.csv"}')
    print('\nFirst recorded scanpath:')
    print(trial[['image_name','participant_id','record_index','fixation_index',
                 'x_display','y_display','duration_ms','x_image','y_image','valid_for_image']].head().to_string(index=False))


if __name__ == '__main__':
    main()
