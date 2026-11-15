"""Fixation validation and recorded COCO-FreeView dataset commands."""
import argparse
import json
from .data import read_fixations


def main(argv=None):
    parser = argparse.ArgumentParser(prog='edugaze')
    commands = parser.add_subparsers(dest='command', required=True)
    validate = commands.add_parser('validate', help='Validate a canonical fixation CSV')
    validate.add_argument('csv')
    validate.add_argument('--width', type=int, required=True)
    validate.add_argument('--height', type=int, required=True)
    coco = commands.add_parser('coco', help='Inspect/export official COCO-FreeView JSON')
    coco.add_argument('json')
    coco.add_argument('--images', required=True)
    coco.add_argument('--image', help='Exact photograph basename')
    coco.add_argument('--participant', help='Original participant identifier')
    coco.add_argument('--export', help='New normalized CSV; requires matched images')
    coco.add_argument('--report-invalid-trials', action='store_true',
                      help='Skip malformed scanpaths and report their source indexes')
    args = parser.parse_args(argv)
    try:
        if args.command == 'validate':
            records = read_fixations(args.csv, width=args.width, height=args.height)
            result = {'fixations': len(records),
                      'participants': len({r.participant_id for r in records}),
                      'images': sorted({r.image_name for r in records})}
            print(json.dumps(result))
            return
        from .datasets import CocoFreeView
        dataset = CocoFreeView(args.json, args.images, strict=not args.report_invalid_trials)
        selected = dataset.select(image=args.image, participant=args.participant)
        inventory = dataset.inventory()
        if args.export:
            dataset.export_csv(args.export, image=args.image, participant=args.participant)
        print(json.dumps({'source_scanpaths': len(dataset.metadata),
                          'selected_scanpaths': int(selected.record_index.nunique()),
                          'selected_fixations': len(selected),
                          'selected_images': int(selected.image_name.nunique()),
                          'selected_participants': sorted(selected.participant_id.unique().tolist()),
                          'valid_fixations': int(selected.valid_fixation.sum()),
                          'image_matches': inventory.image_status.value_counts().to_dict(),
                          'rejected_scanpaths': dataset.issues}, indent=2))
        selected_names = set(selected.image_name.unique())
        print(inventory[inventory.image_name.isin(selected_names)].head(10).to_string(index=False))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
