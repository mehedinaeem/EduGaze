"""Foundation commands; no model training or analysis is performed."""
import argparse
import json
from .data import read_fixations


def main(argv=None):
    parser = argparse.ArgumentParser(prog="edugaze")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="Validate a canonical fixation CSV")
    validate.add_argument("csv")
    validate.add_argument("--width", type=int, required=True)
    validate.add_argument("--height", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        records = read_fixations(args.csv, width=args.width, height=args.height)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({"fixations": len(records),
                      "participants": len({r.participant_id for r in records}),
                      "images": sorted({r.image_name for r in records})}))
