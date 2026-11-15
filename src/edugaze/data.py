"""Validated fixation CSV records. Rows retain their recorded order.

Coordinates are pixels on the presented stimulus canvas, with origin at the
upper left. Durations are milliseconds. No filtering or inference is implicit.
"""
import csv
from dataclasses import dataclass
from math import isfinite
from pathlib import Path


@dataclass(frozen=True)
class Fixation:
    image_name: str
    participant_id: str
    x: float
    y: float
    duration_ms: float

    def __post_init__(self):
        if not self.image_name.strip() or not self.participant_id.strip():
            raise ValueError("Image name and participant ID must be nonempty")
        if not all(isfinite(v) for v in (self.x, self.y, self.duration_ms)):
            raise ValueError("Coordinates and duration must be finite")
        if self.x < 0 or self.y < 0 or self.duration_ms <= 0:
            raise ValueError("Coordinates must be nonnegative and duration positive")


def read_fixations(path: str | Path, *, width: int, height: int) -> list[Fixation]:
    """Load canonical CSV, rejecting invalid rows instead of silently dropping them.

    Required columns: image_name,participant_id,x,y,duration_ms.
    Canvas boundaries are half-open: 0 <= x < width, 0 <= y < height.
    Trials must be kept in separate files until a trial-aware schema is added.
    """
    if (isinstance(width, bool) or isinstance(height, bool)
            or not isinstance(width, int) or not isinstance(height, int)
            or width <= 0 or height <= 0):
        raise ValueError("Canvas dimensions must be positive integers")
    required = {"image_name", "participant_id", "x", "y", "duration_ms"}
    records = []
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing columns: {sorted(required - set(reader.fieldnames or []))}")
        for line, row in enumerate(reader, start=2):
            try:
                record = Fixation(row["image_name"], row["participant_id"],
                                  float(row["x"]), float(row["y"]), float(row["duration_ms"]))
                if record.x >= width or record.y >= height:
                    raise ValueError("Fixation lies outside the stimulus canvas")
            except (ValueError, TypeError, AttributeError) as exc:
                raise ValueError(f"Invalid fixation at CSV row {line}: {exc}") from exc
            records.append(record)
    if not records:
        raise ValueError("Fixation file contains no records")
    return records
