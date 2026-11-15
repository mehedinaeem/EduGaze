"""Explicit project paths, independent of the working directory."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    def __post_init__(self):
        object.__setattr__(self, "root", Path(self.root).resolve())

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def synthetic(self) -> Path:
        return self.root / "data" / "synthetic"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"
