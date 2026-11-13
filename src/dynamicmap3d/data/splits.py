"""Access to the frozen windows in configs/splits.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class WindowSpec:
    key: str          # "development" or "eval0", "eval1", ...
    role: str
    sequence: str
    first: int
    last: int

    @property
    def frames(self) -> range:
        return range(self.first, self.last + 1)

    @property
    def tag(self) -> str:
        return f"{self.sequence}_{self.first:06d}_{self.last:06d}"


def _spec(key: str, d: dict) -> WindowSpec:
    return WindowSpec(key, d["role"], str(d["sequence"]).zfill(2), int(d["first_frame"]), int(d["last_frame"]))


def all_windows(path: str | Path = "configs/splits.yaml") -> dict[str, WindowSpec]:
    s = yaml.safe_load(Path(path).read_text())
    out = {"development": _spec("development", s["development"])}
    for i, e in enumerate(s["evaluation"]):
        out[f"eval{i}"] = _spec(f"eval{i}", e)
    return out


def get_window(key: str, path: str | Path = "configs/splits.yaml") -> WindowSpec:
    w = all_windows(path)
    if key not in w:
        raise KeyError(f"unknown window '{key}', expected one of {sorted(w)}")
    return w[key]
