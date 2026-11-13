"""SemanticKITTI point-wise labels.

Each ``.label`` file holds one uint32 per scan point: the lower 16 bits are the semantic class,
the upper 16 bits the instance id (temporally consistent within a sequence).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

MOVING_CLASSES = {
    252: "moving-car",
    253: "moving-bicyclist",
    254: "moving-person",
    255: "moving-motorcyclist",
    256: "moving-on-rails",
    257: "moving-bus",
    258: "moving-truck",
    259: "moving-other-vehicle",
}
IGNORE_CLASSES = {0: "unlabeled", 1: "outlier"}
# Static movable objects: hard negatives for motion detection (e.g. parked cars).
STATIC_MOVABLE = {10: "car", 11: "bicycle", 13: "bus", 15: "motorcycle", 18: "truck",
                  20: "other-vehicle", 30: "person", 31: "bicyclist", 32: "motorcyclist"}
STRUCTURE = {50: "building", 51: "fence", 52: "other-structure", 70: "vegetation", 71: "trunk",
             80: "pole", 81: "traffic-sign", 99: "other-object"}
GROUND = {40: "road", 44: "parking", 48: "sidewalk", 49: "other-ground", 60: "lane-marking",
          72: "terrain"}

_MOVING_ARR = np.array(sorted(MOVING_CLASSES), dtype=np.uint16)
_IGNORE_ARR = np.array(sorted(IGNORE_CLASSES), dtype=np.uint16)


def decode_labels(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(raw, dtype=np.uint32)
    return (raw & 0xFFFF).astype(np.uint16), (raw >> 16).astype(np.uint16)


def labels_from_bytes(buf: bytes) -> tuple[np.ndarray, np.ndarray]:
    if len(buf) % 4:
        raise ValueError("label buffer size is not a multiple of 4")
    return decode_labels(np.frombuffer(buf, dtype=np.uint32))


def read_labels(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    return labels_from_bytes(Path(path).read_bytes())


def is_moving(sem: np.ndarray) -> np.ndarray:
    return np.isin(sem, _MOVING_ARR)


def is_ignored(sem: np.ndarray) -> np.ndarray:
    return np.isin(sem, _IGNORE_ARR)
