"""Minimal FFmpeg pipes for the demo capture (no imageio dependency in this project's environment).

The FFmpeg binary is resolved in this order: $FFMPEG, imageio-ffmpeg if installed, ffmpeg on PATH,
then the binary bundled with the sibling LiDAR-MapLab project's virtual environment.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np


def ffmpeg_exe() -> str:
    if os.environ.get("FFMPEG"):
        return os.environ["FFMPEG"]
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass
    if shutil.which("ffmpeg"):
        return shutil.which("ffmpeg")
    sib = Path(__file__).resolve().parents[2] / "LiDAR-MapLab/.venv/Lib/site-packages/imageio_ffmpeg/binaries"
    found = sorted(sib.glob("ffmpeg*.exe"))
    if found:
        return str(found[-1])
    raise RuntimeError("FFmpeg not found: set FFMPEG=<path to ffmpeg.exe>")


def probe(path: str | Path) -> dict:
    """Width, height, fps, duration and codec, parsed from ffmpeg's stream info."""
    r = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, text=True)
    s = r.stderr
    m = re.search(r"Video: (\w+).*?, (\d{2,5})x(\d{2,5})", s)
    fps = re.search(r"([\d.]+) fps", s)
    dur = re.search(r"Duration: (\d+):(\d+):([\d.]+)", s)
    return {"codec": m.group(1) if m else None, "width": int(m.group(2)) if m else None,
            "height": int(m.group(3)) if m else None, "fps": float(fps.group(1)) if fps else None,
            "duration_s": (int(dur.group(1)) * 3600 + int(dur.group(2)) * 60 + float(dur.group(3))) if dur else None}


def read_frames(path: str | Path):
    """Yield RGB uint8 frames (H, W, 3); raises if the decoder reports an error."""
    info = probe(path)
    w, h = info["width"], info["height"]
    p = subprocess.Popen([ffmpeg_exe(), "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    n = w * h * 3
    try:
        while True:
            buf = p.stdout.read(n)
            if len(buf) < n:
                break
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        p.stdout.close()
        err = p.stderr.read().decode(errors="replace").strip()
        if p.wait() not in (0, None) or err:
            if err:
                raise RuntimeError(f"decode error in {path}: {err[:500]}")


class Writer:
    """H.264 (yuv420p, +faststart) MP4 writer fed with RGB frames."""

    def __init__(self, path: str | Path, w: int, h: int, fps: int = 30, crf: int = 20, preset: str = "slow"):
        self.p = subprocess.Popen([ffmpeg_exe(), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                   "-s", f"{w}x{h}", "-framerate", str(fps), "-i", "-", "-c:v", "libx264",
                                   "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
                                   "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)

    def append(self, frame: np.ndarray) -> None:
        self.p.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())

    def close(self) -> None:
        self.p.stdin.close()
        if self.p.wait():
            raise RuntimeError("ffmpeg encode failed")
