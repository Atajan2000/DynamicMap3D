"""Capture the real Rerun Viewer window (Windows): screenshots and MP4, window pixels only.

Copied unchanged (apart from this note) from the sibling LiDAR-MapLab project (lidar_maplab/viz/capture.py).

Privacy and correctness by construction: frames come from Win32 ``PrintWindow`` with
``PW_RENDERFULLCONTENT``, which renders *only the viewer window* into an off-screen bitmap. Other
applications, the taskbar or notifications cannot appear in a capture, even when they overlap
the viewer. (An earlier screen-region approach did capture an overlapping window; it was removed
and that file deleted.)

* ``ViewerSession`` launches the real viewer binary (``rerun_cli/rerun.exe``) with a recording and
  a layout (.rbl) and sizes its client area to the requested resolution. It does not make the
  window topmost and does not take keyboard focus.
* ``screenshot`` writes one PNG. ``record`` streams frames to ffmpeg (``imageio-ffmpeg``) at a
  fixed wall-clock frame rate, repeating the latest frame if a grab is late, so playback speed in
  the video equals real time.

Windows only. On other platforms use ``rerun --screenshot-to`` (start-of-playback state only)
or a screen recorder.
"""

from __future__ import annotations

import ctypes
import subprocess
import sys
import time
from ctypes import wintypes
from pathlib import Path

import numpy as np

__all__ = ["ViewerSession", "monitor_rects", "CaptureError", "viewer_executable", "is_blank"]


class CaptureError(RuntimeError):
    pass


def _require_windows() -> None:
    if sys.platform != "win32":
        raise CaptureError("window capture is implemented for Windows only (Win32 PrintWindow)")


_u32 = _g32 = None


def _api():
    """user32/gdi32 with explicit 64-bit-safe signatures."""
    global _u32, _g32
    if _u32 is None:
        u, g = ctypes.windll.user32, ctypes.windll.gdi32
        try:
            u.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))  # per-monitor v2: physical pixels
        except Exception:
            pass
        H = ctypes.c_void_p
        u.GetDC.restype, u.GetDC.argtypes = H, [wintypes.HWND]
        u.ReleaseDC.argtypes = [wintypes.HWND, H]
        u.PrintWindow.argtypes = [wintypes.HWND, H, wintypes.UINT]
        g.CreateCompatibleDC.restype, g.CreateCompatibleDC.argtypes = H, [H]
        g.CreateCompatibleBitmap.restype, g.CreateCompatibleBitmap.argtypes = H, [H, ctypes.c_int, ctypes.c_int]
        g.SelectObject.restype, g.SelectObject.argtypes = H, [H, H]
        g.DeleteObject.argtypes = [H]
        g.DeleteDC.argtypes = [H]
        g.GetDIBits.argtypes = [H, H, wintypes.UINT, wintypes.UINT, ctypes.c_void_p, ctypes.c_void_p, wintypes.UINT]
        _u32, _g32 = u, g
    return _u32, _g32


def monitor_rects() -> list[tuple[tuple[int, int, int, int], bool]]:
    """[((x, y, w, h), is_primary)] for every monitor, in virtual-desktop pixels."""
    _require_windows()
    u, _ = _api()
    rects = []

    class MONITORINFO(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT),
                    ("rcWork", wintypes.RECT), ("dwFlags", wintypes.DWORD)]

    @ctypes.WINFUNCTYPE(ctypes.c_int, wintypes.HMONITOR, wintypes.HDC, ctypes.POINTER(wintypes.RECT), wintypes.LPARAM)
    def cb(hmon, hdc, lprect, lparam):
        mi = MONITORINFO()
        mi.cbSize = ctypes.sizeof(MONITORINFO)
        u.GetMonitorInfoW(hmon, ctypes.byref(mi))
        r = mi.rcMonitor
        rects.append(((r.left, r.top, r.right - r.left, r.bottom - r.top), bool(mi.dwFlags & 1)))
        return 1

    u.EnumDisplayMonitors(None, None, cb, 0)
    return rects


def _default_rect(width: int, height: int) -> tuple[int, int, int, int]:
    """Prefer a non-primary monitor of exactly the target size (keeps the main screen free)."""
    mons = monitor_rects()
    for (x, y, w, h), primary in sorted(mons, key=lambda m: m[1]):  # non-primary first
        if (w, h) == (width, height):
            return x, y, w, h
    (x, y, _, _), _ = next(m for m in mons if m[1])
    return x, y, width, height


def viewer_executable() -> Path:
    """The real viewer binary in ``rerun_cli`` (``Scripts/rerun.exe`` is only a launcher that starts
    it as a grandchild process, so its PID would not own the window)."""
    import importlib.util
    spec = importlib.util.find_spec("rerun_cli")
    if spec and spec.submodule_search_locations:
        exe = Path(list(spec.submodule_search_locations)[0]) / ("rerun.exe" if sys.platform == "win32" else "rerun")
        if exe.is_file():
            return exe
    fallback = Path(sys.executable).with_name("rerun.exe" if sys.platform == "win32" else "rerun")
    if fallback.is_file():
        return fallback
    raise CaptureError("rerun viewer executable not found (pip install rerun-sdk)")


def _windows_of_pid(pid: int) -> list[int]:
    u, _ = _api()
    found = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    def cb(hwnd, lparam):
        owner = wintypes.DWORD()
        u.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value == pid and u.IsWindowVisible(hwnd) and u.GetWindowTextLengthW(hwnd) > 0:
            found.append(hwnd)
        return True

    u.EnumWindows(cb, 0)
    return found


def _state_dir() -> Path:
    """Where the native viewer persists per-recording blueprints (incl. camera state)."""
    import os
    return Path(os.environ.get("APPDATA", Path.home())) / "rerun" / "data" / "blueprints"


def _state_snapshot() -> dict[Path, float]:
    d = _state_dir()
    return {p: p.stat().st_mtime for p in d.glob("*.rbl")} if d.is_dir() else {}


def is_blank(img: np.ndarray, background: tuple[int, int, int] = (13, 13, 13), min_fraction: float = 0.01) -> bool:
    """True if less than ``min_fraction`` of pixels differ from black and from the background."""
    a = img.astype(np.int32)
    differs = (np.abs(a - np.array(background)).sum(axis=2) > 40) & (a.sum(axis=2) > 30)
    return float(differs.mean()) < min_fraction


def _caption_overlay(w: int, h: int, text: str, bar_h: int = 44, bottom_margin: int = 32):
    """Pre-render a caption bar: (y0, y1, rgb, alpha) arrays blended into each frame."""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGBA", (w, bar_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("segoeui.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    tw = d.textlength(text, font=font)
    x0 = 16
    d.rounded_rectangle([x0 - 10, 4, x0 + tw + 14, bar_h - 4], radius=8, fill=(0, 0, 0, 185))
    d.text((x0 + 2, bar_h // 2), text, font=font, fill=(240, 239, 236, 255), anchor="lm")
    a = np.asarray(img).astype(np.float32)
    y1 = h - bottom_margin
    return y1 - bar_h, y1, a[:, :, :3], a[:, :, 3:4] / 255.0


def _apply_overlay(bgra: np.ndarray, overlay) -> np.ndarray:
    y0, y1, rgb, alpha = overlay
    out = bgra.copy()
    region = out[y0:y1, :, :3].astype(np.float32)
    bgr = rgb[:, :, ::-1]
    out[y0:y1, :, :3] = (region * (1 - alpha) + bgr * alpha).astype(np.uint8)
    return out


class _Grabber:
    """Reusable PrintWindow -> numpy grabber for one window (client area)."""

    PW_CLIENTONLY_RENDERFULLCONTENT = 0x1 | 0x2

    def __init__(self, hwnd: int):
        self.u, self.g = _api()
        self.hwnd = hwnd
        r = wintypes.RECT()
        self.u.GetClientRect(hwnd, ctypes.byref(r))
        self.w, self.h = r.right - r.left, r.bottom - r.top
        if self.w <= 0 or self.h <= 0:
            raise CaptureError("window has an empty client area")
        self.hdc_win = self.u.GetDC(hwnd)
        self.hdc_mem = self.g.CreateCompatibleDC(self.hdc_win)
        self.hbmp = self.g.CreateCompatibleBitmap(self.hdc_win, self.w, self.h)
        self.old = self.g.SelectObject(self.hdc_mem, self.hbmp)

        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG), ("biHeight", wintypes.LONG),
                        ("biPlanes", wintypes.WORD), ("biBitCount", wintypes.WORD),
                        ("biCompression", wintypes.DWORD), ("biSizeImage", wintypes.DWORD),
                        ("biXPelsPerMeter", wintypes.LONG), ("biYPelsPerMeter", wintypes.LONG),
                        ("biClrUsed", wintypes.DWORD), ("biClrImportant", wintypes.DWORD)]

        self.bmi = BITMAPINFOHEADER()
        self.bmi.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        self.bmi.biWidth, self.bmi.biHeight = self.w, -self.h  # negative: top-down rows
        self.bmi.biPlanes, self.bmi.biBitCount, self.bmi.biCompression = 1, 32, 0
        self.buf = (ctypes.c_ubyte * (self.w * self.h * 4))()

    def grab_bgra(self) -> np.ndarray:
        if not self.u.PrintWindow(self.hwnd, self.hdc_mem, self.PW_CLIENTONLY_RENDERFULLCONTENT):
            raise CaptureError("PrintWindow failed")
        self.g.GetDIBits(self.hdc_mem, self.hbmp, 0, self.h, self.buf, ctypes.byref(self.bmi), 0)
        return np.frombuffer(self.buf, dtype=np.uint8).reshape(self.h, self.w, 4)

    def grab_rgb(self) -> np.ndarray:
        return self.grab_bgra()[:, :, 2::-1].copy()

    def close(self) -> None:
        self.g.SelectObject(self.hdc_mem, self.old)
        self.g.DeleteObject(self.hbmp)
        self.g.DeleteDC(self.hdc_mem)
        self.u.ReleaseDC(self.hwnd, self.hdc_win)


class ViewerSession:
    """A real Rerun Viewer process showing ``recording`` with ``layout`` (.rbl)."""

    def __init__(self, recording: str | Path, layout: str | Path, size: tuple[int, int] = (1920, 1080),
                 rect: tuple[int, int, int, int] | None = None, rerun_exe: str | Path | None = None,
                 port: int = 9877):
        _require_windows()
        self.recording, self.layout = Path(recording), Path(layout)
        for p in (self.recording, self.layout):
            if not p.is_file():
                raise CaptureError(f"missing file: {p}")
        exe = Path(rerun_exe) if rerun_exe else viewer_executable()
        self.size, self.port = size, port
        # The viewer persists the active blueprint (including camera state) per recording and
        # restores it next time, overriding the .rbl given on the command line; this made later
        # sessions start with stale cameras (verified). Persistence cannot be disabled from the
        # CLI in rerun 0.41, so every session removes the state files it created or modified.
        self._state_before = _state_snapshot()
        self.rect = rect or _default_rect(*size)
        args = [str(exe), "--port", str(port), "--window-size", f"{size[0]}x{size[1]}"]
        self.proc = subprocess.Popen(args + [str(self.recording), str(self.layout)], stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
        self._grabber: _Grabber | None = None
        try:  # never leave an orphaned viewer behind if startup fails
            self.hwnd = self._wait_for_window()
            self._place()
            self._grabber = _Grabber(self.hwnd)
            if (self._grabber.w, self._grabber.h) != tuple(size):
                raise CaptureError(f"client area is {self._grabber.w}x{self._grabber.h}, expected {size}")
        except BaseException:
            self.close()
            raise

    def _wait_for_window(self, timeout: float = 30.0) -> int:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise CaptureError(f"viewer exited with code {self.proc.returncode}")
            wins = _windows_of_pid(self.proc.pid)
            if wins:
                return wins[0]
            time.sleep(0.25)
        raise CaptureError("viewer window did not appear")

    def _place(self) -> None:
        """Borderless, client area exactly ``size`` at ``rect``; not topmost, no focus change."""
        u, _ = _api()
        GWL_STYLE, WS_CAPTION, WS_THICKFRAME = -16, 0x00C00000, 0x00040000
        u.SetWindowLongW(self.hwnd, GWL_STYLE, u.GetWindowLongW(self.hwnd, GWL_STYLE) & ~WS_CAPTION & ~WS_THICKFRAME)
        SWP_NOZORDER, SWP_NOACTIVATE, SWP_FRAMECHANGED = 0x0004, 0x0010, 0x0020
        x, y, _, _ = self.rect
        w, h = self.size
        u.SetWindowPos(self.hwnd, None, x, y, w, h, SWP_NOZORDER | SWP_NOACTIVATE | SWP_FRAMECHANGED)
        time.sleep(1.0)

    def grab(self) -> np.ndarray:
        return self._grabber.grab_rgb()

    def screenshot(self, path: str | Path, require_content: bool = True) -> Path:
        from PIL import Image
        img = self.grab()
        if require_content and is_blank(img):
            raise CaptureError("captured frame is blank (viewer not rendered yet?)")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).save(path, optimize=True)
        return path

    def record(self, path: str | Path, seconds: float, fps: int = 15, crf: int = 20,
               caption: str | None = None) -> dict:
        """Record ``seconds`` of the window to H.264 MP4 at a fixed wall-clock frame rate.

        ``caption`` is drawn into a bar above the viewer's time panel; it labels the scene and
        never covers or alters plotted data.
        """
        import imageio_ffmpeg
        g = self._grabber
        overlay = _caption_overlay(g.w, g.h, caption) if caption else None
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo",
               "-pix_fmt", "bgra", "-s", f"{g.w}x{g.h}", "-framerate", str(fps), "-i", "-",
               "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(path)]
        ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        n_total = int(round(seconds * fps))
        grabs = 0
        t0 = time.perf_counter()
        def grab():
            f = g.grab_bgra()
            return (_apply_overlay(f, overlay) if overlay else f).tobytes()

        frame = grab()
        written = 0
        while written < n_total:
            due = int((time.perf_counter() - t0) * fps)  # frames that should exist by now
            if written <= due:
                frame = grab()
                grabs += 1
            while written <= min(due, n_total - 1):
                ff.stdin.write(frame)
                written += 1
            time.sleep(max(0.0, (written / fps) - (time.perf_counter() - t0)))
        ff.stdin.close()
        if ff.wait() != 0:
            raise CaptureError("ffmpeg failed to encode the capture")
        return {"file": str(path), "frames": written, "unique_grabs": grabs, "fps": fps,
                "seconds": written / fps, "duplicated": written - grabs}

    def press_space(self) -> None:
        """Toggle play/pause by posting a key event to the viewer window only (no focus change)."""
        u, _ = _api()
        u.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        scan = 0x39
        u.PostMessageW(self.hwnd, 0x0100, 0x20, 1 | (scan << 16))  # WM_KEYDOWN VK_SPACE
        u.PostMessageW(self.hwnd, 0x0101, 0x20, 1 | (scan << 16) | (1 << 30) | (1 << 31))  # WM_KEYUP

    def close(self) -> None:
        if self._grabber is not None:
            self._grabber.close()
            self._grabber = None
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        before = getattr(self, "_state_before", None)
        if before is not None:
            for path, mtime in _state_snapshot().items():
                if path not in before or mtime != before[path]:
                    path.unlink(missing_ok=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
