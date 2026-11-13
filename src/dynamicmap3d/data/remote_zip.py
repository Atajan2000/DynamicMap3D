"""Read individual members of a remote ZIP archive with HTTP Range requests.

Only the central directory and the requested members are transferred. Every response must be
``206 Partial Content`` with the requested ``Content-Range``; a server that ignores ``Range`` and
starts sending the whole archive (84.8 GB for the KITTI odometry Velodyne zip) aborts the read.
CRC-32 of every member is verified by :mod:`zipfile` on read.
"""

from __future__ import annotations

import io
import struct
import time
import zlib
import urllib.request
import zipfile
from collections import OrderedDict


class RangeError(RuntimeError):
    pass


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP Range requests with a small LRU block cache."""

    def __init__(self, url: str, block: int = 1 << 20, max_blocks: int = 16, timeout: float = 120):
        self.url, self.block, self.max_blocks, self.timeout = url, block, max_blocks, timeout
        self.pos = 0
        self.bytes_fetched = 0
        self.requests = 0
        self._cache: OrderedDict[int, bytes] = OrderedDict()
        head = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(head, timeout=timeout) as r:
            if r.headers.get("Accept-Ranges", "").lower() != "bytes":
                raise RangeError(f"{url} does not advertise byte ranges")
            self.size = int(r.headers["Content-Length"])

    # io.RawIOBase interface -------------------------------------------------------------
    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        if n == 0:
            return b""
        if n > self.block:  # large member payload: one exact request, not cached
            data = self._get(self.pos, self.pos + n - 1)
        else:
            out = bytearray()
            while len(out) < n:
                i, o = divmod(self.pos + len(out), self.block)
                out += self._block(i)[o : o + n - len(out)]
            data = bytes(out)
        self.pos += len(data)
        return data

    def readinto(self, b) -> int:
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)

    def read_exact(self, start: int, n: int) -> bytes:
        """Transfer exactly bytes [start, start + n): one uncached request, no block read-ahead."""
        return self._get(start, start + n - 1) if n > 0 else b""

    # internals --------------------------------------------------------------------------
    def _block(self, i: int) -> bytes:
        if i in self._cache:
            self._cache.move_to_end(i)
            return self._cache[i]
        s = i * self.block
        data = self._get(s, min(s + self.block, self.size) - 1)
        self._cache[i] = data
        if len(self._cache) > self.max_blocks:
            self._cache.popitem(last=False)
        return data

    def _get(self, start: int, end: int, retries: int = 8) -> bytes:
        want = end - start + 1
        last: Exception | None = None
        for attempt in range(retries):
            req = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    cr = r.headers.get("Content-Range", "")
                    if r.status != 206 or not cr.startswith(f"bytes {start}-{end}/"):
                        raise RangeError(f"server ignored Range (status {r.status}, '{cr}')")
                    data = r.read(want)
                self.requests += 1
                if len(data) == want:
                    self.bytes_fetched += len(data)
                    return data
                last = IOError(f"short read {len(data)}/{want}")
            except RangeError:
                raise
            except Exception as e:  # IncompleteRead, timeouts, resets
                last = e
            time.sleep(min(2**attempt, 30))
        raise IOError(f"range {start}-{end} failed after {retries} attempts: {last}")


def open_remote_zip(url: str) -> tuple[zipfile.ZipFile, HttpRangeFile]:
    rf = HttpRangeFile(url)
    return zipfile.ZipFile(rf), rf


def fetch_member(rf: HttpRangeFile, header_offset: int, compress_size: int, file_size: int,
                 crc: int) -> bytes:
    """Fetch one member using central-directory metadata cached earlier (no CD re-read).

    Parses the local file header, transfers exactly the member payload, and verifies size and
    CRC-32. Supports stored (0) and deflate (8) members.

    Both reads are exact-size requests. (An earlier version read the 30-byte header through the
    1 MiB block cache, which transferred ~1 MB of unneeded read-ahead per member.)
    """
    def exact(start: int, n: int) -> bytes:
        if hasattr(rf, "read_exact"):
            return rf.read_exact(start, n)
        rf.seek(start)
        return rf.read(n)

    hdr = exact(header_offset, 30)
    if hdr[:4] != b"PK\x03\x04":
        raise zipfile.BadZipFile(f"no local header at offset {header_offset}")
    method = struct.unpack("<H", hdr[8:10])[0]
    name_len, extra_len = struct.unpack("<HH", hdr[26:30])
    payload = exact(header_offset + 30 + name_len + extra_len, compress_size)
    if method == 0:
        data = payload
    elif method == 8:
        data = zlib.decompress(payload, -15)
    else:
        raise zipfile.BadZipFile(f"unsupported compression method {method}")
    if len(data) != file_size or (zlib.crc32(data) & 0xFFFFFFFF) != crc:
        raise zipfile.BadZipFile(f"size/CRC mismatch for member at offset {header_offset}")
    return data
