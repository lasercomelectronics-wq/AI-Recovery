from __future__ import annotations

import mmap
from pathlib import Path


class MMapReader:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._fd = None
        self._map: mmap.mmap | None = None

    def __enter__(self) -> "MMapReader":
        self._fd = self.path.open("rb")
        self._map = mmap.mmap(self._fd.fileno(), 0, access=mmap.ACCESS_READ)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._map is not None:
            self._map.close()
        if self._fd is not None:
            self._fd.close()

    def read_all(self) -> bytes:
        if self._map is None:
            raise RuntimeError("MMapReader not opened")
        return bytes(self._map)
