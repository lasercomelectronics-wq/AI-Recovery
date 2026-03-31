from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from datetime import datetime

MB = 1024 * 1024
DEFAULT_BLOCK_SIZE = 128 * MB
DEFAULT_OVERLAP_SIZE = 32 * MB

@dataclass(slots=True)
class Block:
    offset: int
    data: memoryview
    chunk_index: int
    chunk_end: int


class RawReader:
    def __init__(
        self,
        path: str | Path,
        block_size: int = DEFAULT_BLOCK_SIZE,
        overlap_size: int = DEFAULT_OVERLAP_SIZE,
        governor: "HardwareGovernor | None" = None,
    ) -> None:
        self.path = Path(path)
        self.block_size = int(block_size)
        self.overlap_size = int(overlap_size)
        self._validate_sizes()
        self._fd = None
        self.size = 0
        # Preallocated buffer used for all block reads (no mmap).
        self._buffer: bytearray | None = None
        self._governor = governor

    def _validate_sizes(self) -> None:
        if self.block_size <= 0:
            raise ValueError(f"block_size debe ser positivo en bytes. Recibido: {self.block_size!r}")
        if self.overlap_size < 0:
            raise ValueError(f"overlap_size no puede ser negativo. Recibido: {self.overlap_size!r}")
        if self.overlap_size >= self.block_size:
            raise ValueError(
                "Configuración inválida en RawReader: overlap_size debe ser menor que block_size. "
                f"Recibido block_size={self.block_size} bytes, overlap_size={self.overlap_size} bytes. "
                "Revisá unidades (MB vs bytes), por ejemplo: "
                "block_size=512*1024*1024 y overlap_size=64*1024*1024."
            )

    def __enter__(self) -> "RawReader":
        self._fd = self.path.open("rb")
        self.size = self.path.stat().st_size
        alloc_len = min(self.block_size, self.size) if self.size > 0 else self.block_size
        self._buffer = bytearray(alloc_len)

        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._fd is not None:
            self._fd.close()
            self._fd = None
        self._buffer = None

    def iter_blocks(self, start_offset: int = 0) -> Iterator[Block]:
        if self._fd is None:
            raise RuntimeError("RawReader must be opened with context manager")

        offset = max(0, min(start_offset, self.size))
        step = self.block_size - self.overlap_size

        chunk_index = 0

        if step <= 0:
            raise ValueError(f"RawReader fatal: overlap ({self.overlap_size}) >= block_size ({self.block_size})")

        if self._buffer is None:
            raise RuntimeError("Internal buffer not initialized")

        last_offset = -1
        while offset < self.size:
            if offset <= last_offset:
                raise RuntimeError(f"Pointer Trap Fatal: el reader no presenta progreso monotónico. ({offset} <= {last_offset})")
            last_offset = offset

            # RAM backpressure: if system RAM is above the hard cap, do not ingest the next block.
            if self._governor is not None:
                self._governor.hold_if_ram_over_sync()

            end = min(offset + self.block_size, self.size)
            read_len = end - offset

            # Ensure buffer is large enough for the current file.
            if len(self._buffer) < read_len:
                self._buffer = bytearray(read_len)

            self._fd.seek(offset, 0)  # os.SEEK_SET explícito
            view = memoryview(self._buffer)[:read_len]
            # Debug probe: log before and after readinto to detect I/O starvation
            try:
                with open("debug_vulcan.log", "a", encoding="utf-8") as _lf:
                    _lf.write(f"[{datetime.now().isoformat()}] RawReader: about to readinto {read_len} bytes at offset {offset}\n")
            except Exception:
                pass
            got = self._fd.readinto(view)
            try:
                with open("debug_vulcan.log", "a", encoding="utf-8") as _lf:
                    _lf.write(f"[{datetime.now().isoformat()}] RawReader: readinto returned {got} bytes at offset {offset}\n")
            except Exception:
                pass
            if not got:
                break

            block_view = view[:got]
            yield Block(offset=offset, data=block_view, chunk_index=chunk_index, chunk_end=offset + got)

            if end >= self.size:
                break
            offset += step
            chunk_index += 1

    def read_range(self, start: int, end: int) -> bytes:
        if self._fd is None:
            raise RuntimeError("RawReader must be opened with context manager")

        start = max(0, min(start, self.size))
        end = max(start, min(end, self.size))

        # RAM backpressure: avoid reading additional data if the host is above the hard cap.
        if self._governor is not None:
            self._governor.hold_if_ram_over_sync()

        self._fd.seek(start)
        return self._fd.read(end - start)

    def buffer_health(self) -> str:
        return "safe-traditional-io"
