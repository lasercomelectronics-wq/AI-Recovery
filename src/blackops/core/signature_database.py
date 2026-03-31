from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable

import numpy as np


class ContainerType(IntEnum):
    UNKNOWN = 0
    MP4 = 1
    AVI = 2
    MPEG_TS = 3
    MPEG_PS = 4
    WMV = 5
    MKV = 6
    FLV = 7
    MOV = 8
    _3GP = 9


@dataclass(slots=True)
class FileSignature:
    container: ContainerType
    name: str
    magic: np.ndarray


class SignatureDatabase:
    """Optimized signature database backed by NumPy arrays."""

    def __init__(self) -> None:
        self._signatures: dict[str, FileSignature] = {}
        self._load_defaults()

    def _load_defaults(self) -> None:
        defaults: Iterable[tuple[ContainerType, str, bytes]] = (
            (ContainerType.MP4, "MP4 ftyp", bytes.fromhex("0000001866747970")),
            (ContainerType.MOV, "QuickTime moov", bytes.fromhex("000000146d6f6f76")),
            (ContainerType.AVI, "AVI RIFF", bytes.fromhex("52494646")),
            (ContainerType.MPEG_TS, "MPEG-TS", bytes.fromhex("47")),
            (ContainerType.MPEG_PS, "MPEG-PS", bytes.fromhex("000001ba")),
            (ContainerType.WMV, "ASF", bytes.fromhex("3026B2758E66CF11")),
            (ContainerType.MKV, "Matroska", bytes.fromhex("1A45DFA3")),
            (ContainerType.FLV, "FLV", bytes.fromhex("464C5601")),
            (ContainerType._3GP, "3GPP", bytes.fromhex("0000001466747970336770")),
        )
        for ctype, name, magic in defaults:
            self.add(name, ctype, magic)

    def add(self, name: str, container: ContainerType, magic: bytes) -> None:
        self._signatures[name] = FileSignature(
            container=container,
            name=name,
            magic=np.frombuffer(magic, dtype=np.uint8).copy(),
        )

    def detect_container(self, data: bytes | np.ndarray) -> ContainerType:
        arr = np.frombuffer(data, dtype=np.uint8) if isinstance(data, (bytes, bytearray, memoryview)) else data
        for sig in self._signatures.values():
            n = sig.magic.size
            if arr.size >= n and np.array_equal(arr[:n], sig.magic):
                return sig.container
        return ContainerType.UNKNOWN

    def find_all_offsets(self, data: bytes | np.ndarray) -> dict[str, list[int]]:
        arr = np.frombuffer(data, dtype=np.uint8) if isinstance(data, (bytes, bytearray, memoryview)) else data
        found: dict[str, list[int]] = {}
        for name, sig in self._signatures.items():
            n = sig.magic.size
            if n == 0 or arr.size < n:
                found[name] = []
                continue
            offsets: list[int] = []
            # vectorized prefilter by first byte
            candidates = np.where(arr[: arr.size - n + 1] == sig.magic[0])[0]
            for idx in candidates.tolist():
                if np.array_equal(arr[idx : idx + n], sig.magic):
                    offsets.append(int(idx))
            found[name] = offsets
        return found

    @property
    def signatures(self) -> dict[str, FileSignature]:
        return dict(self._signatures)
