from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class NalUnit:
    offset: int
    size: int
    nal_type: int


def find_h264_nals(data: bytes | np.ndarray) -> list[NalUnit]:
    arr = np.frombuffer(data, dtype=np.uint8) if isinstance(data, (bytes, bytearray, memoryview)) else data
    if arr.size < 5:
        return []

    out: list[NalUnit] = []
    i = 0
    while i + 4 < arr.size:
        if arr[i] == 0 and arr[i + 1] == 0 and ((arr[i + 2] == 1) or (arr[i + 2] == 0 and arr[i + 3] == 1)):
            start = i + 3 if arr[i + 2] == 1 else i + 4
            nal_type = int(arr[start] & 0x1F) if start < arr.size else 0
            j = start + 1
            while j + 4 < arr.size:
                if arr[j] == 0 and arr[j + 1] == 0 and (arr[j + 2] == 1 or (arr[j + 2] == 0 and arr[j + 3] == 1)):
                    break
                j += 1
            out.append(NalUnit(offset=int(i), size=int(j - i), nal_type=nal_type))
            i = j
        else:
            i += 1
    return out
