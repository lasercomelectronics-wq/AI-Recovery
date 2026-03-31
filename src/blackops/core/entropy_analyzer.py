from __future__ import annotations

import numpy as np

from blackops.gpu.entropy import sliding_entropy_cupy


class EntropyAnalyzer:
    def __init__(self, window: int = 4096, step: int = 1024) -> None:
        self.window = window
        self.step = step

    def profile(self, data: bytes) -> np.ndarray:
        return sliding_entropy_cupy(data, window=self.window, step=self.step)

    def find_contiguous_candidates(self, ent: np.ndarray, threshold: float = 0.75) -> list[int]:
        """Return candidate window indexes whose entropy is near local median."""
        if ent.size == 0:
            return []
        median = float(np.median(ent))
        allowed = median * threshold
        return [i for i, value in enumerate(ent.tolist()) if value >= allowed]
