from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np

from blackops.core.entropy_analyzer import EntropyAnalyzer

MEMORY_CEILING_BYTES = 2 * 1024 * 1024 * 1024


@dataclass(slots=True)
class Fragment:
    offset: int
    data: bytes
    signature: str


@dataclass(slots=True)
class FragmentEdge:
    a: int
    b: int
    score: float


class FragmentGraph:
    def __init__(self) -> None:
        self.entropy = EntropyAnalyzer(window=2048, step=512)

    def edge_score(self, left: Fragment, right: Fragment) -> float:
        signature_score = 1.0 if left.signature.split("_")[0] == right.signature.split("_")[0] else 0.2
        left_tail = left.data[-65536:] if len(left.data) > 65536 else left.data
        right_head = right.data[:65536]
        e_left = self.entropy.profile(left_tail)
        e_right = self.entropy.profile(right_head)

        entropy_score = 0.0
        if e_left.size and e_right.size:
            entropy_score = 1.0 - min(abs(float(np.mean(e_left)) - float(np.mean(e_right))) / 8.0, 1.0)

        dist = abs(right.offset - (left.offset + len(left.data)))
        distance_score = 1.0 if dist < 8 * 1024 * 1024 else 0.4
        return 0.45 * signature_score + 0.4 * entropy_score + 0.15 * distance_score

    def best_chain(self, fragments: list[Fragment], threshold: float = 0.65) -> list[Fragment]:
        chains = self.beam_search_chains(fragments, beam_width=6, threshold=threshold, max_depth=12)
        return chains[0] if chains else []

    def beam_search_chains(
        self,
        fragments: list[Fragment],
        beam_width: int = 5,
        threshold: float = 0.65,
        max_depth: int = 10,
    ) -> list[list[Fragment]]:
        if not fragments:
            return []

        ordered = sorted(fragments, key=lambda f: f.offset)
        beams: list[tuple[list[Fragment], float]] = [([f], 0.0) for f in ordered[: min(len(ordered), beam_width)]]

        for depth in range(max_depth):
            candidates: list[tuple[list[Fragment], float]] = []
            for chain, chain_score in beams:
                last = chain[-1]
                expanded = False
                for nxt in ordered:
                    if nxt.offset <= last.offset:
                        continue
                    score = self.edge_score(last, nxt)
                    if score >= threshold:
                        candidates.append((chain + [nxt], chain_score + score))
                        expanded = True
                if not expanded:
                    candidates.append((chain, chain_score))

            candidates.sort(key=lambda c: (c[1], len(c[0])), reverse=True)

            estimated_memory = sum(
                sys.getsizeof(chain) + sum(sys.getsizeof(f) for f in chain)
                for chain, _ in candidates[:beam_width]
            )
            if estimated_memory > MEMORY_CEILING_BYTES:
                scores = [score for _, score in candidates]
                if scores:
                    percentile_25 = float(np.percentile(scores, 25))
                    candidates = [(c, s) for c, s in candidates if s >= percentile_25]

            beams = candidates[:beam_width]
            if not beams:
                break

        return [chain for chain, _ in beams]
