from __future__ import annotations

import asyncio
from dataclasses import dataclass

from blackops.core.fragment_graph import Fragment, FragmentGraph
from blackops.core.video_signature_engine import VideoSignatureEngine


@dataclass(slots=True)
class ReconstructionResult:
    ok: bool
    container: str
    recovered_data: bytes
    reason: str = ""
    score: float = 0.0


class VideoReconstructor:
    def __init__(self) -> None:
        self.engine = VideoSignatureEngine()
        self.graph = FragmentGraph()

    async def reconstruct_async(self, fragments: list[Fragment], container_hint: str) -> ReconstructionResult:
        chains = await asyncio.to_thread(
            self.graph.beam_search_chains, fragments, 8, 0.6, 14
        )
        if not chains:
            return ReconstructionResult(False, container_hint, b"", "no_chain", 0.0)

        best: ReconstructionResult | None = None
        for chain in chains:
            merged = b"".join(fragment.data for fragment in chain)
            result = self._validate_container(merged, container_hint)
            if best is None or result.score > best.score:
                best = result

        return best if best is not None else ReconstructionResult(False, container_hint, b"", "no_valid_chain", 0.0)

    def reconstruct(self, fragments: list[Fragment], container_hint: str) -> ReconstructionResult:
        chains = self.graph.beam_search_chains(fragments, beam_width=8, threshold=0.6, max_depth=14)
        if not chains:
            return ReconstructionResult(False, container_hint, b"", "no_chain", 0.0)

        best: ReconstructionResult | None = None
        for chain in chains:
            merged = b"".join(fragment.data for fragment in chain)
            result = self._validate_container(merged, container_hint)
            if best is None or result.score > best.score:
                best = result

        return best if best is not None else ReconstructionResult(False, container_hint, b"", "no_valid_chain", 0.0)

    def _validate_container(self, merged: bytes, container_hint: str) -> ReconstructionResult:
        family = container_hint.split("_")[0]

        if family in {"mp4", "3gp"}:
            atoms = self.engine.parse_mp4_atoms(merged)
            has_ftyp = bool(atoms["ftyp"])
            has_mdat = bool(atoms["mdat"])
            has_index = bool(atoms["moov"] or atoms["moof"])
            score = 0.0
            score += 0.35 if has_ftyp else 0.0
            score += 0.35 if has_mdat else 0.0
            score += 0.30 if has_index else 0.0
            if score >= 0.75:
                return ReconstructionResult(True, "mp4", merged, "validated_mp4", score)
            return ReconstructionResult(False, "mp4", b"", "invalid_mp4_atoms", score)

        if family == "avi":
            score = 0.0
            if b"RIFF" in merged:
                score += 0.5
            if b"AVI " in merged:
                score += 0.5
            if score >= 0.8:
                return ReconstructionResult(True, "avi", merged, "validated_avi", score)
            return ReconstructionResult(False, "avi", b"", "invalid_avi", score)

        if family == "wmv":
            guid = bytes.fromhex("3026B2758E66CF11A6D900AA0062CE6C")
            score = 1.0 if guid in merged else 0.0
            if score >= 0.8:
                return ReconstructionResult(True, "wmv", merged, "validated_wmv", score)
            return ReconstructionResult(False, "wmv", b"", "invalid_wmv", score)

        if family == "flv":
            score = 1.0 if merged.startswith(bytes.fromhex("464C5601")) else 0.0
            if score >= 0.8:
                return ReconstructionResult(True, "flv", merged, "validated_flv", score)
            return ReconstructionResult(False, "flv", b"", "invalid_flv", score)

        if family == "mpeg":
            score = 0.0
            if bytes.fromhex("000001BA") in merged:
                score += 0.5
            if bytes.fromhex("000001B3") in merged:
                score += 0.5
            if score >= 0.5:
                return ReconstructionResult(True, "mpeg", merged, "validated_mpeg", score)
            return ReconstructionResult(False, "mpeg", b"", "invalid_mpeg", score)

        if family == "ts":
            sync = merged.count(bytes([0x47]))
            score = min(sync / 50.0, 1.0)
            if score >= 0.6:
                return ReconstructionResult(True, "ts", merged, "validated_ts", score)
            return ReconstructionResult(False, "ts", b"", "invalid_ts", score)

        return ReconstructionResult(False, family, b"", "unsupported_or_invalid", 0.0)
