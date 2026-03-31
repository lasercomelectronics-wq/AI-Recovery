from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class VideoHit:
    kind: str
    family: str
    offset: int
    confidence: float
    signature_id: str


class VideoSignatureEngine:
    """Deep signature scanner for forensic video carving using external signature registry."""

    def __init__(self, signature_db: str | None = None) -> None:
        self.signature_db = Path(signature_db) if signature_db else None
        self.registry = self._load_registry(self.signature_db)

    @staticmethod
    def _load_registry(path: Path | None) -> list[dict]:
        if path is not None:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload.get("signatures", [])

        signatures_dir = Path(__file__).resolve().parents[1] / "signatures"
        merged: dict[str, dict] = {}
        for file in sorted(signatures_dir.glob("video_signatures*.json")):
            payload = json.loads(file.read_text(encoding="utf-8"))
            for sig in payload.get("signatures", []):
                sid = str(sig.get("id", f"anon_{len(merged)}"))
                merged[sid] = sig
        return list(merged.values())

    def stats(self) -> dict:
        families: dict[str, int] = {}
        for sig in self.registry:
            fam = sig.get("family", "unknown")
            families[fam] = families.get(fam, 0) + 1
        return {"total": len(self.registry), "families": families}

    def scan_block(self, data: bytes, base_offset: int) -> list[VideoHit]:
        hits: list[VideoHit] = []
        for sig in self.registry:
            pattern = self._pattern_bytes(sig)
            if not pattern:
                continue

            start = 0
            iteration_count = 0
            max_iter = len(data)

            while True:
                iteration_count += 1
                if iteration_count > max_iter:
                    raise RuntimeError(f"Loop infinito detectado en bloque para la firma '{sig.get('id', 'unknown')}'.")

                idx = data.find(pattern, start)
                if idx < 0:
                    break

                next_index = idx + max(len(pattern), 1)
                if next_index <= idx:
                    raise RuntimeError("Violación de progreso: el parser no avanza.")

                # offset-aware signatures (e.g. atom/tag expected at offset 4 or brand at 8)
                expected_offset = sig.get("offset")
                if isinstance(expected_offset, int):
                    # if expected offset is absolute from block start, bias confidence
                    local_confidence = float(sig.get("confidence", 0.7))
                    if idx != expected_offset:
                        local_confidence *= 0.7
                    if sig.get("family") == "ts":
                        local_confidence *= self._ts_confidence(data, idx)
                    if local_confidence >= 0.45:
                        hits.append(
                            VideoHit(
                                kind=str(sig.get("kind", "unknown")),
                                family=str(sig.get("family", "unknown")),
                                offset=base_offset + idx,
                                confidence=min(local_confidence, 0.99),
                                signature_id=str(sig.get("id", "unknown")),
                            )
                        )
                else:
                    local_confidence = float(sig.get("confidence", 0.7))
                    if sig.get("family") == "ts":
                        local_confidence *= self._ts_confidence(data, idx)
                    if local_confidence >= 0.45:
                        hits.append(
                            VideoHit(
                                kind=str(sig.get("kind", "unknown")),
                                family=str(sig.get("family", "unknown")),
                                offset=base_offset + idx,
                                confidence=min(local_confidence, 0.99),
                                signature_id=str(sig.get("id", "unknown")),
                            )
                        )

                start = next_index
        return hits

    @staticmethod
    def _pattern_bytes(sig: dict) -> bytes:
        if "pattern_hex" in sig:
            return bytes.fromhex(str(sig["pattern_hex"]))
        if "pattern_ascii" in sig:
            return str(sig["pattern_ascii"]).encode("ascii", errors="ignore")
        return b""

    @staticmethod
    def _ts_confidence(data: bytes, idx: int) -> float:
        checks = 0
        ok = 0
        for k in range(1, 8):
            pos = idx + 188 * k
            if pos < len(data):
                checks += 1
                if data[pos] == 0x47:
                    ok += 1
        return (ok / checks) if checks else 0.0

    def parse_mp4_atoms(self, blob: bytes) -> dict[str, list[int]]:
        atoms: dict[str, list[int]] = {"ftyp": [], "moov": [], "mdat": [], "moof": []}
        i, n = 0, len(blob)
        while i + 8 <= n:
            size = int.from_bytes(blob[i : i + 4], "big", signed=False)
            tag = blob[i + 4 : i + 8]
            if tag in (b"ftyp", b"moov", b"mdat", b"moof"):
                atoms[tag.decode()].append(i)

            if size == 1 and i + 16 <= n:
                size = int.from_bytes(blob[i + 8 : i + 16], "big", signed=False)
            if size < 8:
                i += 1
            else:
                i += size
        return atoms

    def validate_mp4_structure(self, blob: bytes) -> bool:
        atoms = self.parse_mp4_atoms(blob)
        has_index = bool(atoms["moov"] or atoms["moof"])
        return bool(atoms["ftyp"] and atoms["mdat"]) and has_index
