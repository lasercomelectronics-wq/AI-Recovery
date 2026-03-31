from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SignatureHit:
    kind: str
    offset: int


class SignatureEngine:
    """Deep scanner for media signatures and simple structural validation."""

    SIGNATURES: dict[str, bytes] = {
        "jpeg": bytes.fromhex("FFD8FF"),
        "mp4_ftyp": b"ftyp",
        "mp4_moov": b"moov",
        "mp4_mdat": b"mdat",
        "avi": bytes.fromhex("52494646"),
        "avi_subtype": b"AVI ",
        "mpeg_ps": bytes.fromhex("000001BA"),
        "mpeg_seq": bytes.fromhex("000001B3"),
        "wmv_guid": bytes.fromhex("3026B2758E66CF11"),
        "flv": bytes.fromhex("464C5601"),
        "ts_sync": bytes([0x47]),
    }

    def scan_block(self, data: bytes, base_offset: int) -> list[SignatureHit]:
        hits: list[SignatureHit] = []

        for name, magic in self.SIGNATURES.items():
            start = 0
            iteration_count = 0
            max_iter = len(data)

            while start < len(data):
                iteration_count += 1
                if iteration_count > max_iter:
                    raise RuntimeError(f"Loop infinito detectado en bloque para la firma '{name}'.")

                idx = data.find(magic, start)
                if idx < 0:
                    break

                next_index = idx + max(len(magic), 1)
                if next_index <= idx:
                    raise RuntimeError("Violación de progreso: el parser no avanza.")

                if name == "ts_sync" and (idx + 188 < len(data)) and data[idx + 188] != 0x47:
                    start = next_index
                    continue
                hits.append(SignatureHit(kind=name, offset=base_offset + idx))
                start = next_index

        return self._deduplicate_nearby_hits(hits)

    def _deduplicate_nearby_hits(self, hits: list[SignatureHit], tolerance: int = 16) -> list[SignatureHit]:
        """Remove duplicate hits within tolerance bytes (off-by-one mitigation)."""
        if not hits:
            return []

        sorted_hits = sorted(hits, key=lambda h: h.offset)
        deduped: list[SignatureHit] = [sorted_hits[0]]

        for hit in sorted_hits[1:]:
            last = deduped[-1]
            if abs(hit.offset - last.offset) > tolerance or hit.kind != last.kind:
                deduped.append(hit)

        return deduped

    def parse_mp4_atoms(self, data: bytes) -> dict[str, list[int]]:
        """Locate ftyp/moov/mdat atoms, including moov at tail."""
        atoms: dict[str, list[int]] = {"ftyp": [], "moov": [], "mdat": []}
        i = 0
        n = len(data)
        while i + 8 <= n:
            size = int.from_bytes(data[i : i + 4], "big", signed=False)
            atom = data[i + 4 : i + 8]
            if atom in (b"ftyp", b"moov", b"mdat"):
                atoms[atom.decode()].append(i)
            if size < 8:
                i += 1
            else:
                i += size
        return atoms
