from __future__ import annotations

import hashlib
from dataclasses import dataclass
from io import BytesIO

from PIL import Image

from blackops.core.entropy_analyzer import EntropyAnalyzer
from blackops.carving.images.engine import ImageCarvingEngine
from blackops.carving.images.scorer import WeightedScorer
from blackops.carving.images.base import ImageFormat


@dataclass(slots=True)
class Candidate:
    offset: int
    size: int
    score: int
    image_format: str
    content_hash: str
    fragmented: bool = False


class ImageProcessor:
    JPEG_HEADER = bytes.fromhex("FFD8FF")
    JPEG_EOI = bytes.fromhex("FFD9")
    JPEG_SOS = bytes.fromhex("FFDA")

    def __init__(self, min_score: int = 40) -> None:
        self.min_score = min_score
        self.entropy = EntropyAnalyzer()
        self.engine = ImageCarvingEngine()
        self.scorer = WeightedScorer()

    def process(self, data: bytes, offset: int) -> list[Candidate]:
        """Carve images from block. Returns list of candidates (empty if none meet threshold).

        Callers must iterate: always returns a list, never singleton.
        """
        view = memoryview(data)
        candidates = []
        
        for result in self.engine.carve(view, offset):
            # Refine score with entropy and global metrics
            result = self.scorer.refine(result, view[result.offset_in_block:])
            
            if result.score >= self.min_score:
                blob = data[result.offset_in_block : result.offset_in_block + result.total_size]
                candidates.append(Candidate(
                    offset=offset + result.offset_in_block,
                    size=result.total_size,
                    score=int(result.score),
                    image_format=result.format.name.lower(),
                    content_hash=hashlib.sha256(blob).hexdigest(),
                    fragmented=not result.integrity_complete,
                ))
        return candidates

    def validate_image(self, img_data: bytes, min_resolution: int) -> tuple[bool, tuple[int, int]]:
        try:
            with Image.open(BytesIO(img_data)) as img:
                width, height = img.size
                if width < min_resolution or height < min_resolution:
                    return False, (width, height)
                img.verify()
                return True, (width, height)
        except Exception:
            return False, (0, 0)

    def reassemble_fragmented_jpeg(self, head_blob: bytes, candidates: list[bytes]) -> bytes | None:
        """Indexar -> Validar -> Reensamblar por continuidad + perfil de entropía."""
        if self.JPEG_EOI in head_blob:
            return head_blob

        best = None
        best_score = -1.0
        for fragment in candidates:
            merged = head_blob + fragment
            if self.JPEG_EOI not in merged:
                continue
            ent = self.entropy.profile(merged[: min(len(merged), 2 * 1024 * 1024)])
            score = float(ent.mean()) if ent.size else 0.0
            if score > best_score:
                best_score = score
                best = merged[: merged.find(self.JPEG_EOI) + 2]
        return best
