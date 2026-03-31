from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Iterator, Optional
import numpy as np
from .base import ImageFormat, Signature, ValidationResult, ForensicEvidence
from .formats.jpeg import JPEGValidator
from .formats.png import PNGValidator
from .formats.tiff import TIFFValidator
from .formats.webp import WebPValidator
from .formats.heif import HEIFValidator
from .formats.common import BMPValidator, GIFValidator, PSDValidator, ICOValidator

@dataclass(slots=True)
class IndexEntry:
    offset: int
    format: ImageFormat
    confidence: float = 1.0

class SignatureIndexer:
    """Fast signature indexing before deep parsing."""
    
    def __init__(self, signatures: List[Signature]):
        self.signatures = signatures
        # Optimization: group magics by first byte
        self._by_first_byte: Dict[int, List[Signature]] = {}
        for sig in signatures:
            fb = sig.magic[0]
            if fb not in self._by_first_byte:
                self._by_first_byte[fb] = []
            self._by_first_byte[fb].append(sig)

    def scan(self, data: memoryview) -> List[IndexEntry]:
        """Scan data for potential signatures."""
        entries = []
        arr = np.frombuffer(data, dtype=np.uint8)
        
        # Performance: find candidates by first byte across all signatures
        for fb, sigs in self._by_first_byte.items():
            # Vectorized search for first byte
            offsets = np.where(arr == fb)[0]
            for offset in offsets:
                for sig in sigs:
                    # Account for offset hint (e.g. magic at offset 4)
                    effective_offset = offset - sig.offset_hint
                    if effective_offset < 0:
                        continue
                        
                    sig_len = len(sig.magic)
                    if offset + sig_len <= len(data):
                        # Use tobytes().startswith to be safe and handle the offset correctly
                        if data[offset : offset + sig_len].tobytes() == sig.magic:
                            entries.append(IndexEntry(offset=int(effective_offset), format=sig.format))
        
        # Sort by offset for sequential processing
        entries.sort(key=lambda x: x.offset)
        return entries

class ImageCarvingEngine:
    """Main engine for image carving with pre-indexing and deep validation."""
    
    def __init__(self):
        self.signatures = self._load_default_signatures()
        self.indexer = SignatureIndexer(self.signatures)
        self.validators: Dict[ImageFormat, type] = {
            ImageFormat.JPEG: JPEGValidator,
            ImageFormat.PNG: PNGValidator,
            ImageFormat.TIFF: TIFFValidator,
            ImageFormat.WEBP: WebPValidator,
            ImageFormat.HEIC: HEIFValidator,
            ImageFormat.AVIF: HEIFValidator,
            ImageFormat.BMP: BMPValidator,
            ImageFormat.GIF: GIFValidator,
            ImageFormat.PSD: PSDValidator,
            ImageFormat.ICO: ICOValidator,
        }
    
    def _load_default_signatures(self) -> List[Signature]:
        return [
            Signature(bytes.fromhex("FFD8FF"), ImageFormat.JPEG),
            Signature(bytes.fromhex("89504E470D0A1A0A"), ImageFormat.PNG),
            Signature(bytes.fromhex("474946383761"), ImageFormat.GIF),
            Signature(bytes.fromhex("474946383961"), ImageFormat.GIF),
            Signature(bytes.fromhex("424D"), ImageFormat.BMP),
            Signature(bytes.fromhex("49492A00"), ImageFormat.TIFF),
            Signature(bytes.fromhex("4D4D002A"), ImageFormat.TIFF),
            Signature(bytes.fromhex("52494646"), ImageFormat.WEBP), # RIFF
            Signature(bytes.fromhex("0000000066747970"), ImageFormat.HEIC, offset_hint=4), # Broad ISO BMFF
            Signature(bytes.fromhex("38425053"), ImageFormat.PSD),
            Signature(bytes.fromhex("00000100"), ImageFormat.ICO),
            Signature(bytes.fromhex("0000000C6A502020"), ImageFormat.J2K),
            Signature(bytes.fromhex("44445320"), ImageFormat.DDS),
        ]

    def register_validator(self, fmt: ImageFormat, validator_cls: type):
        self.validators[fmt] = validator_cls

    def carve(self, block_data: memoryview, base_offset: int) -> Iterator[ValidationResult]:
        """Orchestrate the carving process on a block of data."""
        potential_matches = self.indexer.scan(block_data)
        
        for entry in potential_matches:
            if entry.format in self.validators:
                validator_cls = self.validators[entry.format]
                # Slice data starting from the signature
                candidate_view = block_data[entry.offset:]
                validator = validator_cls(candidate_view)
                result = validator.validate()
                
                if result.score > 0:
                    result.offset_in_block = entry.offset
                    yield result
