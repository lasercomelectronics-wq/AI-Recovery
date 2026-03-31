from __future__ import annotations
from enum import Enum, auto
from ..base import BaseValidator, ImageFormat, ValidationResult, ForensicEvidence

class WebPValidator(BaseValidator):
    def __init__(self, data: memoryview):
        super().__init__(data)
        self.cursor = 0
        self.width = 0
        self.height = 0
        self.type = ""
        
    def _read_u32_le(self) -> int:
        if self.cursor + 4 > len(self.data): return -1
        val = self.data[self.cursor] | (self.data[self.cursor+1] << 8) | \
              (self.data[self.cursor+2] << 16) | (self.data[self.cursor+3] << 24)
        self.cursor += 4
        return val

    def validate(self) -> ValidationResult:
        if len(self.data) < 12 or self.data[0:4] != b'RIFF' or self.data[8:12] != b'WEBP':
            return ValidationResult(ImageFormat.WEBP, score=0.0)
            
        self.evidence.log("RIFF Header and WEBP FourCC found")
        riff_size = self._read_u32_le() + 8
        self.cursor = 12 # Start of chunks
        
        found_vp8 = False
        
        while self.cursor < riff_size and self.cursor < len(self.data):
            fourcc = self.data[self.cursor:self.cursor+4].tobytes()
            self.cursor += 4
            size = self._read_u32_le()
            if size < 0: break
            
            type_str = fourcc.decode('ascii', errors='ignore')
            self.evidence.log(f"WebP Chunk: {type_str}, Size: {size}")
            
            if type_str == "VP8 ":
                self._parse_vp8(self.data[self.cursor:self.cursor+size])
                found_vp8 = True
            elif type_str == "VP8L":
                self._parse_vp8l(self.data[self.cursor:self.cursor+size])
                found_vp8 = True
            elif type_str == "VP8X":
                self._parse_vp8x(self.data[self.cursor:self.cursor+size])
                found_vp8 = True
                
            self.cursor += size
            if size % 2 != 0: self.cursor += 1 # RIFF padding

        score = 30.0 # Header
        rationale = "RIFF/WEBP Header validated. "
        
        if found_vp8:
            score += 40.0
            rationale += f"VP8 data found: {self.width}x{self.height}. "
            if self.width > 0 and self.height > 0:
                score += 30.0
        
        return ValidationResult(
            format=ImageFormat.WEBP,
            score=min(100.0, score),
            width=self.width,
            height=self.height,
            integrity_complete=found_vp8,
            evidence=self.evidence,
            total_size=self.cursor
        )

    def _parse_vp8(self, chunk: memoryview):
        if len(chunk) < 10: return
        # VP8 bitstream header check
        if chunk[3:6] == b'\x9D\x01\x2A':
            self.width = (chunk[6] | (chunk[7] << 8)) & 0x3FFF
            self.height = (chunk[8] | (chunk[9] << 8)) & 0x3FFF
            
    def _parse_vp8l(self, chunk: memoryview):
        if len(chunk) < 5 or chunk[0] != 0x2F: return
        # Simple VP8L width/height parsing (bits 1-14 and 15-28)
        bits = chunk[1] | (chunk[2] << 8) | (chunk[3] << 16) | (chunk[4] << 24)
        self.width = (bits & 0x3FFF) + 1
        self.height = ((bits >> 14) & 0x3FFF) + 1
        
    def _parse_vp8x(self, chunk: memoryview):
        if len(chunk) < 10: return
        # Extended header: 24-bit width and 24-bit height
        self.width = (chunk[4] | (chunk[5] << 8) | (chunk[6] << 16)) + 1
        self.height = (chunk[7] | (chunk[8] << 8) | (chunk[9] << 16)) + 1
