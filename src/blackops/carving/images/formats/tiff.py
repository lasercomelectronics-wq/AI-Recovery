from __future__ import annotations
import struct
from enum import Enum, auto
from ..base import BaseValidator, ImageFormat, ValidationResult, ForensicEvidence

class TIFFValidator(BaseValidator):
    def __init__(self, data: memoryview):
        super().__init__(data)
        self.cursor = 0
        self.endian = '<' # Default LE
        self.width = 0
        self.height = 0
        self.ifd_count = 0
        
    def _read_u16(self) -> int:
        if self.cursor + 2 > len(self.data): return -1
        val = struct.unpack_from(f"{self.endian}H", self.data, self.cursor)[0]
        self.cursor += 2
        return val

    def _read_u32(self) -> int:
        if self.cursor + 4 > len(self.data): return -1
        val = struct.unpack_from(f"{self.endian}I", self.data, self.cursor)[0]
        self.cursor += 4
        return val

    def validate(self) -> ValidationResult:
        if len(self.data) < 8:
            return ValidationResult(ImageFormat.TIFF, score=0.0)
            
        magic = self.data[0:2]
        if magic == b'II':
            self.endian = '<'
            self.evidence.log("TIFF Little Endian [II]")
        elif magic == b'MM':
            self.endian = '>'
            self.evidence.log("TIFF Big Endian [MM]")
        else:
            return ValidationResult(ImageFormat.TIFF, score=0.0)
            
        self.cursor = 2
        version = self._read_u16()
        if version != 42:
            self.evidence.log(f"Invalid TIFF version: {version}")
            return ValidationResult(ImageFormat.TIFF, score=0.0)
            
        self.evidence.add_step("TIFF_IFD_TRAVERSAL")
        ifd_offset = self._read_u32()
        
        # Traverse IFD tree
        while ifd_offset > 0 and ifd_offset < len(self.data):
            self.cursor = ifd_offset
            self.ifd_count += 1
            num_entries = self._read_u16()
            if num_entries < 0 or self.cursor + (num_entries * 12) > len(self.data):
                self.evidence.log(f"Corrupt IFD at {ifd_offset}")
                break
                
            self.evidence.log(f"IFD #{self.ifd_count} found at {ifd_offset} with {num_entries} entries")
            
            for _ in range(num_entries):
                tag = self._read_u16()
                type_id = self._read_u16()
                count = self._read_u32()
                value_or_offset = self._read_u32()
                
                # Tag 256: ImageWidth, 257: ImageHeight
                if tag == 256: self.width = value_or_offset
                elif tag == 257: self.height = value_or_offset
                
            # Last 4 bytes of IFD is offset to next IFD
            ifd_offset = self._read_u32()
            if self.ifd_count > 20: # Safety break
                self.evidence.log("Too many IFDs, potential recursion trap.")
                break

        score = 30.0 # Magic
        rationale = "TIFF Header validated. "
        
        if self.ifd_count > 0:
            score += 40.0
            rationale += f"{self.ifd_count} IFD(s) parsed. "
            if self.width > 0 and self.height > 0:
                score += 30.0
                rationale += f"Dimensions extracted: {self.width}x{self.height}. "
        
        return ValidationResult(
            format=ImageFormat.TIFF,
            score=min(100.0, score),
            width=self.width,
            height=self.height,
            integrity_complete=self.ifd_count > 0,
            evidence=self.evidence,
            total_size=self.cursor
        )
