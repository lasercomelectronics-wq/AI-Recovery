from __future__ import annotations
import zlib
from enum import Enum, auto
from ..base import BaseValidator, ImageFormat, ValidationResult, ForensicEvidence

class PNGState(Enum):
    MAGIC = auto()
    IHDR = auto()
    CHUNKS = auto()
    COMPLETED = auto()
    ERROR = auto()

class PNGValidator(BaseValidator):
    def __init__(self, data: memoryview, full_crc_check: bool = False):
        super().__init__(data)
        self.cursor = 8 # Skip magic
        self.state = PNGState.MAGIC
        self.full_crc_check = full_crc_check
        self.width = 0
        self.height = 0
        self.chunks_read = 0
        
    def _read_u32(self) -> int:
        if self.cursor + 4 > len(self.data):
            return -1
        val = (self.data[self.cursor] << 24) | (self.data[self.cursor+1] << 16) | \
              (self.data[self.cursor+2] << 8) | self.data[self.cursor+3]
        self.cursor += 4
        return val

    def validate(self) -> ValidationResult:
        if len(self.data) < 8 or self.data[0:8] != b'\x89\x50\x4E\x47\x0D\x0A\x1A\x0A':
            return ValidationResult(ImageFormat.PNG, score=0.0)
            
        self.evidence.log("PNG Magic recognized [89 50 4E 47...]")
        self.state = PNGState.IHDR
        
        found_iend = False
        
        while self.state != PNGState.COMPLETED and self.state != PNGState.ERROR:
            # Chunk Header (Len + Type)
            if self.cursor + 8 > len(self.data):
                self.evidence.log(f"Truncated chunk header at offset {self.cursor}")
                self.state = PNGState.COMPLETED
                break
                
            length = self._read_u32()
            chunk_type = self.data[self.cursor:self.cursor+4].tobytes()
            self.cursor += 4
            
            if length < 0:
                self.state = PNGState.ERROR
                break
            
            self.chunks_read += 1
            type_str = chunk_type.decode('ascii', errors='ignore')
            
            # Start of data
            data_start = self.cursor
            self.cursor += length
            
            # CRC
            if self.cursor + 4 > len(self.data):
                self.evidence.log(f"Truncated CRC for chunk {type_str}")
                self.state = PNGState.COMPLETED
                break
            
            crc_read = self._read_u32()
            
            # Level 1 CRC: Always check IHDR
            should_check_crc = (type_str == "IHDR") or self.full_crc_check
            
            if should_check_crc:
                # CRC is calculated over chunk_type + chunk_data
                to_check = self.data[data_start-4 : data_start+length].tobytes()
                crc_calc = zlib.crc32(to_check) & 0xFFFFFFFF
                if crc_calc != crc_read:
                    self.evidence.log(f"CRC Mismatch in {type_str}: read {crc_read:08X}, calc {crc_calc:08X}")
                    if type_str == "IHDR":
                        self.state = PNGState.ERROR
                        break
                    else:
                        # Non-critical chunk CRC error
                        pass
                else:
                    self.evidence.log(f"CRC Verified for {type_str}")

            if type_str == "IHDR":
                if length != 13:
                    self.state = PNGState.ERROR
                    break
                view = self.data[data_start:data_start+13]
                self.width = (view[0] << 24) | (view[1] << 16) | (view[2] << 8) | view[3]
                self.height = (view[4] << 24) | (view[5] << 16) | (view[6] << 8) | view[7]
                self.evidence.log(f"IHDR processed: {self.width}x{self.height}")
                self.state = PNGState.CHUNKS
            elif type_str == "IEND":
                self.evidence.log("IEND found, image complete.")
                found_iend = True
                self.state = PNGState.COMPLETED
            elif type_str == "IDAT":
                # IDAT chunks contain the actual compressed pixel data
                pass
                
        # Calculate score
        score = 25.0 # Magic
        rationale = "PNG Magic identified. "
        
        if self.width > 0:
            score += 35.0
            rationale += f"IHDR validated with size {self.width}x{self.height}. "
        
        if self.chunks_read > 2: # At least IHDR + IDAT + IEND
            score += 15.0
        
        if found_iend:
            score += 25.0
            rationale += "IEND trailer found. "
        else:
            rationale += "Missing IEND trailer (possible fragmentation). "
            
        if self.state == PNGState.ERROR:
            score = 0
            rationale = "Fatal structural error in PNG."

        self.evidence.confidence_rationale = rationale.strip()
        
        return ValidationResult(
            format=ImageFormat.PNG,
            score=min(100.0, score),
            width=self.width,
            height=self.height,
            integrity_complete=found_iend,
            evidence=self.evidence,
            total_size=self.cursor
        )
