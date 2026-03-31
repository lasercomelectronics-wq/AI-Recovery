from __future__ import annotations
from enum import Enum, auto
from ..base import BaseValidator, ImageFormat, ValidationResult, ForensicEvidence

class JPEGState(Enum):
    INIT = auto()
    FIND_HEADER = auto()
    READ_MARKER = auto()
    SCAN_DATA = auto()
    COMPLETED = auto()
    ERROR = auto()

class JPEGValidator(BaseValidator):
    """
    Finite State Machine for JPEG validation.
    SOI -> APPn/DQT/DHT/SOF -> SOS -> EOI
    """
    
    def __init__(self, data: memoryview):
        super().__init__(data)
        self.cursor = 0
        self.state = JPEGState.INIT
        self.width = 0
        self.height = 0
        self.precision = 0
        self.components = 0
    
    def _read_u16(self) -> int:
        if self.cursor + 2 > len(self.data):
            return -1
        val = (self.data[self.cursor] << 8) | self.data[self.cursor + 1]
        self.cursor += 2
        return val

    def validate(self) -> ValidationResult:
        self.evidence.add_step("JPEG_FSM_START")
        
        # Initial check for SOI
        if len(self.data) < 2 or self.data[0:2] != b'\xFF\xD8':
            self.evidence.log("Missing SOI marker")
            return ValidationResult(ImageFormat.JPEG, score=0.0)
            
        self.cursor = 2
        self.evidence.log("SOI found [FF D8]")
        self.state = JPEGState.READ_MARKER
        
        found_sof = False
        found_sos = False
        found_eoi = False
        
        while self.state != JPEGState.COMPLETED and self.state != JPEGState.ERROR:
            # Bounds check
            if self.cursor + 2 > len(self.data):
                self.evidence.log(f"Partial image reached EOF at offset {self.cursor}")
                self.state = JPEGState.COMPLETED # Partial is still valid but with lower score
                break
                
            # Markers start with FF
            if self.data[self.cursor] != 0xFF:
                self.evidence.log(f"Expected marker FF at {self.cursor}, found {self.data[self.cursor]:02X}")
                # Heuristic: if we already found SOS, maybe it's just noisy data, but SOS marks start of entropy coded data
                if found_sos:
                    self.state = JPEGState.SCAN_DATA
                    continue
                else:
                    self.state = JPEGState.ERROR
                    break
            
            # Skip potential padding FFs
            while self.cursor < len(self.data) and self.data[self.cursor] == 0xFF:
                self.cursor += 1
            
            if self.cursor >= len(self.data):
                self.state = JPEGState.COMPLETED
                break
                
            marker = self.data[self.cursor]
            self.cursor += 1
            
            if marker == 0xD9: # EOI
                self.evidence.log(f"EOI found [FF D9] at offset {self.cursor}")
                found_eoi = True
                self.state = JPEGState.COMPLETED
            elif marker == 0xDA: # SOS (Start of Scan)
                self.evidence.log(f"SOS found [FF DA] at offset {self.cursor}")
                found_sos = True
                # Move until EOI or end of block
                self.state = JPEGState.SCAN_DATA
            elif 0xC0 <= marker <= 0xC3: # SOF0, SOF1, SOF2, SOF3
                length = self._read_u16()
                if length < 8:
                    self.state = JPEGState.ERROR
                    break
                self.precision = self.data[self.cursor]
                self.height = (self.data[self.cursor + 1] << 8) | self.data[self.cursor + 2]
                self.width = (self.data[self.cursor + 3] << 8) | self.data[self.cursor + 4]
                self.components = self.data[self.cursor + 5]
                self.evidence.log(f"SOF found [FF {marker:02X}] {self.width}x{self.height}, comps={self.components}")
                found_sof = True
                self.cursor += length - 2
            elif 0xD0 <= marker <= 0xD7: # RSTn (Reset markers)
                # Standalone, no length
                pass
            elif marker == 0x01: # TEM
                pass
            else:
                # Regular marker with length
                length = self._read_u16()
                if length < 2:
                    self.state = JPEGState.ERROR
                    break
                self.cursor += length - 2
        
        if self.state == JPEGState.SCAN_DATA:
            # Efficient scan for EOI in remaining data
            rem = self.data[self.cursor:]
            eoi_idx = rem.tobytes().find(b'\xFF\xD9')
            if eoi_idx >= 0:
                self.cursor += eoi_idx + 2
                found_eoi = True
                self.evidence.log(f"EOI recovered via scanning at relative offset {eoi_idx}")
            self.state = JPEGState.COMPLETED

        # Calculate weighted score
        score = 20.0 # Base for SOI
        rationale = "SOI found. "
        
        if found_sof:
            score += 30.0
            rationale += "Structural SOF found. "
            if 1 < self.width < 10000 and 1 < self.height < 10000:
                score += 20.0
                rationale += f"Valid dimensions extracted: {self.width}x{self.height}. "
        
        if found_sos:
            score += 15.0
            rationale += "SOS markers present. "
            
        if found_eoi:
            score += 15.0
            rationale += "EOI terminator verified. "
        else:
            rationale += "EOI missing (fragmented?). "
            
        if self.state == JPEGState.ERROR:
            score *= 0.5
            rationale += "Structural error detected. "

        self.evidence.confidence_rationale = rationale.strip()
        
        return ValidationResult(
            format=ImageFormat.JPEG,
            score=min(100.0, score),
            width=self.width,
            height=self.height,
            integrity_complete=found_eoi,
            evidence=self.evidence,
            total_size=self.cursor
        )
