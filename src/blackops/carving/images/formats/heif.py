from __future__ import annotations
from enum import Enum, auto
from ..base import BaseValidator, ImageFormat, ValidationResult, ForensicEvidence

class HEIFValidator(BaseValidator):
    def __init__(self, data: memoryview):
        super().__init__(data)
        self.cursor = 0
        self.width = 0
        self.height = 0
        self.brands = []
        
    def _read_u32(self) -> int:
        if self.cursor + 4 > len(self.data): return -1
        val = (self.data[self.cursor] << 24) | (self.data[self.cursor+1] << 16) | \
              (self.data[self.cursor+2] << 8) | self.data[self.cursor+3]
        self.cursor += 4
        return val

    def validate(self) -> ValidationResult:
        self.evidence.add_step("HEIF_BOX_PARSING")
        
        # ISO BMFF starts with 'ftyp' box
        first_len = self._read_u32()
        if first_len < 8 or first_len > len(self.data):
            return ValidationResult(ImageFormat.HEIC, score=0.0)
            
        ftyp_magic = self.data[self.cursor:self.cursor+4].tobytes()
        if ftyp_magic != b'ftyp':
            return ValidationResult(ImageFormat.HEIC, score=0.0)
            
        self.cursor = 4
        major_brand = self.data[self.cursor:self.cursor+4].tobytes().decode('ascii', errors='ignore')
        self.brands.append(major_brand)
        self.evidence.log(f"Major brand: {major_brand}")
        
        # Determine if HEIC or AVIF
        fmt = ImageFormat.HEIC
        if major_brand in ["avif", "avis"]:
            fmt = ImageFormat.AVIF
            
        # Skip ftyp remaining part
        self.cursor = first_len
        
        found_meta = False
        found_mdat = False
        
        while self.cursor + 8 <= len(self.data):
            box_len = self._read_u32()
            if box_len < 0: break
            if box_len == 1: # Large box (64-bit size)
                box_len = (self._read_u32() << 32) | self._read_u32()
                
            box_type = self.data[self.cursor:self.cursor+4].tobytes().decode('ascii', errors='ignore')
            self.cursor += 4
            self.evidence.log(f"ISO BMFF Box: {box_type}, Size: {box_len}")
            
            if box_type == "meta":
                found_meta = True
                # Meta box contains iprp, ispe (dimensions)
                self._parse_meta(self.data[self.cursor:self.cursor+box_len-8])
            elif box_type == "mdat":
                found_mdat = True
                
            if box_len > 0:
                self.cursor += box_len - 8
            else: # size 0 means until end of file
                break
                
        score = 20.0 # ftyp found
        rationale = "ftyp header present. "
        
        if found_meta:
            score += 40.0
            rationale += "Meta box parsed. "
            if self.width > 0 and self.height > 0:
                score += 20.0
                rationale += f"Dimensions extracted: {self.width}x{self.height}. "
                
        if found_mdat:
            score += 20.0
            rationale += "Mdat box found (possible fragmented data). "
            
        return ValidationResult(
            format=fmt,
            score=min(100.0, score),
            width=self.width,
            height=self.height,
            integrity_complete=found_meta and found_mdat,
            evidence=self.evidence,
            total_size=self.cursor
        )

    def _parse_meta(self, meta_data: memoryview):
        # The meta box has a 4-byte version/flags header
        if len(meta_data) < 4: return
        cursor = 4
        while cursor + 8 <= len(meta_data):
            b_len = (meta_data[cursor] << 24) | (meta_data[cursor+1] << 16) | \
                    (meta_data[cursor+2] << 8) | meta_data[cursor+3]
            b_type = meta_data[cursor+4:cursor+8].tobytes().decode('ascii', errors='ignore')
            
            if b_type == "iprp":
                self._parse_iprp(meta_data[cursor+8:cursor+b_len])
            
            cursor += b_len
            
    def _parse_iprp(self, iprp_data: memoryview):
        cursor = 0
        while cursor + 8 <= len(iprp_data):
            b_len = (iprp_data[cursor] << 24) | (iprp_data[cursor+1] << 16) | \
                    (iprp_data[cursor+2] << 8) | iprp_data[cursor+3]
            b_type = iprp_data[cursor+4:cursor+8].tobytes().decode('ascii', errors='ignore')
            
            if b_type == "ipco":
                self._parse_ipco(iprp_data[cursor+8:cursor+b_len])
                
            cursor += b_len

    def _parse_ipco(self, ipco_data: memoryview):
        cursor = 0
        while cursor + 8 <= len(ipco_data):
            b_len = (ipco_data[cursor] << 24) | (ipco_data[cursor+1] << 16) | \
                    (ipco_data[cursor+2] << 8) | ipco_data[cursor+3]
            b_type = ipco_data[cursor+4:cursor+8].tobytes().decode('ascii', errors='ignore')
            
            if b_type == "ispe": # Image Spatial Extents (Dimensions)
                self.width = (ipco_data[cursor+12] << 24) | (ipco_data[cursor+13] << 16) | \
                             (ipco_data[cursor+14] << 8) | ipco_data[cursor+15]
                self.height = (ipco_data[cursor+16] << 24) | (ipco_data[cursor+17] << 16) | \
                              (ipco_data[cursor+18] << 8) | ipco_data[cursor+19]
                
            cursor += b_len
