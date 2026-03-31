from __future__ import annotations
from ..base import BaseValidator, ImageFormat, ValidationResult, ForensicEvidence

class BMPValidator(BaseValidator):
    def validate(self) -> ValidationResult:
        if len(self.data) < 14 or self.data[0:2] != b'BM':
            return ValidationResult(ImageFormat.BMP, score=0.0)
        
        # File Size (offset 2)
        expected_size = self.data[2] | (self.data[3] << 8) | (self.data[4] << 16) | (self.data[5] << 24)
        # Header size (offset 14)
        header_size = self.data[14] | (self.data[15] << 8) | (self.data[16] << 16) | (self.data[17] << 24)
        
        width = self.data[18] | (self.data[19] << 8) | (self.data[20] << 16) | (self.data[21] << 24)
        height = self.data[22] | (self.data[23] << 8) | (self.data[24] << 16) | (self.data[25] << 24)
        
        score = 30.0 # Magic
        if 1 < width < 65535 and 1 < height < 65535:
            score += 40.0
        if expected_size <= len(self.data):
            score += 30.0
            
        return ValidationResult(ImageFormat.BMP, score=score, width=width, height=height, total_size=expected_size)

class GIFValidator(BaseValidator):
    def validate(self) -> ValidationResult:
        if len(self.data) < 10 or self.data[0:6] not in [b'GIF87a', b'GIF89a']:
            return ValidationResult(ImageFormat.GIF, score=0.0)
            
        width = self.data[6] | (self.data[7] << 8)
        height = self.data[8] | (self.data[9] << 8)
        
        # Scanning for trailer 0x3B
        eoi_idx = self.data.tobytes().find(b'\x3B')
        
        score = 30.0 # Magic
        if eoi_idx >= 0:
            score += 50.0
            
        return ValidationResult(ImageFormat.GIF, score=score, width=width, height=height, total_size=(eoi_idx + 1 if eoi_idx >= 0 else 0))

class PSDValidator(BaseValidator):
    def validate(self) -> ValidationResult:
        if len(self.data) < 26 or self.data[0:4] != b'8BPS':
            return ValidationResult(ImageFormat.PSD, score=0.0)
        
        version = (self.data[4] << 8) | self.data[5]
        if version not in [1, 2]: return ValidationResult(ImageFormat.PSD, score=0.0)
        
        # PSD header: Channels(2), Height(4), Width(4), Depth(2), Mode(2)
        height = (self.data[12] << 24) | (self.data[13] << 16) | (self.data[14] << 8) | self.data[15]
        width = (self.data[16] << 24) | (self.data[17] << 16) | (self.data[18] << 8) | self.data[19]
        
        return ValidationResult(ImageFormat.PSD, score=80.0, width=width, height=height)

class ICOValidator(BaseValidator):
    def validate(self) -> ValidationResult:
        if len(self.data) < 6 or self.data[0:4] != b'\x00\x00\x01\x00':
            return ValidationResult(ImageFormat.ICO, score=0.0)
            
        num_images = self.data[4] | (self.data[5] << 8)
        if num_images == 0: return ValidationResult(ImageFormat.ICO, score=10.0)
        
        # Direct width/height from the first icon entry
        width = self.data[6] if self.data[6] != 0 else 256
        height = self.data[7] if self.data[7] != 0 else 256
        
        return ValidationResult(ImageFormat.ICO, score=90.0, width=width, height=height)
