from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Dict, Any

class ImageFormat(Enum):
    JPEG = auto()
    PNG = auto()
    GIF = auto()
    BMP = auto()
    TIFF = auto()
    WEBP = auto()
    HEIC = auto()
    AVIF = auto()
    PSD = auto()
    ICO = auto()
    J2K = auto()
    CR2 = auto()
    NEF = auto()
    ARW = auto()
    DNG = auto()
    TGA = auto()
    PCX = auto()
    DDS = auto()
    UNKNOWN = auto()

@dataclass(slots=True)
class ForensicEvidence:
    evidence_log: List[str] = field(default_factory=list)
    decision_path: List[str] = field(default_factory=list)
    confidence_rationale: str = ""
    
    def log(self, message: str):
        self.evidence_log.append(message)
    
    def add_step(self, step: str):
        self.decision_path.append(step)

@dataclass(slots=True)
class ValidationResult:
    format: ImageFormat
    score: float # 0.0 to 100.0
    width: Optional[int] = None
    height: Optional[int] = None
    integrity_complete: bool = False
    evidence: ForensicEvidence = field(default_factory=ForensicEvidence)
    metadata: Dict[str, Any] = field(default_factory=dict)
    offset_in_block: int = 0
    total_size: int = 0

class BaseValidator:
    """Base class for FSM-based image validators."""
    
    def __init__(self, data: memoryview):
        self.data = data
        self.evidence = ForensicEvidence()
    
    def validate(self) -> ValidationResult:
        raise NotImplementedError("Subclasses must implement validate()")

@dataclass(slots=True)
class Signature:
    magic: bytes
    format: ImageFormat
    offset_hint: int = 0 # Usually 0, but some formats might have offset magics
