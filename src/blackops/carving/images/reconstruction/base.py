from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
from ..base import ValidationResult, ImageFormat

@dataclass(slots=True)
class CandidateNode:
    offset: int
    data_size: int
    format: ImageFormat
    result: ValidationResult
    
    # Affinity hints
    next_offset_hint: Optional[int] = None
    affinity_score: float = 0.0
    
    # Adjacency list for graph-based reconstruction
    neighbors: List[CandidateNode] = field(default_factory=list)

class OffsetAffinity:
    """Calculates affinity between two candidate fragments."""
    
    def calculate(self, head: CandidateNode, tail: CandidateNode) -> float:
        # 1. Physical proximity score
        dist = tail.offset - (head.offset + head.data_size)
        prox_score = max(0.0, 1.0 - (abs(dist) / 1024 / 1024)) # 1MB range
        
        # 2. Format match score
        fmt_score = 1.0 if head.format == tail.format else 0.0
        
        return (prox_score * 0.4) + (fmt_score * 0.6)
