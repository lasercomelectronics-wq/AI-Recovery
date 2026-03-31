from __future__ import annotations
import math
from typing import List
from .base import ValidationResult, ForensicEvidence

class WeightedScorer:
    """Refinement of scores based on multi-factor evidence."""
    
    @staticmethod
    def calculate_entropy(data: memoryview) -> float:
        """Calculate Shannon entropy of the data block."""
        if not data: return 0.0
        counts = [0] * 256
        for byte in data:
            counts[byte] += 1
        entropy = 0.0
        length = len(data)
        for count in counts:
            if count == 0: continue
            p = count / length
            entropy -= p * math.log2(p)
        return entropy

    def refine(self, result: ValidationResult, data: memoryview) -> ValidationResult:
        """Apply high-level refinements to the structural score."""
        
        # 1. Entropy sanity check
        # Images (especially compressed like JPEG/PNG) should have high entropy (usually > 7.0)
        # BMPs might have lower entropy depending on content.
        entropy = self.calculate_entropy(data[:4096])
        result.metadata['entropy'] = entropy
        
        if entropy < 1.0: # Basically zero-filled or repeating
            result.score *= 0.1
            result.evidence.log(f"Low entropy detected ({entropy:.2f}), likely false positive.")
        elif entropy < 4.0 and result.format not in [None]: # Low entropy for compressed formats
            result.score *= 0.5
            result.evidence.log(f"Suspiciously low entropy for compressed format ({entropy:.2f}).")
        
        # 2. Confidence rationale weighting
        # (This is already mostly handled in validators, but we can add global penalties)
        
        return result
