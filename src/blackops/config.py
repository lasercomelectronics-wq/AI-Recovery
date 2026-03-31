from __future__ import annotations

import os
from dataclasses import dataclass


MB = 1024 * 1024


def _int_from_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return int(default)
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} debe ser un entero en bytes. Recibido: {raw!r}") from exc


@dataclass(frozen=True, slots=True)
class ScanConfig:
    block_size_bytes: int = 512 * MB
    overlap_size_bytes: int = 64 * MB

    def __post_init__(self) -> None:
        if self.block_size_bytes <= 0:
            raise ValueError("block_size_bytes debe ser mayor que cero.")
        if self.overlap_size_bytes < 0:
            raise ValueError("overlap_size_bytes no puede ser negativo.")
        if self.overlap_size_bytes >= self.block_size_bytes:
            raise ValueError(
                "Configuración inválida: overlap_size_bytes debe ser menor que block_size_bytes. "
                f"Recibido block_size_bytes={self.block_size_bytes}, overlap_size_bytes={self.overlap_size_bytes}. "
                "Revisá unidades (MB vs bytes). Ejemplo correcto: 512*1024*1024 y 64*1024*1024."
            )

    @classmethod
    def from_env(cls) -> "ScanConfig":
        return cls(
            block_size_bytes=_int_from_env("BLACKOPS_BLOCK_SIZE_BYTES", 512 * MB),
            overlap_size_bytes=_int_from_env("BLACKOPS_OVERLAP_SIZE_BYTES", 64 * MB),
        )


DEFAULT_SCAN_CONFIG = ScanConfig.from_env()
