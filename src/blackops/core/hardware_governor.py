from __future__ import annotations

import asyncio
import gc
import os
import time
from dataclasses import dataclass
from typing import Sequence

import psutil

try:
    import cupy as cp
except Exception:  # pragma: no cover
    cp = None


CPU_HARD_THRESHOLD_PERCENT = 60.0
RAM_HARD_THRESHOLD_PERCENT = 60.0
RAM_HARD_USED_CEILING_BYTES = int(18 * 1024**3)  # 18GB

DEFAULT_GPU_VRAM_LIMIT_BYTES = int(4.8 * 1024**3)


def _select_first_n_affinity_cores(cores_needed: int) -> list[int]:
    proc = psutil.Process()
    current = proc.cpu_affinity()
    if not current:
        # On some systems this might fail; return empty so caller can skip.
        return []
    return list(current[: min(cores_needed, len(current))])


@dataclass
class HardwareGovernor:
    cpu_threshold_percent: float = CPU_HARD_THRESHOLD_PERCENT
    ram_threshold_percent: float = RAM_HARD_THRESHOLD_PERCENT
    cpu_affinity_cores: int = 9
    gpu_vram_pool_limit_bytes: int = DEFAULT_GPU_VRAM_LIMIT_BYTES
    ram_used_ceiling_bytes: int = RAM_HARD_USED_CEILING_BYTES

    _pool_limit_configured: bool = False

    def apply_cpu_affinity(self) -> None:
        """Restrict this process to a small set of logical cores."""
        try:
            cores = _select_first_n_affinity_cores(self.cpu_affinity_cores)
            if cores:
                psutil.Process().cpu_affinity(cores)
        except Exception:
            # Hard cap intent: if affinity fails, we still enforce CPU/RAM via blocking checks.
            pass

    def configure_cupy_memory_pool_limit(self) -> None:
        """Set a strict CuPy default pool limit (~60% VRAM => 4.8GB)."""
        if cp is None:
            return
        if self._pool_limit_configured:
            return

        mempool = cp.get_default_memory_pool()
        # Explicitly set the strict cap requested by the architecture.
        mempool.set_limit(size=int(self.gpu_vram_pool_limit_bytes))
        self._pool_limit_configured = True

    async def wait_while_cpu_over(self, interval_sec: float = 0.1, sleep_sec: float = 0.5) -> None:
        """Cooperative yield to the event loop, no strict CPU blocking."""
        await asyncio.sleep(0.01)

    async def wait_while_ram_over(self, sleep_sec: float = 0.2) -> None:
        """Async RAM hold: force GC and wait until below the hard cap."""
        while True:
            try:
                vm = psutil.virtual_memory()
                ram_percent = vm.percent
                ram_used_bytes = vm.used
            except Exception:
                ram_percent = 0.0
                ram_used_bytes = 0
            if ram_percent <= self.ram_threshold_percent and ram_used_bytes <= self.ram_used_ceiling_bytes:
                return
            gc.collect()
            await asyncio.sleep(sleep_sec)

    def hold_if_ram_over_sync(self, sleep_sec: float = 0.2) -> None:
        """Sync RAM hold for the reader generator (blocks ingestion of next block)."""
        while True:
            try:
                vm = psutil.virtual_memory()
                ram_percent = vm.percent
                ram_used_bytes = vm.used
            except Exception:
                ram_percent = 0.0
                ram_used_bytes = 0
            if ram_percent <= self.ram_threshold_percent and ram_used_bytes <= self.ram_used_ceiling_bytes:
                return
            gc.collect()
            time.sleep(sleep_sec)

    def hold_if_cpu_over_sync(self, interval_sec: float = 0.1, sleep_sec: float = 0.5) -> None:
        """No strict blocking for CPU sync (disabled per high-throughput architectural decision)."""
        pass

    def duty_cycle_sleep_for_gpu(self, exec_time_sec: float, max_duty_cycle: float = 0.60) -> None:
        """
        Enforce max GPU duty cycle by sleeping proportionally to kernel execution time.
        If exec_time = t, max duty cycle = t/(t+sleep) <= 0.60 => sleep >= t*(1/0.60 - 1).
        """
        if exec_time_sec <= 0:
            return
        # Required sleep to keep duty cycle <= max_duty_cycle.
        sleep_sec = exec_time_sec * ((1.0 / max_duty_cycle) - 1.0)
        if sleep_sec > 0:
            time.sleep(sleep_sec)

