from __future__ import annotations

import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

import psutil

try:
    import cupy as cp
except Exception:
    cp = None
from rich.layout import Layout
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text


@dataclass(slots=True)
class RuntimeControls:
    mode: str = "performance"  # performance=60%, silent=20%
    ui_latency_sec: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def toggle(self) -> str:
        with self._lock:
            self.mode = "silent" if self.mode == "performance" else "performance"
            return self.mode

    def set_latency(self, latency: float) -> None:
        with self._lock:
            self.ui_latency_sec = max(0.0, float(latency))

    def snapshot(self) -> tuple[str, float]:
        with self._lock:
            return self.mode, self.ui_latency_sec


RUNTIME_CONTROLS = RuntimeControls()
_HOTKEY_STARTED = False


def start_hotkey_listener() -> None:
    global _HOTKEY_STARTED
    if _HOTKEY_STARTED:
        return
    if not sys.stdin.isatty():
        return

    def _loop() -> None:
        while True:
            try:
                key = sys.stdin.read(1)
            except Exception:
                return
            if not key:
                time.sleep(0.05)
                continue
            if key.lower() == "l":
                RUNTIME_CONTROLS.toggle()

    t = threading.Thread(target=_loop, daemon=True, name="blackops-hotkey-listener")
    t.start()
    _HOTKEY_STARTED = True


@dataclass(slots=True)
class TelemetryState:
    mode: str
    source: str
    destination: str
    total_size: int
    offset: int = 0
    blocks: int = 0
    recovered: int = 0
    fragmented: int = 0
    chunk_start: int = 0
    chunk_end: int = 0
    chunk_index: int = 0
    buffer_health: str = "unknown"
    incidents: list[str] = field(default_factory=list)
    speed_samples: deque[tuple[datetime, float]] = field(default_factory=lambda: deque(maxlen=600))
    # (Legacy) fields kept for backwards compatibility.
    last_preview_snapshot: str = ""
    last_preview_thumb: str = ""


class TelemetryUI:
    def __init__(self, state: TelemetryState) -> None:
        self.state = state
        self._render_interval_sec = 1.0  # strict 1Hz dashboard updates
        self._last_render_ts = -1e9
        self._cached_layout: Layout | None = None
        self.progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        )
        self.main_task = self.progress.add_task("Disco RAW", total=state.total_size)

    def update(self, offset: int, mbps: float) -> Layout:
        # Always track state for later renders, but render the heavy Rich layout at 1Hz.
        tick_start = time.perf_counter()
        self.state.offset = offset
        self.state.speed_samples.append((datetime.now(), mbps))
        now_ts = time.perf_counter()
        if now_ts - self._last_render_ts < self._render_interval_sec and self._cached_layout is not None:
            return self._cached_layout

        # Render heavy Rich layout (tables/panels) only at 1Hz.
        self.progress.update(self.main_task, completed=offset)

        # Fix ETA matemático: usar ventana móvil de velocidad reciente en vez de promedio temporal.
        recent_samples = list(self.state.speed_samples)[-10:]
        sample = [v for t, v in recent_samples]
        avg_speed = sum(sample) / max(len(sample), 1)
        remaining = max(self.state.total_size - offset, 0)
        
        WARMUP_THRESHOLD = 2 * 128 * 1024 * 1024
        MIN_VALID_SPEED = 1e-3  # MB/s
        MAX_REASONABLE_ETA = 48 * 3600  # 48 hours

        if offset < WARMUP_THRESHOLD:
            eta_str = "Calculando..."
        elif avg_speed <= MIN_VALID_SPEED:
            eta_str = "Calculando..."
        else:
            eta_sec = (remaining / (1024 * 1024)) / avg_speed
            if eta_sec > MAX_REASONABLE_ETA:
                eta_str = "∞"
            else:
                eta_str = f"{eta_sec/60:.1f} min"

        layout = Layout()
        layout.split_column(
            Layout(name="header", size=5),
            Layout(name="telemetry", size=13),
            Layout(name="progress", size=6),
            Layout(name="incidents", size=6),
            Layout(name="live_recon", size=6),
        )

        rt_mode, ui_latency = RUNTIME_CONTROLS.snapshot()
        header_txt = (
            f"BLACKOPS-VULCAN | {self.state.mode} | RAW: {self.state.source} | OUT: {self.state.destination} "
            f"| Governor: {'60%' if rt_mode == 'performance' else '20%'} | Toggle: [L]"
        )
        layout["header"].update(Panel(header_txt, title="Control"))

        chunk_total = max(self.state.chunk_end - self.state.chunk_start, 1)
        chunk_rel = max(0.0, min((offset - self.state.chunk_start) / chunk_total, 1.0)) * 100.0

        table = Table.grid(expand=True)
        table.add_column()
        table.add_column()
        table.add_row("Offset Absoluto", f"0x{offset:016X} ({offset:,})")
        table.add_row("Chunk", f"#{self.state.chunk_index} | 0x{self.state.chunk_start:X}..0x{self.state.chunk_end:X}")
        table.add_row("Progreso Chunk", f"{chunk_rel:.1f}%")
        table.add_row("Bloques", f"{self.state.blocks:,}")
        table.add_row("Velocidad", f"{avg_speed:.2f} MB/s")
        table.add_row("ETA", eta_str)

        gpu_txt = "N/A"
        if cp is not None:
            try:
                free_b, total_b = cp.cuda.runtime.memGetInfo()
                used = 100.0 * (1.0 - (free_b / total_b)) if total_b else 0.0
                gpu_txt = f"{used:.1f}%"
            except Exception:
                gpu_txt = "N/A"

        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent
        cpu_style = "red" if cpu >= 60.0 else "green"
        ram_style = "red" if ram >= 60.0 else "green"
        gpu_num = None
        if gpu_txt.endswith("%"):
            try:
                gpu_num = float(gpu_txt.rstrip("%"))
            except Exception:
                gpu_num = None
        gpu_style = "red" if (gpu_num is not None and gpu_num >= 60.0) else "green"
        table.add_row(
            "CPU / RAM / GPU",
            f"[{cpu_style}]{cpu:.1f}%[/{cpu_style}] / [{ram_style}]{ram:.1f}%[/{ram_style}] / [{gpu_style}]{gpu_txt}[/{gpu_style}]",
        )
        table.add_row("Buffer Health", self.state.buffer_health)
        table.add_row("UI Latencia", f"{ui_latency:.2f}s")
        table.add_row("Recuperadas", f"{self.state.recovered} (fragmentadas: {self.state.fragmented})")
        layout["telemetry"].update(Panel(table, title="Telemetría"))

        layout["progress"].update(Panel(self.progress, title="Progreso"))

        radar = Table.grid(expand=True)
        radar.add_column()
        radar.add_column()
        radar.add_row("GPU Radar", "RTX 4060 (batch sync activo)")
        radar.add_row("Estado", "OK" if gpu_txt != "N/A" else "Sin telemetría CuPy")
        radar.add_row("Power Mode", "Performance (60%)" if rt_mode == "performance" else "Silencioso (20%)")

        inc = "\n".join(self.state.incidents[-6:]) or "Sin incidentes"
        layout["incidents"].split_row(
            Layout(Panel(inc, title="Incidentes")),
            Layout(Panel(radar, title="Radar GPU")),
        )

        # Live Recon multi-track panel (active stateful carving sessions).
        try:
            from blackops.core.live_recon import get_active_carve_tracks
        except Exception:
            from src.blackops.core.live_recon import get_active_carve_tracks

        tracks = get_active_carve_tracks()
        tracks_table = Table.grid(expand=True)
        tracks_table.add_column()
        tracks_table.add_column()
        tracks_table.add_column()
        tracks_table.add_column()
        tracks_table.add_row(
            "[bold]Offset Original[/bold]",
            "[bold]Tipo[/bold]",
            "[bold]Tamaño Escrito[/bold]",
            "[bold]Status[/bold]",
        )

        if not tracks:
            tracks_table.add_row("—", "—", "—", "—")
        else:
            for tr in tracks:
                off = int(tr["offset_original"])  # type: ignore[arg-type]
                typ = str(tr["type_label"])  # type: ignore[arg-type]
                part_bytes = int(tr["part_written_bytes"])  # type: ignore[arg-type]
                size_mb = part_bytes / (1024 * 1024) if part_bytes else 0.0
                status = str(tr["status"])  # type: ignore[arg-type]
                tracks_table.add_row(f"0x{off:016X}", typ, f"{size_mb:.2f} MB", status)

        layout["live_recon"].update(Panel(tracks_table, title="Live Recon (Multi-track)"))

        elapsed = time.perf_counter() - tick_start
        RUNTIME_CONTROLS.set_latency(elapsed)
        self._last_render_ts = now_ts
        self._cached_layout = layout
        return layout
