from __future__ import annotations

import asyncio
import gc
import os
import sys
import threading
from pathlib import Path

import typer

from recuperador_interactivo import orchestrate, preguntar_ruta_destino, preguntar_ruta_origen
from src.blackops.core.nal_reconstructor import find_h264_nals
from src.blackops.core.reader import RawReader
from src.blackops.core.hardware_governor import HardwareGovernor
from src.blackops.core.signature_database import SignatureDatabase
from src.blackops.core.video_signature_engine import VideoSignatureEngine
from src.blackops.gpu.entropy import entropy_cupy, sliding_entropy_cupy
from src.blackops.ui.telemetry import RUNTIME_CONTROLS, start_hotkey_listener

app = typer.Typer(help="BlackOps-Vulcan Orchestrator CLI")

# ─────────────────────────────────────────────────────────────────
# Modos registrados: se agregan nuevos como entrada en este dict.
# ─────────────────────────────────────────────────────────────────
AVAILABLE_MODES: dict[str, str] = {
    "1": "video",
    "2": "image",
}

MENU_TEXT = """
╔══════════════════════════════════════════════════╗
║        BLACKOPS-VULCAN  ·  Modo de Análisis      ║
╠══════════════════════════════════════════════════╣
║                                                  ║
║   1.  Analizar videos                            ║
║   2.  Analizar imágenes                          ║
║                                                  ║
╚══════════════════════════════════════════════════╝
"""


def _run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: dict[str, object] = {}
    error: dict[str, BaseException] = {}

    def _worker() -> None:
        try:
            result["value"] = asyncio.run(coro)
        except BaseException as exc:  # pragma: no cover
            error["exc"] = exc

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join()

    if "exc" in error:
        raise error["exc"]
    return result.get("value")


def _apply_thread_limit(mode: str) -> int:
    max_threads = 9 if mode == "performance" else 3
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(max_threads)
    return max_threads


def _ask_mode() -> str:
    """Menú interactivo: devuelve 'video' o 'image'."""
    typer.echo(MENU_TEXT)
    while True:
        choice = input("  Seleccioná una opción (1/2): ").strip()
        if choice in AVAILABLE_MODES:
            return AVAILABLE_MODES[choice]
        typer.secho("  ❌ Opción inválida. Intentá de nuevo.", fg=typer.colors.RED)


def _run_pipeline(
    raw_path: Path,
    out_dir: Path,
    mode: str,
    clean: bool,
) -> None:
    """Lógica compartida por el comando CLI y el flujo interactivo."""
    if not raw_path.is_file():
        typer.secho(
            "El archivo RAW no existe o la ruta es incorrecta",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    out_dir.mkdir(parents=True, exist_ok=True)

    start_hotkey_listener()
    runtime_mode, _ = RUNTIME_CONTROLS.snapshot()
    threads = _apply_thread_limit(runtime_mode)

    typer.clear()

    try:
        log = _run_async(
            orchestrate(raw_path=raw_path, out_dir=out_dir, mode=mode, clean_destination=clean)
        )
    finally:
        gc.collect()

    typer.echo("Pipeline finalizado.")
    typer.echo(f"Log: {out_dir / 'log_recuperacion.json'}")
    summary = {k: v for k, v in log.items() if k != 'incidentes'}
    typer.echo(f"Governor activo: {'60%' if runtime_mode == 'performance' else '20%'} | Hilos máximos: {threads}")
    typer.echo(f"Resumen: {summary}")


# ─────────────────────────────────────────────────────────────────
# Comandos typer
# ─────────────────────────────────────────────────────────────────

@app.command("run")
def run_pipeline(
    raw: Path | None = typer.Option(
        None,
        exists=False,
        path_type=Path,
        help="Ruta del archivo RAW/IMG",
    ),
    out: Path | None = typer.Option(
        None,
        path_type=Path,
        help="Directorio de salida",
    ),
    mode: str = typer.Option("", help="Modo: video|image (omitir para menú interactivo)"),
    clean: bool = typer.Option(False, help="Limpiar salida antes de ejecutar"),
) -> None:
    """Orquestador principal de toda la suite BlackOps-Vulcan."""

    # ── Resolución de modo ──
    if mode and mode not in {"video", "image"}:
        raise typer.BadParameter("mode debe ser 'video' o 'image'")

    selected_mode = mode if mode else _ask_mode()

    # ── Resolución de rutas ──
    raw_path = raw.expanduser().resolve() if raw is not None else preguntar_ruta_origen()
    out_dir = out.expanduser().resolve() if out is not None else preguntar_ruta_destino()

    _run_pipeline(raw_path, out_dir, selected_mode, clean)


@app.command("scan")
def scan(
    input_file: Path = typer.Argument(..., exists=True, readable=True),
    window: int = typer.Option(4096, help="Sliding entropy window size"),
    step: int = typer.Option(1024, help="Sliding entropy step"),
) -> None:
    db = SignatureDatabase()

    governor = HardwareGovernor()
    governor.apply_cpu_affinity()
    governor.configure_cupy_memory_pool_limit()

    with RawReader(input_file, governor=governor) as reader:
        sample = reader.read_range(0, min(reader.size, 1024 * 1024))

        detected = db.detect_container(sample)
        offsets = db.find_all_offsets(sample)
        entropy = entropy_cupy(sample)
        sliding = sliding_entropy_cupy(sample, window=window, step=step)
        nals = find_h264_nals(sample)

    typer.echo(f"File: {input_file}")
    typer.echo(f"Detected container: {detected.name}")
    typer.echo(f"Global entropy: {entropy:.4f}")
    typer.echo(f"Sliding windows: {len(sliding)}")
    typer.echo(f"NAL units detected (sample): {len(nals)}")

    top = sorted(((k, len(v)) for k, v in offsets.items()), key=lambda x: x[1], reverse=True)[:5]
    typer.echo("Top signature hits:")
    for name, count in top:
        typer.echo(f"  - {name}: {count}")


@app.command("scan-video")
def scan_video(
    input_file: Path = typer.Argument(..., exists=True, readable=True),
    sample_mb: int = typer.Option(16, help="Sample size in MB"),
) -> None:
    engine = VideoSignatureEngine()
    governor = HardwareGovernor()
    governor.apply_cpu_affinity()
    governor.configure_cupy_memory_pool_limit()

    with RawReader(input_file, governor=governor) as reader:
        data = reader.read_range(0, min(reader.size, sample_mb * 1024 * 1024))

    stats = engine.stats()
    hits = engine.scan_block(data, 0)
    typer.echo(f"Registry signatures: {stats['total']} | Families: {stats['families']}")
    typer.echo(f"Video signatures found: {len(hits)}")
    for hit in hits[:30]:
        typer.echo(f"- {hit.kind} @ 0x{hit.offset:X} (conf={hit.confidence:.2f})")


if __name__ == "__main__":
    app()
