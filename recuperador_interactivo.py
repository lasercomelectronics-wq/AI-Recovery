from __future__ import annotations

import argparse
import asyncio
import gc
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import psutil
import time

from rich.console import Console
from rich.live import Live

sys.path.insert(0, "src")

from blackops.core.fragment_graph import Fragment
from blackops.core.processor import ImageProcessor
from blackops.core.reader import RawReader
from blackops.core.live_recon import ActiveCarveSession, init_ffmpeg_worker, stop_ffmpeg_worker
from blackops.core.signature_engine import SignatureEngine
from blackops.core.video_reconstructor import VideoReconstructor
from blackops.core.video_signature_engine import VideoSignatureEngine
from blackops.core.hardware_governor import HardwareGovernor
from blackops.dedup.filter import DedupFilter
from blackops.ui.telemetry import TelemetryState, TelemetryUI

MIN_SCORE = 40
MAX_FILE_SIZE = 32 * 1024 * 1024
MIN_RESOLUTION = 600
RESERVA_SEGURIDAD = 500 * 1024 * 1024
MB = 1024 * 1024
BLOCK_SIZE = 128 * MB
OVERLAP_SIZE = 32 * MB
IMAGENES_POR_CARPETA = 1000
UI_REFRESH_SECONDS = 1

_err_console = Console(stderr=True)


def preguntar_ruta_origen() -> Path:
    while True:
        ruta = input("Ruta archivo RAW/IMG: ").strip()
        path = Path(ruta).expanduser().resolve()
        if path.is_file():
            return path
        print("❌ Ruta inválida")


def preguntar_ruta_destino() -> Path:
    while True:
        ruta = input("Ruta destino: ").strip()
        if not ruta:
            print("❌ Vacía")
            continue
        path = Path(ruta).expanduser().resolve()
        try:
            path.mkdir(parents=True, exist_ok=True)
            return path
        except OSError as exc:
            print(f"❌ {exc}")


def crear_nueva_subcarpeta(destino_base: Path, numero: int) -> tuple[Path, str]:
    name = f"{numero:03d}" if numero <= 999 else f"{numero:04d}"
    p = destino_base / name
    p.mkdir(parents=True, exist_ok=True)
    return p, name


def maybe_cleanup(destino: str | Path, force: bool = False) -> None:
    base = Path(destino)
    items = [*base.glob("*.json"), *[d for d in base.iterdir() if d.is_dir() and d.name.isdigit()]]
    if not items:
        return
    if not force:
        if input("Limpiar destino previo? (s/n): ").lower().strip() != "s":
            return
    for item in items:
        if item.is_dir():
            shutil.rmtree(item, ignore_errors=True)
        else:
            item.unlink(missing_ok=True)


def _watchdog_log(msg: str) -> None:
    with open("debug_vulcan.log", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat()}] {msg}\n")


def run_image_mode(
    reader: RawReader,
    out_dir: Path,
    ui: TelemetryUI,
    state: TelemetryState,
    governor: HardwareGovernor,
) -> dict:
    processor = ImageProcessor(min_score=MIN_SCORE)
    dedup = DedupFilter()
    signatures = SignatureEngine()

    signature_hits: dict[str, list[int]] = defaultdict(list)

    # Stateful JPEG carving sessions (CARVING_IN_PROGRESS).
    jpeg_sessions: dict[int, tuple[ActiveCarveSession, int]] = {}  # start_offset -> (session, score)

    previews_dir = out_dir / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)
    carving_dir = out_dir / "_carving" / "images"
    carving_dir.mkdir(parents=True, exist_ok=True)

    saved_total = duplicated = discarded_small = corrupted = 0
    fragmented_saved = 0
    fragment_idx = 0

    folder_idx = 1
    in_folder = 0
    current_dir, _ = crear_nueva_subcarpeta(out_dir, folder_idx)

    last_t = datetime.now()
    last_off = 0

    for processed, block in enumerate(reader.iter_blocks(), start=1):
        if processed == 1:
            _watchdog_log(f"Iniciando Chunk #{processed-1} (offset {block.offset})")

        # Cooperative yield without blocking
        time.sleep(0.01)
        governor.hold_if_ram_over_sync(sleep_sec=0.2)
        # Update chunk tracking for Telemetry UI
        state.chunk_index = block.chunk_index
        state.chunk_start = block.offset
        state.chunk_end = block.chunk_end
        state.blocks = processed
        
        now = datetime.now()
        dt = max((now - last_t).total_seconds(), 1e-6)
        mbps = ((block.offset - last_off) / (1024 * 1024)) / dt
        last_t, last_off = now, block.offset
        
        # update internal state for main thread renderer; avoid calling Rich UI from worker thread
        try:
            state.offset = block.offset
            # Append a speed sample for the main thread to render (best-effort)
            state.speed_samples.append((datetime.now(), mbps))
        except Exception:
            pass

        try:
            data = bytes(block.data)
            _watchdog_log(f"Block 0x{block.offset:X}: read {len(data)} bytes, invoking signature scan")
            for hit in signatures.scan_block(data, block.offset):
                signature_hits[hit.kind].append(hit.offset)

            _watchdog_log(f"Block 0x{block.offset:X}: calling processor.process()")
            candidates = processor.process(data, block.offset)
            _watchdog_log(f"Block 0x{block.offset:X}: processor.process() returned {len(candidates)} candidates")
            for candidate in candidates:
                if candidate.score >= MIN_SCORE and not dedup.is_duplicate(candidate.content_hash):
                    if candidate.fragmented:
                        start_off = candidate.offset
                        if start_off not in jpeg_sessions:
                            part_path = carving_dir / f"jpeg_{start_off:016X}.part"
                            sess = ActiveCarveSession.create(
                                family="jpg",
                                start_offset=start_off,
                                part_path=part_path,
                                previews_dir=previews_dir,
                                ext="jpg",
                                max_bytes=2 * 1024**3,
                            )
                            jpeg_sessions[start_off] = (sess, int(candidate.score))
                            corrupted += 1
                            state.fragmented = len(jpeg_sessions)
                    else:
                        img_data = reader.read_range(
                            candidate.offset,
                            candidate.offset + min(candidate.size + 512, MAX_FILE_SIZE),
                        )
                        ok, size = processor.validate_image(img_data, MIN_RESOLUTION)
                        if ok:
                            free = shutil.disk_usage(str(out_dir)).free
                            if free < len(img_data) + RESERVA_SEGURIDAD:
                                state.incidents.append("Espacio insuficiente: detención segura")
                                break

                            if in_folder >= IMAGENES_POR_CARPETA:
                                folder_idx += 1
                                in_folder = 0
                                current_dir, _ = crear_nueva_subcarpeta(out_dir, folder_idx)

                            in_folder += 1
                            saved_total += 1
                            state.recovered = saved_total
                            path = current_dir / f"img_{in_folder:04d}_{size[0]}x{size[1]}_{candidate.score}pts.jpg"
                            path.write_bytes(img_data)
                            dedup.add_hash(candidate.content_hash)
                        else:
                            discarded_small += 1

            # Stream append for all active sessions for this block.
            for start_off, (sess, score_pts) in list(jpeg_sessions.items()):
                appended, new_head = sess.write_from_block(block_offset=block.offset, block_data=block.data)
                _ = new_head  # GPU calibration not required for JPEG thumbnails; keep CPU low-impact.

                snap = sess.maybe_snapshot(block_idx=processed)
                if snap is not None:
                    state.last_preview_snapshot = str(snap.snapshot_path)
                    state.last_preview_thumb = str(snap.thumb_path) if snap.thumb_path else ""
                    state.incidents.append(f"JPEG disponible para previsualización (Live Recon): {snap.snapshot_path}")

                if sess.is_complete_for_preview():
                    ok, size = processor.validate_image(sess.part_path.read_bytes(), MIN_RESOLUTION)
                    if ok:
                        if in_folder >= IMAGENES_POR_CARPETA:
                            folder_idx += 1
                            in_folder = 0
                            current_dir, _ = crear_nueva_subcarpeta(out_dir, folder_idx)

                        in_folder += 1
                        saved_total += 1
                        fragmented_saved += 1
                        state.recovered = saved_total
                        fragment_idx += 1
                        out_path = current_dir / f"frag_{fragment_idx:04d}_{size[0]}x{size[1]}.jpg"
                        sess.close()
                        if out_path.exists():
                            state.incidents.append(f"Colisión de nombre, conservando parte: {out_path}")
                        else:
                            sess.part_path.replace(out_path)
                    else:
                        corrupted += 1
                        sess.close()

                    jpeg_sessions.pop(start_off, None)

        except Exception as exc:
            state.incidents.append(f"Bloque 0x{block.offset:X}: {exc}")

        if processed % 16 == 0:
            gc.collect()

        if processed == 1:
            _watchdog_log(f"Terminando Chunk #{processed-1} (offset {block.offset})")

    # Close leftover sessions.
    for _, (sess, _) in jpeg_sessions.items():
        sess.close()

    state.mode = "image-post"
    return {
        "imagenes_total": saved_total,
        "imagenes_fragmentadas_recuperadas": fragmented_saved,
        "duplicados": duplicated,
        "descartadas_pequenas": discarded_small,
        "corruptas_detectadas": corrupted,
        "firmas_detectadas": {k: len(v) for k, v in signature_hits.items()},
    }


def run_video_mode(
    reader: RawReader,
    out_dir: Path,
    ui: TelemetryUI,
    state: TelemetryState,
    governor: HardwareGovernor,
) -> dict:
    sig = VideoSignatureEngine()

    video_dir = out_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    # /previews/ es el directorio para Live Recon snapshots/thumbnail.
    previews_dir = out_dir / "previews"
    previews_dir.mkdir(parents=True, exist_ok=True)

    # .part live carving files.
    carving_dir = video_dir / "_carving"
    carving_dir.mkdir(parents=True, exist_ok=True)

    def _is_header_signature(hit) -> bool:
        # Start-of-container markers: ftyp/ftyp_brand, RIFF/AVI, FLV header, mpeg pack/sequence, ts sync.
        sid = str(getattr(hit, "signature_id", ""))
        fam = str(getattr(hit, "family", ""))
        if fam in {"mp4", "3gp"}:
            return sid.startswith("mp4_brand_") or sid.startswith("mp4_atom_ftyp")
        if fam == "avi":
            return sid.startswith("avi_riff") or sid.startswith("avi_formtype")
        if fam == "flv":
            return sid.startswith("flv_header")
        if fam in {"mpeg", "ts"}:
            return sid.startswith("mpeg_ps_pack") or sid.startswith("mpeg_seq_header") or sid.startswith("mpeg_ts_sync") or sid.startswith("x_ts_pat")
        return False

    def _start_offset_from_hit(hit) -> int:
        sid = str(getattr(hit, "signature_id", ""))
        fam = str(getattr(hit, "family", ""))
        # mp4 ftyp_brand fires at offset 8 inside the atom; rewind 4 bytes to include the atom size.
        if fam in {"mp4", "3gp"} and (sid.startswith("mp4_brand_") or sid.startswith("mp4_atom_ftyp")):
            return max(0, int(hit.offset) - 4)
        return max(0, int(hit.offset))

    # Active sessions: stateful inter-block carving.
    sessions: list[ActiveCarveSession] = []
    sessions_by_start: dict[int, ActiveCarveSession] = {}
    completed_offsets: set[tuple[int, str]] = set()
    recovered_videos = 0

    last_t = datetime.now()
    last_off = 0

    phase_metrics = {"hits_total": 0, "live_recon_snapshots": 0}

    for processed, block in enumerate(reader.iter_blocks(), start=1):
        if processed == 1:
            _watchdog_log(f"Iniciando Chunk #{processed-1} (offset {block.offset})")

        time.sleep(0.01)
        governor.hold_if_ram_over_sync(sleep_sec=0.2)
        # Update chunk tracking for Telemetry UI
        state.chunk_index = block.chunk_index
        state.chunk_start = block.offset
        state.chunk_end = block.chunk_end
        state.blocks = processed

        now = datetime.now()
        dt = max((now - last_t).total_seconds(), 1e-6)
        mbps = ((block.offset - last_off) / (1024 * 1024)) / dt
        last_t, last_off = now, block.offset
        
        try:
            state.offset = block.offset
            state.speed_samples.append((datetime.now(), mbps))
        except Exception:
            pass

        try:
            data = bytes(block.data)
            hits = sig.scan_block(data, block.offset)
            phase_metrics["hits_total"] += len(hits)

            # Stateful chunk handoff: open sessions on headers found in this block.
            for hit in hits:
                if not _is_header_signature(hit):
                    continue
                start_off = _start_offset_from_hit(hit)
                family_str = str(hit.family)
                if start_off in sessions_by_start:
                    continue
                if (start_off, family_str) in completed_offsets:
                    continue
                
                # Create a new .part for this container as soon as header appears.
                part_path = carving_dir / f"carve_{start_off:016X}.part"
                session = ActiveCarveSession.create(
                    family=family_str,
                    start_offset=start_off,
                    part_path=part_path,
                    previews_dir=previews_dir,
                    ext=None,
                    max_bytes=2 * 1024**3,
                )
                sessions.append(session)
                sessions_by_start[start_off] = session

            # Stream append from this block into each active session.
            for sess in list(sessions):
                appended, new_head = sess.write_from_block(block_offset=block.offset, block_data=block.data)
                if appended > 0 and new_head:
                    sess.gpu_entropy_calibration(new_head=new_head)

                snap = sess.maybe_snapshot(block_idx=processed)
                if snap is not None:
                    phase_metrics["live_recon_snapshots"] += 1
                    state.last_preview_snapshot = str(snap.snapshot_path)
                    state.last_preview_thumb = str(snap.thumb_path) if snap.thumb_path else ""
                    state.incidents.append(
                        f"Video/Payload para previsualización disponible (Live Recon): {snap.snapshot_path}"
                    )

                # Finalize when enough structure markers are present.
                # Usa el Validador Estructural (VALID, INCOMPLETE, INVALID)
                status = sess.is_complete_for_preview()
                if status == "VALID":
                    ext = sess.ext
                    recovered_videos += 1
                    state.recovered = recovered_videos
                    final_out = video_dir / f"recovered_{recovered_videos:04d}.{ext}"
                    sess.close()
                    # Overwrite protection
                    if not final_out.exists():
                        sess.part_path.replace(final_out)
                    else:
                        state.incidents.append(f"Final path collision, keeping part: {sess.part_path}")
                    sessions.remove(sess)
                    sessions_by_start.pop(sess.start_offset, None)
                    completed_offsets.add((sess.start_offset, sess.family))

                elif status == "INVALID":
                    # Mover el falso positivo a cuarentena, no suma a recuperados
                    sess.close()
                    quarantine_dir = out_dir / "_quarantine"
                    quarantine_dir.mkdir(parents=True, exist_ok=True)
                    quar_path = quarantine_dir / f"invalid_{sess.start_offset:016X}.part"
                    if not quar_path.exists():
                        sess.part_path.replace(quar_path)
                    else:
                        sess.part_path.unlink(missing_ok=True)
                    sessions.remove(sess)
                    sessions_by_start.pop(sess.start_offset, None)
                    # Añadir al completed_offsets para evitar re-apertura en el mismo offset
                    completed_offsets.add((sess.start_offset, sess.family))
        except Exception as exc:
            state.incidents.append(f"Video Live Recon bloque 0x{block.offset:X}: {exc}")

        if processed == 1:
            _watchdog_log(f"Terminando Chunk #{processed-1} (offset {block.offset})")

    # Finalize remaining sessions.
    for sess in list(sessions):
        try:
            status = sess.is_complete_for_preview()
            if status == "VALID":
                ext = sess.ext
                recovered_videos += 1
                state.recovered = recovered_videos
                final_out = video_dir / f"recovered_{recovered_videos:04d}.{ext}"
                sess.close()
                if not final_out.exists():
                    sess.part_path.replace(final_out)
            elif status == "INCOMPLETE":
                # Marcar explícitamente como parcial (truncado por EOF)
                ext = sess.ext
                final_out = video_dir / f"partial_{sess.start_offset:016X}.{ext}"
                sess.close()
                if not final_out.exists():
                    sess.part_path.replace(final_out)
            elif status == "INVALID":
                sess.close()
                quarantine_dir = out_dir / "_quarantine"
                quarantine_dir.mkdir(parents=True, exist_ok=True)
                quar_path = quarantine_dir / f"invalid_{sess.start_offset:016X}.part"
                if not quar_path.exists():
                    sess.part_path.replace(quar_path)
                else:
                    sess.part_path.unlink(missing_ok=True)
        finally:
            if sess in sessions:
                sessions.remove(sess)
            sessions_by_start.pop(sess.start_offset, None)
            completed_offsets.add((sess.start_offset, sess.family))

    state.mode = "video-post"
    return {
        "videos_recuperados": recovered_videos,
        "phase_metrics": phase_metrics,
        "signature_registry": sig.stats(),
    }


async def orchestrate(
    raw_path: str | Path,
    out_dir: str | Path,
    mode: str = "video",
    clean_destination: bool = False,
) -> dict:
    raw_p = Path(raw_path).expanduser().resolve()
    out_p = Path(out_dir).expanduser().resolve()
    out_p.mkdir(parents=True, exist_ok=True)
    maybe_cleanup(out_p, force=clean_destination)
    start = datetime.now()

    governor = HardwareGovernor()
    governor.apply_cpu_affinity()
    governor.configure_cupy_memory_pool_limit()
    worker_task = init_ffmpeg_worker()

    try:
        with RawReader(raw_p, block_size=BLOCK_SIZE, overlap_size=OVERLAP_SIZE, governor=governor) as reader:
            state = TelemetryState(
                mode=f"{mode}-scan",
                source=str(raw_p),
                destination=str(out_p),
                total_size=reader.size,
            )
            ui = TelemetryUI(state)

            with Live(ui.update(0, 0.0), refresh_per_second=1 / UI_REFRESH_SECONDS, screen=True) as live:
                loop = asyncio.get_running_loop()
                _watchdog_log(f"Orchestrate: scheduling worker for mode={mode}")
                try:
                    if mode == "image":
                        t = loop.run_in_executor(None, run_image_mode, reader, out_p, ui, state, governor)
                    else:
                        t = loop.run_in_executor(None, run_video_mode, reader, out_p, ui, state, governor)
                except Exception as exc:  # pragma: no cover - capture unexpected scheduling errors
                    _watchdog_log(f"Orchestrate: scheduling failed: {exc}")
                    raise
                _watchdog_log("Orchestrate: worker scheduled")
                
                while not t.done():
                    live.update(ui._cached_layout or ui.update(state.offset, 0.0))  # Avoid overwriting mbps stats
                    await asyncio.sleep(1.0)
                
                results = t.result()
    finally:
        await stop_ffmpeg_worker(worker_task)

    duration_min = (datetime.now() - start).total_seconds() / 60
    log_data = {
        "fecha": datetime.now().isoformat(),
        "modo": mode,
        "origen": str(raw_p),
        "destino": str(out_p),
        "duracion_min": duration_min,
        "bloques_procesados": state.blocks,
        "incidentes": state.incidents[-200:],
        **results,
    }
    log_path = out_p / "log_recuperacion.json"
    log_tmp = out_p / "log_recuperacion.json.tmp"
    log_tmp.write_text(json.dumps(log_data, indent=2), encoding="utf-8")
    log_tmp.replace(log_path)
    return log_data


async def main() -> None:
    parser = argparse.ArgumentParser(description="Recuperador forense interactivo")
    parser.add_argument("--mode", choices=["image", "video"], default="video", help="Modo operativo")
    parser.add_argument("--raw", default=None, help="Ruta RAW/IMG (si se omite, se pide interactivo)")
    parser.add_argument("--out", default=None, help="Ruta destino (si se omite, se pide interactivo)")
    parser.add_argument("--clean", action="store_true", help="Limpiar destino sin preguntar")
    args = parser.parse_args()

    raw_path = (
        Path(args.raw).expanduser().resolve()
        if args.raw
        else preguntar_ruta_origen()
    )
    out_dir = (
        Path(args.out).expanduser().resolve()
        if args.out
        else preguntar_ruta_destino()
    )

    if not raw_path.is_file():
        _err_console.print("[red]El archivo RAW no existe o la ruta es incorrecta[/red]")
        sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)

    log = await orchestrate(raw_path=raw_path, out_dir=out_dir, mode=args.mode, clean_destination=args.clean)
    print(f"\n✅ Finalizado modo {args.mode}. Resultados: { {k: v for k, v in log.items() if k not in ('incidentes', 'fecha')} }")


if __name__ == "__main__":
    asyncio.run(main())
