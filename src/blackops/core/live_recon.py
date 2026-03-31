from __future__ import annotations

import asyncio
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from blackops.gpu.entropy import entropy_cupy
from PIL import Image
from datetime import datetime

ACTIVE_SESSIONS: dict[str, "ActiveCarveSession"] = {}

_FFMPEG_QUEUE: asyncio.Queue | None = None

def init_ffmpeg_worker() -> asyncio.Task | None:
    global _FFMPEG_QUEUE
    _FFMPEG_QUEUE = asyncio.Queue()
    try:
        loop = asyncio.get_running_loop()
        return loop.create_task(_ffmpeg_worker_loop())
    except RuntimeError:
        return None

async def stop_ffmpeg_worker(task: asyncio.Task | None) -> None:
    if _FFMPEG_QUEUE is not None:
        await _FFMPEG_QUEUE.put(None)
    if task is not None:
        await task

async def _ffmpeg_worker_loop() -> None:
    while True:
        if _FFMPEG_QUEUE is None:
            break
        item = await _FFMPEG_QUEUE.get()
        if item is None:
            break
        snapshot_path, thumb_path, session = item
        await _async_ffmpeg_thumbnail(snapshot_path, thumb_path)
        # Notify session (optional, but keeps state clean)
        session.preview_status = "Listo"
        try:
            session.last_thumb_path = thumb_path if thumb_path.exists() and thumb_path.stat().st_size > 0 else None
        except Exception:
            pass


def _mkdir_parents(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _safe_rel_snapshot_name(session_id: str, block_idx: int, ext: str) -> str:
    return f"{session_id}_block{block_idx:06d}.{ext}"


def _stream_contains_markers(path: Path, markers: list[bytes], chunk_size: int = 8 * 1024 * 1024) -> dict[bytes, bool]:
    """
    Search markers in a file without loading it entirely.
    Notes:
      - This is a best-effort substring scan to decide "completion" for previews.
      - For mp4, we primarily need presence of atoms (ftyp/moov/mdat).
    """
    found: dict[bytes, bool] = {m: False for m in markers}
    if not path.exists():
        return found

    # Keep a small overlap to catch markers spanning chunk boundaries.
    keep_tail = max((len(m) for m in markers), default=0)
    tail = b""
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            data = tail + chunk
            for m in markers:
                if not found[m] and m in data:
                    found[m] = True
            tail = data[-keep_tail:]
            if all(found.values()):
                break
    return found


def validate_mp4_integrity(path: Path) -> str:
    """
    Returns "VALID", "INCOMPLETE", or "INVALID".

    Parser iterativo con stack de boundaries por contenedor.
    Cada átomo contenedor (moov, trak, mdia, etc.) empuja su límite al stack,
    y sus hijos se parsean exclusivamente dentro de ese rango.

    Usa offset explícito en lugar de f.tell() para control determinista del flujo.
    Valida header_size dinámico (8 vs 16 para extended 64-bit atoms).
    """

    # Átomos que son contenedores y cuyos hijos deben parsearse recursivamente.
    CONTAINER_ATOMS: set[bytes] = {
        b"moov", b"trak", b"mdia", b"minf", b"stbl", b"dinf",
        b"moof", b"traf", b"edts", b"udta", b"meta", b"mvex",
        b"sinf", b"schi", b"ipro",
    }

    try:
        file_size: int = path.stat().st_size
        if file_size < 8:
            return "INCOMPLETE"

        with path.open("rb") as f:
            # --- Validación rápida: el primer átomo DEBE ser ftyp ---
            f.seek(0, 0)
            first_header = f.read(8)
            if len(first_header) < 8 or first_header[4:8] != b"ftyp":
                return "INVALID"

            # --- Estado global de detección ---
            found_ftyp: bool = False
            found_moov: bool = False
            found_mdat: bool = False
            found_moof: bool = False

            # --- Stack iterativo: (offset_inicio, boundary_fin) ---
            # Arrancamos con el rango completo del archivo.
            stack: list[tuple[int, int]] = [(0, file_size)]

            while stack:
                offset, boundary = stack.pop()

                while offset < boundary:
                    # ¿Podemos leer al menos la cabecera mínima (8 bytes)?
                    if offset + 8 > boundary:
                        break

                    f.seek(offset, 0)
                    raw_header = f.read(8)
                    if len(raw_header) < 8:
                        break

                    size = int.from_bytes(raw_header[0:4], "big")
                    tag = raw_header[4:8]
                    header_size = 8

                    # --- Extended 64-bit size ---
                    if size == 1:
                        header_size = 16
                        if offset + 16 > boundary:
                            return "INCOMPLETE"
                        ext = f.read(8)
                        if len(ext) < 8:
                            return "INCOMPLETE"
                        size = int.from_bytes(ext, "big")
                    elif size == 0:
                        # size==0 significa "hasta el final del contenedor actual"
                        size = boundary - offset

                    # --- Validación de coherencia de tamaño ---
                    if size < header_size:
                        return "INVALID"

                    atom_end = offset + size
                    if atom_end > boundary:
                        # El átomo promete más bytes de los que caben: truncado.
                        return "INCOMPLETE"

                    # --- Registro de átomos clave ---
                    if tag == b"ftyp":
                        found_ftyp = True
                    elif tag == b"mdat":
                        found_mdat = True
                    elif tag == b"moov":
                        found_moov = True
                    elif tag == b"moof":
                        found_moof = True

                    # --- Descenso a contenedores ---
                    if tag in CONTAINER_ATOMS:
                        # Empujar al stack: parsear hijos dentro de [offset+header_size, atom_end)
                        # Pero primero avanzamos el cursor del nivel actual para que
                        # al regresar del stack, continuemos después de este contenedor.
                        # Usamos un truco: empujamos primero el "resto del nivel actual"
                        # y luego el interior del contenedor (LIFO: interior se procesa primero).
                        rest_offset = atom_end
                        if rest_offset < boundary:
                            stack.append((rest_offset, boundary))
                        # Ahora empujamos el interior del contenedor.
                        child_start = offset + header_size
                        if child_start < atom_end:
                            stack.append((child_start, atom_end))
                        # Cortamos el while interno; el stack controlará la continuación.
                        break
                    else:
                        # Átomo hoja: saltar al siguiente hermano.
                        offset = atom_end

            # --- Evaluación final ---
            if not found_ftyp:
                return "INVALID"
            if found_mdat and (found_moov or found_moof):
                return "VALID"
            return "INCOMPLETE"

    except Exception:
        return "INVALID"


def _guess_video_extension_from_family(family: str) -> str:
    if family in {"jpg", "jpeg"}:
        return "jpg"
    if family in {"mp4", "3gp"}:
        return "mp4"
    if family == "avi":
        return "avi"
    if family == "wmv":
        return "wmv"
    if family == "flv":
        return "flv"
    if family == "mpeg":
        return "mpeg"
    if family == "ts":
        return "ts"
    return "bin"


async def _async_ffmpeg_thumbnail(
    snapshot_path: Path,
    thumb_path: Path,
    *,
    max_size: tuple[int, int] = (320, 180),
    timeout_sec: float = 3.0,
) -> bool:
    """
    Best-effort thumbnail extraction using ffmpeg in async mode.
    Requirements:
      - aggressive error tolerance: -err_detect ignore_err
      - output first intelligible frame: -frames:v 1
      - strict timeout: kill process after timeout_sec
    """
    if not shutil.which("ffmpeg"):
        return False

    w, h = max_size
    # Aggressive flags to tolerate corrupt payloads.
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-err_detect",
        "ignore_err",
        "-fflags",
        "+discardcorrupt",
        "-flags2",
        "+fast",
        "-i",
        str(snapshot_path),
        "-frames:v",
        "1",
        "-vf",
        f"scale={w}:{h}:force_original_aspect_ratio=decrease",
        "-an",
        "-sn",
        str(thumb_path),
    ]

    with open("debug_vulcan.log", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat()}] Antes de llamar a ffmpeg para {snapshot_path.name}\n")

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.DEVNULL,  # type: ignore[attr-defined]
        stderr=asyncio.subprocess.DEVNULL,  # type: ignore[attr-defined]
    )
    try:
        await asyncio.wait_for(proc.communicate(), timeout=timeout_sec)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        finally:
            await proc.communicate()
        with open("debug_vulcan.log", "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().isoformat()}] TIMEOUT ffmpeg para {snapshot_path.name}\n")
        return False

    with open("debug_vulcan.log", "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now().isoformat()}] Despues de ffmpeg para {snapshot_path.name}\n")

    return thumb_path.exists() and thumb_path.stat().st_size > 0


def _pil_thumbnail(snapshot_path: Path, thumb_path: Path, max_size: tuple[int, int] = (320, 240)) -> bool:
    try:
        with Image.open(snapshot_path) as im:
            im.thumbnail(max_size)
            im.save(thumb_path, quality=85)
        return thumb_path.exists() and thumb_path.stat().st_size > 0
    except Exception:
        return False


@dataclass
class SnapshotInfo:
    snapshot_path: Path
    thumb_path: Optional[Path] = None


@dataclass
class ActiveCarveSession:
    session_id: str
    family: str
    ext: str

    # Absolute source offsets (for overlap de-dup).
    start_offset: int
    next_write_offset: int

    part_path: Path
    previews_dir: Path
    max_bytes: int = 2 * 1024**3  # safety bound; prevents infinite carving

    blocks_written: int = 0
    last_snapshot_block_idx: int = 0
    last_tail_valid: bytes = b""
    bytes_written_total: int = 0

    preview_status: str = "Generando Preview"  # or "Listo"
    last_snapshot_path: Optional[Path] = None
    last_thumb_path: Optional[Path] = None

    _fd = None

    @classmethod
    def create(
        cls,
        *,
        family: str,
        start_offset: int,
        part_path: Path,
        previews_dir: Path,
        ext: Optional[str] = None,
        max_bytes: int,
    ) -> "ActiveCarveSession":
        sid = uuid.uuid4().hex[:10]
        used_ext = ext or _guess_video_extension_from_family(family)
        _mkdir_parents(previews_dir)

        # Ensure part directory exists.
        part_path.parent.mkdir(parents=True, exist_ok=True)
        # Create/append from scratch for this session.
        with part_path.open("wb"):
            pass

        session = cls(
            session_id=sid,
            family=family,
            ext=used_ext,
            start_offset=start_offset,
            next_write_offset=start_offset,
            part_path=part_path,
            previews_dir=previews_dir,
            max_bytes=max_bytes,
        )
        ACTIVE_SESSIONS[session.session_id] = session
        return session

    def _open_part_if_needed(self) -> None:
        if self._fd is None:
            self._fd = self.part_path.open("ab", buffering=0)

    def close(self) -> None:
        try:
            if self._fd is not None:
                self._fd.close()
        finally:
            self._fd = None
        ACTIVE_SESSIONS.pop(self.session_id, None)

    def write_from_block(self, *, block_offset: int, block_data: memoryview) -> tuple[int, bytes]:
        """
        Append only bytes not already written (overlap de-dup),
        based on absolute offsets.
        Returns number of bytes appended.
        """
        if self.next_write_offset >= self.start_offset + self.max_bytes:
            return 0, b""

        block_start = block_offset
        block_end = block_offset + len(block_data)

        write_start_abs = max(self.next_write_offset, block_start)
        write_end_abs = min(self.start_offset + self.max_bytes, block_end)

        if write_end_abs <= write_start_abs:
            return 0, b""

        local_start = write_start_abs - block_start
        local_end = write_end_abs - block_start

        self._open_part_if_needed()
        chunk = block_data[local_start:local_end]
        # chunk is a view into block_data; write it immediately.
        self._fd.write(chunk)

        appended = local_end - local_start
        self.next_write_offset = write_end_abs
        self.blocks_written += 1
        self.bytes_written_total += appended

        # Keep a small head for GPU calibration.
        new_head = bytes(chunk[:4096]) if appended >= 1 else b""

        # Keep last 4KB valid context for GPU entropy calibration.
        if appended > 0:
            tail = bytes(chunk[max(0, appended - 4096) : appended])
            self.last_tail_valid = tail
        return appended, new_head

    def maybe_snapshot(self, *, block_idx: int) -> SnapshotInfo | None:
        """
        Every N blocks, create a reproducible snapshot by copying the .part to /previews/.
        For thumbnails, attempt ffmpeg extraction; for unsupported formats it will just skip.
        """
        if block_idx - self.last_snapshot_block_idx < 5:
            return None
        self.last_snapshot_block_idx = block_idx

        snapshot_name = _safe_rel_snapshot_name(self.session_id, block_idx, self.ext)
        snapshot_path = self.previews_dir / snapshot_name
        thumb_path = self.previews_dir / f"{self.session_id}_block{block_idx:06d}.thumb.jpg"

        # Make sure file handle is flushed before copy.
        if self._fd is not None:
            self._fd.flush()

        shutil.copyfile(self.part_path, snapshot_path)
        self.last_snapshot_path = snapshot_path

        # Thumbnail extraction:
        if self.ext in {"jpg", "jpeg"}:
            ok_thumb = _pil_thumbnail(snapshot_path, thumb_path)
            self.preview_status = "Listo"
            self.last_thumb_path = thumb_path if ok_thumb else None
            self.prune_old_previews()
            return SnapshotInfo(snapshot_path=snapshot_path, thumb_path=self.last_thumb_path)

        # MP4/video thumbnail: async ffmpeg with strict timeout (3s).
        self.preview_status = "Generando Preview"
        self.last_thumb_path = None

        global _FFMPEG_QUEUE
        if _FFMPEG_QUEUE is not None:
            try:
                _FFMPEG_QUEUE.put_nowait((snapshot_path, thumb_path, self))
            except asyncio.QueueFull:
                pass

        self.prune_old_previews()

        # Snapshot info returns without waiting for the thumbnail.
        return SnapshotInfo(snapshot_path=snapshot_path, thumb_path=None)

    def prune_old_previews(self) -> None:
        """
        Keep only last 3 snapshots per session to avoid SSD saturation.
        """
        if not self.previews_dir.exists():
            return

        # Snapshot files: <session_id>_block*.{ext}
        snapshot_glob = f"{self.session_id}_block*.{self.ext}"
        snapshots = sorted(
            self.previews_dir.glob(snapshot_glob),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        # Delete old snapshots except those needed globally (simplified protection since ffmpeg queue runs linearly)

        for old in snapshots[3:]:
            try:
                old.unlink(missing_ok=True)
            except Exception:
                pass

            # Delete matching thumbnail (same block idx).
            try:
                stem = old.stem  # <session_id>_block000123
                block_idx_str = stem.split("_block")[-1]
                block_idx = int(block_idx_str)
                thumb = self.previews_dir / f"{self.session_id}_block{block_idx:06d}.thumb.jpg"
                thumb.unlink(missing_ok=True)
            except Exception:
                pass

        # Also prune extra thumbnails that have no corresponding kept snapshot.
        kept_snapshot_stems = {s.stem for s in snapshots[:3]}
        for thumb in self.previews_dir.glob(f"{self.session_id}_block*.thumb.jpg"):
            try:
                # thumb.stem is: <session_id>_block000123.thumb
                # We'll reconstruct block idx by splitting.
                stem = thumb.name.replace(".thumb.jpg", "")
                # stem is: <session_id>_block000123
                if stem not in kept_snapshot_stems:
                    thumb.unlink(missing_ok=True)
            except Exception:
                pass

    def is_complete_for_preview(self) -> str:
        """
        Determine if the current .part contains enough markers to consider it "recoverable".
        Retorna string de estado estricto: "VALID", "INCOMPLETE", o "INVALID".
        """
        if self.family in {"mp4", "3gp"}:
            return validate_mp4_integrity(self.part_path)
        
        # Legacy para otros formatos (best effort fallback)
        if self.family == "avi":
            markers = [b"RIFF", b"AVI "]
            found = _stream_contains_markers(self.part_path, markers)
            return "VALID" if bool(found[b"RIFF"] and found[b"AVI "]) else "INCOMPLETE"
        if self.family == "flv":
            markers = [b"FLV\x01"]
            found = _stream_contains_markers(self.part_path, markers)
            return "VALID" if bool(found[b"FLV\x01"]) else "INCOMPLETE"
        if self.family == "wmv":
            markers = [bytes.fromhex("3026B2758E66CF11A6D900AA0062CE6C")]
            found = _stream_contains_markers(self.part_path, markers)
            return "VALID" if bool(found[markers[0]]) else "INCOMPLETE"
        if self.family == "mpeg":
            markers = [b"\x00\x00\x01\xBA", b"\x00\x00\x01\xB3"]
            found = _stream_contains_markers(self.part_path, markers)
            return "VALID" if bool(found[markers[0]] and found[markers[1]]) else "INCOMPLETE"
        if self.family == "ts":
            markers = [b"G", b"\x47"]  # weak marker
            found = _stream_contains_markers(self.part_path, [b"\x47"])
            return "VALID" if bool(found[b"\x47"]) else "INCOMPLETE"
        if self.family in {"jpg", "jpeg"}:
            # JPEG SOI/EOI markers. If EOI is present, we consider it recoverable for snapshot preview.
            markers = [bytes.fromhex("FFD8FF"), bytes.fromhex("FFD9")]
            found = _stream_contains_markers(self.part_path, markers)
            return "VALID" if bool(found[markers[0]] and found[markers[1]]) else "INCOMPLETE"
        return "INCOMPLETE"

    def gpu_entropy_calibration(self, *, new_head: bytes) -> None:
        """
        GPU context calibration:
        Use the last 4KB tail valid context + new_head to drive an entropy computation on GPU.
        This doesn't change correctness of the carve, but it satisfies "GPU context continuity"
        for delta computations.
        """
        if not new_head:
            return
        if not self.last_tail_valid:
            # Initialize tail context with current head.
            self.last_tail_valid = new_head[-4096:]
            return

        # Keep total payload small to avoid memory pressure.
        payload = self.last_tail_valid[-4096:] + new_head[:4096]
        try:
            _ = entropy_cupy(payload)
        except Exception:
            # If GPU isn't available or caps interfere, don't break carving.
            pass


def get_active_carve_tracks() -> list[dict[str, object]]:
    """
    Return a lightweight view of active stateful carving sessions for UI.
    """
    tracks: list[dict[str, object]] = []
    # Copy to avoid mutation during UI rendering.
    sessions = list(ACTIVE_SESSIONS.values())
    sessions.sort(key=lambda s: s.start_offset)
    for s in sessions:
        tracks.append(
            {
                "offset_original": s.start_offset,
                "type_label": "Video" if s.family not in {"jpg", "jpeg"} else "JPEG",
                "part_written_bytes": s.bytes_written_total,
                "status": s.preview_status,
            }
        )
    return tracks

