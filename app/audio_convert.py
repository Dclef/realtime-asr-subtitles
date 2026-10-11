"""Optional ffmpeg fallback. Happy path is PCM→WAV in-process (no ffmpeg)."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def _find_ffmpeg() -> str | None:
    env = os.getenv("FFMPEG_PATH", "").strip()
    if env and Path(env).is_file():
        return env
    which = shutil.which("ffmpeg")
    if which:
        return which
    # Common Windows install from this project
    for candidate in (
        r"E:\ffmpeg\bin\ffmpeg.exe",
        r"C:\ffmpeg\bin\ffmpeg.exe",
    ):
        if Path(candidate).is_file():
            return candidate
    return None


def _ext(mime_type: str) -> str:
    m = (mime_type or "").lower()
    if "wav" in m:
        return "wav"
    if "ogg" in m:
        return "ogg"
    if "mp4" in m or "m4a" in m:
        return "mp4"
    if "mpeg" in m or "mp3" in m:
        return "mp3"
    return "webm"


async def ensure_wav(audio_bytes: bytes, mime_type: str = "audio/wav") -> tuple[bytes, str]:
    """Return (bytes, mime). Pass through if already wav; else optional ffmpeg.

    Preferred clients (desktop overlay + web AudioContext) already send 16 kHz
    mono WAV packed in-process — no ffmpeg needed.
    """
    if not audio_bytes:
        return audio_bytes, mime_type or "audio/wav"
    if "wav" in (mime_type or "").lower() or audio_bytes[:4] == b"RIFF":
        return audio_bytes, "audio/wav"

    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        logger.warning("non-wav input and ffmpeg missing; uploading original %s", mime_type)
        return audio_bytes, mime_type or "audio/webm"

    ext = _ext(mime_type)
    with tempfile.TemporaryDirectory(prefix="asr_") as td:
        src = Path(td) / f"in.{ext}"
        dst = Path(td) / "out.wav"
        src.write_bytes(audio_bytes)
        proc = await asyncio.create_subprocess_exec(
            ffmpeg,
            "-y",
            "-i",
            str(src),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-f",
            "wav",
            str(dst),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        )
        _, err = await proc.communicate()
        if proc.returncode != 0 or not dst.exists():
            msg = (err or b"").decode("utf-8", errors="ignore")[-400:]
            raise RuntimeError(f"ffmpeg 转码失败：{msg or proc.returncode}")
        return dst.read_bytes(), "audio/wav"
