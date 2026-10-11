"""Pack raw PCM samples into a 16-bit mono WAV in memory (no ffmpeg)."""

from __future__ import annotations

import io
import wave
from typing import Union

import numpy as np


def float32_to_wav_bytes(
    samples: Union[np.ndarray, bytes],
    sample_rate: int = 16000,
) -> bytes:
    """
    Convert float32 PCM in [-1, 1] (or raw int16 LE bytes) to a WAV file blob.
    Always outputs 16-bit mono PCM WAV at the given sample rate.
    """
    if isinstance(samples, (bytes, bytearray)):
        # Assume already int16 little-endian mono
        pcm = np.frombuffer(samples, dtype=np.int16)
    else:
        arr = np.asarray(samples, dtype=np.float32).reshape(-1)
        arr = np.clip(arr, -1.0, 1.0)
        pcm = (arr * 32767.0).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def resample_linear(samples: np.ndarray, src_rate: int, dst_rate: int = 16000) -> np.ndarray:
    """Simple linear resample float32 mono. No-op if rates match."""
    if src_rate == dst_rate or samples.size == 0:
        return samples.astype(np.float32, copy=False)
    duration = samples.size / float(src_rate)
    n_out = max(1, int(round(duration * dst_rate)))
    x_old = np.linspace(0.0, 1.0, num=samples.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, samples.astype(np.float32)).astype(np.float32)
