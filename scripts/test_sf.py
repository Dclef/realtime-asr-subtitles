"""Smoke: SiliconFlow transcription with language=ja (+ optional JP wav)."""
from __future__ import annotations

import io
import os
import sys
import wave
from pathlib import Path

from dotenv import load_dotenv
import httpx

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
key = os.environ.get("SILICONFLOW_API_KEY", "")
print("key_set", bool(key), "len", len(key))
if not key:
    sys.exit(1)

wav_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
if wav_path and wav_path.is_file():
    wav = wav_path.read_bytes()
    print("wav", wav_path, len(wav))
else:
    rate = 16000
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * rate)
    wav = buf.getvalue()
    print("wav silence", len(wav))

url = "https://api.siliconflow.cn/v1/audio/transcriptions"
headers = {"Authorization": f"Bearer {key}"}
model = os.getenv("ASR_SMOKE_MODEL") or "Qwen/Qwen3-ASR-1.7B"
prompt = "请只用日语假名汉字转录，不要翻译成中文或英文。"
files = {
    "file": ("chunk.wav", wav, "audio/wav"),
    "model": (None, model),
    "language": (None, "ja"),
    "prompt": (None, prompt),
}
print("form fields: model=%s language=ja prompt=%r" % (model, prompt[:40]))
r = httpx.post(url, headers=headers, files=files, timeout=90)
print("status", r.status_code)
print(r.text[:1500])
