"""Mock ASR for smoke tests without API keys."""

from __future__ import annotations

from typing import Optional


class MockASR:
    name = "mock"

    def __init__(self) -> None:
        self.language: Optional[str] = None
        self._n = 0
        self._last_text = ""

    def check_config(self) -> Optional[str]:
        return None

    def set_language(self, language: Optional[str]) -> None:
        self.language = None if language in (None, "", "auto") else language

    def reset(self) -> None:
        self._n = 0
        self._last_text = ""

    async def transcribe(self, audio_bytes: bytes, mime_type: str = "audio/webm") -> dict:
        if not audio_bytes or len(audio_bytes) < 50:
            return {"ok": True, "text": "", "is_final": False, "provider": self.name}
        self._n += 1
        lang = self.language or "auto"
        text = f"[模拟字幕 #{self._n}] 收到 {len(audio_bytes)} 字节音频（lang={lang}）"
        if text == self._last_text:
            return {
                "ok": True,
                "text": text,
                "is_final": False,
                "duplicate": True,
                "provider": self.name,
            }
        self._last_text = text
        return {"ok": True, "text": text, "is_final": True, "provider": self.name}
