"""Qwen/Qwen3-ASR-1.7B via SiliconFlow OpenAI-compatible /audio/transcriptions."""

from __future__ import annotations

import io
import logging
import os
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
SILICONFLOW_DEFAULT = "https://api.siliconflow.cn/v1"

_LANGUAGE_PROMPTS = {
    "ja": "请只用日语假名汉字转录，不要翻译成中文或英文。",
    "zh": "请只用简体中文转录，不要翻译成日语或英文。",
    "en": "Transcribe in English only. Do not translate.",
}


class Qwen3ASR:
    name = "qwen-sf"

    def __init__(self) -> None:
        self.sf_key = os.getenv("SILICONFLOW_API_KEY", "").strip()
        self.sf_base = (
            os.getenv("SILICONFLOW_BASE_URL") or SILICONFLOW_DEFAULT
        ).rstrip("/")
        self.model = os.getenv("QWEN_ASR_MODEL") or MODEL_ID
        # Ignore legacy hosted flash name if present in old .env
        if self.model.strip().lower() in ("qwen3-asr-flash", "qwen3-asr"):
            self.model = MODEL_ID
        self.language: Optional[str] = None
        self._last_text = ""

    def check_config(self) -> Optional[str]:
        if not self.sf_key:
            return (
                "Qwen3-ASR 需要 SILICONFLOW_API_KEY。"
                "请在设置中填写硅基流动 API Key，或复制 .env.example 为 .env。"
            )
        return None

    def set_language(self, language: Optional[str]) -> None:
        self.language = None if language in (None, "", "auto") else language

    def reset(self) -> None:
        self._last_text = ""

    async def transcribe(self, audio_bytes: bytes, mime_type: str = "audio/wav") -> dict:
        err = self.check_config()
        if err:
            return {"ok": False, "error": err}
        if not audio_bytes or len(audio_bytes) < 100:
            return {"ok": True, "text": "", "is_final": False, "provider": self.name}

        if audio_bytes[:4] == b"RIFF" or "wav" in (mime_type or "").lower():
            mime_type = "audio/wav"
        else:
            return {
                "ok": False,
                "error": "请发送 WAV（PCM）。桌面 Overlay / 浏览器新版均直接发 WAV。",
            }

        url = f"{self.sf_base}/audio/transcriptions"
        headers = {"Authorization": f"Bearer {self.sf_key}"}
        files = {
            "file": ("chunk.wav", io.BytesIO(audio_bytes), "audio/wav"),
            "model": (None, self.model),
        }
        if self.language:
            files["language"] = (None, self.language)
            prompt = _LANGUAGE_PROMPTS.get(self.language)
            if prompt:
                files["prompt"] = (None, prompt)

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(url, headers=headers, files=files)
        except httpx.TimeoutException:
            return {"ok": False, "error": "SiliconFlow（Qwen3-ASR）请求超时。"}
        except httpx.RequestError as exc:
            return {"ok": False, "error": f"无法连接 SiliconFlow：{exc}"}

        if resp.status_code == 401:
            return {"ok": False, "error": "SILICONFLOW_API_KEY 无效或已过期。"}
        if resp.status_code >= 400:
            return {
                "ok": False,
                "error": f"SiliconFlow 错误 ({resp.status_code})：{resp.text[:300]}",
            }
        try:
            payload = resp.json()
        except Exception:
            return {"ok": False, "error": "SiliconFlow 返回无法解析的响应。"}

        text = (payload.get("text") or "").strip()
        if not text:
            return {"ok": True, "text": "", "is_final": False, "provider": self.name}
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
