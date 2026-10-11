"""Shared ASR provider interface and factory (SiliconFlow API-only)."""

from __future__ import annotations

import os
from typing import Optional, Protocol


class ASRProvider(Protocol):
    name: str

    def check_config(self) -> Optional[str]:
        """Chinese error if misconfigured, else None."""
        ...

    def set_language(self, language: Optional[str]) -> None:
        ...

    def reset(self) -> None:
        ...

    async def transcribe(self, audio_bytes: bytes, mime_type: str = "audio/webm") -> dict:
        ...


def resolve_provider_key() -> str:
    """
    Normalize ASR_PROVIDER / ASR_MODEL into:
      sensevoice | qwen-sf | mock
    """
    raw = (
        os.getenv("ASR_PROVIDER")
        or os.getenv("ASR_MODEL")
        or "qwen-sf"
    ).strip()
    key = raw.lower()
    aliases = {
        "sensevoice": "sensevoice",
        "sensevoicesmall": "sensevoice",
        "funaudiollm/sensevoicesmall": "sensevoice",
        "funaudiollm/sensevoice": "sensevoice",
        "qwen-sf": "qwen-sf",
        "qwen3": "qwen-sf",
        "qwen": "qwen-sf",
        "qwen3-asr": "qwen-sf",
        "qwen/qwen3-asr-1.7b": "qwen-sf",
        "qwen/qwen3-asr-1.7b-hf": "qwen-sf",
        "mock": "mock",
        "demo": "mock",
    }
    return aliases.get(key, "qwen-sf" if key not in ("sensevoice", "qwen-sf", "mock") else key)


def create_provider(provider_key: Optional[str] = None):
    from app.asr_mock import MockASR
    from app.asr_qwen import Qwen3ASR
    from app.asr_sensevoice import SenseVoiceASR

    key = (provider_key or resolve_provider_key()).lower()
    if key in ("sensevoice",):
        return SenseVoiceASR()
    if key in ("qwen-sf", "qwen3"):
        return Qwen3ASR()
    if key == "mock":
        return MockASR()
    return Qwen3ASR()


def provider_meta(key: str) -> dict:
    catalog = {
        "qwen-sf": {
            "id": "qwen-sf",
            "label": "Qwen3-ASR-1.7B",
            "hf_id": "Qwen/Qwen3-ASR-1.7B",
            "api": "SiliconFlow /v1/audio/transcriptions",
            "env": ["SILICONFLOW_API_KEY"],
            "wired": True,
        },
        "sensevoice": {
            "id": "sensevoice",
            "label": "SenseVoiceSmall",
            "hf_id": "FunAudioLLM/SenseVoiceSmall",
            "api": "SiliconFlow /v1/audio/transcriptions",
            "env": ["SILICONFLOW_API_KEY"],
            "wired": True,
        },
        "mock": {
            "id": "mock",
            "label": "Mock (smoke test)",
            "hf_id": None,
            "api": "local stub, no network",
            "env": [],
            "wired": True,
        },
    }
    # Map legacy qwen3 → qwen-sf meta
    if key == "qwen3":
        key = "qwen-sf"
    return catalog.get(key, catalog["qwen-sf"])
