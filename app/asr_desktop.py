"""Sync SiliconFlow ASR for the desktop overlay (Qwen3-ASR + SenseVoiceSmall)."""

from __future__ import annotations

import io
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

SF_DEFAULT = "https://api.siliconflow.cn/v1"

_KANA = re.compile(r"[\u3040-\u30ff]")
_CJK = re.compile(r"[\u4e00-\u9fff]")
_LATIN = re.compile(r"[A-Za-z]")
_HANGUL = re.compile(r"[\uac00-\ud7af]")

# OpenAI-style prompt bias (SiliconFlow accepts the field).
_LANGUAGE_PROMPTS: dict[str, str] = {
    "ja": (
        "请只用日语假名汉字转录，不要翻译成中文或英文。"
        "Transcribe in Japanese kana/kanji only. Do not translate to Chinese or English."
    ),
    "zh": "请只用简体中文转录，不要翻译成日语或英文。",
    "en": "Transcribe in English only. Do not translate to Chinese or Japanese.",
}

_LANGUAGE_PROMPTS_STRICT: dict[str, str] = {
    "ja": (
        "【强制】只用日语假名与汉字原样转录语音内容。"
        "绝对禁止输出中文译文、英文译文或中英混排。"
        "language Japanese. Output Japanese transcription only."
    ),
    "zh": "【强制】只用简体中文转录。禁止日语假名或英文。language Chinese.",
    "en": "STRICT: English transcription only. No Chinese or Japanese. language English.",
}


@dataclass(frozen=True)
class AsrModelOption:
    key: str
    label: str
    model_id: str
    backend: str  # always "siliconflow"
    needs_env: str


ASR_CATALOG: list[AsrModelOption] = [
    AsrModelOption(
        "qwen-sf",
        "Qwen3-ASR-1.7B (硅基流动)",
        "Qwen/Qwen3-ASR-1.7B",
        "siliconflow",
        "SILICONFLOW_API_KEY",
    ),
    AsrModelOption(
        "sensevoice",
        "SenseVoiceSmall (硅基流动)",
        "FunAudioLLM/SenseVoiceSmall",
        "siliconflow",
        "SILICONFLOW_API_KEY",
    ),
]

_CATALOG_BY_KEY = {m.key: m for m in ASR_CATALOG}


def strip_sensevoice_tags(text: str) -> str:
    return re.sub(r"<\|[^|]*\|>", "", text or "").strip()


def _env(name: str) -> str:
    return os.getenv(name, "").strip()


def normalize_asr_language(raw: str | None) -> str | None:
    """Map UI/env language to API code zh|ja|en, or None for auto."""
    if raw is None:
        return None
    key = str(raw).strip().lower()
    if key in ("", "auto", "detect", "none"):
        return None
    aliases = {
        "zh": "zh",
        "cn": "zh",
        "chinese": "zh",
        "zh-cn": "zh",
        "zh_cn": "zh",
        "中文": "zh",
        "汉语": "zh",
        "ja": "ja",
        "jp": "ja",
        "japanese": "ja",
        "日语": "ja",
        "日本語": "ja",
        "en": "en",
        "eng": "en",
        "english": "en",
        "英语": "en",
    }
    return aliases.get(key, key if key in ("zh", "ja", "en") else None)


def normalize_detected_language(raw: object) -> str | None:
    """Normalize API response language (e.g. 'Japanese' / 'ja') to zh|ja|en."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() in ("none", "null", "unknown", "auto"):
        return None
    return normalize_asr_language(s)


def default_asr_language() -> str | None:
    """Default ASR lock from env; prefer Japanese for JP-video use case."""
    return normalize_asr_language(os.getenv("ASR_LANGUAGE") or "ja")


def text_looks_like_locked_language(text: str, locked: str) -> bool:
    """Heuristic: does transcript look consistent with the UI language lock?"""
    t = (text or "").strip()
    if not t:
        return True
    kana = bool(_KANA.search(t))
    cjk = len(_CJK.findall(t))
    latin = len(_LATIN.findall(t))
    hangul = bool(_HANGUL.search(t))
    if locked == "ja":
        if kana:
            return True
        if cjk > 0 and latin == 0 and not hangul:
            return False
        if latin >= max(1, cjk) * 2:
            return False
        return True
    if locked == "zh":
        if kana or hangul:
            return False
        if cjk > 0:
            return True
        if latin >= 2 and cjk == 0:
            return False
        return True
    if locked == "en":
        if kana or hangul:
            return False
        if latin == 0 and cjk > 0:
            return False
        return True
    return True


def available_models() -> list[AsrModelOption]:
    """Always expose the two SiliconFlow models (key checked at request time)."""
    return list(ASR_CATALOG)


def pick_default_model_key() -> str:
    """Prefer env ASR_PROVIDER / ASR_MODEL; else Qwen3-ASR then SenseVoice."""
    raw = (_env("ASR_PROVIDER") or _env("ASR_MODEL") or "").lower()
    aliases = {
        "sensevoice": "sensevoice",
        "sensevoicesmall": "sensevoice",
        "funaudiollm/sensevoicesmall": "sensevoice",
        "qwen-sf": "qwen-sf",
        "qwen": "qwen-sf",
        "qwen3": "qwen-sf",
        "qwen3-asr": "qwen-sf",
        "qwen/qwen3-asr-1.7b": "qwen-sf",
        "qwen/qwen3-asr-1.7b-hf": "qwen-sf",
    }
    if raw in aliases:
        return aliases[raw]
    return "qwen-sf"


def model_display_name(key: str) -> str:
    m = _CATALOG_BY_KEY.get(key)
    if not m:
        return key
    return f"{m.model_id}"


class MultiAsrClient:
    """Sync transcription client used by the desktop audio worker."""

    def __init__(self, model_key: Optional[str] = None) -> None:
        self._last = ""
        self.language: Optional[str] = default_asr_language()
        self._http = httpx.Client(timeout=httpx.Timeout(45.0, connect=10.0))
        self.reload_credentials()
        self.set_model(model_key or pick_default_model_key())
        logger.info(
            "MultiAsrClient init model=%s language=%s",
            self.model_id,
            self.language,
        )

    def reload_credentials(self) -> None:
        """Re-read API key / base URL from environment (after settings save)."""
        self.sf_key = _env("SILICONFLOW_API_KEY")
        self.sf_base = (_env("SILICONFLOW_BASE_URL") or SF_DEFAULT).rstrip("/")

    def set_model(self, key: str) -> None:
        if key not in _CATALOG_BY_KEY:
            key = pick_default_model_key()
        self.model_key = key
        self.option = _CATALOG_BY_KEY[key]
        self.model_id = self.option.model_id

    def set_language(self, language: Optional[str]) -> None:
        """Force ASR language (zh|ja|en) or None/auto for detection."""
        self.language = normalize_asr_language(language)
        logger.info("ASR language lock → %s", self.language)

    @property
    def language_label(self) -> str:
        labels = {"zh": "中文", "ja": "日语", "en": "英语"}
        if self.language in labels:
            return labels[self.language]
        return "自动"

    @property
    def label(self) -> str:
        return f"{self.option.label}"

    @property
    def short_name(self) -> str:
        return self.model_id

    def check(self) -> str | None:
        if not self.sf_key:
            return "缺少 API Key，请点击「设置」填写硅基流动密钥。"
        return None

    def reset(self) -> None:
        self._last = ""

    def _mismatch(self, text: str, detected: str | None) -> bool:
        locked = self.language
        if not locked or not text:
            return False
        if detected and detected != locked:
            return True
        return not text_looks_like_locked_language(text, locked)

    def transcribe_wav(self, wav_bytes: bytes) -> dict:
        err = self.check()
        if err:
            return {"ok": False, "error": err}
        if not wav_bytes or len(wav_bytes) < 44:
            return {"ok": True, "text": ""}

        result = self._via_siliconflow(wav_bytes, strict=False)

        if not result.get("ok"):
            return result
        text = strip_sensevoice_tags(result.get("text") or "")
        detected = result.get("detected_language")
        form_fields = result.get("form_fields")

        # SiliconFlow Qwen3-ASR often ignores language=; retry once with strict prompt.
        if text and self._mismatch(text, detected):
            logger.warning(
                "ASR language mismatch locked=%s detected=%s text=%r; retry strict prompt",
                self.language,
                detected,
                text[:80],
            )
            retry = self._via_siliconflow(wav_bytes, strict=True)
            if retry.get("ok"):
                text2 = strip_sensevoice_tags(retry.get("text") or "")
                det2 = retry.get("detected_language")
                form_fields = retry.get("form_fields") or form_fields
                if text2 and not self._mismatch(text2, det2):
                    text = text2
                    detected = det2
                    result = retry
                else:
                    bad = text2 or text
                    logger.warning(
                        "ASR mismatch after retry locked=%s detected=%s text=%r — reject",
                        self.language,
                        det2 or detected,
                        bad[:80],
                    )
                    return {
                        "ok": True,
                        "text": "",
                        "language_mismatch": True,
                        "rejected_text": bad,
                        "detected_language": det2 or detected,
                        "locked_language": self.language,
                        "form_fields": form_fields,
                        "model": self.model_id,
                    }

        if not text:
            out = {"ok": True, "text": "", "model": self.model_id}
            if form_fields:
                out["form_fields"] = form_fields
            if detected:
                out["detected_language"] = detected
            return out
        if text == self._last:
            return {
                "ok": True,
                "text": text,
                "duplicate": True,
                "model": self.model_id,
                "detected_language": detected,
                "form_fields": form_fields,
            }
        self._last = text
        return {
            "ok": True,
            "text": text,
            "model": self.model_id,
            "detected_language": detected,
            "form_fields": form_fields,
        }

    def _via_siliconflow(self, wav_bytes: bytes, *, strict: bool = False) -> dict:
        url = f"{self.sf_base}/audio/transcriptions"
        headers = {"Authorization": f"Bearer {self.sf_key}"}
        files: dict = {
            "file": ("chunk.wav", io.BytesIO(wav_bytes), "audio/wav"),
            "model": (None, self.model_id),
        }
        form_fields: dict[str, str] = {"model": self.model_id}

        if self.language:
            files["language"] = (None, self.language)
            form_fields["language"] = self.language
            prompts = _LANGUAGE_PROMPTS_STRICT if strict else _LANGUAGE_PROMPTS
            prompt = prompts.get(self.language)
            if prompt:
                files["prompt"] = (None, prompt)
                form_fields["prompt"] = prompt[:120]

        if self.language:
            assert files.get("language") == (
                None,
                self.language,
            ), f"missing language form field: {files.keys()}"

        logger.info(
            "SiliconFlow /audio/transcriptions form=%s strict=%s",
            form_fields,
            strict,
        )

        try:
            resp = self._http.post(url, headers=headers, files=files)
        except httpx.TimeoutException:
            return {"ok": False, "error": "请求超时"}
        except httpx.RequestError as exc:
            return {"ok": False, "error": f"网络错误：{exc}"}

        if resp.status_code == 401:
            return {"ok": False, "error": "API Key 无效"}
        if resp.status_code >= 400:
            return {
                "ok": False,
                "error": f"API 错误 ({resp.status_code})：{resp.text[:240]}",
            }
        try:
            payload = resp.json()
        except Exception:
            return {"ok": False, "error": "响应无法解析"}

        detected = normalize_detected_language(payload.get("language"))
        return {
            "ok": True,
            "text": payload.get("text") or "",
            "detected_language": detected,
            "raw_detected_language": payload.get("language"),
            "form_fields": form_fields,
        }
