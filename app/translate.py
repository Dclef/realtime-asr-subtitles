"""ASR → Simplified Chinese translation via SiliconFlow chat API."""

from __future__ import annotations

import logging
import os
import re
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

SF_DEFAULT = "https://api.siliconflow.cn/v1"
DEFAULT_SF_CHAT = "Qwen/Qwen3-8B"

_KANA = re.compile(r"[\u3040-\u30ff]")
_CJK = re.compile(r"[\u4e00-\u9fff]")
_LATIN = re.compile(r"[A-Za-z]")
_HANGUL = re.compile(r"[\uac00-\ud7af]")

_ZH_SYSTEM = (
    "你是专业字幕翻译。把用户给出的字幕行完整翻译成自然、通顺的简体中文。"
    "硬性规则：\n"
    "1. 只输出简体中文译文，一个字也不要解释、不加引号、不加前后缀。\n"
    "2. 禁止在译文中保留任何日文假名（平假名/片假名）、日文汉字原文、罗马字/romaji、"
    "英文单词或英日中混排；若原文是日语、英语或混合语，必须整句重写成纯中文。\n"
    "3. 专有名词也尽量用通行中文译名；实在无法意译时用常见中文音译，仍不得夹杂拉丁字母或假名。\n"
    "4. 不要输出双语对照、不要中英并写、不要中日并写。"
)

_ZH_RETRY_SYSTEM = (
    "上次输出不合格。请重新翻译成【纯简体中文】。"
    "绝对禁止：平假名、片假名、romaji、英文单词、中英/中日混排。"
    "只输出一行通顺的简体中文，不要任何其它字符语种。"
)


def detect_lang(text: str) -> str:
    """Return 'zh', 'ja', 'ko', 'en', or 'mixed' for short subtitle lines."""
    if not text or not text.strip():
        return "mixed"
    cjk = len(_CJK.findall(text))
    kana = len(_KANA.findall(text))
    hangul = len(_HANGUL.findall(text))
    latin = len(_LATIN.findall(text))
    if kana > 0:
        return "ja"
    if hangul > 0:
        return "ko"
    if cjk == 0 and latin == 0:
        return "mixed"
    if latin >= max(1, cjk) * 2:
        return "en"
    if cjk > 0 and latin == 0:
        return "zh"
    if cjk >= max(1, latin) * 0.6:
        return "zh"
    if latin > 0 and cjk == 0:
        return "en"
    return "mixed"


def has_japanese_kana(text: str) -> bool:
    return bool(_KANA.search(text or ""))


def is_mostly_latin(text: str) -> bool:
    if not text or not text.strip():
        return False
    cjk = len(_CJK.findall(text))
    latin = len(_LATIN.findall(text))
    if latin == 0:
        return False
    if cjk == 0 and latin >= 2:
        return True
    return latin >= max(1, cjk)


def is_pure_chinese_secondary(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    if has_japanese_kana(t):
        return False
    if is_mostly_latin(t):
        return False
    if _HANGUL.search(t):
        return False
    lang = detect_lang(t)
    if lang == "zh":
        return True
    if lang == "mixed" and len(_CJK.findall(t)) > 0 and not _LATIN.search(t):
        return True
    return False


class Translator:
    """Sync translator; returns empty string on failure (ASR still shown)."""

    def __init__(self) -> None:
        self.last_error: str = ""
        self._cache: dict[str, str] = {}
        self.reload_credentials()

    def reload_credentials(self) -> None:
        """Re-read key / base / model from environment (after settings save)."""
        self.sf_key = os.getenv("SILICONFLOW_API_KEY", "").strip()
        self.sf_base = (os.getenv("SILICONFLOW_BASE_URL") or SF_DEFAULT).rstrip("/")
        self.sf_model = os.getenv("TRANSLATE_MODEL") or DEFAULT_SF_CHAT

    def available(self) -> bool:
        return bool(self.sf_key)

    def backend_label(self) -> str:
        if self.sf_key:
            return f"SiliconFlow/{self.sf_model}"
        return "(no translate key)"

    def translate_pair(self, text: str) -> tuple[str, str]:
        """
        Return (primary, secondary).
        primary = original ASR; secondary = pure Simplified Chinese only.
        """
        text = (text or "").strip()
        if not text:
            return "", ""
        target = (os.getenv("TRANSLATE_TARGET") or "zh").strip().lower()
        target_zh = target in ("zh", "cn", "chinese", "zh-cn", "zh_cn", "中文")
        lang = detect_lang(text)
        if target_zh and lang == "zh":
            return text, ""
        if text in self._cache:
            return text, self._cache[text]
        if not self.available():
            logger.warning("translator unavailable: no API key")
            return text, ""

        if target in ("en", "english", "eng"):
            instruction = (
                "Translate the following subtitle line into natural English. "
                "Output ONLY the translation, no quotes, no explanation."
            )
            translated = self._chat(instruction, text)
            if not translated:
                translated = self._chat(
                    "You are a translator. Output only the translation.",
                    f"{instruction}\n\nText:\n{text}",
                )
            if translated and translated.strip().lower() == text.strip().lower():
                return text, ""
            if translated:
                self._cache[text] = translated
                self._trim_cache()
            return text, translated or ""

        translated = self._chat(_ZH_SYSTEM, text)
        if not translated:
            translated = self._chat(
                "你是字幕翻译。只输出纯简体中文译文。",
                f"{_ZH_SYSTEM}\n\n原文：\n{text}",
            )

        if translated:
            if translated.strip().lower() == text.strip().lower() and lang == "zh":
                return text, ""
            if not is_pure_chinese_secondary(translated):
                logger.warning(
                    "secondary not pure Chinese (%r -> %r); retry stricter",
                    text,
                    translated,
                )
                retry = self._chat(
                    _ZH_RETRY_SYSTEM,
                    f"原文：{text}\n上次不合格输出：{translated}\n请给出纯简体中文：",
                )
                if retry and is_pure_chinese_secondary(retry):
                    translated = retry
                else:
                    logger.warning(
                        "refuse mixed secondary after retry (%r -> %r / %r)",
                        text,
                        translated,
                        retry,
                    )
                    self.last_error = self.last_error or "secondary mixed language refused"
                    return text, ""
            self._cache[text] = translated
            self._trim_cache()
        else:
            logger.warning(
                "translate returned empty for: %s last_error=%s",
                text[:80],
                getattr(self, "last_error", ""),
            )
        return text, translated or ""

    def _trim_cache(self) -> None:
        if len(self._cache) > 200:
            for k in list(self._cache.keys())[:50]:
                self._cache.pop(k, None)

    def _chat(self, system_or_instruction: str, user_text: str) -> str:
        if not self.sf_key:
            return ""
        return self._post_chat(
            self.sf_base,
            self.sf_key,
            self.sf_model,
            system_or_instruction,
            user_text,
        )

    def _post_chat(
        self,
        base: str,
        api_key: str,
        model: str,
        instruction: str,
        user_text: str,
    ) -> str:
        url = f"{base}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": model,
            "messages": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": user_text},
            ],
            "temperature": 0.1,
            "max_tokens": 256,
        }
        if "Qwen3" in (model or "") or "qwen3" in (model or "").lower():
            body["enable_thinking"] = False
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(url, headers=headers, json=body)
        except Exception as exc:
            self.last_error = f"request failed: {exc}"
            logger.warning("translate request failed: %s", exc)
            return ""
        if resp.status_code >= 400:
            self.last_error = f"HTTP {resp.status_code}: {resp.text[:300]}"
            logger.warning("translate HTTP %s: %s", resp.status_code, resp.text[:300])
            return ""
        self.last_error = ""
        try:
            content = (
                resp.json()
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
        except Exception:
            return ""
        content = (content or "").strip()
        content = re.sub(r"<think>[\s\S]*?</think>", "", content, flags=re.I).strip()
        content = re.sub(r"^```\w*\n?|```$", "", content).strip()
        content = content.strip("\"'")
        return content
