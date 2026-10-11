"""Local GUI/API settings (API key, base URL, translate model).

Persists to settings.json next to the project root. File is gitignored.
API keys are never logged.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

SF_DEFAULT_BASE = "https://api.siliconflow.cn/v1"
DEFAULT_TRANSLATE_MODEL = "Qwen/Qwen3-8B"

SETTINGS_FILENAME = "settings.json"

# Keys written to settings.json (and mirrored into os.environ).
FIELD_API_KEY = "api_key"
FIELD_BASE_URL = "base_url"
FIELD_TRANSLATE_MODEL = "translate_model"
FIELD_ASR_PROVIDER = "asr_provider"
FIELD_OVERLAY_OPACITY = "overlay_opacity"  # 0.15–1.0 panel mask alpha
FIELD_CHUNK_SECONDS = "chunk_seconds"  # ASR hop length (GUI slider)
FIELD_OVERLAP_SECONDS = "overlap_seconds"  # overlap; omit/None → auto ~18% of chunk
FIELD_LATENCY_PRESET = "latency_preset"  # realtime|accurate|precise|custom

# Default panel background alpha (0–255 scale used in paint; stored as 0.0–1.0)
DEFAULT_OVERLAY_OPACITY = 0.70
DEFAULT_CHUNK_SECONDS = 6.5
CHUNK_SEC_MIN = 1.5
CHUNK_SEC_MAX = 10.0
OVERLAP_RATIO = 0.18  # auto-derive overlap ≈ 15–20% of chunk


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def settings_path(root: Optional[Path] = None) -> Path:
    return (root or project_root()) / SETTINGS_FILENAME


def default_settings() -> dict[str, object]:
    return {
        FIELD_API_KEY: "",
        FIELD_BASE_URL: SF_DEFAULT_BASE,
        FIELD_TRANSLATE_MODEL: DEFAULT_TRANSLATE_MODEL,
        FIELD_ASR_PROVIDER: "qwen-sf",
        FIELD_OVERLAY_OPACITY: DEFAULT_OVERLAY_OPACITY,
        FIELD_CHUNK_SECONDS: DEFAULT_CHUNK_SECONDS,
        FIELD_OVERLAP_SECONDS: derive_overlap_seconds(DEFAULT_CHUNK_SECONDS),
        FIELD_LATENCY_PRESET: "accurate",
    }


def clamp_chunk_seconds(value: object) -> float:
    """Clamp ASR chunk hop to [1.5, 10.0] seconds (0.1s steps)."""
    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return DEFAULT_CHUNK_SECONDS
    return max(CHUNK_SEC_MIN, min(CHUNK_SEC_MAX, round(v, 1)))


def derive_overlap_seconds(chunk_sec: object) -> float:
    """Auto overlap ≈ 18% of chunk, clamped to a sensible range."""
    c = clamp_chunk_seconds(chunk_sec)
    ov = round(c * OVERLAP_RATIO, 1)
    return max(0.3, min(2.0, ov))


def clamp_overlap_seconds(value: object, chunk_sec: object | None = None) -> float:
    """Clamp overlap; if value missing/invalid, derive from chunk."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return derive_overlap_seconds(chunk_sec if chunk_sec is not None else DEFAULT_CHUNK_SECONDS)
    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return derive_overlap_seconds(chunk_sec if chunk_sec is not None else DEFAULT_CHUNK_SECONDS)
    return max(0.0, min(5.0, round(v, 1)))


def clamp_opacity(value: object) -> float:
    """Clamp overlay opacity to a usable range (panel still readable)."""
    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return DEFAULT_OVERLAY_OPACITY
    # Accept 0–100 as percent
    if v > 1.0:
        v = v / 100.0
    return max(0.15, min(1.0, v))


def load_settings(root: Optional[Path] = None) -> dict[str, object]:
    """Load settings.json; missing file → defaults. Never logs secrets."""
    path = settings_path(root)
    data = default_settings()
    if not path.is_file():
        return data
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("settings.json unreadable: %s", type(exc).__name__)
        return data
    if not isinstance(raw, dict):
        return data
    for key in data:
        val = raw.get(key)
        if val is None:
            continue
        if key == FIELD_OVERLAY_OPACITY:
            data[key] = clamp_opacity(val)
        elif key == FIELD_CHUNK_SECONDS:
            data[key] = clamp_chunk_seconds(val)
        elif key == FIELD_OVERLAP_SECONDS:
            data[key] = clamp_overlap_seconds(val, raw.get(FIELD_CHUNK_SECONDS))
        elif key == FIELD_API_KEY:
            data[key] = str(val)
        else:
            data[key] = str(val).strip()
    # Normalize base URL
    base = (data.get(FIELD_BASE_URL) or "").strip() or SF_DEFAULT_BASE
    data[FIELD_BASE_URL] = base.rstrip("/")
    if not data.get(FIELD_TRANSLATE_MODEL):
        data[FIELD_TRANSLATE_MODEL] = DEFAULT_TRANSLATE_MODEL
    if not data.get(FIELD_ASR_PROVIDER):
        data[FIELD_ASR_PROVIDER] = "qwen-sf"
    data[FIELD_OVERLAY_OPACITY] = clamp_opacity(data.get(FIELD_OVERLAY_OPACITY))
    data[FIELD_CHUNK_SECONDS] = clamp_chunk_seconds(data.get(FIELD_CHUNK_SECONDS))
    # If overlap absent in file, auto-derive from chunk
    if FIELD_OVERLAP_SECONDS not in raw or raw.get(FIELD_OVERLAP_SECONDS) is None:
        data[FIELD_OVERLAP_SECONDS] = derive_overlap_seconds(data[FIELD_CHUNK_SECONDS])
    else:
        data[FIELD_OVERLAP_SECONDS] = clamp_overlap_seconds(
            data.get(FIELD_OVERLAP_SECONDS), data[FIELD_CHUNK_SECONDS]
        )
    preset = str(data.get(FIELD_LATENCY_PRESET) or "accurate").strip().lower()
    if preset not in ("realtime", "accurate", "precise", "custom"):
        preset = "accurate"
    data[FIELD_LATENCY_PRESET] = preset
    return data


def save_settings(data: dict[str, Any], root: Optional[Path] = None) -> Path:
    """Write settings.json (creates parent). Does not log api_key."""
    path = settings_path(root)
    merged = default_settings()
    for key in merged:
        if key not in data or data[key] is None:
            continue
        if key == FIELD_OVERLAY_OPACITY:
            merged[key] = clamp_opacity(data[key])
        elif key == FIELD_CHUNK_SECONDS:
            merged[key] = clamp_chunk_seconds(data[key])
        elif key == FIELD_OVERLAP_SECONDS:
            merged[key] = clamp_overlap_seconds(
                data[key], data.get(FIELD_CHUNK_SECONDS, merged.get(FIELD_CHUNK_SECONDS))
            )
        elif key == FIELD_API_KEY:
            merged[key] = str(data[key])
        else:
            merged[key] = str(data[key]).strip()
    base = (merged.get(FIELD_BASE_URL) or "").strip() or SF_DEFAULT_BASE
    merged[FIELD_BASE_URL] = base.rstrip("/")
    merged[FIELD_OVERLAY_OPACITY] = clamp_opacity(merged.get(FIELD_OVERLAY_OPACITY))
    merged[FIELD_CHUNK_SECONDS] = clamp_chunk_seconds(merged.get(FIELD_CHUNK_SECONDS))
    merged[FIELD_OVERLAP_SECONDS] = clamp_overlap_seconds(
        merged.get(FIELD_OVERLAP_SECONDS), merged[FIELD_CHUNK_SECONDS]
    )
    preset = str(merged.get(FIELD_LATENCY_PRESET) or "accurate").strip().lower()
    if preset not in ("realtime", "accurate", "precise", "custom"):
        preset = "accurate"
    merged[FIELD_LATENCY_PRESET] = preset
    path.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "Saved settings to %s (api_key set=%s, base_url=%s, translate_model=%s, "
        "opacity=%.2f, chunk=%.1fs, overlap=%.1fs, preset=%s)",
        path.name,
        bool((merged.get(FIELD_API_KEY) or "").strip()),
        merged.get(FIELD_BASE_URL),
        merged.get(FIELD_TRANSLATE_MODEL),
        float(merged.get(FIELD_OVERLAY_OPACITY) or DEFAULT_OVERLAY_OPACITY),
        float(merged.get(FIELD_CHUNK_SECONDS) or DEFAULT_CHUNK_SECONDS),
        float(merged.get(FIELD_OVERLAP_SECONDS) or 0.0),
        merged.get(FIELD_LATENCY_PRESET),
    )
    return path


def update_settings(partial: dict[str, Any], root: Optional[Path] = None) -> dict[str, object]:
    """Load-merge-save so partial GUI updates do not wipe other keys."""
    data = load_settings(root)
    for key, val in partial.items():
        if val is not None:
            data[key] = val
    # If chunk changed without an explicit overlap, auto-derive (~18%).
    if FIELD_CHUNK_SECONDS in partial and FIELD_OVERLAP_SECONDS not in partial:
        data[FIELD_OVERLAP_SECONDS] = derive_overlap_seconds(data.get(FIELD_CHUNK_SECONDS))
    save_settings(data, root)
    return load_settings(root)


def apply_settings_to_env(root: Optional[Path] = None) -> dict[str, object]:
    """
    Load settings.json and push into os.environ for ASR/translate clients.

    Precedence: settings.json overrides empty .env values for these keys;
    if settings has a non-empty api_key / base_url / translate_model, it wins.
    If settings.json is missing, seed from existing env (e.g. .env) without writing.
    """
    data = load_settings(root)
    path = settings_path(root)

    # Seed from env if settings file absent or fields empty (one-way migrate).
    env_key = (os.getenv("SILICONFLOW_API_KEY") or "").strip()
    env_base = (os.getenv("SILICONFLOW_BASE_URL") or "").strip()
    env_tm = (os.getenv("TRANSLATE_MODEL") or "").strip()
    env_prov = (os.getenv("ASR_PROVIDER") or os.getenv("ASR_MODEL") or "").strip()

    changed = False
    if not (data.get(FIELD_API_KEY) or "").strip() and env_key:
        data[FIELD_API_KEY] = env_key
        changed = True
    if (not (data.get(FIELD_BASE_URL) or "").strip() or data[FIELD_BASE_URL] == SF_DEFAULT_BASE) and env_base:
        data[FIELD_BASE_URL] = env_base.rstrip("/")
        changed = True
    if not (data.get(FIELD_TRANSLATE_MODEL) or "").strip() and env_tm:
        data[FIELD_TRANSLATE_MODEL] = env_tm
        changed = True
    if env_prov and data.get(FIELD_ASR_PROVIDER) == "qwen-sf" and not path.is_file():
        # Only adopt env provider on first run (no settings file yet)
        data[FIELD_ASR_PROVIDER] = env_prov

    if changed and not path.is_file():
        # First-run migrate: write settings.json from .env so GUI sees the key
        # without committing .env. User can later edit via GUI.
        try:
            save_settings(data, root)
        except OSError as exc:
            logger.warning("Could not migrate settings.json: %s", type(exc).__name__)

    # settings.json wins when present / non-empty
    api_key = (data.get(FIELD_API_KEY) or "").strip()
    base_url = (data.get(FIELD_BASE_URL) or "").strip() or SF_DEFAULT_BASE
    translate_model = (data.get(FIELD_TRANSLATE_MODEL) or "").strip() or DEFAULT_TRANSLATE_MODEL
    asr_provider = (data.get(FIELD_ASR_PROVIDER) or "").strip() or "qwen-sf"

    if api_key:
        os.environ["SILICONFLOW_API_KEY"] = api_key
    if base_url:
        os.environ["SILICONFLOW_BASE_URL"] = base_url.rstrip("/")
    if translate_model:
        os.environ["TRANSLATE_MODEL"] = translate_model
    if asr_provider:
        os.environ["ASR_PROVIDER"] = asr_provider

    chunk = clamp_chunk_seconds(data.get(FIELD_CHUNK_SECONDS))
    overlap = clamp_overlap_seconds(data.get(FIELD_OVERLAP_SECONDS), chunk)
    data[FIELD_CHUNK_SECONDS] = chunk
    data[FIELD_OVERLAP_SECONDS] = overlap
    os.environ["ASR_CHUNK_SECONDS"] = f"{chunk:.1f}"
    os.environ["ASR_OVERLAP_SECONDS"] = f"{overlap:.1f}"
    preset = str(data.get(FIELD_LATENCY_PRESET) or "accurate").strip().lower()
    if preset in ("realtime", "accurate", "precise", "custom"):
        os.environ["ASR_LATENCY_PRESET"] = preset
        data[FIELD_LATENCY_PRESET] = preset

    # Opacity is GUI-only (not an API env); keep on the returned dict.
    data[FIELD_OVERLAY_OPACITY] = clamp_opacity(data.get(FIELD_OVERLAY_OPACITY))

    return data


def mask_key(key: str) -> str:
    """Show only last 4 chars for UI; empty → empty."""
    k = (key or "").strip()
    if not k:
        return ""
    if len(k) <= 4:
        return "****"
    return "…" + k[-4:]
