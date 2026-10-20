#!/usr/bin/env python3
"""Desktop always-on-top subtitle overlay (PyQt5 → cloud ASR + bilingual).

Captures **system / loopback audio** (browser/video playback) by default, or
microphone. PCM → 16 kHz mono WAV in-process (no ffmpeg) → SiliconFlow ASR
(Qwen3-ASR / SenseVoiceSmall). Optional bilingual line → 简体中文.
"""

from __future__ import annotations

import io
import logging
import os
import queue
import re
import sys
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from dotenv import load_dotenv
from PyQt5.QtCore import Qt, QPoint, pyqtSignal, QObject, QTimer
from PyQt5.QtGui import QFont, QColor, QPainter, QBrush, QPen, QMouseEvent
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QSizePolicy,
    QComboBox,
    QMessageBox,
    QCheckBox,
    QSlider,
)

from app.asr_desktop import (
    MultiAsrClient,
    available_models,
    default_asr_language,
    normalize_asr_language,
    pick_default_model_key,
    strip_sensevoice_tags,
)
from app.settings_dialog import SettingsDialog
from app.settings_store import (
    DEFAULT_CHUNK_SECONDS,
    FIELD_CHUNK_SECONDS,
    FIELD_LATENCY_PRESET,
    FIELD_OVERLAP_SECONDS,
    FIELD_OVERLAY_OPACITY,
    apply_settings_to_env,
    clamp_chunk_seconds,
    clamp_opacity,
    clamp_overlap_seconds,
    derive_overlap_seconds,
    load_settings,
    update_settings,
)
from app.translate import Translator

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
apply_settings_to_env(BASE_DIR)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

SF_DEFAULT = "https://api.siliconflow.cn/v1"
TARGET_SR = 16000

# Preferred substrings for loopback / stereo-mix style devices (case-insensitive).
LOOPBACK_NAME_HINTS = (
    "loopback",
    "stereo mix",
    "立体声混音",
    "what u hear",
    "what you hear",
    "wave out mix",
    "cable output",
    "vb-audio",
    "vb-cable",
    "cable-a output",
    "cable-b output",
)

NO_LOOPBACK_HELP = (
    "未检测到系统声音回环设备，无法字幕浏览器/视频播放音频。\n\n"
    "插拔耳机后请先点「刷新设备」——程序会按当前默认播放设备"
    "（扬声器/耳机）重新枚举 WASAPI 回环。\n\n"
    "请任选其一：\n"
    "1) 确认耳机已成为 Windows 默认播放设备后点「刷新设备」；\n"
    "2) 启用「立体声混音」：右键任务栏喇叭 → 声音设置 → 更多声音设置 → "
    "「录制」选项卡 → 右键空白处勾选「显示禁用的设备」→ 启用「立体声混音」；\n"
    "3) 部分 USB/蓝牙耳机没有可用回环：可改回板载扬声器/3.5mm 耳机，"
    "或安装 VB-Audio Virtual Cable（https://vb-audio.com/Cable/）"
    "并将系统/浏览器输出设为 CABLE Input；\n"
    "4) 本程序优先用当前默认播放设备的 WASAPI Loopback（无需立体声混音）。"
    "若仍失败，请更新声卡驱动后重试。\n\n"
    "也可临时切换到「麦克风」模式听写人声。"
)


# Latency presets: longer chunks + overlap favor ASR accuracy (JP sentences).
# chunk_sec = new-audio hop; ASR window ≈ chunk_sec + overlap_sec.
# GUI slider (1.5–10s) is primary; presets snap the slider. Overlap ≈ 18% of chunk
# when using the slider (named presets keep their catalog overlap).
LATENCY_PRESETS: dict[str, dict] = {
    "realtime": {
        "label": "实时",
        "chunk_sec": 1.5,
        "overlap_sec": 0.5,
    },
    "accurate": {
        "label": "准确",
        "chunk_sec": 6.5,
        "overlap_sec": 1.2,
    },
    "precise": {
        "label": "更准",
        "chunk_sec": 8.0,
        "overlap_sec": 1.5,
    },
}


def _nearest_preset_key(chunk_sec: float) -> str:
    """Return named preset if chunk matches within 0.05s, else custom."""
    for key, p in LATENCY_PRESETS.items():
        if abs(float(p["chunk_sec"]) - chunk_sec) < 0.05:
            return key
    return "custom"


def _latency_preset_key() -> str:
    raw = (os.getenv("ASR_LATENCY_PRESET") or "").strip().lower()
    aliases = {
        "realtime": "realtime",
        "real-time": "realtime",
        "rt": "realtime",
        "live": "realtime",
        "实时": "realtime",
        "accurate": "accurate",
        "quality": "accurate",
        "准确": "accurate",
        "precise": "precise",
        "max": "precise",
        "best": "precise",
        "更准": "precise",
        "custom": "custom",
        "自定义": "custom",
    }
    if raw in aliases:
        return aliases[raw]
    # Legacy: ASR_CHUNK_SECONDS length picks quality tier
    try:
        env_chunk = float(os.getenv("ASR_CHUNK_SECONDS") or "")
    except ValueError:
        env_chunk = None
    if env_chunk is not None:
        return _nearest_preset_key(env_chunk)
    # Default: accurate (6.5s+overlap) — prefer quality over realtime lag
    return "accurate"


def _preset_defaults(key: str | None = None) -> tuple[str, float, float]:
    """Resolve (preset_key, chunk_sec, overlap_sec) from settings / env / catalog."""
    cfg = load_settings(BASE_DIR)
    cfg_chunk = cfg.get(FIELD_CHUNK_SECONDS)
    cfg_ov = cfg.get(FIELD_OVERLAP_SECONDS)
    cfg_preset = str(cfg.get(FIELD_LATENCY_PRESET) or "").strip().lower()

    # Explicit named preset from UI → catalog values
    if key in LATENCY_PRESETS:
        p = LATENCY_PRESETS[key]
        return key, float(p["chunk_sec"]), float(p["overlap_sec"])

    # Startup / custom: prefer settings.json slider values
    if cfg_chunk is not None:
        chunk = clamp_chunk_seconds(cfg_chunk)
    else:
        try:
            env_chunk = os.getenv("ASR_CHUNK_SECONDS")
            chunk = (
                clamp_chunk_seconds(float(env_chunk))
                if env_chunk is not None and str(env_chunk).strip() != ""
                else DEFAULT_CHUNK_SECONDS
            )
        except ValueError:
            chunk = DEFAULT_CHUNK_SECONDS

    if key == "custom" or cfg_preset == "custom":
        k = "custom"
    else:
        k = _nearest_preset_key(chunk)
        if cfg_preset in LATENCY_PRESETS and abs(
            chunk - float(LATENCY_PRESETS[cfg_preset]["chunk_sec"])
        ) < 0.05:
            k = cfg_preset

    if k in LATENCY_PRESETS and abs(chunk - float(LATENCY_PRESETS[k]["chunk_sec"])) < 0.05:
        overlap = float(LATENCY_PRESETS[k]["overlap_sec"])
    else:
        k = "custom"
        if cfg_ov is not None:
            overlap = clamp_overlap_seconds(cfg_ov, chunk)
        else:
            try:
                env_ov = os.getenv("ASR_OVERLAP_SECONDS")
                overlap = (
                    clamp_overlap_seconds(float(env_ov), chunk)
                    if env_ov is not None and str(env_ov).strip() != ""
                    else derive_overlap_seconds(chunk)
                )
            except ValueError:
                overlap = derive_overlap_seconds(chunk)
    return k, clamp_chunk_seconds(chunk), float(overlap)


def _chunk_seconds() -> float:
    return _preset_defaults()[1]


def _bilingual_default() -> bool:
    raw = (os.getenv("BILINGUAL") or os.getenv("ASR_BILINGUAL") or "1").strip().lower()
    return raw not in ("0", "false", "off", "no")


def _asr_language_default() -> str:
    """UI default: Japanese (ja) for JP videos; env ASR_LANGUAGE overrides."""
    return normalize_asr_language(os.getenv("ASR_LANGUAGE") or "ja") or "ja"


def float32_to_wav_bytes(samples: np.ndarray, sample_rate: int = TARGET_SR) -> bytes:
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


def resample_linear(samples: np.ndarray, src_rate: int, dst_rate: int = TARGET_SR) -> np.ndarray:
    if src_rate == dst_rate or samples.size == 0:
        return samples.astype(np.float32, copy=False)
    n_out = max(1, int(round(samples.size * float(dst_rate) / float(src_rate))))
    x_old = np.linspace(0.0, 1.0, num=samples.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, samples.astype(np.float32)).astype(np.float32)


def _name_looks_loopback(name: str) -> bool:
    low = (name or "").lower()
    return any(h in low for h in LOOPBACK_NAME_HINTS)


def _strip_loopback_suffix(name: str) -> str:
    """'Speakers (Realtek) [Loopback]' → 'Speakers (Realtek)'."""
    n = (name or "").strip()
    for suf in (" [Loopback]", " (Loopback)", " [loopback]"):
        if n.endswith(suf):
            return n[: -len(suf)].strip()
    return n


def _names_match_output(loopback_name: str, output_name: str) -> bool:
    if not loopback_name or not output_name:
        return False
    lb = _strip_loopback_suffix(loopback_name)
    out = (output_name or "").strip()
    return lb == out or out in lb or lb in out


# ---------------------------------------------------------------------------
# Audio device discovery
# ---------------------------------------------------------------------------


@dataclass
class AudioDevice:
    """One selectable capture endpoint."""

    key: str  # stable id for combo
    label: str  # UI text
    kind: str  # "loopback" | "mic"
    backend: str  # "pyaudiowpatch" | "soundcard" | "sounddevice"
    backend_ref: object  # backend-specific handle / index
    device_name: str = ""  # raw endpoint name (for re-resolve after plug)


def current_default_playback_name() -> str:
    """Best-effort name of the current default WASAPI / system playback device."""
    try:
        import pyaudiowpatch as pyaudio

        p = pyaudio.PyAudio()
        try:
            try:
                info = p.get_default_wasapi_device(d_out=True)
                return (info.get("name") or "").strip()
            except Exception:
                wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
                di = p.get_device_info_by_index(wasapi["defaultOutputDevice"])
                return (di.get("name") or "").strip()
        finally:
            p.terminate()
    except Exception:
        pass
    try:
        import soundcard as sc

        return (sc.default_speaker().name or "").strip()
    except Exception:
        return ""


def _add_loopback(
    loopbacks: list[AudioDevice],
    seen_lb: set[str],
    *,
    key: str,
    name: str,
    label: str,
    backend: str,
    backend_ref: object,
    default_out_name: str,
) -> None:
    if key in seen_lb:
        return
    # Deduplicate by stripped endpoint name across backends
    bare = _strip_loopback_suffix(name).lower()
    if bare and any(_strip_loopback_suffix(d.device_name).lower() == bare for d in loopbacks):
        return
    seen_lb.add(key)
    score_prefix = "★ " if _names_match_output(name, default_out_name) else ""
    loopbacks.append(
        AudioDevice(
            key=key,
            label=f"{score_prefix}{label}",
            kind="loopback",
            backend=backend,
            backend_ref=backend_ref,
            device_name=name,
        )
    )


def list_audio_devices() -> tuple[list[AudioDevice], list[AudioDevice]]:
    """Return (loopback_devices, mic_devices).

    Prefer WASAPI loopback of the *current default playback device* (speakers or
    headphones). Re-enumerate every call so「刷新设备」/plug-in picks up changes.
    """
    loopbacks: list[AudioDevice] = []
    mics: list[AudioDevice] = []
    seen_lb: set[str] = set()
    seen_mic: set[str] = set()
    log = logging.getLogger("overlay.audio")

    # --- 1) pyaudiowpatch WASAPI loopback (best on Windows) ---
    try:
        import pyaudiowpatch as pyaudio

        p = pyaudio.PyAudio()
        try:
            wasapi = None
            try:
                wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
            except Exception as exc:
                log.warning("WASAPI host unavailable: %s", exc)
                wasapi = None

            default_out_name = ""
            default_out_index: int | None = None
            if wasapi is not None:
                try:
                    di = p.get_default_wasapi_device(d_out=True)
                    default_out_name = (di.get("name") or "").strip()
                    default_out_index = int(di["index"])
                except Exception:
                    try:
                        default_out_index = int(wasapi["defaultOutputDevice"])
                        di = p.get_device_info_by_index(default_out_index)
                        default_out_name = (di.get("name") or "").strip()
                    except Exception as exc:
                        log.warning("default WASAPI output: %s", exc)

            # Official helpers: default loopback + all loopback endpoints
            try:
                for info in p.get_loopback_device_info_generator():
                    idx = int(info["index"])
                    name = info.get("name") or f"loopback-{idx}"
                    _add_loopback(
                        loopbacks,
                        seen_lb,
                        key=f"paw-lb-{idx}",
                        name=name,
                        label=f"系统 · {name} (WASAPI)",
                        backend="pyaudiowpatch",
                        backend_ref=idx,
                        default_out_name=default_out_name,
                    )
            except Exception as exc:
                log.warning("loopback generator: %s", exc)

            try:
                info = p.get_default_wasapi_loopback()
                idx = int(info["index"])
                name = info.get("name") or f"loopback-{idx}"
                _add_loopback(
                    loopbacks,
                    seen_lb,
                    key=f"paw-lb-{idx}",
                    name=name,
                    label=f"系统 · {name} (WASAPI)",
                    backend="pyaudiowpatch",
                    backend_ref=idx,
                    default_out_name=default_out_name or _strip_loopback_suffix(name),
                )
            except Exception as exc:
                log.info("get_default_wasapi_loopback: %s", exc)

            # Walk EVERY WASAPI render endpoint → analogue loopback
            # (covers headphones / USB / secondary speakers after plug-in)
            if wasapi is not None:
                wasapi_index = int(wasapi["index"])
                for i in range(p.get_device_count()):
                    try:
                        info = p.get_device_info_by_index(i)
                    except Exception:
                        continue
                    if int(info.get("hostApi", -1)) != wasapi_index:
                        continue
                    if info.get("isLoopbackDevice"):
                        continue
                    if int(info.get("maxOutputChannels") or 0) <= 0:
                        continue
                    try:
                        lb = p.get_wasapi_loopback_analogue_by_dict(info)
                    except Exception:
                        try:
                            lb = p.get_wasapi_loopback_analogue_by_index(i)
                        except Exception:
                            continue
                    idx = int(lb["index"])
                    name = lb.get("name") or f"loopback-{idx}"
                    _add_loopback(
                        loopbacks,
                        seen_lb,
                        key=f"paw-lb-{idx}",
                        name=name,
                        label=f"系统 · {name} (WASAPI)",
                        backend="pyaudiowpatch",
                        backend_ref=idx,
                        default_out_name=default_out_name,
                    )

            # Fallback scan: any device flagged isLoopbackDevice
            for i in range(p.get_device_count()):
                try:
                    info = p.get_device_info_by_index(i)
                except Exception:
                    continue
                name = info.get("name") or f"device-{i}"
                if info.get("isLoopbackDevice"):
                    idx = int(info.get("index", i))
                    _add_loopback(
                        loopbacks,
                        seen_lb,
                        key=f"paw-lb-{idx}",
                        name=name,
                        label=f"系统 · {name} (WASAPI)",
                        backend="pyaudiowpatch",
                        backend_ref=idx,
                        default_out_name=default_out_name,
                    )
                elif int(info.get("maxInputChannels") or 0) > 0 and not info.get(
                    "isLoopbackDevice"
                ):
                    if "mapper" in name.lower() or "主声音捕获" in name:
                        continue
                    if name in seen_mic:
                        continue
                    seen_mic.add(name)
                    mics.append(
                        AudioDevice(
                            key=f"paw-mic-{i}",
                            label=f"麦克风 · {name}",
                            kind="mic",
                            backend="pyaudiowpatch",
                            backend_ref=int(info["index"]),
                            device_name=name,
                        )
                    )

            # Ensure the ★ marker lands on current default even if name matching
            # was fuzzy (e.g. Realtek speakers vs headphones share branding).
            if default_out_name and loopbacks:
                for d in loopbacks:
                    if d.backend != "pyaudiowpatch":
                        continue
                    if _names_match_output(d.device_name, default_out_name):
                        if not d.label.startswith("★ "):
                            d.label = "★ " + d.label
                    elif d.label.startswith("★ ") and not _names_match_output(
                        d.device_name, default_out_name
                    ):
                        d.label = d.label[2:]

            if default_out_index is not None and not any(
                d.label.startswith("★ ") for d in loopbacks
            ):
                try:
                    lb = p.get_wasapi_loopback_analogue_by_index(default_out_index)
                    idx = int(lb["index"])
                    name = lb.get("name") or f"loopback-{idx}"
                    _add_loopback(
                        loopbacks,
                        seen_lb,
                        key=f"paw-lb-{idx}",
                        name=name,
                        label=f"系统 · {name} (WASAPI)",
                        backend="pyaudiowpatch",
                        backend_ref=idx,
                        default_out_name=name,  # force ★
                    )
                except Exception as exc:
                    log.info("analogue for default out idx %s: %s", default_out_index, exc)
        finally:
            p.terminate()
    except Exception as exc:
        log.warning("pyaudiowpatch enumerate failed: %s", exc)

    # --- 2) soundcard: loopback for default + ALL speakers ---
    try:
        import soundcard as sc

        default_sp_name = ""
        try:
            default_sp_name = sc.default_speaker().name or ""
        except Exception:
            pass

        speakers = []
        try:
            speakers = list(sc.all_speakers())
        except Exception as exc:
            log.warning("soundcard all_speakers: %s", exc)

        for sp in speakers:
            try:
                lb = sc.get_microphone(id=str(sp.name), include_loopback=True)
            except Exception:
                continue
            if not getattr(lb, "isloopback", True):
                continue
            name = getattr(lb, "name", None) or sp.name
            _add_loopback(
                loopbacks,
                seen_lb,
                key=f"sc-lb-{sp.name}",
                name=name,
                label=f"系统 · {name} (soundcard loopback)",
                backend="soundcard",
                backend_ref=lb,
                default_out_name=default_sp_name,
            )

        for mic in sc.all_microphones(include_loopback=True):
            name = mic.name
            if getattr(mic, "isloopback", False) or _name_looks_loopback(name):
                _add_loopback(
                    loopbacks,
                    seen_lb,
                    key=f"sc-lb-{name}",
                    name=name,
                    label=f"系统 · {name} (soundcard)",
                    backend="soundcard",
                    backend_ref=mic,
                    default_out_name=default_sp_name,
                )
            else:
                if name in seen_mic:
                    continue
                seen_mic.add(name)
                mics.append(
                    AudioDevice(
                        key=f"sc-mic-{name}",
                        label=f"麦克风 · {name}",
                        kind="mic",
                        backend="soundcard",
                        backend_ref=mic,
                        device_name=name,
                    )
                )
    except Exception as exc:
        log.warning("soundcard enumerate failed: %s", exc)

    # --- 3) sounddevice named Stereo Mix / Cable / etc. ---
    try:
        import sounddevice as sd

        hostapis = sd.query_hostapis()
        for i, d in enumerate(sd.query_devices()):
            name = d.get("name") or f"dev-{i}"
            max_in = int(d.get("max_input_channels") or 0)
            if max_in <= 0:
                continue
            host_name = ""
            try:
                host_name = hostapis[int(d["hostapi"])]["name"]
            except Exception:
                pass
            if "WDM-KS" in host_name:
                continue
            if _name_looks_loopback(name):
                _add_loopback(
                    loopbacks,
                    seen_lb,
                    key=f"sd-lb-{i}",
                    name=name,
                    label=f"系统 · {name} ({host_name or 'sounddevice'})",
                    backend="sounddevice",
                    backend_ref=i,
                    default_out_name="",
                )
            else:
                if "mapper" in name.lower() or "主声音捕获" in name:
                    continue
                if name in seen_mic:
                    continue
                if not mics or "WASAPI" in host_name or host_name.startswith("MME"):
                    seen_mic.add(name)
                    mics.append(
                        AudioDevice(
                            key=f"sd-mic-{i}",
                            label=f"麦克风 · {name} ({host_name})",
                            kind="mic",
                            backend="sounddevice",
                            backend_ref=i,
                            device_name=name,
                        )
                    )
    except Exception as exc:
        log.warning("sounddevice enumerate failed: %s", exc)

    loopbacks.sort(key=lambda d: (0 if d.label.startswith("★") else 1, d.label))
    log.info(
        "enumerated %d loopback(s), %d mic(s); default_playback=%r",
        len(loopbacks),
        len(mics),
        current_default_playback_name(),
    )
    return loopbacks, mics


def pick_default_loopback(loopbacks: list[AudioDevice]) -> Optional[AudioDevice]:
    """Prefer ★ (current default playback) WASAPI loopback."""
    if not loopbacks:
        return None
    for d in loopbacks:
        if d.label.startswith("★"):
            return d
    # Re-check against live default name (in case ★ marking missed)
    default_name = current_default_playback_name()
    if default_name:
        for d in loopbacks:
            if _names_match_output(d.device_name, default_name):
                return d
    for d in loopbacks:
        if d.backend == "pyaudiowpatch":
            return d
    return loopbacks[0]


def resolve_pyaudiowpatch_loopback_index(device: AudioDevice) -> int:
    """Re-resolve WASAPI loopback index by name (indices shift after plug-in)."""
    import pyaudiowpatch as pyaudio

    preferred = int(device.backend_ref)
    want = _strip_loopback_suffix(device.device_name)
    p = pyaudio.PyAudio()
    try:
        # Prefer current default playback's loopback when this device was ★ / default
        if device.label.startswith("★") or not want:
            try:
                info = p.get_default_wasapi_loopback()
                return int(info["index"])
            except Exception:
                pass
        if want:
            for info in p.get_loopback_device_info_generator():
                if _names_match_output(info.get("name") or "", want):
                    return int(info["index"])
            # Match against current outputs' analogues
            try:
                wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
                for i in range(p.get_device_count()):
                    info = p.get_device_info_by_index(i)
                    if int(info.get("hostApi", -1)) != int(wasapi["index"]):
                        continue
                    if int(info.get("maxOutputChannels") or 0) <= 0:
                        continue
                    if info.get("isLoopbackDevice"):
                        continue
                    if not _names_match_output(
                        (info.get("name") or "") + " [Loopback]", want
                    ) and not _names_match_output(info.get("name") or "", want):
                        continue
                    try:
                        lb = p.get_wasapi_loopback_analogue_by_dict(info)
                        return int(lb["index"])
                    except Exception:
                        continue
            except Exception:
                pass
        # Last resort: stored index if still a loopback
        try:
            info = p.get_device_info_by_index(preferred)
            if info.get("isLoopbackDevice"):
                return preferred
        except Exception:
            pass
        raise RuntimeError(
            f"找不到回环设备「{device.device_name or preferred}」。"
            "请点「刷新设备」后重试（插拔耳机后索引会变化）。"
        )
    finally:
        p.terminate()


# ---------------------------------------------------------------------------
# Capture backends → float32 mono blocks into a queue
# ---------------------------------------------------------------------------


class CaptureSession:
    """Open one device and push float32 mono numpy chunks into audio_q."""

    def __init__(
        self,
        device: AudioDevice,
        audio_q: queue.Queue,
        stop_event: threading.Event,
    ) -> None:
        self.device = device
        self.audio_q = audio_q
        self.stop_event = stop_event
        self.sample_rate = TARGET_SR
        self._closer: Optional[Callable[[], None]] = None

    def start(self) -> int:
        """Start capture; return used sample rate. Raises on failure."""
        if self.device.backend == "pyaudiowpatch":
            return self._start_pyaudiowpatch()
        if self.device.backend == "soundcard":
            return self._start_soundcard()
        if self.device.backend == "sounddevice":
            return self._start_sounddevice()
        raise RuntimeError(f"未知后端：{self.device.backend}")

    def close(self) -> None:
        if self._closer:
            try:
                self._closer()
            except Exception:
                pass
            self._closer = None

    def _start_pyaudiowpatch(self) -> int:
        import pyaudiowpatch as pyaudio

        # Re-resolve index: plugging headphones shifts PortAudio device indices.
        if self.device.kind == "loopback":
            idx = resolve_pyaudiowpatch_loopback_index(self.device)
        else:
            idx = int(self.device.backend_ref)
        p = pyaudio.PyAudio()
        info = p.get_device_info_by_index(idx)
        sr = int(info.get("defaultSampleRate") or 48000)
        ch = max(1, int(info.get("maxInputChannels") or 2))
        # Prefer stereo for loopback, then downmix
        if self.device.kind == "loopback":
            ch = min(ch, 2) if ch >= 2 else ch
        else:
            ch = 1 if ch >= 1 else ch
        frames_per_buffer = 2048

        def callback(in_data, frame_count, time_info, status):  # noqa: ANN001
            if self.stop_event.is_set():
                return (None, pyaudio.paComplete)
            arr = np.frombuffer(in_data, dtype=np.float32)
            if ch > 1:
                arr = arr.reshape(-1, ch).mean(axis=1)
            else:
                arr = arr.reshape(-1)
            self.audio_q.put(arr.copy())
            return (None, pyaudio.paContinue)

        stream = p.open(
            format=pyaudio.paFloat32,
            channels=ch,
            rate=sr,
            input=True,
            input_device_index=idx,
            frames_per_buffer=frames_per_buffer,
            stream_callback=callback,
        )
        stream.start_stream()
        self.sample_rate = sr

        def _close() -> None:
            try:
                stream.stop_stream()
                stream.close()
            except Exception:
                pass
            try:
                p.terminate()
            except Exception:
                pass

        self._closer = _close
        return sr

    def _start_soundcard(self) -> int:
        mic = self.device.backend_ref
        sr = 48000
        ch = 2 if self.device.kind == "loopback" else 1
        recorder = mic.recorder(samplerate=sr, channels=ch)
        recorder.__enter__()
        self.sample_rate = sr
        stop = self.stop_event
        audio_q = self.audio_q

        def _reader() -> None:
            block = 2048
            try:
                while not stop.is_set():
                    data = recorder.record(numframes=block)
                    if data.ndim > 1:
                        mono = data.mean(axis=1).astype(np.float32)
                    else:
                        mono = data.astype(np.float32)
                    audio_q.put(mono)
            except Exception:
                pass

        t = threading.Thread(target=_reader, name="sc-reader", daemon=True)
        t.start()

        def _close() -> None:
            stop.set()
            try:
                recorder.__exit__(None, None, None)
            except Exception:
                pass

        self._closer = _close
        return sr

    def _start_sounddevice(self) -> int:
        import sounddevice as sd

        idx = int(self.device.backend_ref)
        info = sd.query_devices(idx)
        default_sr = int(info.get("default_samplerate") or 48000)
        max_in = int(info.get("max_input_channels") or 1)
        channels = 2 if (self.device.kind == "loopback" and max_in >= 2) else 1
        try_rates = [TARGET_SR, default_sr, 48000, 44100]
        stream = None
        used_sr = default_sr
        last_err: Exception | None = None
        audio_q = self.audio_q
        stop = self.stop_event

        def callback(indata, frames, time_info, status):  # noqa: ANN001
            if stop.is_set():
                return
            if indata.shape[1] > 1:
                mono = indata.mean(axis=1).astype(np.float32)
            else:
                mono = indata[:, 0].copy()
            audio_q.put(mono)

        for rate in try_rates:
            try:
                stream = sd.InputStream(
                    device=idx,
                    samplerate=rate,
                    channels=channels,
                    dtype="float32",
                    blocksize=0,
                    callback=callback,
                )
                stream.start()
                used_sr = rate
                break
            except Exception as exc:
                last_err = exc
                stream = None
        if stream is None:
            raise RuntimeError(f"无法打开设备：{last_err}")

        self.sample_rate = used_sr

        def _close() -> None:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass

        self._closer = _close
        return used_sr


# ---------------------------------------------------------------------------
# ASR + translation wired in MultiAsrClient / Translator (app/)
# ---------------------------------------------------------------------------



class Bridge(QObject):
    # original, translation ("" for partials), is_final
    transcript = pyqtSignal(str, str, bool)
    status = pyqtSignal(str)
    error = pyqtSignal(str)


class AudioWorker:
    """Background capture (loopback or mic) + chunked ASR (+ optional translate)."""

    def __init__(
        self,
        bridge: Bridge,
        client: MultiAsrClient,
        translator: Translator,
    ) -> None:
        self.bridge = bridge
        self.client = client
        self.translator = translator
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._audio_q: queue.Queue[np.ndarray] = queue.Queue()
        preset, chunk, overlap = _preset_defaults()
        self.latency_preset = preset
        self.chunk_sec = chunk
        self.overlap_sec = overlap
        self.bilingual = _bilingual_default()
        self.device: Optional[AudioDevice] = None

    def start(self, device: AudioDevice) -> None:
        if self._thread and self._thread.is_alive():
            return
        self.device = device
        self._stop.clear()
        self.client.reset()
        # drain queue
        while not self._audio_q.empty():
            try:
                self._audio_q.get_nowait()
            except queue.Empty:
                break
        self._thread = threading.Thread(target=self._run, name="audio-asr", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def apply_preset(self, key: str) -> None:
        """Apply 「实时」/「准确」/「更准」hop + overlap (takes effect on next start)."""
        if key in LATENCY_PRESETS:
            k = key
            chunk = float(LATENCY_PRESETS[key]["chunk_sec"])
            overlap = float(LATENCY_PRESETS[key]["overlap_sec"])
        else:
            k, chunk, overlap = _preset_defaults(key)
        self.latency_preset = k
        self.chunk_sec = clamp_chunk_seconds(chunk)
        self.overlap_sec = clamp_overlap_seconds(overlap, self.chunk_sec)

    def apply_chunk(self, chunk_sec: float, *, persist_preset: str | None = None) -> None:
        """Apply slider hop; auto-derive overlap (~18%). Takes effect on next start."""
        chunk = clamp_chunk_seconds(chunk_sec)
        overlap = derive_overlap_seconds(chunk)
        if persist_preset is not None:
            k = persist_preset
        else:
            k = _nearest_preset_key(chunk)
            if k in LATENCY_PRESETS:
                # Snap to catalog overlap when landing exactly on a preset
                overlap = float(LATENCY_PRESETS[k]["overlap_sec"])
            else:
                k = "custom"
        self.latency_preset = k
        self.chunk_sec = chunk
        self.overlap_sec = overlap

    def _finalize_pending(self, pending: str | None, mode: str) -> None:
        if not pending:
            return
        secondary = ""
        if self.bilingual:
            self.bridge.status.emit("翻译中…")
            _, secondary = self.translator.translate_pair(pending)
            if not secondary:
                err = getattr(self.translator, "last_error", "") or ""
                # Already-Chinese ASR leaves secondary empty on purpose.
                from app.translate import detect_lang

                if detect_lang(pending) != "zh":
                    tip = f"翻译失败{('：' + err) if err else ''}"
                    self.bridge.status.emit(tip)
        self.bridge.transcript.emit(pending, secondary or "", True)
        if secondary or not self.bilingual:
            self.bridge.status.emit(f"录音中 · {mode} · {self.client.short_name}")

    def _run(self) -> None:
        assert self.device is not None
        session = CaptureSession(self.device, self._audio_q, self._stop)
        try:
            used_sr = session.start()
        except Exception as exc:
            kind = "系统声音" if self.device.kind == "loopback" else "麦克风"
            self.bridge.error.emit(f"无法打开{kind}：{exc}")
            return

        mode = "系统回环" if self.device.kind == "loopback" else "麦克风"
        preset_label = LATENCY_PRESETS.get(self.latency_preset, {}).get(
            "label", "自定义" if self.latency_preset == "custom" else self.latency_preset
        )
        self.bridge.status.emit(
            f"录音中 · {mode} · {preset_label} · ASR {self.client.short_name} · "
            f"块{self.chunk_sec:.1f}s/叠{self.overlap_sec:.1f}s · "
            f"{used_sr}→{TARGET_SR} Hz"
        )
        buf: list[np.ndarray] = []
        buf_samples = 0
        need = max(1, int(round(self.chunk_sec * used_sr)))
        overlap_n = max(0, int(round(self.overlap_sec * used_sr)))
        overlap_tail = np.zeros(0, dtype=np.float32)
        pending_text: str | None = None
        # If API is slower than hop, drop backlog so UI stays near-live.
        max_backlog = need * 3

        try:
            while not self._stop.is_set():
                try:
                    block = self._audio_q.get(timeout=0.2)
                except queue.Empty:
                    continue
                buf.append(block)
                buf_samples += block.size
                if buf_samples > max_backlog:
                    # Keep newest ~1 hop so lag does not compound.
                    chunk_all = np.concatenate(buf)
                    keep = chunk_all[-need:]
                    buf = [keep]
                    buf_samples = keep.size
                if buf_samples < need:
                    continue
                chunk = np.concatenate(buf)
                buf = []
                buf_samples = 0
                if chunk.size > need:
                    leftover = chunk[need:]
                    chunk = chunk[:need]
                    if leftover.size:
                        buf.append(leftover)
                        buf_samples = leftover.size
                mono_cap = chunk.reshape(-1).astype(np.float32, copy=False)
                if overlap_tail.size:
                    mono_cap = np.concatenate([overlap_tail, mono_cap])
                if overlap_n > 0:
                    overlap_tail = chunk.reshape(-1)[-overlap_n:].astype(
                        np.float32, copy=True
                    )
                else:
                    overlap_tail = np.zeros(0, dtype=np.float32)

                mono = mono_cap
                if used_sr != TARGET_SR:
                    mono = resample_linear(mono, used_sr, TARGET_SR)
                # Silence → finalize open partial line (end of utterance).
                if float(np.max(np.abs(mono))) < 0.008:
                    if pending_text:
                        self._finalize_pending(pending_text, mode)
                        pending_text = None
                        self.client.reset()
                    continue

                wav = float32_to_wav_bytes(mono, TARGET_SR)
                self.bridge.status.emit("识别中…")
                t0 = time.perf_counter()
                result = self.client.transcribe_wav(wav)
                api_ms = (time.perf_counter() - t0) * 1000.0
                if not result.get("ok"):
                    self.bridge.error.emit(result.get("error") or "识别失败")
                    self.bridge.status.emit("出错")
                    continue
                if result.get("language_mismatch"):
                    rejected = (result.get("rejected_text") or "")[:40]
                    det = result.get("detected_language") or "?"
                    locked = result.get("locked_language") or self.client.language or "?"
                    self.bridge.status.emit(
                        f"跳过非锁定语言 · 期望{locked} 检出{det} · "
                        f"{rejected!r} · API {api_ms:.0f}ms"
                    )
                    continue
                text = (result.get("text") or "").strip()
                if not text or result.get("duplicate"):
                    self.bridge.status.emit(
                        f"录音中 · {mode} · {self.client.short_name} · "
                        f"API {api_ms:.0f}ms"
                    )
                    continue

                warn = ""
                if api_ms > self.chunk_sec * 1000.0 * 1.25:
                    warn = " · API偏慢"

                # Bilingual OR quality / longer chunks: each hop is a committed line →
                # translate to 中文 immediately (do not wait for silence).
                if (
                    self.bilingual
                    or self.latency_preset in ("accurate", "precise")
                    or self.chunk_sec >= 3.5
                ):
                    if text != pending_text:
                        self._finalize_pending(text, mode)
                        pending_text = text
                    self.bridge.status.emit(
                        f"录音中 · {mode} · {preset_label} · "
                        f"{self.client.short_name} · API {api_ms:.0f}ms{warn}"
                    )
                    continue

                # Realtime + bilingual off: partial first, finalize on silence/next.
                if pending_text and pending_text != text:
                    self._finalize_pending(pending_text, mode)

                pending_text = text
                self.bridge.transcript.emit(text, "", False)
                self.bridge.status.emit(
                    f"录音中 · {mode} · {preset_label} · "
                    f"{self.client.short_name} · API {api_ms:.0f}ms{warn}"
                )
        finally:
            if pending_text:
                self._finalize_pending(pending_text, mode)
            session.close()
            self.bridge.status.emit("已停止")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------


class OverlayWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("实时字幕 Overlay")
        flags = (
            Qt.Window
            | Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
        )
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(780, 200)
        self.resize(1180, 240)

        self._drag_pos: QPoint | None = None
        self._mask_only = False
        self._chrome_height = 0  # filled after build; used to shrink/restore
        self.translator = Translator()
        self.client = MultiAsrClient(pick_default_model_key())
        self.client.set_language(_asr_language_default())
        self.bridge = Bridge()
        self.worker = AudioWorker(self.bridge, self.client, self.translator)
        self._recording = False
        self._history: list[tuple[str, str]] = []
        self._loopbacks: list[AudioDevice] = []
        self._mics: list[AudioDevice] = []
        self._devices_by_key: dict[str, AudioDevice] = {}

        self._build_ui()
        self.bridge.transcript.connect(self._on_transcript)
        self.bridge.status.connect(self._on_status)
        self.bridge.error.connect(self._on_error)

        # Overlay panel opacity (遮罩透明度) from settings.json
        self._overlay_opacity = clamp_opacity(
            load_settings(BASE_DIR).get(FIELD_OVERLAY_OPACITY)
        )

        self._last_default_playback = current_default_playback_name()
        self._refresh_devices(show_help_if_empty=True)
        self._sync_model_banner()
        self._apply_overlay_opacity(self._overlay_opacity)

        # Plug/unplug headphones → Windows default playback changes; refresh list.
        self._device_poll = QTimer(self)
        self._device_poll.setInterval(2500)
        self._device_poll.timeout.connect(self._poll_default_playback)
        self._device_poll.start()

        cfg = self.client.check()
        if cfg:
            self.status_lbl.setText(cfg)
            self.primary_lbl.setText("请点击「设置」填写硅基流动 API Key")
        else:
            self.status_lbl.setText(
                f"就绪 · {self.client.short_name} · 默认捕获系统声音(视频)"
            )
            self.primary_lbl.setText("播放视频后点「开始」→ 双语字幕显示系统音频内容")
            self.secondary_lbl.setText("")

        screen = QApplication.primaryScreen()
        if screen:
            geo = screen.availableGeometry()
            self.move(geo.left() + 40, geo.bottom() - self.height() - 60)

    def _combo_style(self) -> str:
        return (
            "QComboBox { background: #333; color: #eee; border-radius: 6px; "
            "padding: 2px 8px; font-size: 12px; }"
            "QComboBox QAbstractItemView { background: #2a2a2e; color: #eee; }"
        )

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)
        self._root_layout = root

        # Chrome (toolbars) — can be hidden for 「仅遮罩」clean subtitle look
        self.chrome_wrap = QWidget()
        chrome_v = QVBoxLayout(self.chrome_wrap)
        chrome_v.setContentsMargins(0, 0, 0, 0)
        chrome_v.setSpacing(6)

        bar = QHBoxLayout()
        bar.setSpacing(8)

        self.title_lbl = QLabel("字幕 HUD")
        self.title_lbl.setStyleSheet("color: #ccc; font-size: 12px;")
        bar.addWidget(self.title_lbl)

        self.source_combo = QComboBox()
        self.source_combo.setFixedHeight(28)
        self.source_combo.setMinimumWidth(140)
        self.source_combo.setStyleSheet(self._combo_style())
        self.source_combo.addItem("系统声音(视频)", "loopback")
        self.source_combo.addItem("麦克风", "mic")
        self.source_combo.setCurrentIndex(0)
        self.source_combo.currentIndexChanged.connect(self._on_source_mode_changed)
        bar.addWidget(self.source_combo)

        self.device_combo = QComboBox()
        self.device_combo.setFixedHeight(28)
        self.device_combo.setMinimumWidth(220)
        self.device_combo.setStyleSheet(self._combo_style())
        bar.addWidget(self.device_combo, stretch=1)

        self.btn_refresh = QPushButton("刷新设备")
        self.btn_refresh.setFixedHeight(28)
        self.btn_refresh.setStyleSheet(self._btn_style("#444"))
        self.btn_refresh.clicked.connect(lambda: self._refresh_devices(True))
        bar.addWidget(self.btn_refresh)

        self.btn_toggle = QPushButton("开始")
        self.btn_toggle.setFixedHeight(28)
        self.btn_toggle.setStyleSheet(self._btn_style("#2d8f4e"))
        self.btn_toggle.clicked.connect(self._toggle)
        bar.addWidget(self.btn_toggle)

        self.btn_clear = QPushButton("清空")
        self.btn_clear.setFixedHeight(28)
        self.btn_clear.setStyleSheet(self._btn_style("#555"))
        self.btn_clear.clicked.connect(self._clear)
        bar.addWidget(self.btn_clear)

        self.btn_settings = QPushButton("设置")
        self.btn_settings.setFixedHeight(28)
        self.btn_settings.setStyleSheet(self._btn_style("#3d5a80"))
        self.btn_settings.setToolTip("API Key / Base URL / 遮罩透明度")
        self.btn_settings.clicked.connect(self._open_settings)
        bar.addWidget(self.btn_settings)

        self.btn_quit = QPushButton("关闭")
        self.btn_quit.setFixedHeight(28)
        self.btn_quit.setStyleSheet(self._btn_style("#8b3a3a"))
        self.btn_quit.clicked.connect(self.close)
        bar.addWidget(self.btn_quit)

        self.btn_mask_only = QPushButton("仅遮罩")
        self.btn_mask_only.setFixedHeight(28)
        self.btn_mask_only.setStyleSheet(self._btn_style("#5a4a7a"))
        self.btn_mask_only.setToolTip(
            "隐藏工具栏，只保留字幕遮罩（更干净）。\n"
            "恢复：双击遮罩，或按 Esc。"
        )
        self.btn_mask_only.clicked.connect(self._enter_mask_only)
        bar.addWidget(self.btn_mask_only)

        chrome_v.addLayout(bar)

        # Row 2: model / bilingual / chunk
        bar2 = QHBoxLayout()
        bar2.setSpacing(8)

        self.model_banner = QLabel("")
        self.model_banner.setStyleSheet(
            "color: #9cf; font-size: 12px; font-weight: bold;"
        )
        bar2.addWidget(self.model_banner)

        self.model_combo = QComboBox()
        self.model_combo.setFixedHeight(28)
        self.model_combo.setMinimumWidth(260)
        self.model_combo.setStyleSheet(self._combo_style())
        self._fill_model_combo()
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        bar2.addWidget(self.model_combo)

        lang_lbl = QLabel("识别语言")
        lang_lbl.setStyleSheet("color: #aaa; font-size: 12px;")
        bar2.addWidget(lang_lbl)

        self.lang_combo = QComboBox()
        self.lang_combo.setFixedHeight(28)
        self.lang_combo.setMinimumWidth(90)
        self.lang_combo.setStyleSheet(self._combo_style())
        self.lang_combo.addItem("日语", "ja")
        self.lang_combo.addItem("中文", "zh")
        self.lang_combo.addItem("英语", "en")
        want_lang = _asr_language_default()
        idx_lang = self.lang_combo.findData(want_lang)
        self.lang_combo.setCurrentIndex(idx_lang if idx_lang >= 0 else 0)
        self.lang_combo.setToolTip(
            "锁定 ASR 识别语言（传给 API language=ja|zh|en）。\n"
            "日语视频请选「日语」，避免串成中文/英文。\n"
            "主字幕=识别原文；副字幕=简体中文翻译。"
        )
        self.lang_combo.currentIndexChanged.connect(self._on_language_changed)
        bar2.addWidget(self.lang_combo)
        self.client.set_language(self.lang_combo.currentData() or "ja")

        self.bilingual_chk = QCheckBox("双语字幕")
        self.bilingual_chk.setChecked(_bilingual_default())
        self.bilingual_chk.setStyleSheet("color: #eee; font-size: 12px;")
        self.bilingual_chk.setToolTip(
            "定稿后：主行=所选识别语言原文；副行=纯简体中文翻译（无日英混排）"
        )
        self.bilingual_chk.stateChanged.connect(self._on_bilingual_toggled)
        bar2.addWidget(self.bilingual_chk)
        self.worker.bilingual = self.bilingual_chk.isChecked()

        lat_lbl = QLabel("延迟")
        lat_lbl.setStyleSheet("color: #aaa; font-size: 12px;")
        bar2.addWidget(lat_lbl)

        self.latency_combo = QComboBox()
        self.latency_combo.setFixedHeight(28)
        self.latency_combo.setMinimumWidth(100)
        self.latency_combo.setStyleSheet(self._combo_style())
        self.latency_combo.addItem("实时", "realtime")
        self.latency_combo.addItem("准确", "accurate")
        self.latency_combo.addItem("更准", "precise")
        self.latency_combo.addItem("自定义", "custom")
        want = self.worker.latency_preset
        idx = self.latency_combo.findData(want)
        self.latency_combo.setCurrentIndex(idx if idx >= 0 else 1)
        self.latency_combo.setToolTip(
            "快捷预设（会同步右侧滑条）：\n"
            "实时 1.5s · 准确 6.5s · 更准 8.0s；\n"
            "自定义 = 滑条任意 1.5–10s（重叠约 18%）。"
        )
        self.latency_combo.currentIndexChanged.connect(self._on_latency_changed)
        bar2.addWidget(self.latency_combo)

        chunk_lbl = QLabel("块")
        chunk_lbl.setStyleSheet("color: #aaa; font-size: 12px;")
        bar2.addWidget(chunk_lbl)

        self.chunk_slider = QSlider(Qt.Horizontal)
        self.chunk_slider.setMinimum(15)   # 1.5s
        self.chunk_slider.setMaximum(100)  # 10.0s
        self.chunk_slider.setFixedWidth(140)
        self.chunk_slider.setFixedHeight(22)
        self.chunk_slider.setValue(int(round(self.worker.chunk_sec * 10)))
        self.chunk_slider.setStyleSheet(
            "QSlider::groove:horizontal {"
            " height: 6px; background: #3a3a45; border-radius: 3px; }"
            "QSlider::handle:horizontal {"
            " width: 14px; height: 14px; margin: -4px 0;"
            " background: #6a8cff; border-radius: 7px; }"
            "QSlider::sub-page:horizontal {"
            " background: #5a7ae0; border-radius: 3px; }"
        )
        self.chunk_slider.setToolTip(
            "ASR 切片时长 1.5–10s（拖动调节）。\n"
            "重叠自动 ≈ 块长的 18%。写入 settings.json，下次开始生效。"
        )
        self.chunk_slider.valueChanged.connect(self._on_chunk_slider_changed)
        self.chunk_slider.sliderReleased.connect(self._persist_chunk_settings)
        bar2.addWidget(self.chunk_slider)

        self.chunk_value_lbl = QLabel(
            f"{self.worker.chunk_sec:.1f}s/+{self.worker.overlap_sec:.1f}s"
        )
        self.chunk_value_lbl.setMinimumWidth(88)
        self.chunk_value_lbl.setStyleSheet(
            "color: #9cb4ff; font-size: 12px; font-weight: bold;"
        )
        bar2.addWidget(self.chunk_value_lbl)

        self.status_lbl = QLabel("")
        self.status_lbl.setStyleSheet("color: #8af; font-size: 12px;")
        bar2.addWidget(self.status_lbl, stretch=1)

        chrome_v.addLayout(bar2)
        root.addWidget(self.chrome_wrap)

        # Tiny restore control visible only in mask-only mode
        self.btn_restore_chrome = QPushButton("控")
        self.btn_restore_chrome.setFixedSize(28, 22)
        self.btn_restore_chrome.setStyleSheet(
            "QPushButton { background: rgba(60,50,90,180); color: #ddd; "
            "border: none; border-radius: 4px; font-size: 11px; }"
            "QPushButton:hover { background: rgba(90,70,130,220); }"
        )
        self.btn_restore_chrome.setToolTip("显示工具栏（或双击遮罩 / Esc）")
        self.btn_restore_chrome.clicked.connect(self._exit_mask_only)
        self.btn_restore_chrome.hide()
        root.addWidget(self.btn_restore_chrome, alignment=Qt.AlignLeft)

        # Subtitle panel: primary (large) + secondary (smaller translation)
        panel = QVBoxLayout()
        panel.setSpacing(2)
        panel.setContentsMargins(10, 10, 10, 10)

        self.primary_lbl = QLabel("")
        self.primary_lbl.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        self.primary_lbl.setWordWrap(True)
        self.primary_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        font = QFont("Microsoft YaHei UI", 22)
        font.setBold(True)
        self.primary_lbl.setFont(font)
        self.primary_lbl.setStyleSheet("color: #fff; background: transparent;")
        panel.addWidget(self.primary_lbl)

        self.secondary_lbl = QLabel("")
        self.secondary_lbl.setAlignment(Qt.AlignCenter | Qt.AlignTop)
        self.secondary_lbl.setWordWrap(True)
        sec_font = QFont("Microsoft YaHei UI", 14)
        self.secondary_lbl.setFont(sec_font)
        self.secondary_lbl.setStyleSheet("color: #c8d0e0; background: transparent;")
        panel.addWidget(self.secondary_lbl)

        self.panel_wrap = QWidget()
        self.panel_wrap.setLayout(panel)
        # alpha set by _apply_overlay_opacity
        self.panel_wrap.setStyleSheet(
            "background: rgba(0,0,0,180); border-radius: 10px;"
        )
        root.addWidget(self.panel_wrap, stretch=1)

        self.tip_lbl = QLabel(
            "识别语言默认「日语」(可改中/英) · 主字幕=原文 · 副字幕=简体中文 · "
            "块时长可拖滑条 1.5–10s（重叠≈18%）· 系统回环 · 「仅遮罩」藏工具栏"
        )
        self.tip_lbl.setStyleSheet("color: #888; font-size: 11px;")
        self.tip_lbl.setWordWrap(True)
        root.addWidget(self.tip_lbl)

    def _fill_model_combo(self) -> None:
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        models = available_models()
        if not models:
            self.model_combo.addItem("（无可用模型）", "")
        else:
            for m in models:
                self.model_combo.addItem(f"{m.label}", m.key)
            want = self.client.model_key
            idx = self.model_combo.findData(want)
            if idx < 0:
                idx = self.model_combo.findData("qwen-sf")
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)
        self.model_combo.blockSignals(False)

    def _sync_model_banner(self) -> None:
        bi = "双语开" if self.worker.bilingual else "双语关"
        pl = LATENCY_PRESETS.get(self.worker.latency_preset, {}).get(
            "label",
            "自定义" if self.worker.latency_preset == "custom" else self.worker.latency_preset,
        )
        lang = getattr(self.client, "language_label", None) or "日语"
        self.model_banner.setText(
            f"ASR：{self.client.short_name}  ·  语种:{lang}  ·  {bi}  ·  "
            f"{pl} {self.worker.chunk_sec:.1f}s/+{self.worker.overlap_sec:.1f}s"
        )
        if hasattr(self, "chunk_value_lbl"):
            self.chunk_value_lbl.setText(
                f"{self.worker.chunk_sec:.1f}s/+{self.worker.overlap_sec:.1f}s"
            )

    def _on_model_changed(self, _index: int = 0) -> None:
        key = self.model_combo.currentData()
        if not key:
            return
        was = self._recording
        if was:
            self.worker.stop()
            self._recording = False
            self.btn_toggle.setText("开始")
            self.btn_toggle.setStyleSheet(self._btn_style("#2d8f4e"))
        self.client.set_model(key)
        self.client.reset()
        self._sync_model_banner()
        self.status_lbl.setText(f"已切换 ASR → {self.client.short_name}")

    def _on_language_changed(self, _index: int = 0) -> None:
        code = self.lang_combo.currentData()
        if not code:
            return
        was = self._recording
        if was:
            self.worker.stop()
            self._recording = False
            self.btn_toggle.setText("开始")
            self.btn_toggle.setStyleSheet(self._btn_style("#2d8f4e"))
        self.client.set_language(str(code))
        self.client.reset()
        self._sync_model_banner()
        self.status_lbl.setText(
            f"识别语言 → {self.client.language_label}"
            f"（每请求必带 language={self.client.language} + prompt）"
        )

    def _on_bilingual_toggled(self, state: int) -> None:
        self.worker.bilingual = state == Qt.Checked
        self._sync_model_banner()

    def _on_latency_changed(self, _index: int = 0) -> None:
        key = self.latency_combo.currentData()
        if not key:
            return
        if key == "custom":
            # Keep current slider value; just mark custom
            self.worker.apply_chunk(self.chunk_slider.value() / 10.0, persist_preset="custom")
        else:
            self.worker.apply_preset(str(key))
            self.chunk_slider.blockSignals(True)
            self.chunk_slider.setValue(int(round(self.worker.chunk_sec * 10)))
            self.chunk_slider.blockSignals(False)
        self._sync_model_banner()
        self._persist_chunk_settings()
        pl = LATENCY_PRESETS.get(self.worker.latency_preset, {}).get(
            "label", "自定义"
        )
        self.status_lbl.setText(
            f"延迟 → {pl} "
            f"({self.worker.chunk_sec:.1f}s 切片 / "
            f"{self.worker.overlap_sec:.1f}s 重叠，下次开始生效)"
        )

    def _on_chunk_slider_changed(self, value: int) -> None:
        chunk = clamp_chunk_seconds(value / 10.0)
        self.worker.apply_chunk(chunk)
        # Sync combo without re-entrancy
        self.latency_combo.blockSignals(True)
        idx = self.latency_combo.findData(self.worker.latency_preset)
        if idx < 0:
            idx = self.latency_combo.findData("custom")
        if idx >= 0:
            self.latency_combo.setCurrentIndex(idx)
        self.latency_combo.blockSignals(False)
        self._sync_model_banner()
        self.status_lbl.setText(
            f"块时长 → {self.worker.chunk_sec:.1f}s "
            f"(重叠 {self.worker.overlap_sec:.1f}s≈18%，下次开始生效)"
        )

    def _persist_chunk_settings(self) -> None:
        """Write chunk/overlap/preset to settings.json (merge, keep API key)."""
        try:
            update_settings(
                {
                    FIELD_CHUNK_SECONDS: self.worker.chunk_sec,
                    FIELD_OVERLAP_SECONDS: self.worker.overlap_sec,
                    FIELD_LATENCY_PRESET: self.worker.latency_preset,
                },
                BASE_DIR,
            )
            # Keep env in sync for any code paths that read getenv
            os.environ["ASR_CHUNK_SECONDS"] = f"{self.worker.chunk_sec:.1f}"
            os.environ["ASR_OVERLAP_SECONDS"] = f"{self.worker.overlap_sec:.1f}"
            os.environ["ASR_LATENCY_PRESET"] = self.worker.latency_preset
        except OSError as exc:
            self.status_lbl.setText(f"无法保存块时长：{exc}")

    @staticmethod
    def _btn_style(bg: str) -> str:
        return (
            f"QPushButton {{ background: {bg}; color: white; border: none; "
            f"border-radius: 6px; padding: 4px 14px; font-size: 13px; }}"
            f"QPushButton:hover {{ opacity: 0.9; }}"
        )

    def _source_mode(self) -> str:
        return self.source_combo.currentData() or "loopback"

    def _refresh_devices(self, show_help_if_empty: bool = False) -> None:
        prev_key = ""
        try:
            prev_key = self.device_combo.currentData() or ""
        except Exception:
            prev_key = ""
        self._last_default_playback = current_default_playback_name()
        self._loopbacks, self._mics = list_audio_devices()
        self._devices_by_key = {d.key: d for d in self._loopbacks + self._mics}
        self._fill_device_combo(prefer_key=prev_key)
        if self._source_mode() == "loopback" and self._loopbacks:
            picked = pick_default_loopback(self._loopbacks)
            tip = f"已刷新 · 默认播放「{self._last_default_playback or '?'}」"
            if picked:
                tip += f" → {picked.device_name or picked.label}"
            self.status_lbl.setText(tip)
        if show_help_if_empty and self._source_mode() == "loopback" and not self._loopbacks:
            self.status_lbl.setText("未找到系统回环设备（插耳机后请再点刷新）")
            QMessageBox.warning(self, "需要系统声音回环", NO_LOOPBACK_HELP)

    def _fill_device_combo(self, prefer_key: str = "") -> None:
        self.device_combo.blockSignals(True)
        self.device_combo.clear()
        mode = self._source_mode()
        devices = self._loopbacks if mode == "loopback" else self._mics
        if not devices:
            if mode == "loopback":
                self.device_combo.addItem("（无回环设备 — 插耳机后点刷新）", "")
            else:
                self.device_combo.addItem("（未检测到麦克风）", "")
        else:
            for d in devices:
                self.device_combo.addItem(d.label, d.key)
            if mode == "loopback":
                # Prefer current default (★); else keep prior selection if still present
                default = pick_default_loopback(devices)
                chosen = None
                if default:
                    chosen = default.key
                elif prefer_key and self.device_combo.findData(prefer_key) >= 0:
                    chosen = prefer_key
                if chosen:
                    idx = self.device_combo.findData(chosen)
                    if idx >= 0:
                        self.device_combo.setCurrentIndex(idx)
            elif prefer_key:
                idx = self.device_combo.findData(prefer_key)
                if idx >= 0:
                    self.device_combo.setCurrentIndex(idx)
        self.device_combo.blockSignals(False)

    def _poll_default_playback(self) -> None:
        """When headphones plug in, default playback switches — refresh combo."""
        try:
            name = current_default_playback_name()
        except Exception:
            return
        if not name or name == getattr(self, "_last_default_playback", None):
            return
        prev = self._last_default_playback
        self._last_default_playback = name
        logging.getLogger("overlay.audio").info(
            "default playback changed: %r -> %r", prev, name
        )
        was_recording = self._recording
        if was_recording:
            self.worker.stop()
            self._recording = False
            self.btn_toggle.setText("开始")
            self.btn_toggle.setStyleSheet(self._btn_style("#2d8f4e"))
        self._refresh_devices(show_help_if_empty=False)
        if self._source_mode() != "loopback":
            self.status_lbl.setText(f"播放设备已切换为「{name}」")
            return
        if not self._loopbacks:
            self.status_lbl.setText(
                f"播放设备已切换为「{name}」，但未找到回环 — 可点刷新或看帮助"
            )
            QMessageBox.warning(self, "需要系统声音回环", NO_LOOPBACK_HELP)
            return
        default = pick_default_loopback(self._loopbacks)
        if default:
            idx = self.device_combo.findData(default.key)
            if idx >= 0:
                self.device_combo.setCurrentIndex(idx)
        if was_recording and default is not None:
            try:
                self.worker.start(default)
                self._recording = True
                self.btn_toggle.setText("停止")
                self.btn_toggle.setStyleSheet(self._btn_style("#c45c26"))
                self.status_lbl.setText(
                    f"已切换到「{name}」回环并继续字幕 · {default.label}"
                )
            except Exception as exc:
                self.status_lbl.setText(f"切换回环失败：{exc}（请点刷新后重试）")
        else:
            self.status_lbl.setText(
                f"检测到播放设备切换为「{name}」，已选对应回环 — 请点「开始」"
            )

    def _on_source_mode_changed(self, _index: int = 0) -> None:
        if self._recording:
            self.worker.stop()
            self._recording = False
            self.btn_toggle.setText("开始")
            self.btn_toggle.setStyleSheet(self._btn_style("#2d8f4e"))
        self._fill_device_combo()
        if self._source_mode() == "loopback" and not self._loopbacks:
            self.status_lbl.setText("未找到系统回环 — 点击刷新或查看帮助")
        else:
            mode = "系统声音" if self._source_mode() == "loopback" else "麦克风"
            self.status_lbl.setText(f"已切换到{mode}")

    def _selected_device(self) -> Optional[AudioDevice]:
        key = self.device_combo.currentData()
        if not key:
            return None
        return self._devices_by_key.get(key)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        chrome_a = max(40, min(220, int(round(self._overlay_opacity * 160))))
        border_a = max(60, min(220, int(round(self._overlay_opacity * 180))))
        painter.setBrush(QBrush(QColor(20, 20, 24, chrome_a)))
        painter.setPen(QPen(QColor(80, 80, 90, border_a), 1))
        painter.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 12, 12)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._drag_pos is not None and event.buttons() & Qt.LeftButton:
            self.move(event.globalPos() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        self._drag_pos = None

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton and self._mask_only:
            self._exit_mask_only()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key_Escape and self._mask_only:
            self._exit_mask_only()
            event.accept()
            return
        super().keyPressEvent(event)

    def _enter_mask_only(self) -> None:
        """Hide toolbars/tip; keep only the subtitle mask bar."""
        if self._mask_only:
            return
        self._mask_only = True
        # Remember chrome height so we can shrink the window cleanly
        chrome_h = self.chrome_wrap.height() + self.tip_lbl.height() + 12
        self._chrome_height = max(chrome_h, 70)
        self.chrome_wrap.hide()
        self.tip_lbl.hide()
        self.btn_restore_chrome.show()
        self.setFocus(Qt.OtherFocusReason)
        # Shrink toward a subtitle-bar footprint
        new_h = max(110, self.height() - self._chrome_height)
        self.setMinimumSize(480, 90)
        self.resize(self.width(), new_h)
        self.status_lbl.setText("仅遮罩 · 双击或 Esc 恢复工具栏")

    def _exit_mask_only(self) -> None:
        """Show toolbars/tip again."""
        if not self._mask_only:
            return
        self._mask_only = False
        self.btn_restore_chrome.hide()
        self.chrome_wrap.show()
        self.tip_lbl.show()
        self.setMinimumSize(780, 200)
        grow = self._chrome_height or 100
        self.resize(max(self.width(), 780), max(self.height() + grow, 240))
        self.status_lbl.setText("已恢复工具栏")

    def _apply_overlay_opacity(self, opacity: float) -> None:
        """Update chrome + subtitle panel mask alpha from 0.15–1.0."""
        self._overlay_opacity = clamp_opacity(opacity)
        # Subtitle panel: map 0.15–1.0 → alpha ~40–230
        alpha = int(round(40 + self._overlay_opacity * 190))
        alpha = max(40, min(230, alpha))
        if hasattr(self, "panel_wrap"):
            self.panel_wrap.setStyleSheet(
                f"background: rgba(0,0,0,{alpha}); border-radius: 10px;"
            )
        self.update()

    def _open_settings(self) -> None:
        was = self._recording
        if was:
            self.worker.stop()
            self._recording = False
            self.btn_toggle.setText("开始")
            self.btn_toggle.setStyleSheet(self._btn_style("#2d8f4e"))

        prev_opacity = self._overlay_opacity
        dlg = SettingsDialog(
            self,
            on_opacity_preview=self._apply_overlay_opacity,
            current_provider=self.client.model_key,
        )
        if dlg.exec_() != dlg.Accepted or not dlg.saved_settings:
            # Revert live preview if cancelled
            self._apply_overlay_opacity(prev_opacity)
            return

        # Reload env + clients from settings.json
        apply_settings_to_env(BASE_DIR)
        self.client.reload_credentials()
        self.translator.reload_credentials()
        self._apply_overlay_opacity(
            float(dlg.saved_settings.get(FIELD_OVERLAY_OPACITY) or prev_opacity)
        )
        # Apply chunk slider values saved from settings dialog
        saved_chunk = dlg.saved_settings.get(FIELD_CHUNK_SECONDS)
        if saved_chunk is not None:
            self.worker.apply_chunk(
                float(saved_chunk),
                persist_preset=str(
                    dlg.saved_settings.get(FIELD_LATENCY_PRESET) or "custom"
                ),
            )
            self.chunk_slider.blockSignals(True)
            self.chunk_slider.setValue(int(round(self.worker.chunk_sec * 10)))
            self.chunk_slider.blockSignals(False)
            self.latency_combo.blockSignals(True)
            idx = self.latency_combo.findData(self.worker.latency_preset)
            if idx < 0:
                idx = self.latency_combo.findData("custom")
            if idx >= 0:
                self.latency_combo.setCurrentIndex(idx)
            self.latency_combo.blockSignals(False)
            self._sync_model_banner()
        self._fill_model_combo()
        cfg = self.client.check()
        if cfg:
            self.status_lbl.setText(cfg)
            self.primary_lbl.setText("请点击「设置」填写硅基流动 API Key")
        else:
            self.status_lbl.setText(
                f"设置已保存 · {self.client.short_name} · 翻译={self.translator.backend_label()}"
            )

    def _toggle(self) -> None:
        if self._recording:
            self.worker.stop()
            self._recording = False
            self.btn_toggle.setText("开始")
            self.btn_toggle.setStyleSheet(self._btn_style("#2d8f4e"))
            self.status_lbl.setText("已停止")
            return

        err = self.client.check()
        if err:
            self.status_lbl.setText(err)
            return

        if self._source_mode() == "loopback":
            # Re-enumerate so headphones-as-default is available before capture
            prev_key = self.device_combo.currentData() or ""
            self._refresh_devices(show_help_if_empty=False)
            if not self._loopbacks:
                QMessageBox.warning(self, "需要系统声音回环", NO_LOOPBACK_HELP)
                return
            # Keep manual pick if still listed; otherwise ★ current default playback
            if prev_key and self.device_combo.findData(prev_key) >= 0:
                self.device_combo.setCurrentIndex(self.device_combo.findData(prev_key))
            else:
                default = pick_default_loopback(self._loopbacks)
                if default:
                    idx = self.device_combo.findData(default.key)
                    if idx >= 0:
                        self.device_combo.setCurrentIndex(idx)

        device = self._selected_device()
        if device is None:
            self.status_lbl.setText("请先选择音频设备")
            return

        # apply UI settings before start
        self.worker.bilingual = self.bilingual_chk.isChecked()
        lat = self.latency_combo.currentData()
        if lat and lat != "custom" and lat in LATENCY_PRESETS:
            self.worker.apply_preset(str(lat))
        else:
            self.worker.apply_chunk(
                self.chunk_slider.value() / 10.0,
                persist_preset="custom" if lat == "custom" else None,
            )
        key = self.model_combo.currentData()
        if key:
            self.client.set_model(key)
        lang = self.lang_combo.currentData()
        if lang:
            self.client.set_language(str(lang))

        self.worker.start(device)
        self._recording = True
        self.btn_toggle.setText("停止")
        self.btn_toggle.setStyleSheet(self._btn_style("#c45c26"))
        kind = "系统声音" if device.kind == "loopback" else "麦克风"
        self._sync_model_banner()
        self.status_lbl.setText(
            f"启动{kind}… ASR={self.client.short_name} · {device.label}"
        )

    def _clear(self) -> None:
        self._history.clear()
        self.client.reset()
        self.primary_lbl.setText("")
        self.secondary_lbl.setText("")

    def _on_transcript(
        self, text: str, translation: str = "", is_final: bool = True
    ) -> None:
        if is_final:
            self._history.append((text, translation or ""))
            self.primary_lbl.setText(text)
            self.primary_lbl.setStyleSheet("color: #fff; background: transparent;")
            if self.worker.bilingual and translation:
                self.secondary_lbl.setText(translation)
                self.secondary_lbl.show()
            else:
                self.secondary_lbl.setText("")
        else:
            # Partial: quick primary line only (no translate yet).
            self.primary_lbl.setText(text + " …")
            self.primary_lbl.setStyleSheet("color: #d0d8e8; background: transparent;")
            # Keep last finalized translation visible until next final.
        self._sync_model_banner()

    def _on_status(self, msg: str) -> None:
        self.status_lbl.setText(msg)

    def _on_error(self, msg: str) -> None:
        self.status_lbl.setText(f"错误：{msg}")

    def closeEvent(self, event) -> None:  # noqa: N802
        self.worker.stop()
        time.sleep(0.2)
        event.accept()


def main() -> int:
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    win = OverlayWindow()
    win.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
