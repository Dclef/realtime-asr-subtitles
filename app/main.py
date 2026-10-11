"""FastAPI: overlay UI + WebSocket ASR bridge (cloud API only)."""

from __future__ import annotations

import base64
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.asr_base import create_provider, provider_meta, resolve_provider_key
from app.settings_store import apply_settings_to_env

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
apply_settings_to_env(BASE_DIR)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("asr")

STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Realtime ASR Subtitles Overlay", docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8765",
        "http://localhost:8765",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _chunk_seconds() -> float:
    try:
        return float(os.getenv("ASR_CHUNK_SECONDS") or "6.5")
    except ValueError:
        return 2.5


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict[str, Any]:
    key = resolve_provider_key()
    meta = provider_meta(key)
    provider = create_provider(key)
    cfg_err = provider.check_config()
    return {
        "ok": True,
        "provider": key,
        "meta": meta,
        "configured": cfg_err is None,
        "config_error": cfg_err,
        "chunk_seconds": _chunk_seconds(),
        "providers": [provider_meta(k) for k in ("qwen-sf", "sensevoice", "mock")],
    }


@app.get("/api/providers")
async def list_providers() -> dict[str, Any]:
    return {
        "default": resolve_provider_key(),
        "providers": [provider_meta(k) for k in ("qwen-sf", "sensevoice", "mock")],
    }


@app.websocket("/ws/asr")
async def ws_asr(websocket: WebSocket) -> None:
    await websocket.accept()
    provider_key = resolve_provider_key()
    # Client may override via config message
    asr = create_provider(provider_key)
    closed = False

    async def send_json(payload: dict) -> None:
        if closed:
            return
        try:
            await websocket.send_text(json.dumps(payload, ensure_ascii=False))
        except Exception:
            pass

    try:
        cfg_err = asr.check_config()
        await send_json(
            {
                "type": "status",
                "status": "connected",
                "provider": provider_key,
                "meta": provider_meta(provider_key),
                "chunk_seconds": _chunk_seconds(),
                "message": f"已连接（{provider_key}）",
            }
        )
        if cfg_err:
            await send_json({"type": "error", "message": cfg_err})

        while True:
            raw = await websocket.receive()
            if raw.get("type") == "websocket.disconnect":
                break

            if "text" in raw and raw["text"] is not None:
                try:
                    msg = json.loads(raw["text"])
                except json.JSONDecodeError:
                    await send_json({"type": "error", "message": "无效的 JSON 消息"})
                    continue

                mtype = msg.get("type")
                if mtype == "config":
                    language = msg.get("language") or "auto"
                    # Allow runtime provider switch from UI
                    want = (msg.get("provider") or provider_key or "").strip()
                    if want:
                        mapped = _map_client_provider(want)
                        if mapped != provider_key:
                            provider_key = mapped
                            asr = create_provider(provider_key)
                            cfg_err = asr.check_config()
                            if cfg_err:
                                await send_json({"type": "error", "message": cfg_err})
                    asr.set_language(language)
                    asr.reset()
                    await send_json(
                        {
                            "type": "status",
                            "status": "configured",
                            "language": language,
                            "provider": provider_key,
                            "meta": provider_meta(provider_key),
                            "message": f"provider={provider_key}，语言={language}",
                        }
                    )
                elif mtype == "ping":
                    await send_json({"type": "pong"})
                elif mtype == "audio":
                    b64 = msg.get("data") or ""
                    mime = msg.get("mime") or "audio/webm"
                    try:
                        audio_bytes = base64.b64decode(b64)
                    except Exception:
                        await send_json({"type": "error", "message": "音频数据解码失败"})
                        continue
                    await _transcribe_and_push(asr, audio_bytes, mime, send_json)
                elif mtype == "stop":
                    asr.reset()
                    await send_json(
                        {"type": "status", "status": "stopped", "message": "已停止"}
                    )
                else:
                    await send_json(
                        {"type": "error", "message": f"未知消息类型：{mtype}"}
                    )

            elif "bytes" in raw and raw["bytes"] is not None:
                await _transcribe_and_push(
                    asr, raw["bytes"], "audio/webm", send_json
                )

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as exc:
        logger.exception("WebSocket error")
        await send_json({"type": "error", "message": f"服务异常：{exc}"})
    finally:
        closed = True


def _map_client_provider(want: str) -> str:
    """Map UI values / HF ids to internal keys."""
    w = want.strip().lower()
    aliases = {
        "sensevoice": "sensevoice",
        "sensevoicesmall": "sensevoice",
        "funaudiollm/sensevoicesmall": "sensevoice",
        "qwen-sf": "qwen-sf",
        "qwen3": "qwen-sf",
        "qwen": "qwen-sf",
        "qwen/qwen3-asr-1.7b": "qwen-sf",
        "qwen/qwen3-asr-1.7b-hf": "qwen-sf",
        "mock": "mock",
        "demo": "mock",
    }
    if w in aliases:
        return aliases[w]
    if w in ("sensevoice", "qwen-sf", "mock"):
        return w
    return "qwen-sf"


async def _transcribe_and_push(asr, audio_bytes: bytes, mime: str, send_json) -> None:
    cfg = asr.check_config()
    if cfg:
        await send_json({"type": "error", "message": cfg})
        return
    result = await asr.transcribe(audio_bytes, mime)
    if result.get("duplicate"):
        return
    if not result.get("ok"):
        await send_json(
            {"type": "error", "message": result.get("error") or "ASR 失败"}
        )
        return
    text = (result.get("text") or "").strip()
    if not text:
        return
    await send_json(
        {
            "type": "transcript",
            "text": text,
            "is_final": bool(result.get("is_final")),
            "provider": result.get("provider") or asr.name,
        }
    )


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
