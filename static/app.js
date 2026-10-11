(() => {
  const $ = (id) => document.getElementById(id);

  const overlayRoot = $("overlayRoot");
  const statusDot = $("statusDot");
  const statusText = $("statusText");
  const errorBanner = $("errorBanner");
  const providerSelect = $("providerSelect");
  const langSelect = $("langSelect");
  const btnMic = $("btnMic");
  const btnClear = $("btnClear");
  const historyPanel = $("historyPanel");
  const subtitleText = $("subtitleText");

  let ws = null;
  let mediaStream = null;
  let audioCtx = null;
  let processor = null;
  let sourceNode = null;
  let recording = false;
  let chunkSeconds = 2.5;
  let lastFinal = "";
  let reconnectTimer = null;
  let pcmChunks = [];
  let pcmSamples = 0;
  const TARGET_RATE = 16000;

  function setStatus(kind, text) {
    statusDot.className = "status-dot" + (kind ? " " + kind : "");
    statusText.textContent = text;
  }

  function showError(msg) {
    if (!msg) {
      errorBanner.textContent = "";
      errorBanner.classList.remove("visible");
      return;
    }
    errorBanner.textContent = msg;
    errorBanner.classList.add("visible");
    setStatus("error", "错误");
  }

  function wsUrl() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    return `${proto}//${location.host}/ws/asr`;
  }

  function connect() {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
      return;
    }
    setStatus("", "连接中…");
    ws = new WebSocket(wsUrl());

    ws.onopen = () => {
      setStatus("connected", "已连接");
      showError("");
      sendConfig();
    };

    ws.onclose = () => {
      setStatus("", "已断开");
      if (recording) stopRecording(false);
      if (!reconnectTimer) {
        reconnectTimer = setTimeout(() => {
          reconnectTimer = null;
          connect();
        }, 1500);
      }
    };

    ws.onerror = () => {
      showError("WebSocket 连接失败，请确认服务已启动。");
    };

    ws.onmessage = (ev) => {
      let msg;
      try {
        msg = JSON.parse(ev.data);
      } catch {
        return;
      }
      handleMessage(msg);
    };
  }

  function sendConfig() {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(
      JSON.stringify({
        type: "config",
        language: langSelect.value,
        provider: providerSelect.value,
      })
    );
  }

  function handleMessage(msg) {
    switch (msg.type) {
      case "status":
        if (msg.chunk_seconds) chunkSeconds = Number(msg.chunk_seconds) || chunkSeconds;
        if (msg.status === "connected" || msg.status === "configured") {
          setStatus(recording ? "recording" : "connected", msg.message || msg.status);
        } else if (msg.status === "stopped") {
          setStatus("connected", msg.message || "已停止");
        }
        break;
      case "error":
        showError(msg.message || "未知错误");
        break;
      case "transcript":
        onTranscript(msg.text || "", !!msg.is_final);
        break;
      default:
        break;
    }
  }

  function onTranscript(text, isFinal) {
    if (!text) return;
    if (isFinal && text === lastFinal) return;
    subtitleText.textContent = text;
    subtitleText.classList.toggle("partial", !isFinal);
    if (isFinal) {
      lastFinal = text;
      appendHistory(text, true);
    }
  }

  function appendHistory(text, isFinal) {
    const line = document.createElement("div");
    line.className = "history-line" + (isFinal ? " final" : "");
    line.textContent = text;
    historyPanel.appendChild(line);
    while (historyPanel.children.length > 40) {
      historyPanel.removeChild(historyPanel.firstChild);
    }
    historyPanel.scrollTop = historyPanel.scrollHeight;
  }

  function downsampleTo16k(float32, inRate) {
    if (inRate === TARGET_RATE) return float32;
    const ratio = inRate / TARGET_RATE;
    const outLen = Math.floor(float32.length / ratio);
    const out = new Float32Array(outLen);
    for (let i = 0; i < outLen; i++) {
      const idx = Math.floor(i * ratio);
      out[i] = float32[idx];
    }
    return out;
  }

  function encodeWav(float32, sampleRate) {
    const n = float32.length;
    const buffer = new ArrayBuffer(44 + n * 2);
    const view = new DataView(buffer);
    const writeStr = (off, s) => {
      for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i));
    };
    writeStr(0, "RIFF");
    view.setUint32(4, 36 + n * 2, true);
    writeStr(8, "WAVE");
    writeStr(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeStr(36, "data");
    view.setUint32(40, n * 2, true);
    let off = 44;
    for (let i = 0; i < n; i++) {
      let s = Math.max(-1, Math.min(1, float32[i]));
      view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      off += 2;
    }
    return buffer;
  }

  function flushPcm() {
    if (!recording || pcmSamples < TARGET_RATE * 0.4) return;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const merged = new Float32Array(pcmSamples);
    let o = 0;
    for (const c of pcmChunks) {
      merged.set(c, o);
      o += c.length;
    }
    pcmChunks = [];
    pcmSamples = 0;
    const wav = encodeWav(merged, TARGET_RATE);
    const b64 = arrayBufferToBase64(wav);
    ws.send(
      JSON.stringify({
        type: "audio",
        mime: "audio/wav",
        data: b64,
      })
    );
  }

  async function startRecording() {
    showError("");
    connect();
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      await waitForOpen(4000);
    }
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      showError("无法连接后端 WebSocket。");
      return;
    }

    sendConfig();

    try {
      mediaStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          channelCount: 1,
        },
      });
    } catch (err) {
      const name = err && err.name;
      if (name === "NotAllowedError" || name === "PermissionDeniedError") {
        showError("麦克风权限被拒绝，请在浏览器设置中允许后重试。");
      } else if (name === "NotFoundError") {
        showError("未找到麦克风设备。");
      } else {
        showError("无法打开麦克风：" + (err.message || String(err)));
      }
      return;
    }

    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    sourceNode = audioCtx.createMediaStreamSource(mediaStream);
    processor = audioCtx.createScriptProcessor(4096, 1, 1);
    pcmChunks = [];
    pcmSamples = 0;
    const threshold = Math.floor(TARGET_RATE * chunkSeconds);

    processor.onaudioprocess = (ev) => {
      if (!recording) return;
      const input = ev.inputBuffer.getChannelData(0);
      const down = downsampleTo16k(input, audioCtx.sampleRate);
      pcmChunks.push(new Float32Array(down));
      pcmSamples += down.length;
      if (pcmSamples >= threshold) flushPcm();
    };

    sourceNode.connect(processor);
    processor.connect(audioCtx.destination);

    recording = true;
    btnMic.classList.add("recording");
    overlayRoot.classList.add("recording");
    setStatus("recording", "识别中…");
  }

  function stopRecording(notifyServer = true) {
    recording = false;
    btnMic.classList.remove("recording");
    overlayRoot.classList.remove("recording");
    try {
      flushPcm();
    } catch (_) {}
    if (processor) {
      try {
        processor.disconnect();
      } catch (_) {}
      processor.onaudioprocess = null;
      processor = null;
    }
    if (sourceNode) {
      try {
        sourceNode.disconnect();
      } catch (_) {}
      sourceNode = null;
    }
    if (audioCtx) {
      try {
        audioCtx.close();
      } catch (_) {}
      audioCtx = null;
    }
    stopTracks();
    pcmChunks = [];
    pcmSamples = 0;
    if (notifyServer && ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "stop" }));
    }
    setStatus("connected", "已停止");
  }

  function stopTracks() {
    if (mediaStream) {
      mediaStream.getTracks().forEach((t) => t.stop());
      mediaStream = null;
    }
  }

  function arrayBufferToBase64(buffer) {
    const bytes = new Uint8Array(buffer);
    let binary = "";
    const chunk = 0x8000;
    for (let i = 0; i < bytes.length; i += chunk) {
      binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
    }
    return btoa(binary);
  }

  function waitForOpen(ms) {
    return new Promise((resolve) => {
      const start = Date.now();
      const tick = () => {
        if (ws && ws.readyState === WebSocket.OPEN) return resolve(true);
        if (Date.now() - start > ms) return resolve(false);
        setTimeout(tick, 100);
      };
      tick();
    });
  }

  btnMic.addEventListener("click", () => {
    if (recording) stopRecording(true);
    else startRecording();
  });

  btnClear.addEventListener("click", () => {
    historyPanel.innerHTML = "";
    subtitleText.textContent = "";
    lastFinal = "";
    showError("");
  });

  providerSelect.addEventListener("change", () => {
    if (recording) stopRecording(true);
    sendConfig();
  });
  langSelect.addEventListener("change", sendConfig);

  fetch("/api/health")
    .then((r) => r.json())
    .then((h) => {
      if (h.provider) providerSelect.value = h.provider;
      if (h.chunk_seconds) chunkSeconds = h.chunk_seconds;
      if (h.config_error && h.provider !== "mock") {
        showError(h.config_error);
      }
    })
    .catch(() => {})
    .finally(connect);
})();
