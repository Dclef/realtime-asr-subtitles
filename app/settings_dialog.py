"""Settings dialog: SiliconFlow API + overlay opacity (open-source friendly UI)."""

from __future__ import annotations

from typing import Callable, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from app.settings_store import (
    DEFAULT_CHUNK_SECONDS,
    DEFAULT_OVERLAY_OPACITY,
    DEFAULT_TRANSLATE_MODEL,
    FIELD_API_KEY,
    FIELD_ASR_PROVIDER,
    FIELD_BASE_URL,
    FIELD_CHUNK_SECONDS,
    FIELD_LATENCY_PRESET,
    FIELD_OVERLAP_SECONDS,
    FIELD_OVERLAY_OPACITY,
    FIELD_TRANSLATE_MODEL,
    SF_DEFAULT_BASE,
    clamp_chunk_seconds,
    clamp_opacity,
    clamp_overlap_seconds,
    derive_overlap_seconds,
    load_settings,
    mask_key,
    save_settings,
)


DIALOG_STYLE = """
QDialog {
    background: #1e1e24;
    color: #e8e8ec;
}
QGroupBox {
    font-weight: bold;
    font-size: 13px;
    border: 1px solid #3a3a45;
    border-radius: 8px;
    margin-top: 12px;
    padding: 14px 12px 10px 12px;
    color: #d0d4e0;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: #9cb4ff;
}
QLabel {
    color: #c8ccd8;
    font-size: 12px;
}
QLineEdit {
    background: #2a2a32;
    color: #f0f0f4;
    border: 1px solid #4a4a58;
    border-radius: 6px;
    padding: 7px 10px;
    font-size: 13px;
    selection-background-color: #3d5a80;
}
QLineEdit:focus {
    border: 1px solid #6a8cff;
}
QSlider::groove:horizontal {
    height: 6px;
    background: #3a3a45;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    width: 16px;
    height: 16px;
    margin: -5px 0;
    background: #6a8cff;
    border-radius: 8px;
}
QSlider::sub-page:horizontal {
    background: #5a7ae0;
    border-radius: 3px;
}
QCheckBox {
    color: #c8ccd8;
    font-size: 12px;
    spacing: 8px;
}
QPushButton {
    background: #3a3a48;
    color: #eee;
    border: none;
    border-radius: 6px;
    padding: 7px 16px;
    font-size: 13px;
}
QPushButton:hover {
    background: #4a4a5a;
}
QPushButton#primaryBtn {
    background: #3d6ad8;
    font-weight: bold;
}
QPushButton#primaryBtn:hover {
    background: #4d7aef;
}
QDialogButtonBox QPushButton {
    min-width: 80px;
}
"""


class SettingsDialog(QDialog):
    """Modal settings for API credentials and overlay appearance."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        on_opacity_preview: Optional[Callable[[float], None]] = None,
        current_provider: str = "qwen-sf",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("设置 · API 与外观")
        self.setModal(True)
        self.setMinimumWidth(480)
        self.setStyleSheet(DIALOG_STYLE)
        self._on_opacity_preview = on_opacity_preview
        self._settings = load_settings()
        if current_provider:
            self._settings[FIELD_ASR_PROVIDER] = current_provider

        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(18, 16, 18, 14)

        title = QLabel("硅基流动 API & 字幕遮罩")
        title.setFont(QFont("Microsoft YaHei UI", 14, QFont.Bold))
        title.setStyleSheet("color: #eef0f8; margin-bottom: 2px;")
        root.addWidget(title)

        hint = QLabel(
            "密钥仅保存在本机 settings.json（已 gitignore），不会上传或写入日志。"
            "申请 Key：https://cloud.siliconflow.cn"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #888; font-size: 11px; margin-bottom: 6px;")
        root.addWidget(hint)

        # --- API group ---
        api_box = QGroupBox("API 连接")
        form = QFormLayout(api_box)
        form.setSpacing(10)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.key_edit = QLineEdit()
        self.key_edit.setEchoMode(QLineEdit.Password)
        self.key_edit.setPlaceholderText("sk-… 硅基流动 API Key")
        self.key_edit.setText(str(self._settings.get(FIELD_API_KEY) or ""))
        self.key_edit.setClearButtonEnabled(True)
        form.addRow("API Key", self.key_edit)

        self.show_key_chk = QCheckBox("显示密钥")
        self.show_key_chk.toggled.connect(self._toggle_key_visible)
        form.addRow("", self.show_key_chk)

        self.base_edit = QLineEdit()
        self.base_edit.setPlaceholderText(SF_DEFAULT_BASE)
        self.base_edit.setText(
            str(self._settings.get(FIELD_BASE_URL) or SF_DEFAULT_BASE)
        )
        form.addRow("Base URL", self.base_edit)

        self.translate_edit = QLineEdit()
        self.translate_edit.setPlaceholderText(DEFAULT_TRANSLATE_MODEL)
        self.translate_edit.setText(
            str(self._settings.get(FIELD_TRANSLATE_MODEL) or DEFAULT_TRANSLATE_MODEL)
        )
        self.translate_edit.setToolTip(
            "双语翻译用的对话模型（与 ASR 共用同一 Key / Base URL）"
        )
        form.addRow("翻译模型", self.translate_edit)

        root.addWidget(api_box)

        # --- Appearance group ---
        look_box = QGroupBox("外观")
        look_layout = QVBoxLayout(look_box)

        row = QHBoxLayout()
        op_lbl = QLabel("遮罩透明度")
        row.addWidget(op_lbl)

        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setMinimum(15)  # 15%
        self.opacity_slider.setMaximum(100)
        opacity = clamp_opacity(self._settings.get(FIELD_OVERLAY_OPACITY))
        self.opacity_slider.setValue(int(round(opacity * 100)))
        self.opacity_slider.setToolTip("字幕条背景遮罩不透明度（即时预览）")
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        row.addWidget(self.opacity_slider, stretch=1)

        self.opacity_value_lbl = QLabel(f"{self.opacity_slider.value()}%")
        self.opacity_value_lbl.setMinimumWidth(40)
        self.opacity_value_lbl.setStyleSheet("color: #9cb4ff; font-weight: bold;")
        row.addWidget(self.opacity_value_lbl)
        look_layout.addLayout(row)

        look_hint = QLabel("数值越高遮罩越实；过低可能影响字幕可读性。")
        look_hint.setStyleSheet("color: #777; font-size: 11px;")
        look_layout.addWidget(look_hint)
        root.addWidget(look_box)

        # --- ASR chunk group ---
        asr_box = QGroupBox("ASR 切片时长")
        asr_layout = QVBoxLayout(asr_box)

        crow = QHBoxLayout()
        crow.addWidget(QLabel("块时长"))
        self.chunk_slider = QSlider(Qt.Horizontal)
        self.chunk_slider.setMinimum(15)   # 1.5s * 10
        self.chunk_slider.setMaximum(100)  # 10.0s * 10
        chunk = clamp_chunk_seconds(
            self._settings.get(FIELD_CHUNK_SECONDS, DEFAULT_CHUNK_SECONDS)
        )
        self.chunk_slider.setValue(int(round(chunk * 10)))
        self.chunk_slider.setToolTip(
            "ASR 切片 hop（1.5–10s）。重叠自动约 18% 块长。\n"
            "也可在主窗口工具栏拖动同一滑条。"
        )
        self.chunk_slider.valueChanged.connect(self._on_chunk_changed)
        crow.addWidget(self.chunk_slider, stretch=1)
        self.chunk_value_lbl = QLabel(f"{chunk:.1f}s")
        self.chunk_value_lbl.setMinimumWidth(48)
        self.chunk_value_lbl.setStyleSheet("color: #9cb4ff; font-weight: bold;")
        crow.addWidget(self.chunk_value_lbl)
        asr_layout.addLayout(crow)

        ov = clamp_overlap_seconds(
            self._settings.get(FIELD_OVERLAP_SECONDS), chunk
        )
        self.overlap_hint_lbl = QLabel(
            f"重叠自动 ≈ {ov:.1f}s（约 18% 块长，下次「开始」生效）"
        )
        self.overlap_hint_lbl.setStyleSheet("color: #777; font-size: 11px;")
        asr_layout.addWidget(self.overlap_hint_lbl)
        root.addWidget(asr_box)

        # Key status line
        keyed = bool(str(self._settings.get(FIELD_API_KEY) or "").strip())
        status = (
            f"当前密钥：已配置（{mask_key(str(self._settings.get(FIELD_API_KEY)))}）"
            if keyed
            else "当前密钥：未配置"
        )
        self.status_lbl = QLabel(status)
        self.status_lbl.setStyleSheet("color: #8af; font-size: 11px;")
        root.addWidget(self.status_lbl)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel
        )
        buttons.button(QDialogButtonBox.Save).setText("保存")
        buttons.button(QDialogButtonBox.Save).setObjectName("primaryBtn")
        buttons.button(QDialogButtonBox.Cancel).setText("取消")
        buttons.accepted.connect(self._on_save)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self._saved: Optional[dict] = None

    def _toggle_key_visible(self, checked: bool) -> None:
        self.key_edit.setEchoMode(
            QLineEdit.Normal if checked else QLineEdit.Password
        )

    def _on_opacity_changed(self, value: int) -> None:
        self.opacity_value_lbl.setText(f"{value}%")
        if self._on_opacity_preview:
            self._on_opacity_preview(value / 100.0)

    def _on_chunk_changed(self, value: int) -> None:
        chunk = clamp_chunk_seconds(value / 10.0)
        ov = derive_overlap_seconds(chunk)
        self.chunk_value_lbl.setText(f"{chunk:.1f}s")
        self.overlap_hint_lbl.setText(
            f"重叠自动 ≈ {ov:.1f}s（约 18% 块长，下次「开始」生效）"
        )

    def _on_save(self) -> None:
        api_key = self.key_edit.text().strip()
        base = self.base_edit.text().strip() or SF_DEFAULT_BASE
        translate = self.translate_edit.text().strip() or DEFAULT_TRANSLATE_MODEL
        opacity = clamp_opacity(self.opacity_slider.value() / 100.0)
        chunk = clamp_chunk_seconds(self.chunk_slider.value() / 10.0)
        overlap = derive_overlap_seconds(chunk)
        if not api_key:
            reply = QMessageBox.question(
                self,
                "未填写 API Key",
                "未填写 API Key，识别与翻译将无法工作。仍要保存吗？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return
        # Preserve latency preset unless chunk no longer matches a named preset
        preset = str(self._settings.get(FIELD_LATENCY_PRESET) or "custom")
        preset_chunks = {"realtime": 1.5, "accurate": 6.5, "precise": 8.0}
        if abs(preset_chunks.get(preset, -1) - chunk) > 0.05:
            preset = "custom"
            for name, sec in preset_chunks.items():
                if abs(sec - chunk) < 0.05:
                    preset = name
                    break
        data = {
            FIELD_API_KEY: api_key,
            FIELD_BASE_URL: base.rstrip("/"),
            FIELD_TRANSLATE_MODEL: translate,
            FIELD_ASR_PROVIDER: self._settings.get(FIELD_ASR_PROVIDER) or "qwen-sf",
            FIELD_OVERLAY_OPACITY: opacity,
            FIELD_CHUNK_SECONDS: chunk,
            FIELD_OVERLAP_SECONDS: overlap,
            FIELD_LATENCY_PRESET: preset,
        }
        try:
            save_settings(data)
        except OSError as exc:
            QMessageBox.critical(self, "保存失败", f"无法写入 settings.json：{exc}")
            return
        self._saved = data
        self.accept()

    @property
    def saved_settings(self) -> Optional[dict]:
        return self._saved
