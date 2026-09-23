"""Frame enhancement, judged live against the real footage.

Enhancement is a property of a recording setup, not of the software: a well lit
tank may need none, a turbid one under uneven lighting may need a lot. Choosing
it from a terminal means picking numbers, running something, opening the output
and guessing. Here the video is already on screen.

What is chosen here is applied to everything downstream — the thresholds, the
background model, the identification images and the Detectron2 pipeline alike —
so the preview never disagrees with what is actually segmented.

The controls follow the pattern of the thresholds widgets beside them: a named
preset in a combo box, with the individual numbers revealed only when "Custom"
is chosen. Most people never need the numbers.
"""

from pathlib import Path

from qtpy.QtCore import Qt, Signal  # type: ignore[reportPrivateImportUsage]
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from idtrackerai.extra_tools.detectron2_pipeline import preprocessing as fp
from idtrackerai.GUI_tools import WrappedLabel

NONE = "None"
GENTLE = "Gentle"
STANDARD = "Standard"
STRONG = "Strong"
CUSTOM = "Custom"

# Named starting points, so the common case is one choice rather than four
# numbers. The values differ only in CLAHE strength: that is the knob that
# actually changes how visible the animals are.
PRESETS = {
    NONE: {**fp.DEFAULT_SETTINGS, "enhance": False},
    GENTLE: {**fp.DEFAULT_SETTINGS, "enhance": True, "clahe_clip": 1.0},
    STANDARD: {**fp.DEFAULT_SETTINGS, "enhance": True, "clahe_clip": 1.5},
    STRONG: {**fp.DEFAULT_SETTINGS, "enhance": True, "clahe_clip": 3.0},
}


class _ValueSlider(QWidget):
    """A slider with its value beside it, like the threshold sliders.

    Works in integer steps internally because QSlider is integer-only, and
    shows the real value, so "1.5" is never displayed as "15".
    """

    valueChanged = Signal(float)

    def __init__(self, minimum: float, maximum: float, step: float, decimals: int = 1):
        super().__init__()
        self._step = step
        self._decimals = decimals
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(int(round(minimum / step)), int(round(maximum / step)))
        self.slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.label = QLabel()
        self.label.setMinimumWidth(34)
        self.label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        layout.addWidget(self.slider)
        layout.addWidget(self.label)

        self.slider.valueChanged.connect(self._changed)
        self._changed(self.slider.value())

    def _changed(self, raw: int) -> None:
        value = raw * self._step
        self.label.setText(f"{value:.{self._decimals}f}")
        self.valueChanged.emit(value)

    def value(self) -> float:
        return self.slider.value() * self._step

    def setValue(self, value: float) -> None:
        self.slider.setValue(int(round(value / self._step)))


class EnhancementWidget(QWidget):
    """Picks the enhancement applied to every frame before segmentation."""

    settingsChanged = Signal(dict)
    settingsCommitted = Signal(dict)
    profileSaved = Signal(object)
    splitChanged = Signal(float)

    def __init__(self):
        super().__init__()
        self._loading = False

        self.title = QLabel("Frame\nenhancement")
        self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.preset = QComboBox()
        self.preset.addItems([NONE, GENTLE, STANDARD, STRONG, CUSTOM])
        self.preset.setCurrentText(NONE)
        self.preset.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.compare = QSlider(Qt.Orientation.Horizontal)
        self.compare.setRange(0, 100)
        self.compare.setValue(100)
        self.compare.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.compare_left = QLabel("original")
        self.compare_right = QLabel("enhanced")

        self.save_button = QToolButton()
        self.save_button.setText("Save setup...")
        self.load_button = QToolButton()
        self.load_button.setText("Load setup...")
        for b in (self.save_button, self.load_button):
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        # revealed only for Custom
        self.contrast = _ValueSlider(0.5, 8.0, 0.1, decimals=1)
        self.detail = _ValueSlider(4, 32, 1, decimals=0)
        self.evenness = _ValueSlider(5, 80, 1, decimals=0)
        self.contrast_label = QLabel("Contrast boost")
        self.detail_label = QLabel("Local detail")
        self.evenness_label = QLabel("Lighting evenness")

        self.summary = WrappedLabel(framed=True)
        self.error = WrappedLabel()
        self.error.setVisible(False)
        self.error.setStyleSheet("color: #c0392b;")
        self.hint = WrappedLabel()
        self.hint.setVisible(False)
        self.hint.setStyleSheet("color: #b9770e;")

        top = QHBoxLayout()
        top.addWidget(self.title)
        top.addWidget(self.preset)
        top.addWidget(self.compare_left)
        top.addWidget(self.compare)
        top.addWidget(self.compare_right)
        top.addWidget(self.load_button)
        top.addWidget(self.save_button)

        self.custom_rows = QWidget()
        grid = QGridLayout(self.custom_rows)
        grid.setContentsMargins(0, 0, 0, 0)
        for row, (label, widget) in enumerate((
            (self.contrast_label, self.contrast),
            (self.detail_label, self.detail),
            (self.evenness_label, self.evenness),
        )):
            grid.addWidget(label, row, 0)
            grid.addWidget(widget, row, 1)
        self.custom_rows.setVisible(False)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        layout.addLayout(top)
        layout.addWidget(self.custom_rows)
        layout.addWidget(self.summary)
        layout.addWidget(self.hint)
        layout.addWidget(self.error)

        self._apply(PRESETS[NONE])

        self.preset.currentTextChanged.connect(self._preset_changed)
        for slider in (self.contrast, self.detail, self.evenness):
            slider.valueChanged.connect(lambda _v: self._changed())
            slider.slider.sliderReleased.connect(self._committed)
        self.compare.valueChanged.connect(lambda v: self.splitChanged.emit(v / 100.0))
        self.save_button.clicked.connect(self.save_profile)
        self.load_button.clicked.connect(self.load_profile)

    # --------------------------------------------------------------- settings
    def settings(self) -> dict:
        if self.preset.currentText() != CUSTOM:
            return dict(PRESETS[self.preset.currentText()])
        return {
            "enhance": True,
            "clahe_clip": round(self.contrast.value(), 2),
            "clahe_tile": int(self.detail.value()),
            "illumination_sigma": float(self.evenness.value()),
            "illumination_downsample": fp.DEFAULT_SETTINGS["illumination_downsample"],
            "correct_lighting": True,
        }

    def setSettings(self, settings: dict | None) -> None:
        merged = {**fp.DEFAULT_SETTINGS, **(settings or {})}
        name = CUSTOM
        for preset_name, preset in PRESETS.items():
            if all(merged.get(k) == preset[k] for k in fp.SETTING_KEYS):
                name = preset_name
                break
        self._loading = True
        try:
            self.preset.setCurrentText(name)
            self._apply(merged)
        finally:
            self._loading = False
        self._changed()
        self._committed()

    def _apply(self, settings: dict) -> None:
        self.contrast.setValue(float(settings["clahe_clip"]))
        self.detail.setValue(float(settings["clahe_tile"]))
        self.evenness.setValue(float(settings["illumination_sigma"]))

    # --------------------------------------------------------------- reacting
    def _preset_changed(self, name: str) -> None:
        self.custom_rows.setVisible(name == CUSTOM)
        if name != CUSTOM and not self._loading:
            self._loading = True
            try:
                self._apply(PRESETS[name])
            finally:
                self._loading = False
        self._changed()
        if not self._loading:
            self._committed()

    def _changed(self, *_args) -> None:
        if self._loading:
            return
        settings = self.settings()
        enhancing = settings["enhance"]
        for widget in (self.compare, self.compare_left, self.compare_right,
                       self.save_button):
            widget.setEnabled(enhancing)
        self.summary.setText(fp.describe(settings))
        self.clear_error()
        self.settingsChanged.emit(settings)

    def _committed(self, *_args) -> None:
        if not self._loading:
            self.settingsCommitted.emit(self.settings())

    def set_hint(self, message: str) -> None:
        """A caution shown under the summary, or cleared when empty."""
        self.hint.setText(message)
        self.hint.setVisible(bool(message))

    # ----------------------------------------------------------------- errors
    def show_error(self, message: str) -> None:
        self.error.setText(f"Preview failed: {message}")
        self.error.setVisible(True)

    def clear_error(self) -> None:
        self.error.setVisible(False)

    # --------------------------------------------------------------- profiles
    def save_profile(self) -> None:
        name, _ = QFileDialog.getSaveFileName(
            self, "Save this setup's enhancement", "preprocess_profile.json",
            filter="Setup profile (*.json)",
        )
        if not name:
            return
        try:
            path = fp.save_profile(Path(name), self.settings(), name=Path(name).stem)
        except (OSError, fp.PreprocessingError) as exc:
            QMessageBox.warning(self, "Could not save the setup", str(exc))
            return
        self.profileSaved.emit(path)

    def load_profile(self) -> None:
        name, _ = QFileDialog.getOpenFileName(
            self, "Open a setup's enhancement", "", filter="Setup profile (*.json)"
        )
        if not name:
            return
        try:
            settings = fp.load_profile(Path(name))
        except fp.PreprocessingError as exc:
            QMessageBox.warning(self, "Could not read the setup", str(exc))
            return
        self.setSettings(settings)

    # --------------------------------------------------------------- tooltips
    def setToolTips(self, tips: dict) -> None:
        self.title.setToolTip(tips["enhancement"])
        self.preset.setToolTip(tips["enhancement_preset"])
        for w in (self.compare, self.compare_left, self.compare_right):
            w.setToolTip(tips["enhancement_compare"])
        self.save_button.setToolTip(tips["enhancement_save"])
        self.load_button.setToolTip(tips["enhancement_load"])
        for label, widget, key in (
            (self.contrast_label, self.contrast, "enhancement_contrast"),
            (self.detail_label, self.detail, "enhancement_detail"),
            (self.evenness_label, self.evenness, "enhancement_evenness"),
        ):
            label.setToolTip(tips[key])
            widget.setToolTip(tips[key])
            widget.slider.setToolTip(tips[key])
