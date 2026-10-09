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

from qtpy.QtCore import Qt, QTimer, Signal  # type: ignore[reportPrivateImportUsage]
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
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
from idtrackerai.segmentation_app.widgets.light_edits import (
    LightEditGroups,
    _ValueSlider,  # noqa: F401  (re-exported)
)

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

# light edit -> tooltip name; the tone contrast is told apart from the CLAHE one
_LIGHT_TIPS = {
    **{k: k for k in (
        "exposure", "brightness", "gamma", "shadows", "highlights", "blacks",
        "whites", "sharpness", "denoise",
    )},
    "contrast": "contrast_tone",
}


class EnhancementWidget(QWidget):
    """Picks the enhancement applied to every frame before segmentation."""

    settingsChanged = Signal(dict)
    settingsCommitted = Signal(dict)
    profileSaved = Signal(object)
    splitChanged = Signal(float)

    def __init__(self):
        super().__init__()
        self._loading = False
        # the fields Custom has no slider for, kept from what was loaded so a
        # profile with them set differently is not silently rewritten
        self._extra = {
            k: fp.DEFAULT_SETTINGS[k]
            for k in ("illumination_downsample", "correct_lighting")
        }
        # wheel, keyboard and groove clicks change a slider without ever
        # releasing it, so a value is also committed once it has settled
        self._settle = QTimer(self)
        self._settle.setSingleShot(True)
        self._settle.setInterval(300)
        self._settle.timeout.connect(self._committed)

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
        self.groups = LightEditGroups()
        self.contrast = self.groups.sliders["clahe_clip"]
        self.detail = self.groups.sliders["clahe_tile"]
        self.evenness = self.groups.sliders["illumination_sigma"]
        self.contrast_label = self.groups.labels["clahe_clip"]
        self.detail_label = self.groups.labels["clahe_tile"]
        self.evenness_label = self.groups.labels["illumination_sigma"]
        self.reset_button = QToolButton()
        self.reset_button.setText("Reset all")
        self.reset_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)

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

        self.custom_area = QWidget()
        custom = QVBoxLayout(self.custom_area)
        custom.setContentsMargins(0, 0, 0, 0)
        custom.addWidget(self.groups)
        custom.addWidget(self.reset_button, alignment=Qt.AlignmentFlag.AlignRight)
        self.custom_area.setVisible(False)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        layout.addLayout(top)
        layout.addWidget(self.custom_area)
        layout.addWidget(self.summary)
        layout.addWidget(self.hint)
        layout.addWidget(self.error)

        self._apply(PRESETS[NONE])

        self.preset.currentTextChanged.connect(self._preset_changed)
        self.groups.valueChanged.connect(self._slider_moved)
        self.groups.released.connect(self._released)
        self.reset_button.clicked.connect(self._reset_all)
        self.compare.valueChanged.connect(lambda v: self.splitChanged.emit(v / 100.0))
        self.save_button.clicked.connect(self.save_profile)
        self.load_button.clicked.connect(self.load_profile)

    # --------------------------------------------------------------- settings
    def settings(self) -> dict:
        if self.preset.currentText() != CUSTOM:
            return dict(PRESETS[self.preset.currentText()])
        return {
            "enhance": True,
            **self.groups.values(),
            **self._extra,
        }

    def setSettings(self, settings: dict | None) -> None:
        if isinstance(settings, dict):
            merged = {**fp.DEFAULT_SETTINGS, **settings}
        else:
            # no enhancement was asked for (a session without one, a .toml
            # without the key, or one saved by an older build holding only the
            # setting names). The defaults would read as the Standard preset
            # and switch enhancement on, so this has to mean "None".
            merged = dict(PRESETS[NONE])
        name = CUSTOM
        for preset_name, preset in PRESETS.items():
            if all(merged.get(k) == preset[k] for k in fp.SETTING_KEYS):
                name = preset_name
                break
        self._extra = {
            k: merged[k] for k in ("illumination_downsample", "correct_lighting")
        }
        self._loading = True
        try:
            self.preset.setCurrentText(name)
            self._apply(merged)
        finally:
            self._loading = False
        self._changed()
        self._committed()

    def _apply(self, settings: dict) -> None:
        self.groups.setValues(settings)

    # --------------------------------------------------------------- reacting
    def _preset_changed(self, name: str) -> None:
        self.custom_area.setVisible(name == CUSTOM)
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

    def _slider_moved(self) -> None:
        if not self._loading and self.preset.currentText() != CUSTOM:
            # the sliders already hold the preset's numbers, so Custom starts
            # from them; guarded so this is one change, not a change per signal
            self._loading = True
            try:
                self.preset.setCurrentText(CUSTOM)
            finally:
                self._loading = False
        self._changed()
        if not self._loading:
            self._settle.start()

    def _reset_all(self) -> None:
        self._loading = True
        try:
            self.groups.resetAll()
        finally:
            self._loading = False
        self._slider_moved()
        self._released()

    def _released(self) -> None:
        self._settle.stop()
        self._committed()

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
        self.reset_button.setToolTip(tips["enhancement_reset"])
        for label, widget, key in (
            (self.contrast_label, self.contrast, "enhancement_contrast"),
            (self.detail_label, self.detail, "enhancement_detail"),
            (self.evenness_label, self.evenness, "enhancement_evenness"),
            *(
                (self.groups.labels[k], self.groups.sliders[k], f"enhancement_{t}")
                for k, t in _LIGHT_TIPS.items()
            ),
        ):
            label.setToolTip(tips[key])
            widget.setToolTip(tips[key])
            widget.slider.setToolTip(tips[key])
