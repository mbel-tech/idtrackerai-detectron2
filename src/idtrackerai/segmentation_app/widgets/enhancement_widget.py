"""Controls for the frame enhancement, judged live against the real footage.

Enhancement is a property of a recording setup, not of the software: a well lit
tank may need none, a turbid one under uneven lighting may need strong CLAHE.
Choosing it from a terminal means picking numbers, running the sampler, opening
the output and judging — with no feedback loop. Here the video is already on
screen, so the setting can be judged where it matters.

What is chosen here is saved as the setup's profile, and that profile is what
carries the settings through annotation, training and inference. The controls
and the drawing are deliberately separate: this widget owns no painting, and
:class:`EnhancementPreview` owns no controls.
"""

from pathlib import Path

from qtpy.QtCore import Qt, Signal  # type: ignore[reportPrivateImportUsage]
from qtpy.QtWidgets import (
    QButtonGroup,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from idtrackerai.extra_tools.detectron2_pipeline import preprocessing as fp
from idtrackerai.GUI_tools import WrappedLabel

OFF = "Off"
CLAHE = "CLAHE"
CLAHE_AND_LIGHT = "CLAHE + even lighting"


class EnhancementWidget(QWidget):
    """Picks the enhancement for this recording setup."""

    settingsChanged = Signal(dict)   # live, on every change
    settingsCommitted = Signal(dict)  # on release, worth persisting
    profileSaved = Signal(object)     # Path
    splitChanged = Signal(float)

    def __init__(self):
        super().__init__()
        self._loading = False

        self.mode_group = QButtonGroup(self)
        self.off_button = QRadioButton(OFF)
        self.clahe_button = QRadioButton(CLAHE)
        self.full_button = QRadioButton(CLAHE_AND_LIGHT)
        for i, button in enumerate((self.off_button, self.clahe_button, self.full_button)):
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self.mode_group.addButton(button, i)
        self.full_button.setChecked(True)

        self.clip = QDoubleSpinBox()
        self.clip.setRange(0.1, 40.0)
        self.clip.setSingleStep(0.25)
        self.clip.setDecimals(2)

        self.tile = QSpinBox()
        # A tile grid of zero makes cv2.createCLAHE raise, from inside a paint
        # handler where the exception would be swallowed and shown as a blank
        # frame. The minimum is the guard.
        self.tile.setRange(1, 64)

        self.sigma = QDoubleSpinBox()
        self.sigma.setRange(0.1, 200.0)
        self.sigma.setSingleStep(1.0)
        self.sigma.setDecimals(1)

        self.downsample = QSpinBox()
        # Minimum 2 rather than 1: at 1 the Gaussian runs at full resolution,
        # which on 1080p takes long enough to make the live preview stutter.
        # The command line still allows 1 for a one-off.
        self.downsample.setRange(2, 16)

        self.split = QSlider(Qt.Orientation.Horizontal)
        self.split.setRange(0, 100)
        self.split.setValue(100)
        self.split.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        self.summary = WrappedLabel(framed=True)
        self.error = WrappedLabel()
        self.error.setVisible(False)
        self.error.setStyleSheet("color: #c0392b;")

        self.save_button = QPushButton("Save profile...")
        self.load_button = QPushButton("Load profile...")
        for button in (self.save_button, self.load_button):
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        form = QFormLayout()
        form.addRow("CLAHE clip limit", self.clip)
        form.addRow("CLAHE tile grid", self.tile)
        form.addRow("Lighting sigma", self.sigma)
        form.addRow("Lighting downsample", self.downsample)

        modes = QHBoxLayout()
        for button in (self.off_button, self.clahe_button, self.full_button):
            modes.addWidget(button)
        modes.addStretch()

        compare = QHBoxLayout()
        compare.addWidget(QLabel("raw"))
        compare.addWidget(self.split)
        compare.addWidget(QLabel("enhanced"))

        buttons = QHBoxLayout()
        buttons.addWidget(self.load_button)
        buttons.addWidget(self.save_button)
        buttons.addStretch()

        layout = QVBoxLayout()
        self.setLayout(layout)
        layout.addLayout(modes)
        layout.addLayout(form)
        layout.addLayout(compare)
        layout.addWidget(self.summary)
        layout.addWidget(self.error)
        layout.addLayout(buttons)

        self.setSettings(fp.DEFAULT_SETTINGS)

        self.mode_group.idToggled.connect(self._mode_changed)
        for box in (self.clip, self.tile, self.sigma, self.downsample):
            box.valueChanged.connect(self._changed)
            box.editingFinished.connect(self._committed)
        self.split.valueChanged.connect(
            lambda v: self.splitChanged.emit(v / 100.0)
        )
        self.save_button.clicked.connect(self.save_profile)
        self.load_button.clicked.connect(self.load_profile)

    # --------------------------------------------------------------- settings
    def settings(self) -> dict:
        enhance = not self.off_button.isChecked()
        return {
            "enhance": enhance,
            "clahe_clip": self.clip.value(),
            "clahe_tile": self.tile.value(),
            "illumination_sigma": self.sigma.value(),
            "illumination_downsample": self.downsample.value(),
            "correct_lighting": self.full_button.isChecked(),
        }

    def setSettings(self, settings: dict) -> None:
        merged = {**fp.DEFAULT_SETTINGS, **(settings or {})}
        self._loading = True
        try:
            if not merged["enhance"]:
                self.off_button.setChecked(True)
            elif merged["correct_lighting"]:
                self.full_button.setChecked(True)
            else:
                self.clahe_button.setChecked(True)
            self.clip.setValue(float(merged["clahe_clip"]))
            self.tile.setValue(int(merged["clahe_tile"]))
            self.sigma.setValue(float(merged["illumination_sigma"]))
            self.downsample.setValue(int(merged["illumination_downsample"]))
        finally:
            self._loading = False
        self._changed()
        self._committed()

    # ---------------------------------------------------------------- reacting
    def _mode_changed(self, _id: int, checked: bool) -> None:
        if checked:
            self._changed()
            self._committed()

    def _changed(self, *_args) -> None:
        if self._loading:
            return
        settings = self.settings()
        enhancing = settings["enhance"]
        lighting = settings["correct_lighting"]
        self.clip.setEnabled(enhancing)
        self.tile.setEnabled(enhancing)
        self.sigma.setEnabled(enhancing and lighting)
        self.downsample.setEnabled(enhancing and lighting)
        self.split.setEnabled(enhancing)
        self.summary.setText(fp.describe(settings))
        self.clear_error()
        self.settingsChanged.emit(settings)

    def _committed(self, *_args) -> None:
        if not self._loading:
            self.settingsCommitted.emit(self.settings())

    # ----------------------------------------------------------------- errors
    def show_error(self, message: str) -> None:
        self.error.setText(f"Preview failed: {message}")
        self.error.setVisible(True)

    def clear_error(self) -> None:
        self.error.setVisible(False)

    # --------------------------------------------------------------- profiles
    def save_profile(self) -> None:
        name, _ = QFileDialog.getSaveFileName(
            self, "Save enhancement profile", "preprocess_profile.json",
            filter="Profile (*.json)",
        )
        if not name:
            return
        try:
            path = fp.save_profile(Path(name), self.settings(), name=Path(name).stem)
        except (OSError, fp.PreprocessingError) as exc:
            QMessageBox.warning(self, "Could not save the profile", str(exc))
            return
        self.profileSaved.emit(path)

    def load_profile(self) -> None:
        name, _ = QFileDialog.getOpenFileName(
            self, "Open enhancement profile", "", filter="Profile (*.json)"
        )
        if not name:
            return
        try:
            settings = fp.load_profile(Path(name))
        except fp.PreprocessingError as exc:
            QMessageBox.warning(self, "Could not read the profile", str(exc))
            return
        self.setSettings(settings)
