"""The light edits, as four collapsible groups of sliders.

Grouped the way photo editors group them (Light, Tone, Detail, Evenness) so the
thirteen numbers are not one long list. The three CLAHE and evenness sliders
that existed before live here too; they are just more rows.
"""

from qtpy.QtCore import Qt, Signal  # type: ignore[reportPrivateImportUsage]
from qtpy.QtCore import QEvent
from qtpy.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from idtrackerai.extra_tools.detectron2_pipeline import preprocessing as fp


class _ValueSlider(QWidget):
    """A slider with its value beside it, like the threshold sliders.

    Works in integer steps internally because QSlider is integer-only, and
    shows the real value, so "1.5" is never displayed as "15". Double-clicking
    returns it to its default. It lives here, and is re-exported by
    enhancement_widget, so the two modules do not import each other.
    """

    valueChanged = Signal(float)

    def __init__(
        self,
        minimum: float,
        maximum: float,
        step: float,
        decimals: int = 1,
        default: float | None = None,
    ):
        super().__init__()
        self._step = step
        self._decimals = decimals
        self._minimum = minimum
        self._maximum = maximum
        self.default = minimum if default is None else default
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(int(round(minimum / step)), int(round(maximum / step)))
        self.slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.label = QLabel()
        self.label.setMinimumWidth(40)
        self.label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        layout.addWidget(self.slider)
        layout.addWidget(self.label)

        self.slider.valueChanged.connect(self._changed)
        for w in (self.slider, self.label):
            w.installEventFilter(self)
        self._changed(self.slider.value())

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QEvent.Type.MouseButtonDblClick:
            self.reset()
            return True
        return super().eventFilter(obj, event)

    def _changed(self, raw: int) -> None:
        value = self._round(raw)
        # a sign only where the slider goes both ways, so "+20" reads as an edit
        sign = "+" if self._minimum < 0 and value else ""
        self.label.setText(f"{value:{sign}.{self._decimals}f}")
        self.valueChanged.emit(value)

    def _round(self, raw: int) -> float:
        # 3 * 0.1 is 0.30000000000000004, which must not leak into settings
        return round(raw * self._step, self._decimals)

    def range(self) -> tuple[float, float]:
        return self._minimum, self._maximum

    def value(self) -> float:
        return self._round(self.slider.value())

    def setValue(self, value: float) -> None:
        self.slider.setValue(int(round(value / self._step)))

    def reset(self) -> None:
        self.setValue(self.default)


# settings key -> (label, minimum, maximum, step, decimals)
_ROWS = {
    "exposure": ("Exposure", *fp.LIGHT_EDIT_RANGES["exposure"], 0.1, 1),
    "brightness": ("Brightness", *fp.LIGHT_EDIT_RANGES["brightness"], 1, 0),
    "gamma": ("Gamma", *fp.LIGHT_EDIT_RANGES["gamma"], 0.05, 2),
    "contrast": ("Contrast", *fp.LIGHT_EDIT_RANGES["contrast"], 1, 0),
    "shadows": ("Shadows", *fp.LIGHT_EDIT_RANGES["shadows"], 1, 0),
    "highlights": ("Highlights", *fp.LIGHT_EDIT_RANGES["highlights"], 1, 0),
    "blacks": ("Blacks", *fp.LIGHT_EDIT_RANGES["blacks"], 1, 0),
    "whites": ("Whites", *fp.LIGHT_EDIT_RANGES["whites"], 1, 0),
    "sharpness": ("Sharpness", *fp.LIGHT_EDIT_RANGES["sharpness"], 1, 0),
    "denoise": ("Denoise", *fp.LIGHT_EDIT_RANGES["denoise"], 1, 0),
    "clahe_clip": ("Local contrast (CLAHE)", 0.5, 8.0, 0.1, 1),
    "clahe_tile": ("Local detail", 4, 32, 1, 0),
    "illumination_sigma": ("Lighting evenness", 5, 80, 1, 0),
}

GROUPS = {
    "Light": ("exposure", "brightness", "gamma"),
    "Tone": ("contrast", "shadows", "highlights", "blacks", "whites"),
    "Detail": ("sharpness", "denoise", "clahe_clip", "clahe_tile"),
    "Evenness": ("illumination_sigma",),
}
# Tone is the one most people leave alone, so it starts folded away
_CLOSED = ("Tone",)


class LightEditGroups(QWidget):
    """Every enhancement slider, in collapsible groups."""

    valueChanged = Signal()
    released = Signal()

    def __init__(self):
        super().__init__()
        self.sliders: dict[str, _ValueSlider] = {}
        self.labels: dict[str, QLabel] = {}
        self.toggles: dict[str, QToolButton] = {}
        self.members = GROUPS

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        for title, keys in GROUPS.items():
            toggle = QToolButton()
            toggle.setText(title)
            toggle.setCheckable(True)
            toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            toggle.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self.toggles[title] = toggle
            body = QWidget()
            grid = QGridLayout(body)
            grid.setContentsMargins(12, 0, 0, 0)
            for row, key in enumerate(keys):
                label, low, high, step, decimals = _ROWS[key]
                slider = _ValueSlider(
                    low, high, step, decimals, default=fp.DEFAULT_SETTINGS[key]
                )
                slider.valueChanged.connect(lambda _v: self.valueChanged.emit())
                slider.slider.sliderReleased.connect(self.released)
                self.sliders[key] = slider
                self.labels[key] = QLabel(label)
                grid.addWidget(self.labels[key], row, 0)
                grid.addWidget(slider, row, 1)
            toggle.toggled.connect(
                lambda open_, t=toggle, b=body: self._set_open(t, b, open_)
            )
            self._set_open(toggle, body, title not in _CLOSED)
            layout.addWidget(toggle)
            layout.addWidget(body)

    @staticmethod
    def _set_open(toggle: QToolButton, body: QWidget, open_: bool) -> None:
        toggle.setChecked(open_)
        toggle.setArrowType(
            Qt.ArrowType.DownArrow if open_ else Qt.ArrowType.RightArrow
        )
        body.setVisible(open_)

    def values(self) -> dict:
        # int or float as in the defaults, so a settings dict compares equal
        # to one read from a profile
        return {
            key: type(fp.DEFAULT_SETTINGS[key])(slider.value())
            for key, slider in self.sliders.items()
        }

    def setValues(self, values: dict) -> None:
        for key, slider in self.sliders.items():
            slider.setValue(float(values[key]))

    def resetAll(self) -> None:
        for slider in self.sliders.values():
            slider.reset()
