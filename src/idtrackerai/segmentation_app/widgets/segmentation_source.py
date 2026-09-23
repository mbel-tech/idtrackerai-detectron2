"""Chooses where the animals' outlines come from: idtracker.ai's own
thresholding, or contours precomputed by an external segmentation model.
"""

import logging
from pathlib import Path

from qtpy.QtCore import Qt, Signal  # type: ignore[reportPrivateImportUsage]
from qtpy.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QSizePolicy,
    QToolButton,
    QWidget,
)

from idtrackerai.base.animals_detection.external_contours import ExternalContours
from idtrackerai.GUI_tools import WrappedLabel

THRESHOLDING = "Thresholding"
EXTERNAL = "External contours"
DETECTRON2 = "Detectron2 pipeline"


class SegmentationSourceWidget(QWidget):
    """Switches between thresholding and an external contour file.

    Emits the chosen file's path, or None for thresholding. The file is checked
    against the loaded video as soon as either changes, so a mismatched sidecar
    is caught here rather than after a long tracking run.
    """

    valueChanged = Signal(object)  # Path | None
    modeChanged = Signal(str)      # one of the three constants above

    def __init__(self):
        super().__init__()
        self.path: Path | None = None
        self.n_frames: int | None = None
        self.video_size: tuple[int, int] | None = None

        self.label = QLabel("Segmentation")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.source = QComboBox()
        self.source.addItems((THRESHOLDING, EXTERNAL, DETECTRON2))
        self.source.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.source.setSizePolicy(
            QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum
        )

        self.file_label = WrappedLabel(framed=True)
        self.file_label.setVisible(False)
        self.file_label.clicked.connect(self.choose_file)

        self.browse = QToolButton()
        self.browse.setText("Load contours")
        self.browse.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.browse.setVisible(False)
        self.browse.clicked.connect(self.choose_file)

        layout = QHBoxLayout()
        self.setLayout(layout)
        layout.addWidget(self.label)
        layout.addWidget(self.source)
        layout.addWidget(self.file_label)
        layout.addWidget(self.browse)

        self.source.currentTextChanged.connect(self.source_changed)

    # ------------------------------------------------------------- video info
    def set_video_info(self, n_frames: int, video_size: tuple[int, int]) -> None:
        """Called when a video is loaded, so the sidecar can be checked."""
        self.n_frames = n_frames
        self.video_size = video_size
        if self.path is not None:
            problem = self.describe_or_reject(self.path)
            if problem:
                QMessageBox.warning(self, "External contours do not fit", problem)
                self.clear_to_thresholding()

    # ------------------------------------------------------------- user input
    def source_changed(self, text: str) -> None:
        external = text == EXTERNAL
        self.file_label.setVisible(external and self.path is not None)
        self.browse.setVisible(external)
        self.modeChanged.emit(text)

        if text == DETECTRON2:
            # Preparing a model, not tracking with one. There is no contour
            # file yet, so this must not open a file dialog the way EXTERNAL
            # does; the panel below takes over.
            self.valueChanged.emit(None)
            return

        if not external:
            self.valueChanged.emit(None)
            return

        if self.path is None and not self.choose_file():
            # nothing loaded and the dialog was dismissed, so stay on the
            # thresholding path rather than leaving the app in a state it
            # cannot segment in
            self.clear_to_thresholding()
            return

        self.valueChanged.emit(self.path)

    def choose_file(self) -> bool:
        start = str(self.path.parent if self.path else Path.cwd())
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open external contours", start, filter="Contours (*.h5 *.hdf5)"
        )
        if not file_name:
            return False

        path = Path(file_name)
        problem = self.describe_or_reject(path)
        if problem:
            QMessageBox.warning(self, "Could not use this contour file", problem)
            return False

        self.path = path
        self.file_label.setText(path.name)
        self.file_label.setVisible(True)
        if self.source.currentText() != EXTERNAL:
            self.source.setCurrentText(EXTERNAL)  # re-enters source_changed
        else:
            self.valueChanged.emit(self.path)
        return True

    def clear_to_thresholding(self) -> None:
        self.path = None
        self.file_label.setVisible(False)
        self.browse.setVisible(False)
        # blockSignals so we emit once, below, instead of twice
        self.source.blockSignals(True)
        self.source.setCurrentText(THRESHOLDING)
        self.source.blockSignals(False)
        self.valueChanged.emit(None)

    # ------------------------------------------------------------ validation
    def describe_or_reject(self, path: Path) -> str | None:
        """Returns a human-readable problem, or None if the file is usable.

        Also updates the tooltip with what the file says about itself, so the
        user can confirm they picked the right model's output.
        """
        try:
            contours = ExternalContours(path)
        except FileNotFoundError:
            return f"{path} does not exist."
        except Exception as exc:
            return f"{path.name} could not be read as a contour file.\n\n{exc}"

        try:
            if self.n_frames is not None and contours.n_frames != self.n_frames:
                return (
                    f"{path.name} covers {contours.n_frames} frames but the video"
                    f" has {self.n_frames}.\n\nThese contours were probably"
                    " computed from a different clip."
                )
            if (
                self.video_size is not None
                and (contours.width, contours.height) != self.video_size
            ):
                return (
                    f"{path.name} was computed at {contours.width}x"
                    f"{contours.height} but the video is {self.video_size[0]}x"
                    f"{self.video_size[1]}."
                )

            described = "\n".join(
                f"{key}: {value}"
                for key, value in contours.attrs.items()
                if key not in ("format_version",)
            )
            self.file_label.setToolTip(described)
            logging.info("External contours selected: %s", contours)
            return None
        finally:
            contours.close()

    # ------------------------------------------------------------- app access
    def setValue(self, path: Path | str | None) -> None:
        """Restores the widget from a session or .toml file."""
        if path is None:
            self.clear_to_thresholding()
            return

        path = Path(path)
        problem = self.describe_or_reject(path)
        if problem:
            QMessageBox.warning(self, "External contours unavailable", problem)
            self.clear_to_thresholding()
            return

        self.path = path
        self.file_label.setText(path.name)
        self.file_label.setVisible(True)
        self.browse.setVisible(True)
        self.source.setCurrentText(EXTERNAL)
        self.valueChanged.emit(self.path)

    def value(self) -> Path | None:
        """The contour file tracking will use, or None.

        Detectron2 mode deliberately returns None: it is a preparation mode,
        and nothing about it belongs in the saved parameters.
        """
        if self.source.currentText() == EXTERNAL:
            return self.path
        return None

    def mode(self) -> str:
        return self.source.currentText()

    def uses_external_contours(self) -> bool:
        return self.value() is not None
