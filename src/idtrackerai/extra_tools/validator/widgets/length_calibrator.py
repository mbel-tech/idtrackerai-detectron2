from qtpy.QtCore import QPointF, Qt, Signal  # type: ignore[reportPrivateImportUsage]
from qtpy.QtGui import QColor, QColorConstants
from qtpy.QtWidgets import QInputDialog, QVBoxLayout, QWidget

from idtrackerai.GUI_tools import (
    AddBtn,
    CanvasMouseEvent,
    CanvasPainter,
    CustomList,
    LightPopUp,
    point_colors,
)
from idtrackerai.utils import LengthCalibration


class LengthCalibrator(QWidget):
    needToDraw = Signal()
    current_calibration: LengthCalibration | None = None
    preview_point: tuple[float, float] | None = None
    """Where the cursor is while the second calibration point is being placed."""

    def __init__(self) -> None:
        super().__init__()

        self.add = AddBtn()
        self.list = CustomList(max_n_row=0)

        layout = QVBoxLayout()
        layout.setSpacing(2)
        self.setLayout(layout)
        layout.addWidget(self.add)
        layout.addWidget(self.list)

        self.add.clicked.connect(self.add_clicked)
        self.calibrations: list[LengthCalibration] = []
        self.list.ListChanged.connect(self.needToDraw.emit)
        self.list.removedItemIndex.connect(self.remove_item)
        self.color_count = -1
        self.popup = LightPopUp()
        self.info_dialog_already_displayed: bool = False

    def click_event(self, event: CanvasMouseEvent) -> None:
        if (
            not self.isVisible()
            or not self.isEnabled()
            or self.current_calibration is None
        ):
            return

        if event.button == Qt.MouseButton.LeftButton:
            self.current_calibration.add_point(event.int_xy_data)
            if self.current_calibration.has_two_points():
                self.end_calibration()

    def move_event(self, event: CanvasMouseEvent) -> None:
        """Draws the calibration line as the cursor moves.

        Placing two points blind and only then seeing the line means redoing the
        calibration whenever it lands off the mark. Following the cursor makes
        the second click a confirmation rather than a guess.
        """
        if (
            not self.isVisible()
            or not self.isEnabled()
            or self.current_calibration is None
            or self.current_calibration.point_A is None
        ):
            return

        self.preview_point = event.xy_data
        self.needToDraw.emit()

    def end_calibration(self) -> None:
        if self.current_calibration is None:
            return

        self.preview_point = None
        self.needToDraw.emit()

        # A positive minimum: the distance is a denominator in
        # LengthCalibration.value(), so zero would be a division by zero later.
        # Positional arguments: PyQt6 names the bounds min/max and PySide6 names
        # them minValue/maxValue, and qtpy does not paper over the difference.
        distance, ok = QInputDialog.getDouble(
            self,
            "idtracker.ai",
            "Enter real distance between the two points:",
            1.0,  # value
            1e-9,  # minimum; the distance is a denominator in value()
            2147483647.0,  # maximum, Qt's own default
            4,  # decimals
        )
        if not ok:
            self.current_calibration = None
            return

        self.current_calibration.distance = distance
        self.calibrations.append(self.current_calibration)
        self.list.add_str(
            str(self.current_calibration), color=QColor(self.current_calibration.color)
        )
        self.current_calibration = None

    def add_clicked(self) -> None:
        if not self.info_dialog_already_displayed:
            self.popup.info(
                "Length Calibration",
                "Indicate two points by clicking in the video and then enter the"
                " numeric value of the distance between these two. The average factor"
                " of all calibrations will be included in the trajectory files.",
            )
            self.info_dialog_already_displayed = True

        self.color_count = (
            0 if self.color_count == len(point_colors) - 1 else self.color_count + 1
        )
        self.current_calibration = LengthCalibration(point_colors[self.color_count])
        self.needToDraw.emit()

    def remove_item(self, item_index: int) -> None:
        """Removes the calibration shown on row `item_index`.

        Matching on `str(calibration)` instead, as this used to, deletes the
        first calibration with the same points and distance rather than the one
        whose row was clicked.
        """
        if 0 <= item_index < len(self.calibrations):
            self.calibrations.pop(item_index)
            self.needToDraw.emit()

    def load(self, calibrations: list[LengthCalibration] | None) -> None:
        self.list.clear()
        # The widget list and self.calibrations are indexed together now, so
        # they have to be cleared together.
        self.calibrations = []
        if calibrations is None:
            return

        for calibration in calibrations:
            self.color_count = (
                0 if self.color_count == len(point_colors) - 1 else self.color_count + 1
            )
            calibration.color = point_colors[self.color_count]
            self.calibrations.append(calibration)
            self.list.add_str(str(calibration), color=QColor(calibration.color))
        self.current_calibration = None

    def get_calibrations(self) -> list[LengthCalibration]:
        return [c for c in self.calibrations if c.completed()]

    def paint_on_canvas(self, painter: CanvasPainter) -> None:
        painter.setPen(QColorConstants.Black)
        for calibration in self.calibrations:
            assert calibration.point_A is not None
            assert calibration.point_B is not None
            painter.setPen(QColor(calibration.color))
            painter.drawLine(*(calibration.point_A + calibration.point_B))  # type: ignore
            painter.setPen(QColorConstants.Black)
            painter.drawText(
                QPointF(
                    0.5 * (calibration.point_A[0] + calibration.point_B[0]),
                    0.5 * (calibration.point_A[1] + calibration.point_B[1]),
                ),
                f"{calibration.distance}",
            )
            painter.setBrush(QColor(calibration.color))
            painter.drawBigPoint(*calibration.point_A)
            painter.drawBigPoint(*calibration.point_B)

        calibration = self.current_calibration
        if calibration is None:
            return
        painter.setBrush(QColor(calibration.color))

        if calibration.point_A is not None:
            painter.drawBigPoint(*calibration.point_A)

            if self.preview_point is not None:
                painter.setPen(QColor(calibration.color))
                painter.drawLine(
                    QPointF(*calibration.point_A), QPointF(*self.preview_point)
                )

        if calibration.point_B is not None:
            painter.drawBigPoint(*calibration.point_B)
