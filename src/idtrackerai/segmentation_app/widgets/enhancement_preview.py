"""Draws the enhanced frame over the video player, live.

This is a painter, not a control. It is a ``QObject`` rather than a ``QWidget``
and never appears in a layout: it exists only to listen to the player's
``painting_time`` signal, exactly as :class:`FrameAnalyzer` does for blob
polygons. Keeping it separate from the controls means the controls can be
tested without a canvas, and this can be tested with a bare ``QPainter`` on a
``QImage`` and no widget at all.

It paints over the frame the player has already drawn rather than setting
``VideoPlayer.is_muted``. Muting would make this object responsible for
supplying the background for every other painter on that signal, and any path
that hides the panel without unmuting would leave a black canvas. Overdrawing
is self-correcting: stop painting and the raw frame is back.
"""

import logging

import numpy as np
from qtpy.QtCore import QObject, QPointF, QRectF, Signal  # type: ignore[reportPrivateImportUsage]
from qtpy.QtGui import QImage, QPixmap

from idtrackerai.extra_tools.detectron2_pipeline import preprocessing as fp

# Matches VideoPlayer.video_drawing_origin, so the enhanced image lands exactly
# on top of the frame the player drew.
DRAWING_ORIGIN = QPointF(-0.5, -0.5)


class EnhancementPreview(QObject):
    """Overdraws the current frame with its enhanced version."""

    failed = Signal(str)

    def __init__(self):
        super().__init__()
        self.settings: dict = dict(fp.DEFAULT_SETTINGS)
        self.enabled = False
        self.split = 1.0  # 1.0 = fully enhanced, 0.0 = fully raw
        self._generation = 0
        self._settings_key: tuple = ()
        self._key: tuple | None = None
        self._pixmap: QPixmap | None = None
        self._refresh_settings_key()

    # ----------------------------------------------------------------- inputs
    def set_settings(self, settings: dict) -> None:
        self.settings = dict(settings)
        self._refresh_settings_key()
        self._invalidate()

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)
        if enabled:
            # a previous failure disabled this; give it another chance
            self._invalidate()

    def set_split(self, split: float) -> None:
        self.split = max(0.0, min(1.0, float(split)))

    def bump_generation(self) -> None:
        """Invalidate the cache because the frame numbering changed.

        Reordering video paths changes which image a frame number refers to,
        and the cache key alone cannot see that.
        """
        self._generation += 1
        self._invalidate()

    def _refresh_settings_key(self) -> None:
        self._settings_key = tuple(sorted(self.settings.items()))

    def _invalidate(self) -> None:
        self._key = None
        self._pixmap = None

    # ---------------------------------------------------------------- painting
    def paint_on_canvas(self, painter, frame_number: int, frame) -> None:
        if not self.enabled or frame is None:
            return
        if not self.settings.get("enhance", True) or self.split <= 0.0:
            return  # the player's raw frame is already what we want to show

        try:
            self._ensure_pixmap(frame_number, frame)
            if self._pixmap is None:
                return
            painter.save()
            if self.split < 1.0:
                painter.setClipRect(
                    QRectF(
                        DRAWING_ORIGIN.x(),
                        DRAWING_ORIGIN.y(),
                        self._pixmap.width() * self.split,
                        self._pixmap.height(),
                    )
                )
            painter.drawPixmap(DRAWING_ORIGIN, self._pixmap)
            painter.restore()
        except Exception as exc:  # noqa: BLE001
            # Canvas.paintEvent catches Exception and only logs the message, so
            # a bug here would show as a blank or stale frame with no traceback.
            # Disable so the log is not spammed once per repaint, and tell the
            # panel so it can say something visible.
            logging.exception("Enhancement preview failed")
            self.enabled = False
            self._invalidate()
            self.failed.emit(str(exc))

    def _ensure_pixmap(self, frame_number: int, frame: np.ndarray) -> None:
        # frame.shape matters because toggling the player's "Enable color"
        # switches the same frame number between 2D grayscale and 3D BGR.
        key = (self._generation, frame_number, frame.shape, self._settings_key)
        if key == self._key and self._pixmap is not None:
            return

        enhanced = fp.enhance(
            frame,
            clahe_clip=self.settings["clahe_clip"],
            clahe_tile=self.settings["clahe_tile"],
            downsample=self.settings["illumination_downsample"],
            sigma=self.settings["illumination_sigma"],
            correct_lighting=self.settings["correct_lighting"],
        )
        if enhanced is frame:
            # Frames come from an lru_cache and are shared; handing the very
            # same buffer to a QImage that does not copy would be a hazard.
            enhanced = enhanced.copy()
        enhanced = np.ascontiguousarray(enhanced)

        height, width = enhanced.shape
        # QImage does not copy; QPixmap.fromImage does, so building both in one
        # expression keeps the buffer alive exactly as long as it is needed.
        self._pixmap = QPixmap.fromImage(
            QImage(enhanced.data, width, height, width, QImage.Format.Format_Grayscale8)
        )
        self._key = key
