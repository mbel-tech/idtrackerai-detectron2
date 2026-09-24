"""The SAM 3 panel's wiring, checked without a display or a GPU.

These are not tests of how the panel looks. They pin the two things that are
easy to break silently and expensive to discover late: that the controls stay
disabled until a run would actually work, and that every control the panel asks
for a tooltip by name actually has one.
"""

import pytest
import toml

pytest.importorskip("qtpy", reason="the Segmentation App needs Qt")

from pathlib import Path  # noqa: E402

TOOLTIPS = (
    Path(__file__).resolve().parents[1]
    / "src/idtrackerai/segmentation_app/tooltips.toml"
)


@pytest.fixture(scope="module")
def qt_app():
    """One offscreen QApplication for the module."""
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def panel(qt_app):
    from idtrackerai.segmentation_app.widgets import EnhancementWidget, Sam3Panel

    return Sam3Panel(EnhancementWidget())


def test_the_selector_offers_sam3(qt_app):
    from idtrackerai.segmentation_app.widgets.segmentation_source import (
        DETECTRON2,
        EXTERNAL,
        SAM3,
        THRESHOLDING,
        SegmentationSourceWidget,
    )

    widget = SegmentationSourceWidget()
    modes = [widget.source.itemText(i) for i in range(widget.source.count())]

    assert modes == [THRESHOLDING, EXTERNAL, DETECTRON2, SAM3]


def test_choosing_sam3_does_not_open_a_file_dialog(qt_app):
    """SAM 3 mode produces contours; it must not demand one up front.

    If this regressed, selecting the mode would block on a modal dialog, which
    in an offscreen test means hanging rather than failing.
    """
    from idtrackerai.segmentation_app.widgets.segmentation_source import (
        SAM3,
        SegmentationSourceWidget,
    )

    widget = SegmentationSourceWidget()
    emitted = []
    widget.valueChanged.connect(emitted.append)

    widget.source.setCurrentText(SAM3)

    assert widget.mode() == SAM3
    assert emitted == [None]
    assert widget.paths == []


def test_nothing_runs_before_a_checkpoint_is_chosen(panel):
    assert not panel.prelabel_button.isEnabled()
    assert not panel.export_button.isEnabled()


def test_export_stays_disabled_without_a_gpu(panel, tmp_path):
    """A whole video on CPU is not slow, it is impractical. Do not offer it."""
    panel.weights = tmp_path / "sam3.pt"
    panel.set_video_context([tmp_path / "clip.mp4"])

    assert panel.export_button.isEnabled() is panel.has_gpu


def test_drafting_needs_a_frames_folder(panel, tmp_path):
    panel.weights = tmp_path / "sam3.pt"
    panel._refresh_enabled()
    assert not panel.prelabel_button.isEnabled()

    panel.frames_dir = tmp_path
    panel._refresh_enabled()
    assert panel.prelabel_button.isEnabled()


def test_an_empty_prompt_blocks_everything(panel, tmp_path):
    panel.weights = tmp_path / "sam3.pt"
    panel.frames_dir = tmp_path
    panel.prompt.setText("   ")
    panel._refresh_enabled()

    assert not panel.prelabel_button.isEnabled()


def test_the_colab_route_is_offered_exactly_when_it_is_needed(panel):
    assert panel.colab_hint.isVisibleTo(panel) is not panel.has_gpu


def test_every_control_it_labels_has_a_tooltip(panel):
    """setToolTips skips missing keys silently, so check they are all there."""
    tips = toml.load(TOOLTIPS)
    wanted = {
        "sam3_weights",
        "sam3_prompt",
        "sam3_score_threshold",
        "sam3_max_instances",
        "sam3_frames",
        "sam3_prelabel",
        "sam3_overwrite",
        "sam3_export",
    }

    assert wanted <= set(tips)
    panel.setToolTips(tips)
    assert panel.prompt.toolTip()
    assert panel.export_button.toolTip()
