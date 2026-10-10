"""Checks that pre-labelled frames are something the dataset step can read.

The point of pre-labelling is to save annotation time, which it only does if
the files it writes are ordinary LabelMe annotations. So the test that matters
is the round trip: draw polygons, write them, and build a COCO dataset from the
result without the dataset step complaining.

A fake backend stands in for SAM 3, so none of this needs torch, sam3 or a GPU.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from idtrackerai.extra_tools.detectron2_pipeline.dataset import (
    DatasetRequest,
    build_coco_dataset,
    is_labelme,
)
from idtrackerai.extra_tools.detectron2_pipeline.prelabel import (
    labelme_document,
    prelabel,
    simplify,
)

HEIGHT, WIDTH = 120, 160


def ellipse_mask(cx, cy, a=20, b=11):
    mask = np.zeros((HEIGHT, WIDTH), np.uint8)
    cv2.ellipse(mask, (cx, cy), (a, b), 0, 0, 360, 1, -1)
    return mask


class FakePredictor:
    """Finds two animals in every frame, wherever the frame came from."""

    def __init__(self, masks=None):
        self.masks = masks
        self.description = "fake"

    def predict(self, frame):
        masks = self.masks if self.masks is not None else [
            ellipse_mask(45, 60),
            ellipse_mask(115, 60),
        ]
        return list(masks), [0.9] * len(masks)


@pytest.fixture
def frames(tmp_path):
    """Three sampled frames, named the way the sampling step names them."""
    folder = tmp_path / "frames"
    folder.mkdir()
    for index in (0, 30, 60):
        image = np.full((HEIGHT, WIDTH, 3), 90, np.uint8)
        cv2.imwrite(str(folder / f"clip_f{index:06d}.png"), image)
    # the sampling step leaves this beside the frames; it is not an annotation
    (folder / "sampling_manifest.json").write_text(
        json.dumps({"frames": [], "enhancement": {}}), encoding="utf-8"
    )
    return folder


def test_it_writes_one_annotation_per_frame(frames):
    summary = prelabel(frames, FakePredictor(), label="fish")

    assert summary["written"] == 3
    assert summary["polygons"] == 6
    for index in (0, 30, 60):
        assert (frames / f"clip_f{index:06d}.json").is_file()


def test_what_it_writes_looks_like_labelme(frames):
    prelabel(frames, FakePredictor(), label="fish")
    path = frames / "clip_f000000.json"

    assert is_labelme(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["imagePath"] == "clip_f000000.png"
    assert (data["imageWidth"], data["imageHeight"]) == (WIDTH, HEIGHT)
    assert len(data["shapes"]) == 2
    for shape in data["shapes"]:
        assert shape["shape_type"] == "polygon"
        assert shape["label"] == "fish"
        assert len(shape["points"]) >= 3


def test_machine_drawn_outlines_are_marked_as_such(frames):
    prelabel(frames, FakePredictor(), label="fish")
    data = json.loads((frames / "clip_f000000.json").read_text(encoding="utf-8"))

    assert data["flags"]["sam3_prelabel"] is True
    assert all(s["flags"]["sam3_prelabel"] for s in data["shapes"])


def test_corrected_annotations_are_never_clobbered(frames):
    """The whole point is saving hand-correction, so it must not destroy it."""
    corrected = frames / "clip_f000000.json"
    corrected.write_text(
        json.dumps({"shapes": [], "corrected": "by hand"}), encoding="utf-8"
    )

    summary = prelabel(frames, FakePredictor(), label="fish")

    assert summary["skipped_existing"] == 1
    assert summary["written"] == 2
    assert json.loads(corrected.read_text(encoding="utf-8"))["corrected"] == "by hand"


def test_overwrite_redraws_deliberately(frames):
    corrected = frames / "clip_f000000.json"
    corrected.write_text(json.dumps({"shapes": []}), encoding="utf-8")

    summary = prelabel(frames, FakePredictor(), label="fish", overwrite=True)

    assert summary["written"] == 3
    assert len(json.loads(corrected.read_text(encoding="utf-8"))["shapes"]) == 2


def test_a_frame_with_nothing_found_gets_no_file(frames):
    """An empty annotation would hide that the frame still needs drawing."""
    summary = prelabel(frames, FakePredictor(masks=[]), label="fish")

    assert summary["written"] == 0
    assert len(summary["empty"]) == 3
    assert not (frames / "clip_f000000.json").exists()


def test_the_manifest_is_not_mistaken_for_a_frame(frames):
    summary = prelabel(frames, FakePredictor(), label="fish")
    manifest = json.loads(
        (frames / "sampling_manifest.json").read_text(encoding="utf-8")
    )

    assert summary["frames"] == 3
    assert "enhancement" in manifest  # untouched


# ------------------------------------------------------------- simplification
def test_simplify_reduces_vertex_count():
    contour = np.array(
        [[int(60 + 40 * np.cos(t)), int(60 + 25 * np.sin(t))]
         for t in np.linspace(0, 2 * np.pi, 400, endpoint=False)],
        dtype=np.int32,
    )
    reduced = simplify(contour, 0.01)

    assert len(reduced) < len(contour)
    assert len(reduced) >= 3


def test_simplify_never_destroys_a_polygon():
    """Over-simplifying a small blob into a line would fail dataset validation."""
    tiny = np.array([[10, 10], [12, 10], [12, 12], [10, 12]], dtype=np.int32)
    reduced = simplify(tiny, 0.9)

    assert len(reduced) >= 3


def test_document_points_are_plain_floats():
    """json.dumps cannot serialise numpy scalars, and would fail at write time."""
    polygon = np.array([[1, 2], [30, 4], [5, 60]], dtype=np.int32)
    document = labelme_document(
        Path("f.png"), WIDTH, HEIGHT, [polygon], "fish"
    )
    encoded = json.dumps(document)

    assert json.loads(encoded)["shapes"][0]["points"][1] == [30.0, 4.0]


# ------------------------------------------------------ the round trip itself
def test_the_dataset_step_accepts_what_was_drawn(frames, tmp_path):
    """The one that matters: pre-labels build a dataset with no complaints."""
    prelabel(frames, FakePredictor(), label="fish")

    report = build_coco_dataset(
        DatasetRequest(input_dir=frames, output_dir=tmp_path / "dataset")
    )

    assert not report.problems, report.problems
    assert "fish" in report.label_counts
    assert report.train_annotations + report.val_annotations == 6
