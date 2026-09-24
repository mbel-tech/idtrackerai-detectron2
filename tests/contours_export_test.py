"""Exercises the contour exporter with a stand-in model.

The sidecar *format* is covered by ``external_contours_test.py``. What was not
covered is the exporter that writes it: the loop that reads a video, asks a
model for masks, cleans them, applies the overlap policy and turns them into
polygons. That loop is now shared by Detectron2 and SAM 3, so a bug in it is a
bug in both.

A fake backend stands in for the model, which keeps these tests free of torch,
detectron2, sam3 and any GPU, and lets each one state exactly which masks the
model returned.
"""

import json
from argparse import Namespace

import cv2
import h5py
import numpy as np
import pytest

from idtrackerai.base.animals_detection.external_contours import (
    ExternalContours,
    write_contours,
)
from idtrackerai.extra_tools.detectron2_pipeline.inference import export_video
from idtrackerai.extra_tools.detectron2_pipeline.sam3_predictor import _as_binary_mask

HEIGHT, WIDTH = 120, 160
N_FRAMES = 6


def ellipse_mask(cx, cy, a=14, b=7):
    mask = np.zeros((HEIGHT, WIDTH), np.uint8)
    cv2.ellipse(mask, (cx, cy), (a, b), 0, 0, 360, 1, -1)
    return mask


class FakePredictor:
    """Returns pre-arranged masks, one list per frame, ignoring the pixels."""

    def __init__(self, masks_per_frame, scores_per_frame=None):
        self.masks_per_frame = masks_per_frame
        self.scores_per_frame = scores_per_frame
        self.description = "fake"
        self.seen = 0

    def predict(self, frame):
        index = self.seen
        self.seen += 1
        masks = self.masks_per_frame[index]
        if self.scores_per_frame is None:
            scores = [0.9] * len(masks)
        else:
            scores = self.scores_per_frame[index]
        return list(masks), list(scores)


@pytest.fixture
def video(tmp_path):
    """A short readable video. Its pixels do not matter; its length does."""
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (WIDTH, HEIGHT)
    )
    if not writer.isOpened():
        pytest.skip("no MJPG encoder available in this OpenCV build")
    for i in range(N_FRAMES):
        frame = np.full((HEIGHT, WIDTH, 3), 40 + i, np.uint8)
        writer.write(frame)
    writer.release()

    capture = cv2.VideoCapture(str(path))
    readable = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if readable != N_FRAMES:
        pytest.skip(f"video round-tripped as {readable} frames, not {N_FRAMES}")
    return path


def make_args(**overrides):
    """The exporter's knobs, at the defaults its argument parser uses."""
    args = Namespace(
        backend="detectron2",
        prompt=None,
        weights="model_final.pth",
        score_threshold=0.7,
        max_instances=5,
        min_component=0,
        dilate=0,
        dedup_iou=0.7,
        min_area=1.0,
        on_overlap="merge",
        merge_overlap=0.15,
        limit=0,
        local_cache=None,
        keep_cache=False,
        progress_every=1000,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def run_export(video, tmp_path, masks_per_frame, scores_per_frame=None, **overrides):
    output = tmp_path / "contours.h5"
    predictor = FakePredictor(masks_per_frame, scores_per_frame)
    args = make_args(**overrides)
    stats = export_video(
        video,
        output,
        args,
        predictor,
        predictor.description,
        lambda frame: frame,
        {"enhance": False},
        write_contours,
    )
    return output, stats


def test_every_frame_is_represented(video, tmp_path):
    """Two animals per frame arrive as two contours per frame."""
    masks = [
        [ellipse_mask(40, 60), ellipse_mask(120, 60)] for _ in range(N_FRAMES)
    ]
    output, stats = run_export(video, tmp_path, masks)

    contours = ExternalContours(output)
    assert contours.n_frames == N_FRAMES
    assert (contours.width, contours.height) == (WIDTH, HEIGHT)
    for frame in range(N_FRAMES):
        assert len(contours.contours_in_frame(frame)) == 2

    assert stats["frames"] == N_FRAMES
    assert stats["contours"] == 2 * N_FRAMES
    assert stats["empty_frames"] == 0


def test_a_frame_with_no_detection_stays_in_place(video, tmp_path):
    """A blank frame must keep its slot, or every later frame shifts."""
    masks = [[ellipse_mask(40, 60)] for _ in range(N_FRAMES)]
    masks[3] = []
    output, stats = run_export(video, tmp_path, masks)

    contours = ExternalContours(output)
    assert contours.n_frames == N_FRAMES
    assert contours.contours_in_frame(3) == [] or len(
        contours.contours_in_frame(3)
    ) == 0
    assert len(contours.contours_in_frame(4)) == 1
    assert stats["empty_frames"] == 1


def test_overlapping_animals_merge_into_one_blob(video, tmp_path):
    """The default policy hands crossings to idtracker.ai as a single blob."""
    overlapping = [ellipse_mask(80, 60), ellipse_mask(84, 60)]
    masks = [list(overlapping) for _ in range(N_FRAMES)]
    output, _ = run_export(video, tmp_path, masks, dedup_iou=1.0)

    contours = ExternalContours(output)
    assert len(contours.contours_in_frame(0)) == 1


def test_split_policy_keeps_animals_apart(video, tmp_path):
    """--on-overlap split resolves the occlusion instead."""
    overlapping = [ellipse_mask(80, 60), ellipse_mask(84, 60)]
    masks = [list(overlapping) for _ in range(N_FRAMES)]
    output, _ = run_export(
        video, tmp_path, masks, on_overlap="split", dedup_iou=1.0
    )

    contours = ExternalContours(output)
    assert len(contours.contours_in_frame(0)) == 2


def test_tiny_detections_are_dropped(video, tmp_path):
    """min_area filters specks that would otherwise become blobs."""
    masks = [[ellipse_mask(40, 60), ellipse_mask(120, 60, a=1, b=1)] for _ in range(N_FRAMES)]
    output, _ = run_export(video, tmp_path, masks, min_area=50.0)

    contours = ExternalContours(output)
    assert len(contours.contours_in_frame(0)) == 1


def test_the_sidecar_records_which_model_made_it(video, tmp_path):
    """A contour file has to say where it came from, now two things can write one."""
    masks = [[ellipse_mask(40, 60)] for _ in range(N_FRAMES)]
    output, _ = run_export(
        video, tmp_path, masks, backend="sam3", prompt="fish", weights="sam3.pt"
    )

    with h5py.File(output, "r") as handle:
        attrs = dict(handle.attrs)

    assert attrs["backend"] == "sam3"
    assert attrs["prompt"] == "fish"
    assert attrs["model"] == "fake"
    assert json.loads(attrs["enhancement"]) == {"enhance": False}


def test_detectron2_does_not_claim_a_prompt(video, tmp_path):
    """The prompt attribute belongs to SAM 3 and must not appear otherwise."""
    masks = [[ellipse_mask(40, 60)] for _ in range(N_FRAMES)]
    output, _ = run_export(video, tmp_path, masks)

    with h5py.File(output, "r") as handle:
        attrs = dict(handle.attrs)

    assert attrs["backend"] == "detectron2"
    assert "prompt" not in attrs


def test_a_partial_file_is_not_left_behind(video, tmp_path):
    """The .partial is renamed into place, never left for resume logic to find."""
    masks = [[ellipse_mask(40, 60)] for _ in range(N_FRAMES)]
    output, _ = run_export(video, tmp_path, masks)

    assert output.is_file()
    assert not output.with_suffix(output.suffix + ".partial").exists()


# ------------------------------------------------- SAM 3 mask normalisation
# SAM 3 hands back masks with a leading channel axis, and sometimes as
# probabilities rather than booleans. clean_mask downstream assumes a plain
# binary image, so this is where that assumption is made true.


def test_a_channel_axis_is_removed():
    mask = np.zeros((1, HEIGHT, WIDTH), np.uint8)
    mask[0, 10:20, 10:20] = 1
    out = _as_binary_mask(mask)
    assert out.shape == (HEIGHT, WIDTH)
    assert out.sum() == 100


def test_probabilities_become_binary():
    mask = np.zeros((HEIGHT, WIDTH), np.float32)
    mask[10:20, 10:20] = 0.9
    mask[30:40, 30:40] = 0.2
    out = _as_binary_mask(mask)
    assert out.dtype == np.uint8
    assert set(np.unique(out)) <= {0, 1}
    assert out.sum() == 100


def test_an_already_binary_mask_is_unchanged():
    mask = np.zeros((HEIGHT, WIDTH), np.uint8)
    mask[5:15, 5:15] = 1
    out = _as_binary_mask(mask)
    assert out.dtype == np.uint8
    assert out.sum() == 100
