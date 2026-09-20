import cv2
import h5py
import numpy as np
import pytest

from idtrackerai import Blob
from idtrackerai.base.animals_detection.external_contours import (
    FORMAT_VERSION,
    ExternalContours,
    validate_against_video,
    write_contours,
)
from idtrackerai.base.animals_detection.segmentation import process_frame

HEIGHT, WIDTH = 240, 320


def ellipse_mask(cx, cy, a=18, b=7, angle=0):
    mask = np.zeros((HEIGHT, WIDTH), np.uint8)
    cv2.ellipse(mask, (cx, cy), (a, b), angle, 0, 360, 255, -1)
    return mask


def contours_of(mask):
    found, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    return [c[:, 0] for c in found]


@pytest.fixture
def sidecar(tmp_path):
    """Five animals across twelve frames, with a dropout on frame 7."""
    contours_per_frame = []
    for frame in range(12):
        n_animals = 3 if frame == 7 else 5
        frame_contours = []
        for i in range(n_animals):
            mask = ellipse_mask(40 + i * 55 + frame * 2, 60 + (i % 3) * 60)
            frame_contours += contours_of(mask)
        contours_per_frame.append(frame_contours)

    path = write_contours(
        tmp_path / "contours.h5",
        contours_per_frame,
        width=WIDTH,
        height=HEIGHT,
        source="test",
    )
    return path, contours_per_frame


def test_round_trip_is_lossless(sidecar):
    path, expected = sidecar
    contours = ExternalContours(path)

    assert contours.n_frames == len(expected)
    assert (contours.width, contours.height) == (WIDTH, HEIGHT)

    for frame, expected_contours in enumerate(expected):
        got = contours.contours_in_frame(frame)
        assert len(got) == len(expected_contours)
        for a, b in zip(got, expected_contours):
            assert a.dtype == np.int32
            assert np.array_equal(a, b)


def test_frames_outside_the_video_are_empty_not_an_error(sidecar):
    contours = ExternalContours(sidecar[0])
    assert contours.contours_in_frame(-1) == []
    assert contours.contours_in_frame(10_000) == []


def test_blob_features_come_from_the_contour(sidecar):
    """Every blob attribute is derived, so a contour is all that is needed."""
    contours = ExternalContours(sidecar[0])
    contour = contours.contours_in_frame(0)[0]
    blob = Blob(contour, frame_number=0)

    assert blob.centroid[0] == pytest.approx(40, abs=2)
    assert blob.centroid[1] == pytest.approx(60, abs=2)
    assert blob.area > 0
    assert blob.extension > 0
    assert blob.bbox_corners.bottom < blob.bbox_corners.top
    assert blob.bbox_corners.left < blob.bbox_corners.right
    assert blob.get_bbox_mask().any()


def test_process_frame_reads_the_sidecar(sidecar):
    path, expected = sidecar
    frame = np.full((HEIGHT, WIDTH), 30, np.uint8)

    areas, contours, returned = process_frame(
        frame,
        intensity_ths=[0, 255],
        area_ths=[1, np.inf],
        external_contours=path,
        frame_number=0,
    )

    assert len(contours) == len(expected[0])
    assert len(areas) == len(contours)
    assert returned is frame, "the frame must pass through for bbox images"


def test_dropout_frames_pass_through_untouched(sidecar):
    """Missing detections stay missing; crossing detection deals with them."""
    path, _ = sidecar
    _, contours, _ = process_frame(
        np.zeros((HEIGHT, WIDTH), np.uint8),
        intensity_ths=[0, 255],
        area_ths=[1, np.inf],
        external_contours=path,
        frame_number=7,
    )
    assert len(contours) == 3


def test_area_threshold_still_filters(sidecar):
    path, _ = sidecar
    _, contours, _ = process_frame(
        np.zeros((HEIGHT, WIDTH), np.uint8),
        intensity_ths=[0, 255],
        area_ths=[10_000, np.inf],
        external_contours=path,
        frame_number=0,
    )
    assert contours == []


def test_roi_drops_whole_contours(sidecar):
    path, _ = sidecar
    roi = np.zeros((HEIGHT, WIDTH), np.uint8)
    roi[:, :160] = 1

    _, contours, _ = process_frame(
        np.zeros((HEIGHT, WIDTH), np.uint8),
        intensity_ths=[0, 255],
        area_ths=[1, np.inf],
        ROI_mask=roi,
        external_contours=path,
        frame_number=0,
    )
    assert len(contours) == 3
    for contour in contours:
        assert contour[:, 0].mean() < 160


def test_missing_frame_number_is_rejected(sidecar):
    from idtrackerai import IdtrackeraiError

    with pytest.raises(IdtrackeraiError, match="frame number"):
        process_frame(
            np.zeros((HEIGHT, WIDTH), np.uint8),
            intensity_ths=[0, 255],
            area_ths=[1, np.inf],
            external_contours=sidecar[0],
        )


def test_mismatched_video_is_caught_before_tracking(sidecar):
    path, expected = sidecar
    validate_against_video(path, len(expected), WIDTH, HEIGHT)

    with pytest.raises(ValueError, match="frames"):
        validate_against_video(path, len(expected) + 1, WIDTH, HEIGHT)

    with pytest.raises(ValueError, match="1920x1080"):
        validate_against_video(path, len(expected), 1920, 1080)


def test_wrong_format_version_is_rejected(tmp_path):
    path = tmp_path / "bad.h5"
    with h5py.File(path, "w") as file:
        file.create_dataset("vertices", data=np.zeros((1, 2), np.int32))
        file.create_dataset("vertex_offsets", data=np.zeros(2, np.int64))
        file.create_dataset("frame_offsets", data=np.zeros(2, np.int64))
        file.attrs.update(
            format_version=FORMAT_VERSION + 1, n_frames=1, width=WIDTH, height=HEIGHT
        )

    with pytest.raises(ValueError, match="format version"):
        ExternalContours(path)


def test_missing_file_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        ExternalContours(tmp_path / "absent.h5")


def test_empty_frames_survive(tmp_path):
    path = write_contours(tmp_path / "empty.h5", [[], [], []], width=WIDTH, height=HEIGHT)
    contours = ExternalContours(path)
    assert all(contours.contours_in_frame(i) == [] for i in range(3))


def test_opencv_shaped_contours_are_accepted(tmp_path):
    """write_contours should take (n, 1, 2) straight from cv2.findContours."""
    found, _ = cv2.findContours(
        ellipse_mask(100, 100), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS
    )
    assert found[0].ndim == 3

    path = write_contours(tmp_path / "cv.h5", [list(found)], width=WIDTH, height=HEIGHT)
    contour = ExternalContours(path).contours_in_frame(0)[0]
    assert contour.ndim == 2 and contour.shape[1] == 2


def test_malformed_contour_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="expected"):
        write_contours(
            tmp_path / "bad.h5",
            [[np.zeros((5, 3), np.int32)]],
            width=WIDTH,
            height=HEIGHT,
        )
