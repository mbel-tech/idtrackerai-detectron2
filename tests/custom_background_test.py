"""A background the apps saved must not be enhanced a second time."""

import json

import cv2
import numpy as np
import pytest

from idtrackerai.base.animals_detection.segmentation import (
    apply_enhancement,
    load_custom_background,
)
from idtrackerai.utils import IdtrackeraiError

ENHANCEMENT = {
    "enhance": True,
    "clahe_clip": 3.0,
    "clahe_tile": 8,
    "illumination_sigma": 25.0,
    "illumination_downsample": 4,
    "correct_lighting": True,
}


@pytest.fixture
def raw(tmp_path):
    rng = np.random.default_rng(0)
    image = (rng.random((64, 64)) * 120 + 60).astype(np.uint8)
    path = tmp_path / "raw.png"
    cv2.imwrite(str(path), image)
    return image, path


def test_enhancement_is_not_idempotent():
    """The reason a saved, already enhanced background has to be recognised."""
    rng = np.random.default_rng(1)
    image = (rng.random((64, 64)) * 120 + 60).astype(np.uint8)
    once = apply_enhancement(image, ENHANCEMENT)
    assert not np.array_equal(once, apply_enhancement(once, ENHANCEMENT))


def test_raw_image_without_sidecar_is_enhanced(raw):
    image, path = raw
    loaded = load_custom_background(str(path), enhancement=ENHANCEMENT)
    assert np.array_equal(loaded, apply_enhancement(image, ENHANCEMENT))


def test_no_enhancement_leaves_the_image_alone(raw):
    image, path = raw
    assert np.array_equal(load_custom_background(str(path)), image)


def test_saved_enhanced_background_is_not_enhanced_again(raw, tmp_path):
    image, _ = raw
    enhanced = apply_enhancement(image, ENHANCEMENT)
    path = tmp_path / "saved.png"
    cv2.imwrite(str(path), enhanced)
    path.with_suffix(".enhancement.json").write_text(
        json.dumps({"enhancement": ENHANCEMENT})
    )
    loaded = load_custom_background(str(path), enhancement=ENHANCEMENT)
    assert np.array_equal(loaded, enhanced)


def test_background_saved_with_other_settings_is_refused(raw, tmp_path):
    image, _ = raw
    path = tmp_path / "saved.png"
    cv2.imwrite(str(path), apply_enhancement(image, ENHANCEMENT))
    path.with_suffix(".enhancement.json").write_text(
        json.dumps({"enhancement": {**ENHANCEMENT, "clahe_clip": 1.0}})
    )
    with pytest.raises(IdtrackeraiError):
        load_custom_background(str(path), enhancement=ENHANCEMENT)
