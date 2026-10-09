"""Light-edit settings: schema defaults, range validation and reporting."""

import itertools

import cv2
import numpy as np
import pytest

from idtrackerai.extra_tools.detectron2_pipeline import preprocessing as fp


def test_defaults_are_no_change():
    for key in fp.LIGHT_EDIT_RANGES:
        assert key in fp.DEFAULT_SETTINGS
        assert key in fp.SETTING_KEYS
    assert fp.DEFAULT_SETTINGS["gamma"] == 1.0
    for key in fp.LIGHT_EDIT_RANGES:
        if key != "gamma":
            assert fp.DEFAULT_SETTINGS[key] == 0


def test_partial_dict_and_old_profile_are_completed():
    merged = fp.normalize_settings({"clahe_clip": 2.0})
    assert merged == {**fp.DEFAULT_SETTINGS, "clahe_clip": 2.0}


@pytest.mark.parametrize(
    "key, value",
    [("exposure", 3.1), ("gamma", 0.2), ("denoise", 11), ("sharpness", -1), ("blacks", 101)],
)
def test_out_of_range_is_rejected(key, value):
    with pytest.raises(fp.PreprocessingError, match=key):
        fp.normalize_settings({key: value})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
def test_nan_inf_and_bool_are_rejected(value):
    with pytest.raises(fp.PreprocessingError, match="exposure"):
        fp.normalize_settings({"exposure": value})


def test_integers_are_accepted_for_float_keys():
    merged = fp.normalize_settings({"exposure": 1, "gamma": 2})
    assert merged["exposure"] == 1
    assert merged["gamma"] == 2


def test_describe_lists_only_non_default_edits():
    text = fp.describe({**fp.DEFAULT_SETTINGS, "exposure": 0.5, "shadows": 20})
    assert "exposure=+0.5" in text
    assert "shadows=+20" in text
    assert "gamma" not in text
    base = fp.describe(fp.DEFAULT_SETTINGS)
    assert "=+" not in base
    assert base.endswith("downsample=4")


def test_missing_recorded_key_counts_as_default():
    recorded = {"clahe_clip": 1.5}
    warning = fp.check_settings_match(
        recorded, {**fp.DEFAULT_SETTINGS, "exposure": 1.0}, "the model"
    )
    assert warning is not None and "exposure" in warning
    assert fp.check_settings_match(recorded, dict(fp.DEFAULT_SETTINGS), "the model") is None


# ------------------------------------------------------------ image operations

ramp = np.tile(np.arange(256, dtype=np.uint8), (32, 1))
_rng_frame = np.random.default_rng(0).integers(0, 255, (64, 64), dtype=np.uint8)
_NEUTRAL = dict(
    exposure=0, brightness=0, contrast=0, gamma=1.0, shadows=0, highlights=0, blacks=0, whites=0
)


def _lut(**kw):
    return fp.tone_lut(**{**_NEUTRAL, **kw})


@pytest.mark.parametrize("frame", [_rng_frame, ramp])
def test_default_output_is_identical_to_the_old_pipeline(frame):
    old = fp.apply_clahe(fp.correct_illumination(fp.to_gray(frame), 4, 25.0), 1.5, 8)
    assert np.array_equal(fp.enhance_with_settings(frame, fp.DEFAULT_SETTINGS), old)
    off = fp.apply_clahe(fp.to_gray(frame), 1.5, 8)
    settings = {**fp.DEFAULT_SETTINGS, "correct_lighting": False}
    assert np.array_equal(fp.enhance_with_settings(frame, settings), off)


def test_default_lut_is_identity():
    assert np.array_equal(_lut(), np.arange(256))


@pytest.mark.parametrize(
    "kwargs, index, direction",
    [
        ({"exposure": 1}, 128, "up"),
        ({"exposure": -1}, 128, "down"),
        ({"brightness": 50}, 128, "up"),
        ({"brightness": -50}, 128, "down"),
        ({"gamma": 2.0}, 128, "up"),
        ({"gamma": 0.5}, 128, "down"),
        ({"contrast": 100}, 128, "same"),
        ({"contrast": 100}, 192, "up"),
        ({"contrast": 100}, 64, "down"),
        ({"contrast": -100}, 0, "grey"),
        ({"contrast": -100}, 255, "grey"),
        ({"shadows": 100}, 32, "up"),
        ({"shadows": 100}, 200, "same"),
        ({"highlights": -100}, 224, "down"),
        ({"highlights": -100}, 32, "same"),
        ({"blacks": 50}, 20, "zero"),
        ({"whites": 50}, 230, "full"),
    ],
)
def test_each_control_moves_the_ramp_the_right_way(kwargs, index, direction):
    out = int(_lut(**kwargs)[index])
    if direction == "up":
        assert out > index
    elif direction == "down":
        assert out < index
    elif direction == "same":
        assert abs(out - index) <= 1
    elif direction == "grey":
        assert abs(out - 128) <= 1
    elif direction == "zero":
        assert out == 0
    elif direction == "full":
        assert out == 255


def test_lut_is_monotonic_for_any_in_range_combination():
    ranges = fp.LIGHT_EDIT_RANGES
    names = list(_NEUTRAL)
    choices = [
        sorted({ranges[n][0], _NEUTRAL[n], ranges[n][1]}) for n in names
    ]
    for combo in itertools.product(*choices):
        lut = fp.tone_lut(*combo)
        assert np.all(np.diff(lut.astype(int)) >= 0), dict(zip(names, combo))


def test_extreme_levels_stay_finite():
    lut = _lut(blacks=100, whites=-100)
    assert lut.dtype == np.uint8 and lut.shape == (256,)
    assert np.all(np.diff(lut.astype(int)) >= 0)


def test_denoise_reduces_noise_and_sharpen_increases_edges():
    rng = np.random.default_rng(1)
    noisy = np.clip(100 + rng.normal(0, 10, (64, 64)), 0, 255).astype(np.uint8)
    assert fp.denoise(noisy, 8).std() < noisy.std()
    step = np.full((32, 32), 80, np.uint8)
    step[:, 16:] = 160
    assert np.abs(np.diff(fp.sharpen(step, 100).astype(int), axis=1)).max() > 80


def test_zero_strength_returns_the_input_values():
    assert np.array_equal(fp.denoise(_rng_frame, 0), _rng_frame)
    assert np.array_equal(fp.sharpen(_rng_frame, 0), _rng_frame)


def test_tiny_and_flat_frames_do_not_raise():
    maxed = {
        **fp.DEFAULT_SETTINGS,
        **{k: hi for k, (_, hi) in fp.LIGHT_EDIT_RANGES.items()},
    }
    for frame in (np.zeros((4, 4), np.uint8), np.full((40, 40), 90, np.uint8)):
        out = fp.enhance_with_settings(frame, maxed)
        assert out.dtype == np.uint8 and out.shape == frame.shape


def test_missing_keys_are_filled():
    a = fp.enhance_with_settings(_rng_frame, {"clahe_clip": 2.0})
    b = fp.enhance_with_settings(_rng_frame, {**fp.DEFAULT_SETTINGS, "clahe_clip": 2.0})
    assert np.array_equal(a, b)
