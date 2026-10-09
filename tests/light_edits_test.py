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
    [
        ("exposure", 3.1),
        ("gamma", 0.2),
        ("denoise", 11),
        ("sharpness", -1),
        ("blacks", 101),
    ],
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
    assert (
        fp.check_settings_match(recorded, dict(fp.DEFAULT_SETTINGS), "the model")
        is None
    )


# ------------------------------------------------------------ image operations

ramp = np.tile(np.arange(256, dtype=np.uint8), (32, 1))
_rng_frame = np.random.default_rng(0).integers(0, 255, (64, 64), dtype=np.uint8)
_NEUTRAL = dict(
    exposure=0,
    brightness=0,
    contrast=0,
    gamma=1.0,
    shadows=0,
    highlights=0,
    blacks=0,
    whites=0,
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
        ({"blacks": -50}, 20, "zero"),
        ({"blacks": 50}, 0, "up"),
        ({"blacks": 50}, 255, "same"),
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
    choices = [sorted({ranges[n][0], _NEUTRAL[n], ranges[n][1]}) for n in names]
    for combo in itertools.product(*choices):
        lut = fp.tone_lut(*combo)
        assert np.all(np.diff(lut.astype(int)) >= 0), dict(zip(names, combo))


def test_extreme_levels_stay_finite():
    # the black point meets the white point: the guard keeps it finite
    for kwargs in ({"blacks": -100, "whites": 100}, {"blacks": 100, "whites": -100}):
        lut = _lut(**kwargs)
        assert lut.dtype == np.uint8 and lut.shape == (256,)
        assert np.all(np.diff(lut.astype(int)) >= 0)


def test_positive_blacks_lifts_and_stays_monotonic():
    lut = _lut(blacks=50)
    assert lut[0] > 0
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


# ------------------------------------------------------------- call sites
import argparse
import json

from idtrackerai.base.animals_detection import segmentation
from idtrackerai.extra_tools.detectron2_pipeline import sampling
from idtrackerai.extra_tools.detectron2_pipeline.errors import (
    PreprocessingError,
    SamplingError,
)

_OLD_KEYS = (
    "enhance",
    "clahe_clip",
    "clahe_tile",
    "illumination_sigma",
    "illumination_downsample",
    "correct_lighting",
)


def test_all_call_sites_agree():
    variants = [
        {
            "clahe_clip": 2.0,
            "exposure": 0.7,
            "shadows": 30,
            "sharpness": 40,
            "denoise": 3,
        },
        {
            "clahe_clip": 2.0,
            "exposure": 0.7,
            "gamma": 1.4,
            "correct_lighting": True,
            "illumination_downsample": 2,
            "illumination_sigma": 10.0,
        },
        {
            "clahe_clip": 3.0,
            "blacks": 20,
            "correct_lighting": False,
            "illumination_downsample": 8,
            "illumination_sigma": 40.0,
        },
    ]
    for extra in variants:
        settings = {**fp.DEFAULT_SETTINGS, **extra}
        want = fp.enhance_with_settings(_rng_frame, settings)
        assert np.array_equal(
            segmentation.apply_enhancement(_rng_frame, settings), want
        )
        assert np.array_equal(
            fp.make_enhancer_from(settings)(_rng_frame)[:, :, 0], want
        )


def _parse(argv):
    parser = argparse.ArgumentParser()
    fp.add_arguments(parser)
    return parser.parse_args(argv)


def test_cli_flags_reach_the_settings():
    settings = fp.resolve_settings(_parse(["--exposure", "0.5", "--gamma", "1.4"]))[0]
    assert settings["exposure"] == 0.5 and settings["gamma"] == 1.4
    assert settings["shadows"] == 0
    assert fp.resolve_settings(_parse([]))[0] == fp.DEFAULT_SETTINGS


def test_cli_flag_out_of_range_is_rejected():
    with pytest.raises(PreprocessingError):
        fp.resolve_settings(_parse(["--denoise", "50"]))


def test_profile_round_trip_with_light_edits(tmp_path):
    settings = {
        **fp.DEFAULT_SETTINGS,
        "exposure": 0.5,
        "gamma": 1.4,
        "shadows": 30,
        "denoise": 3,
    }
    path = fp.save_profile(tmp_path / "p.json", settings)
    assert fp.load_profile(path) == settings

    old = tmp_path / "old.json"
    old.write_text(json.dumps({k: fp.DEFAULT_SETTINGS[k] for k in _OLD_KEYS}))
    assert fp.normalize_settings(fp.load_profile(old)) == fp.DEFAULT_SETTINGS


def _record(folder, **overrides):
    folder.mkdir(exist_ok=True)
    settings = {**fp.DEFAULT_SETTINGS, **overrides}
    (folder / "preprocess_profile.json").write_text(json.dumps(settings))


def test_sampling_refuses_frames_enhanced_with_different_light_edits(tmp_path):
    _record(tmp_path, exposure=0.5)
    with pytest.raises(SamplingError):
        sampling.check_enhancement_unchanged(
            tmp_path, {**fp.DEFAULT_SETTINGS, "exposure": 1.0}
        )

    old = tmp_path / "old"
    old.mkdir()
    (old / "preprocess_profile.json").write_text(
        json.dumps({k: fp.DEFAULT_SETTINGS[k] for k in _OLD_KEYS})
    )
    with pytest.raises(SamplingError):
        sampling.check_enhancement_unchanged(
            old, {**fp.DEFAULT_SETTINGS, "exposure": 1.0}
        )
    sampling.check_enhancement_unchanged(old, dict(fp.DEFAULT_SETTINGS))

    same = tmp_path / "same"
    _record(same)
    sampling.check_enhancement_unchanged(same, dict(fp.DEFAULT_SETTINGS))


def test_preview_uses_the_shared_function(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    try:  # a missing libEGL/libGL raises a plain ImportError, so no importorskip
        from qtpy.QtGui import QGuiApplication, QImage

        from idtrackerai.segmentation_app.widgets.enhancement_preview import (
            EnhancementPreview,
        )

        app = QGuiApplication.instance() or QGuiApplication([])  # noqa: F841
    except ImportError as exc:
        pytest.skip(f"Qt or its platform libraries (libEGL/libGL) unavailable: {exc}")
    settings = {**fp.DEFAULT_SETTINGS, "exposure": 1.0}
    preview = EnhancementPreview()
    preview.set_settings(settings)
    preview._ensure_pixmap(0, _rng_frame)
    image = preview._pixmap.toImage().convertToFormat(QImage.Format.Format_Grayscale8)
    h, w = _rng_frame.shape
    bits = image.constBits()
    bits.setsize(image.sizeInBytes())
    got = np.frombuffer(bits, np.uint8).reshape(h, image.bytesPerLine())[:, :w]
    assert np.array_equal(got, fp.enhance_with_settings(_rng_frame, settings))


def test_old_format_background_sidecar_matches_identical_settings(
    tmp_path, monkeypatch
):
    """A sidecar written before the light edits existed holds only the six
    original keys; it must still count as built with the same settings."""
    import json

    from idtrackerai import Session
    from idtrackerai.base.animals_detection.segmentation import load_custom_background

    path = tmp_path / "background.png"
    monkeypatch.setattr(Session, "background_path", property(lambda self: path))
    image = np.tile(np.linspace(40, 90, 32, dtype=np.uint8), (32, 1))
    cv2.imencode(".png", image)[1].tofile(path)

    old = {k: fp.DEFAULT_SETTINGS[k] for k in _OLD_KEYS}
    old.update(enhance=True, clahe_clip=2.0)
    session = object.__new__(Session)
    session.enhancement = dict(old)
    session.background_settings_path.write_text(
        json.dumps({"enhancement": old}), encoding="utf-8"
    )

    assert session.background_matches_enhancement() is True
    # same settings: the saved image is used as is, not enhanced again
    assert np.array_equal(load_custom_background(str(path), None, old), image)
