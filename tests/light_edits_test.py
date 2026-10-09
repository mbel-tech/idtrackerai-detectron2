"""Light-edit settings: schema defaults, range validation and reporting."""

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
