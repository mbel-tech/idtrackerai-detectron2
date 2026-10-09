"""A background sidecar that cannot be understood must read as a mismatch."""

import json

import pytest

from idtrackerai import Session
from idtrackerai.extra_tools.detectron2_pipeline import preprocessing as fp

_OLD_KEYS = (
    "enhance",
    "clahe_clip",
    "clahe_tile",
    "illumination_sigma",
    "illumination_downsample",
    "correct_lighting",
)


@pytest.fixture
def session(tmp_path, monkeypatch):
    monkeypatch.setattr(
        Session, "background_path", property(lambda self: tmp_path / "background.png")
    )
    session = object.__new__(Session)
    session.enhancement = {"clahe_clip": 2.0}
    return session


def _write_sidecar(session, recorded):
    session.background_settings_path.write_text(
        json.dumps({"enhancement": recorded}), encoding="utf-8"
    )


def _current(session):
    return dict(session.effective_enhancement())


def test_sidecar_with_unknown_key_does_not_match(session):
    _write_sidecar(session, {**_current(session), "from_a_newer_build": 1})
    assert session.background_matches_enhancement() is False


def test_sidecar_with_out_of_range_value_does_not_match(session):
    _write_sidecar(session, {**_current(session), "exposure": 99})
    assert session.background_matches_enhancement() is False


def test_old_six_key_sidecar_still_matches(session):
    old = {k: fp.DEFAULT_SETTINGS[k] for k in _OLD_KEYS}
    old.update(enhance=True, clahe_clip=2.0)
    session.enhancement = dict(old)
    _write_sidecar(session, old)
    assert session.background_matches_enhancement() is True
