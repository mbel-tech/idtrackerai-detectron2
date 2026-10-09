"""The light edits panel in the Segmentation App's enhancement widget."""

import pytest

from idtrackerai.extra_tools.detectron2_pipeline import preprocessing as fp

LIGHT_KEYS = tuple(fp.LIGHT_EDIT_RANGES)


@pytest.fixture(scope="module")
def widgets():
    """The widget classes, with a QApplication, or a clean skip without Qt."""
    mp = pytest.MonkeyPatch()
    mp.setenv("QT_QPA_PLATFORM", "offscreen")
    try:  # a missing libEGL/libGL raises a plain ImportError, so no importorskip
        from qtpy.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])
    except ImportError as exc:
        mp.undo()
        pytest.skip(f"Qt or its platform libraries (libEGL/libGL) unavailable: {exc}")
    from idtrackerai.segmentation_app.widgets import enhancement_widget
    from idtrackerai.segmentation_app.widgets.light_edits import LightEditGroups

    yield enhancement_widget, LightEditGroups
    del app
    mp.undo()


@pytest.fixture
def widget(widgets):
    return widgets[0].EnhancementWidget()


def test_slider_value_is_exact_after_reset(widgets):
    groups = widgets[1]()
    assert set(LIGHT_KEYS) <= set(groups.sliders)
    for key, slider in groups.sliders.items():
        lo, hi = slider.range()
        slider.setValue(lo + (hi - lo) * 0.3)
        slider.reset()
        assert slider.value() == fp.DEFAULT_SETTINGS[key], key


def test_moving_a_slider_switches_to_custom(widgets, widget):
    widget.setSettings(widgets[0].PRESETS["Standard"])
    assert widget.preset.currentText() == "Standard"
    widget.groups.sliders["exposure"].setValue(0.5)
    assert widget.preset.currentText() == "Custom"
    assert widget.settings()["exposure"] == 0.5


def test_reset_all_returns_to_no_change(widget):
    widget.setSettings({**fp.DEFAULT_SETTINGS, "shadows": 40, "gamma": 1.4})
    widget.groups.sliders["exposure"].setValue(1.5)
    widget.groups.resetAll()
    got = widget.groups.values()
    assert {k: got[k] for k in LIGHT_KEYS} == {
        k: fp.DEFAULT_SETTINGS[k] for k in LIGHT_KEYS
    }


def test_reset_all_button_commits(widget):
    widget.setSettings({**fp.DEFAULT_SETTINGS, "shadows": 40})
    seen = []
    widget.settingsCommitted.connect(seen.append)
    widget.reset_button.click()
    assert seen and seen[-1]["shadows"] == 0


def test_set_settings_round_trips_a_custom_set(widget):
    custom = {**fp.DEFAULT_SETTINGS, "shadows": 40, "gamma": 1.4}
    widget.setSettings(custom)
    assert widget.settings() == custom
    assert widget.preset.currentText() == "Custom"


def test_set_settings_selects_a_preset_only_on_full_match(widgets, widget):
    standard = widgets[0].PRESETS["Standard"]
    widget.setSettings(standard)
    assert widget.preset.currentText() == "Standard"
    widget.setSettings({**standard, "exposure": 0.1})
    assert widget.preset.currentText() == "Custom"


def test_old_settings_without_light_edits_load(widget):
    widget.setSettings(
        {
            "clahe_clip": 1.5,
            "clahe_tile": 8,
            "illumination_sigma": 25.0,
            "illumination_downsample": 4,
            "correct_lighting": True,
            "enhance": True,
        }
    )
    assert widget.preset.currentText() == "Standard"


def test_groups_are_collapsible(widget):
    groups = widget.groups
    assert len(groups.toggles) == 4
    for title, toggle in groups.toggles.items():
        sliders = groups.members[title]
        assert sliders
        for want_open in (False, True):
            toggle.setChecked(want_open)
            assert all(
                groups.sliders[k].isVisibleTo(groups) == want_open for k in sliders
            )
    # every group starts open, so an active Tone edit is never hidden
    assert all(t.isChecked() for t in groups.toggles.values())
    groups.toggles["Tone"].setChecked(False)
    widget.setSettings({**fp.DEFAULT_SETTINGS, "shadows": 40})
    assert not groups.toggles["Tone"].isChecked()
    assert not groups.sliders["shadows"].isVisibleTo(groups)


def test_clahe_label_is_renamed(widget):
    assert widget.contrast_label.text() == "Local contrast (CLAHE)"


def test_tooltips_cover_every_light_edit(widget):
    import tomllib
    from pathlib import Path

    path = Path(fp.__file__).parents[2] / "segmentation_app" / "tooltips.toml"
    tips = tomllib.loads(path.read_text(encoding="utf-8"))
    widget.setToolTips(tips)
    assert widget.groups.sliders["shadows"].toolTip()


@pytest.fixture
def warnings_seen(monkeypatch, widgets):
    seen = []
    monkeypatch.setattr(
        widgets[0].QMessageBox, "warning", lambda *a, **k: seen.append(a[1:])
    )
    return seen


@pytest.mark.parametrize("bad", ["5.0", "NaN"])
def test_profile_with_bad_value_warns_and_changes_nothing(
    widget, widgets, warnings_seen, monkeypatch, tmp_path, bad
):
    path = tmp_path / "bad.json"
    path.write_text('{"exposure": %s}' % bad)
    monkeypatch.setattr(
        widgets[0].QFileDialog, "getOpenFileName", lambda *a, **k: (str(path), "")
    )
    widget.setSettings({**fp.DEFAULT_SETTINGS, "shadows": 40})
    before = widget.settings()
    widget.load_profile()
    assert widget.settings() == before
    assert len(warnings_seen) == 1
    assert warnings_seen[0][0] == "Could not read the setup"


def test_set_settings_with_bad_value_warns_and_changes_nothing(widget, warnings_seen):
    widget.setSettings({**fp.DEFAULT_SETTINGS, "shadows": 40})
    before = widget.settings()
    widget.setSettings({"exposure": 99})
    assert widget.settings() == before
    assert len(warnings_seen) == 1
    assert warnings_seen[0][0] == "Invalid enhancement settings"
