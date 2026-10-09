# Light edits panel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add ten grayscale light edits (exposure, brightness, contrast, gamma, shadows, highlights, blacks, whites, sharpness, denoise) to frame enhancement, with a grouped collapsible panel in the Segmentation App.

**Architecture:** The edits are new keys in the shared settings dict in `preprocessing.py`, applied by one new function, `enhance_with_settings`, which every call site (segmentation, background model, GUI preview, Detectron2 enhancer, demo) uses instead of listing keys by hand. Tone edits are folded into one cached 256-entry lookup table; denoise and sharpen are skipped at their defaults, so default output is identical to today's.

**Tech Stack:** Python, OpenCV (`cv2.LUT`, `bilateralFilter`, `GaussianBlur`), NumPy, qtpy/PyQt6, pytest.

**Spec:** `docs/superpowers/specs/2026-10-09-light-edits-panel-design.md`

## Global Constraints

- Grayscale only. No saturation, hue or white-balance control.
- `preprocessing.py` may import only `cv2`, `numpy` and the standard library (the Colab bundle loads it by path, see `bundle.py`).
- Every new key defaults to no change: `exposure` 0.0, `brightness` 0, `contrast` 0, `gamma` 1.0, `shadows` 0, `highlights` 0, `blacks` 0, `whites` 0, `sharpness` 0, `denoise` 0.
- Ranges: `exposure` -3.0 to 3.0, `brightness`/`contrast`/`shadows`/`highlights`/`blacks`/`whites` -100 to 100, `gamma` 0.3 to 3.0, `sharpness` 0 to 100, `denoise` 0 to 10.
- Order: denoise, lighting evenness, tone, CLAHE, sharpen.
- Unknown settings keys still raise. Old profiles, sessions and `.toml` files load unchanged with missing keys filled from defaults.
- The `clahe_clip` key keeps its name; its UI label becomes "Local contrast (CLAHE)".
- Existing tests (`enhancement_settings_test.py`, `custom_background_test.py`, `detectron2_pipeline_fixes_test.py`) pass unchanged.
- Tests run with `QT_QPA_PLATFORM=offscreen`. If `cv2` or `pytest` are missing, run `pip install -e ".[dev]"` first.

## Review Focus

- NaN or infinite values in a profile or flag (JSON allows `NaN`): rejected with `PreprocessingError`, not passed on to the lookup table.
- `true`/`false` given for a numeric key: rejected, as the existing keys already do.
- Extreme levels (`blacks` 100 with `whites` -100) would make the black point meet the white point: the table must stay finite and monotonic, not divide by zero.
- A model trained before this change, run with light edits on: `check_settings_match` must warn instead of staying silent.
- Slider float round-off (`3 * 0.1`): a slider reset must give exactly the default, or a no-change Custom never matches a preset and profiles differ by `1e-17`.

## File Structure

- Modify `src/idtrackerai/extra_tools/detectron2_pipeline/preprocessing.py`: schema, validation, image ops, `enhance_with_settings`, CLI flags, `describe`, `_demo`.
- Modify `src/idtrackerai/base/animals_detection/segmentation.py`: `apply_enhancement` delegates to `enhance_with_settings`.
- Modify `src/idtrackerai/segmentation_app/widgets/enhancement_preview.py`: use `enhance_with_settings`.
- Create `src/idtrackerai/segmentation_app/widgets/light_edits.py`: the collapsible groups of sliders.
- Modify `src/idtrackerai/segmentation_app/widgets/enhancement_widget.py`: wire the groups in, Custom settings, Reset all, relabel CLAHE.
- Modify `src/idtrackerai/segmentation_app/tooltips.toml`: tooltips for the new sliders.
- Create `tests/light_edits_test.py` (core, call sites) and `tests/light_edits_widget_test.py` (widget).
- Modify `USER_GUIDE.md` and `docs/detectron2-pipeline.md`: describe the panel.

---

### Task 1: Settings schema and validation

**Files:**
- Modify: `src/idtrackerai/extra_tools/detectron2_pipeline/preprocessing.py` (`DEFAULT_SETTINGS`, `normalize_settings`, `describe`, `check_settings_match`)
- Test: `tests/light_edits_test.py`

**Interfaces:**
- Produces: `fp.LIGHT_EDIT_RANGES: dict[str, tuple[float, float]]` (key to inclusive min, max) and the ten keys in `fp.DEFAULT_SETTINGS` / `fp.SETTING_KEYS`. `fp.normalize_settings` validates them. `fp.describe(settings)` appends only non-default edits, e.g. `"exposure=+0.5, shadows=+20"`.

- [ ] **Step 1: Write the failing tests** in `tests/light_edits_test.py` (module imports `preprocessing as fp`, `pytest`):
  - `test_defaults_are_no_change`: every key of `LIGHT_EDIT_RANGES` is in `DEFAULT_SETTINGS`, and `DEFAULT_SETTINGS["gamma"] == 1.0` while the other nine equal `0` / `0.0`.
  - `test_partial_dict_and_old_profile_are_completed`: `normalize_settings({"clahe_clip": 2.0})` equals `{**DEFAULT_SETTINGS, "clahe_clip": 2.0}`.
  - `test_out_of_range_is_rejected` (parametrized over `("exposure", 3.1)`, `("gamma", 0.2)`, `("denoise", 11)`, `("sharpness", -1)`, `("blacks", 101)`): raises `fp.PreprocessingError` matching the key name.
  - `test_nan_inf_and_bool_are_rejected` (parametrized `float("nan")`, `float("inf")`, `True`): raises `PreprocessingError` for `exposure`.
  - `test_integers_are_accepted_for_float_keys`: `{"exposure": 1, "gamma": 2}` normalizes without error.
  - `test_describe_lists_only_non_default_edits`: `describe({**DEFAULT_SETTINGS, "exposure": 0.5, "shadows": 20})` contains `"exposure=+0.5"` and `"shadows=+20"` and not `"gamma"`; with defaults the string equals the pre-change output.
  - `test_missing_recorded_key_counts_as_default`: `check_settings_match({"clahe_clip": 1.5}, {**DEFAULT_SETTINGS, "exposure": 1.0}, "the model")` returns a string mentioning `exposure`; with `exposure` 0.0 it returns `None`.

- [ ] **Step 2: Run** `python -m pytest tests/light_edits_test.py -q`. Expected: FAIL (`LIGHT_EDIT_RANGES` missing).

- [ ] **Step 3: Implement** in `preprocessing.py`: `LIGHT_EDIT_RANGES` with the ranges from Global Constraints; add the ten defaults to `DEFAULT_SETTINGS`. In `normalize_settings`, for each light-edit key require `isinstance(v, (int, float))`, not `bool`, `math.isfinite(v)` and inside its range, else `PreprocessingError(f"enhancement '{key}' must be a number between {lo} and {hi}")`. In `check_settings_match`, take the recorded value as `recorded.get(key, DEFAULT_SETTINGS.get(key))` for keys of `current` that have a default (the existing guard `key in recorded` is replaced by `key in current`, and a `recorded` with no settings at all still returns `None`). In `describe`, after the existing parts append the non-default edits formatted `name=+x` (signed, trimmed trailing zeros).

- [ ] **Step 4: Run** the Step 2 command plus `python -m pytest tests/enhancement_settings_test.py tests/detectron2_pipeline_fixes_test.py -q`. Expected: all PASS.

- [ ] **Step 5: Commit** `git add -A && git commit -m "Add light-edit settings to the enhancement schema"`

---

### Task 2: Image operations and `enhance_with_settings`

**Files:**
- Modify: `src/idtrackerai/extra_tools/detectron2_pipeline/preprocessing.py`
- Test: `tests/light_edits_test.py`

**Interfaces:**
- Consumes: Task 1 keys and `LIGHT_EDIT_RANGES`.
- Produces:
  - `tone_lut(exposure, brightness, contrast, gamma, shadows, highlights, blacks, whites) -> np.ndarray` (shape `(256,)`, uint8, `functools.lru_cache`d on its arguments, so the returned array is made read-only).
  - `denoise(gray, strength: float) -> np.ndarray`, `sharpen(gray, amount: float) -> np.ndarray`.
  - `enhance(frame, ..., **light_edits)`: the existing signature gains the ten keys as keyword arguments with the no-change defaults.
  - `enhance_with_settings(frame: np.ndarray, settings: dict) -> np.ndarray`: single-channel uint8; fills missing keys from `DEFAULT_SETTINGS`; ignores the `enhance` flag, like `apply_enhancement` today.

- [ ] **Step 1: Write the failing tests** (module-level helper `ramp = np.tile(np.arange(256, dtype=np.uint8), (32, 1))`, `rng` frame `integers(0, 255, (64, 64), uint8)`):
  - `test_default_output_is_identical_to_the_old_pipeline`: for the random frame and for `ramp`, `enhance_with_settings(f, DEFAULT_SETTINGS)` equals `apply_clahe(correct_illumination(to_gray(f), 4, 25.0), 1.5, 8)`, and equals the same with `correct_lighting=False` composition when that setting is off.
  - `test_default_lut_is_identity`: `tone_lut(0, 0, 0, 1.0, 0, 0, 0, 0)` equals `np.arange(256)`.
  - `test_each_control_moves_the_ramp_the_right_way` (parametrized `(kwarg, value, mid_sample_should_be)`): applying `tone_lut` at index 128: `exposure=1` raises it, `exposure=-1` lowers it, `brightness=50` raises, `brightness=-50` lowers, `gamma=2.0` raises, `gamma=0.5` lowers, `contrast=100` leaves 128 within 1 of itself but maps index 192 higher and 64 lower, `contrast=-100` maps everything to within 1 of 128, `shadows=100` raises index 32 and leaves index 200 unchanged, `highlights=-100` lowers index 224 and leaves index 32 unchanged, `blacks=-50` maps index 20 to 0 (crushed) and `blacks=50` lifts index 0 above 0, `whites=50` maps index 230 to 255.
  - `test_lut_is_monotonic_for_any_in_range_combination`: over a grid of the extremes and 0 for all eight tone settings (`itertools.product`), `np.all(np.diff(lut.astype(int)) >= 0)`.
  - `test_extreme_levels_stay_finite`: `tone_lut(blacks=100, whites=-100, ...)` has no exception, dtype uint8, monotonic.
  - `test_denoise_reduces_noise_and_sharpen_increases_edges`: on a flat 100 frame plus Gaussian noise (seeded), `denoise(f, 8).std() < f.std()`; on a step-edge frame, `sharpen(f, 100)` has larger max gradient than `f`.
  - `test_zero_strength_returns_the_input_values`: `denoise(f, 0)` and `sharpen(f, 0)` equal `f`.
  - `test_tiny_and_flat_frames_do_not_raise`: `enhance_with_settings` on a `4x4` frame and a constant frame, with every light edit at its max, returns uint8 of the same shape.
  - `test_missing_keys_are_filled`: `enhance_with_settings(f, {"clahe_clip": 2.0})` equals the call with `{**DEFAULT_SETTINGS, "clahe_clip": 2.0}`.

- [ ] **Step 2: Run** `python -m pytest tests/light_edits_test.py -q`. Expected: new tests FAIL (`tone_lut` undefined).

- [ ] **Step 3: Implement `tone_lut`** with `x = np.arange(256) / 255.0` and these stages, each followed by `np.clip(x, 0, 1)`:
  1. Levels: `black_in = -0.5 * blacks / 100` (positive blacks lifts); `white_in = 1 - 0.5 * whites / 100`; `white_in = max(white_in, black_in + 0.05)`; `x = (x - black_in) / (white_in - black_in)`.
  2. `x = x * 2 ** exposure`.
  3. `x = x ** (1 / gamma)`.
  4. Bumps with amplitude `0.25 * v / 100`: `x += 0.25 * shadows/100 * np.where(x < 0.5, (1 - 2*x)**2, 0)` and `x += 0.25 * highlights/100 * np.where(x > 0.5, (2*x - 1)**2, 0)` (amplitude is capped at 0.25 because that is the largest that keeps the curve monotonic).
  5. `x = 0.5 + (x - 0.5) * (1 + contrast / 100)` (linear on both sides: +100 doubles the slope, -100 flattens to grey).
  6. `x = x + brightness / 100`.
  Return `np.round(x * 255).astype(np.uint8)`.

  Then: `denoise` is `cv2.bilateralFilter(gray, 5, 8 * strength, strength)` (return `gray` unchanged when `strength <= 0`); `sharpen` is `cv2.addWeighted(gray, 1 + a, cv2.GaussianBlur(gray, (0, 0), 2.0), -a, 0)` with `a = amount / 50` (unchanged when `amount <= 0`). Extend `enhance()`: denoise, then the existing lighting step, then `cv2.LUT(gray, tone_lut(...))` only when the eight tone settings differ from the defaults, then CLAHE, then sharpen. Add `enhance_with_settings(frame, settings)` merging over `DEFAULT_SETTINGS` and mapping `illumination_downsample`/`illumination_sigma` to `enhance()`'s `downsample`/`sigma`.

- [ ] **Step 4: Run** `python -m pytest tests/light_edits_test.py tests/enhancement_settings_test.py -q`. Expected: all PASS. Then adjust the shadows/highlights and sharpen values only if a direction test fails, staying inside the ranges in Global Constraints.

- [ ] **Step 5: Commit** `git add -A && git commit -m "Add tone, denoise and sharpen steps to the enhancement"`

---

### Task 3: Route every call site through one function; CLI flags and demo

**Files:**
- Modify: `preprocessing.py` (`make_enhancer_from`, `add_arguments`, `_explicit_from_args`, `_demo`)
- Modify: `src/idtrackerai/base/animals_detection/segmentation.py:175-189` (`apply_enhancement`)
- Modify: `src/idtrackerai/segmentation_app/widgets/enhancement_preview.py:122-129` (`_ensure_pixmap`)
- Modify: `src/idtrackerai/extra_tools/detectron2_pipeline/sampling.py:275-295` (`check_enhancement_unchanged`)
- Test: `tests/light_edits_test.py`

**Interfaces:**
- Consumes: `enhance_with_settings`, `normalize_enhancement` (existing, `segmentation.py:166`).
- Produces: CLI flags `--exposure`, `--brightness`, `--contrast`, `--gamma`, `--shadows`, `--highlights`, `--blacks`, `--whites`, `--sharpness`, `--denoise` (all `type=float`, `default=None`), picked up by `_explicit_from_args`.

- [ ] **Step 1: Write the failing tests**:
  - `test_all_call_sites_agree`: with `settings = {**DEFAULT_SETTINGS, "clahe_clip": 2.0, "exposure": 0.7, "shadows": 30, "sharpness": 40, "denoise": 3}` on a random frame, `apply_enhancement(f, settings)` equals `enhance_with_settings(f, settings)` equals `make_enhancer_from(settings)(f)[:, :, 0]`.
  - `test_cli_flags_reach_the_settings`: build a parser with `fp.add_arguments`, parse `["--exposure", "0.5", "--gamma", "1.4"]`, `fp.resolve_settings(args)[0]` has `exposure == 0.5` and `gamma == 1.4` and `shadows == 0`; with no flags it equals `DEFAULT_SETTINGS`.
  - `test_cli_flag_out_of_range_is_rejected`: `--denoise 50` makes `resolve_settings` raise `PreprocessingError` (it runs `normalize_settings` on the merged result; add that call).
  - `test_profile_round_trip_with_light_edits`: `save_profile` then `load_profile` returns the same ten values; an old profile file written with only the six old keys loads and normalizes.
  - `test_sampling_refuses_frames_enhanced_with_different_light_edits`: `sampling.check_enhancement_unchanged(folder, settings)` (`sampling.py:275`) raises `SamplingError` when the folder's recorded profile has `exposure` 0.5 and `settings` has 1.0, and also when the recorded profile is an old one without light-edit keys and `settings` has `exposure` 1.0 (missing recorded key counts as the default). It does not raise when both are at defaults.
  - `test_preview_uses_the_shared_function` (needs Qt, offscreen): construct `EnhancementPreview`, set settings with `exposure=1.0`, call `_ensure_pixmap(0, frame)`, and compare the pixmap's image bytes to `enhance_with_settings(frame, settings)`.

- [ ] **Step 2: Run** `python -m pytest tests/light_edits_test.py -q`. Expected: the new tests FAIL.

- [ ] **Step 3: Implement.** `make_enhancer_from`, `apply_enhancement` (after its `normalize_enhancement`) and `_ensure_pixmap` call `enhance_with_settings`; delete their hand-written keyword lists. `make_enhancer_from` still returns `for_detectron2`'s three-channel form (`cv2.cvtColor(..., GRAY2BGR)` around `enhance_with_settings`). In `sampling.check_enhancement_unchanged`, compare `recorded.get(key, DEFAULT_SETTINGS.get(key))` instead of skipping keys absent from `recorded` (import `DEFAULT_SETTINGS` beside `SETTING_KEYS` at `sampling.py:29-33`, both import branches). Add the ten flags in `add_arguments` and the names to the tuple in `_explicit_from_args`. In `resolve_settings`, run `normalize_settings` on the final dict before returning. Rewrite `_demo` to produce its strip as `[gray, enhance_with_settings(frame, settings)]` with caption `"grayscale | enhanced"` (drop its private re-implementation of the steps).

- [ ] **Step 4: Run** `QT_QPA_PLATFORM=offscreen python -m pytest tests -q -k "not smoke"`. Expected: all PASS.

- [ ] **Step 5: Commit** `git add -A && git commit -m "Use one enhancement function at every call site; add CLI flags"`

---

### Task 4: Panel widget

**Files:**
- Create: `src/idtrackerai/segmentation_app/widgets/light_edits.py`
- Modify: `src/idtrackerai/segmentation_app/widgets/enhancement_widget.py` (`_ValueSlider`, `EnhancementWidget.__init__`, `settings`, `setSettings`, `_apply`, `_changed`, `setToolTips`)
- Modify: `src/idtrackerai/segmentation_app/tooltips.toml` (after line 30)
- Test: `tests/light_edits_widget_test.py`

**Interfaces:**
- Consumes: `fp.LIGHT_EDIT_RANGES`, `fp.DEFAULT_SETTINGS`, existing `_ValueSlider`.
- Produces: in `light_edits.py`, `class LightEditGroups(QWidget)` with `valueChanged = Signal()`, `released = Signal()`, `values() -> dict` (the ten edits plus `clahe_clip`, `clahe_tile`, `illumination_sigma`), `setValues(dict)`, `resetAll()`, and `sliders: dict[str, _ValueSlider]` keyed by settings key. `_ValueSlider` gains `default: float`, a signed display, `reset()` bound to double-click, and `value()` rounds to its decimals.

- [ ] **Step 1: Write the failing tests** (a module-scoped `QApplication` fixture; `EnhancementWidget()` built directly):
  - `test_slider_value_is_exact_after_reset`: for every key in `LightEditGroups().sliders`, after `setValue(<non-default>)` then `reset()`, `value() == fp.DEFAULT_SETTINGS[key]` (exact equality; catches `3 * 0.1`).
  - `test_moving_a_slider_switches_to_custom`: start at preset `Standard`, set `groups.sliders["exposure"].setValue(0.5)`, assert `preset.currentText() == "Custom"` and `settings()["exposure"] == 0.5`.
  - `test_reset_all_returns_to_no_change`: after several edits, `resetAll()` makes the ten edit values equal the defaults.
  - `test_set_settings_round_trips_a_custom_set`: `setSettings({**DEFAULT_SETTINGS, "shadows": 40, "gamma": 1.4})` then `settings()` returns the same dict; `preset.currentText() == "Custom"`.
  - `test_set_settings_selects_a_preset_only_on_full_match`: `setSettings(PRESETS["Standard"])` selects `Standard`; the same dict with `exposure=0.1` selects `Custom`.
  - `test_old_settings_without_light_edits_load`: `setSettings({"clahe_clip": 1.5, "clahe_tile": 8, "illumination_sigma": 25.0, "illumination_downsample": 4, "correct_lighting": True, "enhance": True})` selects `Standard`.
  - `test_groups_are_collapsible`: each of the four group toggles hides and shows its sliders, and the open/closed state survives a `setSettings` call.
  - `test_clahe_label_is_renamed`: `widget.contrast_label.text() == "Local contrast (CLAHE)"`.

- [ ] **Step 2: Run** `QT_QPA_PLATFORM=offscreen python -m pytest tests/light_edits_widget_test.py -q`. Expected: FAIL (module missing).

- [ ] **Step 3: Implement `LightEditGroups`** in `light_edits.py`: four collapsible sections (a checkable `QToolButton` arrow header toggling a body `QWidget`) titled Light (exposure, brightness, gamma), Tone (contrast, shadows, highlights, blacks, whites), Detail (sharpness, denoise, CLAHE strength, CLAHE tile), Evenness (lighting evenness), each row a label plus a `_ValueSlider` built from `LIGHT_EDIT_RANGES` (steps: exposure 0.1, gamma 0.05, everything else 1; decimals 1, 2, 0). Move the existing `contrast`, `detail` and `evenness` sliders into it, keeping their `contrast`/`detail`/`evenness` attribute names on `EnhancementWidget` so tooltips and tests keep working. Keep the open/closed state in the instance, not in `setValues`. In `EnhancementWidget`: replace `custom_rows` with `LightEditGroups`, add a "Reset all" `QToolButton` in the Custom area, extend `settings()` (Custom branch) and `_apply()` to the ten keys, make any slider move set the preset to `Custom` (guarded by `_loading`), and relabel `contrast_label` to "Local contrast (CLAHE)". Add tooltip entries `enhancement_exposure`, `_brightness`, `_gamma`, `_shadows`, `_highlights`, `_blacks`, `_whites`, `_sharpness`, `_denoise`, `_contrast_tone`, `_reset` to `tooltips.toml` in the existing `<qt>` style, and set them in `setToolTips`; the denoise tooltip says it is slower on long videos and off by default.

- [ ] **Step 4: Run** `QT_QPA_PLATFORM=offscreen python -m pytest tests -q -k "not smoke"`. Expected: all PASS. Then launch the app offscreen once (`QT_QPA_PLATFORM=offscreen python -c "from idtrackerai.segmentation_app.main import *"` or the project's own run entry point) to confirm it imports and `tooltips.toml` still parses.

- [ ] **Step 5: Commit** `git add -A && git commit -m "Add the light edits panel to the enhancement widget"`

---

### Task 5: Docs and full verification

**Files:**
- Modify: `USER_GUIDE.md:222-236` (Step 1, Enhancement), `docs/detectron2-pipeline.md:95`

- [ ] **Step 1: Update the docs.** In `USER_GUIDE.md` add a short subsection under Step 1 naming the four groups and the Reset all button, saying the edits are grayscale-only, travel with the profile and the model, and that denoise is slower. Update the stale "Off / CLAHE / CLAHE + even lighting" line in `docs/detectron2-pipeline.md:95` to the preset names plus the light edits.

- [ ] **Step 2: Run the full suite** `QT_QPA_PLATFORM=offscreen python -m pytest tests -q`. Expected: all PASS (including `smoke_test.py` if its dependencies are installed; if torch is missing, report which tests were skipped).

- [ ] **Step 3: Run the repo's checks** `black --check src tests && isort --check src tests` (settings in `pyproject.toml`). Expected: no changes needed; otherwise apply and re-run.

- [ ] **Step 4: Commit and push** `git add -A && git commit -m "Document the light edits panel" && git push -u origin fatina-del-codice/wizardly-keller-axqagf`

## Self-review notes

- Spec coverage: processing order and tone/sharpen/denoise (Task 2); schema, ranges, old-profile loading, `check_settings_match` (Task 1); call sites, CLI flags, describe, demo, sampling (Tasks 1 and 3); panel, groups, reset, relabel, tooltips, `light_edits.py` split (Task 4); error handling is the range checks of Task 1 plus the existing preview-failure path; testing list is spread across the tasks; Colab constraint is in Global Constraints and checked by Task 3 importing nothing new.
- Decisions the spec left open and this plan pins: level points are `-0.5*blacks/100` (positive blacks lifts) and `1 - 0.5*whites/100` with a 0.05 minimum gap; shadow/highlight bump amplitude is capped at 0.25 (the monotonic limit); contrast is linear on both sides; denoise is `bilateralFilter(gray, 5, 8*s, s)`; sharpen uses sigma 2.0 and amount `sharpness/50`.
