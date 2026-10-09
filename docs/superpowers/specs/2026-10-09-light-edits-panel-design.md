# Light edits panel for frame enhancement: design

Date: 2026-10-09. Status: awaiting review.

## Problem

Frame enhancement offers three numbers (CLAHE strength, CLAHE tile size, lighting evenness) behind four presets. When footage is too dark, too flat, noisy or soft in a way those cannot fix, there is nothing else to adjust. Users want the familiar photo-editor controls (exposure, shadows, highlights and so on) to make animals stand out from the background before segmentation, judged live on their own footage.

## Goals

- A set of light edits (exposure, brightness, contrast, gamma, shadows, highlights, blacks, whites, sharpness, denoise) next to the existing CLAHE and lighting evenness controls, with the live original/enhanced preview the app already has.
- The edits apply to everything downstream exactly as enhancement does today: thresholds, background model, identification images, Detectron2 training and inference. The preview never disagrees with what is segmented.
- Existing profiles, sessions, `.toml` files, trained models and command-line flags keep working and produce identical output.
- A setup's edits are saved in the same profile file and travel with the trained weights.

## Non-goals

- Anything colour-related (saturation, vibrance, white balance, hue). The pipeline is grayscale from its first step, so these would have no effect. Revisit only if the pipeline becomes colour-aware.
- Local, region-based edits (masks, gradients, brushes).
- Per-video or per-frame-range settings. Settings stay one set per setup.
- Changing how presets behave. Presets keep their current values.

## Processing order

All work happens in `preprocessing.enhance()` on the grayscale frame:

1. Denoise
2. Lighting evenness (existing `correct_illumination`)
3. Tone adjustment (new, one lookup table)
4. CLAHE (existing `apply_clahe`)
5. Sharpen

Every new setting defaults to "no change", and a step at its default is skipped without touching the pixels. With default settings the output is therefore byte-identical to today's. Denoising comes first so it does not amplify noise through the later steps. Sharpening comes last so it acts on the final tonal range.

## The tone step

Exposure, brightness, contrast, gamma, shadows, highlights, blacks and whites are folded into a single 256-entry uint8 lookup table, applied with `cv2.LUT`. It is built from the settings alone (not the image), so it costs the same on every frame and is cached per settings value.

Applied to the normalised value `x` in [0, 1], in this order:

1. Blacks and whites: a levels remap with input black point `black_in = -0.5 * blacks / 100` and white point `white_in = 1 - 0.5 * whites / 100`. Positive blacks lifts the blacks (flatter dark end), negative crushes them; positive whites brightens and saturates the whites, negative pulls them down. Out-of-range values clip.
2. Exposure: multiply by `2 ** exposure`.
3. Gamma: `x ** (1 / gamma)`.
4. Shadows and highlights: add a smooth bump, `shadows * w_dark(x)` and `highlights * w_bright(x)`, where `w_dark` falls from 1 at black to 0 at mid-grey and `w_bright` rises from 0 at mid-grey to 1 at white. Positive shadows lifts the dark end; negative highlights recovers the bright end. These are global curves, so they cannot create halos.
5. Contrast: scale around mid-grey, `0.5 + (x - 0.5) * k`, with `k = 1 + contrast / 100` for positive values and a gentler mapping for negative ones, so -100 is flat grey and +100 doubles the slope.
6. Brightness: add `brightness / 100`.

The result is clipped to [0, 1] and scaled to 0 to 255. The table is monotonic non-decreasing for any in-range combination, which a test enforces: an edit may crush or lift tones but never invert them.

Ranges:

| Key | Range | No change |
| --- | --- | --- |
| `exposure` | -3.0 to 3.0 (EV stops) | 0.0 |
| `brightness` | -100 to 100 | 0 |
| `contrast` | -100 to 100 | 0 |
| `gamma` | 0.3 to 3.0 | 1.0 |
| `shadows` | -100 to 100 | 0 |
| `highlights` | -100 to 100 | 0 |
| `blacks` | -100 to 100 | 0 |
| `whites` | -100 to 100 | 0 |
| `sharpness` | 0 to 100 | 0 |
| `denoise` | 0 to 10 | 0 |

Sharpness is an unsharp mask (`gray + amount * (gray - blur(gray))`, with a fixed small radius). Denoise is `cv2.bilateralFilter` with strength scaled from the setting. It is the only step with a real per-frame cost, and it is off by default. The UI notes this next to the slider.

## Settings schema

- `DEFAULT_SETTINGS` gains the ten keys above, with the no-change values. `SETTING_KEYS` follows automatically.
- `normalize_settings` also checks the new keys are numbers inside their ranges, with the same error style as today. Unknown keys still raise, so a misspelt key is never silently ignored.
- The key `contrast` is the new tone contrast. The existing `clahe_clip` keeps its name, but the UI relabels it "Local contrast (CLAHE)" so the two are not confused.
- Old profiles, sessions and `.toml` files load unchanged; missing keys fill from the defaults. A profile saved with the new keys is rejected by older builds, because unknown keys are an error there by design. This is accepted.
- `check_settings_match` compares a model's recorded settings with the current ones. A recorded dict from before this change has no new keys; a missing recorded key counts as that key's default, so a model trained without light edits warns when run with them.

## Call sites that enumerate settings by name

These list the keys explicitly instead of looping over `SETTING_KEYS`, so each needs the new keys added. Missing one would silently drop an edit at that stage, which is the training/inference mismatch this module exists to prevent.

- `preprocessing.enhance()`, `for_detectron2()` and `make_enhancer_from()`: pass the new settings through.
- `segmentation.apply_enhancement()` in `base/animals_detection`: same.
- `preprocessing.add_arguments()` and `_explicit_from_args()`: one flag per new key (for example `--exposure`, `--shadows`), defaulting to `None` like the existing ones, so `resolve_settings` layering is unchanged.
- `preprocessing.describe()`: lists only the non-default edits, so the summary line stays short.
- `preprocessing._demo()`: runs the same function the pipeline does instead of re-implementing the steps, so the demo strip cannot drift from the pipeline.
- `sampling.py`: already compares via `SETTING_KEYS`; a test confirms the new keys take part.
- The Colab bundle loads `preprocessing.py` by path. It must keep depending only on `cv2` and `numpy`.

## The panel

`EnhancementWidget` keeps the preset dropdown (None, Gentle, Standard, Strong, Custom), the original/enhanced compare slider, Save setup and Load setup. Choosing Custom reveals the sliders in collapsible groups, each group remembering whether it was open:

- **Light:** exposure, brightness, gamma.
- **Tone:** contrast, shadows, highlights, blacks, whites.
- **Detail:** sharpness, denoise, local contrast (CLAHE) strength and tile.
- **Evenness:** lighting evenness.

Each slider shows its value, resets to the no-change value on double-click, and has a tooltip. A "Reset all" button returns every slider to no change. Moving any slider while a preset is selected switches the dropdown to Custom, and `setSettings` selects a preset only when every key (new ones included) matches it, as it does now. The existing `_ValueSlider` is reused, extended with the reset and a signed display. Slider changes keep the existing flow: `settingsChanged` updates the preview live; `settingsCommitted` (on release or after 300 ms) refreshes the background model and Detectron2 panel.

The widget is already 330 lines. The groups go in a new `light_edits.py` beside it, so `enhancement_widget.py` only wires them in.

## Error handling

- Out-of-range values from a profile or flag raise `PreprocessingError` naming the key and its range. In the UI the sliders make this impossible; loading a bad profile shows the existing warning dialog.
- A preview failure shows the existing inline "Preview failed" message and does not touch the saved settings.

## Testing

- **Identity:** default settings give output identical to the current `enhance()` on test frames (guards existing models).
- **Direction:** each control moves mean brightness, spread or sharpness in the right direction on a synthetic gradient.
- **Lookup table:** monotonic non-decreasing across a grid of in-range settings; identity at defaults.
- **Schema:** out-of-range and unknown keys raise; partial and old-format dicts load; profile save/load round-trips; `check_settings_match` treats missing recorded keys as defaults.
- **Call sites:** the same settings give identical output through `make_enhancer_from`, `apply_enhancement` and the CLI flags, so training, inference and segmentation cannot disagree.
- **Widget:** moving a slider switches to Custom; Reset all returns to the None preset's values; `setSettings` round-trips every preset and a custom set; double-click resets one slider.

Existing tests (`enhancement_settings_test.py`, `custom_background_test.py`, `detectron2_pipeline_fixes_test.py`) must pass unchanged.

## Open items

None. The shadows/highlights bump shape and the sharpen radius are tuned against the test footage during implementation, within the ranges above.
