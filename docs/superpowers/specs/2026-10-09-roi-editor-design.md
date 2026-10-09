# ROI editor for the Segmentation App: design

Date: 2026-10-09. Status: awaiting review.

## Problem

Defining a region of interest means clicking every corner of a polygon, choosing a shape type in a pop-up for each shape, and managing a list of text lines. This is slow for irregular arenas (labyrinth floors made of several strips), and the result cannot be corrected except by redrawing a shape. Users work with a handful of setups, each reused for many videos.

## Goals

- A simple arena (rectangle, circle, ellipse) takes about five seconds.
- An irregular arena is selected by clicking regions and fixing edges with a brush, not by placing corner points.
- What will be tracked is always visible on the video.
- A finished arena is saved once and reused for later videos of the same setup.
- Existing `.toml` files, sessions and downstream tools keep working.

## Non-goals (for now)

- A "Suggest floor" button that proposes the region automatically.
- Rotating, warping or automatically aligning a saved arena. Only translation by hand and uniform scaling for the same proportions.
- Changing how exclusive ROIs behave.

## User experience

The polygon list, the Add button and the type pop-up in `ROI_widget.py` are replaced. When "Regions of interest" is ticked, a tool strip appears over the video and a panel appears beside it.

**Tool strip:** Brush (B), Eraser (E), Click select (W), Rectangle (R), Ellipse (O), Undo (Ctrl+Z), Redo (Ctrl+Y).

**Panel:**
- Include / Exclude toggle. Alt flips it temporarily while a tool is in use.
- Brush size slider. `[` and `]` change it.
- Click select tolerance slider.
- Saved arenas: a dropdown with previews, Save as..., Import..., Export..., Clear.

**Behaviour:**
- Rectangle and Ellipse are drawn by dragging. Until committed (Enter, or choosing another tool) they show handles for moving and resizing. Including or excluding applies when they commit.
- Click select floods outward from the clicked pixel on the same enhanced frame that is shown, limited by the tolerance. The result is lightly smoothed and holes smaller than a few pixels are filled. Clicking adds to the mask in Include mode and removes in Exclude mode.
- Brush and Eraser paint into the mask. Shift-click draws a straight line from the last point.
- The area outside the mask is dimmed on the video at all times. There is no separate preview step.
- Pan with Space + drag and zoom with the wheel, as in the existing canvas.

## Data model

The single source of truth is a mask: a `uint8` array of the video height and width, values 0 or 255.

- All tools edit the mask. Undo and redo keep up to 30 steps, each stored as the changed bounding box only, so memory stays small at high resolutions.
- A new module `segmentation_app/widgets/roi_mask.py` holds the mask operations with no Qt dependency, so they can be tested directly: fill from a shape, flood select, brush stroke, mask to polygon strings, polygon strings to mask.
- `ROI_widget.py` becomes the Qt layer: tool strip, panel, canvas interaction and painting. It keeps the same outward interface so `main.py` changes little: `valueChanged(mask)`, `getValue()`, `setValue(values, exclusive)`, `paint_on_canvas`, `set_video_size`, `click_event`.

## Files and compatibility

**Loading an old file.** `roi_list` strings (`+ Polygon [...]`, `- Ellipse {...}`) are drawn into the mask in order, using the existing `build_ROI_mask_from_list`. The result is then editable with the new tools. The text lines are not shown to the user.

**Saving.** Saving parameters writes both:
- `roi_list`: polygon strings derived from the mask. Each connected region becomes one `+ Polygon`, each hole one `- Polygon`, simplified with `approxPolyDP` at 0.5 px. This keeps the validator, which reads `roi_list` to draw the regions, and any other reader working unchanged.
- `roi_mask`: the path to the mask PNG written beside the `.toml`, as a new optional session parameter. When present it is the exact mask and takes precedence.

**Session.** `session.py` gets a new annotated field `roi_mask: Path | str | None = None`. The annotation is required, because unknown keys in a loaded `.toml` are reported as unrecognised. In `prepare_tracking`, line 375 builds `ROI_mask` from `roi_list`. It changes to: if `roi_mask` is set, load that PNG (grayscale, thresholded at 127) and require its size to equal the video size, otherwise raise `IdtrackeraiError` naming both sizes. If it is not set, behaviour is unchanged.

**Colab.** The notebook inherits the `.toml`, so the mask PNG must be next to it. The Colab bundle step and its on-screen instructions are updated to say so, and `roi_mask` is stored as a path relative to the `.toml` whenever that is possible.

**Exclusive ROIs.** The existing checkbox stays. Regions are the connected components of the mask, found by the same `find_exclusive_contours` call as today.

## Arena library

- Location: a per-user folder (`%APPDATA%\idtrackerai\arenas` on Windows, `~/.config/idtrackerai/arenas` elsewhere). Created on first save.
- Each arena is one directory named after the arena, holding `mask.png`, `reference.jpg` (a downscaled frame from the video it was drawn on, at most 640 px wide), and `arena.json` (name, video width and height, creation date, optional note).
- Names are unique. Saving over an existing name asks for confirmation. Names are sanitised for the file system.
- Export writes one `.zip` containing that directory. Import reads it and validates that it contains the three files and a mask that opens as an image.
- The dropdown shows each arena's name, size and a thumbnail of the mask over the reference frame.

**Applying an arena to a video.**
- Same size: the mask loads and is shown as an outline over the new video so the fit can be judged. Arrow keys nudge it by 1 px (Shift: 10 px), and dragging with the Move handle shifts it as a whole.
- Different size, same proportions (within 1%): offer to scale the mask, using nearest-neighbour resampling.
- Different proportions: refuse, and say why. The mask can still be loaded unscaled on request.

## Error handling

- Empty mask while "Regions of interest" is ticked: the existing warning on Close and track stays, and tracking is not started with an empty region.
- Unreadable or wrong-sized `roi_mask` PNG: `IdtrackeraiError` with the path and both sizes. In the app, a message box, then the ROI stays empty.
- Library folder not writable, arena corrupt, import of an invalid zip: a message box with the reason. The app keeps running and the current mask is not touched.
- Click select on a frame that fails to enhance: fall back to the raw frame, and log it.

## Testing

Unit tests (no GUI) for `roi_mask.py`:
- shape fill, brush stroke and eraser, and include/exclude semantics;
- flood select on synthetic images (a bright strip between dark walls, with noise and with a gap), and its tolerance;
- mask to polygon strings and back reproduces the mask within a small pixel tolerance, including holes and multiple regions;
- loading every ROI type the old format supports gives the same mask as `build_ROI_mask_from_list`;
- undo and redo restore exact masks, and the 30-step limit holds.

Library tests with a temporary folder: save, list, load, overwrite refusal, export, import, corrupt import.

Session tests: `roi_mask` precedence over `roi_list`, size mismatch error, unrecognised-parameter handling, and that a `.toml` round trip through `toml_format` keeps `roi_mask`.

GUI smoke test with `QT_QPA_PLATFORM=offscreen`: build the widget, simulate a drag with each tool, check the emitted mask, and check that an old `roi_list` loads.

## Open points for review

- Keyboard shortcuts could collide with existing shortcuts in the Segmentation App. The implementation plan starts by listing the existing ones.
- Whether the saved-arena preview should show the mask over the reference frame or the mask alone. The design assumes over the frame.
