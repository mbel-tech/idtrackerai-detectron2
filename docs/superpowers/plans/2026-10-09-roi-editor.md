# ROI editor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the polygon-list ROI widget of the Segmentation App with a mask-based editor (brush, eraser, click select, rectangle, ellipse, move, undo) plus a per-user library of saved arenas.

**Architecture:** One `uint8` mask (0/255, video size) is the source of truth. Pure numpy/cv2 helpers live in `idtrackerai/utils/roi_mask.py` and `roi_arenas.py` (no Qt, unit tested). `Canvas` gains an opt-in tool mode that forwards press/drag/release instead of panning. `ROI_widget.py` becomes the Qt layer. Saving writes both the derived `roi_list` polygons (for older readers) and a mask PNG referenced by a new `roi_mask` session parameter, which wins when present.

**Tech Stack:** Python 3.13, PyQt6 through `qtpy`, OpenCV (`cv2`), numpy, pytest. Spec: `docs/superpowers/specs/2026-10-09-roi-editor-design.md`.

**Conventions for every task**
- Python: `C:\Users\marti\AppData\Local\Programs\Python\Python313\python.exe`. In steps below it is written as `python`.
- Run GUI-touching tests with `QT_QPA_PLATFORM=offscreen` (PowerShell: `$env:QT_QPA_PLATFORM="offscreen"`).
- Working directory: `C:\Users\marti\Projects\idtrackeraixdetectron2`.
- Commit messages end with the line `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`.
- All work happens on branch `roi-editor`. Do **not** push to `master`: the user's Colab notebook installs from `master`.

**Decisions that differ from the spec text** (apply them; Task 7 updates the spec):
- Click select uses key **C** (W and S are remapped to up/down by `other_utils.py`).
- Pan with **middle-button drag or Ctrl+left drag** (Space already plays/pauses the video).
- Nudge uses four **panel buttons** (arrow keys step video frames).

---

## File structure

| File | Responsibility |
|---|---|
| `src/idtrackerai/utils/roi_mask.py` (new) | Mask helpers: regions (rectangle, ellipse, stroke, flood select), apply, shift, scale, PNG load/save, mask to/from `roi_list` strings, undo history |
| `src/idtrackerai/utils/roi_arenas.py` (new) | Arena library on disk: save, list, load, delete, export, import |
| `src/idtrackerai/utils/__init__.py` | Export the new helpers |
| `src/idtrackerai/utils/py_utils.py` | `load_toml` resolves a relative `roi_mask` against the `.toml` folder |
| `src/idtrackerai/session.py` | New `roi_mask` parameter, `build_roi_mask()`, temp-file cleanup |
| `src/idtrackerai/GUI_tools/widgets_utils/canvas.py` | Opt-in tool mode: `press_event`, `drag_event`, `release_event` |
| `src/idtrackerai/segmentation_app/widgets/ROI_widget.py` | The editor UI and tools (rewritten) |
| `src/idtrackerai/segmentation_app/main.py` | Wiring, save/load of `roi_mask`, Enter/Escape |
| `src/idtrackerai/segmentation_app/tooltips.toml` | New ROI tooltip |
| `src/idtrackerai/segmentation_app/widgets/detectron2_panel.py` | Colab bundle text mentions the mask PNG |
| `tests/roi_mask_test.py`, `tests/roi_arenas_test.py`, `tests/roi_session_test.py`, `tests/canvas_tool_mode_test.py`, `tests/roi_widget_test.py` (new) | Tests |

---

### Task 0: Branch

**Files:** none

- [ ] **Step 1: Create the branch**

```bash
git checkout -b roi-editor
git status --short
```
Expected: `Switched to a new branch 'roi-editor'` and empty status.

---

### Task 1: Mask helpers (`roi_mask.py`)

**Files:**
- Create: `src/idtrackerai/utils/roi_mask.py`
- Modify: `src/idtrackerai/utils/__init__.py`
- Test: `tests/roi_mask_test.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/roi_mask_test.py`:

```python
"""Mask helpers behind the ROI editor."""

import cv2
import numpy as np
import pytest

from idtrackerai.utils import build_ROI_mask_from_list
from idtrackerai.utils.roi_mask import (
    OFF,
    ON,
    MaskHistory,
    apply_region,
    ellipse_region,
    empty_mask,
    flood_region,
    load_mask_png,
    mask_to_roi_list,
    proportions_match,
    rectangle_region,
    save_mask_png,
    scale_mask,
    shift_mask,
    stroke_region,
)


def iou(a, b):
    a, b = a > 0, b > 0
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else np.logical_and(a, b).sum() / union


def test_apply_region_include_and_exclude():
    mask = empty_mask(50, 40)
    region = rectangle_region(mask.shape, (10, 10), (30, 20))
    apply_region(mask, region, include=True)
    assert mask[15, 20] == ON and mask[5, 5] == OFF
    apply_region(mask, ellipse_region(mask.shape, (20, 15), (4, 4), 0), include=False)
    assert mask[15, 20] == OFF and mask[11, 11] == ON


def test_rectangle_region_accepts_corners_in_any_order():
    shape = (40, 50)
    a = rectangle_region(shape, (10, 10), (30, 20))
    b = rectangle_region(shape, (30, 20), (10, 10))
    assert np.array_equal(a, b) and a.sum() > 0


def test_stroke_region_single_point_and_line():
    shape = (60, 80)
    dot = stroke_region(shape, [(40, 30)], radius=5)
    assert dot[30, 40] and not dot[30, 55]
    line = stroke_region(shape, [(10, 30), (70, 30)], radius=3)
    assert line[30, 10] and line[30, 40] and line[30, 70] and not line[10, 40]


def test_stroke_region_is_clipped_to_the_frame():
    region = stroke_region((20, 20), [(-5, -5), (30, 30)], radius=4)
    assert region.shape == (20, 20) and region.any()


def strip_frame():
    rng = np.random.default_rng(0)
    gray = np.full((120, 200), 70, np.uint8)
    gray[20:30, :] = 220
    gray[70:80, :] = 220
    return np.clip(gray + rng.normal(0, 4, gray.shape), 0, 255).astype(np.uint8)


def test_flood_region_fills_the_strip_between_walls():
    region = flood_region(strip_frame(), (100, 50), tolerance=12)
    rows = np.nonzero(region.any(axis=1))[0]
    assert 28 <= rows[0] <= 36 and 64 <= rows[-1] <= 72
    assert region[50, 5] and region[50, 195]
    assert not region[10, 100] and not region[100, 100]


def test_flood_region_low_tolerance_selects_less():
    frame = strip_frame()
    loose = flood_region(frame, (100, 50), tolerance=30).sum()
    tight = flood_region(frame, (100, 50), tolerance=1).sum()
    assert tight < loose


def test_flood_region_outside_the_frame_is_empty():
    assert not flood_region(strip_frame(), (-3, 500), tolerance=12).any()


def test_flood_region_fills_tiny_holes():
    frame = np.full((60, 60), 80, np.uint8)
    frame[30, 30] = 250  # one noisy pixel
    region = flood_region(frame, (5, 5), tolerance=10)
    assert region[30, 30]


def test_mask_to_roi_list_round_trip_with_hole_and_island():
    mask = empty_mask(300, 200)
    cv2.rectangle(mask, (20, 20), (280, 180), ON, -1)
    cv2.rectangle(mask, (60, 60), (240, 140), OFF, -1)
    cv2.rectangle(mask, (120, 90), (160, 110), ON, -1)
    lines = mask_to_roi_list(mask)
    assert lines[0].startswith("+ Polygon") and any(l.startswith("- Polygon") for l in lines)
    rebuilt = build_ROI_mask_from_list(lines, 300, 200)
    assert iou(mask, rebuilt) > 0.97


def test_mask_to_roi_list_empty_mask_gives_empty_list():
    assert mask_to_roi_list(empty_mask(30, 20)) == []


def test_mask_to_roi_list_two_separate_regions():
    mask = empty_mask(200, 100)
    cv2.rectangle(mask, (10, 10), (80, 40), ON, -1)
    cv2.rectangle(mask, (110, 50), (190, 90), ON, -1)
    lines = mask_to_roi_list(mask)
    assert sum(l.startswith("+") for l in lines) == 2
    assert iou(mask, build_ROI_mask_from_list(lines, 200, 100)) > 0.97


def test_png_round_trip_thresholds_at_127(tmp_path):
    mask = empty_mask(40, 30)
    mask[5:15, 5:15] = ON
    path = tmp_path / "ñ mask.png"  # a non-ASCII path on purpose
    save_mask_png(path, mask)
    assert np.array_equal(load_mask_png(path), mask)
    gray = np.full((30, 40), 100, np.uint8)
    gray[:5] = 200
    cv2.imencode(".png", gray)[1].tofile(tmp_path / "g.png")
    loaded = load_mask_png(tmp_path / "g.png")
    assert set(np.unique(loaded)) == {OFF, ON} and loaded[0, 0] == ON and loaded[10, 0] == OFF


def test_load_mask_png_rejects_unreadable_file(tmp_path):
    (tmp_path / "bad.png").write_bytes(b"not an image")
    with pytest.raises(ValueError):
        load_mask_png(tmp_path / "bad.png")


def test_shift_mask_moves_and_does_not_wrap():
    mask = empty_mask(30, 30)
    mask[10:12, 10:12] = ON
    moved = shift_mask(mask, 5, -3)
    assert moved[7, 15] == ON and moved[10, 10] == OFF
    edge = shift_mask(mask, 25, 0)
    assert edge[10, 35 - 30] == OFF and edge.sum() < mask.sum() + 1


def test_scale_mask_and_proportions():
    mask = empty_mask(100, 50)
    mask[10:20, 10:30] = ON
    scaled = scale_mask(mask, 200, 100)
    assert scaled.shape == (100, 200) and scaled[30, 40] == ON
    assert set(np.unique(scaled)) <= {OFF, ON}
    assert proportions_match(100, 50, 200, 100)
    assert proportions_match(100, 50, 202, 100, tolerance=0.02)
    assert not proportions_match(100, 50, 100, 100)


def test_history_undo_redo_restores_exact_masks():
    mask = empty_mask(60, 40)
    history = MaskHistory(limit=30)
    snapshots = [mask.copy()]
    for x in (5, 20, 40):
        before = mask.copy()
        apply_region(mask, rectangle_region(mask.shape, (x, 5), (x + 8, 15)), True)
        history.record(before, mask)
        snapshots.append(mask.copy())
    for expected in reversed(snapshots[:-1]):
        assert history.undo(mask)
        assert np.array_equal(mask, expected)
    assert not history.undo(mask)
    for expected in snapshots[1:]:
        assert history.redo(mask)
        assert np.array_equal(mask, expected)
    assert not history.redo(mask)


def test_history_ignores_no_ops_and_respects_the_limit():
    mask = empty_mask(30, 30)
    history = MaskHistory(limit=3)
    history.record(mask.copy(), mask)  # nothing changed
    assert not history.can_undo
    for i in range(6):
        before = mask.copy()
        mask[i, i] = ON
        history.record(before, mask)
    undone = 0
    while history.undo(mask):
        undone += 1
    assert undone == 3


def test_new_edit_after_undo_drops_the_redo_branch():
    mask = empty_mask(30, 30)
    history = MaskHistory()
    before = mask.copy()
    mask[1, 1] = ON
    history.record(before, mask)
    history.undo(mask)
    before = mask.copy()
    mask[2, 2] = ON
    history.record(before, mask)
    assert not history.can_redo
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/roi_mask_test.py -q -p no:cacheprovider`
Expected: collection error `ModuleNotFoundError: No module named 'idtrackerai.utils.roi_mask'`.

- [ ] **Step 3: Write the implementation**

Create `src/idtrackerai/utils/roi_mask.py`:

```python
"""Mask operations behind the ROI editor.

The region of interest is one uint8 mask the size of the video, 255 inside and
0 outside. Everything here is numpy and OpenCV only, so it is tested without
a GUI and the Session can read a saved mask without importing the app.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ON = 255
OFF = 0


def empty_mask(width: int, height: int) -> np.ndarray:
    return np.zeros((height, width), np.uint8)


# ------------------------------------------------------------------ regions
def rectangle_region(
    shape: tuple[int, int], p0: tuple[float, float], p1: tuple[float, float]
) -> np.ndarray:
    """A filled rectangle between two corners given in any order."""
    region = np.zeros(shape, np.uint8)
    x0, x1 = sorted((int(round(p0[0])), int(round(p1[0]))))
    y0, y1 = sorted((int(round(p0[1])), int(round(p1[1]))))
    cv2.rectangle(region, (x0, y0), (x1, y1), ON, -1)
    return region


def ellipse_region(
    shape: tuple[int, int],
    center: tuple[float, float],
    axes: tuple[float, float],
    angle: float = 0.0,
) -> np.ndarray:
    region = np.zeros(shape, np.uint8)
    cv2.ellipse(
        region,
        (int(round(center[0])), int(round(center[1]))),
        (max(int(round(axes[0])), 0), max(int(round(axes[1])), 0)),
        angle,
        0,
        360,
        ON,
        -1,
    )
    return region


def stroke_region(
    shape: tuple[int, int], points: list[tuple[float, float]], radius: float
) -> np.ndarray:
    """A thick line through the points with round ends. One point is a dot."""
    region = np.zeros(shape, np.uint8)
    radius = max(int(round(radius)), 1)
    pts = [(int(round(x)), int(round(y))) for x, y in points]
    for a, b in zip(pts, pts[1:]):
        cv2.line(region, a, b, ON, thickness=2 * radius)
    for p in pts:
        cv2.circle(region, p, radius, ON, -1)
    return region


def flood_region(
    gray: np.ndarray,
    seed: tuple[float, float],
    tolerance: float,
    blur: bool = True,
    min_hole: int = 25,
) -> np.ndarray:
    """The connected area around ``seed`` whose brightness is within
    ``tolerance`` of the seed pixel (a magic wand).

    Brightness is compared to the seed, not to the neighbour, so a gradient
    cannot leak the selection across a wall. The frame is blurred first so
    sensor noise does not fragment the area, and holes smaller than
    ``min_hole`` pixels are filled.
    """
    height, width = gray.shape[:2]
    x, y = int(round(seed[0])), int(round(seed[1]))
    if not (0 <= x < width and 0 <= y < height):
        return np.zeros((height, width), np.uint8)

    work = cv2.GaussianBlur(gray, (5, 5), 0) if blur else gray.copy()
    fill = np.zeros((height + 2, width + 2), np.uint8)
    flags = 4 | (ON << 8) | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE
    tol = float(tolerance)
    cv2.floodFill(work, fill, (x, y), 0, tol, tol, flags)
    region = np.ascontiguousarray(fill[1:-1, 1:-1])

    region = cv2.medianBlur(region, 5)
    holes = cv2.bitwise_not(region)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=4)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] < min_hole:
            region[labels == label] = ON
    return region


def apply_region(mask: np.ndarray, region: np.ndarray, include: bool) -> None:
    """Adds the region to the mask, or removes it, in place."""
    mask[region > 0] = ON if include else OFF


# ------------------------------------------------------------ transformations
def shift_mask(mask: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """The mask moved by whole pixels. Whatever leaves the frame is lost."""
    height, width = mask.shape
    matrix = np.float32([[1, 0, round(dx)], [0, 1, round(dy)]])
    return cv2.warpAffine(
        mask, matrix, (width, height), flags=cv2.INTER_NEAREST, borderValue=OFF
    )


def scale_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)


def proportions_match(
    width_a: int, height_a: int, width_b: int, height_b: int, tolerance: float = 0.01
) -> bool:
    """Whether two frame sizes have the same aspect ratio within a fraction."""
    ratio = (width_b / height_b) / (width_a / height_a)
    return abs(ratio - 1.0) <= tolerance


# ------------------------------------------------------------------- files
def save_mask_png(path: Path | str, mask: np.ndarray) -> None:
    # imencode + tofile: cv2.imwrite fails on some non-ASCII Windows paths
    cv2.imencode(".png", mask)[1].tofile(str(path))


def load_mask_png(path: Path | str) -> np.ndarray:
    """A saved mask as 0/255. Raises ValueError if the file is not an image."""
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE) if data.size else None
    if image is None:
        raise ValueError(f"{path} could not be read as an image")
    return np.where(image > 127, ON, OFF).astype(np.uint8)


# ----------------------------------------------------------- roi_list strings
def mask_to_roi_list(mask: np.ndarray, epsilon: float = 0.5) -> list[str]:
    """The mask as ``"+ Polygon [[x, y], ...]"`` lines, the format of ``roi_list``.

    Nested regions alternate: an outline is ``+``, a hole in it ``-``, an island
    in the hole ``+`` again. Lines are ordered outermost first, because the
    lines are applied in order. This is an approximation within about a pixel
    of the edge; the mask PNG is the exact record.
    """
    contours, hierarchy = cv2.findContours(
        np.ascontiguousarray(mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
    )
    if hierarchy is None:
        return []
    hierarchy = hierarchy[0]

    found: list[tuple[int, str]] = []
    for index, contour in enumerate(contours):
        depth, parent = 0, hierarchy[index][3]
        while parent != -1:
            depth += 1
            parent = hierarchy[parent][3]
        points = cv2.approxPolyDP(contour, epsilon, True)[:, 0, :]
        if len(points) < 3:
            continue
        sign = "+" if depth % 2 == 0 else "-"
        vertices = ", ".join(f"[{x:.1f}, {y:.1f}]" for x, y in points)
        found.append((depth, f"{sign} Polygon [{vertices}]"))
    found.sort(key=lambda item: item[0])
    return [line for _, line in found]


# ------------------------------------------------------------------ history
class MaskHistory:
    """Undo and redo for a mask, storing only the changed rectangle per step."""

    def __init__(self, limit: int = 30):
        self.limit = limit
        self._undo: list[tuple[slice, slice, np.ndarray, np.ndarray]] = []
        self._redo: list[tuple[slice, slice, np.ndarray, np.ndarray]] = []

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()

    def record(self, before: np.ndarray, after: np.ndarray) -> None:
        """Remember a change that has already been applied to the mask."""
        changed = before != after
        if not changed.any():
            return
        rows = np.nonzero(changed.any(axis=1))[0]
        cols = np.nonzero(changed.any(axis=0))[0]
        ys = slice(int(rows[0]), int(rows[-1]) + 1)
        xs = slice(int(cols[0]), int(cols[-1]) + 1)
        self._undo.append((ys, xs, before[ys, xs].copy(), after[ys, xs].copy()))
        del self._undo[: -self.limit]
        self._redo.clear()

    def undo(self, mask: np.ndarray) -> bool:
        if not self._undo:
            return False
        ys, xs, before, after = self._undo.pop()
        mask[ys, xs] = before
        self._redo.append((ys, xs, before, after))
        return True

    def redo(self, mask: np.ndarray) -> bool:
        if not self._redo:
            return False
        ys, xs, before, after = self._redo.pop()
        mask[ys, xs] = after
        self._undo.append((ys, xs, before, after))
        return True
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/roi_mask_test.py -q -p no:cacheprovider`
Expected: all tests pass (about 17).

- [ ] **Step 5: Commit**

```bash
git add src/idtrackerai/utils/roi_mask.py tests/roi_mask_test.py
git commit -m "Add mask helpers for the ROI editor

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Arena library (`roi_arenas.py`)

**Files:**
- Create: `src/idtrackerai/utils/roi_arenas.py`
- Test: `tests/roi_arenas_test.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/roi_arenas_test.py`:

```python
"""The on-disk library of saved arenas."""

import json
import zipfile

import cv2
import numpy as np
import pytest

from idtrackerai.utils.roi_arenas import (
    ArenaError,
    delete_arena,
    export_arena,
    import_arena,
    list_arenas,
    load_arena,
    sanitize_name,
    save_arena,
)


@pytest.fixture
def mask():
    m = np.zeros((60, 80), np.uint8)
    m[10:30, 20:60] = 255
    return m


@pytest.fixture
def frame():
    return np.random.default_rng(0).integers(0, 255, (60, 80, 3), dtype=np.uint8)


def test_sanitize_name():
    assert sanitize_name("  Labyrinth, 3 levels ") == "Labyrinth, 3 levels"
    assert sanitize_name('a/b\\c:d*e?f"g<h>i|j') == "a_b_c_d_e_f_g_h_i_j"
    with pytest.raises(ArenaError):
        sanitize_name("   ")
    with pytest.raises(ArenaError):
        sanitize_name("..")


def test_save_list_and_load_round_trip(tmp_path, mask, frame):
    saved = save_arena(tmp_path, "Tank A", mask, frame, note="north camera")
    assert (saved.width, saved.height) == (80, 60)
    arenas = list_arenas(tmp_path)
    assert [a.name for a in arenas] == ["Tank A"]
    arena, loaded = load_arena(tmp_path, "Tank A")
    assert arena.note == "north camera"
    assert np.array_equal(loaded, mask)
    assert (tmp_path / "Tank A" / "reference.jpg").is_file()


def test_reference_frame_is_downscaled(tmp_path, mask):
    big = np.zeros((1080, 1920, 3), np.uint8)
    big_mask = cv2.resize(mask, (1920, 1080), interpolation=cv2.INTER_NEAREST)
    save_arena(tmp_path, "Big", big_mask, big)
    reference = cv2.imdecode(
        np.fromfile(tmp_path / "Big" / "reference.jpg", dtype=np.uint8), cv2.IMREAD_COLOR
    )
    assert reference.shape[1] <= 640


def test_saving_without_a_frame_is_allowed(tmp_path, mask):
    save_arena(tmp_path, "NoFrame", mask, None)
    assert not (tmp_path / "NoFrame" / "reference.jpg").exists()
    assert list_arenas(tmp_path)[0].name == "NoFrame"


def test_overwrite_needs_permission(tmp_path, mask, frame):
    save_arena(tmp_path, "A", mask, frame)
    with pytest.raises(ArenaError):
        save_arena(tmp_path, "A", mask, frame)
    other = np.zeros_like(mask)
    other[0:10, 0:10] = 255
    save_arena(tmp_path, "A", other, frame, overwrite=True)
    assert np.array_equal(load_arena(tmp_path, "A")[1], other)


def test_empty_mask_is_refused(tmp_path):
    with pytest.raises(ArenaError):
        save_arena(tmp_path, "Empty", np.zeros((10, 10), np.uint8), None)


def test_load_missing_arena(tmp_path):
    with pytest.raises(ArenaError):
        load_arena(tmp_path, "nope")


def test_list_skips_broken_folders(tmp_path, mask, frame):
    save_arena(tmp_path, "Good", mask, frame)
    (tmp_path / "Broken").mkdir()
    (tmp_path / "Broken" / "arena.json").write_text("{not json")
    assert [a.name for a in list_arenas(tmp_path)] == ["Good"]


def test_list_on_a_missing_folder_is_empty(tmp_path):
    assert list_arenas(tmp_path / "does_not_exist") == []


def test_delete(tmp_path, mask, frame):
    save_arena(tmp_path, "A", mask, frame)
    delete_arena(tmp_path, "A")
    assert list_arenas(tmp_path) == []
    with pytest.raises(ArenaError):
        delete_arena(tmp_path, "A")


def test_export_import_round_trip(tmp_path, mask, frame):
    library, other = tmp_path / "lib", tmp_path / "other"
    save_arena(library, "A", mask, frame, note="n")
    archive = tmp_path / "a.zip"
    export_arena(library, "A", archive)
    imported = import_arena(other, archive)
    assert imported.name == "A" and imported.note == "n"
    assert np.array_equal(load_arena(other, "A")[1], mask)
    with pytest.raises(ArenaError):
        import_arena(other, archive)
    import_arena(other, archive, overwrite=True)


def test_import_rejects_unexpected_members(tmp_path, mask, frame):
    save_arena(tmp_path / "lib", "A", mask, frame)
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for name in ("mask.png", "arena.json"):
            zf.write(tmp_path / "lib" / "A" / name, name)
        zf.writestr("../escape.txt", "x")
    with pytest.raises(ArenaError):
        import_arena(tmp_path / "other", archive)
    assert not (tmp_path / "escape.txt").exists()


def test_import_rejects_a_zip_without_a_valid_mask(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("mask.png", b"not an image")
        zf.writestr("arena.json", json.dumps({"name": "X", "width": 1, "height": 1}))
    with pytest.raises(ArenaError):
        import_arena(tmp_path / "lib", archive)


def test_import_rejects_a_file_that_is_not_a_zip(tmp_path):
    (tmp_path / "x.zip").write_bytes(b"nope")
    with pytest.raises(ArenaError):
        import_arena(tmp_path / "lib", tmp_path / "x.zip")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/roi_arenas_test.py -q -p no:cacheprovider`
Expected: `ModuleNotFoundError: No module named 'idtrackerai.utils.roi_arenas'`.

- [ ] **Step 3: Write the implementation**

Create `src/idtrackerai/utils/roi_arenas.py`:

```python
"""A per-user library of saved arenas (region-of-interest masks).

Each arena is a folder named after it, holding ``mask.png``, ``arena.json`` and
optionally ``reference.jpg``, a downscaled frame of the video it was drawn on.
No Qt here, so it is tested directly.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .roi_mask import load_mask_png, save_mask_png

MASK_FILE = "mask.png"
INFO_FILE = "arena.json"
REFERENCE_FILE = "reference.jpg"
MEMBERS = {MASK_FILE, INFO_FILE, REFERENCE_FILE}
REFERENCE_MAX_WIDTH = 640
_FORBIDDEN = re.compile(r'[\\/:*?"<>|\x00-\x1f]')


class ArenaError(Exception):
    """The library could not do what was asked; the message says why."""


class ArenaExistsError(ArenaError):
    """An arena with that name is already saved (the caller may offer to replace it)."""


@dataclass
class Arena:
    name: str
    width: int
    height: int
    created: str = ""
    note: str = ""


def default_arenas_dir() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
        return base / "idtrackerai" / "arenas"
    return Path.home() / ".config" / "idtrackerai" / "arenas"


def sanitize_name(name: str) -> str:
    cleaned = _FORBIDDEN.sub("_", name.strip()).strip(" .")
    if not cleaned:
        raise ArenaError("An arena needs a name.")
    return cleaned


def reference_path(root: Path, name: str) -> Path:
    return Path(root) / name / REFERENCE_FILE


def _info_of(folder: Path) -> Arena:
    try:
        data = json.loads((folder / INFO_FILE).read_text(encoding="utf-8"))
        return Arena(
            name=str(data["name"]),
            width=int(data["width"]),
            height=int(data["height"]),
            created=str(data.get("created", "")),
            note=str(data.get("note", "")),
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ArenaError(f"{folder.name} is not a readable arena ({exc})") from exc


def list_arenas(root: Path | str) -> list[Arena]:
    root = Path(root)
    if not root.is_dir():
        return []
    arenas = []
    for folder in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if not folder.is_dir():
            continue
        try:
            arenas.append(_info_of(folder))
        except ArenaError:
            continue  # a broken folder must not hide the good ones
    return arenas


def save_arena(
    root: Path | str,
    name: str,
    mask: np.ndarray,
    reference_frame: np.ndarray | None = None,
    note: str = "",
    overwrite: bool = False,
) -> Arena:
    root = Path(root)
    name = sanitize_name(name)
    if not mask.any():
        raise ArenaError("The region is empty, so there is nothing to save.")
    folder = root / name
    if folder.exists() and not overwrite:
        raise ArenaExistsError(f"An arena called {name!r} already exists.")

    height, width = mask.shape
    arena = Arena(name, width, height, datetime.now().isoformat(timespec="seconds"), note)
    try:
        root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".saving-", dir=root))
        try:
            save_mask_png(staging / MASK_FILE, mask)
            if reference_frame is not None and reference_frame.size:
                frame = reference_frame
                if frame.shape[1] > REFERENCE_MAX_WIDTH:
                    scale = REFERENCE_MAX_WIDTH / frame.shape[1]
                    frame = cv2.resize(
                        frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
                    )
                cv2.imencode(".jpg", frame)[1].tofile(str(staging / REFERENCE_FILE))
            (staging / INFO_FILE).write_text(
                json.dumps(arena.__dict__, indent=2), encoding="utf-8"
            )
            if folder.exists():
                shutil.rmtree(folder)
            staging.rename(folder)
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
    except OSError as exc:
        raise ArenaError(f"Could not save the arena: {exc}") from exc
    return arena


def load_arena(root: Path | str, name: str) -> tuple[Arena, np.ndarray]:
    folder = Path(root) / name
    if not folder.is_dir():
        raise ArenaError(f"There is no arena called {name!r}.")
    arena = _info_of(folder)
    try:
        mask = load_mask_png(folder / MASK_FILE)
    except (OSError, ValueError) as exc:
        raise ArenaError(f"The mask of {name!r} could not be read ({exc})") from exc
    return arena, mask


def delete_arena(root: Path | str, name: str) -> None:
    folder = Path(root) / name
    if not folder.is_dir():
        raise ArenaError(f"There is no arena called {name!r}.")
    try:
        shutil.rmtree(folder)
    except OSError as exc:
        raise ArenaError(f"Could not delete {name!r}: {exc}") from exc


def export_arena(root: Path | str, name: str, zip_path: Path | str) -> None:
    folder = Path(root) / name
    if not folder.is_dir():
        raise ArenaError(f"There is no arena called {name!r}.")
    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for member in MEMBERS:
                if (folder / member).is_file():
                    archive.write(folder / member, member)
    except OSError as exc:
        raise ArenaError(f"Could not write {zip_path}: {exc}") from exc


def import_arena(root: Path | str, zip_path: Path | str, overwrite: bool = False) -> Arena:
    try:
        archive = zipfile.ZipFile(zip_path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise ArenaError(f"{zip_path} is not a zip file ({exc})") from exc
    with archive:
        names = set(archive.namelist())
        if not names <= MEMBERS or not {MASK_FILE, INFO_FILE} <= names:
            raise ArenaError(
                "This zip is not an exported arena: it must hold "
                f"{MASK_FILE} and {INFO_FILE} and nothing else but {REFERENCE_FILE}."
            )
        with tempfile.TemporaryDirectory() as tmp:
            for member in names:  # fixed, vetted names only: no path from the zip is used
                Path(tmp, member).write_bytes(archive.read(member))
            try:
                info = _info_of(Path(tmp))
                mask = load_mask_png(Path(tmp) / MASK_FILE)
            except (ArenaError, ValueError) as exc:
                raise ArenaError(f"The arena in this zip is damaged ({exc})") from exc
            reference = None
            if REFERENCE_FILE in names:
                reference = cv2.imdecode(
                    np.fromfile(str(Path(tmp) / REFERENCE_FILE), dtype=np.uint8),
                    cv2.IMREAD_COLOR,
                )
            return save_arena(
                root, info.name, mask, reference, info.note, overwrite=overwrite
            )
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/roi_arenas_test.py -q -p no:cacheprovider`
Expected: all pass (14 tests). If `test_reference_frame_is_downscaled` fails on the width, check that `cv2.resize` is given `fx`/`fy`, not `dsize`.

- [ ] **Step 5: Commit**

```bash
git add src/idtrackerai/utils/roi_arenas.py tests/roi_arenas_test.py
git commit -m "Add the saved-arena library for the ROI editor

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Session `roi_mask` parameter

**Files:**
- Modify: `src/idtrackerai/session.py` (import near line 17-30, field after line 114, line 375)
- Modify: `src/idtrackerai/utils/py_utils.py:133-135` (`load_toml`)
- Test: `tests/roi_session_test.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/roi_session_test.py`:

```python
"""The exact ROI mask is a session parameter that wins over roi_list."""

import cv2
import numpy as np
import pytest

from idtrackerai import IdtrackeraiError, Session
from idtrackerai.utils import load_toml
from idtrackerai.utils.roi_mask import empty_mask, save_mask_png

ROI_LIST = ["+ Polygon [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]"]


def make_session(width=60, height=40):
    session = Session()
    session.width, session.height = width, height
    return session


def test_roi_mask_is_a_recognised_parameter():
    assert Session().set_parameters(roi_mask="x.png") == set()


def test_roi_mask_png_wins_over_roi_list(tmp_path):
    mask = empty_mask(60, 40)
    mask[20:30, 30:50] = 255
    save_mask_png(tmp_path / "m.png", mask)
    session = make_session()
    session.roi_list = ROI_LIST
    session.roi_mask = str(tmp_path / "m.png")
    assert np.array_equal(session.build_roi_mask(), mask)


def test_without_roi_mask_the_polygons_are_used():
    session = make_session()
    session.roi_list = ROI_LIST
    mask = session.build_roi_mask()
    assert mask[5, 5] == 255 and mask[30, 50] == 0


def test_no_roi_at_all_is_none():
    assert make_session().build_roi_mask() is None


def test_size_mismatch_names_both_sizes(tmp_path):
    save_mask_png(tmp_path / "m.png", empty_mask(100, 100))
    session = make_session(60, 40)
    session.roi_mask = str(tmp_path / "m.png")
    with pytest.raises(IdtrackeraiError, match="100x100.*60x40"):
        session.build_roi_mask()


def test_unreadable_mask_is_an_idtrackerai_error(tmp_path):
    (tmp_path / "m.png").write_bytes(b"nope")
    session = make_session()
    session.roi_mask = str(tmp_path / "m.png")
    with pytest.raises(IdtrackeraiError):
        session.build_roi_mask()


def test_temporary_mask_is_removed_after_use(tmp_path):
    temp = tmp_path / "video.tmp_roi_mask.png"
    save_mask_png(temp, empty_mask(10, 10))
    session = make_session()
    session.roi_mask = str(temp)
    session.discard_temporary_roi_mask()
    assert not temp.exists() and session.roi_mask is None


def test_a_normal_mask_file_is_kept(tmp_path):
    keep = tmp_path / "arena_roi.png"
    save_mask_png(keep, empty_mask(10, 10))
    session = make_session()
    session.roi_mask = str(keep)
    session.discard_temporary_roi_mask()
    assert keep.exists() and session.roi_mask == str(keep)


def test_load_toml_resolves_a_relative_roi_mask(tmp_path):
    folder = tmp_path / "setups"
    folder.mkdir()
    (folder / "a.toml").write_text('roi_mask = "a_roi.png"\n', encoding="utf-8")
    loaded = load_toml(folder / "a.toml")
    assert loaded["roi_mask"] == str((folder / "a_roi.png").resolve())


def test_load_toml_keeps_an_absolute_roi_mask(tmp_path):
    absolute = (tmp_path / "m.png").resolve()
    (tmp_path / "a.toml").write_text(f"roi_mask = '{absolute}'\n", encoding="utf-8")
    assert load_toml(tmp_path / "a.toml")["roi_mask"] == str(absolute)
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/roi_session_test.py -q -p no:cacheprovider`
Expected: failures, `AttributeError: ... 'build_roi_mask'` and the recognised-parameter test failing.

- [ ] **Step 3: Implement in `session.py`**

Add the import next to the other `.utils` imports (after the existing `from .utils import (...)` block):

```python
from .utils.roi_mask import load_mask_png
```

Add the field directly after `roi_list: list[str] | str | None = None` (line 114):

```python
    roi_mask: Path | str | None = None
    """Path to a PNG mask (white = tracked) that replaces ``roi_list`` when set.
    It holds the exact pixels the Segmentation App's ROI editor produced, while
    ``roi_list`` is only a polygon approximation kept for older readers."""
```

Replace line 375 (`self.ROI_mask = build_ROI_mask_from_list(self.roi_list, self.width, self.height)`) with:

```python
        self.ROI_mask = self.build_roi_mask()
        self.discard_temporary_roi_mask()
```

Add these two methods to `Session`, directly above `def bkg_model` property block (before `@property\n    def bkg_model`):

```python
    TEMPORARY_ROI_MASK_SUFFIX = ".tmp_roi_mask.png"

    def build_roi_mask(self) -> np.ndarray | None:
        """The region of interest as a 0/255 mask, or None when there is none.

        The exact mask file wins over the polygon list.
        """
        if self.roi_mask:
            path = resolve_path(self.roi_mask)
            try:
                mask = load_mask_png(path)
            except (OSError, ValueError) as exc:
                raise IdtrackeraiError(
                    f"Could not read the ROI mask {path}: {exc}"
                ) from exc
            if mask.shape != (self.height, self.width):
                raise IdtrackeraiError(
                    f"The ROI mask {path.name} is {mask.shape[1]}x{mask.shape[0]} "
                    f"but the video is {self.width}x{self.height}."
                )
            return mask
        return build_ROI_mask_from_list(self.roi_list, self.width, self.height)

    def discard_temporary_roi_mask(self) -> None:
        """Delete the mask the Segmentation App wrote beside the video for this
        run. The session keeps its own copy in the preprocessing folder."""
        if self.roi_mask and str(self.roi_mask).endswith(
            self.TEMPORARY_ROI_MASK_SUFFIX
        ):
            Path(self.roi_mask).unlink(missing_ok=True)
            self.roi_mask = None
```

- [ ] **Step 4: Implement in `py_utils.py`**

In `load_toml`, directly after the loop that turns empty strings into `None` (the `for key, value in toml_dict.items(): if value == "": toml_dict[key] = None` block), add:

```python
        roi_mask = toml_dict.get("roi_mask")
        if roi_mask and not Path(roi_mask).is_absolute():
            # relative to the .toml, so a setup folder can be copied as a whole
            toml_dict["roi_mask"] = str(path.resolve().parent / roi_mask)
```

- [ ] **Step 5: Run the tests**

Run: `python -m pytest tests/roi_session_test.py tests/enhancement_settings_test.py -q -p no:cacheprovider`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/idtrackerai/session.py src/idtrackerai/utils/py_utils.py tests/roi_session_test.py
git commit -m "Add roi_mask session parameter, preferred over roi_list

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Canvas tool mode

**Files:**
- Modify: `src/idtrackerai/GUI_tools/widgets_utils/canvas.py` (signals at line 72-77, `__init__` at 84-99, mouse handlers 150-189)
- Test: `tests/canvas_tool_mode_test.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/canvas_tool_mode_test.py`:

```python
"""In tool mode a left drag goes to the tool, and the view pans another way."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from qtpy.QtCore import QEvent, QPointF, Qt
from qtpy.QtGui import QMouseEvent
from qtpy.QtWidgets import QApplication

from idtrackerai.GUI_tools.widgets_utils.canvas import Canvas

app = QApplication.instance() or QApplication([])
LEFT, MIDDLE, NONE = (
    Qt.MouseButton.LeftButton,
    Qt.MouseButton.MiddleButton,
    Qt.MouseButton.NoButton,
)


def mouse(kind, x, y, button=LEFT, held=LEFT, mods=Qt.KeyboardModifier.NoModifier):
    return QMouseEvent(kind, QPointF(x, y), button, held, mods)


def press(c, x, y, button=LEFT, mods=Qt.KeyboardModifier.NoModifier):
    c.mousePressEvent(mouse(QEvent.Type.MouseButtonPress, x, y, button, button, mods))


def move(c, x, y, held=LEFT):
    c.mouseMoveEvent(mouse(QEvent.Type.MouseMove, x, y, NONE, held))


def release(c, x, y, button=LEFT):
    c.mouseReleaseEvent(mouse(QEvent.Type.MouseButtonRelease, x, y, button, NONE))


@pytest.fixture
def canvas():
    c = Canvas()
    c.resize(200, 100)
    c.show()
    c.grab()  # a paint is what gives the canvas its coordinate mapping
    return c


@pytest.fixture
def seen(canvas):
    log = []
    canvas.press_event.connect(lambda e: log.append(("press", e.xy_data)))
    canvas.drag_event.connect(lambda e: log.append(("drag", e.xy_data)))
    canvas.release_event.connect(lambda e: log.append(("release", e.xy_data)))
    canvas.click_event.connect(lambda e: log.append(("click", e.xy_data)))
    return log


def test_default_mode_drag_pans_and_sends_no_tool_events(canvas, seen):
    before = canvas.centerX
    press(canvas, 50, 50)
    move(canvas, 70, 50)
    release(canvas, 70, 50)
    assert canvas.centerX != before
    assert not [e for e in seen if e[0] in ("press", "drag", "release")]


def test_tool_mode_sends_press_drag_release_and_does_not_pan(canvas, seen):
    canvas.set_tool_mode(True)
    before = (canvas.centerX, canvas.centerY)
    press(canvas, 50, 50)
    move(canvas, 60, 55)
    move(canvas, 70, 60)
    release(canvas, 70, 60)
    assert [name for name, _ in seen] == ["press", "drag", "drag", "release"]
    assert (canvas.centerX, canvas.centerY) == before


def test_tool_mode_middle_button_pans(canvas, seen):
    canvas.set_tool_mode(True)
    before = canvas.centerX
    press(canvas, 50, 50, MIDDLE)
    move(canvas, 70, 50, MIDDLE)
    release(canvas, 70, 50, MIDDLE)
    assert canvas.centerX != before
    assert not [e for e in seen if e[0] in ("press", "drag", "release")]


def test_tool_mode_ctrl_left_pans(canvas, seen):
    canvas.set_tool_mode(True)
    before = canvas.centerX
    press(canvas, 50, 50, LEFT, Qt.KeyboardModifier.ControlModifier)
    move(canvas, 70, 50)
    release(canvas, 70, 50)
    assert canvas.centerX != before
    assert not [e for e in seen if e[0] in ("press", "drag", "release")]


def test_leaving_tool_mode_restores_click_events(canvas, seen):
    canvas.set_tool_mode(True)
    canvas.set_tool_mode(False)
    press(canvas, 50, 50)
    release(canvas, 50, 50)
    assert [name for name, _ in seen] == ["click"]


def test_tool_mode_cursor_is_a_crosshair(canvas):
    canvas.set_tool_mode(True)
    assert canvas.cursor().shape() == Qt.CursorShape.CrossCursor
    canvas.set_tool_mode(False)
    assert canvas.cursor().shape() == Qt.CursorShape.PointingHandCursor
```

- [ ] **Step 2: Run to verify they fail**

Run: `$env:QT_QPA_PLATFORM="offscreen"; python -m pytest tests/canvas_tool_mode_test.py -q -p no:cacheprovider`
Expected: `AttributeError: 'Canvas' object has no attribute 'press_event'` or `set_tool_mode`.

- [ ] **Step 3: Implement**

In `canvas.py`, after the `painting_time = Signal(CanvasPainter)` line add:

```python
    press_event = Signal(CanvasMouseEvent)
    drag_event = Signal(CanvasMouseEvent)
    release_event = Signal(CanvasMouseEvent)
    """Left press, drag and release, emitted instead of panning while
    :meth:`set_tool_mode` is on. Pan then with the middle button or Ctrl + left."""
```

In `__init__`, replace `self.setCursor(Qt.CursorShape.PointingHandCursor)` with:

```python
        self.tool_mode = False
        self._tool_dragging = False
        self._idle_cursor = Qt.CursorShape.PointingHandCursor
        self.setCursor(self._idle_cursor)
```

Add this method before `paintEvent`:

```python
    def set_tool_mode(self, enabled: bool) -> None:
        """While on, a left drag is handed to a tool instead of panning the view."""
        self.tool_mode = enabled
        self._tool_dragging = False
        self._idle_cursor = (
            Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.PointingHandCursor
        )
        self.setCursor(self._idle_cursor)
```

Replace `mousePressEvent`, `mouseReleaseEvent` and `mouseMoveEvent` with:

```python
    def mousePressEvent(self, event: QMouseEvent):
        wants_tool = (
            self.tool_mode
            and event.button() == Qt.MouseButton.LeftButton
            and not event.modifiers() & Qt.KeyboardModifier.ControlModifier
        )
        if wants_tool:
            self._tool_dragging = True
            self.setFocus()
            self.press_event.emit(
                CanvasMouseEvent(
                    event.button(), self.zoom, self.to_physical_units(event.pos())
                )
            )
            return
        self.mouse_press_position = event.pos()
        self.mouse_pressed = True
        self.click_origin = (event.pos().x(), event.pos().y())

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._tool_dragging:
            self._tool_dragging = False
            self.release_event.emit(
                CanvasMouseEvent(
                    event.button(), self.zoom, self.to_physical_units(event.pos())
                )
            )
            return
        self.mouse_pressed = False
        self.setCursor(self._idle_cursor)
        displacement = self.zoom * (event.pos() - self.mouse_press_position)
        if abs(displacement.x()) < 0.4 and abs(displacement.y()) < 0.4:
            # ignore small movements and treat them as clicks
            self.setFocus()
            self.click_event.emit(
                CanvasMouseEvent(
                    event.button(),
                    self.zoom,
                    self.to_physical_units(self.mouse_press_position),
                )
            )

    def mouseMoveEvent(self, event: QMouseEvent):
        moved = CanvasMouseEvent(
            event.buttons(), self.zoom, self.to_physical_units(event.pos())
        )
        self.move_event.emit(moved)
        if self._tool_dragging:
            self.drag_event.emit(moved)
            return
        if self.mouse_pressed:
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            pos = event.pos()
            self.centerX -= self.zoom * (pos.x() - self.click_origin[0])
            self.centerY -= self.zoom * (pos.y() - self.click_origin[1])
            self.click_origin = (pos.x(), pos.y())
            self.update()
```

- [ ] **Step 4: Run the tests**

Run: `$env:QT_QPA_PLATFORM="offscreen"; python -m pytest tests/canvas_tool_mode_test.py -q -p no:cacheprovider`
Expected: 6 passed. The existing default behaviour is unchanged when `tool_mode` is off.

- [ ] **Step 5: Commit**

```bash
git add src/idtrackerai/GUI_tools/widgets_utils/canvas.py tests/canvas_tool_mode_test.py
git commit -m "Let the canvas hand left drags to a tool instead of panning

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 5: The editor widget

**Files:**
- Rewrite: `src/idtrackerai/segmentation_app/widgets/ROI_widget.py`
- Test: `tests/roi_widget_test.py`

The widget keeps the interface `main.py` already uses (`CheckBox`, `exclusive_rois`, `valueChanged`, `needToDraw`, `getValue`, `setValue`, `paint_on_canvas`, `set_video_size`) and adds `press_event`, `drag_event`, `release_event`, `move_event`, `toolModeChanged`, `install_shortcuts`, `set_frame`, `set_enhancement`, `getMask`, `load_mask_file`, `commit_pending`, `cancel_pending`.

- [ ] **Step 1: Write the failing tests**

Create `tests/roi_widget_test.py`:

```python
"""The ROI editor widget, driven the way the canvas drives it."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from qtpy.QtCore import Qt
from qtpy.QtGui import QImage
from qtpy.QtWidgets import QApplication, QInputDialog, QMessageBox

from idtrackerai.GUI_tools import CanvasMouseEvent, CanvasPainter
from idtrackerai.segmentation_app.widgets.ROI_widget import (
    BRUSH,
    ELLIPSE,
    ERASER,
    MOVE,
    RECT,
    SELECT,
    ROIWidget,
)
from idtrackerai.utils import build_ROI_mask_from_list
from idtrackerai.utils.roi_mask import empty_mask, save_mask_png

app = QApplication.instance() or QApplication([])
LEFT = Qt.MouseButton.LeftButton
ON, OFF = 255, 0


def ev(x, y, zoom=1.0):
    return CanvasMouseEvent(LEFT, zoom, (x, y))


def drag(w, *points):
    w.press_event(ev(*points[0]))
    for p in points[1:]:
        w.drag_event(ev(*p))
    w.release_event(ev(*points[-1]))


@pytest.fixture
def widget(tmp_path):
    w = ROIWidget(arenas_dir=tmp_path / "arenas")
    w.set_video_size((200, 100))
    w.CheckBox.setChecked(True)
    return w


def strip_frame():
    rng = np.random.default_rng(0)
    gray = np.full((100, 200), 70, np.uint8)
    gray[15:25, :] = 220
    gray[60:70, :] = 220
    return np.clip(gray + rng.normal(0, 4, gray.shape), 0, 255).astype(np.uint8)


# ----------------------------------------------------------------- basics
def test_unchecked_means_no_roi(tmp_path):
    w = ROIWidget(arenas_dir=tmp_path)
    w.set_video_size((200, 100))
    assert w.getValue() is None and w.getMask() is None


def test_checked_but_empty_gives_an_empty_list(widget):
    assert widget.getValue() == []


def test_tool_mode_signal_follows_the_checkbox(tmp_path):
    w = ROIWidget(arenas_dir=tmp_path)
    w.set_video_size((200, 100))
    states = []
    w.toolModeChanged.connect(states.append)
    w.CheckBox.setChecked(True)
    w.CheckBox.setChecked(False)
    assert states == [True, False]


# ------------------------------------------------------------ brush/eraser
def test_brush_paints_and_publishes_once_on_release(widget):
    published = []
    widget.valueChanged.connect(published.append)
    widget.set_tool(BRUSH)
    widget.brush_slider.setValue(6)
    widget.press_event(ev(50, 50))
    widget.drag_event(ev(100, 50))
    assert published == []  # not while still painting
    widget.release_event(ev(100, 50))
    assert len(published) == 1
    assert published[0][50, 75] == ON and published[0][10, 10] == OFF
    assert widget.getValue()[0].startswith("+ Polygon")


def test_eraser_removes(widget):
    widget.set_tool(BRUSH)
    widget.brush_slider.setValue(8)
    drag(widget, (30, 50), (170, 50))
    widget.set_tool(ERASER)
    drag(widget, (90, 50), (110, 50))
    mask = widget.getMask()
    assert mask[50, 40] == ON and mask[50, 100] == OFF


def test_exclude_mode_makes_the_brush_remove(widget):
    widget.set_tool(BRUSH)
    widget.brush_slider.setValue(8)
    drag(widget, (30, 50), (170, 50))
    widget.exclude_button.click()
    drag(widget, (90, 50), (110, 50))
    assert widget.getMask()[50, 100] == OFF
    widget.include_button.click()
    drag(widget, (90, 50), (110, 50))
    assert widget.getMask()[50, 100] == ON


def test_alt_flips_the_mode_while_held(widget, monkeypatch):
    widget.set_tool(BRUSH)
    widget.brush_slider.setValue(8)
    drag(widget, (30, 50), (170, 50))
    monkeypatch.setattr(
        QApplication, "keyboardModifiers", staticmethod(lambda: Qt.KeyboardModifier.AltModifier)
    )
    drag(widget, (90, 50), (110, 50))
    assert widget.getMask()[50, 100] == OFF


# ----------------------------------------------------------------- shapes
def test_rectangle_stays_pending_until_committed(widget):
    widget.set_tool(RECT)
    drag(widget, (20, 20), (80, 60))
    assert widget.pending is not None
    assert not widget.getMask().any()
    widget.commit_pending()
    mask = widget.getMask()
    assert widget.pending is None
    assert mask[40, 50] == ON and mask[10, 10] == OFF and mask[40, 120] == OFF


def test_ellipse_commit_fills_an_inscribed_ellipse(widget):
    widget.set_tool(ELLIPSE)
    drag(widget, (40, 20), (160, 80))
    widget.commit_pending()
    mask = widget.getMask()
    assert mask[50, 100] == ON and mask[21, 41] == OFF  # a bounding-box corner


def test_dragging_a_corner_resizes_the_pending_shape(widget):
    widget.set_tool(RECT)
    drag(widget, (20, 20), (80, 60))
    drag(widget, (80, 60), (120, 80))  # grab the bottom-right handle
    widget.commit_pending()
    assert widget.getMask()[75, 110] == ON


def test_dragging_inside_moves_the_pending_shape(widget):
    widget.set_tool(RECT)
    drag(widget, (20, 20), (80, 60))
    drag(widget, (50, 40), (110, 40))  # inside the box, not on a handle
    widget.commit_pending()
    mask = widget.getMask()
    assert mask[40, 120] == ON and mask[40, 30] == OFF


def test_pressing_outside_commits_and_starts_a_new_shape(widget):
    widget.set_tool(RECT)
    drag(widget, (20, 20), (60, 50))
    drag(widget, (120, 20), (170, 60))
    assert widget.getMask()[35, 40] == ON  # the first one was committed
    assert widget.pending is not None  # the second one is pending
    widget.commit_pending()
    assert widget.getMask()[40, 150] == ON


def test_cancel_discards_the_pending_shape(widget):
    widget.set_tool(RECT)
    drag(widget, (20, 20), (80, 60))
    widget.cancel_pending()
    widget.commit_pending()
    assert not widget.getMask().any()


def test_a_click_without_a_drag_leaves_no_shape(widget):
    widget.set_tool(RECT)
    drag(widget, (50, 50), (50, 50))
    assert widget.pending is None


def test_changing_tool_commits_the_pending_shape(widget):
    widget.set_tool(RECT)
    drag(widget, (20, 20), (80, 60))
    widget.set_tool(BRUSH)
    assert widget.getMask()[40, 50] == ON


def test_toggling_exclude_before_commit_removes_instead(widget):
    widget.set_tool(BRUSH)
    widget.brush_slider.setValue(30)
    drag(widget, (60, 50), (140, 50))
    widget.set_tool(RECT)
    drag(widget, (80, 30), (120, 70))
    widget.exclude_button.click()
    widget.commit_pending()
    assert widget.getMask()[50, 100] == OFF and widget.getMask()[50, 70] == ON


# ----------------------------------------------------------- click select
def test_click_select_fills_the_floor_between_walls(widget):
    widget.set_frame(strip_frame())
    widget.set_tool(SELECT)
    widget.tolerance_slider.setValue(12)
    widget.press_event(ev(100, 40))
    widget.release_event(ev(100, 40))
    mask = widget.getMask()
    assert mask[40, 5] == ON and mask[40, 195] == ON
    assert mask[20, 100] == OFF and mask[90, 100] == OFF


def test_click_select_without_a_frame_does_nothing(widget):
    widget.set_tool(SELECT)
    widget.press_event(ev(100, 40))
    widget.release_event(ev(100, 40))
    assert not widget.getMask().any()


# ------------------------------------------------------------ move / nudge
def test_move_tool_shifts_the_mask_and_can_be_undone(widget):
    widget.set_tool(RECT)
    drag(widget, (20, 20), (60, 50))
    widget.commit_pending()
    widget.set_tool(MOVE)
    drag(widget, (40, 30), (70, 40))
    assert widget.getMask()[45, 60] == ON and widget.getMask()[30, 30] == OFF
    widget.undo()
    assert widget.getMask()[30, 30] == ON


def test_nudge_buttons_shift_by_one_pixel(widget):
    widget.set_tool(RECT)
    drag(widget, (20, 20), (60, 50))
    widget.commit_pending()
    before = widget.getMask()
    widget.nudge(1, 0)
    assert np.array_equal(widget.getMask()[:, 1:], before[:, :-1])


# ------------------------------------------------------------ undo / redo
def test_undo_and_redo_walk_through_the_edits(widget):
    widget.set_tool(BRUSH)
    widget.brush_slider.setValue(5)
    drag(widget, (20, 20), (40, 20))
    first = widget.getMask()
    drag(widget, (20, 60), (40, 60))
    widget.undo()
    assert np.array_equal(widget.getMask(), first)
    widget.undo()
    assert not widget.getMask().any()
    widget.redo()
    assert np.array_equal(widget.getMask(), first)


def test_undo_cancels_a_pending_shape_first(widget):
    widget.set_tool(BRUSH)
    widget.brush_slider.setValue(5)
    drag(widget, (20, 20), (40, 20))
    widget.set_tool(RECT)
    drag(widget, (80, 30), (120, 70))
    widget.undo()
    assert widget.pending is None and widget.getMask()[20, 30] == ON


# ------------------------------------------------- loading and old formats
OLD_ROI = [
    "+ Polygon [[10.0, 10.0], [100.0, 10.0], [100.0, 60.0], [10.0, 60.0]]",
    "- Ellipse {'center': [50, 35], 'axes': [10, 10], 'angle': 0}",
]


def test_old_roi_list_loads_into_the_mask(widget):
    widget.setValue(OLD_ROI, False)
    assert np.array_equal(widget.getMask(), build_ROI_mask_from_list(OLD_ROI, 200, 100))
    assert widget.CheckBox.isChecked()


def test_old_roi_list_waits_for_the_video_size(tmp_path):
    w = ROIWidget(arenas_dir=tmp_path)
    w.setValue(OLD_ROI, False)  # no video yet
    w.set_video_size((200, 100))
    assert np.array_equal(w.getMask(), build_ROI_mask_from_list(OLD_ROI, 200, 100))


def test_setting_no_value_unticks_the_region(widget):
    widget.setValue(OLD_ROI, False)
    widget.setValue(None, False)
    assert not widget.CheckBox.isChecked() and widget.getValue() is None


def test_load_mask_file(widget, tmp_path):
    mask = empty_mask(200, 100)
    mask[10:50, 20:90] = ON
    save_mask_png(tmp_path / "m.png", mask)
    widget.load_mask_file(tmp_path / "m.png", False)
    assert np.array_equal(widget.getMask(), mask)


def test_load_mask_file_of_the_wrong_size_raises(widget, tmp_path):
    save_mask_png(tmp_path / "m.png", empty_mask(50, 50))
    with pytest.raises(ValueError, match="50x50"):
        widget.load_mask_file(tmp_path / "m.png", False)


def test_exclusive_checkbox_appears_only_with_several_regions(widget):
    widget.set_tool(RECT)
    drag(widget, (10, 10), (40, 40))
    widget.commit_pending()
    assert widget.exclusive_rois.isHidden()
    drag(widget, (100, 10), (140, 40))
    widget.commit_pending()
    assert not widget.exclusive_rois.isHidden()


# ----------------------------------------------------------------- arenas
def test_save_and_apply_an_arena(widget, monkeypatch):
    monkeypatch.setattr(QInputDialog, "getText", staticmethod(lambda *a, **k: ("Tank", True)))
    widget.set_tool(RECT)
    drag(widget, (20, 20), (80, 60))
    widget.commit_pending()
    saved = widget.getMask()
    widget.save_arena_as()
    assert widget.arena_combo.findData("Tank") > 0
    widget.clear_region()
    assert not widget.getMask().any()
    widget.apply_arena("Tank")
    assert np.array_equal(widget.getMask(), saved)


def test_arena_of_the_same_proportions_is_scaled_after_asking(widget, monkeypatch, tmp_path):
    from idtrackerai.utils import roi_arenas

    small = empty_mask(100, 50)
    small[10:30, 10:60] = ON
    roi_arenas.save_arena(widget.arenas_dir, "Small", small, None)
    monkeypatch.setattr(
        QMessageBox, "question", staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes)
    )
    widget.apply_arena("Small")
    mask = widget.getMask()
    assert mask.shape == (100, 200) and mask[40, 60] == ON and mask[90, 150] == OFF


def test_arena_of_other_proportions_is_refused(widget, monkeypatch):
    from idtrackerai.utils import roi_arenas

    square = empty_mask(100, 100)
    square[10:30, 10:60] = ON
    roi_arenas.save_arena(widget.arenas_dir, "Square", square, None)
    warned = []
    monkeypatch.setattr(
        QMessageBox, "warning", staticmethod(lambda *a, **k: warned.append(a[2]))
    )
    widget.apply_arena("Square")
    assert warned and not widget.getMask().any()


# ------------------------------------------------------------------ paint
def test_painting_with_every_overlay_does_not_raise(widget):
    widget.set_tool(BRUSH)
    drag(widget, (20, 20), (60, 40))
    widget.set_tool(RECT)
    drag(widget, (80, 30), (120, 70))
    widget.move_event(ev(30, 30))
    image = QImage(200, 100, QImage.Format.Format_ARGB32)
    painter = CanvasPainter(image, 1.0)
    try:
        widget.paint_on_canvas(painter, 0, None)
        widget.set_tool(BRUSH)
        widget.move_event(ev(30, 30))
        widget.paint_on_canvas(painter, 0, None)
    finally:
        painter.end()
```

- [ ] **Step 2: Run to verify they fail**

Run: `$env:QT_QPA_PLATFORM="offscreen"; python -m pytest tests/roi_widget_test.py -q -p no:cacheprovider -x`
Expected: `ImportError: cannot import name 'BRUSH' from ... ROI_widget`.

- [ ] **Step 3: Rewrite `ROI_widget.py`**

Replace the whole file with:

```python
"""Region of interest editor: paint, select and shape the area to track.

The region is one mask the size of the video (255 = tracked). Every tool edits
that mask, what lies outside it is dimmed on the video, and a finished mask can
be kept as a named arena and reused on later videos of the same setup. The mask
maths lives in ``idtrackerai.utils.roi_mask``; this file is the Qt layer.
"""

from __future__ import annotations

from itertools import cycle
from math import hypot
from pathlib import Path

import numpy as np
from qtpy.QtCore import QPointF, QRectF, Qt  # type: ignore[reportPrivateImportUsage]
from qtpy.QtCore import Signal  # type: ignore[reportPrivateImportUsage]
from qtpy.QtGui import QColor, QImage, QKeySequence, QPen, QShortcut
from qtpy.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from idtrackerai.base.fragmentation import find_exclusive_contours
from idtrackerai.extra_tools.detectron2_pipeline import preprocessing as fp
from idtrackerai.GUI_tools import CanvasMouseEvent, CanvasPainter, get_path_from_points
from idtrackerai.utils import build_ROI_mask_from_list, roi_arenas
from idtrackerai.utils.roi_mask import (
    OFF,
    ON,
    MaskHistory,
    apply_region,
    ellipse_region,
    empty_mask,
    flood_region,
    load_mask_png,
    mask_to_roi_list,
    proportions_match,
    rectangle_region,
    scale_mask,
    shift_mask,
    stroke_region,
)

BRUSH, ERASER, SELECT, RECT, ELLIPSE, MOVE = (
    "brush",
    "eraser",
    "select",
    "rect",
    "ellipse",
    "move",
)
TOOLS = {
    BRUSH: ("Brush", "B"),
    ERASER: ("Eraser", "E"),
    SELECT: ("Click select", "C"),
    RECT: ("Rectangle", "R"),
    ELLIPSE: ("Ellipse", "O"),
    MOVE: ("Move", "M"),
}
SHAPE_TOOLS = (RECT, ELLIPSE)
HANDLE_PIXELS = 8  # how close, on screen, a press must be to grab a corner
INCLUDE_COLOR = QColor(47, 182, 165)
EXCLUDE_COLOR = QColor(231, 76, 60)


class ROIWidget(QWidget):
    """Tools, options and saved arenas for the region of interest."""

    needToDraw = Signal()
    valueChanged = Signal(object)  # np.ndarray | None, the mask for the analyzer
    toolModeChanged = Signal(bool)  # True while the canvas should hand drags to us
    exclusive_ROI_colors = [
        QColor(0, 0, 190, 200),
        QColor(0, 200, 0, 200),
        QColor(255, 100, 0, 200),
        QColor(205, 254, 0, 200),
        QColor(138, 102, 66, 200),
        QColor(200, 50, 250, 200),
    ]

    def __init__(self, parent=None, arenas_dir: Path | str | None = None):
        super().__init__(parent)
        self.arenas_dir = (
            Path(arenas_dir) if arenas_dir else roi_arenas.default_arenas_dir()
        )
        self.mask: np.ndarray | None = None
        self.history = MaskHistory()
        self.video_size = (1, 1)
        self.tool = BRUSH
        self.include = True
        self.pending: dict | None = None
        self.exclusive_ROI_paths: list = []

        self._frame: np.ndarray | None = None
        self._enhancement: dict | None = None
        self._overlay: QImage | None = None
        self._overlay_buffer: np.ndarray | None = None
        self._cursor_xy: tuple[float, float] | None = None
        self._last_point: tuple[float, float] | None = None
        self._before: np.ndarray | None = None
        self._drag: dict | None = None
        self._pending_roi_list: list[str] | None = None

        self._build_ui()
        self.refresh_arenas()
        self._sync_buttons()

    # --------------------------------------------------------------------- UI
    def _build_ui(self) -> None:
        self.CheckBox = QCheckBox("Regions of interest")
        self.CheckBox.stateChanged.connect(self.CheckBox_changed)

        self.exclusive_rois = QCheckBox("Exclusive ROIs")
        self.exclusive_rois.setEnabled(False)
        self.exclusive_rois.setVisible(False)
        self.exclusive_rois.stateChanged.connect(lambda _s: self.needToDraw.emit())

        self.panel = QWidget()
        self.panel.setEnabled(False)

        group = QButtonGroup(self)
        group.setExclusive(True)
        self.tool_buttons: dict[str, QToolButton] = {}
        tools_row = QHBoxLayout()
        for tool, (label, key) in TOOLS.items():
            button = QToolButton()
            button.setText(label)
            button.setToolTip(f"{label} ({key})")
            button.setCheckable(True)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.clicked.connect(lambda _c, t=tool: self.set_tool(t))
            group.addButton(button)
            tools_row.addWidget(button)
            self.tool_buttons[tool] = button
        self.tool_buttons[BRUSH].setChecked(True)

        mode_group = QButtonGroup(self)
        mode_group.setExclusive(True)
        self.include_button = QToolButton()
        self.include_button.setText("Include")
        self.exclude_button = QToolButton()
        self.exclude_button.setText("Exclude")
        for button in (self.include_button, self.exclude_button):
            button.setCheckable(True)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            mode_group.addButton(button)
        self.include_button.setChecked(True)
        self.include_button.setToolTip("Add to the region. Hold Alt to flip while using a tool.")
        self.exclude_button.setToolTip("Take away from the region. Hold Alt to flip while using a tool.")
        self.include_button.clicked.connect(lambda: self._mode_changed(True))
        self.exclude_button.clicked.connect(lambda: self._mode_changed(False))

        self.undo_button = QToolButton()
        self.undo_button.setText("Undo")
        self.undo_button.setToolTip("Undo (Ctrl+Z)")
        self.undo_button.clicked.connect(self.undo)
        self.redo_button = QToolButton()
        self.redo_button.setText("Redo")
        self.redo_button.setToolTip("Redo (Ctrl+Y)")
        self.redo_button.clicked.connect(self.redo)
        for button in (self.undo_button, self.redo_button):
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)

        mode_row = QHBoxLayout()
        mode_row.addWidget(self.include_button)
        mode_row.addWidget(self.exclude_button)
        mode_row.addStretch(1)
        mode_row.addWidget(self.undo_button)
        mode_row.addWidget(self.redo_button)

        self.brush_slider = QSlider(Qt.Orientation.Horizontal)
        self.brush_slider.setRange(1, 150)
        self.brush_slider.setValue(12)
        self.brush_slider.setToolTip("Brush radius in video pixels. [ and ] change it.")
        self.brush_slider.valueChanged.connect(lambda _v: self.needToDraw.emit())
        self.tolerance_slider = QSlider(Qt.Orientation.Horizontal)
        self.tolerance_slider.setRange(1, 60)
        self.tolerance_slider.setValue(12)
        self.tolerance_slider.setToolTip(
            "How different from the clicked pixel a neighbour may be and still be selected"
        )
        for slider in (self.brush_slider, self.tolerance_slider):
            slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        sliders = QVBoxLayout()
        for text, slider in (
            ("Brush size", self.brush_slider),
            ("Select tolerance", self.tolerance_slider),
        ):
            row = QHBoxLayout()
            row.addWidget(QLabel(text))
            row.addWidget(slider)
            sliders.addLayout(row)

        nudge_row = QHBoxLayout()
        nudge_row.addWidget(QLabel("Nudge (Shift = 10 px)"))
        for text, dx, dy in (("◀", -1, 0), ("▶", 1, 0), ("▲", 0, -1), ("▼", 0, 1)):
            button = QToolButton()
            button.setText(text)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.clicked.connect(lambda _c, a=dx, b=dy: self.nudge(a, b))
            nudge_row.addWidget(button)
        nudge_row.addStretch(1)

        self.arena_combo = QComboBox()
        self.arena_combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.arena_combo.activated.connect(self._arena_activated)
        self.save_arena_button = QToolButton()
        self.save_arena_button.setText("Save as…")
        self.save_arena_button.clicked.connect(self.save_arena_as)
        self.import_button = QToolButton()
        self.import_button.setText("Import…")
        self.import_button.clicked.connect(self.import_arena_file)
        self.export_button = QToolButton()
        self.export_button.setText("Export…")
        self.export_button.clicked.connect(self.export_arena_file)
        self.delete_button = QToolButton()
        self.delete_button.setText("Delete")
        self.delete_button.clicked.connect(self.delete_selected_arena)
        self.clear_button = QToolButton()
        self.clear_button.setText("Clear")
        self.clear_button.clicked.connect(self.clear_region)
        for button in (
            self.save_arena_button,
            self.import_button,
            self.export_button,
            self.delete_button,
            self.clear_button,
        ):
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        arena_buttons = QHBoxLayout()
        for button in (
            self.save_arena_button,
            self.import_button,
            self.export_button,
            self.delete_button,
            self.clear_button,
        ):
            arena_buttons.addWidget(button)

        panel_layout = QVBoxLayout(self.panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(2)
        panel_layout.addLayout(tools_row)
        panel_layout.addLayout(mode_row)
        panel_layout.addLayout(sliders)
        panel_layout.addLayout(nudge_row)
        panel_layout.addWidget(self.arena_combo)
        panel_layout.addLayout(arena_buttons)

        layout = QVBoxLayout(self)
        layout.setSpacing(2)
        layout.addWidget(self.CheckBox)
        layout.addWidget(self.panel)
        layout.addWidget(self.exclusive_rois, alignment=Qt.AlignmentFlag.AlignRight)

    def install_shortcuts(self, target: QWidget) -> None:
        """Keyboard shortcuts that work while ``target`` (the canvas) has focus."""
        for tool, (_label, key) in TOOLS.items():
            self._shortcut(target, key, lambda t=tool: self._shortcut_tool(t))
        self._shortcut(target, "Ctrl+Z", self.undo)
        self._shortcut(target, "Ctrl+Y", self.redo)
        self._shortcut(target, "[", lambda: self.brush_slider.setValue(self.brush_slider.value() - 3))
        self._shortcut(target, "]", lambda: self.brush_slider.setValue(self.brush_slider.value() + 3))

    def _shortcut(self, target: QWidget, key: str, slot) -> None:
        shortcut = QShortcut(QKeySequence(key), target)
        shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
        shortcut.activated.connect(slot)

    def _shortcut_tool(self, tool: str) -> None:
        if self._active():
            self.set_tool(tool)

    # --------------------------------------------------------------- state
    def _active(self) -> bool:
        return self.CheckBox.isChecked() and self.mask is not None

    def CheckBox_changed(self, _state=None) -> None:
        enabled = self.CheckBox.isChecked()
        self.panel.setEnabled(enabled)
        self.exclusive_rois.setEnabled(enabled)
        if enabled:
            self._ensure_mask()
        else:
            self.pending = None
            self._drag = None
        self._after_change()
        self.toolModeChanged.emit(self._active())

    def _ensure_mask(self) -> None:
        width, height = self.video_size
        if self.mask is None or self.mask.shape != (height, width):
            self.mask = empty_mask(width, height)
            self.history.clear()

    def set_video_size(self, video_size) -> None:
        width, height = video_size
        self.video_size = (width, height)
        if self.mask is not None and self.mask.shape != (height, width):
            self.mask = empty_mask(width, height) if self.CheckBox.isChecked() else None
            self.history.clear()
            self.pending = None
        if self._pending_roi_list is not None and (width, height) != (1, 1):
            values, self._pending_roi_list = self._pending_roi_list, None
            self._ensure_mask()
            self._replace_mask(build_ROI_mask_from_list(values, width, height), record=False)
            return
        if self.CheckBox.isChecked():
            self._ensure_mask()
        self._after_change()

    def set_frame(self, frame: np.ndarray | None) -> None:
        """The frame on screen, which click select reads."""
        if frame is not None and getattr(frame, "size", 0):
            self._frame = frame

    def set_enhancement(self, settings: dict | None) -> None:
        self._enhancement = settings

    def set_tool(self, tool: str) -> None:
        self.commit_pending()
        self.tool = tool
        self.tool_buttons[tool].setChecked(True)
        self._last_point = None
        self.needToDraw.emit()

    def _mode_changed(self, include: bool) -> None:
        self.include = include
        if self.pending is not None:
            self.pending["include"] = include
        self.needToDraw.emit()

    def _effective_include(self) -> bool:
        base = False if self.tool == ERASER else self.include
        alt = bool(QApplication.keyboardModifiers() & Qt.KeyboardModifier.AltModifier)
        return base != alt

    # ----------------------------------------------------------- mask updates
    def _replace_mask(self, new: np.ndarray, record: bool = True) -> None:
        before = self.mask
        self.mask = new
        if record and before is not None and before.shape == new.shape:
            self.history.record(before, new)
        else:
            self.history.clear()
        self._after_change()

    def setMask(self, mask: np.ndarray, record: bool = True) -> None:
        self.CheckBox.setChecked(True)
        self.commit_pending()
        self._replace_mask(np.where(mask > 0, ON, OFF).astype(np.uint8), record)

    def _after_change(self) -> None:
        self._overlay = None
        self._update_exclusive_paths()
        self._publish()
        self._sync_buttons()
        self.needToDraw.emit()

    def _publish(self) -> None:
        active = self.CheckBox.isChecked() and self.mask is not None
        self.valueChanged.emit(self.mask.copy() if active else None)

    def _update_exclusive_paths(self) -> None:
        self.exclusive_ROI_paths = []
        if self.CheckBox.isChecked() and self.mask is not None:
            for contour, holes in find_exclusive_contours(self.mask):
                path = get_path_from_points(contour)
                for hole in holes:
                    path -= get_path_from_points(hole)
                self.exclusive_ROI_paths.append(path)
        if len(self.exclusive_ROI_paths) == 1:
            self.exclusive_ROI_paths.clear()
        self.exclusive_rois.setVisible(len(self.exclusive_ROI_paths) > 0)

    def _sync_buttons(self) -> None:
        self.undo_button.setEnabled(self.history.can_undo or self.pending is not None)
        self.redo_button.setEnabled(self.history.can_redo)
        has_mask = self.mask is not None and bool(self.mask.any())
        self.save_arena_button.setEnabled(has_mask)
        self.clear_button.setEnabled(has_mask)
        chosen = self._selected_arena() is not None
        self.export_button.setEnabled(chosen)
        self.delete_button.setEnabled(chosen)

    # ----------------------------------------------------------- value in/out
    def getMask(self) -> np.ndarray | None:
        return self.mask.copy() if self._active() else None

    def getValue(self) -> list[str] | None:
        """The polygon strings older readers use, derived from the mask."""
        if not self._active():
            return None
        return mask_to_roi_list(self.mask)

    def setValue(self, values: list[str] | str | None, exclusive_roi: bool) -> None:
        self.pending = None
        self._pending_roi_list = None
        self.history.clear()
        if not values:
            self.mask = None
            self.CheckBox.setChecked(False)
            self._after_change()
            return
        if isinstance(values, str):
            values = [values]
        self.mask = None
        if self.video_size == (1, 1):
            self._pending_roi_list = list(values)
        self.CheckBox.setChecked(True)
        if self._pending_roi_list is None:
            width, height = self.video_size
            self._replace_mask(build_ROI_mask_from_list(values, width, height), record=False)
        self.exclusive_rois.setChecked(exclusive_roi)

    def load_mask_file(self, path: Path | str, exclusive_roi: bool = False) -> None:
        """Use a saved mask PNG. Raises ValueError if it cannot be used."""
        mask = load_mask_png(path)
        width, height = self.video_size
        if mask.shape != (height, width):
            raise ValueError(
                f"The ROI mask {Path(path).name} is {mask.shape[1]}x{mask.shape[0]} "
                f"but the video is {width}x{height}."
            )
        self.pending = None
        self._pending_roi_list = None
        self.CheckBox.setChecked(True)
        self._replace_mask(mask, record=False)
        self.exclusive_rois.setChecked(exclusive_roi)

    # ------------------------------------------------------------ mouse input
    def press_event(self, event: CanvasMouseEvent) -> None:
        if not self._active() or event.button != Qt.MouseButton.LeftButton:
            return
        x, y = event.xy_data
        if self.tool in SHAPE_TOOLS:
            self._press_shape(x, y, event.zoom)
            return

        self.commit_pending()
        include = self._effective_include()
        self._before = self.mask.copy()
        if self.tool in (BRUSH, ERASER):
            shift = bool(QApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier)
            points = [self._last_point, (x, y)] if shift and self._last_point else [(x, y)]
            self._paint(points, include)
            self._drag = {"kind": "stroke", "include": include}
        elif self.tool == SELECT:
            self._select_at(x, y, include)
        elif self.tool == MOVE:
            self._drag = {"kind": "move", "origin": (x, y)}
        self._last_point = (x, y)

    def drag_event(self, event: CanvasMouseEvent) -> None:
        if self._drag is None or not self._active():
            return
        x, y = event.xy_data
        self._cursor_xy = (x, y)
        kind = self._drag["kind"]
        if kind == "stroke":
            self._paint([self._last_point or (x, y), (x, y)], self._drag["include"])
            self._last_point = (x, y)
        elif kind == "move":
            ox, oy = self._drag["origin"]
            self.mask = shift_mask(self._before, x - ox, y - oy)
            self._overlay = None
        elif self.pending is not None:
            self._drag_shape(kind, x, y)
        self.needToDraw.emit()

    def release_event(self, event: CanvasMouseEvent) -> None:
        drag, self._drag = self._drag, None
        if drag is None or not self._active():
            return
        kind = drag["kind"]
        if kind in ("stroke", "move"):
            self.history.record(self._before, self.mask)
            self._before = None
            self._after_change()
        elif self.pending is not None:
            p = self.pending
            p["x0"], p["x1"] = sorted((p["x0"], p["x1"]))
            p["y0"], p["y1"] = sorted((p["y0"], p["y1"]))
            if p["x1"] - p["x0"] < 2 and p["y1"] - p["y0"] < 2:
                self.pending = None
            self._sync_buttons()
            self.needToDraw.emit()

    def move_event(self, event: CanvasMouseEvent) -> None:
        """The cursor moved, so the brush outline can follow it."""
        self._cursor_xy = event.xy_data
        if self._active() and self.tool in (BRUSH, ERASER):
            self.needToDraw.emit()

    def _radius(self) -> int:
        return self.brush_slider.value()

    def _paint(self, points, include: bool) -> None:
        region = stroke_region(self.mask.shape, points, self._radius())
        apply_region(self.mask, region, include)
        self._overlay = None

    def _select_at(self, x: float, y: float, include: bool) -> None:
        gray = self._gray_for_select()
        if gray is None or gray.shape != self.mask.shape:
            self._before = None
            return
        region = flood_region(gray, (x, y), self.tolerance_slider.value())
        apply_region(self.mask, region, include)
        self.history.record(self._before, self.mask)
        self._before = None
        self._after_change()

    def _gray_for_select(self) -> np.ndarray | None:
        """The frame as the user sees it: enhanced when enhancement is on."""
        if self._frame is None:
            return None
        settings = self._enhancement
        if settings and settings.get("enhance"):
            return fp.enhance(
                self._frame,
                clahe_clip=settings["clahe_clip"],
                clahe_tile=settings["clahe_tile"],
                downsample=settings["illumination_downsample"],
                sigma=settings["illumination_sigma"],
                correct_lighting=settings["correct_lighting"],
            )
        return fp.to_gray(self._frame)

    # ----------------------------------------------------------- pending shapes
    def _press_shape(self, x: float, y: float, zoom: float) -> None:
        hit = self._hit(x, y, zoom) if self.pending is not None else None
        if self.pending is not None and hit is None:
            self.commit_pending()
        if self.pending is not None and hit is not None:
            p = self.pending
            self._drag = {
                "kind": "shape_move" if hit == "move" else "resize",
                "handle": hit,
                "origin": (x, y),
                "box": (p["x0"], p["y0"], p["x1"], p["y1"]),
            }
            return
        self.pending = {
            "kind": self.tool,
            "x0": x,
            "y0": y,
            "x1": x,
            "y1": y,
            "include": self._effective_include(),
        }
        self._drag = {"kind": "new"}
        self._sync_buttons()

    def _hit(self, x: float, y: float, zoom: float) -> str | None:
        p = self.pending
        reach = HANDLE_PIXELS * zoom
        corners = {
            "tl": (p["x0"], p["y0"]),
            "tr": (p["x1"], p["y0"]),
            "bl": (p["x0"], p["y1"]),
            "br": (p["x1"], p["y1"]),
        }
        for name, (cx, cy) in corners.items():
            if hypot(x - cx, y - cy) <= reach:
                return name
        if p["x0"] <= x <= p["x1"] and p["y0"] <= y <= p["y1"]:
            return "move"
        return None

    def _drag_shape(self, kind: str, x: float, y: float) -> None:
        p = self.pending
        if kind == "new":
            p["x1"], p["y1"] = x, y
        elif kind == "resize":
            x0, y0, x1, y1 = self._drag["box"]
            handle = self._drag["handle"]
            if handle[1] == "l":
                x0 = x
            else:
                x1 = x
            if handle[0] == "t":
                y0 = y
            else:
                y1 = y
            p["x0"], p["y0"], p["x1"], p["y1"] = x0, y0, x1, y1
        elif kind == "shape_move":
            ox, oy = self._drag["origin"]
            x0, y0, x1, y1 = self._drag["box"]
            dx, dy = x - ox, y - oy
            p["x0"], p["y0"], p["x1"], p["y1"] = x0 + dx, y0 + dy, x1 + dx, y1 + dy

    def _pending_region(self) -> np.ndarray:
        p = self.pending
        shape = self.mask.shape
        if p["kind"] == RECT:
            return rectangle_region(shape, (p["x0"], p["y0"]), (p["x1"], p["y1"]))
        center = ((p["x0"] + p["x1"]) / 2, (p["y0"] + p["y1"]) / 2)
        axes = (abs(p["x1"] - p["x0"]) / 2, abs(p["y1"] - p["y0"]) / 2)
        return ellipse_region(shape, center, axes)

    def commit_pending(self) -> None:
        if self.pending is None or not self._active():
            self.pending = None
            return
        region = self._pending_region()
        include = self.pending["include"]
        self.pending = None
        before = self.mask.copy()
        apply_region(self.mask, region, include)
        self.history.record(before, self.mask)
        self._after_change()

    def cancel_pending(self) -> None:
        if self.pending is not None:
            self.pending = None
            self._drag = None
            self._sync_buttons()
            self.needToDraw.emit()

    # --------------------------------------------------------- undo and nudge
    def undo(self) -> None:
        if self.pending is not None:
            self.cancel_pending()
            return
        if self._active() and self.history.undo(self.mask):
            self._after_change()

    def redo(self) -> None:
        if self._active() and self.history.redo(self.mask):
            self._after_change()

    def nudge(self, dx: int, dy: int) -> None:
        if not self._active():
            return
        self.commit_pending()
        step = 10 if QApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier else 1
        before = self.mask.copy()
        self.mask = shift_mask(self.mask, dx * step, dy * step)
        self.history.record(before, self.mask)
        self._after_change()

    def clear_region(self) -> None:
        if not self._active():
            return
        self.commit_pending()
        width, height = self.video_size
        self._replace_mask(empty_mask(width, height), record=True)

    # ------------------------------------------------------------------ arenas
    def refresh_arenas(self, select: str | None = None) -> None:
        self.arena_combo.blockSignals(True)
        self.arena_combo.clear()
        self.arena_combo.addItem("Saved arenas…", None)
        for arena in roi_arenas.list_arenas(self.arenas_dir):
            self.arena_combo.addItem(f"{arena.name} ({arena.width}x{arena.height})", arena.name)
        if select:
            index = self.arena_combo.findData(select)
            if index >= 0:
                self.arena_combo.setCurrentIndex(index)
        self.arena_combo.blockSignals(False)
        self._sync_buttons()

    def _selected_arena(self) -> str | None:
        return self.arena_combo.currentData()

    def _arena_activated(self, _index: int) -> None:
        name = self._selected_arena()
        if name:
            self.apply_arena(name)
        self._sync_buttons()

    def save_arena_as(self) -> None:
        self.commit_pending()
        if not self._active() or not self.mask.any():
            QMessageBox.warning(self, "Nothing to save", "Draw a region first.")
            return
        name, accepted = QInputDialog.getText(self, "Save arena", "Name for this arena:")
        if not accepted:
            return
        try:
            name = roi_arenas.sanitize_name(name)
            try:
                roi_arenas.save_arena(self.arenas_dir, name, self.mask, self._frame)
            except roi_arenas.ArenaExistsError:
                answer = QMessageBox.question(
                    self, "Replace arena?", f"An arena called {name!r} already exists. Replace it?"
                )
                if answer != QMessageBox.StandardButton.Yes:
                    return
                roi_arenas.save_arena(
                    self.arenas_dir, name, self.mask, self._frame, overwrite=True
                )
        except roi_arenas.ArenaError as exc:
            QMessageBox.warning(self, "Could not save the arena", str(exc))
            return
        self.refresh_arenas(select=name)

    def apply_arena(self, name: str) -> None:
        if self.video_size == (1, 1):
            QMessageBox.warning(self, "No video", "Open a video before using an arena.")
            return
        try:
            arena, mask = roi_arenas.load_arena(self.arenas_dir, name)
        except roi_arenas.ArenaError as exc:
            QMessageBox.warning(self, "Could not open the arena", str(exc))
            return
        width, height = self.video_size
        if mask.shape != (height, width):
            if not proportions_match(arena.width, arena.height, width, height):
                QMessageBox.warning(
                    self,
                    "Arena does not fit this video",
                    f"{name!r} was drawn on a {arena.width}x{arena.height} video and "
                    f"this one is {width}x{height}, which has different proportions.",
                )
                return
            answer = QMessageBox.question(
                self,
                "Different video size",
                f"{name!r} was drawn on {arena.width}x{arena.height} and this video is "
                f"{width}x{height}. Scale the region to fit?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            mask = scale_mask(mask, width, height)
        self.setMask(mask, record=True)

    def export_arena_file(self) -> None:
        name = self._selected_arena()
        if not name:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export arena", f"{name}.zip", "Arena (*.zip)"
        )
        if not path:
            return
        try:
            roi_arenas.export_arena(self.arenas_dir, name, path)
        except roi_arenas.ArenaError as exc:
            QMessageBox.warning(self, "Could not export the arena", str(exc))

    def import_arena_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import arena", "", "Arena (*.zip)"
        )
        if not path:
            return
        try:
            try:
                arena = roi_arenas.import_arena(self.arenas_dir, path)
            except roi_arenas.ArenaExistsError:
                answer = QMessageBox.question(
                    self, "Replace arena?", "An arena with this name exists. Replace it?"
                )
                if answer != QMessageBox.StandardButton.Yes:
                    return
                arena = roi_arenas.import_arena(self.arenas_dir, path, overwrite=True)
        except roi_arenas.ArenaError as exc:
            QMessageBox.warning(self, "Could not import the arena", str(exc))
            return
        self.refresh_arenas(select=arena.name)

    def delete_selected_arena(self) -> None:
        name = self._selected_arena()
        if not name:
            return
        answer = QMessageBox.question(self, "Delete arena", f"Delete the saved arena {name!r}?")
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            roi_arenas.delete_arena(self.arenas_dir, name)
        except roi_arenas.ArenaError as exc:
            QMessageBox.warning(self, "Could not delete the arena", str(exc))
        self.refresh_arenas()

    # ----------------------------------------------------------------- painting
    def _overlay_image(self) -> QImage:
        if self._overlay is None:
            height, width = self.mask.shape
            rgba = np.zeros((height, width, 4), np.uint8)
            rgba[self.mask == OFF, 3] = 150  # dim what will not be tracked
            self._overlay_buffer = np.ascontiguousarray(rgba)
            self._overlay = QImage(
                self._overlay_buffer.data, width, height, 4 * width, QImage.Format.Format_RGBA8888
            )
        return self._overlay

    def paint_on_canvas(self, painter: CanvasPainter, frame_number=None, frame=None) -> None:
        self.set_frame(frame)
        if not self._active():
            return

        painter.drawImage(0, 0, self._overlay_image())
        painter.setBrush(Qt.BrushStyle.NoBrush)

        if self.exclusive_rois.isChecked():
            for color, path in zip(cycle(self.exclusive_ROI_colors), self.exclusive_ROI_paths):
                painter.setPen(color)
                painter.drawPath(path)

        if self.pending is not None:
            p = self.pending
            color = INCLUDE_COLOR if p["include"] else EXCLUDE_COLOR
            pen = QPen(color)
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            box = QRectF(
                min(p["x0"], p["x1"]),
                min(p["y0"], p["y1"]),
                abs(p["x1"] - p["x0"]),
                abs(p["y1"] - p["y0"]),
            )
            if p["kind"] == RECT:
                painter.drawRect(box)
            else:
                painter.drawEllipse(box)
            painter.setPen(Qt.PenStyle.SolidLine)
            painter.setBrush(color)
            for cx, cy in (
                (p["x0"], p["y0"]),
                (p["x1"], p["y0"]),
                (p["x0"], p["y1"]),
                (p["x1"], p["y1"]),
            ):
                painter.drawBigPoint(cx, cy, 9)
            painter.setBrush(Qt.BrushStyle.NoBrush)

        if self.tool in (BRUSH, ERASER) and self._cursor_xy is not None:
            painter.setPen(QColor(255, 255, 255))
            radius = self._radius()
            painter.drawEllipse(QPointF(*self._cursor_xy), radius, radius)
```

- [ ] **Step 4: Run the widget tests**

Run: `$env:QT_QPA_PLATFORM="offscreen"; python -m pytest tests/roi_widget_test.py -q -p no:cacheprovider`
Expected: all pass (about 33). If `test_alt_flips_the_mode_while_held` fails because `QApplication.keyboardModifiers` cannot be patched on the class, patch it on the module instead: `monkeypatch.setattr("idtrackerai.segmentation_app.widgets.ROI_widget.QApplication.keyboardModifiers", ...)`. If `QShortcut` cannot be imported from `qtpy.QtGui`, import it from `qtpy.QtWidgets`.

- [ ] **Step 5: Commit**

```bash
git add src/idtrackerai/segmentation_app/widgets/ROI_widget.py tests/roi_widget_test.py
git commit -m "Rewrite the ROI widget as a mask editor

Brush, eraser, click select, rectangle, ellipse, move, undo and redo, with a
saved-arena library. Keeps the interface the Segmentation App already uses.

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Wire the editor into the Segmentation App

**Files:**
- Modify: `src/idtrackerai/segmentation_app/main.py`
- Modify: `src/idtrackerai/segmentation_app/tooltips.toml:5`
- Modify: `src/idtrackerai/segmentation_app/widgets/detectron2_panel.py` (the tip in `save_colab_bundle`)
- Test: `tests/roi_app_wiring_test.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/roi_app_wiring_test.py`:

```python
"""The Segmentation App saves, loads and hands over the ROI mask."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication, QMessageBox

from idtrackerai import Session
from idtrackerai.GUI_tools import CanvasMouseEvent
from idtrackerai.segmentation_app.main import SegmentationGUI
from idtrackerai.segmentation_app.widgets.ROI_widget import RECT
from idtrackerai.utils import build_ROI_mask_from_list
from idtrackerai.utils.roi_mask import empty_mask, load_mask_png, save_mask_png

app = QApplication.instance() or QApplication([])
LEFT = Qt.MouseButton.LeftButton


@pytest.fixture
def gui(tmp_path, monkeypatch):
    # keep the saved-arena library out of the real user profile
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("HOME", str(tmp_path))
    window = SegmentationGUI()
    window.ROI_Widget.set_video_size((200, 100))
    return window


def draw_rectangle(gui, p0, p1):
    roi = gui.ROI_Widget
    roi.CheckBox.setChecked(True)
    roi.set_tool(RECT)
    roi.press_event(CanvasMouseEvent(LEFT, 1.0, p0))
    roi.drag_event(CanvasMouseEvent(LEFT, 1.0, p1))
    roi.release_event(CanvasMouseEvent(LEFT, 1.0, p1))
    roi.commit_pending()


def test_ticking_the_roi_puts_the_canvas_in_tool_mode(gui):
    canvas = gui.videoPlayer.canvas
    assert not canvas.tool_mode
    gui.ROI_Widget.CheckBox.setChecked(True)
    assert canvas.tool_mode
    gui.ROI_Widget.CheckBox.setChecked(False)
    assert not canvas.tool_mode


def test_canvas_drags_reach_the_roi_widget(gui):
    gui.ROI_Widget.CheckBox.setChecked(True)
    gui.ROI_Widget.set_tool(RECT)
    canvas = gui.videoPlayer.canvas
    canvas.press_event.emit(CanvasMouseEvent(LEFT, 1.0, (20.0, 20.0)))
    canvas.drag_event.emit(CanvasMouseEvent(LEFT, 1.0, (80.0, 60.0)))
    canvas.release_event.emit(CanvasMouseEvent(LEFT, 1.0, (80.0, 60.0)))
    assert gui.ROI_Widget.pending is not None


def test_the_roi_mask_reaches_the_frame_analyzer(gui):
    draw_rectangle(gui, (20.0, 20.0), (80.0, 60.0))
    assert gui.frame_analyzer.ROI_mask is not None
    assert gui.frame_analyzer.ROI_mask[40, 50] == 255


def test_saving_parameters_writes_the_mask_beside_the_toml(gui, tmp_path):
    draw_rectangle(gui, (20.0, 20.0), (80.0, 60.0))
    parameters = {"roi_list": gui.ROI_Widget.getValue(), "number_of_animals": 1}
    out = gui.with_roi_mask_file(parameters, tmp_path / "setup.toml")
    assert out["roi_mask"] == "setup_roi.png"
    assert np.array_equal(load_mask_png(tmp_path / "setup_roi.png"), gui.ROI_Widget.getMask())
    assert list(out)[: len(parameters)] == list(parameters)


def test_no_roi_means_no_mask_file(gui, tmp_path):
    parameters = {"roi_list": None}
    assert gui.with_roi_mask_file(parameters, tmp_path / "setup.toml") == parameters
    assert not (tmp_path / "setup_roi.png").exists()


def test_a_saved_mask_is_preferred_over_the_polygon_list(gui, tmp_path):
    mask = empty_mask(200, 100)
    mask[10:40, 10:90] = 255
    save_mask_png(tmp_path / "m.png", mask)
    gui.session.roi_mask = str(tmp_path / "m.png")
    gui.session.roi_list = ["+ Polygon [[0.0, 0.0], [150.0, 0.0], [150.0, 90.0], [0.0, 90.0]]"]
    gui.load_roi_parameters()
    assert np.array_equal(gui.ROI_Widget.getMask(), mask)


def test_a_missing_mask_falls_back_to_the_polygon_list(gui, tmp_path, monkeypatch):
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: warned.append(a[2])))
    roi_list = ["+ Polygon [[0.0, 0.0], [150.0, 0.0], [150.0, 90.0], [0.0, 90.0]]"]
    gui.session.roi_mask = str(tmp_path / "gone.png")
    gui.session.roi_list = roi_list
    gui.load_roi_parameters()
    assert warned
    assert np.array_equal(gui.ROI_Widget.getMask(), build_ROI_mask_from_list(roi_list, 200, 100))


def test_polygon_only_files_still_load(gui):
    roi_list = ["+ Polygon [[10.0, 10.0], [100.0, 10.0], [100.0, 60.0], [10.0, 60.0]]"]
    gui.session.roi_mask = None
    gui.session.roi_list = roi_list
    gui.load_roi_parameters()
    assert np.array_equal(gui.ROI_Widget.getMask(), build_ROI_mask_from_list(roi_list, 200, 100))


def test_tracking_gets_the_exact_mask_through_a_temporary_file(gui, tmp_path):
    draw_rectangle(gui, (20.0, 20.0), (80.0, 60.0))
    gui.session.video_paths = [tmp_path / "v.mp4"]
    gui.attach_roi_mask_for_tracking({"roi_list": gui.ROI_Widget.getValue()})
    assert str(gui.session.roi_mask).endswith(Session.TEMPORARY_ROI_MASK_SUFFIX)
    assert np.array_equal(load_mask_png(gui.session.roi_mask), gui.ROI_Widget.getMask())


def test_tracking_without_an_roi_clears_a_stale_mask(gui, tmp_path):
    gui.session.video_paths = [tmp_path / "v.mp4"]
    gui.session.roi_mask = str(tmp_path / "old.png")
    gui.attach_roi_mask_for_tracking({"roi_list": None})
    assert gui.session.roi_mask is None
```

- [ ] **Step 2: Run to verify they fail**

Run: `$env:QT_QPA_PLATFORM="offscreen"; python -m pytest tests/roi_app_wiring_test.py -q -p no:cacheprovider -x`
Expected: failure in the fixture or first test (`ROIWidget` still wired the old way: `canvas.tool_mode` stays False).

- [ ] **Step 3: Edit `main.py`**

(a) Imports. Next to the other `idtrackerai.utils` import add:

```python
from idtrackerai.utils.roi_mask import save_mask_png
```

(b) Replace `self.ROI_Widget = ROIWidget(self)` with:

```python
        self.ROI_Widget = ROIWidget()
```

(c) Replace the line `self.videoPlayer.canvas.click_event.connect(self.ROI_Widget.click_event)` with:

```python
        canvas = self.videoPlayer.canvas
        canvas.press_event.connect(self.ROI_Widget.press_event)
        canvas.drag_event.connect(self.ROI_Widget.drag_event)
        canvas.release_event.connect(self.ROI_Widget.release_event)
        canvas.move_event.connect(self.ROI_Widget.move_event)
        self.ROI_Widget.toolModeChanged.connect(canvas.set_tool_mode)
        self.ROI_Widget.install_shortcuts(canvas)
        self.enhancement.settingsChanged.connect(self.ROI_Widget.set_enhancement)
```

(d) In `load_parameters`, replace `self.ROI_Widget.setValue(self.session.roi_list, self.session.exclusive_rois)` with `self.load_roi_parameters()` and add this method directly below `load_parameters`:

```python
    def load_roi_parameters(self) -> None:
        """The exact mask when the parameters name one, otherwise the polygons."""
        if self.session.roi_mask:
            try:
                self.ROI_Widget.load_mask_file(
                    self.session.roi_mask, self.session.exclusive_rois
                )
                return
            except (OSError, ValueError) as exc:
                QMessageBox.warning(
                    self,
                    "ROI mask not used",
                    f"{exc}\n\nThe polygon list in the parameters is used instead.",
                )
        self.ROI_Widget.setValue(self.session.roi_list, self.session.exclusive_rois)
```

(e) Replace `keyPressEvent`:

```python
    def keyPressEvent(self, event: QKeyEvent):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self.ROI_Widget.commit_pending()
        elif event.key() == Qt.Key.Key_Escape:
            self.ROI_Widget.cancel_pending()
```

(f) In `new_video_paths`, delete the line `self.ROI_Widget.list.ListChanged.emit()`. (`set_video_size`, called just above it, now refreshes the mask.)

(g) Add these two methods next to `save_parameters_func`:

```python
    def with_roi_mask_file(self, parameters: dict, toml_path: Path) -> dict:
        """Write the exact ROI mask beside the .toml and name it in the parameters.

        ``roi_list`` stays as the polygon approximation for older readers; the
        name is relative, so the setup folder can be moved as a whole.
        """
        mask = self.ROI_Widget.getMask()
        if mask is None or parameters.get("roi_list") is None:
            return parameters
        png = toml_path.with_name(toml_path.stem + "_roi.png")
        save_mask_png(png, mask)
        return {**parameters, "roi_mask": png.name}

    def attach_roi_mask_for_tracking(self, parameters: dict) -> None:
        """Give the session the exact mask for this run, not the polygons.

        The file sits beside the video and is deleted by the session once it has
        kept its own copy.
        """
        self.session.roi_mask = None
        mask = self.ROI_Widget.getMask()
        if mask is None or parameters.get("roi_list") is None:
            return
        path = Path(self.session.video_paths[0]).with_suffix(
            Session.TEMPORARY_ROI_MASK_SUFFIX
        )
        try:
            save_mask_png(path, mask)
        except OSError as exc:
            logging.warning("Could not write the exact ROI mask (%s); using polygons", exc)
            return
        self.session.roi_mask = str(path)
```

(h) In `save_parameters_func`, change the `try:` block so the mask file is written inside it:

```python
        try:
            parameters = self.with_roi_mask_file(parameters, Path(fileName))
            text = "".join(
                f"{key} = {toml_format(value)}\n" for key, value in parameters.items()
            )
            Path(fileName).write_text(text, encoding="utf_8")
        except (OSError, ValueError, TypeError) as exc:
            QMessageBox.warning(self, "Could not save the parameters", str(exc))
```

(i) In `close_and_track_video`, directly after the line `self.session.enhancement = parameters.get("enhancement")` add:

```python
        self.attach_roi_mask_for_tracking(parameters)
```

- [ ] **Step 4: Edit the tooltip and the Colab tip**

In `tooltips.toml` replace the `region_of_interest` line with:

```toml
region_of_interest = "<qt>Choose the area to track. Tick it, then paint with the brush (B), click a region with Click select (C), or drag a rectangle (R) or ellipse (O). Include/Exclude, or hold Alt, takes areas away. Undo with Ctrl+Z. Save the finished region as a named arena to reuse it. Pan with the middle button or Ctrl + drag.</qt>"
```

In `detectron2_panel.py`, in `save_colab_bundle`, replace the tip text

```python
            "Tip: 'Save parameters' writes a .toml the notebook will inherit, "
            "so your animal count, ROI and area thresholds carry over.",
```

with

```python
            "Tip: 'Save parameters' writes a .toml the notebook will inherit, "
            "so your animal count, ROI and area thresholds carry over. The ROI "
            "is also saved as a mask image beside the .toml (name_roi.png): "
            "upload that file next to the .toml.",
```

- [ ] **Step 5: Run the wiring tests and the neighbours**

Run: `$env:QT_QPA_PLATFORM="offscreen"; python -m pytest tests/roi_app_wiring_test.py tests/roi_widget_test.py tests/canvas_tool_mode_test.py -q -p no:cacheprovider`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/idtrackerai/segmentation_app tests/roi_app_wiring_test.py
git commit -m "Wire the ROI editor into the Segmentation App

Canvas drags reach the tools, saved parameters carry a mask PNG beside the
.toml, tracking receives the exact mask, and old polygon files still load.

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Documentation and final verification

**Files:**
- Modify: `docs/superpowers/specs/2026-10-09-roi-editor-design.md`
- Modify: `USER_GUIDE.md` (append a section)

- [ ] **Step 1: Bring the spec in line with the decisions**

In the spec:
- In "User experience", change `Click select (W)` to `Click select (C)`, add `Move (M)` to the tool strip list, and replace "Pan with Space + drag and zoom with the wheel, as in the existing canvas." with "Pan with the middle button or Ctrl + left drag, and zoom with the wheel. Space keeps playing and pausing the video."
- In "Applying an arena to a video", replace "Arrow keys nudge it by 1 px (Shift: 10 px), and dragging with the Move handle shifts it as a whole." with "Four nudge buttons in the panel move it by 1 px (Shift: 10 px), and the Move tool (M) drags it as a whole. Arrow keys are not used because they step the video frames."
- Delete the "Open points for review" section, since both points are now decided (shortcuts listed above; previews show the mask over the reference frame).

- [ ] **Step 2: Add a user guide section**

Append to `USER_GUIDE.md`:

```markdown
## Regions of interest

Tick **Regions of interest** in the Segmentation App to choose the area that is tracked. Everything outside it is dimmed on the video.

| Tool | Key | Use it to |
|---|---|---|
| Brush | B | Paint the area. Size with the slider, or `[` and `]`. |
| Eraser | E | Paint areas away. |
| Click select | C | Click a region and it fills out to its edges, like a magic wand. Raise the tolerance if it stops too early, lower it if it leaks. |
| Rectangle, Ellipse | R, O | Drag a shape, then adjust its corner handles or drag it. Press Enter to apply it, or Esc to drop it. |
| Move | M | Drag the whole region, for example when the camera shifted a little. The four nudge buttons move it a pixel at a time (Shift for 10). |

**Include / Exclude** decides whether a tool adds to the region or takes from it. Hold **Alt** to flip it while you use a tool. **Ctrl+Z** and **Ctrl+Y** undo and redo. Pan with the middle mouse button, or Ctrl and drag.

**Saved arenas.** Once the region is right, **Save as...** keeps it under a name in your user profile. Later, pick it from the list for any video of the same setup. If the video is a different size with the same proportions, you are asked whether to scale it. **Export...** and **Import...** share an arena as one zip file.

When you save parameters, the region is also written as an image next to the `.toml` (`name_roi.png`) and named in it as `roi_mask`. Tracking uses that exact image. Keep the two files together, and upload both when you track on Colab.
```

- [ ] **Step 3: Run the whole suite**

Run: `$env:QT_QPA_PLATFORM="offscreen"; python -m pytest tests -q --deselect tests/smoke_test.py -p no:cacheprovider`
Expected: everything passes (previous 72 plus about 110 new). If a test that existed before fails, stop and investigate before continuing.

- [ ] **Step 4: Start the app offscreen and load an old file**

Create the old-format file and a check script in the scratchpad, then run it:

```python
# check_roi_app.py
import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import numpy as np
from qtpy.QtWidgets import QApplication
app = QApplication([])
from idtrackerai.segmentation_app.main import SegmentationGUI
from idtrackerai.utils import build_ROI_mask_from_list

gui = SegmentationGUI()
gui.ROI_Widget.set_video_size((640, 480))
roi = ["+ Polygon [[50.0, 50.0], [600.0, 50.0], [600.0, 400.0], [50.0, 400.0]]"]
gui.session.roi_mask = None
gui.session.roi_list = roi
gui.load_roi_parameters()
expected = build_ROI_mask_from_list(roi, 640, 480)
assert np.array_equal(gui.ROI_Widget.getMask(), expected)
assert gui.videoPlayer.canvas.tool_mode
print("old polygon file loads into the new editor: OK")
```

Run: `python <scratchpad>\check_roi_app.py`
Expected: `old polygon file loads into the new editor: OK`.

- [ ] **Step 5: Look at it**

Run the real app against a real video and try each tool once: `idtrackerai`, open a video, tick Regions of interest, brush a stroke, click-select a strip, drag a rectangle and resize it by a corner, Ctrl+Z, Save as..., Clear, pick the arena again, Save parameters, reopen the `.toml`. Report anything that looks or feels wrong rather than fixing it silently.

- [ ] **Step 6: Commit and publish the branch**

```bash
git add USER_GUIDE.md docs/superpowers/specs/2026-10-09-roi-editor-design.md
git commit -m "Document the ROI editor and align the spec with the decisions

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
git push github roi-editor
```
Expected: `* [new branch] roi-editor -> roi-editor`. Do not merge to `master` until the user has tried it, because the Colab notebook installs from `master`.

---

## Self-review against the spec

| Spec requirement | Task |
|---|---|
| One mask is the source of truth; helpers testable without Qt | 1 |
| Brush, eraser, click select, rectangle, ellipse with handles, move, undo/redo | 1, 5 |
| Include/Exclude toggle, Alt flips | 5 |
| Dimmed outside, always visible | 5 (`_overlay_image`) |
| Pan without conflicting with the video player | 4 |
| Saved arenas: save, list, load, delete, export, import, safe zip | 2, 5 |
| Apply arena: same size, scale after asking, refuse other proportions, nudge/move | 5 |
| Old `.toml` and polygon strings still load; `roi_list` still written | 1, 5, 6 |
| `roi_mask` parameter, precedence, size check, relative path | 3, 6 |
| Colab carries the mask PNG | 6 (tip text), 7 (guide) |
| Error handling: empty mask, unreadable or wrong-sized PNG, library errors | 3, 5, 6 |
| Tests listed in the spec | 1, 2, 3, 4, 5, 6 |
| Not in scope: Suggest button, rotate/warp, auto-align | not planned |
