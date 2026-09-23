"""Filesystem metadata must never be mistaken for a video.

macOS writes a companion file named "._<original>" beside every file it copies
onto a non-Apple filesystem, holding resource-fork metadata. The companion
carries the same extension as the file it shadows, so a folder of 49 clips
globs as 98 ".mp4" paths and the first companion OpenCV is handed is reported
as an unreadable video.
"""

from pathlib import Path

import pytest

from idtrackerai.extra_tools.detectron2_pipeline import sampling
from idtrackerai.utils import drop_metadata_sidecars, is_metadata_sidecar


@pytest.mark.parametrize(
    "name, expected",
    [
        ("._B3_N1_segment_2.mp4", True),
        ("._clip.avi", True),
        (".DS_Store", True),
        ("Thumbs.db", True),
        ("desktop.ini", True),
        ("B3_N1_segment_2.mp4", False),
        ("clip.avi", False),
        # a leading dot alone is not enough; the marker is the underscore
        (".hidden_but_real.mp4", False),
        # and the sequence must be at the start, not merely present
        ("recording._2.mp4", False),
    ],
)
def test_recognises_sidecars(name, expected):
    assert is_metadata_sidecar(name) is expected


def test_splits_without_losing_anything():
    names = ["a.mp4", "._a.mp4", "b.mp4", "._b.mp4", ".DS_Store"]
    kept, dropped = drop_metadata_sidecars(names)
    assert [p.name for p in kept] == ["a.mp4", "b.mp4"]
    assert len(kept) + len(dropped) == len(names)


def test_resolve_videos_drops_them(tmp_path: Path):
    """The shape of the real failure: one companion per clip, same extension."""
    for stem in ("clip_01", "clip_02", "clip_03"):
        (tmp_path / f"{stem}.mp4").write_bytes(b"pretend video")
        # 4096 bytes is what macOS actually writes
        (tmp_path / f"._{stem}.mp4").write_bytes(b"\x00" * 4096)

    assert len(list(tmp_path.glob("*.mp4"))) == 6, "fixture should look ambiguous"

    videos = sampling.resolve_videos([tmp_path / "*.mp4"])
    assert len(videos) == 3
    assert all(not v.name.startswith("._") for v in videos)


def test_explicit_paths_are_filtered_too(tmp_path: Path):
    """Not only globs: a multi-select in the file dialog hits the same thing."""
    real = tmp_path / "clip.mp4"
    real.write_bytes(b"pretend video")
    companion = tmp_path / "._clip.mp4"
    companion.write_bytes(b"\x00" * 4096)

    videos = sampling.resolve_videos([real, companion])
    assert videos == [real]
