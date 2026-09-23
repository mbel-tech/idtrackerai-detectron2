"""Where a half-finished preparation is remembered between sessions.

Preparing a Detectron2 model takes days: sample frames, annotate them by hand,
build a dataset, then go to Colab. The app must be closeable in the middle of
that, so each step records what it produced.

The state lives beside the data, not in QSettings, for two reasons. It has to
travel when the folder is copied to another machine, and it has to be readable
and diffable when something looks wrong. It is scoped to the *folder*, not to a
session name, because an annotation corpus belongs to a recording setup and is
shared across every clip and session from that rig.

Disk is the truth and this file is only a hint: :meth:`PrepState.refresh` checks
every claim against what is actually there, so a deleted folder shows as an
incomplete step rather than a step that lies about being done.
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

FORMAT_VERSION = 1
STATE_DIR = "detectron2_prep"
STATE_FILE = "prep_state.json"


def prep_root(video_paths, output_dir=None) -> Path:
    """The folder this preparation belongs to.

    Mirrors how :meth:`Session.prepare_tracking` picks its output location
    (``session.py:286``): the configured output directory, or the folder the
    first video sits in.
    """
    if output_dir:
        return Path(output_dir) / STATE_DIR
    if video_paths:
        return Path(video_paths[0]).parent / STATE_DIR
    raise ValueError("Neither an output directory nor any video path was given")


@dataclass
class PrepState:
    root: Path
    enhancement: dict = field(default_factory=dict)
    profile: str | None = None
    frames_dir: str | None = None
    n_sampled: int = 0
    sampling_complete: bool = False
    annotated: int = 0
    dataset_dir: str | None = None
    n_train: int = 0
    n_val: int = 0
    updated: str = ""

    # --------------------------------------------------------------- on disk
    @property
    def path(self) -> Path:
        return self.root / STATE_FILE

    def _absolute(self, stored: str | None) -> Path | None:
        if not stored:
            return None
        path = Path(stored)
        return path if path.is_absolute() else (self.root / path)

    def _store(self, path: Path | str | None) -> str | None:
        """Relative to the root where possible, so the folder can be moved."""
        if path is None:
            return None
        path = Path(path)
        try:
            return str(path.relative_to(self.root))
        except ValueError:
            return str(path)

    @property
    def frames_path(self) -> Path | None:
        return self._absolute(self.frames_dir)

    @property
    def dataset_path(self) -> Path | None:
        return self._absolute(self.dataset_dir)

    def set_frames_dir(self, path: Path | None) -> None:
        self.frames_dir = self._store(path)

    def set_dataset_dir(self, path: Path | None) -> None:
        self.dataset_dir = self._store(path)

    def set_profile(self, path: Path | None) -> None:
        self.profile = self._store(path)

    # ------------------------------------------------------------ load, save
    @classmethod
    def load(cls, root: Path) -> "PrepState":
        state = cls(root=Path(root))
        path = state.path
        if not path.is_file():
            return state

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logging.warning("Could not read %s (%s); starting fresh", path, exc)
            return state

        version = data.get("format_version")
        if version != FORMAT_VERSION:
            logging.warning(
                "%s has format version %s, expected %s; starting fresh",
                path, version, FORMAT_VERSION,
            )
            return state

        for key, value in data.items():
            if key != "format_version" and hasattr(state, key) and key != "root":
                setattr(state, key, value)
        return state

    def save(self) -> bool:
        """Writes atomically. Returns False if the location is not writable.

        A half-written file after a crash would break resume, so it is written
        beside the target and moved into place. Failure is not fatal: a
        read-only share means the panel works, it just cannot remember.
        """
        data: dict[str, Any] = {"format_version": FORMAT_VERSION}
        data.update({k: v for k, v in asdict(self).items() if k != "root"})
        data["updated"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.updated = data["updated"]

        try:
            self.root.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(STATE_FILE + ".tmp")
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            os.replace(tmp, self.path)
            return True
        except OSError as exc:
            logging.warning("Could not save preparation state to %s: %s", self.path, exc)
            return False

    # ----------------------------------------------------------- reality check
    def refresh(self) -> None:
        """Re-derives every step's status from what is actually on disk."""
        frames = self.frames_path
        if frames is None or not frames.is_dir():
            self.frames_dir = None
            self.n_sampled = 0
            self.annotated = 0
        else:
            self.n_sampled = len(
                [p for p in frames.iterdir() if p.suffix.lower() in (".png", ".jpg")]
            )
            self.annotated = _count_annotations(frames)

        dataset = self.dataset_path
        if dataset is None or not (dataset / "train.json").is_file():
            self.dataset_dir = None
            self.n_train = self.n_val = 0

        profile = self._absolute(self.profile)
        if profile is not None and not profile.is_file():
            self.profile = None

    @property
    def sampling_done(self) -> bool:
        return self.n_sampled > 0

    @property
    def annotation_done(self) -> bool:
        return self.n_sampled > 0 and self.annotated >= self.n_sampled

    @property
    def dataset_done(self) -> bool:
        return self.n_train > 0

    def dataset_is_stale(self) -> bool:
        """True when annotations have changed since the dataset was built."""
        dataset, frames = self.dataset_path, self.frames_path
        if dataset is None or frames is None:
            return False
        report = dataset / "report.json"
        if not report.is_file():
            return False
        newest = max(
            (p.stat().st_mtime for p in frames.glob("*.json")), default=0.0
        )
        return newest > report.stat().st_mtime


def _count_annotations(folder: Path) -> int:
    try:
        from .dataset import is_labelme
    except ImportError:  # loaded by path
        from dataset import is_labelme  # type: ignore[no-redef]
    return sum(1 for p in folder.glob("*.json") if is_labelme(p))
