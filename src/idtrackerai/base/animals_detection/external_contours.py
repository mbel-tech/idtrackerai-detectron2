"""Reading externally computed contours (e.g. Detectron2 instance masks) instead
of thresholding the video.

idtracker.ai's native segmentation turns pixels into contours with
:func:`~idtrackerai.base.animals_detection.segmentation.process_frame`. When an
external instance-segmentation model has already found the animals, that step is
redundant and lossy: the masks would have to be painted back into the video and
re-thresholded. This module lets the contours enter the pipeline directly.

Everything downstream is unaffected. :class:`~idtrackerai.Blob` derives its
centroid, area, bounding box, extension and orientation from the contour, so
supplying a contour is enough to get every blob feature, computed by the same
code as before.

Sidecar file format
-------------------
Contours are stored in a single HDF5 file per video, in a ragged (CSR-like)
layout so that 30k frames do not become 30k datasets::

    /vertices        int32   [n_vertices, 2]    (x, y) of every contour point
    /vertex_offsets  int64   [n_contours + 1]   slices of /vertices per contour
    /frame_offsets   int64   [n_frames + 1]     slices of /vertex_offsets per frame
    /scores          float32 [n_contours]       optional detector confidence

Frame ``f`` owns contours ``frame_offsets[f]:frame_offsets[f + 1]``, and contour
``i`` owns vertices ``vertex_offsets[i]:vertex_offsets[i + 1]``.

Root attributes: ``format_version``, ``n_frames``, ``width``, ``height`` and
free-form provenance (``source``, ``model``, ``score_threshold``, ...).
"""

import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np

FORMAT_VERSION = 1

# h5py handles cannot be pickled and must not be shared across forked processes,
# so each worker opens its own. Keyed by (pid, path) and opened on first use.
_OPEN_FILES: dict[tuple[int, str], "ExternalContours"] = {}


class ExternalContours:
    """Random access to the contours of a single video, by global frame number."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"External contour file not found: {self.path}")

        self._file = h5py.File(self.path, "r")

        version = self._file.attrs.get("format_version", 0)
        if version != FORMAT_VERSION:
            raise ValueError(
                f"{self.path} has contour format version {version}, this "
                f"idtracker.ai expects {FORMAT_VERSION}"
            )

        self._vertices: h5py.Dataset = self._file["vertices"]  # type: ignore
        self._vertex_offsets: np.ndarray = self._file["vertex_offsets"][:]  # type: ignore
        self._frame_offsets: np.ndarray = self._file["frame_offsets"][:]  # type: ignore

        self.n_frames = int(self._file.attrs["n_frames"])  # type: ignore
        self.width = int(self._file.attrs["width"])  # type: ignore
        self.height = int(self._file.attrs["height"])  # type: ignore

    @property
    def attrs(self) -> dict[str, Any]:
        """Provenance recorded by the exporter (model, thresholds, date, ...)."""
        return dict(self._file.attrs)

    def contours_in_frame(self, frame_number: int) -> list[np.ndarray]:
        """Contours of one frame, each with shape [n_points, 2] and dtype int32.

        Frames outside the stored range return no contours rather than raising,
        matching how the native segmentation treats unreadable frames.
        """
        if not 0 <= frame_number < self.n_frames:
            return []

        first = int(self._frame_offsets[frame_number])
        last = int(self._frame_offsets[frame_number + 1])
        if first == last:
            return []

        # one read for the whole frame, then split, rather than a read per contour
        v_first = int(self._vertex_offsets[first])
        v_last = int(self._vertex_offsets[last])
        vertices: np.ndarray = self._vertices[v_first:v_last]

        bounds = self._vertex_offsets[first : last + 1] - v_first
        return [
            vertices[bounds[i] : bounds[i + 1]].astype(np.int32, copy=False)
            for i in range(last - first)
        ]

    def close(self) -> None:
        self._file.close()

    def __repr__(self) -> str:
        n_contours = len(self._vertex_offsets) - 1
        return (
            f"ExternalContours({self.path.name}, {self.n_frames} frames, "
            f"{n_contours} contours, {self.width}x{self.height})"
        )


class ContourSequence:
    """Several files read as one timeline, for a session made of several clips.

    idtracker.ai treats a list of videos as one concatenated recording, while
    the exporter writes one file per clip, so tracking the segments of a trial
    together needs the two views reconciled. Global frame ``f`` belongs to the
    file whose span contains it, and is read at its offset within that file.

    The order of ``paths`` must match the order of the session's video paths.
    Nothing here can detect a wrong order -- the contours would simply be
    applied to the wrong frames -- so the caller pairs them by name and the
    frame counts are checked per clip before tracking starts.
    """

    def __init__(self, paths: Sequence[Path | str]):
        paths = [Path(p) for p in paths]
        if not paths:
            raise ValueError("No external contour files were given")
        self.paths = paths
        self.readers = [ExternalContours(p) for p in paths]

        # global frame -> file, by binary search on the cumulative boundaries
        self._boundaries = np.cumsum(
            [0] + [reader.n_frames for reader in self.readers]
        )
        self.n_frames = int(self._boundaries[-1])

        sizes = {(reader.width, reader.height) for reader in self.readers}
        if len(sizes) != 1:
            detail = ", ".join(
                f"{r.path.name} {r.width}x{r.height}" for r in self.readers
            )
            self.close()
            raise ValueError(
                "External contour files were computed at different sizes, so "
                f"they cannot describe one recording: {detail}"
            )
        self.width, self.height = sizes.pop()

    def file_of(self, frame_number: int) -> tuple[int, int]:
        """(index of the file, frame number within it) for a global frame."""
        index = int(np.searchsorted(self._boundaries, frame_number, "right")) - 1
        return index, frame_number - int(self._boundaries[index])

    def contours_in_frame(self, frame_number: int) -> list[np.ndarray]:
        if not 0 <= frame_number < self.n_frames:
            return []
        index, local = self.file_of(frame_number)
        return self.readers[index].contours_in_frame(local)

    @property
    def attrs(self) -> dict[str, Any]:
        """Provenance of the first file; they come from one export run."""
        return self.readers[0].attrs if self.readers else {}

    def close(self) -> None:
        for reader in getattr(self, "readers", []):
            reader.close()

    def __repr__(self) -> str:
        return (
            f"ContourSequence({len(self.readers)} files, {self.n_frames} frames, "
            f"{self.width}x{self.height})"
        )


def _as_paths(path_or_paths) -> list[Path]:
    """One path or many, always as a list."""
    if isinstance(path_or_paths, (str, Path)):
        return [Path(path_or_paths)]
    return [Path(p) for p in path_or_paths]


def get_external_contours(path_or_paths) -> "ExternalContours | ContourSequence":
    """Per-process cached accessor, safe to call from multiprocessing workers.

    Accepts one path or several. Several are read as one concatenated
    timeline, matching how a session treats several video files.
    """
    paths = _as_paths(path_or_paths)
    key = (os.getpid(), tuple(str(p) for p in paths))
    if key not in _OPEN_FILES:
        _OPEN_FILES[key] = (
            ExternalContours(paths[0]) if len(paths) == 1 else ContourSequence(paths)
        )
        logging.debug(
            "Opened %d external contour file(s) in pid %d", len(paths), os.getpid()
        )
    return _OPEN_FILES[key]


def validate_against_video(
    path_or_paths,
    n_frames: int,
    width: int,
    height: int,
    video_paths: Sequence[Path | str] | None = None,
    per_video_frames: Sequence[int] | None = None,
) -> None:
    """Fails early, before segmenting, if the sidecars do not match the video.

    A silent mismatch here would be expensive: contours would be applied to the
    wrong frames and the error would only surface as nonsensical trajectories.
    With several files the risk is worse, because a wrong *order* still adds up
    to the right total, so when the caller can say how long each clip is, every
    file is checked against its own clip rather than only against the sum.
    """
    paths = _as_paths(path_or_paths)
    if not paths:
        raise ValueError("No external contour files were given")

    readers = [ExternalContours(p) for p in paths]
    try:
        problems = []
        total = sum(reader.n_frames for reader in readers)

        # per clip first: it says which file is wrong, not merely that one is
        if per_video_frames is not None:
            if len(per_video_frames) != len(readers):
                problems.append(
                    f"{len(readers)} contour file(s) were given for "
                    f"{len(per_video_frames)} video(s)"
                )
            else:
                for i, (reader, expected) in enumerate(zip(readers, per_video_frames)):
                    if reader.n_frames != expected:
                        video = (
                            Path(video_paths[i]).name
                            if video_paths is not None and i < len(video_paths)
                            else f"video {i + 1}"
                        )
                        problems.append(
                            f"{reader.path.name} covers {reader.n_frames} frames "
                            f"but {video} has {expected}"
                        )

        if total != n_frames:
            listing = ", ".join(
                f"{r.path.name} ({r.n_frames})" for r in readers
            )
            problems.append(
                f"they cover {total} frames in total but the session has "
                f"{n_frames}: {listing}"
            )

        sizes = {(r.width, r.height) for r in readers}
        if len(sizes) > 1:
            detail = ", ".join(f"{r.path.name} {r.width}x{r.height}" for r in readers)
            problems.append(f"they were computed at different sizes: {detail}")
        elif sizes and sizes != {(width, height)}:
            got_w, got_h = next(iter(sizes))
            problems.append(
                f"they were computed at {got_w}x{got_h} but the video is "
                f"{width}x{height}"
            )

        if problems:
            what = (
                f"External contour file {readers[0].path}"
                if len(readers) == 1
                else f"The {len(readers)} external contour files"
            )
            raise ValueError(f"{what} does not match the video: " + "; ".join(problems))

        if len(readers) == 1:
            logging.info("Using external contours: %s", readers[0])
        else:
            logging.info(
                "Using %d external contour files covering %d frames:",
                len(readers), total,
            )
            for reader in readers:
                logging.info("    %s (%d frames)", reader.path.name, reader.n_frames)
        for key in ("source", "model", "score_threshold", "created"):
            if key in readers[0].attrs:
                logging.info("    %s: %s", key, readers[0].attrs[key])
    finally:
        for reader in readers:
            reader.close()


def write_contours(
    path: Path | str,
    contours_per_frame: list[list[np.ndarray]],
    width: int,
    height: int,
    scores_per_frame: list[list[float]] | None = None,
    **provenance: Any,
) -> Path:
    """Writes a sidecar file. Used by the exporter that runs the detector.

    Parameters
    ----------
    contours_per_frame
        One list of contours per frame, in frame order, each contour an array of
        shape [n_points, 2] holding (x, y). Frames with no detection hold [].
    provenance
        Free-form attributes describing how the contours were produced. Recorded
        in the file and logged at tracking time.
    """
    path = Path(path)

    frame_offsets = np.zeros(len(contours_per_frame) + 1, np.int64)
    vertex_offsets = [0]
    all_vertices = []
    all_scores = []

    for frame_index, contours in enumerate(contours_per_frame):
        for contour_index, contour in enumerate(contours):
            contour = np.asarray(contour)
            if contour.ndim == 3 and contour.shape[1] == 1:
                # cv2.findContours returns (n_points, 1, 2)
                contour = contour[:, 0]
            if contour.ndim != 2 or contour.shape[1] != 2:
                raise ValueError(
                    f"Contour {contour_index} of frame {frame_index} has shape "
                    f"{contour.shape}, expected [n_points, 2]"
                )
            all_vertices.append(contour.astype(np.int32, copy=False))
            vertex_offsets.append(vertex_offsets[-1] + len(contour))
            if scores_per_frame is not None:
                all_scores.append(scores_per_frame[frame_index][contour_index])
        frame_offsets[frame_index + 1] = len(vertex_offsets) - 1

    vertices = (
        np.concatenate(all_vertices)
        if all_vertices
        else np.empty((0, 2), np.int32)
    )

    with h5py.File(path, "w") as file:
        file.create_dataset("vertices", data=vertices, compression="gzip")
        file.create_dataset(
            "vertex_offsets", data=np.asarray(vertex_offsets, np.int64)
        )
        file.create_dataset("frame_offsets", data=frame_offsets)
        if scores_per_frame is not None:
            file.create_dataset(
                "scores", data=np.asarray(all_scores, np.float32)
            )

        file.attrs["format_version"] = FORMAT_VERSION
        file.attrs["n_frames"] = len(contours_per_frame)
        file.attrs["width"] = width
        file.attrs["height"] = height
        for key, value in provenance.items():
            file.attrs[key] = value

    return path
