"""Extracting frames from videos for annotation.

The logic lives here rather than in the command-line script so that the
Segmentation App and the CLI run the same code. :func:`sample_frames` takes
``progress`` and ``abort`` callables, matching the convention
:func:`idtrackerai.base.animals_detection.segmentation.generate_frame_stack`
already uses, which is what lets one function drive a terminal progress bar and
a ``QProgressDialog`` without knowing which it is talking to.

Sampling is stratified by default: each video is cut into as many equal blocks
as there are frames to take, and one frame is drawn from each. Uniform random
sampling over a 30000-frame clip clumps, leaving stretches unrepresented;
stratifying guarantees coverage of the whole recording, including whatever the
lighting does halfway through.
"""

import json
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import cv2

try:
    from .errors import SamplingError
    from .preprocessing import make_enhancer_from, save_profile
except ImportError:  # loaded by path, without the package around it
    from errors import SamplingError  # type: ignore[no-redef]
    from preprocessing import make_enhancer_from, save_profile  # type: ignore[no-redef]


@dataclass
class VideoPlan:
    """How many frames to take from one video, and which."""

    video: Path
    available: int
    requested: int
    indices: list[int] = field(default_factory=list)


@dataclass
class SamplingResult:
    frames: list[dict]
    requested: int
    output: Path
    manifest_path: Path | None = None
    profile_path: Path | None = None
    per_video: dict[str, int] = field(default_factory=dict)
    unreadable: list[str] = field(default_factory=list)
    complete: bool = True

    @property
    def written(self) -> int:
        return len(self.frames)


def video_frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SamplingError(f"Could not open {path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count


def stratified_indices(n_frames: int, n_take: int, rng: random.Random) -> list[int]:
    """One frame from each of n_take equal blocks, at a random point within it."""
    if n_take >= n_frames:
        return list(range(n_frames))
    edges = [round(i * n_frames / n_take) for i in range(n_take + 1)]
    return [
        rng.randrange(edges[i], max(edges[i + 1], edges[i] + 1)) for i in range(n_take)
    ]


def uniform_indices(n_frames: int, n_take: int, rng: random.Random) -> list[int]:
    if n_take >= n_frames:
        return list(range(n_frames))
    return sorted(rng.sample(range(n_frames), n_take))


def allocate(counts: Sequence[int], total: int, even: bool) -> list[int]:
    """Decides how many frames to take from each video."""
    n_videos = len(counts)
    if n_videos == 0:
        raise SamplingError("No videos to sample from")

    if even:
        base, extra = divmod(total, n_videos)
        return [base + (1 if i < extra else 0) for i in range(n_videos)]

    grand_total = sum(counts)
    if grand_total == 0:
        raise SamplingError("The videos contain no frames")
    allocation = [round(total * c / grand_total) for c in counts]

    # rounding can drift; fix it against the longest videos
    while sum(allocation) != total:
        step = 1 if sum(allocation) < total else -1
        order = sorted(range(n_videos), key=lambda i: counts[i], reverse=step > 0)
        for i in order:
            if 0 <= allocation[i] + step <= counts[i]:
                allocation[i] += step
                break
        else:
            break
    return allocation


def resolve_videos(patterns: Sequence[Path]) -> list[Path]:
    """Expands globs and drops anything that is not a file."""
    videos: list[Path] = []
    for pattern in patterns:
        if any(ch in str(pattern) for ch in "*?"):
            videos.extend(sorted(pattern.parent.glob(pattern.name)))
        else:
            videos.append(Path(pattern))
    return [v for v in videos if v.is_file()]


def plan_sampling(
    videos: Sequence[Path],
    n_frames: int,
    seed: int = 0,
    even: bool = False,
    uniform_random: bool = False,
) -> list[VideoPlan]:
    """Works out what would be sampled, without touching a single frame.

    Cheap enough for the GUI to call on every spinbox change, so the user can
    see "600 frames: clip_01 -> 312, clip_02 -> 288" before committing.
    """
    if not videos:
        raise SamplingError("No videos matched")

    counts = [video_frame_count(v) for v in videos]
    allocation = allocate(counts, n_frames, even)
    rng = random.Random(seed)
    pick = uniform_indices if uniform_random else stratified_indices

    plans = []
    for video, available, requested in zip(videos, counts, allocation):
        indices = pick(available, requested, rng) if requested else []
        plans.append(VideoPlan(video, available, requested, indices))
    return plans


def sample_frames(
    videos: Sequence[Path],
    output: Path,
    n_frames: int,
    settings: dict,
    seed: int = 0,
    even: bool = False,
    uniform_random: bool = False,
    image_format: str = "png",
    progress: Callable[[int], None] | None = None,
    abort: Callable[[], bool] | None = None,
) -> SamplingResult:
    """Writes enhanced frames for annotation, plus a manifest and a profile.

    The frames are written already enhanced, with the same settings inference
    will use, so that what gets annotated is what the model will see.

    An aborted run keeps what it has written and reports ``complete=False``.
    That is a deliberate departure from the background-computation thread this
    otherwise mirrors, which discards partial results: two hundred annotated
    frames are genuinely useful, an incomplete background model is not.
    """
    videos = list(videos)
    plans = plan_sampling(videos, n_frames, seed, even, uniform_random)
    output.mkdir(parents=True, exist_ok=True)
    enhance = make_enhancer_from(settings)

    records: list[dict] = []
    per_video: dict[str, int] = {}
    unreadable: list[str] = []
    complete = True

    for plan in plans:
        if plan.requested == 0:
            per_video[plan.video.name] = 0
            continue

        cap = cv2.VideoCapture(str(plan.video))
        written = 0
        for index in plan.indices:
            if abort is not None and abort():
                complete = False
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                unreadable.append(f"{plan.video.name}:{index}")
                continue

            image = enhance(frame)
            name = f"{plan.video.stem}_f{index:06d}.{image_format}"
            # imencode+tofile rather than imwrite, so non-ASCII paths work
            cv2.imencode(f".{image_format}", image)[1].tofile(str(output / name))
            records.append(
                {
                    "file_name": name,
                    "source_video": plan.video.name,
                    "frame_number": index,
                    "width": image.shape[1],
                    "height": image.shape[0],
                }
            )
            written += 1
            if progress is not None:
                progress(len(records))

        cap.release()
        per_video[plan.video.name] = written
        if not complete:
            break

    result = SamplingResult(
        frames=records,
        requested=n_frames,
        output=output,
        per_video=per_video,
        unreadable=unreadable,
        complete=complete,
    )

    if not records:
        return result

    manifest = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "sampling": "uniform-random" if uniform_random else "stratified",
        "allocation": "even" if even else "proportional",
        "complete": complete,
        "enhancement": settings,
        "videos": [str(v) for v in videos],
        "frames": records,
    }
    result.manifest_path = output / "sampling_manifest.json"
    result.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # The settings live beside the frames as a reusable profile, so the same
    # recording setup can be prepared identically next time, and so the
    # conversion step can carry them on towards training.
    result.profile_path = save_profile(
        output / "preprocess_profile.json",
        settings,
        name=output.name,
        notes="Written by the sampling step; reuse with --preprocess-profile",
    )
    return result
