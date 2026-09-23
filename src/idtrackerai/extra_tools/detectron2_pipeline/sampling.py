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
    readable: bool = True
    """False when the file could not be opened or reports no frames.

    Unreadable clips stay in the plan rather than aborting it, so one bad file
    in a folder of fifty does not cost the whole run, and so the caller can
    name exactly which ones were skipped.
    """


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


def frame_count_or_zero(path: Path) -> int:
    """Frame count, or 0 for a file that cannot be read as a video."""
    try:
        return max(0, video_frame_count(path))
    except (SamplingError, cv2.error):
        return 0


def check_distinct_stems(videos: Sequence[Path]) -> None:
    """Refuses a set whose file names collide once the folders are dropped.

    Frames are named after the video stem and all land in one flat folder, so
    two clips called clip_01.mp4 in different folders would overwrite each
    other's frames and corrupt the by-recording split, silently. Better to say
    so before writing anything.
    """
    seen: dict[str, list[Path]] = {}
    for video in videos:
        seen.setdefault(video.stem, []).append(video)
    clashes = {stem: paths for stem, paths in seen.items() if len(paths) > 1}
    if not clashes:
        return
    detail = "; ".join(
        f"{stem}: " + ", ".join(str(p) for p in paths)
        for stem, paths in sorted(clashes.items())
    )
    raise SamplingError(
        "Two or more videos share a file name, and sampled frames are named "
        "after it, so they would overwrite each other. Rename or separate "
        f"them first. {detail}"
    )


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


def _fit_to_capacity(allocation: list[int], counts: Sequence[int], total: int) -> list[int]:
    """No video is asked for more frames than it has, and nothing is lost.

    The shortfall goes round the remaining videos one frame at a time rather
    than onto whichever happens to sort first, so a short clip in the set does
    not quietly hand its share to a single neighbour.
    """
    alloc = [min(a, c) for a, c in zip(allocation, counts)]
    short = total - sum(alloc)
    while short > 0:
        room = [i for i, (a, c) in enumerate(zip(alloc, counts)) if a < c]
        if not room:
            break
        for i in room:
            if short == 0:
                break
            alloc[i] += 1
            short -= 1
    return alloc


def allocate(counts: Sequence[int], total: int, even: bool) -> list[int]:
    """Decides how many frames to take from each video.

    Proportional shares rarely come out whole, and how the leftover frames are
    handed out is the whole ball game when the budget is small: giving them all
    to the longest video means asking for 20 frames from 49 clips samples one
    clip twenty times and 48 clips not at all. So the leftovers go to the
    largest fractional parts, one each -- the standard largest-remainder
    apportionment -- which keeps equal-length clips within one frame of each
    other at any budget.
    """
    n_videos = len(counts)
    if n_videos == 0:
        raise SamplingError("No videos to sample from")

    if even:
        base, extra = divmod(total, n_videos)
        allocation = [base + (1 if i < extra else 0) for i in range(n_videos)]
        return _fit_to_capacity(allocation, counts, total)

    grand_total = sum(counts)
    if grand_total == 0:
        raise SamplingError("The videos contain no frames")

    exact = [total * c / grand_total for c in counts]
    allocation = [int(e) for e in exact]
    leftover = total - sum(allocation)
    if leftover > 0:
        # largest fractional part first, ties to the longer video
        order = sorted(
            range(n_videos),
            key=lambda i: (exact[i] - allocation[i], counts[i]),
            reverse=True,
        )
        for i in order[:leftover]:
            allocation[i] += 1

    return _fit_to_capacity(allocation, counts, total)


def _is_sidecar(path) -> bool:
    """Filesystem metadata that merely looks like a media file.

    macOS writes a "._name.mp4" companion beside every file it copies onto
    a non-Apple filesystem. It carries the same extension, so any *.mp4
    glob picks it up and OpenCV then reports it as an unreadable video.
    """
    name = Path(path).name
    return name.startswith("._") or name in (".DS_Store", "Thumbs.db", "desktop.ini")


def resolve_videos(patterns: Sequence[Path]) -> list[Path]:
    """Expands globs and drops anything that is not a file."""
    videos: list[Path] = []
    for pattern in patterns:
        if any(ch in str(pattern) for ch in "*?"):
            videos.extend(sorted(pattern.parent.glob(pattern.name)))
        else:
            videos.append(Path(pattern))
    return [v for v in videos if v.is_file() and not _is_sidecar(v)]


def plan_sampling(
    videos: Sequence[Path],
    n_frames: int,
    seed: int = 0,
    even: bool = False,
    uniform_random: bool = False,
    counts: Sequence[int] | None = None,
) -> list[VideoPlan]:
    """Works out what would be sampled, without touching a single frame.

    Cheap enough for the GUI to call on every spinbox change, so the user can
    see "600 frames: clip_01 -> 312, clip_02 -> 288" before committing --
    provided it passes ``counts``. Counting frames means opening every file,
    which is the expensive part and the part that does not change while a
    spinbox does, so the caller may pass counts it has already gathered.

    A clip that cannot be read gets a plan with ``readable=False`` and no
    frames, rather than bringing the whole run down.
    """
    if not videos:
        raise SamplingError("No videos matched")

    if counts is None:
        counts = [frame_count_or_zero(v) for v in videos]
    elif len(counts) != len(videos):
        raise SamplingError("Frame counts do not line up with the videos")
    counts = [max(0, int(c)) for c in counts]

    if not any(counts):
        raise SamplingError(
            "None of the videos could be read: "
            + ", ".join(v.name for v in list(videos)[:5])
        )

    allocation = allocate(counts, n_frames, even)
    rng = random.Random(seed)
    pick = uniform_indices if uniform_random else stratified_indices

    plans = []
    for video, available, requested in zip(videos, counts, allocation):
        indices = pick(available, requested, rng) if requested else []
        plans.append(
            VideoPlan(video, available, requested, indices, readable=available > 0)
        )
    return plans


def _merge_manifest(path: Path, manifest: dict, output: Path) -> dict:
    """Folds a new batch into whatever the folder already recorded.

    Sampling a second time into the same folder is an invited flow -- it is how
    an annotation corpus grows -- and overwriting the manifest orphaned every
    earlier frame. An orphaned frame falls back to having its source guessed
    from its name, which used to disagree with the recorded spelling, so one
    clip could end up on both sides of the train/validation split.

    Records whose image has since been deleted are dropped, so the manifest
    does not accumulate ghosts.
    """
    if not path.is_file():
        return manifest
    try:
        old = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return manifest  # unreadable or not JSON: the new batch stands alone

    by_name: dict[str, dict] = {}
    for record in old.get("frames", []):
        name = record.get("file_name")
        if name and (output / name).is_file():
            by_name[name] = record
    for record in manifest["frames"]:
        by_name[record["file_name"]] = record

    merged = dict(manifest)
    merged["frames"] = sorted(by_name.values(), key=lambda r: r["file_name"])

    old_batches = old.get("batches")
    if not old_batches:
        # a manifest written before batches were recorded: keep what it said
        # about itself so the earlier frames' provenance is not simply lost
        old_batches = [
            {
                key: old[key]
                for key in ("created", "seed", "sampling", "allocation",
                            "complete", "enhancement", "videos")
                if key in old
            }
        ]
    merged["batches"] = old_batches + manifest["batches"]

    videos = list(dict.fromkeys(old.get("videos", []) + manifest["videos"]))
    merged["videos"] = videos
    return merged


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
    counts: Sequence[int] | None = None,
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
    check_distinct_stems(videos)
    plans = plan_sampling(videos, n_frames, seed, even, uniform_random, counts)
    output.mkdir(parents=True, exist_ok=True)
    enhance = make_enhancer_from(settings)

    records: list[dict] = []
    # Every video gets an entry up front. An aborted run then reports zero for
    # the clips it never reached, instead of omitting them and reading as
    # though the collection were smaller than it is.
    per_video: dict[str, int] = {plan.video.name: 0 for plan in plans}
    unreadable: list[str] = [p.video.name for p in plans if not p.readable]
    complete = True

    for plan in plans:
        if plan.requested == 0:
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
                    # the stem, matching how the frames are named and how the
                    # dataset step derives a source from a file name when no
                    # manifest covers it; storing the name with its extension
                    # made one clip look like two different recordings
                    "source_video": plan.video.stem,
                    "source_path": str(plan.video),
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

    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    batch = {
        "created": created,
        "seed": seed,
        "sampling": "uniform-random" if uniform_random else "stratified",
        "allocation": "even" if even else "proportional",
        "complete": complete,
        "enhancement": settings,
        "videos": [str(v) for v in videos],
    }
    for record in records:
        record["batch"] = created

    manifest = {
        "created": created,
        "seed": seed,
        "sampling": batch["sampling"],
        "allocation": batch["allocation"],
        "complete": complete,
        # kept at the top level because read_enhancement() looks for it here
        "enhancement": settings,
        "videos": batch["videos"],
        "batches": [batch],
        "frames": records,
    }
    result.manifest_path = output / "sampling_manifest.json"
    manifest = _merge_manifest(result.manifest_path, manifest, output)
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
