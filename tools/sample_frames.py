"""Extracts frames from videos for annotation in LabelMe.

Frames are written already enhanced, with the same function used at inference,
so what gets annotated is what the model will see. Annotating raw frames and
predicting on enhanced ones (or the reverse) is a quiet source of poor
detections that looks like a training problem.

Sampling is stratified by default: each video is cut into as many equal blocks
as there are frames to take, and one frame is drawn from each block. Uniform
random sampling over a 30000-frame clip clumps, leaving stretches of the video
unrepresented; stratifying guarantees coverage of the whole recording, including
whatever the lighting does halfway through.

A manifest records which video and frame number every image came from. That
provenance is what lets the COCO conversion split train and validation by video
instead of at random, which matters: frames a second apart are near-duplicates,
and randomly splitting them puts the same fish, in the same pose, on both sides
of the split. The resulting evaluation score measures memorisation.

Usage
-----
    python sample_frames.py --videos clips/*.mp4 --n-frames 600 --output annotate/

Then open the output folder in LabelMe, draw polygons around each animal with a
single consistent label, and save. LabelMe writes a .json next to each image.
"""

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2

# Make the sibling modules importable however this file is loaded: as a
# script, from another directory, or via importlib from a test.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import frame_preprocessing as fp


def video_frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open {path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return count


def stratified_indices(n_frames: int, n_take: int, rng: random.Random) -> list[int]:
    """One frame from each of n_take equal blocks, at a random point within it."""
    if n_take >= n_frames:
        return list(range(n_frames))
    edges = [round(i * n_frames / n_take) for i in range(n_take + 1)]
    return [rng.randrange(edges[i], max(edges[i + 1], edges[i] + 1)) for i in range(n_take)]


def uniform_indices(n_frames: int, n_take: int, rng: random.Random) -> list[int]:
    if n_take >= n_frames:
        return list(range(n_frames))
    return sorted(rng.sample(range(n_frames), n_take))


def allocate(counts: list[int], total: int, even: bool) -> list[int]:
    """Decides how many frames to take from each video."""
    n_videos = len(counts)
    if even:
        base, extra = divmod(total, n_videos)
        return [base + (1 if i < extra else 0) for i in range(n_videos)]

    grand_total = sum(counts)
    if grand_total == 0:
        raise SystemExit("The videos contain no frames")
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


def main():
    parser = argparse.ArgumentParser(
        description="Sample frames from videos for LabelMe annotation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--videos", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--n-frames", type=int, required=True, help="total frames across all videos"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--even",
        action="store_true",
        help="take the same number from each video instead of proportionally",
    )
    parser.add_argument(
        "--uniform-random",
        action="store_true",
        help="sample uniformly at random instead of one frame per time block",
    )
    parser.add_argument(
        "--format", choices=("png", "jpg"), default="png",
        help="png is lossless; jpg artefacts can shift annotated boundaries",
    )
    fp.add_arguments(parser)
    args = parser.parse_args()

    videos = []
    for pattern in args.videos:
        matches = sorted(pattern.parent.glob(pattern.name)) if "*" in str(pattern) else [pattern]
        videos.extend(m for m in matches if m.is_file())
    if not videos:
        raise SystemExit(f"No videos matched {args.videos}")

    args.output.mkdir(parents=True, exist_ok=True)

    settings, notes = fp.resolve_settings(args)
    for note in notes:
        print(note)
    print(f"enhancement: {fp.describe(settings)}")
    enhance = fp.make_enhancer_from(settings)
    rng = random.Random(args.seed)
    pick = uniform_indices if args.uniform_random else stratified_indices

    counts = [video_frame_count(v) for v in videos]
    allocation = allocate(counts, args.n_frames, args.even)

    print(f"{len(videos)} video(s), {sum(counts)} frames total")
    records = []

    for video, n_frames, n_take in zip(videos, counts, allocation):
        if n_take == 0:
            print(f"  {video.name}: skipped (allocation 0)")
            continue

        indices = pick(n_frames, n_take, rng)
        cap = cv2.VideoCapture(str(video))
        written = 0
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                print(f"  {video.name}: could not read frame {index}, skipping")
                continue

            image = enhance(frame)
            name = f"{video.stem}_f{index:06d}.{args.format}"
            # imencode+tofile rather than imwrite, so non-ASCII paths work
            cv2.imencode(f".{args.format}", image)[1].tofile(str(args.output / name))
            records.append(
                {
                    "file_name": name,
                    "source_video": video.name,
                    "frame_number": index,
                    "width": image.shape[1],
                    "height": image.shape[0],
                }
            )
            written += 1
        cap.release()
        print(f"  {video.name}: {written}/{n_take} frames ({n_frames} available)")

    manifest = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": args.seed,
        "sampling": "uniform-random" if args.uniform_random else "stratified",
        "allocation": "even" if args.even else "proportional",
        "enhancement": settings,
        "videos": [str(v) for v in videos],
        "frames": records,
    }
    manifest_path = args.output / "sampling_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # The settings live beside the frames as a reusable profile, so the same
    # recording setup can be prepared identically next time without retyping
    # flags, and so the choice is reviewable and version controllable.
    profile_path = fp.save_profile(
        args.output / "preprocess_profile.json",
        settings,
        name=args.output.name,
        notes="Written by sample_frames.py; reuse with --preprocess-profile",
    )

    print(f"\n{len(records)} frames written to {args.output}")
    print(f"Manifest: {manifest_path}")
    print(f"Profile:  {profile_path}")
    print(
        "\nNext: open this folder in LabelMe, draw one polygon per animal using a\n"
        "single consistent label, then run labelme_to_coco.py on the folder."
    )


if __name__ == "__main__":
    main()
