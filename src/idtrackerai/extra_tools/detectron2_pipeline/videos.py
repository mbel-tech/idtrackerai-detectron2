"""Checks that a set of videos is readable and consistent before a long export.

Worth running before committing hours of GPU time. A clip that OpenCV cannot
open, or whose frame count reads as zero, fails at the moment it is reached
rather than at the start, which on a 40-hour batch means losing a session to
something that took one second to detect.

Frame counts come from ``cv2.CAP_PROP_FRAME_COUNT``, the same call idtracker.ai
uses in ``Session.get_processing_episodes``. That is deliberate: the contour
file records the frame count, and idtracker.ai refuses a sidecar whose count
disagrees with the video. The two agree as long as both read the *same file*, so
do not re-encode or re-copy a clip between exporting its contours and tracking
it.

Usage
-----
    python check_videos.py --videos clips/*.mp4
    python check_videos.py --videos clips/*.mp4 --contours contours/ --fps 10
"""

import argparse
from collections import Counter
from pathlib import Path

import cv2


def probe(path: Path) -> dict:
    """Reads a video's metadata, and confirms a frame can actually be decoded."""
    info = {
        "path": path,
        "name": path.name,
        "size_mb": path.stat().st_size / 1e6,
        "opened": False,
        "frames": 0,
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "readable": False,
        "problems": [],
    }

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        info["problems"].append("OpenCV cannot open it")
        cap.release()
        return info

    info["opened"] = True
    info["frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    info["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    info["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    info["fps"] = float(cap.get(cv2.CAP_PROP_FPS))

    # Metadata can be present while the stream is unreadable, so decode one frame.
    ok, _ = cap.read()
    info["readable"] = bool(ok)
    if not ok:
        info["problems"].append("metadata reads but no frame decodes")

    cap.release()

    if info["frames"] <= 0:
        info["problems"].append(
            "frame count is 0 — idtracker.ai rejects this too, the file is"
            " probably truncated or its index is broken"
        )
    if info["width"] <= 0 or info["height"] <= 0:
        info["problems"].append("no frame size reported")
    return info


def main():
    parser = argparse.ArgumentParser(
        description="Verify videos before a long inference run",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--videos", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--contours",
        type=Path,
        help="folder of already-exported .h5 files, to show what is left",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="assumed inference throughput, for the time estimate",
    )
    args = parser.parse_args()

    videos: list[Path] = []
    for pattern in args.videos:
        if any(ch in str(pattern) for ch in "*?"):
            videos.extend(sorted(pattern.parent.glob(pattern.name)))
        else:
            videos.append(pattern)
    videos = [v for v in videos if v.is_file()]

    if not videos:
        raise SystemExit(
            f"No videos matched {[str(p) for p in args.videos]}.\n"
            "On Colab, check the Drive mount and that the path is the real\n"
            "folder: a 'Shared with me' folder is not reachable by path until\n"
            "you add a shortcut to it in My Drive."
        )

    print(f"{len(videos)} video(s)\n")
    header = f"{'file':34s} {'frames':>8s} {'size':>9s} {'fps':>6s}  {'dims':>11s}  status"
    print(header)
    print("-" * len(header))

    results = [probe(v) for v in videos]
    for info in results:
        status = "ok"
        if info["problems"]:
            status = "; ".join(info["problems"])
        elif args.contours and (args.contours / f"{info['path'].stem}.h5").exists():
            status = "done"
        print(
            f"{info['name'][:34]:34s} {info['frames']:8d} {info['size_mb']:8.0f}M"
            f" {info['fps']:6.1f}  {info['width']:5d}x{info['height']:<5d}  {status}"
        )

    broken = [i for i in results if i["problems"]]
    good = [i for i in results if not i["problems"]]

    sizes = Counter((i["width"], i["height"]) for i in good)
    names = Counter(i["name"] for i in results)
    duplicates = [n for n, c in names.items() if c > 1]

    total_frames = sum(i["frames"] for i in good)
    total_gb = sum(i["size_mb"] for i in results) / 1000

    print(f"\n{len(good)} usable, {len(broken)} with problems")
    print(f"{total_frames:,} frames, {total_gb:.1f} GB on disk")

    if len(sizes) > 1:
        print(f"\nWARNING: mixed resolutions {dict(sizes)}")
        print(
            "  Each contour file is checked against its own video, so mixed"
            " sizes are\n  fine — but a model trained at one scale may do worse"
            " at another."
        )
    if duplicates:
        print(f"\nWARNING: repeated file names {duplicates}")
        print("  Contour files are named after the video stem, so these collide.")

    if args.contours:
        done = sum(
            1 for i in good if (args.contours / f"{i['path'].stem}.h5").exists()
        )
        remaining = [
            i for i in good if not (args.contours / f"{i['path'].stem}.h5").exists()
        ]
        left_frames = sum(i["frames"] for i in remaining)
        print(f"\n{done}/{len(good)} already exported, {len(remaining)} to go")
        if left_frames:
            print(
                f"  {left_frames:,} frames ~ {left_frames / args.fps / 3600:.1f} h"
                f" at {args.fps:g} fps"
            )
    elif total_frames:
        print(
            f"  ~ {total_frames / args.fps / 3600:.1f} h of inference at"
            f" {args.fps:g} fps"
        )

    if broken:
        print("\nProblems:")
        for info in broken:
            print(f"  {info['name']}: {'; '.join(info['problems'])}")
        raise SystemExit(1)

    print("\nAll videos readable.")


if __name__ == "__main__":
    main()
