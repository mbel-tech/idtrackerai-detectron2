"""Extracts frames from videos for annotation in LabelMe.

Command-line front end. The logic lives in
``idtrackerai.extra_tools.detectron2_pipeline.sampling`` so that the
Segmentation App runs exactly the same code.

Frames are written already enhanced, with the same settings inference will use,
so what gets annotated is what the model will see. Annotating raw frames and
predicting on enhanced ones (or the reverse) is a quiet source of poor
detections that looks like a training problem.

Usage
-----
    python sample_frames.py --videos clips/*.mp4 --n-frames 600 --output annotate/

Then open the output folder in LabelMe, draw polygons around each animal with a
single consistent label, and save. LabelMe writes a .json next to each image.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import frame_preprocessing as fp  # noqa: E402
from _pipeline import module  # noqa: E402

sampling = module("sampling")
errors = module("errors")


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
        "--format",
        choices=("png", "jpg"),
        default="png",
        help="png is lossless; jpg artefacts can shift annotated boundaries",
    )
    fp.add_arguments(parser)
    args = parser.parse_args()

    try:
        videos = sampling.resolve_videos(args.videos)
        if not videos:
            raise errors.SamplingError(f"No videos matched {args.videos}")

        settings, notes = fp.resolve_settings(args)
        for note in notes:
            print(note)
        print(f"enhancement: {fp.describe(settings)}")

        plans = sampling.plan_sampling(
            videos, args.n_frames, args.seed, args.even, args.uniform_random
        )
        total_available = sum(p.available for p in plans)
        print(f"\n{len(videos)} video(s), {total_available} frames total")
        for plan in plans:
            print(
                f"  {plan.video.name}: taking {plan.requested}"
                f" of {plan.available} frames"
            )

        last = [0]

        def report(written: int) -> None:
            if written - last[0] >= 25 or written == args.n_frames:
                last[0] = written
                print(f"    {written}/{args.n_frames} written", flush=True)

        result = sampling.sample_frames(
            videos,
            args.output,
            args.n_frames,
            settings,
            seed=args.seed,
            even=args.even,
            uniform_random=args.uniform_random,
            image_format=args.format,
            progress=report,
        )
    except errors.PipelineError as exc:
        raise SystemExit(str(exc))

    if result.unreadable:
        print(f"\n{len(result.unreadable)} frame(s) could not be read:")
        for item in result.unreadable[:10]:
            print(f"  {item}")

    print(f"\n{result.written} frames written to {result.output}")
    if result.manifest_path:
        print(f"Manifest: {result.manifest_path}")
    if result.profile_path:
        print(f"Profile:  {result.profile_path}")
    print(
        "\nNext: open this folder in LabelMe, draw one polygon per animal using a\n"
        "single consistent label, then run labelme_to_coco.py on the folder."
    )


if __name__ == "__main__":
    main()
