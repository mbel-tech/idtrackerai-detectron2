"""Converts LabelMe polygon annotations into COCO instance segmentation JSON,
split into training and validation sets for Detectron2.

Command-line front end. The logic lives in
``idtrackerai.extra_tools.detectron2_pipeline.dataset`` so that the
Segmentation App runs exactly the same code.

Annotations are validated rather than trusted: two-point polygons, stray
clicks, bounding boxes drawn where a polygon was meant, shapes off the image
edge, and label typos that silently become a second class are all reported.

The split defaults to grouping by source video, because frames sampled from one
clip a few seconds apart are near-duplicates, and splitting those at random
makes the validation score measure memorisation.

Usage
-----
    python labelme_to_coco.py --input annotate/ --output dataset/ --val-fraction 0.15

Produces::

    dataset/train.json                COCO annotations for training
    dataset/val.json                  COCO annotations for validation
    dataset/images/                   the annotated images, copied
    dataset/report.json               validation findings and statistics
    dataset/preprocess_profile.json   the enhancement settings, for training
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _pipeline import module  # noqa: E402

dataset = module("dataset")
errors = module("errors")


def main():
    parser = argparse.ArgumentParser(
        description="LabelMe -> COCO for Detectron2 instance segmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input", type=Path, required=True, help="folder of images + LabelMe .json"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument(
        "--split-by",
        choices=("video", "random"),
        default="video",
        help="'video' keeps frames from one clip on one side of the split",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--labels",
        nargs="+",
        help="labels to keep, in category order. Default: every label found",
    )
    parser.add_argument(
        "--single-class",
        metavar="NAME",
        help="merge every label into one class with this name",
    )
    parser.add_argument(
        "--expected-instances",
        type=int,
        help="flag frames whose instance count differs from this (e.g. 5 fish)",
    )
    parser.add_argument(
        "--min-area", type=float, default=10.0, help="reject polygons smaller than this"
    )
    parser.add_argument(
        "--no-copy-images", action="store_true", help="reference images in place"
    )
    args = parser.parse_args()

    request = dataset.DatasetRequest(
        input_dir=args.input,
        output_dir=args.output,
        val_fraction=args.val_fraction,
        split_by=args.split_by,
        seed=args.seed,
        labels=args.labels,
        single_class=args.single_class,
        expected_instances=args.expected_instances,
        min_area=args.min_area,
        copy_images=not args.no_copy_images,
    )

    try:
        report = dataset.build_coco_dataset(request)
    except errors.PipelineError as exc:
        raise SystemExit(str(exc))

    if report.skipped_json:
        print(f"Ignoring non-annotation JSON: {', '.join(report.skipped_json)}")

    names = [c["name"] for c in report.categories]
    print(f"\nCategories: {names}  (NUM_CLASSES = {len(names)})")
    print(f"Train: {report.train_images:4d} images, {report.train_annotations:5d} instances")
    print(f"Val:   {report.val_images:4d} images, {report.val_annotations:5d} instances")
    if args.split_by == "video":
        print(f"  train videos: {report.train_videos}")
        print(f"  val videos:   {report.val_videos}")

    if report.enhancement:
        print(f"\nEnhancement carried forward: {report.enhancement}")
    else:
        print(
            "\nWARNING: no enhancement record found beside the annotations."
            "\nTraining will record null, so inference falls back to the"
            " built-in defaults\nand the mismatch check cannot fire."
        )

    if report.problems:
        print(f"\n{len(report.problems)} problem(s) — these were dropped:")
        for line in report.problems[:15]:
            print(f"  {line}")
        if len(report.problems) > 15:
            print(f"  ... and {len(report.problems) - 15} more, see report.json")
    if report.notes:
        print(f"\n{len(report.notes)} note(s):")
        for line in report.notes[:10]:
            print(f"  {line}")
        if len(report.notes) > 10:
            print(f"  ... and {len(report.notes) - 10} more, see report.json")

    print(f"\nWritten to {report.output_dir}")
    print(f"Next: python train_detectron2.py --dataset {report.output_dir} --output model/")


if __name__ == "__main__":
    main()
