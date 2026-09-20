"""Converts LabelMe polygon annotations into COCO instance segmentation JSON,
split into training and validation sets for Detectron2.

Annotations are validated rather than trusted. LabelMe will happily save a
two-point "polygon", a shape drawn outside the image, or a label typo that
silently becomes a second class, and each of those either crashes training or
quietly degrades it.

The split defaults to grouping by source video. Frames sampled from one clip a
few seconds apart show the same animals in nearly the same poses, so a random
split puts near-duplicates in both train and validation, and the validation AP
then measures memorisation rather than generalisation. Splitting by video keeps
the two sets genuinely separate. Use ``--split-by random`` only when the frames
really are independent.

Usage
-----
    python labelme_to_coco.py --input annotate/ --output dataset/ --val-fraction 0.15

Produces::

    dataset/train.json      COCO annotations for training
    dataset/val.json        COCO annotations for validation
    dataset/images/         the annotated images, copied
    dataset/report.json     validation findings and dataset statistics
"""

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - cv2 is only needed for area/bbox
    cv2 = None


def polygon_area(points: np.ndarray) -> float:
    """Shoelace area. Not cv2.contourArea, so this works without OpenCV."""
    x, y = points[:, 0], points[:, 1]
    return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2)


def load_labelme(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def source_video_of(file_name: str, manifest: dict | None) -> str:
    """Which clip a frame came from, for the group-aware split."""
    if manifest:
        entry = manifest.get(file_name)
        if entry:
            return entry["source_video"]
    # fall back to the naming sample_frames.py uses: <video stem>_f<number>.png
    stem = Path(file_name).stem
    if "_f" in stem:
        return stem.rsplit("_f", 1)[0]
    return stem


def main():
    parser = argparse.ArgumentParser(
        description="LabelMe -> COCO for Detectron2 instance segmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True, help="folder of images + LabelMe .json")
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

    json_files = sorted(
        p for p in args.input.glob("*.json") if p.name != "sampling_manifest.json"
    )
    if not json_files:
        raise SystemExit(f"No LabelMe .json files in {args.input}")

    manifest = None
    manifest_path = args.input / "sampling_manifest.json"
    if manifest_path.is_file():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = {f["file_name"]: f for f in data["frames"]}
        print(f"Using provenance from {manifest_path.name}")

    problems: list[str] = []
    notes: list[str] = []
    label_counter: Counter[str] = Counter()
    per_image: list[dict] = []

    # ------------------------------------------------------- read and validate
    for json_path in json_files:
        try:
            data = load_labelme(json_path)
        except json.JSONDecodeError as exc:
            problems.append(f"{json_path.name}: not valid JSON ({exc})")
            continue

        image_name = data.get("imagePath") or (json_path.stem + ".png")
        image_path = args.input / Path(image_name).name
        if not image_path.is_file():
            problems.append(
                f"{json_path.name}: image {Path(image_name).name} not found beside it"
            )
            continue

        width = data.get("imageWidth")
        height = data.get("imageHeight")
        if not width or not height:
            problems.append(f"{json_path.name}: missing imageWidth/imageHeight")
            continue

        shapes = []
        for i, shape in enumerate(data.get("shapes", [])):
            label = shape.get("label", "")
            kind = shape.get("shape_type", "polygon")
            points = np.asarray(shape.get("points", []), dtype=float)

            if kind != "polygon":
                problems.append(
                    f"{json_path.name} shape {i}: shape_type is {kind!r}, expected"
                    " 'polygon'. Instance segmentation needs outlines, not boxes."
                )
                continue
            if len(points) < 3:
                problems.append(
                    f"{json_path.name} shape {i} ({label}): only {len(points)}"
                    " point(s), a polygon needs 3"
                )
                continue

            area = polygon_area(points)
            if area < args.min_area:
                problems.append(
                    f"{json_path.name} shape {i} ({label}): area {area:.1f} below"
                    f" --min-area {args.min_area}, probably a stray click"
                )
                continue

            outside = (
                (points[:, 0] < 0).any()
                or (points[:, 1] < 0).any()
                or (points[:, 0] > width).any()
                or (points[:, 1] > height).any()
            )
            if outside:
                notes.append(
                    f"{json_path.name} shape {i} ({label}): extends past the image"
                    " border, clipped"
                )
                points[:, 0] = points[:, 0].clip(0, width)
                points[:, 1] = points[:, 1].clip(0, height)

            label_counter[label] += 1
            shapes.append({"label": label, "points": points, "area": area})

        if not shapes:
            problems.append(f"{json_path.name}: no usable polygons, image excluded")
            continue

        if args.expected_instances and len(shapes) != args.expected_instances:
            notes.append(
                f"{json_path.name}: {len(shapes)} instance(s), expected"
                f" {args.expected_instances}"
            )

        per_image.append(
            {
                "file_name": Path(image_name).name,
                "path": image_path,
                "width": int(width),
                "height": int(height),
                "shapes": shapes,
                "source_video": source_video_of(Path(image_name).name, manifest),
            }
        )

    if not per_image:
        raise SystemExit(
            "No usable annotations found.\n  " + "\n  ".join(problems[:20])
        )

    # ---------------------------------------------------------------- classes
    if args.single_class:
        categories = [{"id": 1, "name": args.single_class}]
        category_of = lambda label: 1  # noqa: E731
    else:
        names = args.labels or sorted(label_counter)
        unknown = set(label_counter) - set(names)
        if unknown:
            problems.append(
                f"labels present but not in --labels, their shapes are dropped:"
                f" {sorted(unknown)}"
            )
        categories = [{"id": i + 1, "name": n} for i, n in enumerate(names)]
        ids = {n: i + 1 for i, n in enumerate(names)}
        category_of = ids.get

    if len(label_counter) > 1 and not args.single_class and not args.labels:
        notes.append(
            f"{len(label_counter)} distinct labels found: {dict(label_counter)}."
            " If these are meant to be one class, pass --single-class."
        )

    # ------------------------------------------------------------------ split
    rng = random.Random(args.seed)
    if args.split_by == "video":
        by_video: dict[str, list[dict]] = defaultdict(list)
        for image in per_image:
            by_video[image["source_video"]].append(image)

        videos = sorted(by_video)
        rng.shuffle(videos)
        if len(videos) < 2:
            notes.append(
                f"All frames come from one source ({videos[0]}), so a by-video split"
                " is impossible; falling back to a random split. The validation"
                " score will be optimistic because neighbouring frames are"
                " near-duplicates."
            )
            images = list(per_image)
            rng.shuffle(images)
            cut = max(1, round(len(images) * args.val_fraction))
            val_images, train_images = images[:cut], images[cut:]
        else:
            target = len(per_image) * args.val_fraction
            val_images, train_images, taken = [], [], 0
            for video in videos:
                group = by_video[video]
                if taken < target and len(val_images) + len(group) <= len(per_image) - 1:
                    val_images += group
                    taken += len(group)
                else:
                    train_images += group
    else:
        notes.append(
            "Random split requested. If these frames come from the same clips,"
            " train and validation will contain near-duplicate images and the"
            " validation AP will overstate real performance."
        )
        images = list(per_image)
        rng.shuffle(images)
        cut = max(1, round(len(images) * args.val_fraction))
        val_images, train_images = images[:cut], images[cut:]

    # ------------------------------------------------------------------ write
    args.output.mkdir(parents=True, exist_ok=True)
    images_dir = args.output / "images"
    if not args.no_copy_images:
        images_dir.mkdir(exist_ok=True)

    def build_coco(subset: list[dict], name: str) -> dict:
        coco = {
            "info": {
                "description": f"{name} set from LabelMe annotations",
                "date_created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": categories,
        }
        annotation_id = 1
        for image_id, image in enumerate(subset, start=1):
            coco["images"].append(
                {
                    "id": image_id,
                    "file_name": image["file_name"],
                    "width": image["width"],
                    "height": image["height"],
                }
            )
            for shape in image["shapes"]:
                category_id = category_of(shape["label"])
                if category_id is None:
                    continue
                points = shape["points"]
                x0, y0 = points.min(0)
                x1, y1 = points.max(0)
                coco["annotations"].append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "segmentation": [points.flatten().round(2).tolist()],
                        "area": round(shape["area"], 2),
                        "bbox": [
                            round(float(x0), 2),
                            round(float(y0), 2),
                            round(float(x1 - x0), 2),
                            round(float(y1 - y0), 2),
                        ],
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1

            if not args.no_copy_images:
                target = images_dir / image["file_name"]
                if not target.exists():
                    shutil.copy2(image["path"], target)
        return coco

    train = build_coco(train_images, "train")
    val = build_coco(val_images, "val")
    (args.output / "train.json").write_text(json.dumps(train), encoding="utf-8")
    (args.output / "val.json").write_text(json.dumps(val), encoding="utf-8")

    report = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input": str(args.input),
        "split_by": args.split_by,
        "categories": categories,
        "label_counts": dict(label_counter),
        "train": {
            "images": len(train["images"]),
            "annotations": len(train["annotations"]),
            "videos": sorted({i["source_video"] for i in train_images}),
        },
        "val": {
            "images": len(val["images"]),
            "annotations": len(val["annotations"]),
            "videos": sorted({i["source_video"] for i in val_images}),
        },
        "problems": problems,
        "notes": notes,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # ----------------------------------------------------------------- report
    print(f"\nCategories: {[c['name'] for c in categories]}  (NUM_CLASSES = {len(categories)})")
    print(f"Train: {len(train['images']):4d} images, {len(train['annotations']):5d} instances")
    print(f"Val:   {len(val['images']):4d} images, {len(val['annotations']):5d} instances")
    if args.split_by == "video":
        print(f"  train videos: {sorted({i['source_video'] for i in train_images})}")
        print(f"  val videos:   {sorted({i['source_video'] for i in val_images})}")

    if problems:
        print(f"\n{len(problems)} problem(s) — these shapes or images were dropped:")
        for line in problems[:15]:
            print(f"  {line}")
        if len(problems) > 15:
            print(f"  ... and {len(problems) - 15} more, see report.json")
    if notes:
        print(f"\n{len(notes)} note(s):")
        for line in notes[:10]:
            print(f"  {line}")
        if len(notes) > 10:
            print(f"  ... and {len(notes) - 10} more, see report.json")

    if not val["images"]:
        print("\nWARNING: the validation set is empty, so training cannot be evaluated.")

    print(f"\nWritten to {args.output}")
    print(f"Next: python train_detectron2.py --dataset {args.output} --output model/")


if __name__ == "__main__":
    main()
