"""Turning LabelMe polygons into a COCO instance-segmentation dataset.

Annotations are validated rather than trusted. LabelMe will happily save a
two-point "polygon", a shape drawn outside the image, or a label typo that
silently becomes a second class, and each of those either crashes training or
quietly degrades it. Everything found is reported rather than raised, so the
caller — a terminal or a GUI panel — can show the whole picture at once.

The split defaults to grouping by source video. Frames sampled from one clip a
few seconds apart show the same animals in nearly the same poses, so a random
split puts near-duplicates in both train and validation, and the validation AP
then measures memorisation rather than generalisation.
"""

import json
import random
import shutil
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    from .errors import DatasetError
except ImportError:  # loaded by path, without the package around it
    from errors import DatasetError  # type: ignore[no-redef]

# Sidecars written by the sampling step, which are not annotations.
RESERVED_JSON = {
    "sampling_manifest.json",
    "preprocess_profile.json",
    "report.json",
}


@dataclass
class DatasetRequest:
    input_dir: Path
    output_dir: Path
    val_fraction: float = 0.15
    split_by: str = "video"
    seed: int = 0
    labels: Sequence[str] | None = None
    single_class: str | None = None
    expected_instances: int | None = None
    min_area: float = 10.0
    copy_images: bool = True


@dataclass
class DatasetReport:
    categories: list[dict] = field(default_factory=list)
    label_counts: dict[str, int] = field(default_factory=dict)
    train_images: int = 0
    train_annotations: int = 0
    train_videos: list[str] = field(default_factory=list)
    val_images: int = 0
    val_annotations: int = 0
    val_videos: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    skipped_json: list[str] = field(default_factory=list)
    enhancement: dict | None = None
    output_dir: Path | None = None

    def as_dict(self) -> dict:
        return {
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "enhancement": self.enhancement,
            "categories": self.categories,
            "label_counts": self.label_counts,
            "train": {
                "images": self.train_images,
                "annotations": self.train_annotations,
                "videos": self.train_videos,
            },
            "val": {
                "images": self.val_images,
                "annotations": self.val_annotations,
                "videos": self.val_videos,
            },
            "problems": self.problems,
            "notes": self.notes,
        }


def polygon_area(points: np.ndarray) -> float:
    """Shoelace area. Not cv2.contourArea, so this works without OpenCV."""
    x, y = points[:, 0], points[:, 1]
    return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) / 2)


def is_labelme(path: Path) -> bool:
    """Recognises an annotation by shape, not by filename.

    Blacklisting names would break the moment another sidecar is added, so a
    file counts as an annotation when it parses as an object with "shapes". A
    file that will not parse counts as one too, so the reader reports it as a
    broken annotation rather than silently ignoring it.
    """
    if path.name in RESERVED_JSON:
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return True
    return isinstance(data, dict) and "shapes" in data


def count_annotated(folder: Path) -> int:
    """How many annotations sit in a folder. Used by the GUI's progress display."""
    return sum(1 for p in folder.glob("*.json") if is_labelme(p))


def source_video_of(file_name: str, manifest: dict | None) -> str:
    """Which clip a frame came from, for the group-aware split."""
    if manifest:
        entry = manifest.get(file_name)
        if entry:
            return entry["source_video"]
    # fall back to the naming the sampling step uses: <video stem>_f<number>.png
    stem = Path(file_name).stem
    if "_f" in stem:
        return stem.rsplit("_f", 1)[0]
    return stem


def read_enhancement(input_dir: Path) -> dict | None:
    """The enhancement settings the frames were prepared with.

    Training records these with the weights so inference can reproduce exactly
    the images the model saw. If they do not make it this far the chain breaks
    silently: training records null, inference falls back to built-in defaults,
    and the mismatch check has nothing to compare against so it cannot fire.
    """
    profile = input_dir / "preprocess_profile.json"
    if profile.is_file():
        return {
            k: v
            for k, v in json.loads(profile.read_text(encoding="utf-8")).items()
            if k not in ("name", "notes")
        }
    manifest = input_dir / "sampling_manifest.json"
    if manifest.is_file():
        return json.loads(manifest.read_text(encoding="utf-8")).get("enhancement")
    return None


def build_coco_dataset(
    request: DatasetRequest,
    progress: Callable[[int], None] | None = None,
    abort: Callable[[], bool] | None = None,
) -> DatasetReport:
    """Validates, converts and splits. Everything found lands in the report."""
    report = DatasetReport(output_dir=request.output_dir)
    all_json = sorted(request.input_dir.glob("*.json"))
    json_files = [p for p in all_json if is_labelme(p)]
    report.skipped_json = [p.name for p in all_json if not is_labelme(p)]

    if not json_files:
        raise DatasetError(f"No LabelMe .json files in {request.input_dir}")

    manifest = None
    manifest_path = request.input_dir / "sampling_manifest.json"
    if manifest_path.is_file():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = {f["file_name"]: f for f in data["frames"]}

    report.enhancement = read_enhancement(request.input_dir)

    label_counter: Counter[str] = Counter()
    per_image: list[dict] = []

    # ------------------------------------------------------ read and validate
    for done, json_path in enumerate(json_files, start=1):
        if abort is not None and abort():
            raise DatasetError("Cancelled")
        if progress is not None:
            progress(done)

        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            report.problems.append(f"{json_path.name}: not valid JSON ({exc})")
            continue

        image_name = data.get("imagePath") or (json_path.stem + ".png")
        image_path = request.input_dir / Path(image_name).name
        if not image_path.is_file():
            report.problems.append(
                f"{json_path.name}: image {Path(image_name).name} not found beside it"
            )
            continue

        width = data.get("imageWidth")
        height = data.get("imageHeight")
        if not width or not height:
            report.problems.append(f"{json_path.name}: missing imageWidth/imageHeight")
            continue

        shapes = []
        for i, shape in enumerate(data.get("shapes", [])):
            label = shape.get("label", "")
            kind = shape.get("shape_type", "polygon")
            points = np.asarray(shape.get("points", []), dtype=float)

            if kind != "polygon":
                report.problems.append(
                    f"{json_path.name} shape {i}: shape_type is {kind!r}, expected"
                    " 'polygon'. Instance segmentation needs outlines, not boxes."
                )
                continue
            if len(points) < 3:
                report.problems.append(
                    f"{json_path.name} shape {i} ({label}): only {len(points)}"
                    " point(s), a polygon needs 3"
                )
                continue

            area = polygon_area(points)
            if area < request.min_area:
                report.problems.append(
                    f"{json_path.name} shape {i} ({label}): area {area:.1f} below"
                    f" the minimum {request.min_area}, probably a stray click"
                )
                continue

            outside = (
                (points[:, 0] < 0).any()
                or (points[:, 1] < 0).any()
                or (points[:, 0] > width).any()
                or (points[:, 1] > height).any()
            )
            if outside:
                report.notes.append(
                    f"{json_path.name} shape {i} ({label}): extends past the image"
                    " border, clipped"
                )
                points[:, 0] = points[:, 0].clip(0, width)
                points[:, 1] = points[:, 1].clip(0, height)

            label_counter[label] += 1
            shapes.append({"label": label, "points": points, "area": area})

        if not shapes:
            report.problems.append(
                f"{json_path.name}: no usable polygons, image excluded"
            )
            continue

        if request.expected_instances and len(shapes) != request.expected_instances:
            report.notes.append(
                f"{json_path.name}: {len(shapes)} instance(s), expected"
                f" {request.expected_instances}"
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

    report.label_counts = dict(label_counter)

    if not per_image:
        raise DatasetError(
            "No usable annotations found.\n  " + "\n  ".join(report.problems[:20])
        )

    # ------------------------------------------------------------- categories
    if request.single_class:
        categories = [{"id": 1, "name": request.single_class}]

        def category_of(_label):
            return 1

    else:
        names = list(request.labels or sorted(label_counter))
        unknown = set(label_counter) - set(names)
        if unknown:
            report.problems.append(
                "labels present but not requested, their shapes are dropped:"
                f" {sorted(unknown)}"
            )
        categories = [{"id": i + 1, "name": n} for i, n in enumerate(names)]
        ids = {n: i + 1 for i, n in enumerate(names)}
        category_of = ids.get

    if len(label_counter) > 1 and not request.single_class and not request.labels:
        report.notes.append(
            f"{len(label_counter)} distinct labels found: {dict(label_counter)}."
            " If these are meant to be one class, set a single class name."
        )
    report.categories = categories

    # ------------------------------------------------------------------ split
    train_images, val_images = _split(per_image, request, report)

    # ------------------------------------------------------------------ write
    request.output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = request.output_dir / "images"
    if request.copy_images:
        images_dir.mkdir(exist_ok=True)

    train = _build_coco(train_images, "train", categories, category_of,
                        images_dir if request.copy_images else None)
    val = _build_coco(val_images, "val", categories, category_of,
                      images_dir if request.copy_images else None)

    (request.output_dir / "train.json").write_text(json.dumps(train), encoding="utf-8")
    (request.output_dir / "val.json").write_text(json.dumps(val), encoding="utf-8")

    report.train_images = len(train["images"])
    report.train_annotations = len(train["annotations"])
    report.train_videos = sorted({i["source_video"] for i in train_images})
    report.val_images = len(val["images"])
    report.val_annotations = len(val["annotations"])
    report.val_videos = sorted({i["source_video"] for i in val_images})

    (request.output_dir / "report.json").write_text(
        json.dumps(report.as_dict(), indent=2), encoding="utf-8"
    )

    # Copy the profile itself too, so the settings travel when only the dataset
    # folder is uploaded to Colab.
    profile = request.input_dir / "preprocess_profile.json"
    if profile.is_file():
        shutil.copy2(profile, request.output_dir / "preprocess_profile.json")
    elif report.enhancement is None:
        report.problems.append(
            "No enhancement record found beside the annotations. Training will"
            " not know how these frames were prepared, and inference will fall"
            " back to the built-in defaults without warning."
        )

    if not report.val_images:
        report.problems.append(
            "The validation set is empty, so training cannot be evaluated."
        )
    return report


def _split(per_image, request: DatasetRequest, report: DatasetReport):
    rng = random.Random(request.seed)

    if request.split_by != "video":
        report.notes.append(
            "Random split requested. If these frames come from the same clips,"
            " train and validation will contain near-duplicate images and the"
            " validation AP will overstate real performance."
        )
        return _random_split(per_image, request.val_fraction, rng)

    by_video: dict[str, list[dict]] = defaultdict(list)
    for image in per_image:
        by_video[image["source_video"]].append(image)

    videos = sorted(by_video)
    rng.shuffle(videos)
    if len(videos) < 2:
        report.notes.append(
            f"All frames come from one source ({videos[0]}), so a by-video split"
            " is impossible; falling back to a random split. The validation"
            " score will be optimistic because neighbouring frames are"
            " near-duplicates."
        )
        return _random_split(per_image, request.val_fraction, rng)

    target = len(per_image) * request.val_fraction
    val_images: list[dict] = []
    train_images: list[dict] = []
    taken = 0
    for video in videos:
        group = by_video[video]
        if taken < target and len(val_images) + len(group) <= len(per_image) - 1:
            val_images += group
            taken += len(group)
        else:
            train_images += group
    return train_images, val_images


def _random_split(per_image, val_fraction: float, rng: random.Random):
    images = list(per_image)
    rng.shuffle(images)
    cut = max(1, round(len(images) * val_fraction))
    return images[cut:], images[:cut]


def _build_coco(subset, name, categories, category_of, images_dir: Path | None) -> dict:
    coco: dict = {
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

        if images_dir is not None:
            target = images_dir / image["file_name"]
            if not target.exists():
                shutil.copy2(image["path"], target)
    return coco
