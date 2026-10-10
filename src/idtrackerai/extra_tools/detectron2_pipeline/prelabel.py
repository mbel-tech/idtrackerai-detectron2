"""Draws a first draft of the annotations with SAM 3, for you to correct.

The slow part of training a detector is not the training, it is outlining a few
hundred animals by hand. SAM 3 can be prompted with a word and will find most
of them, so this step writes LabelMe files next to the sampled frames with the
polygons already in place. You then open them in LabelMe as usual and fix what
is wrong, which is much faster than drawing from nothing.

This sits between ``sampling`` and ``dataset``, and changes neither. The files
it writes are ordinary LabelMe annotations, so the dataset step reads them
without knowing they were machine-drawn.

What it will and will not do
----------------------------
It never overwrites an existing ``.json``. Once you have corrected a frame,
that file is yours; re-running to pick up newly sampled frames cannot destroy
the work. Pass ``--overwrite`` to deliberately redo them.

Every polygon it writes is a **draft**, not an annotation. SAM 3 has not seen
your setup, and on low-contrast animals it misses some and invents others. Each
shape is flagged ``sam3_prelabel`` so machine-drawn outlines stay
distinguishable from hand-drawn ones afterwards, which matters when you come to
describe how the training set was built.

The frames are already enhanced
-------------------------------
``sampling`` applies the enhancement profile before writing each frame, so the
images on disk are the ones the model will be trained on. This step reads them
exactly as they are: enhancing again would show SAM 3 something neither you nor
the trained detector will ever see.

Simplifying the outlines
------------------------
A mask traced at full resolution produces hundreds of vertices, which is a
correct outline and a miserable thing to edit by hand -- dragging one point at
a time through a polygon that dense is worse than redrawing it. The outlines
are simplified to something a person can actually adjust. That is the opposite
of what the contour exporter wants, which is why it does not share this step.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    from .inference import mask_to_contours
except ImportError:  # loaded by path, e.g. from a Colab bundle
    # The bundle unpacks the exporter under the name the notebook invokes,
    # which is not the name of its module, so try both before giving up.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from inference import mask_to_contours  # type: ignore[no-redef]
    except ImportError:
        from detectron2_export_contours import (  # type: ignore[no-redef]
            mask_to_contours,
        )

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# The sidecars the sampling step leaves beside the frames -- sampling_manifest,
# preprocess_profile, report -- need no special handling: find_frames only
# returns images, so they are never candidates in the first place.


def find_frames(folder: Path) -> list[Path]:
    """Every image in the folder, in a stable order."""
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def read_image(path: Path) -> np.ndarray | None:
    """Reads an image, tolerating non-ASCII paths the way sampling writes them."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def simplify(contour: np.ndarray, tolerance: float) -> np.ndarray:
    """Reduces a traced outline to something editable by hand.

    ``tolerance`` is a fraction of the contour's own perimeter, so the same
    value behaves consistently for a fish near the camera and one far from it.
    """
    if tolerance <= 0 or len(contour) <= 3:
        return contour
    epsilon = tolerance * cv2.arcLength(contour.astype(np.int32), True)
    reduced = cv2.approxPolyDP(contour.astype(np.int32), epsilon, True)[:, 0]
    # approxPolyDP can over-simplify a small blob into a line; a polygon needs
    # three points and the dataset step rejects anything with fewer.
    return reduced if len(reduced) >= 3 else contour


def labelme_document(
    image_path: Path, width: int, height: int, polygons: list[np.ndarray], label: str
) -> dict:
    """Builds the LabelMe file the dataset step knows how to read.

    Only the keys that step actually uses are written: ``shapes`` with polygon
    points, ``imagePath``, and the image size. ``imageData`` is deliberately
    left null rather than embedding a base64 copy of a frame that is sitting
    right beside the file.
    """
    return {
        "version": "5.5.0",
        "flags": {"sam3_prelabel": True},
        "shapes": [
            {
                "label": label,
                "points": [[float(x), float(y)] for x, y in polygon],
                "group_id": None,
                "description": "",
                "shape_type": "polygon",
                "flags": {"sam3_prelabel": True},
            }
            for polygon in polygons
        ],
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": int(height),
        "imageWidth": int(width),
    }


def prelabel(
    input_dir: Path,
    predictor,
    label: str,
    tolerance: float = 0.005,
    min_area: float = 1.0,
    overwrite: bool = False,
    progress=None,
    abort=None,
) -> dict:
    """Writes a draft LabelMe file for every frame that has not got one.

    Returns a summary, so a caller can report how much was drawn and how much
    was left alone.
    """
    frames = find_frames(input_dir)
    if not frames:
        raise FileNotFoundError(f"No images in {input_dir}")

    summary = {
        "frames": len(frames),
        "written": 0,
        "skipped_existing": 0,
        "unreadable": [],
        "empty": [],
        "polygons": 0,
    }

    for done, frame_path in enumerate(frames, start=1):
        if abort is not None and abort():
            break

        annotation_path = frame_path.with_suffix(".json")
        if annotation_path.exists() and not overwrite:
            # Corrected by hand, most likely. Never clobber it.
            summary["skipped_existing"] += 1
            if progress is not None:
                progress(done, len(frames))
            continue

        image = read_image(frame_path)
        if image is None:
            summary["unreadable"].append(frame_path.name)
            if progress is not None:
                progress(done, len(frames))
            continue

        masks, _ = predictor.predict(image)

        polygons: list[np.ndarray] = []
        for mask in masks:
            for contour in mask_to_contours(mask, min_area):
                polygons.append(simplify(contour, tolerance))

        if not polygons:
            # Worth naming: a frame SAM 3 found nothing in still needs
            # annotating, and an empty file would hide that.
            summary["empty"].append(frame_path.name)
            if progress is not None:
                progress(done, len(frames))
            continue

        height, width = image.shape[:2]
        document = labelme_document(frame_path, width, height, polygons, label)
        annotation_path.write_text(
            json.dumps(document, indent=2), encoding="utf-8"
        )
        summary["written"] += 1
        summary["polygons"] += len(polygons)

        if progress is not None:
            progress(done, len(frames))

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Pre-label sampled frames with SAM 3, for correction in LabelMe",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="the folder of sampled frames, as written by idtrackerai_d2_sample",
    )
    parser.add_argument(
        "--weights", type=Path, required=True, help="path to sam3.pt"
    )
    parser.add_argument(
        "--prompt",
        required=True,
        help="the word describing the animal, e.g. 'fish'",
    )
    parser.add_argument(
        "--label",
        help="class name written into the annotations (default: the prompt)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="drop detections SAM 3 is less sure of than this",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=0,
        help="expected number of animals per frame; 0 keeps every detection",
    )
    parser.add_argument(
        "--simplify",
        type=float,
        default=0.005,
        help="outline tolerance, as a fraction of each outline's perimeter."
        " Larger means fewer points and easier hand-editing",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=1.0,
        help="discard outlines below this polygon area",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="redraw frames that already have annotations. The default leaves"
        " them alone, so corrections are never lost",
    )
    args = parser.parse_args()

    try:
        from .sam3_predictor import Sam3Predictor
    except ImportError:  # loaded by path, e.g. from a Colab bundle
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from sam3_predictor import Sam3Predictor  # type: ignore[no-redef]

    predictor = Sam3Predictor(
        checkpoint=args.weights,
        prompt=args.prompt,
        device=args.device,
        score_threshold=args.score_threshold,
        max_instances=args.max_instances,
    )

    def progress(done, total):
        if done % 25 == 0 or done == total:
            print(f"  {done}/{total} frames", flush=True)

    summary = prelabel(
        input_dir=args.input_dir,
        predictor=predictor,
        label=args.label or args.prompt,
        tolerance=args.simplify,
        min_area=args.min_area,
        overwrite=args.overwrite,
        progress=progress,
    )

    print(
        f"\n{summary['written']} annotation file(s) written,"
        f" {summary['polygons']} outline(s) drawn"
    )
    if summary["skipped_existing"]:
        print(
            f"  {summary['skipped_existing']} frame(s) already annotated, left"
            " untouched. Pass --overwrite to redraw them."
        )
    if summary["empty"]:
        print(
            f"  {len(summary['empty'])} frame(s) where SAM 3 found nothing:"
            f" {', '.join(summary['empty'][:5])}"
            f"{' ...' if len(summary['empty']) > 5 else ''}"
        )
    if summary["unreadable"]:
        print(f"  {len(summary['unreadable'])} unreadable image(s)")

    print(
        "\nThese are drafts. Open the folder in LabelMe, correct the outlines,\n"
        "and add the animals SAM 3 missed before building the dataset."
    )


if __name__ == "__main__":
    main()
