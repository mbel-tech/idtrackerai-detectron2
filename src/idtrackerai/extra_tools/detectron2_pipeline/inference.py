"""Runs an instance-segmentation model over a video and writes the animals'
outlines to a sidecar file for idtracker.ai.

This replaces the older "enhance the video, then let idtracker.ai re-threshold
it" route. Nothing is painted back into pixels, so no per-clip intensity or area
threshold has to be tuned: the contours the model produces are the contours
idtracker.ai tracks.

Runs where the GPU is (Colab, a workstation) and produces one small .h5 per
video. Tracking then happens wherever idtracker.ai lives; the two no longer have
to be the same machine, and an enhanced video never has to be encoded or moved.

Two models can do the segmenting, and they ask for opposite things:

``--backend detectron2`` (default)
    A Mask R-CNN fine-tuned on frames you annotated yourself. Costs a day of
    annotation and a training run, and in exchange knows your species, your
    tank and your lighting.

``--backend sam3``
    Meta's SAM 3, prompted with a word such as ``fish``. No annotation and no
    training, but it is a general model that has never seen your setup, so it
    is weaker exactly where thresholding is weak: low contrast and small,
    similar animals. Also useful as a first pass to pre-label frames you then
    correct and train on — see ``prelabel.py``.

Whichever runs, the output is the same file in the same format, and everything
after the masks — cleanup, overlap policy, polygons, the sidecar — is shared.

Usage
-----
    python detectron2_export_contours.py \\
        --video clip_01.mp4 \\
        --weights model_final.pth \\
        --output clip_01_contours.h5 \\
        --max-instances 5

or, with no trained model of your own::

    python detectron2_export_contours.py \\
        --backend sam3 --prompt fish \\
        --video clip_01.mp4 \\
        --weights sam3.pt \\
        --output clip_01_contours.h5 \\
        --max-instances 5

Then in the idtracker.ai .toml for that clip::

    external_contours = "clip_01_contours.h5"

Overlap policy
--------------
``--on-overlap`` decides what happens when the model reports two animals whose
masks intersect:

``merge`` (default)
    Strongly overlapping masks become one contour. idtracker.ai sees a single
    blob, its crossing detector flags it, and the usual fragmentation and
    interpolation reconstruct who was who. Use this to keep identity
    reconstruction in idtracker.ai's hands.

``split``
    Contested pixels go to the higher-scoring instance and each animal is
    emitted separately. The detector resolves the occlusion instead. Fewer
    crossings, but mask assignment can flicker between frames, and that
    instability propagates into fragmentation.
"""

import argparse
import importlib.util
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

try:
    from . import preprocessing as fp
except ImportError:  # loaded by path, e.g. from a Colab bundle
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import preprocessing as fp  # type: ignore[no-redef]


def load_training_metadata(weights: Path) -> dict | None:
    """Reads training_metadata.json written beside the weights, if present."""
    path = Path(weights).parent / "training_metadata.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_writer():
    """Imports write_contours from idtrackerai, or straight from the repo.

    On a Colab runtime idtracker.ai is usually not installed, but the file only
    needs numpy and h5py, so loading it by path works fine.
    """
    try:
        from idtrackerai.base.animals_detection.external_contours import (  # type: ignore
            write_contours,
        )

        return write_contours
    except ImportError:
        pass

    # Colab bundle layout: this file is unpacked at <root>/tools/ and the
    # writer keeps its repository path under <root>/src/.
    candidate = (
        Path(__file__).resolve().parent.parent
        / "src/idtrackerai/base/animals_detection/external_contours.py"
    )
    if not candidate.is_file():
        raise ImportError(
            "Could not find external_contours.py. Either install this "
            "idtracker.ai fork, or keep this script inside the repo's tools/ "
            "folder."
        )
    spec = importlib.util.spec_from_file_location("external_contours", candidate)
    module = importlib.util.module_from_spec(spec)  # type: ignore
    sys.modules["external_contours"] = module
    spec.loader.exec_module(module)  # type: ignore
    return module.write_contours


# --------------------------------------------------------------- mask cleanup
def clean_mask(
    mask: np.ndarray, min_component: int, kernel: np.ndarray, dilate: int
) -> np.ndarray:
    """Drops specks, closes pinholes, smooths the boundary, then grows slightly.

    Mirrors the post-processing that was already in use: small connected
    components removed, morphological close then open with a 3x3 ellipse, and a
    1-pixel dilation so the whole animal is inside the outline.
    """
    mask = mask.astype(np.uint8)

    if min_component > 0:
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if n_labels > 1:
            keep = np.zeros(n_labels, bool)
            keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_component
            mask = keep[labels].astype(np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    if dilate > 0:
        mask = cv2.dilate(mask, kernel, iterations=dilate)
    return mask


def deduplicate(masks: list[np.ndarray], scores: list[float], iou_th: float):
    """Removes near-duplicate detections, keeping the more confident one."""
    order = sorted(range(len(masks)), key=lambda i: scores[i], reverse=True)
    kept: list[int] = []
    for i in order:
        duplicate = False
        for j in kept:
            intersection = np.logical_and(masks[i], masks[j]).sum()
            if not intersection:
                continue
            union = np.logical_or(masks[i], masks[j]).sum()
            if intersection / union >= iou_th:
                duplicate = True
                break
        if not duplicate:
            kept.append(i)
    return [masks[i] for i in kept], [scores[i] for i in kept]


def resolve_overlaps(masks: list[np.ndarray], scores: list[float], policy: str,
                     merge_th: float):
    """Applies the chosen overlap policy, returning the masks to be emitted."""
    if len(masks) < 2 or policy == "keep":
        return masks, scores

    if policy == "split":
        # Highest score claims contested pixels; every animal stays separate.
        order = sorted(range(len(masks)), key=lambda i: scores[i], reverse=True)
        claimed = np.zeros_like(masks[0], bool)
        out_masks, out_scores = [], []
        for i in order:
            exclusive = np.logical_and(masks[i], ~claimed)
            if exclusive.any():
                claimed |= exclusive.astype(bool)
                out_masks.append(exclusive.astype(np.uint8))
                out_scores.append(scores[i])
        return out_masks, out_scores

    # policy == "merge": fuse masks that genuinely overlap into one blob and let
    # idtracker.ai's crossing machinery take it from there.
    groups = list(range(len(masks)))

    def root(i):
        while groups[i] != i:
            groups[i] = groups[groups[i]]
            i = groups[i]
        return i

    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            intersection = np.logical_and(masks[i], masks[j]).sum()
            if not intersection:
                continue
            smaller = min(masks[i].sum(), masks[j].sum())
            if smaller and intersection / smaller >= merge_th:
                groups[root(i)] = root(j)

    fused: dict[int, np.ndarray] = {}
    fused_scores: dict[int, float] = {}
    for i, mask in enumerate(masks):
        r = root(i)
        if r in fused:
            fused[r] = np.logical_or(fused[r], mask).astype(np.uint8)
            fused_scores[r] = min(fused_scores[r], scores[i])
        else:
            fused[r] = mask
            fused_scores[r] = scores[i]
    return list(fused.values()), list(fused_scores.values())


def mask_to_contours(mask: np.ndarray, min_area: float) -> list[np.ndarray]:
    """Outer outlines of a binary mask, as [n_points, 2] int32 arrays."""
    found, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS
    )
    return [
        c[:, 0].astype(np.int32)
        for c in found
        if len(c) >= 3 and cv2.contourArea(c) >= min_area
    ]


# ------------------------------------------------------------------ backends
# A backend is anything with a ``predict`` method taking one BGR frame and
# returning its instance masks and their scores:
#
#     predict(frame) -> (list[np.ndarray uint8 H*W], list[float])
#
# Everything after that point in this file — mask cleanup, deduplication, the
# overlap policy, the polygons and the sidecar itself — is the same whichever
# model produced the masks, and the sidecar format has never mentioned
# Detectron2. Keeping the model behind this one small interface is what lets a
# second one be added without touching the export loop.


class Detectron2Predictor:
    """A fine-tuned Mask R-CNN, trained on frames you annotated yourself."""

    def __init__(self, args, metadata: dict | None):
        from detectron2 import model_zoo  # type: ignore
        from detectron2.config import get_cfg  # type: ignore
        from detectron2.engine import DefaultPredictor  # type: ignore

        # Class count and test scale must match training. Take them from the
        # training record unless the user overrode them explicitly, because a
        # mismatch here produces a model that loads cleanly and predicts nonsense.
        num_classes = args.num_classes
        if num_classes is None:
            num_classes = (metadata or {}).get("num_classes", 1)
        min_size = args.min_size
        if min_size is None:
            min_size = (metadata or {}).get("min_size_test", 640)
        config = args.config or (metadata or {}).get(
            "config", "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )

        print(f"Model: {config}, {num_classes} class(es), test scale {min_size}")

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(config))
        cfg.MODEL.WEIGHTS = str(args.weights)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.score_threshold
        cfg.TEST.DETECTIONS_PER_IMAGE = args.max_instances
        cfg.INPUT.MIN_SIZE_TEST = min_size
        cfg.MODEL.DEVICE = args.device
        self._predictor = DefaultPredictor(cfg)
        self.description = config

    def predict(self, frame: np.ndarray):
        instances = self._predictor(frame)["instances"].to("cpu")
        masks = list(instances.pred_masks.numpy().astype(np.uint8))
        scores = [float(s) for s in instances.scores.numpy()]
        return masks, scores


def build_predictor(args, metadata: dict | None):
    """Returns the backend named by ``--backend`` and a string describing it.

    The description is recorded in the sidecar so a contour file says which
    model made it, which matters once two of them can.
    """
    if args.backend == "sam3":
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
    else:
        predictor = Detectron2Predictor(args, metadata)
    return predictor, predictor.description


def export_video(
    video: Path,
    output: Path,
    args,
    predictor,
    model_description: str,
    enhance,
    enhancement: dict,
    write_contours,
    progress=None,
    abort=None,
) -> dict | None:
    """Runs the model over one video and writes its contour file.

    Returns a stats dict, which the batch runner records so a later session can
    report on work done in an earlier one.

    ``progress(done, total)`` is called per frame, for a caller with a progress
    bar to fill. ``abort()`` is polled per frame, following the convention in
    ``sampling.py``; when it returns true this gives up and returns ``None``
    **without writing anything**. A half-exported clip must not leave a file
    behind, because the resume logic counts an existing file as a finished one
    and would skip the rest of the video for good.
    """
    started = datetime.now(timezone.utc)
    local = video
    temporary_copy = None

    if args.local_cache:
        # Drive's FUSE mount is slow for the seeking cv2 does, badly enough to
        # dominate inference time. Copying the clip to local disk first is
        # usually several times faster overall despite the up-front copy.
        args.local_cache.mkdir(parents=True, exist_ok=True)
        temporary_copy = args.local_cache / video.name
        if not temporary_copy.exists():
            print(f"  copying to {temporary_copy} ...", flush=True)
            shutil.copy2(video, temporary_copy)
        local = temporary_copy

    cap = cv2.VideoCapture(str(local))
    if not cap.isOpened():
        raise SystemExit(f"Could not open {local}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.limit:
        n_frames = min(n_frames, args.limit)

    print(f"{video.name}: {n_frames} frames at {width}x{height}", flush=True)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    contours_per_frame: list[list[np.ndarray]] = []
    scores_per_frame: list[list[float]] = []
    empty_frames = 0
    short_frames = 0
    clock = time.monotonic()

    for frame_index in range(n_frames):
        if abort is not None and abort():
            cap.release()
            if temporary_copy is not None and not args.keep_cache:
                temporary_copy.unlink(missing_ok=True)
            return None

        ok, frame = cap.read()
        if not ok:
            # Keep the frame slot so frame numbers stay aligned with the video.
            # idtracker.ai treats a frame with no contours as one with no
            # visible animals, which is what an unreadable frame amounts to.
            contours_per_frame.append([])
            scores_per_frame.append([])
            empty_frames += 1
            continue

        masks, scores = predictor.predict(enhance(frame))

        masks = [clean_mask(m, args.min_component, kernel, args.dilate) for m in masks]
        keep = [i for i, m in enumerate(masks) if m.any()]
        masks = [masks[i] for i in keep]
        scores = [scores[i] for i in keep]

        if args.dedup_iou < 1.0 and len(masks) > 1:
            masks, scores = deduplicate(masks, scores, args.dedup_iou)

        masks, scores = resolve_overlaps(
            masks, scores, args.on_overlap, args.merge_overlap
        )

        frame_contours: list[np.ndarray] = []
        frame_scores: list[float] = []
        for mask, score in zip(masks, scores):
            for contour in mask_to_contours(mask, args.min_area):
                frame_contours.append(contour)
                frame_scores.append(score)

        contours_per_frame.append(frame_contours)
        scores_per_frame.append(frame_scores)

        if not frame_contours:
            empty_frames += 1
        elif len(frame_contours) < args.max_instances:
            short_frames += 1

        if progress is not None:
            progress(frame_index + 1, n_frames)

        if frame_index and frame_index % args.progress_every == 0:
            done = frame_index + 1
            rate = done / max(time.monotonic() - clock, 1e-6)
            remaining = (n_frames - done) / max(rate, 1e-6)
            print(
                f"  {done}/{n_frames} ({100 * done / n_frames:.1f}%)"
                f"  {rate:.1f} fps  ~{remaining / 60:.0f} min left",
                flush=True,
            )

    cap.release()
    if temporary_copy is not None and not args.keep_cache:
        temporary_copy.unlink(missing_ok=True)

    # Write to a .partial first, then move into place. A session that dies
    # mid-write would otherwise leave a truncated .h5 that the resume logic
    # would count as a finished video.
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    # A contour file should say what made it, now that more than one thing can.
    backend_provenance = {"backend": args.backend}
    if args.backend == "sam3":
        backend_provenance["prompt"] = args.prompt

    write_contours(
        partial,
        contours_per_frame,
        width=width,
        height=height,
        scores_per_frame=scores_per_frame,
        # This file is shipped to Colab under a different name, so read it off
        # the file itself rather than hardcoding one of the two.
        source=Path(__file__).name,
        video=video.name,
        model=model_description,
        weights=Path(args.weights).name,
        score_threshold=args.score_threshold,
        max_instances=args.max_instances,
        on_overlap=args.on_overlap,
        enhancement=json.dumps(enhancement),
        **backend_provenance,
        postprocessing=json.dumps(
            {
                "min_component": args.min_component,
                "dilate": args.dilate,
                "dedup_iou": args.dedup_iou,
                "merge_overlap": args.merge_overlap,
            }
        ),
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    partial.replace(output)

    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    total = sum(len(c) for c in contours_per_frame)
    stats = {
        "video": video.name,
        "output": str(output),
        "frames": len(contours_per_frame),
        "contours": total,
        "contours_per_frame": round(total / max(len(contours_per_frame), 1), 3),
        "empty_frames": empty_frames,
        "short_frames": short_frames,
        "seconds": round(elapsed, 1),
        "fps": round(len(contours_per_frame) / max(elapsed, 1e-6), 2),
        "size_mb": round(output.stat().st_size / 1e6, 2),
        "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    print(
        f"  done: {output.name} ({stats['size_mb']} MB), {total} contours,"
        f" {stats['fps']} fps, {empty_frames} empty / {short_frames} short frames",
        flush=True,
    )
    return stats


def _is_sidecar(path) -> bool:
    """Filesystem metadata that merely looks like a media file.

    macOS writes a "._name.mp4" companion beside every file it copies onto
    a non-Apple filesystem. It carries the same extension, so any *.mp4
    glob picks it up and OpenCV then reports it as an unreadable video.
    """
    name = Path(path).name
    return name.startswith("._") or name in (".DS_Store", "Thumbs.db", "desktop.ini")


def resolve_videos(args) -> list[Path]:
    if args.video:
        return [args.video]
    videos: list[Path] = []
    for pattern in args.videos:
        if any(ch in str(pattern) for ch in "*?"):
            videos.extend(sorted(pattern.parent.glob(pattern.name)))
        else:
            videos.append(pattern)
    return [v for v in videos if v.is_file() and not _is_sidecar(v)]


def main():
    parser = argparse.ArgumentParser(
        description="Export Detectron2 instance contours for idtracker.ai",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--backend",
        choices=("detectron2", "sam3"),
        default="detectron2",
        help=(
            "detectron2: a Mask R-CNN fine-tuned on frames you annotated;"
            " sam3: Meta's SAM 3 prompted with a word for the animal, which"
            " needs no annotation or training but is not specific to your setup"
        ),
    )
    parser.add_argument(
        "--prompt",
        help="with --backend sam3, the word describing the animal, e.g. 'fish'",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path, help="a single video")
    source.add_argument(
        "--videos", type=Path, nargs="+", help="several videos, or a glob"
    )
    parser.add_argument("--weights", type=Path, required=True)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--output", type=Path, help="contour file, with --video")
    target.add_argument(
        "--output-dir",
        type=Path,
        help="folder for one <video stem>.h5 per input, with --videos",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="redo videos whose contour file already exists. The default is to"
        " skip them, which is what makes a batch resumable across sessions",
    )
    parser.add_argument(
        "--local-cache",
        type=Path,
        help="copy each video here before decoding, e.g. /content/cache when the"
        " videos live on a slow Drive mount",
    )
    parser.add_argument(
        "--keep-cache", action="store_true", help="do not delete cached copies"
    )
    parser.add_argument(
        "--progress-every", type=int, default=500, help="frames between progress lines"
    )
    parser.add_argument(
        "--config",
        help="model_zoo config the model was fine-tuned from"
        " (default: taken from training_metadata.json)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        help="default: taken from training_metadata.json, else 1",
    )
    parser.add_argument("--score-threshold", type=float, default=0.7)
    parser.add_argument(
        "--max-instances",
        type=int,
        default=5,
        help="expected number of animals; caps detections per frame",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        help="inference scale; default: the value training recorded, else 640",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--min-component",
        type=int,
        default=80,
        help="drop connected components smaller than this, in pixels",
    )
    parser.add_argument("--dilate", type=int, default=1)
    parser.add_argument("--dedup-iou", type=float, default=0.7)
    parser.add_argument(
        "--min-area",
        type=float,
        default=1.0,
        help="discard contours below this polygon area",
    )
    parser.add_argument(
        "--on-overlap",
        choices=("merge", "split", "keep"),
        default="merge",
        help=(
            "merge: overlapping animals become one blob for idtracker.ai's "
            "crossing detector; split: assign contested pixels to the higher "
            "score and keep animals separate; keep: emit masks untouched"
        ),
    )
    parser.add_argument(
        "--merge-overlap",
        type=float,
        default=0.15,
        help=(
            "with --on-overlap merge, fuse two masks when their intersection "
            "covers at least this fraction of the smaller one"
        ),
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="stop after N frames (0 = whole video)"
    )
    fp.add_arguments(parser)
    args = parser.parse_args()

    if args.videos and not args.output_dir:
        parser.error("--videos needs --output-dir")
    if args.video and not args.output:
        parser.error("--video needs --output")

    # Say so rather than ignoring them. A flag that looks accepted but does
    # nothing is how someone ends up believing they set a class count on a
    # model that has no classes.
    detectron2_only = {
        "--num-classes": args.num_classes,
        "--min-size": args.min_size,
        "--config": args.config,
    }
    if args.backend == "sam3":
        if not args.prompt:
            parser.error("--backend sam3 needs --prompt, e.g. --prompt fish")
        used = [flag for flag, value in detectron2_only.items() if value is not None]
        if used:
            verb = "only applies" if len(used) == 1 else "only apply"
            parser.error(f"{', '.join(used)} {verb} to --backend detectron2")
    elif args.prompt:
        parser.error("--prompt only applies to --backend sam3")

    videos = resolve_videos(args)
    if not videos:
        raise SystemExit("No videos matched")

    write_contours = load_writer()
    # SAM 3 was never fine-tuned on this setup, so there is no training record
    # to read and nothing to check the enhancement against. Looking for one
    # would only produce a warning about a file that was never meant to exist.
    metadata = load_training_metadata(args.weights) if args.backend == "detectron2" else None
    # Inference adopts whatever the model was trained with, unless the user
    # deliberately overrides it. Retyping the flags identically at inference
    # time was the old requirement, and an easy thing to get silently wrong.
    trained_with = (metadata or {}).get("enhancement")
    enhancement, notes = fp.resolve_settings(
        args, fallback=trained_with, fallback_label="the model's training record"
    )
    for note in notes:
        print(note)
    print(f"enhancement: {fp.describe(enhancement)}")
    enhance = fp.make_enhancer_from(enhancement)

    if args.backend == "detectron2":
        if metadata is None:
            print(
                f"No training_metadata.json beside {args.weights}; cannot check"
                " that inference matches how the model was trained."
            )
        else:
            mismatch = fp.check_settings_match(trained_with, enhancement, "training")
            if mismatch:
                print(f"\nWARNING: {mismatch}\n")

    def output_for(video: Path) -> Path:
        return args.output if args.output else args.output_dir / f"{video.stem}.h5"

    pending = [v for v in videos if args.overwrite or not output_for(v).exists()]
    done_already = len(videos) - len(pending)
    print(
        f"{len(videos)} video(s); {done_already} already exported,"
        f" {len(pending)} to do"
    )
    if not pending:
        print("Nothing to do. Pass --overwrite to redo them.")
        return

    predictor, model_description = build_predictor(args, metadata)

    log_path = (args.output_dir or args.output.parent) / "export_log.json"
    log = []
    if log_path.is_file():
        try:
            log = json.loads(log_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log = []

    for i, video in enumerate(pending, start=1):
        print(f"\n[{i}/{len(pending)}] {video.name}", flush=True)
        try:
            stats = export_video(
                video,
                output_for(video),
                args,
                predictor,
                model_description,
                enhance,
                enhancement,
                write_contours,
            )
        except KeyboardInterrupt:
            print("\nInterrupted. Finished videos are kept; rerun to continue.")
            break
        if stats is None:  # only reachable with an abort callback, but cheap
            break
        log = [entry for entry in log if entry.get("video") != stats["video"]]
        log.append(stats)
        # rewritten after every video, so a session that dies still leaves a record
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps(log, indent=2), encoding="utf-8")

    total_frames = sum(e["frames"] for e in log)
    empty = sum(e["empty_frames"] for e in log)
    short = sum(e["short_frames"] for e in log)
    print(f"\n{len(log)}/{len(videos)} video(s) exported, {total_frames:,} frames")
    print(
        f"  {empty:,} frames with no detection,"
        f" {short:,} with fewer than {args.max_instances}"
    )
    print(f"  log: {log_path}")
    if empty or short:
        print(
            "\nFrames with missing animals are left as they are, on purpose.\n"
            "Do not fill them in here: idtracker.ai's crossing detection and\n"
            "interpolation are built for exactly this and work from the whole\n"
            "video, not from one previous frame."
        )


if __name__ == "__main__":
    main()
