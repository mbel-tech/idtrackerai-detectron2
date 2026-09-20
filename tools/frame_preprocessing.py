"""The frame enhancement applied before Detectron2, shared by every stage.

This has to be one module, not a copy in each script. The model learns the
appearance of the frames it was annotated on, so the frames extracted for
annotation, the frames used for training and the frames passed to the predictor
at inference must all go through the exact same function. A CLAHE setting that
differs between training and inference is a silent accuracy loss that looks like
a model problem.

The steps follow the established workflow:

1. grayscale
2. correction for uneven illumination, by subtracting a Gaussian-blurred
   background estimate computed on a downsampled copy of the frame
3. CLAHE (clip limit 1.5, 8x8 tiles)

Detectron2's R-50-FPN configs expect 3-channel BGR input, so the enhanced
single-channel image is replicated across three channels by :func:`for_detectron2`.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

# Defaults matching the documented workflow.
CLAHE_CLIP_LIMIT = 1.5
CLAHE_TILE_GRID = 8
ILLUMINATION_DOWNSAMPLE = 4
ILLUMINATION_SIGMA = 25.0
ILLUMINATION_PIVOT = 128


def to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim > 2:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def correct_illumination(
    gray: np.ndarray,
    downsample: int = ILLUMINATION_DOWNSAMPLE,
    sigma: float = ILLUMINATION_SIGMA,
    pivot: int = ILLUMINATION_PIVOT,
) -> np.ndarray:
    """Flattens uneven lighting by removing a blurred estimate of the background.

    The estimate is built on a downsampled copy, which is both faster and
    smoother than blurring at full resolution.

    The result is re-centred on ``pivot`` rather than saturated at zero. A plain
    subtraction clips everything darker than the local background to black,
    which erases exactly the contrast a dark animal on a dark background needs.
    """
    if downsample > 1:
        small = cv2.resize(
            gray, None, fx=1 / downsample, fy=1 / downsample, interpolation=cv2.INTER_AREA
        )
    else:
        small = gray

    background = cv2.GaussianBlur(small, (0, 0), sigma / max(downsample, 1))

    if downsample > 1:
        background = cv2.resize(
            background, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR
        )

    return cv2.addWeighted(gray, 1.0, background, -1.0, pivot)


def apply_clahe(
    gray: np.ndarray,
    clip_limit: float = CLAHE_CLIP_LIMIT,
    tile_grid: int = CLAHE_TILE_GRID,
) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    return clahe.apply(gray)


def enhance(
    frame: np.ndarray,
    clahe_clip: float = CLAHE_CLIP_LIMIT,
    clahe_tile: int = CLAHE_TILE_GRID,
    downsample: int = ILLUMINATION_DOWNSAMPLE,
    sigma: float = ILLUMINATION_SIGMA,
    correct_lighting: bool = True,
) -> np.ndarray:
    """Full enhancement, returning a single-channel uint8 image."""
    gray = to_gray(frame)
    if correct_lighting:
        gray = correct_illumination(gray, downsample, sigma)
    return apply_clahe(gray, clahe_clip, clahe_tile)


def for_detectron2(frame: np.ndarray, **kwargs) -> np.ndarray:
    """Enhanced frame as 3-channel BGR, which is what the R-50-FPN configs want."""
    return cv2.cvtColor(enhance(frame, **kwargs), cv2.COLOR_GRAY2BGR)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Adds the enhancement flags, so every tool exposes the same names."""
    group = parser.add_argument_group("frame enhancement")
    group.add_argument(
        "--no-enhance",
        action="store_true",
        help="feed raw frames to the model (only if it was trained on raw frames)",
    )
    group.add_argument("--clahe-clip", type=float, default=CLAHE_CLIP_LIMIT)
    group.add_argument("--clahe-tile", type=int, default=CLAHE_TILE_GRID)
    group.add_argument("--illumination-sigma", type=float, default=ILLUMINATION_SIGMA)
    group.add_argument(
        "--illumination-downsample", type=int, default=ILLUMINATION_DOWNSAMPLE
    )
    group.add_argument(
        "--no-illumination-correction",
        action="store_true",
        help="apply CLAHE only, leaving uneven lighting in place",
    )


def settings_from_args(args) -> dict:
    """The enhancement settings as a dict, for recording alongside outputs.

    Every tool writes this next to its results so a later run can be checked for
    drift against the settings the model was trained with.
    """
    return {
        "enhance": not args.no_enhance,
        "clahe_clip": args.clahe_clip,
        "clahe_tile": args.clahe_tile,
        "illumination_sigma": args.illumination_sigma,
        "illumination_downsample": args.illumination_downsample,
        "correct_lighting": not args.no_illumination_correction,
    }


def make_enhancer(args):
    """Builds the callable each tool applies to every frame."""
    if args.no_enhance:
        return lambda frame: frame if frame.ndim == 3 else cv2.cvtColor(
            frame, cv2.COLOR_GRAY2BGR
        )

    def enhancer(frame: np.ndarray) -> np.ndarray:
        return for_detectron2(
            frame,
            clahe_clip=args.clahe_clip,
            clahe_tile=args.clahe_tile,
            downsample=args.illumination_downsample,
            sigma=args.illumination_sigma,
            correct_lighting=not args.no_illumination_correction,
        )

    return enhancer


def check_settings_match(recorded: dict | None, current: dict, label: str) -> str | None:
    """Compares enhancement settings, returning a warning when they differ.

    Used to catch the case where a model trained on enhanced frames is later run
    on raw ones, which degrades detection without any error being raised.
    """
    if not recorded:
        return None
    differences = [
        f"{key}: trained with {recorded[key]!r}, now {current[key]!r}"
        for key in current
        if key in recorded and recorded[key] != current[key]
    ]
    if not differences:
        return None
    return (
        f"Frame enhancement differs from {label}:\n  "
        + "\n  ".join(differences)
        + "\nThe model sees different-looking images than it was trained on."
    )


def _demo():
    """Writes a before/after strip, to eyeball the settings on one frame."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("enhancement_demo.png"))
    add_arguments(parser)
    args = parser.parse_args()

    cap = cv2.VideoCapture(str(args.video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Could not read frame {args.frame} of {args.video}")

    gray = to_gray(frame)
    lit = correct_illumination(gray, args.illumination_downsample, args.illumination_sigma)
    final = apply_clahe(lit, args.clahe_clip, args.clahe_tile)
    cv2.imwrite(str(args.output), np.hstack([gray, lit, final]))
    print(f"Wrote {args.output} (grayscale | illumination-corrected | + CLAHE)")


if __name__ == "__main__":
    _demo()
