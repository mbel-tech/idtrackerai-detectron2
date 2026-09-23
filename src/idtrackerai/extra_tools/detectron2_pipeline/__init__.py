"""Producing idtracker.ai contour files with an external segmentation model.

idtracker.ai's own segmentation thresholds each frame. This package holds the
pipeline that replaces it: sampling frames for annotation, converting LabelMe
polygons to COCO, and the frame enhancement all of those stages share.

The logic lives here, in the installed package, rather than in ``tools/``, so
that the Segmentation App and the command-line scripts run the same code. The
scripts under ``tools/`` are thin wrappers over these functions.

Training and inference are deliberately absent: they need a GPU and are driven
from the Colab notebook, not from here.
"""

from .preprocessing import (
    DEFAULT_SETTINGS,
    SETTING_KEYS,
    check_settings_match,
    describe,
    enhance,
    for_detectron2,
    load_profile,
    make_enhancer_from,
    resolve_settings,
    save_profile,
    settings_from_args,
)

__all__ = [
    "DEFAULT_SETTINGS",
    "SETTING_KEYS",
    "check_settings_match",
    "describe",
    "enhance",
    "for_detectron2",
    "load_profile",
    "make_enhancer_from",
    "resolve_settings",
    "save_profile",
    "settings_from_args",
]
