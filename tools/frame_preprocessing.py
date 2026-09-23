"""Locator for the frame-enhancement module, which now lives in the package.

The implementation moved to
``idtrackerai.extra_tools.detectron2_pipeline.preprocessing`` so that the
Segmentation App and these scripts run the same code rather than two copies that
drift. This file stays because the scripts here, and the Colab notebook, import
it by name, and because the tuning CLI is documented at this path.

It resolves the implementation in three places, in order:

1. the installed ``idtrackerai`` package — the normal case
2. ``../src/idtrackerai/.../preprocessing.py`` — running from a git checkout
3. ``./preprocessing.py`` — a Colab bundle, where idtracker.ai is not installed
   and ``make_colab_bundle.py`` ships the implementation next to this file

Tuning a recording setup still works exactly as documented::

    python tools/frame_preprocessing.py --video clip.mp4 --frame 500 \\
        --output check.png --save-profile setups/tank_a.json
"""

import importlib
import importlib.util
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PACKAGE = "idtrackerai.extra_tools.detectron2_pipeline.preprocessing"
_CANDIDATES = (
    _HERE.parent / "src/idtrackerai/extra_tools/detectron2_pipeline/preprocessing.py",
    _HERE / "preprocessing.py",
)


def _load():
    try:
        return importlib.import_module(_PACKAGE)
    except ImportError:
        pass

    for candidate in _CANDIDATES:
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location(
                "idtrackerai_frame_preprocessing", candidate
            )
            module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
            sys.modules[spec.name] = module  # type: ignore[union-attr]
            spec.loader.exec_module(module)  # type: ignore[union-attr]
            return module

    raise ImportError(
        "Could not find the frame-enhancement implementation. Either install "
        "this idtracker.ai fork, run from a git checkout, or rebuild the Colab "
        "bundle with tools/make_colab_bundle.py."
    )


_impl = _load()

# Re-export everything public, so `import frame_preprocessing as fp` keeps
# working unchanged for every caller.
globals().update({k: v for k, v in vars(_impl).items() if not k.startswith("__")})

__all__ = [k for k in vars(_impl) if not k.startswith("_")]


if __name__ == "__main__":
    _impl._demo()
