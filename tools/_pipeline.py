"""Finds the pipeline package, installed or not.

The scripts here are meant to run from a git checkout as well as from an
installed package, and on a Colab runtime where idtracker.ai is not installed at
all. This resolves the package in that order and hands back its modules.

Loading by path uses ``submodule_search_locations`` so the package's own
relative imports keep working; without that, ``from .preprocessing import ...``
inside the package would fail.
"""

import importlib
import importlib.util
import sys
from pathlib import Path

_INSTALLED = "idtrackerai.extra_tools.detectron2_pipeline"
_ALIAS = "_idtrackerai_d2_pipeline"
_DIR = (
    Path(__file__).resolve().parent.parent
    / "src/idtrackerai/extra_tools/detectron2_pipeline"
)


def package():
    """The pipeline package, however it can be reached."""
    try:
        return importlib.import_module(_INSTALLED)
    except ImportError:
        pass

    if _ALIAS in sys.modules:
        return sys.modules[_ALIAS]

    init = _DIR / "__init__.py"
    if not init.is_file():
        raise ImportError(
            "Could not find the Detectron2 pipeline package. Either install "
            "this idtracker.ai fork, or run these scripts from a git checkout."
        )

    spec = importlib.util.spec_from_file_location(
        _ALIAS, init, submodule_search_locations=[str(_DIR)]
    )
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[_ALIAS] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def module(name: str):
    """One submodule, e.g. ``module("sampling")``."""
    return importlib.import_module(f"{package().__name__}.{name}")
