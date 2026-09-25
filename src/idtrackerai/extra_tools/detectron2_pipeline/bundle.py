"""Builds the zip the Colab notebook unpacks.

Ships from the installed package rather than from a repository checkout, so
``idtrackerai_d2_bundle`` works for anyone who pip-installed this fork. That
matters because the GPU stages are the part most likely to be run by someone
who never cloned anything.

Everything except annotation travels. Sampling and dataset building used to be
left behind as "local steps", but a machine with no CUDA card has no reason to
do them locally either, and driving them from the notebook keeps one project's
work in one place. Annotation is the exception and always will be: LabelMe is a
desktop application and Colab has no display, so the notebook samples frames,
hands them over for annotating, and takes them back.

Several files are written at archive paths that differ from their module names,
because the notebook and the scripts locate each other by those paths once
unpacked:

- ``preprocessing`` lands at ``tools/preprocessing.py``, which is where
  ``inference`` looks for it when idtracker.ai is not installed, as it is not
  on a Colab runtime
- ``external_contours`` keeps its repository path under ``src/``, which is
  where ``inference.load_writer`` looks for it
"""

import argparse
import zipfile
from importlib import resources
from pathlib import Path

# (module inside this package, path inside the zip)
PIPELINE_FILES = [
    ("preprocessing.py", "tools/preprocessing.py"),
    ("errors.py", "tools/errors.py"),
    # imported by name from a flat folder on the runtime, which is why each of
    # these falls back to a bare "from errors import ..." with no package
    ("sampling.py", "tools/sampling.py"),
    ("dataset.py", "tools/dataset.py"),
    ("training.py", "tools/train_detectron2.py"),
    ("inference.py", "tools/detectron2_export_contours.py"),
    ("videos.py", "tools/check_videos.py"),
    ("notebooks/colab_detectron2_pipeline.ipynb", "colab_detectron2_pipeline.ipynb"),
]

# files that live elsewhere in idtrackerai
EXTERNAL_FILES = [
    (
        "idtrackerai.base.animals_detection",
        "external_contours.py",
        "src/idtrackerai/base/animals_detection/external_contours.py",
    )
]


def build(output: Path) -> Path:
    """Writes the bundle, reading everything through importlib.resources."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    here = resources.files("idtrackerai.extra_tools.detectron2_pipeline")

    written: list[tuple[str, int]] = []
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, arcname in PIPELINE_FILES:
            source = here
            for part in name.split("/"):
                source = source / part
            data = source.read_bytes()
            zf.writestr(arcname, data)
            written.append((arcname, len(data)))

        for package, name, arcname in EXTERNAL_FILES:
            data = (resources.files(package) / name).read_bytes()
            zf.writestr(arcname, data)
            written.append((arcname, len(data)))

    build.last_contents = written  # type: ignore[attr-defined]
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Build the Colab bundle for the GPU stages",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output", type=Path, default=Path("colab_bundle.zip")
    )
    args = parser.parse_args()

    path = build(args.output)
    print(f"Wrote {path} ({path.stat().st_size / 1024:.0f} KB)")
    for arcname, size in getattr(build, "last_contents", []):
        print(f"  {arcname}  ({size / 1024:.0f} KB)")
    print(
        "\nUpload this to your Drive project folder alongside the dataset and\n"
        "your videos, then open colab_detectron2_pipeline.ipynb in Colab."
    )


if __name__ == "__main__":
    main()
