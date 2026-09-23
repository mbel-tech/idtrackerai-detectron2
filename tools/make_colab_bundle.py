"""Zips exactly the files the Colab notebook needs, in the layout it expects.

Only the GPU stages run on Colab, so the bundle carries training, inference and
the video preflight. Frame sampling and dataset building are local stages done
before anything is uploaded, and they now live in the installed package, so they
are deliberately not shipped.

Two files are placed at paths that differ from their location in the repo:

- the frame-enhancement implementation is written to ``tools/preprocessing.py``,
  which is where ``tools/frame_preprocessing.py`` looks for it when
  idtracker.ai is not installed — as it is not on a Colab runtime
- ``external_contours.py`` keeps its ``src/...`` path, because the exporter
  resolves it relative to the repo layout

    python tools/make_colab_bundle.py

Writes ``colab_bundle.zip``, a few tens of KB. Upload it to Drive and let the
notebook unpack it.
"""

import argparse
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# (path in the repo, path inside the zip)
FILES = [
    ("tools/frame_preprocessing.py", "tools/frame_preprocessing.py"),
    (
        "src/idtrackerai/extra_tools/detectron2_pipeline/preprocessing.py",
        "tools/preprocessing.py",
    ),
    ("tools/train_detectron2.py", "tools/train_detectron2.py"),
    ("tools/detectron2_export_contours.py", "tools/detectron2_export_contours.py"),
    ("tools/check_videos.py", "tools/check_videos.py"),
    ("tools/README.md", "tools/README.md"),
    (
        "src/idtrackerai/base/animals_detection/external_contours.py",
        "src/idtrackerai/base/animals_detection/external_contours.py",
    ),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO / "colab_bundle.zip")
    args = parser.parse_args()

    missing = [src for src, _ in FILES if not (REPO / src).is_file()]
    if missing:
        raise SystemExit("Missing from the repo:\n  " + "\n  ".join(missing))

    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as zf:
        for source, arcname in FILES:
            zf.write(REPO / source, arcname)

    size_kb = args.output.stat().st_size / 1024
    print(f"Wrote {args.output} ({size_kb:.0f} KB)")
    for source, arcname in FILES:
        note = "" if source == arcname else f"   <- {source}"
        print(f"  {arcname}{note}")
    print(
        "\nUpload this to your Drive project folder. The notebook unpacks it to\n"
        "/content/code, which is the layout the GPU scripts expect."
    )


if __name__ == "__main__":
    main()
