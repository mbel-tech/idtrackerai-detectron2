"""Zips exactly the files the Colab notebook needs, in the layout it expects.

The exporter falls back to loading ``write_contours`` from
``../src/idtrackerai/base/animals_detection/external_contours.py`` when
idtracker.ai is not installed, which it will not be on a Colab runtime. So the
bundle keeps that relative layout rather than flattening everything into one
folder.

    python tools/make_colab_bundle.py

Writes ``colab_bundle.zip`` (a few tens of KB). Upload it to Drive and let the
notebook unpack it.
"""

import argparse
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

FILES = [
    "tools/frame_preprocessing.py",
    "tools/sample_frames.py",
    "tools/labelme_to_coco.py",
    "tools/train_detectron2.py",
    "tools/detectron2_export_contours.py",
    "tools/check_videos.py",
    "tools/README.md",
    "src/idtrackerai/base/animals_detection/external_contours.py",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=REPO / "colab_bundle.zip"
    )
    args = parser.parse_args()

    missing = [f for f in FILES if not (REPO / f).is_file()]
    if missing:
        raise SystemExit("Missing from the repo:\n  " + "\n  ".join(missing))

    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in FILES:
            zf.write(REPO / name, name)

    size_kb = args.output.stat().st_size / 1024
    print(f"Wrote {args.output} ({size_kb:.0f} KB)")
    for name in FILES:
        print(f"  {name}")
    print(
        "\nUpload this to your Drive project folder. The notebook unpacks it to\n"
        "/content/code, which is the layout the exporter expects."
    )


if __name__ == "__main__":
    main()
