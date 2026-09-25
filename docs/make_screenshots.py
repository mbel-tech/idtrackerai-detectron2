"""Regenerates the screenshots in USER_GUIDE.md.

The guide shows an interface that changes. Committing the images without the
means to remake them would leave it describing a version of the app nobody is
running within a month, so this script exists to be re-run rather than the
images to be recaptured by hand.

It drives the real Segmentation App offscreen, the way the test suites do, so
what you see is the application and not a mock-up.

    python docs/make_screenshots.py --videos "D:/my footage/*.mp4"

Two kinds of footage are used, deliberately:

* the example clip that ships with idtracker.ai, for the picture of
  thresholding *working* -- it is well lit, the animals are dark on a pale
  background, and it is already public;
* your own clips, for everything else, including the picture of thresholding
  *failing*, which is the reason this fork exists. None of yours are committed
  by this script; whether the screenshots go into a public repository is your
  decision to make.

Pick clips from one recording setup: several sections of the guide are about
how the app treats a collection.
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy.QtGui import QFont, QFontDatabase  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_IMAGES = HERE / "images"

# One size for every capture, so the guide does not jump about.
WIDTH, HEIGHT = 1400, 920

# A font has to be loaded explicitly: the offscreen platform has no system
# font configuration, and every label renders as boxes without one.
FONT_CANDIDATES = [
    r"C:\Windows\Fonts\segoeui.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]


def settle(app, rounds: int = 80) -> None:
    """Let Qt finish laying out and painting before the grab."""
    for _ in range(rounds):
        app.processEvents()
        time.sleep(0.01)


def capture(app, gui, images: Path, name: str) -> Path:
    settle(app)
    path = images / name
    gui.grab().save(str(path))
    print(f"  {name:34s} {path.stat().st_size / 1024:6.0f} KB")
    return path


def wait_until(app, predicate, seconds: float = 120.0) -> bool:
    deadline = time.monotonic() + seconds
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.02)
    return predicate()


def stand_in_contours(clip: Path, where: Path) -> Path:
    """An empty contour file matching a clip, so External contours can be shown.

    Not committed, and no use for tracking. It exists so that screenshot shows
    the real widget in its real state rather than a mode nobody selected --
    choosing the mode with nothing loaded opens a file dialog, which never
    returns without a display.
    """
    import cv2

    from idtrackerai.base.animals_detection.external_contours import write_contours

    cap = cv2.VideoCapture(str(clip))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    where.mkdir(parents=True, exist_ok=True)
    return write_contours(
        where / f"{clip.stem}.h5", [[] for _ in range(n_frames)], width, height,
        source="make_screenshots.py", model="none, an empty stand-in",
    )


def main() -> int:
    from idtrackerai.extra_tools.detectron2_pipeline import sampling

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--videos", required=True,
        help="glob for your own clips, e.g. 'D:/footage/*.mp4'",
    )
    parser.add_argument(
        "--example", type=Path,
        default=Path(sampling.__file__).parents[2] / "data" / "test_A.avi",
        help="the well-lit clip used for the picture of thresholding working",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument(
        "--hard-clip", type=Path,
        help="the clip for the picture of thresholding failing; "
             "defaults to the first one matched",
    )
    parser.add_argument(
        "--hard-frame", type=int, default=14912,
        help="a frame of that clip in which the animals are plainly visible",
    )
    args = parser.parse_args()

    images = args.out
    images.mkdir(parents=True, exist_ok=True)

    app = QApplication.instance() or QApplication([])
    for candidate in FONT_CANDIDATES:
        if Path(candidate).is_file():
            fid = QFontDatabase.addApplicationFont(candidate)
            families = QFontDatabase.applicationFontFamilies(fid)
            if fid != -1 and families:
                app.setFont(QFont(families[0], 9))
                break
    else:
        print("No font found; text may render as boxes.", file=sys.stderr)

    from idtrackerai.segmentation_app.main import SegmentationGUI
    from idtrackerai.segmentation_app.widgets.segmentation_source import (
        DETECTRON2,
        THRESHOLDING,
    )

    clips = sampling.resolve_videos([Path(args.videos)])
    if not clips:
        raise SystemExit(f"No videos matched {args.videos}")
    if not args.example.is_file():
        raise SystemExit(f"Example clip not found: {args.example}")
    print(f"{len(clips)} clip(s) of yours; example {args.example.name}")

    gui = SegmentationGUI()
    gui.resize(WIDTH, HEIGHT)
    gui.show()
    settle(app, 20)

    # ---------------------------------------------------------- it working
    print("\nThresholding, on the example clip")
    gui.open_widget.open_video_paths([str(args.example)])
    settle(app, 40)
    # Opening a video restores whatever enhancement that folder was last
    # prepared with, which is right for the app and wrong for a picture of a
    # first run. Start from nothing, as a newcomer would.
    gui.enhancement.setSettings({"enhance": False})
    gui.segmentation_source.source.setCurrentText(THRESHOLDING)
    gui.n_animals.setValue(8)
    gui.intensity_thresholds.th_type.setCurrentText("Dark animals")
    gui.area_thresholds.setValue([150, float("inf")])
    gui.videoPlayer.setCurrentFrame(250)
    settle(app, 60)
    capture(app, gui, images, "app-thresholding.png")

    # ---------------------------------------------------------- it failing
    print("\nThresholding, on real footage it cannot handle")
    hard = args.hard_clip or clips[0]
    print(f"    using {hard.name} frame {args.hard_frame}")
    gui.open_widget.open_video_paths([str(hard)])
    settle(app, 40)
    gui.enhancement.setSettings({"enhance": False})
    gui.n_animals.setValue(5)
    # These fish are paler than the water, so "bright animals" is the honest
    # attempt -- and it finds the reflections instead.
    gui.intensity_thresholds.th_type.setCurrentText("Bright animals")
    gui.area_thresholds.setValue([800, float("inf")])
    gui.videoPlayer.setCurrentFrame(args.hard_frame)
    settle(app, 60)
    capture(app, gui, images, "app-thresholding-hard.png")

    # ---------------------------------------------------------- the panel
    print("\nThe Detectron2 panel")
    gui.segmentation_source.source.setCurrentText(DETECTRON2)
    settle(app, 40)
    panel = gui.detectron2_panel

    # ------------------------------------------------------ the enhancement
    # After the mode switch, not before: thresholding paints a blue overlay
    # over the whole frame on this footage, which buries the very comparison
    # this picture is of. The overlay stands down in Detectron2 mode.
    print("\nEnhancement")
    gui.enhancement.preset.setCurrentText("Standard")
    gui.enhancement.compare.setValue(50)   # the wipe, half raw and half not
    settle(app, 80)
    capture(app, gui, images, "app-enhancement.png")
    gui.enhancement.compare.setValue(100)

    # Every clip from the setup, which is what the panel's list is for.
    panel.set_video_paths(clips)
    wait_until(app, lambda: len(panel._counts) >= len(clips))
    panel.n_frames.setValue(600)
    panel.update_preview()
    panel._update_video_summary()
    panel._update_grouping_status()

    for index, name in (
        (1, "app-detectron2-sample.png"),
        (2, "app-detectron2-annotate.png"),
        (3, "app-detectron2-dataset.png"),
    ):
        panel.steps.setCurrentIndex(index)
        capture(app, gui, images, name)

    panel.steps.setCurrentIndex(4)
    wait_until(app, lambda: panel.gpu.checked)   # the check runs on a thread
    capture(app, gui, images, "app-detectron2-train.png")

    # ------------------------------------------------- tracking on contours
    print("\nExternal contours")
    import tempfile
    # outside the repository: it is scaffolding, not documentation
    contours = stand_in_contours(hard, Path(tempfile.gettempdir()) / "idtrackerai_shots")
    gui.segmentation_source.setValue(contours)
    settle(app, 60)
    capture(app, gui, images, "app-external-contours.png")

    gui.close()
    app.processEvents()
    print(f"\nWritten to {images}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
