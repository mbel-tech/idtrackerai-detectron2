"""Background workers for the Detectron2 preparation steps.

Both follow the shape :class:`BkgComputationThread` established in
``bkg_widget.py``: a plain ``QThread`` subclass, parameters pushed in before
``start()``, progress reported through signals handed straight to the worker
function, and ``quit()`` overridden to set a flag rather than stop the thread,
so cancelling is cooperative.

Following the deliberate convention documented at ``bkg_widget.py:60``, work
that can raise is kept out of ``run()`` where possible; what cannot be is
caught here and reported through a ``failed`` signal, because an exception
escaping a Qt slot aborts the process.
"""

import logging
from pathlib import Path

from qtpy.QtCore import QThread, Signal  # type: ignore[reportPrivateImportUsage]

from idtrackerai.extra_tools.detectron2_pipeline import dataset as dataset_mod
from idtrackerai.extra_tools.detectron2_pipeline import sampling as sampling_mod
from idtrackerai.extra_tools.detectron2_pipeline.errors import PipelineError


class SamplingThread(QThread):
    """Extracts frames for annotation without blocking the UI."""

    set_progress_value = Signal(int)
    set_progress_max = Signal(int)
    failed = Signal(str)

    def __init__(self):
        super().__init__()
        self.result = None
        self.abort = False

    def set_parameters(
        self,
        videos: list[Path],
        output: Path,
        n_frames: int,
        settings: dict,
        seed: int = 0,
        image_format: str = "png",
    ) -> None:
        self.videos = videos
        self.output = output
        self.n_frames = n_frames
        self.settings = settings
        self.seed = seed
        self.image_format = image_format
        self.result = None
        self.abort = False

    def quit(self):
        # Overridden deliberately, as BkgComputationThread does: this sets a
        # flag the worker checks, so a cancel finishes the current frame
        # rather than tearing the thread down mid-write.
        self.abort = True

    def run(self):
        self.set_progress_max.emit(self.n_frames)
        try:
            self.result = sampling_mod.sample_frames(
                self.videos,
                self.output,
                self.n_frames,
                self.settings,
                seed=self.seed,
                image_format=self.image_format,
                progress=self.set_progress_value.emit,
                abort=lambda: self.abort,
            )
        except PipelineError as exc:
            self.result = None
            self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001
            self.result = None
            logging.exception("Frame sampling failed")
            self.failed.emit(str(exc))


class DatasetThread(QThread):
    """Validates annotations and builds the COCO dataset."""

    set_progress_value = Signal(int)
    set_progress_max = Signal(int)
    failed = Signal(str)

    def __init__(self):
        super().__init__()
        self.result = None
        self.abort = False

    def set_parameters(self, request) -> None:
        self.request = request
        self.result = None
        self.abort = False

    def quit(self):
        self.abort = True

    def run(self):
        try:
            n = len(list(self.request.input_dir.glob("*.json")))
            self.set_progress_max.emit(max(n, 1))
            self.result = dataset_mod.build_coco_dataset(
                self.request,
                progress=self.set_progress_value.emit,
                abort=lambda: self.abort,
            )
        except PipelineError as exc:
            self.result = None
            if not self.abort:
                self.failed.emit(str(exc))
        except Exception as exc:  # noqa: BLE001
            self.result = None
            logging.exception("Dataset build failed")
            self.failed.emit(str(exc))
