"""The guided panel for preparing a Detectron2 model.

Four local steps: choose the enhancement, sample frames, annotate them, build
the dataset. Training and inference are deliberately absent — they need a GPU
and are driven from the Colab notebook — so the panel ends by saying what to
take there.

Each step records what it produced, so the app can be closed in the middle of
a job that takes days. Status is re-derived from disk on every load rather than
trusted from the state file, so a deleted folder shows as incomplete.
"""

import logging
import sys
from pathlib import Path

from qtpy.QtCore import (  # type: ignore[reportPrivateImportUsage]
    QProcess,
    QProcessEnvironment,
    Qt,
    QUrl,
    Signal,
)
from qtpy.QtGui import QDesktopServices
from qtpy.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressDialog,
    QPushButton,
    QSpinBox,
    QToolBox,
    QVBoxLayout,
    QWidget,
)

from idtrackerai.extra_tools.detectron2_pipeline import dataset as dataset_mod
from idtrackerai.extra_tools.detectron2_pipeline import prep_state as prep_state_mod
from idtrackerai.extra_tools.detectron2_pipeline import preprocessing as fp
from idtrackerai.GUI_tools import WrappedLabel

from .enhancement_widget import EnhancementWidget
from .pipeline_threads import DatasetThread, SamplingThread


class Detectron2Panel(QWidget):
    """Walks the user from raw video to a training-ready COCO dataset."""

    needToDraw = Signal()

    def __init__(self, enhancement: EnhancementWidget):
        super().__init__()
        self.enhancement = enhancement
        self.state: prep_state_mod.PrepState | None = None
        self.video_paths: list[Path] = []

        self.sampling_thread = SamplingThread()
        self.dataset_thread = DatasetThread()

        self.steps = QToolBox()
        self.steps.addItem(self._enhancement_page(), "1. Enhancement")
        self.steps.addItem(self._sampling_page(), "2. Sample frames")
        self.steps.addItem(self._annotate_page(), "3. Annotate")
        self.steps.addItem(self._dataset_page(), "4. Build dataset")

        self.handoff = WrappedLabel()
        self.handoff.setText(
            "Training and inference need a GPU and run in "
            "tools/colab_detectron2_pipeline.ipynb. Upload the dataset folder "
            "and your videos to Drive, then come back and choose "
            "'External contours' to track with the result."
        )

        layout = QVBoxLayout()
        self.setLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.steps)
        layout.addWidget(self.handoff)

        self._wire_threads()
        self.enhancement.settingsCommitted.connect(self._enhancement_committed)
        self.enhancement.profileSaved.connect(self._profile_saved)

    # ------------------------------------------------------------------ pages
    def _enhancement_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.enhancement)
        return page

    def _sampling_page(self) -> QWidget:
        page = QWidget()
        self.n_frames = QSpinBox()
        self.n_frames.setRange(1, 100000)
        self.n_frames.setValue(600)
        self.seed = QSpinBox()
        self.seed.setRange(0, 999999)
        self.frames_dir = QLineEdit()
        self.frames_dir.setPlaceholderText("chosen when you run this step")
        self.frames_dir.setReadOnly(True)
        browse = QPushButton("Choose folder...")
        browse.clicked.connect(self._choose_frames_dir)

        self.sample_button = QPushButton("Sample frames")
        self.sample_button.clicked.connect(self.run_sampling)
        self.sample_status = WrappedLabel()

        form = QFormLayout()
        form.addRow("Frames to sample", self.n_frames)
        form.addRow("Seed", self.seed)
        row = QHBoxLayout()
        row.addWidget(self.frames_dir)
        row.addWidget(browse)
        holder = QWidget()
        holder.setLayout(row)
        form.addRow("Output folder", holder)

        layout = QVBoxLayout(page)
        layout.addLayout(form)
        layout.addWidget(self.sample_button)
        layout.addWidget(self.sample_status)
        return page

    def _annotate_page(self) -> QWidget:
        page = QWidget()
        self.class_name = QLineEdit("fish")
        self.launch_button = QPushButton("Open in LabelMe")
        self.launch_button.clicked.connect(self.launch_labelme)
        self.open_folder_button = QPushButton("Open folder")
        self.open_folder_button.clicked.connect(self.open_frames_folder)
        refresh = QPushButton("Refresh count")
        refresh.clicked.connect(self.refresh)
        self.annotate_status = WrappedLabel()

        form = QFormLayout()
        form.addRow("Class name", self.class_name)

        row = QHBoxLayout()
        row.addWidget(self.launch_button)
        row.addWidget(self.open_folder_button)
        row.addWidget(refresh)

        layout = QVBoxLayout(page)
        layout.addLayout(form)
        layout.addLayout(row)
        layout.addWidget(self.annotate_status)
        return page

    def _dataset_page(self) -> QWidget:
        page = QWidget()
        self.expected_instances = QSpinBox()
        self.expected_instances.setRange(0, 1000)
        self.expected_instances.setSpecialValueText("any")
        self.val_fraction = QDoubleSpinBox()
        self.val_fraction.setRange(0.0, 0.9)
        self.val_fraction.setSingleStep(0.05)
        self.val_fraction.setValue(0.15)
        self.dataset_dir = QLineEdit()
        self.dataset_dir.setReadOnly(True)
        self.dataset_dir.setPlaceholderText("chosen when you run this step")
        browse = QPushButton("Choose folder...")
        browse.clicked.connect(self._choose_dataset_dir)

        self.build_button = QPushButton("Build dataset")
        self.build_button.clicked.connect(self.run_dataset)
        self.report_view = QPlainTextEdit()
        self.report_view.setReadOnly(True)
        self.report_view.setMaximumHeight(160)
        self.report_view.setPlaceholderText("the validation report appears here")

        form = QFormLayout()
        form.addRow("Animals per frame", self.expected_instances)
        form.addRow("Validation fraction", self.val_fraction)
        row = QHBoxLayout()
        row.addWidget(self.dataset_dir)
        row.addWidget(browse)
        holder = QWidget()
        holder.setLayout(row)
        form.addRow("Output folder", holder)

        layout = QVBoxLayout(page)
        layout.addLayout(form)
        layout.addWidget(self.build_button)
        layout.addWidget(self.report_view)
        return page

    # ----------------------------------------------------------------- context
    def set_video_context(self, video_paths, output_dir=None) -> None:
        """Called when a video is loaded; finds and reloads any prior work."""
        self.video_paths = [Path(p) for p in video_paths]
        if not self.video_paths:
            return
        try:
            root = prep_state_mod.prep_root(self.video_paths, output_dir)
        except ValueError:
            return

        self.state = prep_state_mod.PrepState.load(root)
        self.state.refresh()

        if self.state.enhancement:
            self.enhancement.setSettings(self.state.enhancement)
        if self.state.frames_path:
            self.frames_dir.setText(str(self.state.frames_path))
        else:
            self.frames_dir.setText(str(root / "annotate"))
        if self.state.dataset_path:
            self.dataset_dir.setText(str(self.state.dataset_path))
        else:
            self.dataset_dir.setText(str(root / "dataset"))

        self.refresh()

    def refresh(self) -> None:
        """Re-derives every step's status from disk and updates the headers."""
        if self.state is None:
            return
        self.state.refresh()
        s = self.state

        self.steps.setItemText(
            0, "1. Enhancement" + (" - done" if s.profile else "")
        )
        if s.sampling_done:
            self.sample_status.setText(
                f"{s.n_sampled} frames in {s.frames_path}"
            )
            self.steps.setItemText(1, f"2. Sample frames - done ({s.n_sampled})")
        else:
            self.sample_status.setText("No frames sampled yet.")
            self.steps.setItemText(1, "2. Sample frames")

        if s.n_sampled:
            self.annotate_status.setText(
                f"{s.annotated} of {s.n_sampled} frames annotated."
            )
            mark = " - done" if s.annotation_done else ""
            self.steps.setItemText(2, f"3. Annotate{mark} ({s.annotated}/{s.n_sampled})")
        else:
            self.annotate_status.setText("Sample some frames first.")
            self.steps.setItemText(2, "3. Annotate")

        if s.dataset_done:
            stale = " - stale, rebuild" if s.dataset_is_stale() else " - done"
            self.steps.setItemText(
                3, f"4. Build dataset{stale} ({s.n_train}/{s.n_val})"
            )
        else:
            self.steps.setItemText(3, "4. Build dataset")

        self.launch_button.setEnabled(bool(s.frames_path and s.n_sampled))
        self.open_folder_button.setEnabled(bool(s.frames_path and s.frames_path.is_dir()))
        self.build_button.setEnabled(s.annotated > 0)

    # ----------------------------------------------------------------- step 1
    def _enhancement_committed(self, settings: dict) -> None:
        if self.state is None:
            return
        self.state.enhancement = settings
        self.state.save()

    def _profile_saved(self, path) -> None:
        if self.state is None:
            return
        self.state.set_profile(Path(path))
        self.state.save()
        self.refresh()

    # ----------------------------------------------------------------- step 2
    def _choose_frames_dir(self) -> None:
        name = QFileDialog.getExistingDirectory(
            self, "Folder for the sampled frames", self.frames_dir.text()
        )
        if name:
            self.frames_dir.setText(name)

    def run_sampling(self) -> None:
        if not self.video_paths:
            QMessageBox.warning(self, "No video", "Open a video first.")
            return
        output = Path(self.frames_dir.text())

        existing = list(output.glob("*.png")) + list(output.glob("*.jpg")) if output.is_dir() else []
        if existing:
            answer = QMessageBox.question(
                self,
                "Folder is not empty",
                f"{output} already holds {len(existing)} image(s).\n\n"
                "Sampling again adds to them, and any annotations you have "
                "already made are kept. Continue?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        self.sampling_thread.set_parameters(
            self.video_paths,
            output,
            self.n_frames.value(),
            self.enhancement.settings(),
            seed=self.seed.value(),
        )
        self.sample_button.setEnabled(False)
        self.sampling_thread.start()

    def _sampling_finished(self) -> None:
        self.sample_button.setEnabled(True)
        result = self.sampling_thread.result
        if result is None:
            self.refresh()
            return
        if self.state is not None:
            self.state.set_frames_dir(result.output)
            self.state.set_profile(result.profile_path)
            self.state.n_sampled = result.written
            self.state.sampling_complete = result.complete
            self.state.enhancement = self.enhancement.settings()
            self.state.save()
        if not result.complete:
            QMessageBox.information(
                self,
                "Sampling cancelled",
                f"{result.written} frames were written and kept. Run the step "
                "again to add more.",
            )
        self.refresh()
        self.steps.setCurrentIndex(2)

    # ----------------------------------------------------------------- step 3
    def launch_labelme(self) -> None:
        folder = self.state.frames_path if self.state else None
        if folder is None or not folder.is_dir():
            QMessageBox.warning(self, "No frames", "Sample some frames first.")
            return

        arguments = ["-m", "labelme", str(folder)]
        label = self.class_name.text().strip()
        if label:
            # Pre-loading the label and validating it exactly turns the
            # label-typo problem the converter can only report into one that
            # cannot happen.
            arguments += ["--labels", label, "--validate-label", "exact"]

        process = QProcess(self)
        environment = QProcessEnvironment.systemEnvironment()
        # LabelMe uses PySide6; this application pins QT_API for its own qtpy.
        # Do not let ours leak into it.
        environment.remove("QT_API")
        process.setProcessEnvironment(environment)
        process.setProgram(sys.executable)
        process.setArguments(arguments)

        started, _pid = process.startDetached()
        if not started:
            QMessageBox.critical(
                self,
                "Could not start LabelMe",
                "Tried to run:\n\n"
                f"  {sys.executable} {' '.join(arguments)}\n\n"
                "LabelMe is a dependency of this package, so this usually means "
                "the environment is broken. Check that "
                "'python -m labelme --help' works.",
            )
            return
        logging.info("Launched LabelMe on %s", folder)

    def open_frames_folder(self) -> None:
        folder = self.state.frames_path if self.state else None
        if folder is None or not folder.is_dir():
            QMessageBox.warning(self, "No frames", "Sample some frames first.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    # ----------------------------------------------------------------- step 4
    def _choose_dataset_dir(self) -> None:
        name = QFileDialog.getExistingDirectory(
            self, "Folder for the dataset", self.dataset_dir.text()
        )
        if name:
            self.dataset_dir.setText(name)

    def run_dataset(self) -> None:
        frames = self.state.frames_path if self.state else None
        if frames is None or not frames.is_dir():
            QMessageBox.warning(self, "No frames", "Sample some frames first.")
            return

        expected = self.expected_instances.value()
        request = dataset_mod.DatasetRequest(
            input_dir=frames,
            output_dir=Path(self.dataset_dir.text()),
            val_fraction=self.val_fraction.value(),
            single_class=self.class_name.text().strip() or None,
            expected_instances=expected or None,
        )
        self.dataset_thread.set_parameters(request)
        self.build_button.setEnabled(False)
        self.dataset_thread.start()

    def _dataset_finished(self) -> None:
        self.build_button.setEnabled(True)
        report = self.dataset_thread.result
        if report is None:
            self.refresh()
            return
        if self.state is not None:
            self.state.set_dataset_dir(report.output_dir)
            self.state.n_train = report.train_images
            self.state.n_val = report.val_images
            self.state.save()
        self.report_view.setPlainText(self._format_report(report))
        self.refresh()

    @staticmethod
    def _format_report(report) -> str:
        lines = [
            f"Categories: {[c['name'] for c in report.categories]}",
            f"Train: {report.train_images} images, {report.train_annotations} instances",
            f"Val:   {report.val_images} images, {report.val_annotations} instances",
        ]
        if report.train_videos or report.val_videos:
            lines.append(f"  train videos: {report.train_videos}")
            lines.append(f"  val videos:   {report.val_videos}")
        if report.enhancement:
            lines.append(f"Enhancement carried forward: {fp.describe(report.enhancement)}")
        else:
            lines.append(
                "WARNING: no enhancement record; training will not know how "
                "these frames were prepared."
            )
        if report.problems:
            lines.append(f"\n{len(report.problems)} problem(s):")
            lines += [f"  {p}" for p in report.problems[:20]]
        if report.notes:
            lines.append(f"\n{len(report.notes)} note(s):")
            lines += [f"  {n}" for n in report.notes[:20]]
        return "\n".join(lines)

    # ---------------------------------------------------------------- threads
    def _wire_threads(self) -> None:
        for thread, finished in (
            (self.sampling_thread, self._sampling_finished),
            (self.dataset_thread, self._dataset_finished),
        ):
            thread.started.connect(lambda t=thread: self._thread_started(t))
            thread.finished.connect(self._thread_finished)
            thread.finished.connect(finished)
            thread.set_progress_value.connect(self._set_progress_value)
            thread.set_progress_max.connect(self._set_progress_max)
            thread.failed.connect(self._thread_failed)

    def _thread_started(self, thread) -> None:
        # Created here rather than before start(), as BkgWidget does, so a run
        # that finishes instantly never flashes a dialog.
        self.progress = QProgressDialog("Working...", "Cancel", 0, 100, self)
        self.progress.setMinimumDuration(200)
        self.progress.setModal(True)
        self.progress.setAutoReset(False)
        self.progress.canceled.connect(thread.quit)

    def _thread_finished(self) -> None:
        if hasattr(self, "progress"):
            self.progress.reset()

    def _set_progress_value(self, value: int) -> None:
        if hasattr(self, "progress"):
            self.progress.setValue(value)

    def _set_progress_max(self, value: int) -> None:
        if hasattr(self, "progress"):
            self.progress.setMaximum(value)

    def _thread_failed(self, message: str) -> None:
        QMessageBox.critical(self, "Step failed", message)

    # ----------------------------------------------------------------- closing
    def close(self) -> bool:
        """Stops any running work and flushes the state before the app exits."""
        for thread in (self.sampling_thread, self.dataset_thread):
            if thread.isRunning():
                thread.quit()
                thread.wait(5000)
        if self.state is not None:
            self.state.save()
        return super().close()
