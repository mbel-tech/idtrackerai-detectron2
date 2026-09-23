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
    QTimer,
    QUrl,
    Signal,
)
from qtpy.QtGui import QDesktopServices
from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
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
from idtrackerai.extra_tools.detectron2_pipeline import sampling as sampling_mod
from idtrackerai.GUI_tools import WrappedLabel

from .enhancement_widget import EnhancementWidget
from .pipeline_threads import DatasetThread, FrameCountThread, SamplingThread


class Detectron2Panel(QWidget):
    """Walks the user from raw video to a training-ready COCO dataset."""

    needToDraw = Signal()

    def __init__(self, enhancement: EnhancementWidget):
        super().__init__()
        self.enhancement = enhancement
        self.state: prep_state_mod.PrepState | None = None

        # What the Segmentation App currently has open, kept apart from the
        # annotation list so the panel can tell "the user has not touched this"
        # from "the user built this deliberately".
        self._open_videos: list[Path] = []
        self._counts: dict[Path, int] = {}
        self._count_keys: dict[Path, tuple] = {}
        self._pending_counts: set[Path] = set()

        self.sampling_thread = SamplingThread()
        self.dataset_thread = DatasetThread()
        self.count_thread = FrameCountThread()
        self.count_thread.counted.connect(self._counts_ready)
        # anything queued while that run was in flight goes now
        self.count_thread.finished.connect(self._drain_count_queue)

        # Counting frames opens every file, so the preview waits for the
        # spinbox to settle rather than recomputing on each keystroke.
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(200)
        self._preview_timer.timeout.connect(self.update_preview)

        self.steps = QToolBox()
        self.steps.addItem(self._enhancement_page(), "1. Enhancement")
        self.steps.addItem(self._sampling_page(), "2. Sample frames")
        self.steps.addItem(self._annotate_page(), "3. Annotate")
        self.steps.addItem(self._dataset_page(), "4. Build dataset")

        self.handoff = WrappedLabel()
        self.handoff.setText(
            "Training and inference need a GPU: upload the dataset and your "
            "videos to Drive and run the Colab notebook, then come back and "
            "choose 'External contours'."
        )

        layout = QVBoxLayout()
        self.setLayout(layout)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.steps)
        layout.addWidget(self.handoff)

        self._wire_threads()
        self.enhancement.profileSaved.connect(self._profile_saved)
        self.enhancement_committed(self.enhancement.settings())

    # ------------------------------------------------------------------ pages
    def _enhancement_page(self) -> QWidget:
        """The enhancement lives above, as its own control.

        It applies to thresholding as well as to this pipeline, so it is not
        owned by this panel. This page just says where it is and what the
        current choice means for the frames about to be sampled.
        """
        page = QWidget()
        self.enhancement_status = WrappedLabel()
        layout = QVBoxLayout(page)
        layout.addWidget(self.enhancement_status)
        return page

    def _sampling_page(self) -> QWidget:
        page = QWidget()

        # The clips to draw frames from are this panel's own list, not the
        # tracking session's. A model should see every recording from a rig,
        # while a session is the one recording being tracked; opening fifty
        # clips just to annotate them would make one absurd concatenated
        # session out of fifty separate experiments.
        self.video_list = QListWidget()
        self.video_list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        # Three rows is enough to show that a list exists and to scroll it.
        # The page has to leave room for the button that acts on the list, and
        # a step whose own action needs scrolling to reach is a bad step.
        self.video_list.setMaximumHeight(58)
        self.add_videos_button = QPushButton("Add videos...")
        self.add_videos_button.clicked.connect(self.add_videos)
        self.add_folder_button = QPushButton("Add folder...")
        self.add_folder_button.clicked.connect(self.add_video_folder)
        self.remove_videos_button = QPushButton("Remove")
        self.remove_videos_button.clicked.connect(self.remove_selected_videos)
        self.reset_videos_button = QPushButton("Use open videos")
        self.reset_videos_button.clicked.connect(self.reset_videos_to_open)
        self.video_summary = WrappedLabel()

        video_buttons = QHBoxLayout()
        video_buttons.setContentsMargins(0, 0, 0, 0)
        for button in (self.add_videos_button, self.add_folder_button,
                       self.remove_videos_button, self.reset_videos_button):
            video_buttons.addWidget(button)

        self.n_frames = QSpinBox()
        self.n_frames.setRange(1, 100000)
        self.n_frames.setValue(600)
        self.n_frames.valueChanged.connect(self._schedule_preview)
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
        self.sample_preview = WrappedLabel(framed=True)

        # Both numbers share a row: they are small, and the page has to fit
        # the button that acts on them without scrolling.
        self.seed_label = QLabel("Selection number")
        counts_row = QHBoxLayout()
        counts_row.setContentsMargins(0, 0, 0, 0)
        counts_row.addWidget(self.n_frames, 1)
        counts_row.addSpacing(8)
        counts_row.addWidget(self.seed_label)
        counts_row.addWidget(self.seed, 1)
        counts_holder = QWidget()
        counts_holder.setLayout(counts_row)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.addRow("Frames to sample", counts_holder)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.frames_dir)
        row.addWidget(browse)
        holder = QWidget()
        holder.setLayout(row)
        form.addRow("Output folder", holder)

        self.video_list_label = QLabel("Videos to sample from")

        layout = QVBoxLayout(page)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addWidget(self.video_list_label)
        layout.addWidget(self.video_list)
        layout.addLayout(video_buttons)
        layout.addWidget(self.video_summary)
        layout.addLayout(form)
        layout.addWidget(self.sample_preview)
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
        self.refresh_button = QPushButton("Refresh count")
        self.refresh_button.clicked.connect(self.refresh)
        self.annotate_status = WrappedLabel()

        form = QFormLayout()
        form.addRow("Class name", self.class_name)

        row = QHBoxLayout()
        row.addWidget(self.launch_button)
        row.addWidget(self.open_folder_button)
        row.addWidget(self.refresh_button)

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

        # How clips are grouped decides whether the validation score means
        # anything. Segments of one recording show the same animals minutes
        # apart, so splitting between them measures memorisation. The grouping
        # is guessed from file names, which cannot always be right, so it is
        # shown rather than applied quietly.
        self.group_by = QComboBox()
        self.group_by.addItem("By recording (groups segments together)", "recording")
        self.group_by.addItem("By file (each clip on its own)", "file")
        self.group_by.addItem("Random (not recommended)", "random")
        self.group_by.currentIndexChanged.connect(self._group_by_changed)
        self.grouping_status = WrappedLabel(framed=True)

        self.build_button = QPushButton("Build dataset")
        self.build_button.clicked.connect(self.run_dataset)
        self.report_view = QPlainTextEdit()
        self.report_view.setReadOnly(True)
        self.report_view.setMaximumHeight(160)
        self.report_view.setPlaceholderText("the validation report appears here")

        form = QFormLayout()
        form.addRow("Animals per frame", self.expected_instances)
        form.addRow("Validation fraction", self.val_fraction)
        form.addRow("Train/validation split", self.group_by)
        row = QHBoxLayout()
        row.addWidget(self.dataset_dir)
        row.addWidget(browse)
        holder = QWidget()
        holder.setLayout(row)
        form.addRow("Output folder", holder)

        layout = QVBoxLayout(page)
        layout.addLayout(form)
        layout.addWidget(self.grouping_status)
        layout.addWidget(self.build_button)
        layout.addWidget(self.report_view)
        return page

    def _group_by_changed(self) -> None:
        if self.state is not None:
            self.state.group_by = self.group_by.currentData()
            self.state.save()
        self._update_grouping_status()
        self._update_video_summary()

    def _update_grouping_status(self) -> None:
        """Shows the grouping that was inferred, before it is acted on."""
        mode = self.group_by.currentData()
        if mode == "random":
            self.grouping_status.setText(
                "Frames will be split at random. Frames from one clip are near "
                "duplicates, so validation will flatter the model."
            )
            return

        videos = self.video_paths
        if not videos:
            self.grouping_status.setText("Add videos to see how they group.")
            return

        groups = dataset_mod.group_videos(
            [v.stem for v in videos],
            self.state.group_overrides if self.state else None,
            mode,
        )
        shown = ", ".join(
            f"{name} ({len(members)})" for name, members in sorted(groups.items())[:6]
        )
        text = f"{len(videos)} clips in {len(groups)} groups: {shown}"
        text += ", ..." if len(groups) > 6 else "."
        if mode == "recording" and len(groups) == 1 < len(videos):
            text += (
                " That puts every clip in one group, which leaves nothing to "
                "split on, so it will fall back to splitting by file."
            )
        elif len(groups) < 2:
            text += " At least two groups are needed to split on them."
        self.grouping_status.setText(text)

    # ----------------------------------------------------------------- context
    def set_video_context(self, video_paths, output_dir=None) -> None:
        """Called when a video is loaded; finds and reloads any prior work."""
        previously_open = list(self._open_videos)
        self._open_videos = [Path(p) for p in video_paths]
        if not self._open_videos:
            return
        try:
            root = prep_state_mod.prep_root(self._open_videos, output_dir)
        except ValueError:
            return

        self.state = prep_state_mod.PrepState.load(root)
        self.state.refresh()

        # The annotation list belongs to the user, so opening another video
        # must not wipe it. It is re-seeded from the open videos only when it
        # was never curated -- empty, or still exactly what was seeded last
        # time -- and otherwise left alone.
        stored = self.state.video_paths
        if stored and stored != previously_open:
            self.set_video_paths(stored)
        else:
            self.set_video_paths(self._open_videos)

        index = self.group_by.findData(self.state.group_by)
        if index >= 0:
            self.group_by.blockSignals(True)
            self.group_by.setCurrentIndex(index)
            self.group_by.blockSignals(False)

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
            # The folder is on screen already, in the Output folder field. A
            # full absolute path here wraps to three lines and pushes this
            # step's own button off the bottom of the page.
            self.sample_status.setText(f"{s.n_sampled} frames sampled.")
            self.sample_status.setToolTip(str(s.frames_path))
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
    def enhancement_committed(self, settings: dict) -> None:
        """Record what the frames will be prepared with, and say so."""
        self.enhancement_status.setText(
            "Set with the Frame enhancement control above, which applies to "
            "thresholding too.\n\nFrames sampled for annotation will be "
            f"written with: {fp.describe(settings)}"
        )
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

    # ------------------------------------------------------ the video list
    @property
    def video_paths(self) -> list[Path]:
        """The clips frames will be drawn from, in list order."""
        return [
            Path(self.video_list.item(i).data(Qt.ItemDataRole.UserRole))
            for i in range(self.video_list.count())
        ]

    def set_video_paths(self, paths) -> None:
        """Replaces the list, dropping duplicates but keeping the order given."""
        unique: list[Path] = []
        seen = set()
        for path in paths:
            path = Path(path)
            if path not in seen:
                seen.add(path)
                unique.append(path)

        self.video_list.clear()
        for path in unique:
            item = QListWidgetItem(path.name)
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            if path.is_file():
                item.setToolTip(str(path))
            else:
                # kept rather than dropped: an unplugged drive should not
                # quietly empty a list the user spent time building
                item.setText(f"{path.name}  (missing)")
                item.setToolTip(f"{path} - not found right now")
            self.video_list.addItem(item)

        if self.state is not None:
            self.state.set_videos(unique)
            self.state.save()
        self._start_counting(unique)
        self._update_video_summary()
        self._update_grouping_status()
        self._schedule_preview()

    def add_videos(self) -> None:
        start = str(self.video_paths[-1].parent) if self.video_paths else ""
        names, _ = QFileDialog.getOpenFileNames(
            self, "Videos to draw annotation frames from", start,
            "Videos (*.mp4 *.avi *.mov *.mkv *.mpg *.mpeg *.wmv *.m4v);;All files (*)",
        )
        if names:
            self.set_video_paths(self.video_paths + [Path(n) for n in names])

    def add_video_folder(self) -> None:
        start = str(self.video_paths[-1].parent) if self.video_paths else ""
        folder = QFileDialog.getExistingDirectory(
            self, "Folder of videos to draw annotation frames from", start
        )
        if not folder:
            return
        found: list[Path] = []
        for suffix in ("mp4", "avi", "mov", "mkv", "mpg", "mpeg", "wmv", "m4v"):
            # resolve_videos drops the macOS "._" companions that otherwise
            # arrive as unreadable videos
            found += sampling_mod.resolve_videos([Path(folder) / f"*.{suffix}"])
        if not found:
            QMessageBox.information(
                self, "No videos there", f"No video files were found in {folder}."
            )
            return
        self.set_video_paths(self.video_paths + sorted(found))

    def remove_selected_videos(self) -> None:
        rows = {index.row() for index in self.video_list.selectedIndexes()}
        if not rows:
            QMessageBox.information(
                self, "Nothing selected", "Select the videos to remove first."
            )
            return
        keep = [p for i, p in enumerate(self.video_paths) if i not in rows]
        self.set_video_paths(keep)

    def reset_videos_to_open(self) -> None:
        self.set_video_paths(self._open_videos)

    def _start_counting(self, videos) -> None:
        """Queues anything new or changed for counting.

        Videos are queued rather than counted directly, because adding a
        second folder while the first is still being counted would otherwise
        mark the new clips as seen and never count them, leaving the preview
        waiting for numbers that were never coming.
        """
        for video in videos:
            try:
                stat = video.stat()
                key = (stat.st_mtime, stat.st_size)
            except OSError:
                self._counts[video] = 0
                self._count_keys.pop(video, None)
                continue
            if self._count_keys.get(video) != key:
                self._count_keys[video] = key
                self._pending_counts.add(video)
        self._drain_count_queue()

    def _drain_count_queue(self) -> None:
        if not self._pending_counts or self.count_thread.isRunning():
            return
        batch = sorted(self._pending_counts)
        self._pending_counts.clear()
        self.count_thread.set_parameters(batch)
        self.count_thread.start()

    def _counts_ready(self, counts: dict) -> None:
        self._counts.update(counts)
        self._update_video_summary()
        self.update_preview()

    def _update_video_summary(self) -> None:
        videos = self.video_paths
        if not videos:
            self.video_summary.setText(
                "No videos yet. Add the clips you want the model trained on - "
                "every recording from this setup, not just the one being tracked."
            )
            return

        groups = dataset_mod.group_videos(
            [v.stem for v in videos],
            self.state.group_overrides if self.state else None,
            self.state.group_by if self.state else "recording",
        )
        known = [self._counts.get(v) for v in videos]
        counted = [c for c in known if c]
        parts = [f"{len(videos)} videos", f"{len(groups)} recordings"]
        if len(counted) == len(videos):
            parts.append(f"{sum(counted):,} frames")
        else:
            parts.append(f"counting frames ({len(counted)}/{len(videos)})")

        missing = [v for v in videos if not v.is_file()]
        if missing:
            parts.append(f"{len(missing)} not found")
        text = ", ".join(parts) + "."

        unreadable = [v for v in videos if v.is_file() and self._counts.get(v) == 0]
        if unreadable:
            text += (
                f" {len(unreadable)} could not be read and will be skipped: "
                + ", ".join(v.name for v in unreadable[:3])
                + ("..." if len(unreadable) > 3 else "")
            )
        self.video_summary.setText(text)

    def _schedule_preview(self) -> None:
        self._preview_timer.start()

    def update_preview(self) -> None:
        """Says what would be sampled, before anything is written."""
        videos = self.video_paths
        if not videos:
            self.sample_preview.setText("")
            return

        counts = [self._counts.get(v) for v in videos]
        if any(c is None for c in counts):
            self.sample_preview.setText("Counting frames...")
            return

        try:
            plans = sampling_mod.plan_sampling(
                videos, self.n_frames.value(), self.seed.value(), counts=counts
            )
        except Exception as exc:  # noqa: BLE001 - a preview must never raise
            self.sample_preview.setText(f"Cannot plan this yet: {exc}")
            return

        taking = [p for p in plans if p.requested]
        empty = [p for p in plans if p.readable and not p.requested]
        shares = sorted(p.requested for p in taking)
        text = (
            f"{sum(p.requested for p in plans)} frames from {len(taking)} of "
            f"{len(plans)} videos"
        )
        if shares:
            text += (
                f", {shares[0]} to {shares[-1]} each"
                if shares[0] != shares[-1]
                else f", {shares[0]} each"
            )
        text += "."
        if empty:
            # the whole point of a collection is that every clip is seen, so
            # this is worth saying loudly rather than leaving to be discovered
            text += (
                f" {len(empty)} videos would get no frames at all - raise the"
                " count to at least one per video."
            )
        self.sample_preview.setText(text)

    # ----------------------------------------------------------------- step 2
    def _choose_frames_dir(self) -> None:
        name = QFileDialog.getExistingDirectory(
            self, "Folder for the sampled frames", self.frames_dir.text()
        )
        if name:
            self.frames_dir.setText(name)

    def run_sampling(self) -> None:
        videos = self.video_paths
        if not videos:
            QMessageBox.warning(
                self, "No videos",
                "Add the videos to draw annotation frames from first.",
            )
            return
        output = Path(self.frames_dir.text())

        gone = [v for v in videos if not v.is_file()]
        if gone:
            QMessageBox.warning(
                self, "Videos not found",
                f"{len(gone)} of the listed videos cannot be found right now, "
                "so no frames can be taken from them:\n\n"
                + "\n".join(str(v) for v in gone[:8])
                + ("\n..." if len(gone) > 8 else "")
                + "\n\nReconnect the drive, or remove them from the list.",
            )
            return

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

        counts = [self._counts.get(v) for v in videos]
        self.sampling_thread.set_parameters(
            videos,
            output,
            self.n_frames.value(),
            self.enhancement.settings(),
            seed=self.seed.value(),
            counts=None if any(c is None for c in counts) else counts,
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
        if result.unreadable:
            # computed by the sampler and, until now, thrown away
            QMessageBox.warning(
                self,
                "Some videos were skipped",
                f"{len(result.unreadable)} video(s) could not be read and "
                "contributed no frames:\n\n"
                + "\n".join(result.unreadable[:10])
                + ("\n..." if len(result.unreadable) > 10 else ""),
            )
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
        mode = self.group_by.currentData()
        request = dataset_mod.DatasetRequest(
            input_dir=frames,
            output_dir=Path(self.dataset_dir.text()),
            val_fraction=self.val_fraction.value(),
            single_class=self.class_name.text().strip() or None,
            expected_instances=expected or None,
            seed=self.seed.value(),
            split_by="random" if mode == "random" else "video",
            group_by="file" if mode == "file" else "recording",
            group_overrides=dict(self.state.group_overrides) if self.state else None,
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
            lines.append(f"  train groups: {report.train_videos}")
            lines.append(f"  val groups:   {report.val_videos}")
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

    # --------------------------------------------------------------- tooltips
    def setToolTips(self, tips: dict) -> None:
        """Explain every control, since none of this is self-evident."""
        pairs = [
            (self.video_list, "d2_video_list"),
            (self.video_list_label, "d2_video_list"),
            (self.add_videos_button, "d2_add_videos"),
            (self.add_folder_button, "d2_add_folder"),
            (self.remove_videos_button, "d2_remove_videos"),
            (self.reset_videos_button, "d2_reset_videos"),
            (self.video_summary, "d2_video_list"),
            (self.n_frames, "d2_n_frames"),
            (self.seed, "d2_seed"),
            (self.seed_label, "d2_seed"),
            (self.sample_preview, "d2_preview"),
            (self.frames_dir, "d2_frames_folder"),
            (self.sample_button, "d2_n_frames"),
            (self.class_name, "d2_class_name"),
            (self.launch_button, "d2_labelme"),
            (self.open_folder_button, "d2_open_folder"),
            (self.refresh_button, "d2_refresh"),
            (self.expected_instances, "d2_expected"),
            (self.val_fraction, "d2_val_fraction"),
            (self.group_by, "d2_group_by"),
            (self.grouping_status, "d2_group_by"),
            (self.dataset_dir, "d2_dataset_folder"),
            (self.build_button, "d2_build"),
            (self.enhancement_status, "enhancement"),
        ]
        for widget, key in pairs:
            if key in tips:
                widget.setToolTip(tips[key])


    # ----------------------------------------------------------------- closing
    def close(self) -> bool:
        """Stops any running work and flushes the state before the app exits."""
        for thread in (self.sampling_thread, self.dataset_thread, self.count_thread):
            if thread.isRunning():
                thread.quit()
                thread.wait(5000)
        if self.state is not None:
            self.state.save()
        return super().close()
