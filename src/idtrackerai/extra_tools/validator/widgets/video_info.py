from qtpy.QtWidgets import QFormLayout, QLabel, QListWidget, QVBoxLayout, QWidget

from idtrackerai import Session


class VideoInfo(QWidget):
    """Read-only summary of the session being validated.

    Frame rate and resolution decide whether a jump is impossible or merely
    fast, and a session assembled from several video files needs its file list
    checked before any of its frame numbers mean anything. All of it is in
    `session.json`, but reading that mid-validation means leaving the Validator.
    """

    def __init__(self) -> None:
        super().__init__()
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.fps_label = QLabel("-")
        self.resolution_label = QLabel("-")
        self.frames_label = QLabel("-")
        self.animals_label = QLabel("-")

        form_layout = QFormLayout()
        form_layout.addRow("Frame rate:", self.fps_label)
        form_layout.addRow("Resolution:", self.resolution_label)
        form_layout.addRow("Total frames:", self.frames_label)
        form_layout.addRow("Animals:", self.animals_label)
        layout.addLayout(form_layout)

        layout.addWidget(QLabel("Video files:"))
        self.video_list = QListWidget()
        layout.addWidget(self.video_list)

    def set_data(self, session: Session | None) -> None:
        if session is None:
            return

        self.fps_label.setText(f"{session.frames_per_second} fps")
        self.resolution_label.setText(f"{session.width} x {session.height}")
        self.frames_label.setText(str(session.number_of_frames))
        self.animals_label.setText(str(session.n_animals))

        self.video_list.clear()
        # Sessions written by older versions may not carry video_paths.
        video_paths = getattr(session, "video_paths", None)
        if video_paths:
            for path in video_paths:
                self.video_list.addItem(str(path))
        else:
            self.video_list.addItem("No video paths found")
