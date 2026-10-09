"""Regression tests for the Detectron2 preparation pipeline's review fixes."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from idtrackerai.extra_tools.detectron2_pipeline import dataset, prep_state, sampling
from idtrackerai.extra_tools.detectron2_pipeline.errors import DatasetError, SamplingError
from idtrackerai.extra_tools.detectron2_pipeline.preprocessing import DEFAULT_SETTINGS


def _png(path: Path, shade: int = 0) -> None:
    cv2.imencode(".png", np.full((20, 20, 3), shade, np.uint8))[1].tofile(str(path))


def _labelme(folder: Path, name: str, image_path: str | None = None) -> None:
    _png(folder / f"{name}.png")
    data = {
        "imageWidth": 20,
        "imageHeight": 20,
        "shapes": [
            {"label": "fish", "shape_type": "polygon",
             "points": [[1, 1], [15, 1], [15, 15], [1, 15]]}
        ],
    }
    if image_path is not None:
        data["imagePath"] = image_path
    (folder / f"{name}.json").write_text(json.dumps(data), encoding="utf-8")


def test_dataset_refreshes_and_prunes_images(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    _labelme(src, "a_f000001")
    _labelme(src, "b_f000001")
    request = dataset.DatasetRequest(src, out, val_fraction=0.5)
    dataset.build_coco_dataset(request)
    assert {p.name for p in (out / "images").iterdir()} == {"a_f000001.png", "b_f000001.png"}

    # a changed source image is re-copied; a dropped annotation is pruned
    _png(src / "a_f000001.png", shade=200)
    (src / "b_f000001.json").unlink()
    dataset.build_coco_dataset(request)
    assert [p.name for p in (out / "images").iterdir()] == ["a_f000001.png"]
    copied = cv2.imdecode(
        np.fromfile(str(out / "images" / "a_f000001.png"), np.uint8), cv2.IMREAD_COLOR
    )
    assert copied.max() == 200


def test_prep_state_stores_posix_and_reads_old_backslashes(tmp_path):
    state = prep_state.PrepState(root=tmp_path)
    state.set_frames_dir(tmp_path / "frames" / "set1")
    assert state.frames_dir == "frames/set1"

    state.dataset_dir = "dataset" + chr(92) + "v1"  # as an older Windows run wrote it
    assert state.save()
    loaded = prep_state.PrepState.load(tmp_path)
    assert loaded.dataset_dir == "dataset/v1"


def test_duplicate_video_names_use_stem(tmp_path, capsys, monkeypatch):
    from idtrackerai.extra_tools.detectron2_pipeline import videos

    for ext in (".mp4", ".avi"):
        (tmp_path / f"clip{ext}").write_bytes(b"not a video")
    monkeypatch.setattr(
        "sys.argv", ["check", "--videos", str(tmp_path / "clip.mp4"), str(tmp_path / "clip.avi")]
    )
    with pytest.raises(SystemExit):  # unreadable fake videos -> exit 1 after the report
        videos.main()
    assert "repeated file names ['clip']" in capsys.readouterr().out


@pytest.mark.parametrize(
    "image_path",
    ["C:" + chr(92) + "Users" + chr(92) + "me" + chr(92) + "frames" + chr(92) + "x_f000002.png",
     "../frames/x_f000002.png", "x_f000002.png", None, ""],
)
def test_labelme_image_path_is_normalised(tmp_path, image_path):
    _labelme(tmp_path, "x_f000002", image_path)
    data = json.loads((tmp_path / "x_f000002.json").read_text())
    assert dataset._image_name_of(data, tmp_path / "x_f000002.json") == "x_f000002.png"


def test_corrupt_manifest_raises_dataset_error(tmp_path):
    _labelme(tmp_path, "x_f000002")
    (tmp_path / "sampling_manifest.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(DatasetError):
        dataset.build_coco_dataset(dataset.DatasetRequest(tmp_path, tmp_path / "out"))


def test_group_split_skips_oversized_first_group():
    per_image = [{"group": "big", "source_video": "big"} for _ in range(20)]
    per_image += [{"group": f"s{i}", "source_video": f"s{i}"} for i in range(10)]
    request = dataset.DatasetRequest(Path("."), Path("."), val_fraction=0.1, seed=3)
    for seed in range(10):
        request.seed = seed
        _, val = dataset._split(per_image, request, dataset.DatasetReport())
        assert len(val) <= 6, seed


def test_resampling_with_different_enhancement_is_refused(tmp_path):
    sampling.save_profile(tmp_path / "preprocess_profile.json", dict(DEFAULT_SETTINGS))
    same = dict(DEFAULT_SETTINGS)
    sampling.check_enhancement_unchanged(tmp_path, same)  # no error
    other = dict(DEFAULT_SETTINGS, clahe_clip=3.0)
    with pytest.raises(SamplingError, match="clahe_clip"):
        sampling.check_enhancement_unchanged(tmp_path, other)
