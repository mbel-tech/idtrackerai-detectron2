"""Fine-tunes a Mask R-CNN on the COCO dataset produced by the dataset step.

Reads the number of classes from the annotations rather than taking it as a
flag: a NUM_CLASSES that disagrees with the data is one of the easiest ways to
get a model that trains without error and predicts nothing useful.

Alongside the weights it writes ``training_metadata.json``, recording the class
names and the frame-enhancement settings the data was prepared with.
detectron2_export_contours.py reads that file and warns if inference is about to
run with different enhancement than training used.

Usage
-----
    python train_detectron2.py --dataset dataset/ --output model/ --epochs 40

Install (Colab, CUDA runtime)::

    pip install 'torch>=2.1' torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'

Detectron2 has no universal wheel; it builds against the installed torch/CUDA
pair, so install torch first and let detectron2 compile against it.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

try:
    from .errors import DatasetError
except ImportError:  # loaded by path, without the package around it
    from errors import DatasetError  # type: ignore[no-redef]


def _load_coco(annotation_file: Path, key: str):
    """One top-level key of a COCO file, or a DatasetError saying what is wrong."""
    try:
        data = json.loads(annotation_file.read_text(encoding="utf-8"))
        return data[key]
    except (OSError, ValueError) as exc:
        raise DatasetError(f"{annotation_file} could not be read as JSON: {exc}")
    except (KeyError, TypeError):
        raise DatasetError(f"{annotation_file} has no {key!r} entry; rebuild the dataset")


def read_categories(annotation_file: Path) -> list[str]:
    try:
        categories = sorted(_load_coco(annotation_file, "categories"),
                            key=lambda c: c["id"])
        names = [c["name"] for c in categories]
    except (KeyError, TypeError):
        raise DatasetError(
            f"{annotation_file} has categories without an id and name; rebuild the dataset"
        )
    ids = [c["id"] for c in categories]
    if ids != list(range(1, len(ids) + 1)):
        raise DatasetError(
            f"{annotation_file} has category ids {ids}; they must run 1..N with no"
            " gaps for Detectron2's contiguous mapping to line up."
        )
    return names


def count_images(annotation_file: Path) -> int:
    return len(_load_coco(annotation_file, "images"))


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Mask R-CNN for instance segmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset", type=Path, required=True,
        help="the dataset folder built by the Segmentation App or idtrackerai_d2_dataset",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config",
        default="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        help="model_zoo config to start from",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="passes over the training set; converted to iterations internally",
    )
    parser.add_argument(
        "--iterations", type=int, help="explicit iteration count, overriding --epochs"
    )
    parser.add_argument("--batch-size", type=int, default=2, help="images per iteration")
    parser.add_argument(
        "--lr",
        type=float,
        default=0.00025,
        help="base learning rate; scale with batch size (0.00025 suits batch 2)",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--min-size-train", type=int, nargs="+", default=[640, 672, 704, 736, 768])
    parser.add_argument(
        "--min-size-test",
        type=int,
        default=640,
        help="must match the value used at inference",
    )
    parser.add_argument(
        "--eval-period",
        type=int,
        default=0,
        help="evaluate every N iterations during training (0 = only at the end)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--check-dataset",
        action="store_true",
        help="write a few annotated samples as images and exit without training",
    )
    args = parser.parse_args()

    train_json = args.dataset / "train.json"
    val_json = args.dataset / "val.json"
    images_dir = args.dataset / "images"
    for path in (train_json, val_json, images_dir):
        if not path.exists():
            raise SystemExit(
                f"Missing {path}. Build the dataset first, in the "
                "Segmentation App or with idtrackerai_d2_dataset, and "
                "upload the whole folder."
            )

    try:
        class_names = read_categories(train_json)
        n_train = count_images(train_json)
        n_val = count_images(val_json)
    except DatasetError as exc:
        raise SystemExit(str(exc))
    if n_train == 0:
        raise SystemExit("The training set is empty.")

    iterations = args.iterations or max(
        1, (n_train * args.epochs) // max(args.batch_size, 1)
    )

    print(f"Classes ({len(class_names)}): {class_names}")
    print(f"Train {n_train} images, val {n_val} images")
    print(
        f"{iterations} iterations at batch {args.batch_size}"
        f" ~ {iterations * args.batch_size / max(n_train, 1):.1f} epochs"
    )

    # Imported here so --help and the checks above work without detectron2.
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.data.datasets import register_coco_instances
    from detectron2.engine import DefaultTrainer
    from detectron2.evaluation import COCOEvaluator
    from detectron2.utils.logger import setup_logger

    setup_logger()

    for name in ("fish_train", "fish_val"):
        if name in DatasetCatalog:
            DatasetCatalog.remove(name)
            MetadataCatalog.remove(name)
    register_coco_instances("fish_train", {}, str(train_json), str(images_dir))
    register_coco_instances("fish_val", {}, str(val_json), str(images_dir))

    if args.check_dataset:
        import cv2
        import numpy as np
        from detectron2.utils.visualizer import Visualizer

        preview = args.output / "dataset_check"
        preview.mkdir(parents=True, exist_ok=True)
        metadata = MetadataCatalog.get("fish_train")
        unreadable = []
        for record in DatasetCatalog.get("fish_train")[:8]:
            # fromfile+imdecode rather than imread, so non-ASCII paths work
            try:
                image = cv2.imdecode(
                    np.fromfile(record["file_name"], dtype=np.uint8), cv2.IMREAD_COLOR
                )
            except OSError:
                image = None
            if image is None:
                unreadable.append(record["file_name"])
                continue
            drawn = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1.0)
            out = drawn.draw_dataset_dict(record)
            target = preview / Path(record["file_name"]).name
            cv2.imencode(target.suffix or ".png", out.get_image()[:, :, ::-1])[1].tofile(
                str(target)
            )
        print(f"\nWrote annotated samples to {preview}")
        if unreadable:
            print(f"\n{len(unreadable)} image(s) could not be read:")
            for name in unreadable:
                print(f"  {name}")
        print("Check that every animal is outlined before spending GPU time.")
        return

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(args.config))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.config)
    cfg.MODEL.DEVICE = args.device
    cfg.DATASETS.TRAIN = ("fish_train",)
    cfg.DATASETS.TEST = ("fish_val",) if n_val else ()
    cfg.DATALOADER.NUM_WORKERS = args.workers
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.MAX_ITER = iterations
    # warm-up and LR drops must fit inside short runs: a fixed 1000-iteration
    # warm-up would never finish in a 50-iteration test, and rounded steps can
    # collide or land past the end
    cfg.SOLVER.WARMUP_ITERS = min(1000, max(1, iterations // 10))
    step1 = max(1, int(iterations * 0.7))
    step2 = max(step1 + 1, int(iterations * 0.9))
    cfg.SOLVER.STEPS = tuple(s for s in (step1, step2) if s < iterations)
    cfg.SOLVER.CHECKPOINT_PERIOD = max(iterations // 5, 1)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(class_names)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.INPUT.MIN_SIZE_TRAIN = tuple(args.min_size_train)
    cfg.INPUT.MIN_SIZE_TEST = args.min_size_test
    cfg.TEST.EVAL_PERIOD = args.eval_period
    cfg.OUTPUT_DIR = str(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    class Trainer(DefaultTrainer):
        @classmethod
        def build_evaluator(cls, cfg, dataset_name, output_folder=None):
            return COCOEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    # DefaultTrainer evaluates once when training ends and returns that result,
    # so evaluating again here would just repeat it
    results = trainer.train() or {}
    if results:
        print("\nValidation results:")
        for task, metrics in results.items():
            for metric, value in metrics.items():
                print(f"  {task}/{metric}: {value:.3f}")

    # How the frames were prepared, recorded with the weights so inference can
    # reproduce it. Checked in order of directness: the profile copied into the
    # dataset, then the dataset report, then a sampling manifest if the dataset
    # happens to sit inside the annotation folder.
    enhancement = None
    for candidate in (
        args.dataset / "preprocess_profile.json",
        args.dataset / "report.json",
        args.dataset.parent / "sampling_manifest.json",
    ):
        if not candidate.is_file():
            continue
        try:
            data = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SystemExit(f"{candidate} could not be read as JSON: {exc}")
        if not isinstance(data, dict):
            continue
        found = data.get("enhancement", data if "clahe_clip" in data else None)
        # a recorded null means "no record", not "enhancement is None"
        if found:
            enhancement = {k: v for k, v in found.items() if k not in ("name", "notes")}
            print(f"Frame enhancement read from {candidate.name}")
            break

    metadata = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": args.config,
        "class_names": class_names,
        "num_classes": len(class_names),
        "min_size_test": args.min_size_test,
        "iterations": iterations,
        "batch_size": args.batch_size,
        "base_lr": args.lr,
        "train_images": n_train,
        "val_images": n_val,
        "enhancement": enhancement,
        "validation": {
            task: {k: float(v) for k, v in metrics.items()}
            for task, metrics in results.items()
        },
    }
    metadata_path = args.output / "training_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if enhancement is None:
        print(
            "\nWARNING: no frame-enhancement record was found alongside the"
            " dataset.\nThe weights will record null, so at inference the"
            " exporter falls back to the\nbuilt-in defaults and the"
            " training/inference mismatch check cannot fire.\nIf these frames"
            " were enhanced, copy the setup's preprocess_profile.json into\n"
            f"{args.dataset} and train again."
        )

    print(f"\nWeights:  {args.output / 'model_final.pth'}")
    print(f"Metadata: {metadata_path}")
    print(
        "\nNext:\n"
        f"  python detectron2_export_contours.py --video clip.mp4"
        f" --weights {args.output / 'model_final.pth'} --output clip_contours.h5"
    )


if __name__ == "__main__":
    main()
