# Detectron2 segmentation for idtracker.ai

Replaces idtracker.ai's intensity thresholding with a fine-tuned Mask R-CNN,
while leaving identity tracking — crossing detection, fragmentation, the
identification network, gap closing — exactly as it was.

Contours go straight from the detector into `Blob`. Nothing is painted back into
a video and re-thresholded, so there are no per-clip intensity or area
thresholds to tune.

## The pipeline

```
   sample_frames.py        frames out of the videos, already enhanced
          |
     [ LabelMe ]           you draw one polygon per animal
          |
   labelme_to_coco.py      validate, convert, split train/val by video
          |
   train_detectron2.py     fine-tune Mask R-CNN, evaluate, record settings
          |
detectron2_export_contours.py   run the model over a clip -> contours.h5
          |
   [ idtracker.ai ]        external_contours = "clip_01_contours.h5"
```

`frame_preprocessing.py` is shared by the first and last stages. That is
deliberate: the model must see the same kind of image at inference that it was
annotated on. Training on CLAHE-enhanced frames and predicting on raw ones costs
accuracy without raising an error, so the settings are recorded at every stage
and checked at inference.

## Running it

```bash
# 1. pull frames for annotation, spread across the whole of every clip
python tools/sample_frames.py --videos clips/*.mp4 --n-frames 600 --output annotate/

# 2. annotate in LabelMe: one polygon per animal, one consistent label
labelme annotate/

# 3. convert and split
python tools/labelme_to_coco.py --input annotate/ --output dataset/ \
    --single-class fish --expected-instances 5 --val-fraction 0.15

# 4. check the annotations landed where you think, before spending GPU time
python tools/train_detectron2.py --dataset dataset/ --output model/ --check-dataset

# 5. train
python tools/train_detectron2.py --dataset dataset/ --output model/ --epochs 40

# 6. one contour file per clip
python tools/detectron2_export_contours.py --video clips/clip_01.mp4 \
    --weights model/model_final.pth --output contours/clip_01.h5 --max-instances 5
```

Then either add `external_contours = "contours/clip_01.h5"` to the clip's
`.toml`, or pick the file in the Segmentation App under **Segmentation →
External contours**.

## Things worth knowing

**The train/validation split is by video, not random.** Frames sampled a second
apart show the same animals in the same poses. Splitting those at random puts
near-duplicates on both sides and the validation AP then measures memorisation.
`--split-by random` exists but warns.

**Annotations are validated, not trusted.** Two-point polygons, stray clicks,
bounding boxes drawn where a polygon was meant, shapes off the image edge and
label typos that silently become a second class are all caught and reported in
`dataset/report.json`.

**`--on-overlap` decides who resolves occlusions.** `merge` (default) fuses
overlapping animals into one blob so idtracker.ai's crossing detector and
interpolation reconstruct identities — the legacy behaviour. `split` lets the
detector separate them instead: fewer crossings, but mask assignment can flicker
frame to frame and that instability propagates into fragmentation.

**Missing detections are left missing.** If the model finds three animals in a
frame instead of five, that frame gets three contours. Do not fill the gap by
reusing the previous frame's masks: a mask *union* becomes one blob with a stale
centroid, which idtracker.ai reads as a crossing that never happened. Its own
interpolation works from the whole video and handles this properly.

**Polygon area is not mask pixel count.** Contours are approximated
(`CHAIN_APPROX_TC89_KCOS`) and interior holes are dropped by `RETR_EXTERNAL`, so
absolute areas differ from thresholded ones by a few percent. This does not
matter — `ModelArea` recalibrates from the data — but do not compare the numbers
to old runs.

**Watch the individual/crossing split on the first run.** Detectron2 areas are
more consistent than thresholded ones, so their standard deviation is smaller,
and `MODEL_AREA_SD_TOLERANCE` (default 4, in `idtrackerai/utils/confparams.py`)
sets the cutoff at `median + 4·std`. A tighter cutoff can flag large individuals
as crossings. If the log reports implausibly many, raise it.

**Better segmentation is not a substitute for validation.** When animals
genuinely occlude each other, identity assignment remains probabilistic. Check
the hard crossings before trusting a batch of trajectories.

## Running the GPU stages in Colab

Annotation stays local. Stages 4–6 move to Colab via
[`colab_detectron2_pipeline.ipynb`](colab_detectron2_pipeline.ipynb).

```bash
python tools/make_colab_bundle.py          # ~27 KB zip of just what Colab needs
```

Upload to `MyDrive/idtrackerai_detectron2/`:

```
colab_bundle.zip
dataset/        from labelme_to_coco.py
```

**The videos must be on Drive too.** Inference decodes every frame, so the clips
themselves have to be reachable from the runtime, not just the dataset. They can
live anywhere on Drive — set `VIDEOS` in the notebook's config cell — which
matters when they run to tens of GB. A folder someone shared with you has no
path until you add a shortcut to it in My Drive (*Shared with me* → right-click
→ *Organise* → *Add shortcut to Drive*).

Open the notebook in Colab, set a GPU runtime, run the cells. Section 2 checks
the mount, opens every clip, decodes a frame from each, and reports what is
already exported before anything long begins:

```bash
python tools/check_videos.py --videos clips/*.mp4 --contours contours/
```

Run it locally too — it exits non-zero on an unopenable or truncated clip.

**Track the same files you exported contours from.** Both the exporter and
idtracker.ai read frame counts with the same `cv2.CAP_PROP_FRAME_COUNT` call, so
they agree by construction — but only for the same file. A re-encoded copy will
be rejected by the sidecar check.

**The export will not finish in one session.** At a plausible 10 fps, 48 clips ×
30 000 frames is about 40 hours, against a session ceiling of 12–24 hours. So it
is built to resume: each clip writes its own `.h5`, reruns skip finished files,
and the file is written to `.partial` and moved into place so a session killed
mid-write cannot leave a truncated file that looks done. Reconnect, rerun the
export cell, repeat.

Two further Colab specifics the notebook handles:

- **Detectron2 is built from source, ~10 minutes.** The wheel is cached to Drive
  and keyed by Python and torch version, so later sessions reuse it and a
  runtime upgrade rebuilds once.
- **Videos are copied to local disk before decoding.** OpenCV seeks constantly,
  and the Drive FUSE mount is slow enough at that to dominate inference time.

What comes back is tens of MB per clip rather than the gigabytes an enhanced
video would be.

## Citing

The tracking is idtracker.ai's; the segmentation here is Detectron2's. Cite
both — see the [repository README](../README.md#citing) for the full list, which
covers the idtracker.ai papers, Detectron2 and Mask R-CNN. Annotation uses
[LabelMe](https://github.com/wkentaro/labelme) (Wada, GPL-3.0) and the image
processing throughout is [OpenCV](https://opencv.org).

## Installing Detectron2

Only stages 4–6 need it; sampling and conversion run on OpenCV and NumPy alone.

```bash
pip install 'torch>=2.1' torchvision --index-url https://download.pytorch.org/whl/cu121
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

There is no universal Detectron2 wheel — it builds against the installed
torch/CUDA pair, so install torch first and let it compile against that.
