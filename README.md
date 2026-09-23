# idtrackerai-detectron2

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Upstream](https://img.shields.io/badge/fork%20of-idtracker.ai-lightgrey)](https://gitlab.com/polavieja_lab/idtrackerai)
[![eLife](https://img.shields.io/badge/cite-10.7554%2FeLife.107602-blue)](https://doi.org/10.7554/eLife.107602)

A fork of [idtracker.ai](https://idtracker.ai) that takes animal outlines from
an external instance-segmentation model — a fine-tuned Detectron2 Mask R-CNN —
instead of finding them by intensity thresholding.

**Identity tracking is untouched.** Crossing detection, fragmentation, the
identification network and gap closing are upstream idtracker.ai's, unmodified.
Only the step that turns pixels into blobs is replaced.

> [!IMPORTANT]
> This is an **unofficial derivative**. It is not affiliated with, endorsed by,
> or supported by the de Polavieja Lab or the Champalimaud Foundation. Please do
> not send problems with this fork to the upstream project — open an issue
> [here](https://github.com/mbel-tech/idtrackerai-detectron2/issues) instead.
>
> For the real thing, go to
> [gitlab.com/polavieja_lab/idtrackerai](https://gitlab.com/polavieja_lab/idtrackerai).

## Why

idtracker.ai segments by thresholding each frame. On low-contrast footage —
unevenly lit tanks, turbid water, fish the same brightness as the background —
that is the limiting step, and no amount of tracking-parameter tuning fixes it.

The usual workaround is to run a segmentation model first, paint its masks back
into the video (darkening the background, brightening the animals), re-encode,
and let idtracker.ai threshold *that*. It works, but it is a round trip through
pixels: precise instance masks are thrown away and approximately recovered, and
every clip needs its own intensity and area thresholds to undo the damage.

This fork removes the round trip. Contours go from the detector into
idtracker.ai's `Blob` objects directly. There is nothing to re-threshold, so
there is nothing to tune per clip.

## How it works

`process_frame()` is the only place in idtracker.ai where pixels become
contours, so it is the only place that changes. Given a contour file it reads
that frame's outlines instead of thresholding, and returns the frame unchanged
so identification images still come from the video's own pixels.

Everything downstream is unaffected, because `Blob` derives its centroid, area,
bounding box, extension and orientation *from the contour*. Supply a contour and
every blob feature exists, computed by the same code as before.

```
Detectron2 Mask R-CNN  ->  contours.h5  ->  idtracker.ai
                                             |
                                             +- crossing detection   unmodified
                                             +- fragmentation        unmodified
                                             +- identification CNN   unmodified
                                             +- gap closing          unmodified
```

When two animals genuinely overlap, the default (`--on-overlap merge`) hands
idtracker.ai a single blob and lets its crossing machinery reconstruct who was
who — the behaviour you already trust. The alternative, letting the detector
split them, is available but shifts that judgement to Mask R-CNN.

## Using it

Install as you would upstream idtracker.ai, from this repository:

```bash
pip install git+https://github.com/mbel-tech/idtrackerai-detectron2
```

The import name is unchanged (`import idtrackerai`), so it cannot be installed
alongside upstream idtracker.ai in the same environment.

### Preparing a model

Open a video in the Segmentation App and set **Segmentation → Detectron2
pipeline**. A guided panel walks through the four local stages:

1. **Enhancement** — CLAHE and illumination correction, previewed live on the
   frame you are looking at, with a raw↔enhanced slider to compare. Enhancement
   belongs to a recording setup, not to the software, so this is where you judge
   it against your own footage and save it as that setup's profile.
2. **Sample frames** — stratified across the whole of every clip, so the model
   sees the recording's full range rather than a clump of it.
3. **Annotate** — opens LabelMe on the folder, preloaded with your class name.
4. **Build dataset** — validates every polygon and converts to COCO, splitting
   train and validation **by source video** so the score is not measuring
   memorisation of near-duplicate frames.

You can close the app between steps; each one records what it produced, and
status is re-derived from disk when you come back.

Training and inference need a GPU and run in
[`tools/colab_detectron2_pipeline.ipynb`](tools/colab_detectron2_pipeline.ipynb).
The same stages are available as command-line scripts — see
**[tools/README.md](tools/README.md)**.

### Tracking with the result

Point a session at the contour file, in the `.toml`:

```toml
external_contours = "contours/clip_01.h5"
```

…or pick it in the Segmentation App under **Segmentation → External contours**.
The intensity thresholds and background subtraction grey out, because they take
no part once contours come from a model.

## Citing

The tracking method is the de Polavieja Lab's work. **Cite their paper**,
whether or not you use this fork:

> Torrents, J., Costa, T., & de Polavieja, G. G. (2026). New idtracker.ai:
> rethinking multi-animal tracking as a representation learning problem to
> increase accuracy and reduce tracking times. *eLife*, 14, RP107602.
> https://doi.org/10.7554/eLife.107602

Depending on what you are reporting, the earlier papers may also be relevant:

> Romero-Ferrero, F., Bergomi, M. G., Hinz, R. C., Heras, F. J. H., &
> de Polavieja, G. G. (2019). idtracker.ai: tracking all individuals in small or
> large collectives of unmarked animals. *Nature Methods*, 16(2), 179–182.
> https://doi.org/10.1038/s41592-018-0295-5

> Pérez-Escudero, A., Vicente-Page, J., Hinz, R. C., Arganda, S., &
> de Polavieja, G. G. (2014). idTracker: tracking individuals in a group by
> automatic identification of unmarked animals. *Nature Methods*, 11(7),
> 743–748. https://doi.org/10.1038/nmeth.2994

If the segmentation stage matters to your results, also cite Detectron2 and
Mask R-CNN:

> Wu, Y., Kirillov, A., Massa, F., Lo, W.-Y., & Girshick, R. (2019).
> Detectron2. https://github.com/facebookresearch/detectron2

> He, K., Gkioxari, G., Dollár, P., & Girshick, R. (2017). Mask R-CNN. In
> *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*.
> https://doi.org/10.1109/ICCV.2017.322

[`CITATION.cff`](CITATION.cff) carries all of this in machine-readable form;
GitHub's *Cite this repository* button reads it.

## Licence

GPL-3.0-or-later, the same licence as upstream idtracker.ai. The original
[`LICENSE`](LICENSE) is included verbatim and unchanged.

[`NOTICE`](NOTICE) records the fork point and lists every modification, as
GPL-3.0 section 5(a) requires. The upstream README is preserved as
[`README.idtrackerai.md`](README.idtrackerai.md).

## Acknowledgements

idtracker.ai is built by the de Polavieja Lab at the Champalimaud Foundation.
This fork is a thin change to one stage of their pipeline; the science, and
nearly all of the code, is theirs.
