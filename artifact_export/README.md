# Long Water Streak Deraining Benchmark — Reproducibility Artifact

This repository accompanies a paper studying whether pretrained deraining/restoration
models generalize, zero-shot, to long, continuous, vertically oriented water streaks
on a camera lens, evaluated on a curated 500-image benchmark built from the daytime
split of the Raindrop Clarity dataset (Jin et al., ECCV 2024).

The original dataset images are not redistributed here due to license restrictions.
This repository provides only the scripts and selected image identifiers needed to
reconstruct the evaluation set and reproduce the study.

## Contents

```
dataset_construction/            Scripts used to build the 500-image benchmark
  find_long_water_streaks_features.py   Streak detector (geometry features + severity)
  build_500_dataset.py                  Diversity sampling used to build the 500-image set
  build_replenish_pool.py               Replenishment sampling for rejected candidates
  selected_500_images_FINAL.txt         The 500 selected image identifiers (<seq>/<frame>.png)

kaggle_notebooks/                 Evaluation scripts for all five restoration models
                                   and both cascade configurations

scoring_scripts/
  score_pairs.py                  PSNR / SSIM / LPIPS scoring

statistics/
  run_stage3_stats.py             Friedman and Wilcoxon statistical tests

selector/
  run_stage4_selector.py          Geometry-aware selector training script

object_detection/
  Stage5_Detection.ipynb          Object-detection evaluation script
```

## Reconstructing the image set

1. Obtain the Raindrop Clarity dataset (daytime training split, `DayRainDrop_Train`)
   from its original source.
2. Use `dataset_construction/selected_500_images_FINAL.txt` to select the 500 image
   pairs (`Drop/<id>` = degraded input, `Clear/<id>` = ground truth).

## Metrics protocol

PSNR and SSIM are computed on the luminance (Y) channel; LPIPS uses a VGG backbone.
All scores are computed at a fixed 256x256 resolution using the official Raindrop
Clarity evaluation protocol, except where noted in the paper (one model's checkpoint
is architecturally fixed to 128x128; see the paper's discussion of this constraint).

## License

The code in this repository is provided for reproducibility purposes. The Raindrop
Clarity dataset is subject to its own license and is not redistributed here.
