# Long Water Streak Deraining Benchmark - Reproducibility Artifact

This repository accompanies a paper studying whether pretrained deraining/restoration
models generalize, zero-shot, to long, continuous, vertically oriented water streaks
on a camera lens, evaluated on a curated 500-image benchmark built from the daytime
split of the Raindrop Clarity dataset (Jin et al., ECCV 2024).

The original dataset images are not redistributed here due to license restrictions.
This repository provides only the scripts and selected image identifiers needed to
reconstruct the evaluation set and reproduce the study.

## Contents

```css
dataset_construction/            Scripts used to build the 500-image benchmark
  find_long_water_streaks_features.py   Streak detector (geometry features + severity)
  build_500_dataset.py                  Diversity sampling used to build the 500-image set
  build_replenish_pool.py               Replenishment sampling for rejected candidates
  selected_500_images_FINAL.txt         The 500 selected image identifiers (<seq>/<frame>.png)

control_experiment/              Files used to reconstruct the compact-raindrop control set
  selected_500_control_images.txt       The 500 control image identifiers (<seq>/<frame>.png)
  build_control_dataset.py              Rebuilds the paired control image set

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
