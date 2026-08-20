"""
score_pairs.py  -  Unified per-image scorer for the Stage 2 500-image benchmark.

Works IDENTICALLY for every configuration (no-restoration, DiT alone, RCDNet
alone, the two cascades, and each new model), so all 8 configs are directly
comparable and the Stage 3 paired tests are valid.

It takes two folders - predicted (restored) images and ground-truth images -
pairs them by relative path, resizes BOTH to a single common resolution
(default 256x256), and computes:
    - PSNR  (luminance/Y channel, official Raindrop Clarity protocol)
    - SSIM  (luminance/Y channel, official protocol)
    - LPIPS (VGG backbone)

using the SAME metric functions as the official evaluation code
(utils.metrics + lpips vgg), so numbers stay comparable to the published
benchmark and to the short paper.

WHY A FIXED RESOLUTION: PSNR/SSIM are resolution-dependent. In the short paper
DiT was scored at 256x256 while no-restoration/RCDNet were at native 720x480,
so those numbers were NOT strictly comparable. Scoring every configuration at
one resolution fixes that permanently.

OUTPUTS (under --out_dir):
    <name>_per_image.csv    one row per image: relative_path, psnr, ssim, lpips
    <name>_summary.json     n, mean/std/median/min/max per metric, resolution
    <name>_summary.txt      same, human-readable

The per-image CSV is the critical Stage 3 input. Do NOT discard it.

USAGE (run from inside the RaindropClarity repo dir so `utils` imports):
    # no restoration (degraded input vs GT):
    python score_pairs.py --pred dataset_500/Drop --gt dataset_500/Clear \
        --name no_restoration --out_dir /kaggle/working/metrics

    # DiT alone (eval_*.py writes results/RainDrop/<model>/<ds>/{output,gt}):
    python score_pairs.py --pred results/.../output --gt results/.../gt \
        --name dit_alone --out_dir /kaggle/working/metrics

    # RCDNet alone (rcdnet_cascade.py writes <out>/Drop + <out>/Clear):
    python score_pairs.py --pred rcd_out/Drop --gt rcd_out/Clear \
        --name rcdnet_alone --out_dir /kaggle/working/metrics

    # smoke test on 20 images before a full run:
    python score_pairs.py --pred ... --gt ... --name test --limit 20
"""

import os
import csv
import json
import glob
import argparse
import statistics as st

import cv2
import numpy as np
import torch
import lpips

from utils.metrics import calculate_psnr, calculate_ssim


def list_pairs(pred_root, gt_root):
    """Pair every PNG under pred_root with the same relative path under gt_root."""
    preds = sorted(glob.glob(os.path.join(pred_root, "**", "*.png"), recursive=True))
    pairs, missing = [], []
    for p in preds:
        rel = os.path.relpath(p, pred_root).replace("\\", "/")
        g = os.path.join(gt_root, rel)
        if os.path.exists(g):
            pairs.append((rel, p, g))
        else:
            missing.append(rel)
    return pairs, missing


def summarize(name, values):
    if not values:
        return {}
    return {
        f"{name}_mean": float(np.mean(values)),
        f"{name}_std": float(np.std(values)),
        f"{name}_median": float(st.median(values)),
        f"{name}_min": float(np.min(values)),
        f"{name}_max": float(np.max(values)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="folder of restored/predicted images")
    ap.add_argument("--gt", required=True, help="folder of ground-truth images")
    ap.add_argument("--name", required=True, help="configuration name (used in output filenames)")
    ap.add_argument("--out_dir", default="metrics")
    ap.add_argument("--size", type=int, default=256,
                    help="common resolution both images are resized to before scoring")
    ap.add_argument("--limit", type=int, default=0,
                    help="score only the first N pairs (0 = all); use for the smoke test")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    loss_fn_vgg = lpips.LPIPS(net="vgg")

    pairs, missing = list_pairs(args.pred, args.gt)
    if args.limit:
        pairs = pairs[:args.limit]
    print(f"[{args.name}] pairs to score: {len(pairs)}   missing gt: {len(missing)}")
    if missing[:5]:
        print("  e.g. missing:", missing[:5])
    if not pairs:
        raise SystemExit("No image pairs found - check --pred and --gt paths.")

    S = (args.size, args.size)
    rows = []
    csv_path = os.path.join(args.out_dir, f"{args.name}_per_image.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["relative_path", "psnr", "ssim", "lpips"])
        for k, (rel, pp, gp) in enumerate(pairs):
            res = cv2.imread(pp, cv2.IMREAD_COLOR)
            gt = cv2.imread(gp, cv2.IMREAD_COLOR)
            if res is None or gt is None:
                print("  WARN unreadable, skipped:", rel)
                continue
            # resize BOTH to the common resolution (INTER_AREA is best for downscale)
            res = cv2.resize(res, S, interpolation=cv2.INTER_AREA)
            gt = cv2.resize(gt, S, interpolation=cv2.INTER_AREA)

            cur_psnr = calculate_psnr(res, gt, test_y_channel=True)
            cur_ssim = calculate_ssim(res, gt, test_y_channel=True)
            # LPIPS: replicate the official repo's exact call (feeds 0-255 float,
            # no [-1,1] normalization) so numbers match the published benchmark.
            tr = torch.from_numpy(res.transpose((2, 0, 1))).float().unsqueeze(0)
            tg = torch.from_numpy(gt.transpose((2, 0, 1))).float().unsqueeze(0)
            cur_lpips = float(loss_fn_vgg(tr, tg).cpu().data.numpy().ravel()[0])

            w.writerow([rel, f"{cur_psnr:.6f}", f"{cur_ssim:.6f}", f"{cur_lpips:.6f}"])
            rows.append((cur_psnr, cur_ssim, cur_lpips))
            if k % 100 == 0:
                print(f"  {k}/{len(pairs)}  running PSNR "
                      f"{np.mean([r[0] for r in rows]):.4f}")

    psnrs = [r[0] for r in rows]; ssims = [r[1] for r in rows]; lps = [r[2] for r in rows]
    summary = {
        "config": args.name,
        "n_images": len(rows),
        "resolution": f"{args.size}x{args.size}",
        "missing_gt": len(missing),
    }
    summary.update(summarize("psnr", psnrs))
    summary.update(summarize("ssim", ssims))
    summary.update(summarize("lpips", lps))

    with open(os.path.join(args.out_dir, f"{args.name}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out_dir, f"{args.name}_summary.txt"), "w") as f:
        f.write(f"Configuration : {args.name}\n")
        f.write(f"Images scored : {len(rows)}\n")
        f.write(f"Resolution    : {args.size}x{args.size}\n")
        f.write(f"PSNR  mean {summary['psnr_mean']:.4f}  std {summary['psnr_std']:.4f}  "
                f"median {summary['psnr_median']:.4f}\n")
        f.write(f"SSIM  mean {summary['ssim_mean']:.4f}  std {summary['ssim_std']:.4f}  "
                f"median {summary['ssim_median']:.4f}\n")
        f.write(f"LPIPS mean {summary['lpips_mean']:.4f}  std {summary['lpips_std']:.4f}  "
                f"median {summary['lpips_median']:.4f}\n")

    print(f"\n[{args.name}] DONE  n={len(rows)}  "
          f"PSNR {summary['psnr_mean']:.4f}  SSIM {summary['ssim_mean']:.4f}  "
          f"LPIPS {summary['lpips_mean']:.4f}")
    print(f"wrote {csv_path}")
    print(f"wrote {args.out_dir}/{args.name}_summary.[json|txt]")


if __name__ == "__main__":
    main()
