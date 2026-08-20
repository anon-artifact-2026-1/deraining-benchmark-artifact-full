"""
build_500_dataset.py  -  Stage 1 dataset construction.

Reads the feature CSV produced by find_long_water_streaks_features.py over the
full daytime split, excludes the 150 images already used in the original study,
then samples NEW long-water-streak images with DIVERSITY across streak length,
width, curvature, and coverage (not just the top-scoring, look-alike examples).
The original 150 are added back to give the 500-image benchmark.

By default it selects a slightly larger buffer of new candidates (--n_new) so
that false positives found during manual verification can be dropped and the set
still reaches 350 verified new images.

Outputs (under --out_dir):
    dataset_500/Drop/<seq>/<frame>.png     degraded inputs (150 + new)
    dataset_500/Clear/<seq>/<frame>.png    matching ground-truth frames
    selected_500_images.txt                identifier list for the whole set
    selected_new_images.txt                just the newly added identifiers
    selection_features.csv                 features for every selected image

The new images should still be visually verified (a contact sheet can be built
separately for that review).
"""

import csv
import shutil
import argparse
import random
from pathlib import Path
from collections import defaultdict


def load_features(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["has_long_water_streak"] = str(r["has_long_water_streak"]).lower() == "true"
        for k in ("longest_streak_height_ratio", "avg_streak_width", "aspect_ratio",
                  "streak_area_coverage", "boundary_contrast", "curvature_waviness",
                  "confidence_score"):
            r[k] = float(r[k])
    return rows


def load_ids(path):
    return set(l.strip() for l in open(path, encoding="utf-8") if l.strip())


def tertiles(values):
    v = sorted(values)
    if not v:
        return (0.0, 0.0)
    return (v[len(v) // 3], v[2 * len(v) // 3])


def bin3(value, cuts):
    lo, hi = cuts
    return 0 if value < lo else (1 if value < hi else 2)


def diverse_sample(candidates, n, seed=42):
    """Round-robin over height x width x curvature x coverage buckets so the
    selection spans the feature space rather than clustering on top scores."""
    random.seed(seed)
    cuts = {
        "height":   tertiles([c["longest_streak_height_ratio"] for c in candidates]),
        "width":    tertiles([c["avg_streak_width"] for c in candidates]),
        "waviness": tertiles([c["curvature_waviness"] for c in candidates]),
        "coverage": tertiles([c["streak_area_coverage"] for c in candidates]),
    }
    buckets = defaultdict(list)
    for c in candidates:
        key = (bin3(c["longest_streak_height_ratio"], cuts["height"]),
               bin3(c["avg_streak_width"], cuts["width"]),
               bin3(c["curvature_waviness"], cuts["waviness"]),
               bin3(c["streak_area_coverage"], cuts["coverage"]))
        buckets[key].append(c)
    for b in buckets.values():
        random.shuffle(b)
    order = list(buckets.keys())
    random.shuffle(order)
    picked = []
    while len(picked) < n and any(buckets[k] for k in order):
        for k in order:
            if buckets[k]:
                picked.append(buckets[k].pop())
                if len(picked) >= n:
                    break
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_csv", required=True)
    ap.add_argument("--dataset_root", required=True,
                    help="folder containing Drop/ and Clear/ (extracted split)")
    ap.add_argument("--existing_150", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_new", type=int, default=350,
                    help="number of new images to select")
    ap.add_argument("--min_confidence", type=float, default=0.0,
                    help="only curate from flagged images at/above this "
                         "detector confidence score (quality floor)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    drop_root = next(p for p in dataset_root.rglob("Drop") if p.is_dir())
    clear_root = next(p for p in dataset_root.rglob("Clear") if p.is_dir())

    out_dir = Path(args.out_dir)
    (out_dir / "dataset_500" / "Drop").mkdir(parents=True, exist_ok=True)
    (out_dir / "dataset_500" / "Clear").mkdir(parents=True, exist_ok=True)

    rows = load_features(args.features_csv)
    by_id = {r["relative_path"]: r for r in rows}
    existing = load_ids(args.existing_150)

    pool = [r for r in rows
            if r["has_long_water_streak"]
            and r["relative_path"] not in existing
            and r["confidence_score"] >= args.min_confidence]

    print(f"Total scanned        : {len(rows)}")
    print(f"Flagged as streak    : {sum(1 for r in rows if r['has_long_water_streak'])}")
    print(f"Already used (excl.) : {len(existing)}")
    print(f"Confidence floor     : {args.min_confidence}")
    print(f"New-candidate pool   : {len(pool)}")

    if len(pool) <= args.n_new:
        print(f"\nNOTE: only {len(pool)} new candidates available; selecting all.")
        new_sel = pool
    else:
        new_sel = diverse_sample(pool, args.n_new, seed=args.seed)

    new_ids = [r["relative_path"] for r in new_sel]
    all_ids = sorted(set(existing) | set(new_ids))

    print(f"New selected         : {len(new_ids)}")
    print(f"Final benchmark size : {len(all_ids)}")

    copied, missing = 0, []
    for rid in all_ids:
        sd, sc = drop_root / rid, clear_root / rid
        if not sd.exists() or not sc.exists():
            missing.append(rid)
            continue
        dd = out_dir / "dataset_500" / "Drop" / rid
        dc = out_dir / "dataset_500" / "Clear" / rid
        dd.parent.mkdir(parents=True, exist_ok=True)
        dc.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sd, dd)
        shutil.copy2(sc, dc)
        copied += 1

    (out_dir / "selected_500_images.txt").write_text("\n".join(all_ids) + "\n", encoding="utf-8")
    (out_dir / "selected_new_images.txt").write_text("\n".join(sorted(new_ids)) + "\n", encoding="utf-8")

    with open(out_dir / "selection_features.csv", "w", newline="", encoding="utf-8") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        for rid in all_ids:
            if rid in by_id:
                wtr.writerow(by_id[rid])

    print(f"\nCopied Drop+Clear    : {copied} pairs")
    if missing:
        print(f"MISSING source       : {len(missing)} (e.g. {missing[:5]})")
    print(f"Dataset at           : {out_dir / 'dataset_500'}")
    print("Wrote selected_500_images.txt, selected_new_images.txt, selection_features.csv")


if __name__ == "__main__":
    main()
