"""
build_replenish_pool.py  -  Stage 1 replenishment candidates.

After manual review of the first 350 candidates, 181 were rejected as not
containing real long water streaks. This pulls a fresh batch of candidates from
the leftover pool so the benchmark can be topped back up to 350 verified new
images.

Pool = detector-flagged (has_long_water_streak) AND confidence >= floor,
       MINUS the original 150,
       MINUS every identifier already used in the first 350 round
             (both the 169 kept and the 181 rejected - rejects must never
              come back round).

Selection uses the same diversity sampling as build_500_dataset.py so the batch
spans streak height / width / curvature / coverage rather than clustering on
top scores.

Output: <out_dir>/NNN__<seq>_<frame>.png   (flat, for arrow-key review)
        <out_dir>/../replenish_candidates.txt
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


def diverse_sample(candidates, n, seed):
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
    ap.add_argument("--features_csv", default="features_full.csv")
    ap.add_argument("--drop_root", default="dataset/DayRainDrop_Train/Drop")
    ap.add_argument("--existing_150", default="positives.txt")
    ap.add_argument("--already_used", nargs="+", default=["selected_new_images.txt"],
                    help="id lists already shown to the reviewer (kept AND rejected)")
    ap.add_argument("--out_dir", default="potential_replenish")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--min_confidence", type=float, default=40.0)
    ap.add_argument("--seed", type=int, default=43)
    args = ap.parse_args()

    rows = load_features(args.features_csv)
    existing = load_ids(args.existing_150)
    used = set()
    for p in args.already_used:
        used |= load_ids(p)

    pool = [r for r in rows
            if r["has_long_water_streak"]
            and r["relative_path"] not in existing
            and r["relative_path"] not in used
            and r["confidence_score"] >= args.min_confidence]

    print(f"Total scanned          : {len(rows)}")
    print(f"Original 150 excluded  : {len(existing)}")
    print(f"Round-1 ids excluded   : {len(used)}")
    print(f"Confidence floor       : {args.min_confidence}")
    print(f"Available pool         : {len(pool)}")

    n = min(args.n, len(pool))
    if n < args.n:
        print(f"NOTE: pool smaller than requested; taking all {n}.")
    sel = diverse_sample(pool, n, seed=args.seed) if len(pool) > n else pool

    sel_ids = sorted(r["relative_path"] for r in sel)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    drop_root = Path(args.drop_root)

    copied, missing = 0, []
    for i, rid in enumerate(sel_ids, start=1):
        src = drop_root / rid
        if not src.exists():
            missing.append(rid)
            continue
        flat = f"{i:03d}__{rid.replace('/', '_')}"
        shutil.copy2(src, out_dir / flat)
        copied += 1

    list_path = Path(f"{Path(args.out_dir).name}_ids.txt")
    list_path.write_text("\n".join(sel_ids) + "\n", encoding="utf-8")

    print(f"\nSelected               : {len(sel_ids)}")
    print(f"Copied to {out_dir}/   : {copied}")
    if missing:
        print(f"MISSING source         : {len(missing)} (e.g. {missing[:5]})")
    print("Wrote replenish_candidates.txt")


if __name__ == "__main__":
    main()
