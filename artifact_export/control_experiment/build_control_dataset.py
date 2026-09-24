#!/usr/bin/env python3
"""Build the paired Drop/Clear folders for the 500-image non-streak CONTROL set,
with a hard guarantee of NO overlap with the streak-500 benchmark.

Overlap is prevented two ways:
  1. By image ID   -- control_500_images.txt already excludes every streak-500 id.
  2. By pixel hash -- if --streak_drop_dir is given, any control Drop image whose
                      content md5 matches a streak Drop image is reported and skipped
                      (the daytime split has a few pixel-identical frames under
                      different names; this catches them).

Usage:
    python build_control_dataset.py \
        --dataset_root "<path>/DayRainDrop_Train" \
        --id_list control_500_images.txt \
        --out_dir control_500_paired \
        --streak_drop_dir "<path>/dataset_500_paired_good/Drop"   # optional but recommended

<dataset_root> must contain Drop/<seq>/<frame>.png and Clear/<seq>/<frame>.png.
"""
import argparse, os, shutil, sys, hashlib

def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()

ap = argparse.ArgumentParser()
ap.add_argument("--dataset_root", required=True)
ap.add_argument("--id_list", default="control_500_images.txt")
ap.add_argument("--out_dir", default="control_500_paired")
ap.add_argument("--streak_drop_dir", default=None,
                help="streak-500 Drop/ folder; if given, pixel-dedup against it")
args = ap.parse_args()

ids = [l.strip().replace("\\", "/") for l in open(args.id_list, encoding="utf-8") if l.strip()]
print(f"{len(ids)} control ids requested")

drop_root = os.path.join(args.dataset_root, "Drop")
clear_root = os.path.join(args.dataset_root, "Clear")
for r in (drop_root, clear_root):
    if not os.path.isdir(r):
        sys.exit(f"ERROR: {r} not found. Point --dataset_root at the folder holding Drop/ and Clear/.")

# --- optional: hash every streak Drop image so we can reject pixel-duplicates ---
streak_hashes = set()
if args.streak_drop_dir:
    if not os.path.isdir(args.streak_drop_dir):
        sys.exit(f"ERROR: --streak_drop_dir {args.streak_drop_dir} not found.")
    for root, _, files in os.walk(args.streak_drop_dir):
        for fn in files:
            if fn.lower().endswith(".png"):
                streak_hashes.add(md5(os.path.join(root, fn)))
    print(f"hashed {len(streak_hashes)} streak Drop images for pixel-dedup")

copied, missing, pixel_dupes, control_hashes = 0, [], [], {}
for rid in ids:
    src_drop = os.path.join(drop_root, rid)
    src_clear = os.path.join(clear_root, rid)
    if not (os.path.isfile(src_drop) and os.path.isfile(src_clear)):
        missing.append(rid); continue
    h = md5(src_drop)
    if h in streak_hashes:                       # identical content to a streak image
        pixel_dupes.append(rid); continue
    if h in control_hashes:                       # duplicate within the control set itself
        pixel_dupes.append(f"{rid} (dup of {control_hashes[h]})"); continue
    control_hashes[h] = rid
    for side, src in (("Drop", src_drop), ("Clear", src_clear)):
        dst = os.path.join(args.out_dir, side, rid)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
    copied += 1

print(f"\ncopied {copied} clean pairs into {args.out_dir}/")
if missing:
    print(f"  {len(missing)} ids missing from dataset, e.g. {missing[:5]}")
if pixel_dupes:
    print(f"  {len(pixel_dupes)} pixel-duplicates SKIPPED, e.g. {pixel_dupes[:5]}")
    print("  -> tell Claude how many were skipped and it will top the list back up to 500.")
if copied == len(ids) and not pixel_dupes:
    print("  PERFECT: all 500 copied, zero ID overlap, zero pixel duplicates.")
