"""
find_long_water_streaks_features.py

Extended version of the long-water-streak detector used to build the original
150-image subset. The DETECTION LOGIC IS UNCHANGED: the same edge extraction,
morphology, geometric filters, and scoring are applied, so the set of images
flagged as streaks (and therefore the selection) is identical to the original
detector. This version only ADDS per-image feature reporting so that each image
can be characterised and later classified by streak severity.

For every image it reports the eight features requested for the expanded study:

    1. longest_streak_height      - height (px) of the tallest streak component
    2. avg_streak_width           - mean width (px) across streak components
    3. aspect_ratio               - aspect ratio of the strongest streak
    4. streak_area_coverage       - total streak area / frame area (0-1)
    5. num_streak_components      - number of accepted streak components
    6. boundary_contrast          - mean horizontal-gradient strength on the
                                    streak side edges (0-255); higher = sharper
                                    refractive borders
    7. curvature_waviness         - waviness (std of row centres) of the
                                    strongest streak
    8. confidence_score           - detector confidence (aggregate streak score)

It then assigns a coarse severity label used for diversity sampling and for the
downstream geometry-aware selector:

    none | low | medium | high

Severity is derived from the detector's own confidence score (which already
combines height, aspect, width, continuity, waviness, and side-edge count), with
thresholds exposed as CLI arguments so they can be tuned to the observed score
distribution.
"""

import cv2
import csv
import shutil
import argparse
import numpy as np
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def count_true_clusters(arr):
    clusters = 0
    in_cluster = False

    for v in arr:
        if v and not in_cluster:
            clusters += 1
            in_cluster = True
        elif not v:
            in_cluster = False

    return clusters


def detect_long_water_streaks(
    image_path,
    min_height_ratio=0.30,
    min_width_ratio=0.012,
    max_width_ratio=0.22,
    min_aspect=2.2,
    min_score=4.5,
    edge_percentile=91,
    debug_path=None
):
    img = cv2.imread(str(image_path))
    if img is None:
        return False, 0.0, [], None

    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Improve local contrast without changing the image content.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Long water streaks usually have strong left/right borders.
    # So we use horizontal gradient, not generic Hough lines.
    sobel_x = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    sobel_x = np.abs(sobel_x)
    sobel_x = cv2.normalize(sobel_x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    threshold_value = max(25, np.percentile(sobel_x, edge_percentile))
    edge = (sobel_x >= threshold_value).astype(np.uint8) * 255

    # Remove tiny round droplet noise.
    edge = cv2.morphologyEx(
        edge,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    )

    # Connect vertical/wavy water-trail edges.
    vertical_kernel_height = max(21, h // 18)
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (5, vertical_kernel_height)
    )

    mask = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, vertical_kernel)

    # Slight horizontal dilation helps join the two sides of a water trail.
    mask = cv2.dilate(
        mask,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1
    )

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    candidates = []

    min_height = int(h * min_height_ratio)
    min_width = max(6, int(w * min_width_ratio))
    max_width = int(w * max_width_ratio)

    for label_id in range(1, num_labels):
        x, y, bw, bh, area = stats[label_id]

        if bh < min_height:
            continue

        if bw < min_width or bw > max_width:
            continue

        aspect = bh / max(bw, 1)

        if aspect < min_aspect:
            continue

        component = (labels[y:y + bh, x:x + bw] == label_id)

        rows_with_pixels = component.any(axis=1)
        row_coverage = rows_with_pixels.mean()

        if row_coverage < 0.45:
            continue

        # Reject single perfectly straight background edges, poles, window borders.
        row_centers = []
        for r in range(component.shape[0]):
            cols = np.where(component[r])[0]
            if len(cols) > 0:
                row_centers.append(cols.mean())

        if len(row_centers) < 5:
            continue

        waviness = float(np.std(row_centers))

        col_support = component.sum(axis=0)
        strong_cols = col_support > (0.12 * bh)
        col_clusters = count_true_clusters(strong_cols)

        # A real water streak often has two side edges or a thicker band.
        # A pole/window border often has one thin straight cluster.
        if col_clusters < 2 and bw < 18 and waviness < 2.0:
            continue

        fill_ratio = area / max(bw * bh, 1)

        if fill_ratio < 0.03 or fill_ratio > 0.75:
            continue

        height_score = 4.0 * (bh / h)
        aspect_score = min(aspect / 3.0, 2.0)
        width_score = min(bw / 30.0, 2.0)
        continuity_score = row_coverage
        waviness_score = min(waviness / 6.0, 1.0)
        cluster_score = min(col_clusters / 2.0, 1.0)

        score = (
            height_score
            + aspect_score
            + width_score
            + continuity_score
            + waviness_score
            + cluster_score
        )

        if score >= min_score:
            # --- NEW: boundary contrast -------------------------------------
            # Mean horizontal-gradient strength on this streak's edge pixels,
            # measured from the same Sobel map the detector already builds.
            # This does not affect the score or the accept/reject decision.
            region = sobel_x[y:y + bh, x:x + bw]
            edge_vals = region[component]
            boundary_contrast = float(edge_vals.mean()) if edge_vals.size else 0.0
            # ----------------------------------------------------------------

            candidates.append({
                "x": int(x),
                "y": int(y),
                "w": int(bw),
                "h": int(bh),
                "area": int(area),
                "aspect": round(float(aspect), 3),
                "fill_ratio": round(float(fill_ratio), 3),
                "row_coverage": round(float(row_coverage), 3),
                "waviness": round(float(waviness), 3),
                "col_clusters": int(col_clusters),
                "boundary_contrast": round(boundary_contrast, 3),
                "score": round(float(score), 3)
            })

    candidates = sorted(candidates, key=lambda c: c["score"], reverse=True)

    total_score = round(sum(c["score"] for c in candidates[:5]), 3)
    has_long_streak = len(candidates) > 0

    # --- NEW: aggregate image-level features -------------------------------
    features = build_image_features(candidates, frame_h=h, frame_w=w,
                                    total_score=total_score)
    # -----------------------------------------------------------------------

    if debug_path is not None and has_long_streak:
        debug_path = Path(debug_path)
        debug_path.parent.mkdir(parents=True, exist_ok=True)

        vis = img.copy()

        for c in candidates:
            x, y, bw, bh = c["x"], c["y"], c["w"], c["h"]
            cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 0, 255), 3)

            label = f"score={c['score']} h={bh} w={bw}"
            cv2.putText(
                vis,
                label,
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

        cv2.imwrite(str(debug_path), vis)

        mask_path = debug_path.with_name(debug_path.stem + "_mask.png")
        cv2.imwrite(str(mask_path), mask)

    return has_long_streak, total_score, candidates, features


def build_image_features(candidates, frame_h, frame_w, total_score):
    """Aggregate the per-component detections into the eight image-level
    features requested for the study. Returns zeros for images with no
    accepted streak component."""
    frame_area = float(frame_h * frame_w)

    if not candidates:
        return {
            "longest_streak_height": 0,
            "longest_streak_height_ratio": 0.0,
            "avg_streak_width": 0.0,
            "aspect_ratio": 0.0,
            "streak_area_coverage": 0.0,
            "num_streak_components": 0,
            "boundary_contrast": 0.0,
            "curvature_waviness": 0.0,
            "confidence_score": 0.0,
        }

    # candidates are sorted by score desc, so candidates[0] is the strongest.
    strongest = candidates[0]
    heights = [c["h"] for c in candidates]
    widths = [c["w"] for c in candidates]
    areas = [c["area"] for c in candidates]
    contrasts = [c["boundary_contrast"] for c in candidates]

    longest_h = max(heights)

    return {
        "longest_streak_height": int(longest_h),
        "longest_streak_height_ratio": round(longest_h / frame_h, 4),
        "avg_streak_width": round(float(np.mean(widths)), 3),
        "aspect_ratio": round(float(strongest["aspect"]), 3),
        "streak_area_coverage": round(float(sum(areas)) / frame_area, 5),
        "num_streak_components": len(candidates),
        "boundary_contrast": round(float(np.mean(contrasts)), 3),
        "curvature_waviness": round(float(strongest["waviness"]), 3),
        "confidence_score": round(float(total_score), 3),
    }


def classify_severity(has_streak, confidence_score, low_thr, high_thr):
    """Coarse severity label derived from the detector's confidence score.
    Thresholds are configurable so they can be tuned to the observed
    score distribution."""
    if not has_streak:
        return "none"
    if confidence_score < low_thr:
        return "low"
    if confidence_score < high_thr:
        return "medium"
    return "high"


FEATURE_COLUMNS = [
    "longest_streak_height",
    "longest_streak_height_ratio",
    "avg_streak_width",
    "aspect_ratio",
    "streak_area_coverage",
    "num_streak_components",
    "boundary_contrast",
    "curvature_waviness",
    "confidence_score",
]


def scan_folder(
    input_dir,
    output_dir,
    csv_path,
    min_height_ratio,
    min_width_ratio,
    max_width_ratio,
    min_aspect,
    min_score,
    edge_percentile,
    low_thr,
    high_thr,
    copy_matches,
    debug
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    csv_path = Path(csv_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = [
        p for p in input_dir.rglob("*")
        if p.suffix.lower() in IMAGE_EXTS
    ]

    rows = []
    matched = 0

    for i, img_path in enumerate(image_paths, start=1):
        rel = img_path.relative_to(input_dir)

        debug_path = None
        if debug:
            debug_path = output_dir / "_debug" / rel

        has_streak, score, candidates, features = detect_long_water_streaks(
            img_path,
            min_height_ratio=min_height_ratio,
            min_width_ratio=min_width_ratio,
            max_width_ratio=max_width_ratio,
            min_aspect=min_aspect,
            min_score=min_score,
            edge_percentile=edge_percentile,
            debug_path=debug_path
        )

        collection_id = rel.parts[0] if len(rel.parts) > 1 else img_path.parent.name
        severity = classify_severity(has_streak, features["confidence_score"],
                                     low_thr, high_thr)

        row = {
            "collection_id": collection_id,
            "relative_path": str(rel).replace("\\", "/"),
            "image_path": str(img_path),
            "has_long_water_streak": has_streak,
            "num_candidates": len(candidates),
            "severity": severity,
        }
        row.update({k: features[k] for k in FEATURE_COLUMNS})
        rows.append(row)

        if has_streak:
            matched += 1
            if copy_matches:
                target = output_dir / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(img_path, target)

        if i % 200 == 0:
            print(f"Scanned {i}/{len(image_paths)} | matched {matched}")

    # streak images first, then by confidence score (descending)
    rows = sorted(
        rows,
        key=lambda r: (r["has_long_water_streak"], r["confidence_score"]),
        reverse=True
    )

    fieldnames = [
        "collection_id",
        "relative_path",
        "image_path",
        "has_long_water_streak",
        "num_candidates",
        "severity",
    ] + FEATURE_COLUMNS

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # severity breakdown for a quick summary
    sev_counts = {"none": 0, "low": 0, "medium": 0, "high": 0}
    for r in rows:
        sev_counts[r["severity"]] += 1

    print("\nDone.")
    print(f"Total images scanned : {len(image_paths)}")
    print(f"Flagged as streak    : {matched}")
    print(f"Candidate-reduction  : {100.0 * (1 - matched / max(len(image_paths),1)):.1f}% "
          f"({matched}/{len(image_paths)} retained)")
    print(f"Severity breakdown   : "
          f"low={sev_counts['low']}  medium={sev_counts['medium']}  high={sev_counts['high']}")
    print(f"CSV saved to         : {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--csv_path", required=True)

    # detection parameters (unchanged defaults from the original detector)
    parser.add_argument("--min_height_ratio", type=float, default=0.30)
    parser.add_argument("--min_width_ratio", type=float, default=0.012)
    parser.add_argument("--max_width_ratio", type=float, default=0.22)
    parser.add_argument("--min_aspect", type=float, default=2.2)
    parser.add_argument("--min_score", type=float, default=4.5)
    parser.add_argument("--edge_percentile", type=float, default=91)

    # severity thresholds on the aggregate confidence score
    parser.add_argument("--low_thr", type=float, default=6.0,
                        help="confidence score below this = low severity")
    parser.add_argument("--high_thr", type=float, default=9.0,
                        help="confidence score at/above this = high severity")

    parser.add_argument("--copy_matches", action="store_true",
                        help="copy flagged images into output_dir")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    scan_folder(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        csv_path=args.csv_path,
        min_height_ratio=args.min_height_ratio,
        min_width_ratio=args.min_width_ratio,
        max_width_ratio=args.max_width_ratio,
        min_aspect=args.min_aspect,
        min_score=args.min_score,
        edge_percentile=args.edge_percentile,
        low_thr=args.low_thr,
        high_thr=args.high_thr,
        copy_matches=args.copy_matches,
        debug=args.debug,
    )
