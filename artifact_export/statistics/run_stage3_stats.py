#!/usr/bin/env python3
"""
Stage 3 - Statistical analysis of the Stage 2 restoration benchmark.

Paired image-level analysis (every config processed the SAME 500 images):
  - Friedman omnibus test across all configs, per metric (+ Kendall's W effect size)
  - Post-hoc pairwise Wilcoxon signed-rank, Holm AND Benjamini-Hochberg corrected
  - Matched-pairs rank-biserial effect size + median paired difference per pair
  - Bootstrap 95% CIs for each config mean; mean Friedman ranks
  - Image-level distribution figures (boxplots) + mean-rank figure
Outputs land in Stage 3/{results,figures,data} and Stage 3/STAGE3_STATISTICAL_ANALYSIS.md
"""
import os, io, csv, zipfile, json
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S2   = os.path.join(ROOT, "Stage 2")
OUT  = os.path.join(ROOT, "Stage 3")
RES  = os.path.join(OUT, "results"); FIG = os.path.join(OUT, "figures"); DAT = os.path.join(OUT, "data")
for d in (RES, FIG, DAT): os.makedirs(d, exist_ok=True)

# ---------- readers ----------
def read_csv_text(text):
    r = csv.reader(io.StringIO(text)); rows = list(r); hdr = rows[0]
    idx = {c: hdr.index(c) for c in ("relative_path", "psnr", "ssim", "lpips")}
    out = {}
    for row in rows[1:]:
        if not row: continue
        out[row[idx["relative_path"]]] = (
            float(row[idx["psnr"]]), float(row[idx["ssim"]]), float(row[idx["lpips"]]))
    return out

def from_zip(zip_path, member):
    with zipfile.ZipFile(os.path.join(S2, zip_path)) as z:
        return read_csv_text(z.read(member).decode())

def from_file(path):
    with open(os.path.join(S2, path), encoding="utf-8") as f:
        return read_csv_text(f.read())

def merge(dicts):
    out = {}
    for d in dicts:
        for k, v in d.items():
            assert k not in out, f"duplicate key {k} across batches"
            out[k] = v
    return out

# ---------- assemble the 8 primary configs (all scored at 256) ----------
dit_b   = [from_zip(f"DiT Alone Outputs/dit_metrics_b{i}.zip", "metrics/dit_alone_per_image.csv") for i in (1, 2, 3)]
nore_b  = [from_zip(f"DiT Alone Outputs/dit_metrics_b{i}.zip", "metrics/no_restoration_per_image.csv") for i in (1, 2, 3)]

CONFIGS = {
    "No restoration": merge(nore_b),
    "RCDNet":         from_zip("RCDNet Alone Outputs/metrics.zip", "metrics/rcdnet_alone_per_image.csv"),
    "AT-GAN":         from_zip("Atgan Alone Outputs/metrics.zip", "metrics/atgan_per_image.csv"),
    "IDT":            from_zip("IDT Alone Outputs/metrics.zip", "metrics/IDT_per_image.csv"),
    "Restormer":      from_file("Restormer Alone Outputs/Restormer_256_Combined/metrics/restormer_per_image.csv"),
    "DiT":            merge(dit_b),
    "RCDNet->DiT":    from_file("RCDNet_DiT_Outputs/RCDNet_DiT_Combined/metrics/rcdnet_then_dit_per_image.csv"),
    "DiT->RCDNet":    from_zip("DiT_RCDNet_Outputs/metrics.zip", "metrics/dit_then_rcdnet_per_image.csv"),
}
ORDER = list(CONFIGS.keys())
# auxiliary: Restormer native 128 (resolution control)
RESTORMER_128 = from_zip("Restormer Alone Outputs/metrics_128.zip", "metrics/restormer_per_image.csv")

# ---------- align on common image keys ----------
keysets = [set(d.keys()) for d in CONFIGS.values()]
common = set.intersection(*keysets)
for name, d in CONFIGS.items():
    print(f"  {name:14s} rows={len(d)}")
print(f"common images across all configs: {len(common)}")
KEYS = sorted(common)
N = len(KEYS)
assert N == 500, f"expected 500 aligned images, got {N}"

METRICS = [("psnr", 0, "higher"), ("ssim", 1, "higher"), ("lpips", 2, "lower")]
# data[metric] -> (k_configs x N) array
data = {}
for m, mi, _ in METRICS:
    data[m] = np.array([[CONFIGS[c][k][mi] for k in KEYS] for c in ORDER])  # shape (K, N)

# ---------- descriptive stats + bootstrap 95% CI ----------
rng = np.random.default_rng(42)
def boot_ci(x, nboot=10000):
    means = rng.choice(x, size=(nboot, len(x)), replace=True).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])

desc_rows = []
for m, mi, _ in METRICS:
    for ci, c in enumerate(ORDER):
        x = data[m][ci]
        lo, hi = boot_ci(x)
        desc_rows.append([c, m, f"{x.mean():.4f}", f"{x.std(ddof=1):.4f}",
                          f"{np.median(x):.4f}", f"{lo:.4f}", f"{hi:.4f}"])
with open(os.path.join(RES, "descriptive_stats.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["config", "metric", "mean", "std", "median", "ci95_low", "ci95_high"])
    w.writerows(desc_rows)

# ---------- Friedman + Kendall's W + mean ranks ----------
K = len(ORDER)
fried_rows, meanrank = [], {}
for m, mi, direction in METRICS:
    arrs = [data[m][ci] for ci in range(K)]
    chi2, p = stats.friedmanchisquare(*arrs)
    W = chi2 / (N * (K - 1))
    fried_rows.append([m, f"{chi2:.3f}", K - 1, f"{p:.3e}", f"{W:.4f}", N])
    # per-image ranks: rank 1 = best.  higher-better -> rank of -value ; lower-better -> rank of value
    M = data[m].T  # (N, K)
    signed = -M if direction == "higher" else M
    ranks = np.apply_along_axis(stats.rankdata, 1, signed)  # (N,K), 1=best
    meanrank[m] = ranks.mean(axis=0)  # per config
with open(os.path.join(RES, "friedman.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["metric", "chi2", "df", "p_value", "kendalls_w", "n_images"])
    w.writerows(fried_rows)
with open(os.path.join(RES, "mean_ranks.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["config"] + [m for m, _, _ in METRICS])
    for ci, c in enumerate(ORDER):
        w.writerow([c] + [f"{meanrank[m][ci]:.3f}" for m, _, _ in METRICS])

# ---------- corrections ----------
def holm(pvals):
    p = np.asarray(pvals); order = np.argsort(p); adj = np.empty_like(p); prev = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]; prev = max(prev, val); adj[idx] = min(prev, 1.0)
    return adj
def bh(pvals):
    p = np.asarray(pvals); m = len(p); order = np.argsort(p); adj = np.empty_like(p)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        idx = order[i]; val = p[idx] * m / (i + 1); prev = min(prev, val); adj[idx] = min(prev, 1.0)
    return adj

def rank_biserial(a, b):
    d = a - b; d = d[d != 0]
    if len(d) == 0: return 0.0, 0.0, 0.0
    r = stats.rankdata(np.abs(d)); Wp = r[d > 0].sum(); Wm = r[d < 0].sum()
    return (Wp - Wm) / (Wp + Wm), Wp, Wm

def mag(r):
    a = abs(r)
    return ("negligible" if a < 0.1 else "small" if a < 0.3 else
            "medium" if a < 0.5 else "large")

pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]
pairwise = {}
for m, mi, direction in METRICS:
    praw, rows = [], []
    for i, j in pairs:
        a, b = data[m][i], data[m][j]
        stat, p = stats.wilcoxon(a, b)  # two-sided
        rrb, _, _ = rank_biserial(a, b)
        meddiff = np.median(a - b)
        # winner (accounting for metric direction)
        if direction == "higher":
            winner = ORDER[i] if meddiff > 0 else ORDER[j]
        else:
            winner = ORDER[i] if meddiff < 0 else ORDER[j]
        praw.append(p)
        rows.append([ORDER[i], ORDER[j], f"{stat:.1f}", p, f"{rrb:+.3f}", mag(rrb),
                     f"{meddiff:+.4f}", winner])
    ph, pb = holm(praw), bh(praw)
    with open(os.path.join(RES, f"wilcoxon_{m}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config_a", "config_b", "wilcoxon_W", "p_raw", "p_holm", "p_bh",
                    "rank_biserial", "effect_mag", "median_diff_a_minus_b", "better"])
        for k, row in enumerate(rows):
            w.writerow(row[:3] + [f"{praw[k]:.3e}", f"{ph[k]:.3e}", f"{pb[k]:.3e}"] + row[4:])
    pairwise[m] = (rows, praw, ph, pb)

# ---------- resolution control: Restormer 256 vs 128 ----------
rc_common = sorted(set(CONFIGS["Restormer"].keys()) & set(RESTORMER_128.keys()))
res_ctrl = {}
for m, mi, direction in METRICS:
    a = np.array([CONFIGS["Restormer"][k][mi] for k in rc_common])
    b = np.array([RESTORMER_128[k][mi] for k in rc_common])
    stat, p = stats.wilcoxon(a, b); rrb, _, _ = rank_biserial(a, b)
    res_ctrl[m] = (a.mean(), b.mean(), np.median(a - b), p, rrb)

# ---------- figures ----------
COL = {"psnr": "PSNR (dB)", "ssim": "SSIM", "lpips": "LPIPS"}
for m, mi, _ in METRICS:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot([data[m][ci] for ci in range(K)], labels=ORDER, showmeans=True, meanline=True)
    ax.set_ylabel(COL[m]); ax.set_title(f"Image-level {COL[m]} distribution across configurations (n={N})")
    ax.tick_params(axis="x", rotation=30)
    for lbl in ax.get_xticklabels(): lbl.set_ha("right")
    ax.grid(axis="y", alpha=.3); fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"box_{m}.png"), dpi=130); plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (m, _, _) in zip(axes, METRICS):
    mr = meanrank[m]; o = np.argsort(mr)
    ax.barh([ORDER[i] for i in o][::-1], [mr[i] for i in o][::-1], color="#4C78A8")
    ax.set_xlabel("mean Friedman rank (1 = best)"); ax.set_title(COL[m]); ax.grid(axis="x", alpha=.3)
fig.suptitle(f"Mean ranks across configurations (n={N}, lower = better)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "mean_ranks.png"), dpi=130); plt.close(fig)

# ---------- write the aligned matrix (reproducibility) ----------
with open(os.path.join(DAT, "aligned_per_image.csv"), "w", newline="") as f:
    w = csv.writer(f)
    head = ["relative_path"]
    for m, _, _ in METRICS:
        head += [f"{c}__{m}" for c in ORDER]
    w.writerow(head)
    for ki, k in enumerate(KEYS):
        row = [k]
        for m, _, _ in METRICS:
            row += [f"{data[m][ci][ki]:.6f}" for ci in range(K)]
        w.writerow(row)

# ---------- dump a JSON of key numbers for the report ----------
summary = {
    "n_images": N, "configs": ORDER,
    "means": {m: {ORDER[ci]: float(data[m][ci].mean()) for ci in range(K)} for m, _, _ in METRICS},
    "friedman": {r[0]: {"chi2": float(r[1]), "df": r[2], "p": r[3], "W": float(r[4])} for r in fried_rows},
    "mean_ranks": {m: {ORDER[ci]: float(meanrank[m][ci]) for ci in range(K)} for m, _, _ in METRICS},
    "res_ctrl": {m: {"mean256": v[0], "mean128": v[1], "med_diff": v[2], "p": v[3], "rrb": v[4]}
                 for m, v in res_ctrl.items()},
}
json.dump(summary, open(os.path.join(RES, "summary.json"), "w"), indent=2)
print("\nDONE. wrote results/, figures/, data/ under Stage 3")
print("Friedman:", {r[0]: r[3] for r in fried_rows})
