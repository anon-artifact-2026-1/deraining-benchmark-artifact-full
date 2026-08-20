#!/usr/bin/env python3
"""
Stage 4 - Geometry-aware restoration selector.

Learns to route each image to the restoration configuration that gives it the best
result, using ONLY the Stage 1 streak-geometry features. Target = best config per image
under a combined score of normalized PSNR + SSIM + LPIPS (the professor's fallback when
downstream-detection labels are not yet available).

Pure numpy (no sklearn/pandas): a CART decision tree + a bootstrap random forest,
evaluated with stratified 5-fold cross-validation. Reports classification accuracy,
feature importance, confusion matrix, a severity cross-tab, and -- the point of the whole
thing -- how much real image quality per-image routing recovers versus always using the
single best model (with an oracle upper bound).

Outputs: Stage 4/results/*.csv, Stage 4/figures/*.png, Stage 4/STAGE4_SELECTOR.md
"""
import os, csv, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT = os.path.join(ROOT, "Stage 1", "features_500_FINAL.csv")
ALIGN = os.path.join(ROOT, "Stage 3", "data", "aligned_per_image.csv")
OUT  = os.path.join(ROOT, "Stage 4"); RES = os.path.join(OUT, "results"); FIG = os.path.join(OUT, "figures")
for d in (RES, FIG): os.makedirs(d, exist_ok=True)
rng = np.random.default_rng(42)

CONFIGS = ["No restoration","RCDNet","AT-GAN","IDT","Restormer","DiT","RCDNet->DiT","DiT->RCDNet"]
FEATURES = ["longest_streak_height_ratio","avg_streak_width","aspect_ratio","streak_area_coverage",
            "num_streak_components","boundary_contrast","curvature_waviness","confidence_score"]

# ---------------- load features + severity ----------------
frows = {r["relative_path"]: r for r in csv.DictReader(open(FEAT, encoding="utf-8"))}
# ---------------- load per-image metrics for all configs ----------------
arows = {r["relative_path"]: r for r in csv.DictReader(open(ALIGN, encoding="utf-8"))}
KEYS = sorted(set(frows) & set(arows))
assert len(KEYS) == 500, f"expected 500 aligned, got {len(KEYS)}"

X = np.array([[float(frows[k][f]) for f in FEATURES] for k in KEYS])          # (500, 8)
severity = np.array([frows[k]["severity"] for k in KEYS])
# metric cubes: (500 images, 8 configs)
def cube(metric):
    return np.array([[float(arows[k][f"{c}__{metric}"]) for c in CONFIGS] for k in KEYS])
PSNR, SSIM, LPIPS = cube("psnr"), cube("ssim"), cube("lpips")

# ---------------- target: best config per image via normalized combined score ----------------
def z(a): return (a - a.mean()) / a.std()
COMBINED = z(PSNR) + z(SSIM) - z(LPIPS)     # higher = better; LPIPS lower is better
y = COMBINED.argmax(axis=1)                 # label = index of best config per image
K = len(CONFIGS)

# ================= pure-numpy CART decision tree =================
class Tree:
    def __init__(self, max_depth=6, min_leaf=5, max_features=None, rng=None):
        self.max_depth, self.min_leaf, self.max_features, self.rng = max_depth, min_leaf, max_features, rng or np.random.default_rng()
        self.fi = None
    def _gini(self, cnt):
        n = cnt.sum()
        if n == 0: return 0.0
        p = cnt / n
        return 1.0 - (p * p).sum()
    def fit(self, X, y, K):
        self.K = K; self.fi = np.zeros(X.shape[1]); self.N = len(y)
        self.root = self._build(X, y, 0)
        if self.fi.sum() > 0: self.fi /= self.fi.sum()
        return self
    def _build(self, X, y, depth):
        cnt = np.bincount(y, minlength=self.K)
        node = {"leaf": True, "proba": cnt / cnt.sum()}
        if depth >= self.max_depth or len(y) < 2 * self.min_leaf or (cnt > 0).sum() == 1:
            return node
        n, d = X.shape
        feats = np.arange(d)
        if self.max_features:
            feats = self.rng.choice(d, size=min(self.max_features, d), replace=False)
        parent_g = self._gini(cnt); best = None
        for f in feats:
            xs = X[:, f]; order = np.argsort(xs); xs_s = xs[order]; ys_s = y[order]
            uniq = np.unique(xs_s)
            if len(uniq) < 2: continue
            thrs = (uniq[:-1] + uniq[1:]) / 2.0
            for t in thrs:
                lm = xs <= t; rm = ~lm
                nl, nr = lm.sum(), rm.sum()
                if nl < self.min_leaf or nr < self.min_leaf: continue
                gl = self._gini(np.bincount(y[lm], minlength=self.K))
                gr = self._gini(np.bincount(y[rm], minlength=self.K))
                gain = parent_g - (nl * gl + nr * gr) / n
                if best is None or gain > best[0]:
                    best = (gain, f, t, lm, rm)
        if best is None or best[0] <= 1e-12:
            return node
        gain, f, t, lm, rm = best
        self.fi[f] += (n / self.N) * gain
        return {"leaf": False, "f": f, "t": t,
                "L": self._build(X[lm], y[lm], depth + 1),
                "R": self._build(X[rm], y[rm], depth + 1)}
    def _proba1(self, x, node):
        while not node["leaf"]:
            node = node["L"] if x[node["f"]] <= node["t"] else node["R"]
        return node["proba"]
    def predict_proba(self, X): return np.array([self._proba1(x, self.root) for x in X])
    def predict(self, X): return self.predict_proba(X).argmax(axis=1)

class Forest:
    def __init__(self, n_trees=300, max_depth=6, min_leaf=5, max_features=3, seed=42):
        self.n_trees, self.max_depth, self.min_leaf, self.max_features = n_trees, max_depth, min_leaf, max_features
        self.rng = np.random.default_rng(seed)
    def fit(self, X, y, K):
        self.K = K; self.trees = []; self.fi = np.zeros(X.shape[1]); n = len(y)
        for _ in range(self.n_trees):
            idx = self.rng.integers(0, n, n)  # bootstrap
            t = Tree(self.max_depth, self.min_leaf, self.max_features, self.rng).fit(X[idx], y[idx], K)
            self.trees.append(t); self.fi += t.fi
        self.fi /= self.n_trees
        return self
    def predict_proba(self, X):
        P = np.zeros((len(X), self.K))
        for t in self.trees: P += t.predict_proba(X)
        return P / self.n_trees
    def predict(self, X): return self.predict_proba(X).argmax(axis=1)

# ================= stratified 5-fold CV =================
def stratified_folks(y, k=5, seed=42):
    r = np.random.default_rng(seed); folds = [[] for _ in range(k)]
    for c in np.unique(y):
        idx = np.where(y == c)[0]; r.shuffle(idx)
        for i, s in enumerate(idx): folds[i % k].append(s)
    return [np.array(sorted(f)) for f in folds]

folds = stratified_folks(y, 5)
oof_pred = np.zeros(len(y), dtype=int)          # out-of-fold RF predictions
oof_pred_tree = np.zeros(len(y), dtype=int)
for i in range(5):
    test = folds[i]; train = np.concatenate([folds[j] for j in range(5) if j != i])
    rf = Forest(seed=100 + i).fit(X[train], y[train], K)
    oof_pred[test] = rf.predict(X[test])
    dt = Tree(max_depth=4, min_leaf=8).fit(X[train], y[train], K)
    oof_pred_tree[test] = dt.predict(X[test])

# full-data models (for feature importance + an interpretable shallow tree)
rf_full = Forest(seed=7).fit(X, y, K)
dt_full = Tree(max_depth=4, min_leaf=8).fit(X, y, K)

# ================= classification metrics =================
def accuracy(yt, yp): return float((yt == yp).mean())
def macro_f1(yt, yp, K):
    fs = []
    for c in range(K):
        tp = ((yp == c) & (yt == c)).sum(); fp = ((yp == c) & (yt != c)).sum(); fn = ((yp != c) & (yt == c)).sum()
        prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn) if tp + fn else 0.0
        fs.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(fs))

label_counts = np.bincount(y, minlength=K)
majority = label_counts.argmax()
acc_rf, acc_dt = accuracy(y, oof_pred), accuracy(y, oof_pred_tree)
acc_major = accuracy(y, np.full_like(y, majority))
f1_rf = macro_f1(y, oof_pred, K)

# ================= routing value (the real point) =================
# For a per-image choice vector `choice` (config index per image), the achieved metric:
def achieved(choice, cube): return cube[np.arange(len(choice)), choice].mean()
oracle_choice = y                                  # always the true best
single_best = COMBINED.mean(axis=0).argmax()       # best config on average
single_choice = np.full(len(y), single_best)
norest_choice = np.full(len(y), CONFIGS.index("No restoration"))

def block(choice):
    return dict(combined=achieved(choice, COMBINED), psnr=achieved(choice, PSNR),
                ssim=achieved(choice, SSIM), lpips=achieved(choice, LPIPS))
routing = {
    "No restoration": block(norest_choice),
    f"Single best ({CONFIGS[single_best]})": block(single_choice),
    "Selector (RF, cross-validated)": block(oof_pred),
    "Oracle (per-image best)": block(oracle_choice),
}
# fraction of the oracle-over-single-best gain that the selector recovers (on combined score)
gap = routing["Oracle (per-image best)"]["combined"] - routing[f"Single best ({CONFIGS[single_best]})"]["combined"]
recovered = (routing["Selector (RF, cross-validated)"]["combined"]
             - routing[f"Single best ({CONFIGS[single_best]})"]["combined"]) / gap if gap else 0.0

# ================= write result CSVs =================
with open(os.path.join(RES, "best_config_labels.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["relative_path","severity","best_config"] + [f"combined__{c}" for c in CONFIGS])
    for i, k in enumerate(KEYS):
        w.writerow([k, severity[i], CONFIGS[y[i]]] + [f"{COMBINED[i,j]:.4f}" for j in range(K)])

with open(os.path.join(RES, "label_distribution.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["config","times_best","pct"])
    for j in range(K): w.writerow([CONFIGS[j], int(label_counts[j]), f"{100*label_counts[j]/len(y):.1f}"])

with open(os.path.join(RES, "cv_metrics.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["model","accuracy","macro_f1"])
    w.writerow(["RandomForest (5-fold OOF)", f"{acc_rf:.4f}", f"{f1_rf:.4f}"])
    w.writerow(["DecisionTree d4 (5-fold OOF)", f"{acc_dt:.4f}", ""])
    w.writerow(["Majority-class baseline", f"{acc_major:.4f}", ""])

cm = np.zeros((K, K), int)
for t, p in zip(y, oof_pred): cm[t, p] += 1
with open(os.path.join(RES, "confusion_matrix.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["true\\pred"] + CONFIGS)
    for i in range(K): w.writerow([CONFIGS[i]] + list(cm[i]))

order = np.argsort(rf_full.fi)[::-1]
with open(os.path.join(RES, "feature_importance.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["feature","importance"])
    for j in order: w.writerow([FEATURES[j], f"{rf_full.fi[j]:.4f}"])

with open(os.path.join(RES, "routing_value.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["strategy","combined","psnr","ssim","lpips"])
    for name, b in routing.items():
        w.writerow([name, f"{b['combined']:.4f}", f"{b['psnr']:.4f}", f"{b['ssim']:.4f}", f"{b['lpips']:.4f}"])

# severity x best-config cross-tab
sev_levels = ["low","medium","high"]
with open(os.path.join(RES, "severity_crosstab.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["severity"] + CONFIGS + ["n"])
    for s in sev_levels:
        mask = severity == s
        row = [int(((y == j) & mask).sum()) for j in range(K)]
        w.writerow([s] + row + [int(mask.sum())])

# ================= figures =================
plt.rcParams.update({"figure.dpi": 130, "font.size": 9})
# label distribution
fig, ax = plt.subplots(figsize=(9, 4.5))
o = np.argsort(label_counts)[::-1]
ax.bar([CONFIGS[j] for j in o], [label_counts[j] for j in o], color="#4C78A8")
ax.set_ylabel("# images where it is the best config"); ax.set_title("How often each configuration is the per-image winner (n=500)")
ax.tick_params(axis="x", rotation=30)
for l in ax.get_xticklabels(): l.set_ha("right")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "label_distribution.png")); plt.close(fig)
# feature importance
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.barh([FEATURES[j] for j in order][::-1], [rf_full.fi[j] for j in order][::-1], color="#59A14F")
ax.set_xlabel("random-forest importance"); ax.set_title("Which geometry features drive the routing decision")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "feature_importance.png")); plt.close(fig)
# confusion matrix
fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(K)); ax.set_yticks(range(K)); ax.set_xticklabels(CONFIGS, rotation=40, ha="right"); ax.set_yticklabels(CONFIGS)
ax.set_xlabel("predicted"); ax.set_ylabel("true (best config)"); ax.set_title("Selector confusion matrix (5-fold OOF)")
for i in range(K):
    for j in range(K):
        if cm[i,j]: ax.text(j, i, cm[i,j], ha="center", va="center", fontsize=7,
                            color="white" if cm[i,j] > cm.max()/2 else "black")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "confusion_matrix.png")); plt.close(fig)
# routing value (combined score)
fig, ax = plt.subplots(figsize=(8, 4.5))
names = list(routing.keys()); vals = [routing[n]["combined"] for n in names]
cols = ["#9AA5B3","#E15759","#4C78A8","#59A14F"]
ax.bar(range(len(names)), vals, color=cols)
ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=15, ha="right")
ax.set_ylabel("mean combined score (higher = better)"); ax.set_title("Value of per-image routing")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "routing_value.png")); plt.close(fig)

# ================= interpretable shallow-tree rules =================
def dump_tree(node, feat, depth=0, lines=None):
    lines = lines if lines is not None else []
    pad = "    " * depth
    if node["leaf"]:
        j = int(np.argmax(node["proba"])); lines.append(f"{pad}-> {CONFIGS[j]}  (p={node['proba'][j]:.2f})")
    else:
        lines.append(f"{pad}if {feat[node['f']]} <= {node['t']:.3f}:")
        dump_tree(node["L"], feat, depth+1, lines)
        lines.append(f"{pad}else:")
        dump_tree(node["R"], feat, depth+1, lines)
    return lines
tree_rules = "\n".join(dump_tree(dt_full.root, FEATURES))
open(os.path.join(RES, "decision_tree_rules.txt"), "w").write(tree_rules)

# ================= console summary =================
print("label distribution (times best):")
for j in o: print(f"   {CONFIGS[j]:14s} {label_counts[j]:3d}  ({100*label_counts[j]/len(y):.1f}%)")
print(f"\nRF  5-fold accuracy = {acc_rf:.3f}   macro-F1 = {f1_rf:.3f}")
print(f"Tree(d4) accuracy   = {acc_dt:.3f}")
print(f"Majority baseline   = {acc_major:.3f}  (always '{CONFIGS[majority]}')")
print("\nrouting value (mean achieved):")
for n, b in routing.items():
    print(f"   {n:32s} combined={b['combined']:+.3f}  PSNR={b['psnr']:.3f}  SSIM={b['ssim']:.3f}  LPIPS={b['lpips']:.4f}")
print(f"\nselector recovers {100*recovered:.0f}% of the oracle-over-single-best gain (combined score)")
print("\ntop features:", [FEATURES[j] for j in order[:4]])

json.dump({"acc_rf": acc_rf, "f1_rf": f1_rf, "acc_dt": acc_dt, "acc_major": acc_major,
           "single_best": CONFIGS[single_best], "recovered_frac": recovered,
           "routing": routing, "label_counts": {CONFIGS[j]: int(label_counts[j]) for j in range(K)},
           "feat_importance": {FEATURES[j]: float(rf_full.fi[j]) for j in order}},
          open(os.path.join(RES, "summary.json"), "w"), indent=2)
print("\nDONE - results/figures written under Stage 4")
