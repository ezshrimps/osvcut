"""Summarise a seam benchmark JSON: means, 95% bootstrap CIs, paired wins and Wilcoxon p-values."""
import json, sys, itertools
import numpy as np
rows = json.load(open(sys.argv[1], encoding="utf8"))
methods = list(dict.fromkeys(r["method"] for r in rows))
keys = [("cd", -1), ("fid_hard", 1), ("cd_hard", -1), ("edge_hard", -1), ("ghost_ssim", 1), ("edge_dev", -1), ("step", -1), ("ms", -1)]
keys = [(k, s) for k, s in keys if any(k in r for r in rows)]
rng = np.random.default_rng(0)
def ci(x):
    x = np.array([v for v in x if v == v])
    if len(x) == 0: return (float("nan"),) * 3
    bs = [rng.choice(x, len(x)).mean() for _ in range(2000)]
    return x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)
print("| method | n | " + " | ".join(k for k, _ in keys) + " |")
print("|---|---|" + "---|" * len(keys))
for m in methods:
    R = [r for r in rows if r["method"] == m]
    cells = []
    for k, _ in keys:
        mu, lo, hi = ci([r.get(k, float("nan")) for r in R])
        cells.append(f"{mu:.4f} [{lo:.4f}, {hi:.4f}]" if k != "ms" else f"{mu:.0f}")
    print(f"| {m} | {len(R)} | " + " | ".join(cells) + " |")
by = {}
for r in rows: by.setdefault((r["clip"], r["t"]), {})[r["method"]] = r
try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None
print()
ref = sys.argv[2] if len(sys.argv) > 2 else methods[-1]
for k, sgn in keys:
    if k == "ms": continue
    for other in methods:
        if other == ref: continue
        pr = [(v[ref][k], v[other][k]) for v in by.values() if ref in v and other in v and v[ref].get(k) == v[ref].get(k) and v[other].get(k) == v[other].get(k)]
        if not pr: continue
        a, b = np.array(pr).T
        wins = int(((a - b) * sgn > 0).sum())
        p = wilcoxon(a, b).pvalue if wilcoxon and len(pr) > 5 and np.any(a != b) else float("nan")
        print(f"{k:10s} {ref} vs {other:12s}: {wins}/{len(pr)} wins, mean diff {np.mean(a - b):+.4f}, p={p:.2g}")
