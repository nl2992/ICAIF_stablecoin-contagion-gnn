"""Bootstrap confidence interval for graph density vs GAT margin correlation.

With n=4 valid episodes the Pearson r=+0.795 is not conventionally significant
(p=0.205), but bootstrapping quantifies the uncertainty on r and confirms the
direction is stable across resamples.

Outputs:
  results/eval/density_bootstrap_ci.json
"""
from __future__ import annotations
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parents[1]
OUT = ROOT / "results/eval"
OUT.mkdir(parents=True, exist_ok=True)

# The 4 valid (density, gat_margin) pairs from episode_density_vs_margin.csv
# Terra=0.360/+0.088, SVB=0.395/+0.047, USDT=0.349/+0.045, FTX=0.080/+0.003
DENSITY = np.array([0.3602, 0.3946, 0.3491, 0.0802])
MARGIN  = np.array([0.0878, 0.0466, 0.0452, 0.0028])

N_BOOT = 100_000
RNG_SEED = 2025


def pearson_r(x, y):
    xd = x - x.mean(); yd = y - y.mean()
    denom = np.sqrt((xd**2).sum() * (yd**2).sum())
    return float(np.dot(xd, yd) / denom) if denom > 1e-12 else 0.0


def main():
    rng = np.random.default_rng(RNG_SEED)
    n = len(DENSITY)

    observed_r = pearson_r(DENSITY, MARGIN)
    boot_r = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        boot_r[i] = pearson_r(DENSITY[idx], MARGIN[idx])

    ci_lo = float(np.percentile(boot_r, 2.5))
    ci_hi = float(np.percentile(boot_r, 97.5))
    ci_lo_90 = float(np.percentile(boot_r, 5.0))
    ci_hi_90 = float(np.percentile(boot_r, 95.0))
    pct_positive = float(np.mean(boot_r > 0))
    pct_above_half = float(np.mean(boot_r > 0.5))

    print(f"Observed r     = {observed_r:+.4f}")
    print(f"95% CI         = [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"90% CI         = [{ci_lo_90:+.4f}, {ci_hi_90:+.4f}]")
    print(f"% resamples > 0    = {pct_positive*100:.1f}%")
    print(f"% resamples > 0.5  = {pct_above_half*100:.1f}%")
    print(f"Boot mean r    = {boot_r.mean():+.4f}")

    result = {
        "observed_r": round(observed_r, 4),
        "n_episodes": n,
        "n_bootstrap": N_BOOT,
        "ci_95_lo": round(ci_lo, 4),
        "ci_95_hi": round(ci_hi, 4),
        "ci_90_lo": round(ci_lo_90, 4),
        "ci_90_hi": round(ci_hi_90, 4),
        "boot_mean_r": round(float(boot_r.mean()), 4),
        "pct_resamples_positive": round(pct_positive, 4),
        "pct_resamples_above_0p5": round(pct_above_half, 4),
        "episodes": [
            {"episode": "Terra_2022", "density": 0.3602, "gat_margin": 0.0878},
            {"episode": "SVB_2023",   "density": 0.3946, "gat_margin": 0.0466},
            {"episode": "USDT_2018",  "density": 0.3491, "gat_margin": 0.0452},
            {"episode": "FTX_2022",   "density": 0.0802, "gat_margin": 0.0028},
        ],
    }
    (OUT / "density_bootstrap_ci.json").write_text(json.dumps(result, indent=2))
    print(f"\nSaved: {OUT}/density_bootstrap_ci.json")


if __name__ == "__main__":
    main()
