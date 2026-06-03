"""
Export per-episode empirical calibration targets for the ABM (schema v1).

For each built episode we measure, from the real 1-min peg-deviation series:
  - ou_half_life_min   : mean-reversion half-life of the trigger's peg deviation
  - peak_depeg_bps     : max |dev| over active non-origin nodes (contagion magnitude)
  - propagation_rho    : fraction of active non-origin nodes that became stressed
  - propagation_lag_min: median minutes from origin onset to first victim onset
  - shock_duration_min : minutes the origin stayed stressed

Writes exports/calibration_v1.csv (+ JSON sidecar).  This is the second half of
the Repo1 -> Repo2 contract (the first is the hub ranking).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from scgnn.data.dataset import list_episodes, load_episode  # noqa: E402
from scgnn.features.labels import _stress_indicator  # noqa: E402

ASSET_CLASS = {"USDC": "fiat_backed", "USDT": "fiat_backed", "TUSD": "fiat_backed",
               "BUSD": "fiat_backed", "USDP": "fiat_backed", "FDUSD": "fiat_backed",
               "DAI": "crypto_backed", "FRAX": "crypto_backed", "UST": "synthetic"}


def ou_half_life_min(price: pd.Series) -> float:
    """AR(1) mean-reversion half-life (minutes) of a price series around its mean."""
    p = price.dropna()
    if len(p) < 60:
        return float("nan")
    lag = p.shift(1).dropna()
    delta = (p.diff()).dropna()
    lag, delta = lag.align(delta, join="inner")
    if lag.std() == 0:
        return float("nan")
    slope = np.polyfit(lag.values, delta.values, 1)[0]
    if slope >= 0:
        return float("nan")
    return float(-np.log(2) / slope)


def main():
    cfg = yaml.safe_load(open("configs/experiment.yaml"))
    thr_map = cfg["labels"]["thresholds_bps"]
    sustained = cfg["labels"]["sustained_min"]
    rows = []
    for name in list_episodes():
        b = load_episode(name)
        idx = pd.to_datetime(b["index_1m_ms"], unit="ms", utc=True)
        origin = b["origin"]
        active = b["active_node_strs"]
        dev = {ns: pd.Series(b["dev_bps_1m"][ns], index=idx) for ns in active}

        def thr(ns):
            return float(thr_map[ASSET_CLASS.get(ns.split("/")[0], "fiat_backed")])

        # origin onset + duration
        origin_onset, shock_dur = None, 0.0
        if origin in dev:
            os_ind = _stress_indicator(dev[origin], thr(origin), sustained)
            shock_dur = float(os_ind.sum())
            on = os_ind[os_ind == 1].index
            origin_onset = on[0] if len(on) else None

        # victims
        victims = [ns for ns in active if ns != origin]
        stressed_flags, lags = [], []
        for ns in victims:
            s_ind = _stress_indicator(dev[ns], thr(ns), sustained)
            became = bool(s_ind.sum() > 0)
            stressed_flags.append(became)
            if became and origin_onset is not None:
                von = s_ind[s_ind == 1].index
                vfirst = von[von > origin_onset]
                if len(vfirst):
                    lags.append((vfirst[0] - origin_onset).total_seconds() / 60.0)

        prop_rho = float(np.mean(stressed_flags)) if stressed_flags else 0.0
        prop_lag = float(np.median(lags)) if lags else float("nan")
        peak_depeg = float(max((dev[ns].abs().max() for ns in victims), default=0.0))
        # OU half-life of origin price (price = dev/1e4 + 1)
        ou = ou_half_life_min(dev[origin] / 1e4 + 1.0) if origin in dev else float("nan")

        rows.append({
            "episode": name, "asset": (origin or "NA").split("/")[0],
            "venue": (origin or "NA/NA").split("/")[1] if origin else "all",
            "ou_half_life_min": round(ou, 2) if ou == ou else None,
            "propagation_rho": round(prop_rho, 4),
            "propagation_lag_min": round(prop_lag, 2) if prop_lag == prop_lag else None,
            "peak_depeg_bps": round(peak_depeg, 2),
            "shock_duration_min": shock_dur,
            "n_victims": len(victims),
            "episode_tag": name, "is_real": True,
        })

    df = pd.DataFrame(rows)
    Path("exports").mkdir(exist_ok=True)
    df.to_csv("exports/calibration_v1.csv", index=False)
    Path("exports/calibration_v1.json").write_text(json.dumps({
        "schema_version": "1", "n_episodes": len(df),
        "rows": df.to_dict(orient="records")}, indent=2))
    print(df.to_string(index=False))
    print("\nWrote exports/calibration_v1.csv")


if __name__ == "__main__":
    main()
