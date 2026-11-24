"""
Figures and a results table for the write-up, built on congestion_model.py.

    python analysis/make_figures.py [--pair harvard_to_porter]

Saves:
  results/fit_results.csv        t0, A, peak delay, held-out MAE (model and baseline) per pair
  results/model_vs_baseline.png  one pair: training-day average, model curve, test days
  results/mae_by_pair.png        held-out MAE per pair, model vs slot-mean baseline
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import congestion_model as cm  # noqa: E402
from fit_model import TRAIN, TEST  # same split as the main analysis  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/travel_times.csv")
ap.add_argument("--ms2", default="data/ms2_volumes.csv")
ap.add_argument("--pair", default="harvard_to_porter")
args = ap.parse_args()
Path("results").mkdir(exist_ok=True)

df = cm.load(args.data)
train, test = cm.split_by_date(df[~df.weekend], TRAIN, TEST)
profile = cm.profile_from_ms2(args.ms2)
t0 = cm.free_flow_quantile(train, 0.05)
fits = cm.fit_shared_beta(train, t0, profile)
beta = fits[0].beta

# ---- results table ----
rows = []
for f in fits:
    g = test[test.pair == f.pair]
    row = {"pair": f.pair, "t0_min": f.t0 / 60, "A": f.a, "beta": beta,
           "peak_delay_min": f.t0 * f.a / 60}
    if len(g):
        row["model_MAE_min"] = cm.score(g.duration_s.values, cm.predict(g, fits, profile))["MAE_min"]
        row["baseline_MAE_min"] = cm.score(g.duration_s.values, cm.baseline_slot_mean(train, g))["MAE_min"]
    rows.append(row)
table = pd.DataFrame(rows).round(3)
table.to_csv("results/fit_results.csv", index=False)
print(table.to_string(index=False))

# ---- figure 1: one pair over the afternoon ----
slots = np.arange(0, 360, 10)
f = next(x for x in fits if x.pair == args.pair)
avg = train[train.pair == args.pair].groupby("slot").duration_s.mean() / 60
curve = cm.travel_time(f.t0, f.a, beta, profile(slots)) / 60
hours = 12 + slots / 60

plt.figure(figsize=(9, 5))
for d, g in test[test.pair == args.pair].groupby("date"):
    plt.plot(12 + g.slot / 60, g.duration_s / 60, color="lightgray", linewidth=1)
plt.plot([], [], color="lightgray", label="Test days (observed)")
plt.plot(12 + avg.index / 60, avg.values, marker="o", markersize=3,
         label="Baseline: training-day average")
plt.plot(hours, curve, linewidth=2.5, label=f"Model (shared beta = {beta:.2f})")
plt.axhline(f.t0 / 60, linestyle="--", color="gray", label="Empty-road time t0")
plt.xticks(range(12, 19), [f"{h}:00" for h in range(12, 19)])
plt.xlabel("Time of day (ET)")
plt.ylabel("Travel time (minutes)")
plt.title(f"{args.pair}: model vs baseline")
plt.legend()
plt.tight_layout()
plt.savefig("results/model_vs_baseline.png", dpi=200)

# ---- figure 2: held-out error by pair ----
if "model_MAE_min" in table:
    t = table.sort_values("baseline_MAE_min")
    y = np.arange(len(t))
    plt.figure(figsize=(8, 5))
    plt.barh(y - 0.2, t.baseline_MAE_min, height=0.4, label="Baseline (slot mean)")
    plt.barh(y + 0.2, t.model_MAE_min, height=0.4, label="Model")
    plt.yticks(y, t.pair)
    plt.xlabel("Held-out mean absolute error (minutes)")
    plt.title("Held-out error by pair")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/mae_by_pair.png", dpi=200)

print("Saved results/fit_results.csv and figures in results/")
