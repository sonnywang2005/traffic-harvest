"""
Step 3: fit the congestion model and compare it with the no-model baseline.

    python analysis/fit_model.py [--seed 0] [--boot 200] [--t0-q 0.05]

Sections printed:
  1. Verification      -- refit synthetic data from the model; recover (A, beta)
  2. Fit               -- A per pair, one shared beta, on training weekdays
  3. Held-out score    -- model vs no-model baselines, same metric, same days
  4. Assumption test   -- shared beta vs per-pair beta
  5. t0 sensitivity    -- refit at the 2nd / 5th / 10th percentile
  6. Bootstrap         -- 90% intervals on A and beta from day resampling
  7. Weekend check     -- scale traffic down and predict weekend times
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import congestion_model as cm  # noqa: E402

TRAIN = [dt.date(2026, 11, d) for d in (12, 13, 16, 17, 18, 19, 20)]
TEST = ([dt.date(2026, 11, d) for d in (23, 24, 30)]
        + [dt.date(2026, 12, d) for d in (1, 2, 3)])


def mae(df, pred):
    return cm.score(df.duration_s.values, pred)["MAE_min"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/travel_times.csv")
    ap.add_argument("--ms2", default="data/ms2_volumes.csv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--t0-q", type=float, default=0.05)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    print(f"seed = {args.seed}   t0 quantile = {args.t0_q}")

    df = cm.load(args.data)
    weekday = df[~df.weekend]
    train, test = cm.split_by_date(weekday, TRAIN, TEST)
    print(f"rows: {len(df)} total, {len(train)} train, {len(test)} test; "
          f"test dates present: {[str(d) for d in sorted(test.date.unique())]}")

    profile = cm.profile_from_ms2(args.ms2)
    tbl = profile.table
    print(f"MS2 profile: {len(tbl)} intervals, noon {tbl.iloc[0]:.0f} -> peak "
          f"{tbl.max():.0f} at {12 + tbl.idxmax() // 60}:{tbl.idxmax() % 60:02d} "
          f"({tbl.max() / tbl.iloc[0] - 1:+.0%})\n")
    t0 = cm.free_flow_quantile(train, args.t0_q)

    # 1. Verification -----------------------------------------------------
    print("== 1. Verification: recover known (A, beta) from synthetic data ==")
    truth = [cm.PairFit(p, float(t0[p]), a, 2.0, True, 0)
             for p, a in [("harvard_to_porter", 0.30), ("porter_to_alewife", 0.60),
                          ("porter_to_davis", 0.15)]]
    synth = cm.simulate(truth, TRAIN, profile, noise_sd=0.05, rng=rng)
    rec = cm.fit_shared_beta(synth, t0, profile)
    for tr, rf in zip(truth, rec):
        print(f"  {tr.pair:20s} A {tr.a:.2f} -> {rf.a:.3f}   success={rf.success}")
    print(f"  shared beta 2.0 -> {rec[0].beta:.3f}")
    zero = cm.travel_time(100.0, 0.0, 2.0, profile(np.arange(0, 360, 10)))
    assert np.allclose(zero, 100.0)
    print("  A=0 limit returns t0 at every slot: OK\n")

    # 2. Fit ----------------------------------------------------------------
    print("== 2. Fit on training weekdays (shared beta) ==")
    fits = cm.fit_shared_beta(train, t0, profile)
    print(f"  beta = {fits[0].beta:.3f}   success={fits[0].success}")
    for f in fits:
        print(f"  {f.pair:20s} t0 {f.t0/60:5.2f} min   A {f.a:.3f}   "
              f"peak delay {f.t0*f.a/60:.2f} min   n={f.n}")
    print()

    # 3. Held-out comparison ------------------------------------------------
    print("== 3. Held-out weekdays: model vs no-model baselines (MAE, minutes) ==")
    if len(test):
        rows = {"model (shared beta)": cm.predict(test, fits, profile),
                "baseline: same-slot mean of train days": cm.baseline_slot_mean(train, test),
                "baseline: Mapbox typical": cm.baseline_mapbox_typical(test)}
        for name, pred in rows.items():
            s = cm.score(test.duration_s.values, pred)
            print(f"  {name:42s} MAE {s['MAE_min']:.2f}   RMSE {s['RMSE_min']:.2f}   "
                  f"bias {s['bias_min']:+.2f}")
        print("  by test day (model / slot-mean):")
        for d, g in test.groupby("date"):
            print(f"    {d}  {mae(g, cm.predict(g, fits, profile)):.2f} / "
                  f"{mae(g, cm.baseline_slot_mean(train, g)):.2f}   (n={len(g)})")
        print("  by pair (model / slot-mean):")
        for p, g in test.groupby("pair"):
            print(f"    {p:20s} {mae(g, cm.predict(g, fits, profile)):.2f} / "
                  f"{mae(g, cm.baseline_slot_mean(train, g)):.2f}")
    print()

    # 4. Assumption test: shared beta ------------------------------------------
    print("== 4. Assumption test: one beta for all pairs vs a beta per pair ==")
    pp = cm.fit_per_pair(train, t0, profile)
    print("  per-pair beta:", "  ".join(f"{f.pair.split('_to_')[0][:4]}>{f.pair.split('_to_')[1][:4]} "
                                        f"{f.beta:.2f}" for f in pp))
    print(f"  training SSE (log): shared {cm.training_sse(train, fits, profile):.3f}   "
          f"per-pair {cm.training_sse(train, pp, profile):.3f}   "
          f"(per-pair has {len(pp) - 1} extra parameters)")
    if len(test):
        print(f"  held-out MAE: shared {mae(test, cm.predict(test, fits, profile)):.3f}   "
              f"per-pair {mae(test, cm.predict(test, pp, profile)):.3f}")
    print()

    # 5. t0 sensitivity ---------------------------------------------------------
    print("== 5. Sensitivity to the t0 quantile ==")
    for q in (0.02, 0.05, 0.10):
        t0q = cm.free_flow_quantile(train, q)
        fq = cm.fit_shared_beta(train, t0q, profile)
        line = f"  q={q:.2f}  beta {fq[0].beta:.2f}  mean A {np.mean([f.a for f in fq]):.3f}"
        if len(test):
            line += f"  held-out MAE {mae(test, cm.predict(test, fq, profile)):.3f}"
        print(line)
    mb = cm.free_flow_mapbox()
    print("  Mapbox no-traffic t0 vs observed 5%:", "  ".join(
        f"{p[:12]} {mb[p]/t0[p]:.2f}x" for p in t0.index))
    print()

    # 6. Bootstrap by day -------------------------------------------------------
    print(f"== 6. Bootstrap over training days ({args.boot} resamples), 90% intervals ==")
    boot = cm.bootstrap_by_day(train, profile, args.boot, rng, t0_q=args.t0_q)
    blo, bhi = cm.interval(boot.groupby("boot").beta.first().values)
    print(f"  beta {fits[0].beta:.3f} [{blo:.3f}, {bhi:.3f}]")
    for p, g in boot.groupby("pair"):
        f = next(x for x in fits if x.pair == p)
        alo, ahi = cm.interval(g.a.values)
        print(f"  {p:20s} A {f.a:.3f} [{alo:.3f}, {ahi:.3f}]")
    print()

    # 7. Weekend check -------------------------------------------------------------
    print("== 7. Weekend check: scale weekday traffic by s, predict weekend times ==")
    wk = df[df.weekend]
    scales = np.linspace(0.3, 1.0, 29)
    maes = [mae(wk, cm.predict(wk, fits, profile, s)) for s in scales]
    i = int(np.argmin(maes))
    print(f"  weekend rows {len(wk)}; best s = {scales[i]:.2f} (MAE {maes[i]:.2f}) vs "
          f"s = 1 (MAE {maes[-1]:.2f}) vs weekday slot-mean baseline "
          f"(MAE {mae(wk, cm.baseline_slot_mean(train, wk)):.2f})")
    print("  NOTE: s is tuned on the weekend data, so this is a fit, not a validation, "
          "until s comes from an outside source (a weekend count at the same station).")


if __name__ == "__main__":
    main()
