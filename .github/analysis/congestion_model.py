"""
Congestion model for the traffic-harvest travel times.

Model (BPR-style)
-----------------
    t(pair, slot) = t0[pair] * (1 + A[pair] * r(slot) ** beta)

where
    t0    free-flow travel time, seconds: the fastest 5% of the pair's own
          training readings (`free_flow_quantile`)
    r     relative traffic level in [0, 1], busiest 15-min interval = 1,
          from the MS2 station counts (`profile_from_ms2`); shape only,
          one profile shared by all pairs
    A     delay at peak traffic as a fraction of t0, one per pair (fitted)
    beta  how sharply delay rises with traffic, SHARED by all pairs (fitted)

Because only the shape of r is known, the volume scale and the usual BPR
coefficient are not separately identifiable: A is their product.

`fit_per_pair` fits a separate beta for every pair; it exists to TEST the
shared-beta assumption, not as the reported model.

Fitting is least squares on log(t / t0), the Gaussian maximum-likelihood
fit with multiplicative noise.  Uncertainty comes from a bootstrap over
days (10-minute samples within one afternoon are not independent), with
t0 recomputed inside each resample.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

SLOT_MINUTES = 10
NOON = 12 * 60          # minutes after midnight
END = 18 * 60


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load(path: str) -> pd.DataFrame:
    """Load the harvested CSV and snap each row to the 10-minute grid.

    Parameters
    ----------
    path : str
        Path to ``data/travel_times.csv``.

    Returns
    -------
    pandas.DataFrame
        One row per (date, pair, slot) with columns ``date``, ``pair``,
        ``slot`` (minutes after noon, 0..350), ``weekend`` (bool),
        ``duration_s`` and ``duration_typical_s``.  Rows outside noon-6pm
        and duplicate (date, pair, slot) rows (from off-schedule runs) are
        dropped, keeping the first.
    """
    df = pd.read_csv(path, parse_dates=["timestamp_et"])
    minutes = df.timestamp_et.dt.hour * 60 + df.timestamp_et.dt.minute
    df["slot"] = (minutes - NOON) // SLOT_MINUTES * SLOT_MINUTES
    df["date"] = df.timestamp_et.dt.date
    df["weekend"] = df.timestamp_et.dt.weekday >= 5
    df = df[(df.slot >= 0) & (df.slot < END - NOON)]
    df = df.drop_duplicates(["date", "pair", "slot"], keep="first")
    return df[["date", "weekend", "pair", "slot", "duration_s",
               "duration_typical_s"]].reset_index(drop=True)


def split_by_date(df: pd.DataFrame, train_dates, test_dates):
    """Return (train, test) frames restricted to the given dates."""
    train = df[df.date.isin(set(train_dates))]
    test = df[df.date.isin(set(test_dates))]
    return train, test


# --------------------------------------------------------------------------- #
# Inputs: free-flow time and volume profile
# --------------------------------------------------------------------------- #
def free_flow_quantile(train: pd.DataFrame, q: float = 0.05) -> pd.Series:
    """t0 per pair: the q-quantile of that pair's *training* readings, seconds.

    The group's choice is q = 0.05 ("fastest 5%").  Use training rows only so
    the held-out comparison stays clean.  `q` is a modeling choice; the runner
    reports the fit's sensitivity to it.
    """
    return train.groupby("pair").duration_s.quantile(q)


def free_flow_mapbox(path: str = "data/freeflow.csv") -> pd.Series:
    """Mapbox no-traffic times from freeflow.py (kept for comparison only).

    These came out *higher* than observed fastest trips.  Note the route
    summaries in that file differ from the live ones for several pairs: the
    no-traffic profile routes differently, so it is not the same trip.
    """
    t = pd.read_csv(path)
    return t.set_index("pair")["freeflow_s"]


def profile_from_ms2(path: str = "data/ms2_volumes.csv"):
    """Relative weekday volume v(slot) from the MS2 15-minute station counts.

    Averages the available count days by interval, keeps noon to 6 pm,
    normalizes so the busiest interval is 1, and interpolates to any slot
    (minutes after noon).  Shape only: the station is ~1.5 miles from the
    pairs and the counts are from 2017.
    """
    d = pd.read_csv(path)
    hh, mm = zip(*(s.split(":") for s in d.interval_start))
    d["min_after_noon"] = (np.array(hh, int) - 12) * 60 + np.array(mm, int)
    d = d[(d.min_after_noon >= 0) & (d.min_after_noon < END - NOON)]
    prof = d.groupby("min_after_noon").volume_15min.mean()
    x = prof.index.values.astype(float)
    y = prof.values / prof.values.max()

    def v(slot_min):
        return np.interp(slot_min, x, y)
    v.table = prof
    return v


def placeholder_profile(slot_min: np.ndarray) -> np.ndarray:
    """Synthetic weekday shape, kept only for the verification cell."""
    knots_min = np.array([0, 120, 180, 240, 300, 330, 359])
    knots_v = np.array([0.60, 0.62, 0.70, 0.85, 1.00, 0.97, 0.90])
    return np.interp(slot_min, knots_min, knots_v)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def travel_time(t0: float, a: float, beta: float, v: np.ndarray) -> np.ndarray:
    """BPR-style travel time: t0 * (1 + a * v**beta)."""
    return t0 * (1.0 + a * np.power(v, beta))


@dataclass
class PairFit:
    pair: str
    t0: float
    a: float
    beta: float
    success: bool
    n: int


def fit_pair(t_obs: np.ndarray, v: np.ndarray, t0: float) -> tuple[float, float, bool]:
    """Fit (a, beta) for one pair by least squares on log(t / t0).

    Parameters are optimized in log space so both stay positive, with bounds
    rather than clipping (Ch. 2 exercise 5).  Returns (a, beta, success).
    """
    y = np.log(t_obs / t0)

    def nll(p):
        a, beta = np.exp(p)
        return np.sum((y - np.log1p(a * np.power(v, beta))) ** 2)

    best = None
    for a0, b0 in [(0.3, 1.0), (0.3, 3.0), (1.0, 2.0), (0.1, 0.5)]:
        res = minimize(nll, np.log([a0, b0]), method="L-BFGS-B",
                       bounds=[(np.log(1e-4), np.log(20)), (np.log(0.05), np.log(10))])
        if best is None or res.fun < best.fun:
            best = res
    a, beta = np.exp(best.x)
    return float(a), float(beta), bool(best.success)


def fit_per_pair(train: pd.DataFrame, t0: pd.Series, profile) -> list[PairFit]:
    """Fit (A, beta) separately for every pair (the assumption-test model)."""
    fits = []
    for pair, g in train.groupby("pair"):
        v = profile(g.slot.values)
        a, beta, ok = fit_pair(g.duration_s.values, v, t0[pair])
        fits.append(PairFit(pair, float(t0[pair]), a, beta, ok, len(g)))
    return fits


def fit_shared_beta(train: pd.DataFrame, t0: pd.Series, profile) -> list[PairFit]:
    """The group's model: one A per pair, one beta shared by all pairs.

    Joint least squares on log(t / t0) over every row of every pair.
    Parameters are log(A_1..A_P, beta); L-BFGS-B with bounds.
    """
    pairs = sorted(train.pair.unique())
    idx = {p: i for i, p in enumerate(pairs)}
    k = train.pair.map(idx).values
    y = np.log(train.duration_s.values / train.pair.map(t0).values)
    v = profile(train.slot.values)
    P = len(pairs)

    def sse(p):
        a = np.exp(p[:P])[k]
        beta = np.exp(p[P])
        return np.sum((y - np.log1p(a * np.power(v, beta))) ** 2)

    best = None
    for b0 in (0.5, 1.0, 2.0, 4.0):
        x0 = np.log(np.r_[np.full(P, 0.3), b0])
        res = minimize(sse, x0, method="L-BFGS-B",
                       bounds=[(np.log(1e-4), np.log(20))] * P + [(np.log(0.05), np.log(10))])
        if best is None or res.fun < best.fun:
            best = res
    a = np.exp(best.x[:P])
    beta = float(np.exp(best.x[P]))
    n = train.groupby("pair").size()
    return [PairFit(p, float(t0[p]), float(a[idx[p]]), beta, bool(best.success), int(n[p]))
            for p in pairs]


fit_all = fit_shared_beta   # the default model


def training_sse(train: pd.DataFrame, fits: list[PairFit], profile) -> float:
    """Sum of squared log residuals on the training rows (for model comparison)."""
    pred = predict(train, fits, profile)
    return float(np.sum((np.log(train.duration_s.values) - np.log(pred)) ** 2))


def predict(df: pd.DataFrame, fits: list[PairFit], profile, volume_scale: float = 1.0) -> np.ndarray:
    """Model prediction, seconds, for each row of `df`.

    `volume_scale` multiplies the profile (e.g. < 1 for weekends).
    """
    by_pair = {f.pair: f for f in fits}
    out = np.empty(len(df))
    for i, (pair, slot) in enumerate(zip(df.pair.values, df.slot.values)):
        f = by_pair[pair]
        out[i] = travel_time(f.t0, f.a, f.beta, volume_scale * profile(np.array([slot])))[0]
    return out


# --------------------------------------------------------------------------- #
# Baselines and scoring
# --------------------------------------------------------------------------- #
def baseline_slot_mean(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """No-model baseline: mean of earlier days, same pair, same 10-minute slot."""
    table = train.groupby(["pair", "slot"]).duration_s.mean()
    return np.array([table[(p, s)] for p, s in zip(test.pair, test.slot)])


def baseline_mapbox_typical(test: pd.DataFrame) -> np.ndarray:
    """Second no-model baseline: Mapbox's own typical time for the slot."""
    return test.duration_typical_s.values


def score(y_true_s: np.ndarray, y_pred_s: np.ndarray) -> dict:
    """MAE and RMSE in minutes, plus mean signed error (bias)."""
    e = (y_pred_s - y_true_s) / 60.0
    return {"MAE_min": float(np.mean(np.abs(e))),
            "RMSE_min": float(np.sqrt(np.mean(e ** 2))),
            "bias_min": float(np.mean(e))}


# --------------------------------------------------------------------------- #
# Verification and uncertainty
# --------------------------------------------------------------------------- #
def simulate(fits: list[PairFit], dates, profile, noise_sd: float, rng: np.random.Generator) -> pd.DataFrame:
    """Generate synthetic data from the model with lognormal noise.

    Used for verification: refitting must recover the known (a, beta).
    """
    rows = []
    slots = np.arange(0, END - NOON, SLOT_MINUTES)
    for d in dates:
        for f in fits:
            t = travel_time(f.t0, f.a, f.beta, profile(slots))
            t = t * np.exp(noise_sd * rng.standard_normal(len(slots)))
            rows += [(d, False, f.pair, s, ts, np.nan) for s, ts in zip(slots, t)]
    return pd.DataFrame(rows, columns=["date", "weekend", "pair", "slot",
                                       "duration_s", "duration_typical_s"])


def bootstrap_by_day(train: pd.DataFrame, profile, n_boot: int,
                     rng: np.random.Generator, fitter=fit_shared_beta,
                     t0_q: float = 0.05) -> pd.DataFrame:
    """Refit on day-resampled training sets; rows are (boot, pair, a, beta).

    t0 is recomputed inside each resample, so its sampling noise is included.
    """
    days = sorted(train.date.unique())
    out = []
    for b in range(n_boot):
        pick = rng.choice(days, size=len(days), replace=True)
        boot = pd.concat([train[train.date == d] for d in pick])
        for f in fitter(boot, free_flow_quantile(boot, t0_q), profile):
            out.append((b, f.pair, f.a, f.beta))
    return pd.DataFrame(out, columns=["boot", "pair", "a", "beta"])


def interval(x: np.ndarray, level: float = 0.90) -> tuple[float, float]:
    lo, hi = np.quantile(x, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)
