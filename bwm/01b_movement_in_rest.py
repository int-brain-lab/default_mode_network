"""Is the BWM 'rest' window (stimOn -1.15..-1.0 s) movement-free?

Downloads only small files per session: _ibl_wheelMoves.intervals and
licks.times (Alyx, BWM sessions of the ~/dmn insertion list). Per trial we
flag wheel movement or a lick inside the rest window, inside the quiescence
window (stimOn -0.4..-0.1) and in a longer ITI window (feedback+1.5 s ..
stimOn-0.5 s). Writes output/01b_movement_trials.pqt and a summary json.
"""
import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from one.api import ONE

from common import ONE_CACHE, OUT, TRIALS_PQT, insertions

one = ONE(base_url="https://alyx.internationalbrainlab.org", cache_dir=ONE_CACHE, silent=True)
trials_all = pd.read_parquet(TRIALS_PQT)
eids = sorted(set(insertions().eid))


def _load(eid, name, collection="alf"):
    try:
        return one.load_dataset(eid, name, collection=collection)
    except Exception:
        return None


def overlap_any(iv, a, b):
    """For windows [a_i, b_i], True if any interval in iv overlaps it."""
    if iv is None or len(iv) == 0:
        return np.zeros(len(a), bool)
    on, off = iv[:, 0], iv[:, 1]
    order = np.argsort(on)
    on, off = on[order], off[order]
    # intervals that start before b and end after a
    j = np.searchsorted(on, b)  # intervals with on < b are [0, j)
    cummax = np.maximum.accumulate(off)
    res = np.zeros(len(a), bool)
    ok = j > 0
    res[ok] = cummax[j[ok] - 1] > a[ok]
    return res


def count_in(ts, a, b):
    if ts is None or len(ts) == 0:
        return np.full(len(a), np.nan)
    ts = np.sort(ts)
    return (np.searchsorted(ts, b) - np.searchsorted(ts, a)).astype(float)


def session(eid):
    wm = _load(eid, "_ibl_wheelMoves.intervals.npy")
    lk = _load(eid, "licks.times.npy")
    g = trials_all[trials_all.eid == eid].sort_values("intervals_0").reset_index(drop=True)
    so = g.stimOn_times.values
    fb_prev = np.r_[np.nan, g.feedback_times.values[:-1]]
    end_prev = np.r_[-np.inf, g.intervals_1.values[:-1]]
    wins = {"rest": (so - 1.15, so - 1.0), "quies": (so - 0.4, so - 0.1),
            "iti_long": (fb_prev + 1.5, so - 0.5)}
    out = pd.DataFrame({"eid": eid, "trial": np.arange(len(g)), "include": g.bwm_include.values,
                        "iti_mask": (so - end_prev) > 1.15,
                        "has_wheel": wm is not None, "has_licks": lk is not None})
    for k, (a, b) in wins.items():
        valid = np.isfinite(a) & np.isfinite(b) & (b > a)
        a2, b2 = np.where(valid, a, 0), np.where(valid, b, 0)
        out[f"{k}_len"] = np.where(valid, b - a, np.nan)
        out[f"{k}_wheel"] = np.where(valid & (wm is not None), overlap_any(wm, a2, b2), np.nan)
        out[f"{k}_licks"] = np.where(valid, count_in(lk, a2, b2), np.nan)
    return out


with ThreadPoolExecutor(8) as ex:
    df = pd.concat(list(ex.map(session, eids)), ignore_index=True)
df.to_parquet(OUT / "01b_movement_trials.pqt")

sel = df.include.astype(bool) & df.iti_mask
summ = {"n_sessions": int(df.eid.nunique()),
        "sessions_with_wheelMoves": int(df.groupby("eid").has_wheel.first().sum()),
        "sessions_with_licks": int(df.groupby("eid").has_licks.first().sum())}
for k in ("rest", "quies", "iti_long"):
    s = sel & df[f"{k}_len"].notna()
    summ[f"{k}_frac_trials_wheel_moving"] = float(np.nanmean(df.loc[s, f"{k}_wheel"]))
    summ[f"{k}_frac_trials_any_lick"] = float(np.nanmean(df.loc[s, f"{k}_licks"] > 0))
    summ[f"{k}_lick_rate_hz"] = float(np.nansum(df.loc[s, f"{k}_licks"]) / np.nansum(df.loc[s & df[f'{k}_licks'].notna(), f"{k}_len"]))
print(json.dumps(summ, indent=1))
json.dump(summ, open(OUT / "01b_movement_summary.json", "w"), indent=1)
