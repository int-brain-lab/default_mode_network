"""Approach A on the full BWM extraction: rest preference and task suppression.

Rest epochs on the 50 ms grid ('still' = no wheel movement, no lick, motion
energy below the session's ME_PCT percentile):
  iti    : feedback + 1.5 s .. next stimOn - 0.5 s, still bins only
  spont  : passive spontaneous period, still bins only
Task: exact per-trial window counts (motor_init, move, feedback; and stim).
Per neuron: rates, RI = (rest - task) / (rest + task), Welch t-test between
per-trial still-ITI rates and per-trial task rates -> suppressed/enhanced.
Also the movement modulation inside the ITI (moving vs still bins).

Usage: python 20_full_A.py [tag] [me_pct] [iti_start]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

from common import OUT, beryl, cosmos, done_eids
from dmnstats import dmn_tests, region_table, split_half

TAG = sys.argv[1] if len(sys.argv) > 1 else "A3"
ME_PCT = float(sys.argv[2]) if len(sys.argv) > 2 else 60
ITI_START = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5
FULL = OUT / "full"
SUBJ = pd.read_csv(OUT / "bwm_insertions.csv").groupby("eid").subject.first()
import os
TASK = tuple(os.environ.get("TASKSET", "motor_init,move,fback").split(","))


def still_mask(s):
    nb = len(s["licks"])
    w = s["wheel_speed"].astype(float)
    lk = s["licks"].astype(float)
    me = s["motion_energy"].astype(float)
    ok = np.ones(nb, bool)
    ok &= ~(np.isfinite(w) & (w > 0.05))
    ok &= ~((lk > 0) & (lk < 255))
    if np.isfinite(me).mean() > 0.5:
        thr = np.nanpercentile(me, ME_PCT)
        ok &= ~(np.isfinite(me) & (me > thr))
    # one-bin margin around any movement
    bad = ~ok
    bad = bad | np.r_[False, bad[:-1]] | np.r_[bad[1:], False]
    return ~bad, ok


def iti_bins(s, nb):
    so, fb, msk = s["stimOn_times"], s["feedback_times"], s["mask"]
    seg = np.full(nb, -1, np.int32)
    BIN = float(s["BIN"])
    for t in range(len(so) - 1):
        if not (msk[t] and np.isfinite(fb[t]) and np.isfinite(so[t + 1])):
            continue
        a, b = int(np.ceil((fb[t] + ITI_START) / BIN)), int(np.floor((so[t + 1] - 0.5) / BIN))
        if b > a:
            seg[a:b] = t
    return seg


def session_neurons(eid):
    s = dict(np.load(FULL / f"{eid}.npz", allow_pickle=True))
    BIN = float(s["BIN"])
    nb = len(s["licks"])
    still, still_nomargin = still_mask(s)
    seg = iti_bins(s, nb)
    sp = s["spont"]
    spont = np.zeros(nb, bool)
    if np.all(np.isfinite(sp)):
        spont[int(sp[0] / BIN) + 20: min(nb, int(sp[1] / BIN))] = True
    msk = s["mask"].astype(bool)
    rows = []
    for f in sorted(FULL.glob(f"{eid}_probe*.npz")):
        p = np.load(f, allow_pickle=True)
        X = p["counts"][:, :nb].astype(np.float32) / BIN
        it = (seg >= 0) & still
        itm = (seg >= 0) & ~still_nomargin
        sps = spont & still
        r_iti = X[:, it].mean(1) if it.sum() else np.full(len(X), np.nan)
        r_itm = X[:, itm].mean(1) if itm.sum() > 50 else np.full(len(X), np.nan)
        r_sp = X[:, sps].mean(1) if sps.sum() > 200 else np.full(len(X), np.nan)
        # per-trial ITI still rates (trials with >= 6 still bins = 0.3 s)
        trs = np.unique(seg[it])
        cnt = np.bincount(seg[it], minlength=len(msk))
        trs = trs[cnt[trs] >= 6]
        if len(trs) < 20:
            continue
        sums = np.zeros((len(msk), X.shape[0]), np.float32)
        np.add.at(sums, seg[it], X[:, it].T)
        iti_tr = sums[trs] / cnt[trs][:, None]
        W = {k: p[f"win_{k}"] / (dur) for k, dur in
             (("motor_init", .15), ("move", .15), ("fback", .3), ("stim", .15), ("quies", .3))}
        task = np.mean([W[k][msk] for k in TASK], axis=0)
        task = task[np.all(np.isfinite(task), 1)]
        r_task = task.mean(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            _, pv = ttest_ind(iti_tr, task, axis=0, equal_var=False)
        r_stim = np.nanmean(W["stim"][msk], 0)
        r_quies = np.nanmean(W["quies"][msk], 0)
        rows.append(pd.DataFrame({
            "eid": eid, "probe": str(p["probe"]), "uuid": p["uuids"], "atlas_id": p["atlas_id"],
            "x": p["xyz"][:, 0], "y": p["xyz"][:, 1], "z": p["xyz"][:, 2],
            "r_iti": r_iti, "r_iti_moving": r_itm, "r_spont": r_sp, "r_task": r_task, "r_stim": r_stim,
            "r_quies": r_quies, "p_iti_task": np.nan_to_num(pv, nan=1.), "n_iti_trials": len(trs),
            "iti_still_s": it.sum() * BIN, "spont_still_s": sps.sum() * BIN}))
    return rows


def ri(a, b):
    return (a - b) / (a + b + 1e-9)


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    eids = done_eids()
    with ProcessPoolExecutor(8) as ex:
        rows = [r for rr in ex.map(session_neurons, eids) for r in rr]
    N = pd.concat(rows, ignore_index=True)
    N["region"] = beryl(N.atlas_id)
    N["cosmos"] = cosmos(N.atlas_id)
    N["subject"] = N.eid.map(SUBJ)
    N = N[~N.region.isin(["root", "void"])]
    N = N[(N.r_iti + N.r_task) / 2 >= 0.5].reset_index(drop=True)
    N["RI_iti"] = ri(N.r_iti, N.r_task)
    N["RI_spont"] = ri(N.r_spont, N.r_task)
    N["RI_iti_quies"] = ri(N.r_iti, N.r_quies)
    N["MI_move"] = ri(N.r_iti_moving, N.r_iti)
    N["suppressed"] = ((N.p_iti_task < 0.01) & (N.RI_iti > 0)).astype(float)
    N["enhanced"] = ((N.p_iti_task < 0.01) & (N.RI_iti < 0)).astype(float)
    N.to_parquet(OUT / f"20_neurons_{TAG}.pqt")
    cols = ["RI_iti", "RI_spont", "RI_iti_quies", "MI_move", "suppressed", "enhanced"]
    R = region_table(N, cols)
    R.to_csv(OUT / f"20_regions_{TAG}.csv")
    res = {"tag": TAG, "me_pct": ME_PCT, "iti_start": ITI_START,
           "n_sessions": int(N.eid.nunique()), "n_subjects": int(N.subject.nunique()),
           "n_neurons": int(len(N)), "n_regions": int(len(R)),
           "median_iti_still_s_per_session": float(N.groupby("eid").iti_still_s.first().median()),
           "median_spont_still_s_per_session": float(N.groupby("eid").spont_still_s.first().median()),
           "sessions_with_spont": int(N.groupby("eid").r_spont.apply(lambda x: np.isfinite(x).any()).sum()),
           "brainwide": {c: round(float(N[c].mean()), 4) for c in cols},
           "reliability_subject_split": {c: split_half(N[np.isfinite(N[c])], R, c) for c in cols},
           "tests": dmn_tests(N, R, cols)}
    Cx = R[R.cosmos == "Isocortex"]
    for c in ("RI_iti", "RI_spont", "suppressed"):
        res[f"ctx_ranking_{c}"] = Cx[c].sort_values(ascending=False).round(3).to_dict()
        res[f"brain_top15_{c}"] = R[c].sort_values(ascending=False).head(15).round(3).to_dict()
    res["by_cosmos"] = R.groupby("cosmos")[cols].mean().round(3).to_dict()
    print(json.dumps(res, indent=1))
    json.dump(res, open(OUT / f"20_full_A_{TAG}.json", "w"), indent=1)
