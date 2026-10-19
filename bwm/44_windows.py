"""Firing rates in task, quiescence, inter-trial rest and grey-screen windows, per neuron and session.

Windows (session averages per neuron, from 50 ms bins unless stated):
  stim       stimOn 0..0.15 s (exact per-trial counts)
  movefb     mean of firstMovement -0.15..0, 0..0.15 and feedback 0..0.3 s (exact counts; the task windows of 42)
  quies      still bins from trial start (intervals_0) to stimOn - 0.1 s: engaged, waiting, not moving
  iti        still bins from trial end (intervals_1) to the next trial start: between trials, not moving
  iti_move   moving bins in the same inter-trial gaps
  grey       still bins of the passive grey screen (first 1 s skipped)
  grey_move  moving bins of the grey screen
Still = no wheel > 0.05 rad/s, no lick, motion energy <= session 60th percentile, one-bin margin (as 42);
moving = any of these, without margin. BWM trial mask applies to all trial-based windows.

Per neuron also: suppressed fraction per rest reference (Welch test of per-trial rest rate vs per-trial movefb
rate, p < 0.01, rest higher), for quies and iti (trials with >= 2 still bins in the window).

Per session, trial-by-trial co-fluctuation between regions in the quies and iti windows: region signal = mean
over its neurons (>= 5, same probe) of the z-scored per-trial rate, with trial number (2nd-order polynomial) and the
per-trial mean motion energy and pupil regressed out; pair correlation across trials.

Usage: python 44_windows.py
"""
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

from common import OUT, beryl, cosmos, done_eids

FULL = OUT / "full"
SUBJ = pd.read_csv(OUT / "bwm_insertions.csv").groupby("eid").subject.first()
MIN_BINS = 40  # 2 s per window per session
MIN_NEU = 5


def masks(s):
    w, lk, me = (s[k].astype(float) for k in ("wheel_speed", "licks", "motion_energy"))
    mov = (np.isfinite(w) & (w > 0.05)) | ((lk > 0) & (lk < 255))
    if np.isfinite(me).mean() > 0.5:
        mov |= np.isfinite(me) & (me > np.nanpercentile(me, 60))
    bad = mov | np.r_[False, mov[:-1]] | np.r_[mov[1:], False]
    return ~bad, mov


def trial_segments(s, nb, BIN):
    """Per-trial bin ranges for quiescence and inter-trial gap (gap after trial t)."""
    i0, i1, so, msk = s["intervals_0"], s["intervals_1"], s["stimOn_times"], s["mask"].astype(bool)
    n = len(so)
    q = np.full((n, 2), -1, int)
    g = np.full((n, 2), -1, int)
    for t in range(n):
        if not msk[t]:
            continue
        if np.isfinite(i0[t]) and np.isfinite(so[t]):
            a, b = int(np.ceil(i0[t] / BIN)), int(np.floor((so[t] - 0.1) / BIN))
            if b > a:
                q[t] = a, min(b, nb)
        if t + 1 < n and np.isfinite(i1[t]) and np.isfinite(i0[t + 1]):
            a, b = int(np.ceil(i1[t] / BIN)), int(np.floor(i0[t + 1] / BIN))
            if b > a:
                g[t] = a, min(b, nb)
    return q, g


def per_trial_rates(X, seg, ok, BIN):
    """X neurons x bins; seg trials x 2; ok bin mask -> trials x neurons rates (NaN if < 2 bins), bins per trial."""
    R = np.full((len(seg), X.shape[0]), np.nan, np.float32)
    nbins = np.zeros(len(seg), int)
    for t, (a, b) in enumerate(seg):
        if a < 0:
            continue
        sel = np.flatnonzero(ok[a:b]) + a
        nbins[t] = len(sel)
        if len(sel) >= 2:
            R[t] = X[:, sel].sum(1) / (len(sel) * BIN)
    return R, nbins


def session(eid):
    s = dict(np.load(FULL / f"{eid}.npz", allow_pickle=True))
    BIN, nb = float(s["BIN"]), len(s["licks"])
    still, mov = masks(s)
    qseg, gseg = trial_segments(s, nb, BIN)
    sp = s["spont"]
    grey = np.zeros(nb, bool)
    if np.all(np.isfinite(sp)):
        grey[int(sp[0] / BIN) + 20: min(nb, int(sp[1] / BIN))] = True
    msk = s["mask"].astype(bool)
    me, pu = s["motion_energy"].astype(float), s["pupil"].astype(float)
    # per-trial nuisance: mean motion energy and pupil in each window
    def tmean(v, seg):
        return np.array([np.nanmean(v[a:b]) if a >= 0 and np.isfinite(v[a:b]).any() else np.nan for a, b in seg])
    nuis = {"quies": (tmean(me, qseg), tmean(pu, qseg)), "iti": (tmean(me, gseg), tmean(pu, gseg))}
    rows, sigs = [], {"quies": [], "iti": []}
    for f in sorted(FULL.glob(f"{eid}_probe*.npz")):
        p = np.load(f, allow_pickle=True)
        X = p["counts"][:, :nb].astype(np.float32)
        W = {k: p[f"win_{k}"] / d for k, d in (("stim", .15), ("motor_init", .15), ("move", .15), ("fback", .3))}
        movefb_tr = np.mean([W[k] for k in ("motor_init", "move", "fback")], 0)  # trials x neurons
        ok_tr = msk & np.all(np.isfinite(movefb_tr), 1)
        out = {"stim": np.nanmean(W["stim"][msk], 0), "movefb": movefb_tr[ok_tr].mean(0)}
        qbins = np.zeros(nb, bool)
        gbins = np.zeros(nb, bool)
        for a, b in qseg[qseg[:, 0] >= 0]:
            qbins[a:b] = True
        for a, b in gseg[gseg[:, 0] >= 0]:
            gbins[a:b] = True
        for name, sel in (("quies", qbins & still), ("iti", gbins & still), ("iti_move", gbins & mov),
                          ("grey", grey & still), ("grey_move", grey & mov)):
            out[name] = X[:, sel].sum(1) / (sel.sum() * BIN) if sel.sum() >= MIN_BINS else np.full(len(X), np.nan)
            out[f"n_{name}"] = int(sel.sum())
        # suppression vs per-trial task rate for the trial-based rest references
        for name, seg in (("quies", qseg), ("iti", gseg)):
            Rt, _ = per_trial_rates(X, seg, still, BIN)
            okr = np.all(np.isfinite(Rt), 1)
            if okr.sum() >= 20 and ok_tr.sum() >= 20:
                with np.errstate(invalid="ignore", divide="ignore"):
                    _, pv = ttest_ind(Rt[okr], movefb_tr[ok_tr], axis=0, equal_var=False)
                out[f"p_{name}"] = np.nan_to_num(pv, nan=1.)
            else:
                out[f"p_{name}"] = np.full(len(X), np.nan)
            # region signals for trial-wise co-fluctuation
            regs = beryl(p["atlas_id"])
            rate_all = X.sum(1) / (nb * BIN)
            for r in np.unique(regs):
                ids = np.flatnonzero((regs == r) & (rate_all >= 0.5))
                if len(ids) < MIN_NEU or r in ("root", "void"):
                    continue
                Z = Rt[:, ids]
                good = np.all(np.isfinite(Z), 1)
                if good.sum() < 50:
                    continue
                Zg = (Z[good] - Z[good].mean(0)) / (Z[good].std(0) + 1e-9)
                sig = np.full(len(Z), np.nan)
                sig[good] = Zg.mean(1)
                sigs[name].append((r, str(p["probe"]), p["xyz"][ids].mean(0), sig))
        rows.append(pd.DataFrame({"eid": eid, "probe": str(p["probe"]), "atlas_id": p["atlas_id"],
                                  "x": p["xyz"][:, 0], "y": p["xyz"][:, 1], "z": p["xyz"][:, 2],
                                  "rate_all": X.sum(1) / (nb * BIN), **{k: v for k, v in out.items()}}))
    # pair correlations across trials, after regressing trial index and nuisance
    pairs = []
    ntr = len(s["stimOn_times"])
    tt = np.linspace(-1, 1, ntr)
    for name, lst in sigs.items():
        if len(lst) < 2:
            continue
        S = np.array([x[3] for x in lst])  # regions x trials
        D = [np.ones(ntr), tt, tt ** 2]
        for v in nuis[name]:
            if np.isfinite(v).mean() > .5:
                D.append(np.where(np.isfinite(v), (v - np.nanmean(v)) / (np.nanstd(v) + 1e-9), 0))
        D = np.array(D).T
        okt = np.all(np.isfinite(S), 0)
        if okt.sum() < 50:
            continue
        Sr = S[:, okt] - (D[okt] @ np.linalg.lstsq(D[okt], S[:, okt].T, rcond=None)[0]).T
        C = np.corrcoef(Sr)
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if lst[i][0] == lst[j][0]:
                    continue
                d = float(np.linalg.norm(lst[i][2] - lst[j][2]) * 1e3)
                pairs.append((name, eid, lst[i][0], lst[j][0], lst[i][1] == lst[j][1], d, float(C[i, j]), int(okt.sum())))
    return rows, pairs


if __name__ == "__main__":
    with ProcessPoolExecutor(6) as ex:
        res = list(ex.map(session, done_eids()))
    N = pd.concat([r for rr in res for r in rr[0]], ignore_index=True)
    N["region"], N["cosmos"] = beryl(N.atlas_id), cosmos(N.atlas_id)
    N["subject"] = N.eid.map(SUBJ)
    N = N[~N.region.isin(["root", "void"]) & (N.rate_all >= 0.5)].reset_index(drop=True)
    N.to_parquet(OUT / "44_neurons.pqt")
    P = pd.DataFrame([p for rr in res for p in rr[1]],
                     columns=["window", "eid", "r1", "r2", "same_probe", "dist_mm", "r", "n_trials"])
    sw = P.r1 > P.r2
    P.loc[sw, ["r1", "r2"]] = P.loc[sw, ["r2", "r1"]].values
    from iblatlas.regions import BrainRegions
    br = BrainRegions()
    acr = np.unique(np.r_[P.r1, P.r2])
    a2c = dict(zip(acr, cosmos(br.acronym2id(acr))))
    P["c1"], P["c2"] = P.r1.map(a2c), P.r2.map(a2c)
    P["subject"] = P.eid.map(SUBJ)
    P.to_parquet(OUT / "44_pairs.pqt")
    print(len(N), N.eid.nunique(), len(P), P.groupby("window").size().to_dict())
    print(N[["stim", "movefb", "quies", "iti", "iti_move", "grey", "grey_move"]].describe().round(2))
