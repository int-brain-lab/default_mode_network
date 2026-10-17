"""ITI rest activity, disengagement and time since feedback (full BWM data).

Per session and Beryl region (>= MIN_NEU neurons, probes pooled):
  y_t = region rate in the still part of ITI t (feedback_t + 1.5 s .. stimOn_t+1
        - 0.5 s; >= 0.3 s still), neurons z-scored over trials, then averaged.
1. Disengagement: partial correlation of y_t with the NEXT trial's log RT
   (firstMovement - stimOn) and with its error (1 - correct), controlling for
   previous feedback type, next-trial |contrast| (log), block, trial number
   (linear + quadratic drift) and the still duration of the ITI. Pupil (ITI
   mean) as an optional extra control. Per region: Fisher z of the partial
   r averaged over sessions; subject bootstrap; DMN vs other cortex.
2. Ramp: region rate in still 0.5 s bins as a function of time since the
   previous feedback (1.5-5 s), slope per region (z-scored rate per s).
Usage: python 23_iti_behaviour.py [tag] [pupil]
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from common import OUT, beryl, cosmos, done_eids
from dmnstats import SETS, PRIMARY, perm_diff

TAG = sys.argv[1] if len(sys.argv) > 1 else "D1"
USE_PUPIL = len(sys.argv) > 2 and sys.argv[2] == "pupil"
FULL = OUT / "full"
MIN_NEU = 5
ITI_START = 1.5
SUBJ = pd.read_csv(OUT / "bwm_insertions.csv").groupby("eid").subject.first()


def partial_r(y, x, C):
    D = np.c_[np.ones(len(y)), C]
    ry = y - D @ np.linalg.lstsq(D, y, rcond=None)[0]
    rx = x - D @ np.linalg.lstsq(D, x, rcond=None)[0]
    if ry.std() == 0 or rx.std() == 0:
        return np.nan
    return float(np.corrcoef(ry, rx)[0, 1])


def session(eid):
    s = dict(np.load(FULL / f"{eid}.npz", allow_pickle=True))
    BIN = float(s["BIN"])
    nb = len(s["licks"])
    w = np.nan_to_num(s["wheel_speed"].astype(float))
    lk = s["licks"].astype(float)
    me = s["motion_energy"].astype(float)
    mov = (w > 0.05) | ((lk > 0) & (lk < 255))
    if np.isfinite(me).mean() > .5:
        mov |= np.nan_to_num(me) > np.nanpercentile(me, 60)
    mov = mov | np.r_[False, mov[:-1]] | np.r_[mov[1:], False]
    still = ~mov
    so, fb, fm = s["stimOn_times"], s["feedback_times"], s["firstMovement_times"]
    msk = s["mask"].astype(bool)
    ntr = len(so)
    seg = np.full(nb, -1, np.int32)
    tsf = np.full(nb, np.nan, np.float32)  # time since previous feedback
    for t in range(ntr - 1):
        if msk[t] and msk[t + 1] and np.isfinite(fb[t]) and np.isfinite(so[t + 1]):
            a, b = int(np.ceil((fb[t] + ITI_START) / BIN)), int(np.floor((so[t + 1] - 0.5) / BIN))
            if b > a:
                seg[a:b] = t
                tsf[a:b] = (np.arange(a, b) + 0.5) * BIN - fb[t]
    it = (seg >= 0) & still
    cnt = np.bincount(seg[it], minlength=ntr)
    trs = np.flatnonzero(cnt >= 6)
    if len(trs) < 40:
        return None
    # per-trial covariates (trial t's ITI precedes trial t+1)
    nxt = trs + 1
    rt = np.log(np.clip(fm[nxt] - so[nxt], 0.05, 10))
    err = (s["feedbackType"][nxt] != 1).astype(float)
    con = np.nan_to_num(s["contrastLeft"][nxt]) + np.nan_to_num(s["contrastRight"][nxt])
    lcon = np.log(con + 0.03)
    prevfb = (s["feedbackType"][trs] == 1).astype(float)
    blk = s["probabilityLeft"][nxt]
    tn = trs / ntr
    dur = cnt[trs] * BIN
    pu = s["pupil"].astype(float)
    sums_p = np.bincount(seg[it], weights=np.nan_to_num(pu[it]), minlength=ntr)
    pup = sums_p[trs] / cnt[trs]
    covs = np.c_[prevfb, lcon, blk, tn, tn ** 2, dur]
    if USE_PUPIL and np.isfinite(pu).mean() > .5:
        covs = np.c_[covs, pup]
    okr = np.isfinite(rt)
    rows, ramps = [], []
    regs_all, X_all = [], []
    for f in sorted(FULL.glob(f"{eid}_probe*.npz")):
        p = np.load(f, allow_pickle=True)
        X = p["counts"][:, :nb].astype(np.float32)
        keep = X.mean(1) / BIN >= 0.5
        regs_all.append(beryl(p["atlas_id"])[keep])
        X_all.append(X[keep])
    if not X_all:
        return None
    reg = np.concatenate(regs_all)
    X = np.concatenate(X_all)
    sums = np.zeros((ntr, X.shape[0]), np.float32)
    np.add.at(sums, seg[it], X[:, it].T)
    Y = sums[trs] / cnt[trs][:, None] / BIN
    sd = Y.std(0)
    good = sd > 0
    Yz = (Y[:, good] - Y[:, good].mean(0)) / sd[good]
    regg = reg[good]
    # ramp: still bins grouped in 0.5 s of time since feedback
    tb = np.floor((tsf - ITI_START) / 0.5)
    rb = it & np.isfinite(tsf) & (tsf < 5.0)
    Xr = X[good][:, rb] / BIN
    Xr = (Xr - Xr.mean(1, keepdims=True)) / (Xr.std(1, keepdims=True) + 1e-9)
    tbr = tb[rb]
    for r in np.unique(regg):
        sel = regg == r
        if sel.sum() < MIN_NEU or r in ("root", "void"):
            continue
        y = Yz[:, sel].mean(1)
        rows.append(dict(eid=eid, region=r, n=int(sel.sum()), n_trials=len(trs),
                         r_rt=partial_r(y[okr], rt[okr], covs[okr]),
                         r_err=partial_r(y, err, covs),
                         r_prevfb=partial_r(y, prevfb, covs[:, 1:]),
                         r_pupil=float(np.corrcoef(y[np.isfinite(pup)], pup[np.isfinite(pup)])[0, 1]) if np.isfinite(pup).mean() > .5 else np.nan))
        yr = Xr[sel].mean(0)
        if len(np.unique(tbr)) >= 3:
            means = pd.Series(yr).groupby(tbr).mean()
            slope = np.polyfit(means.index.values * 0.5 + ITI_START + 0.25, means.values, 1)[0]
            ramps.append(dict(eid=eid, region=r, slope=float(slope)))
    return rows, ramps


if __name__ == "__main__":
    with ProcessPoolExecutor(8) as ex:
        res = [r for r in ex.map(session, done_eids()) if r is not None]
    D = pd.DataFrame([x for r in res for x in r[0]])
    Rm = pd.DataFrame([x for r in res for x in r[1]])
    D = D.merge(Rm, on=["eid", "region"], how="left")
    D["subject"] = D.eid.map(SUBJ)
    D["cosmos"] = cosmos(np.array([__import__("iblatlas.regions", fromlist=["x"]).BrainRegions().acronym2id(a)[0] for a in D.region]))
    D.to_parquet(OUT / f"23_sessions_{TAG}.pqt")
    cols = ["r_rt", "r_err", "r_prevfb", "r_pupil", "slope"]
    for c in cols[:4]:
        D[c + "_z"] = np.arctanh(D[c].clip(-.99, .99))
    zc = [c + "_z" for c in cols[:4]] + ["slope"]
    g = D.groupby("region")
    R = g[zc].mean()
    R["n_sess"] = g.eid.nunique()
    R["n_subj"] = g.subject.nunique()
    R["cosmos"] = g.cosmos.first()
    R = R[R.n_sess >= 3]
    R.to_csv(OUT / f"23_regions_{TAG}.csv")
    rng = np.random.default_rng(0)
    res = {"tag": TAG, "pupil_control": USE_PUPIL, "n_sessions": int(D.eid.nunique()),
           "n_region_sessions": int(len(D)), "n_regions": int(len(R)),
           "grand_mean": {c: round(float(D[c].mean()), 4) for c in zc}}
    # grand-mean CI via subject bootstrap
    subj = D.subject.unique()
    for c in zc:
        bs = []
        for _ in range(1000):
            cnt = pd.Series(rng.choice(subj, len(subj))).value_counts()
            ww = D.subject.map(cnt).fillna(0).values
            ok = np.isfinite(D[c].values)
            bs.append(np.sum(ww[ok] * D[c].values[ok]) / ww[ok].sum())
        res[f"grand_ci|{c}"] = np.quantile(bs, [.025, .975]).round(4).tolist()
    Cx = R[R.cosmos == "Isocortex"]
    for setname, mem in SETS.items():
        isd = Cx.index.isin(mem)
        for oname, oth in (("other_ctx", ~isd), ("higher_order", ~isd & ~Cx.index.isin(PRIMARY))):
            for c in zc:
                v = Cx[c].values
                ok = np.isfinite(v)
                obs, p = perm_diff(v[isd & ok], v[oth & ok])
                res[f"{c}|{setname}|vs_{oname}"] = {"dmn": round(float(v[isd & ok].mean()), 4),
                                                     "other": round(float(v[oth & ok].mean()), 4), "p_perm": round(p, 5)}
    for c in zc:
        res[f"ctx_ranking|{c}"] = Cx[c].sort_values(ascending=False).round(3).to_dict()
        res[f"by_cosmos|{c}"] = R.groupby("cosmos")[c].mean().round(4).to_dict()
    print(json.dumps(res, indent=1))
    json.dump(res, open(OUT / f"23_iti_behaviour_{TAG}.json", "w"), indent=1)
