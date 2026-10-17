"""Approach C: mesoscope (dorsal cortex, many areas imaged simultaneously).

Per session (output/sessions/<eid>.npz from 10_extract_meso.py):
 C-A  rest preference per neuron: pre-stimulus window (stimOn -0.8..-0.1 s,
      trials with a still wheel in the preceding 1 s bins) vs task
      (mean of stim, choice, feedback windows); Welch test -> suppressed.
 C-B  ITI co-fluctuation: 1 s bins from feedback + ITI_START to next
      stimOn - 0.2 s, still bins only, nuisance regression (wheel, licks,
      motion energy, + optional global signal); region signals (Beryl,
      >= MIN_NEU neurons) -> pair r, split-half reliability, distance.
 C-seed  correlation of every neuron with a retrosplenial seed (RSPagl+RSPd,
      or SSp as control seed) on 1 s ITI bins, binned on a 0.5 mm ML-AP grid
      (hemispheres folded) -> map pooled across sessions.
Region labels of most mesoscope sessions are pre-histology estimates.

Usage: python 30_meso_C.py [variant: motion|gsr] [iti_start]
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
SES = OUT / "sessions"
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "motion"
ITI_START = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
TAG = f"{VARIANT}_iti{ITI_START:g}"
MIN_NEU = 10
GRID = 0.5
SEEDS = {"RSP": ["RSPagl", "RSPd", "RSPv"], "SSp": ["SSp-bfd", "SSp-ll", "SSp-ul", "SSp-tr", "SSp-un", "SSp-n", "SSp-m"]}


def session(f):
    from iblatlas.regions import BrainRegions
    br = BrainRegions()
    s = dict(np.load(f, allow_pickle=True))
    eid = str(s["eid"])
    reg = np.array(br.id2acronym(s["atlas_id"], mapping="Beryl"))
    xyz = s["xyz"]
    msk = s["mask"].astype(bool)
    names = list(s["win_names"])
    Xw = s["Xwin"]  # neurons x trials x windows
    BIN = float(s["BIN"])
    w = s["wheel_speed"].astype(float)
    nb = s["Xbin"].shape[1]
    # --- C-A
    so = s["stimOn_times"]
    pre_bin = np.floor((so - 1.0) / BIN).astype(int)
    still_tr = np.array([np.isfinite(so[t]) and 0 <= pre_bin[t] < nb - 1 and
                         np.nan_to_num(w[pre_bin[t]:pre_bin[t] + 2]).max() < 0.05 for t in range(len(so))])
    pre = Xw[:, msk & still_tr, names.index("pre")]
    task = np.mean([Xw[:, msk, names.index(k)] for k in ("stim", "choice", "fback")], 0)
    rows_a = None
    if pre.shape[1] >= 20:
        with np.errstate(invalid="ignore", divide="ignore"):
            _, p = ttest_ind(pre, task, axis=1, equal_var=False)
        rp, rt = np.nanmean(pre, 1), np.nanmean(task, 1)
        rows_a = pd.DataFrame({"eid": eid, "region": reg, "ml": xyz[:, 0], "ap": xyz[:, 1],
                               "r_pre": rp, "r_task": rt,
                               "RI": (rp - rt) / (np.abs(rp) + np.abs(rt) + 1e-9), "p": np.nan_to_num(p, nan=1.)})
    # --- ITI bins
    fb = s["feedback_times"]
    ep = np.zeros(nb, bool)
    for t in range(len(so) - 1):
        if msk[t] and np.isfinite(fb[t]) and np.isfinite(so[t + 1]):
            a, b = int(np.ceil((fb[t] + ITI_START) / BIN)), int(np.floor((so[t + 1] - 0.2) / BIN))
            if b > a:
                ep[a:b] = True
    ep &= ~(np.nan_to_num(w) > 0.05)
    lk = s.get("licks", np.full(nb, np.nan)).astype(float)
    ep &= ~(np.nan_to_num(lk) > 0)
    X = s["Xbin"][:, ep].astype(np.float32).T  # bins x neurons
    good = np.all(np.isfinite(X), 0) & (X.std(0) > 0)
    if ep.sum() < 60 or good.sum() < 50:
        return eid, rows_a, None, None
    X, reg2, xyz2 = X[:, good], reg[good], xyz[good]
    Z = (X - X.mean(0)) / X.std(0)
    D = [np.ones(len(Z))]
    for v in (w[ep], lk[ep], s["motion_energy"][ep].astype(float)):
        if np.isfinite(v).mean() > .5 and np.nanstd(v) > 0:
            v = np.where(np.isfinite(v), v, np.nanmean(v))
            D.append((v - v.mean()) / v.std())
    if VARIANT == "gsr":
        D.append(Z.mean(1))
    D = np.array(D).T
    Z = Z - D @ np.linalg.lstsq(D, Z, rcond=None)[0]
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    rng = np.random.default_rng(0)
    sig, meta = [], []
    for r in np.unique(reg2):
        sel = np.flatnonzero(reg2 == r)
        if len(sel) < MIN_NEU or r in ("root", "void"):
            continue
        rh = []
        for _ in range(10):
            pm = rng.permutation(sel)
            rh.append(np.corrcoef(Z[:, pm[: len(pm) // 2]].mean(1), Z[:, pm[len(pm) // 2:]].mean(1))[0, 1])
        rh = np.nanmean(rh)
        sig.append(Z[:, sel].mean(1))
        meta.append((r, len(sel), *np.abs(xyz2[sel, :2]).mean(0), 2 * rh / (1 + rh)))
    pairs = None
    if len(sig) >= 2:
        C = np.corrcoef(np.array(sig))
        pairs = []
        for i in range(len(meta)):
            for j in range(i + 1, len(meta)):
                d = float(np.hypot(meta[i][2] - meta[j][2], meta[i][3] - meta[j][3]))
                pairs.append((eid, meta[i][0], meta[j][0], d, float(C[i, j]), meta[i][4], meta[j][4], len(Z)))
    # --- seed maps (folded ML)
    seeds = {}
    for sname, sregs in SEEDS.items():
        sm = np.isin(reg2, sregs)
        if sm.sum() < MIN_NEU:
            continue
        seed = Z[:, sm].mean(1)
        seed = (seed - seed.mean()) / seed.std()
        rr = (Z[:, ~sm] * seed[:, None]).mean(0)
        g = np.floor(np.c_[np.abs(xyz2[~sm, 0]), xyz2[~sm, 1]] / GRID).astype(int)
        seeds[sname] = pd.DataFrame({"eid": eid, "gx": g[:, 0], "gy": g[:, 1], "region": reg2[~sm], "r": rr,
                                     "dist_seed": np.hypot(np.abs(xyz2[~sm, 0]) - np.abs(xyz2[sm, 0]).mean(),
                                                           xyz2[~sm, 1] - xyz2[sm, 1].mean())})
    return eid, rows_a, pairs, seeds


if __name__ == "__main__":
    files = sorted(SES.glob("*.npz"))
    with ProcessPoolExecutor(6) as ex:
        res = list(ex.map(session, files))
    A = pd.concat([r[1] for r in res if r[1] is not None], ignore_index=True)
    A.to_parquet(OUT / f"30_neurons_{TAG}.pqt")
    P = pd.DataFrame([p for r in res if r[2] for p in r[2]],
                     columns=["eid", "r1", "r2", "dist_mm", "r", "rel1", "rel2", "n_bins"])
    sw = P.r1 > P.r2
    P.loc[sw, ["r1", "r2"]] = P.loc[sw, ["r2", "r1"]].values
    P.loc[sw, ["rel1", "rel2"]] = P.loc[sw, ["rel2", "rel1"]].values
    P.to_parquet(OUT / f"30_pairs_{TAG}.pqt")
    for sname in SEEDS:
        S = [r[3][sname] for r in res if r[3] and sname in r[3]]
        if S:
            pd.concat(S, ignore_index=True).to_parquet(OUT / f"30_seed_{sname}_{TAG}.pqt")
    print(json.dumps({"tag": TAG, "sessions": len(files), "neurons_A": len(A), "pairs": len(P),
                      "sessions_with_pairs": int(P.eid.nunique())}, indent=1))
