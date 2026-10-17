"""Approach B on the full BWM extraction: spontaneous region-region coupling.

Per session (probes pooled), coarse bins of BIN_B s built from the 50 ms grid.
Epochs: 'spont' (passive spontaneous period) or 'iti' (feedback + 1.5 s ..
next stimOn - 0.5 s). A coarse bin is used only if all its fine bins are in
the epoch (and, with --still, if none of them has movement).
Nuisance regression (per neuron, within epoch) by variant:
  none | motion (wheel, licks, motion energy + 1-bin lags) |
  motion_pupil (+ pupil) | gsr (motion_pupil + global mean of all neurons)
Region signal = mean of z-scored neurons (>= MIN_NEU per region), split-half
reliability over neuron halves; pair r, Fisher z, disattenuated r.
Also per region: correlation of the raw region signal with pupil and with
motion energy ('arousal coupling', pooled across animals later).

Usage: python 21_full_B.py epoch variant bin_s [still]
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from common import DMN_ALL, DMN_CORE, OUT, beryl, cosmos, done_eids

EPOCH = sys.argv[1] if len(sys.argv) > 1 else "spont"
VARIANT = sys.argv[2] if len(sys.argv) > 2 else "motion"
BIN_B = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
STILL = len(sys.argv) > 4 and sys.argv[4] == "still"
TAG = f"{EPOCH}_{VARIANT}_{BIN_B:g}s" + ("_still" if STILL else "")
FULL = OUT / "full"
MIN_NEU = 5
MIN_BINS = 120
ITI_START = 1.5


def coarse(x, k, how="mean"):
    n = len(x) // k
    x = x[: n * k].reshape(n, k, *x.shape[1:])
    return np.nanmean(x, 1) if how == "mean" else x.sum(1)


def session(eid):
    rng = np.random.default_rng(abs(hash(eid)) % 2**32)
    s = dict(np.load(FULL / f"{eid}.npz", allow_pickle=True))
    BIN = float(s["BIN"])
    k = int(round(BIN_B / BIN))
    nb = len(s["licks"])
    ep = np.zeros(nb, bool)
    if EPOCH == "spont":
        sp = s["spont"]
        if not np.all(np.isfinite(sp)):
            return None
        ep[int(sp[0] / BIN) + 20: min(nb, int(sp[1] / BIN))] = True
    else:
        so, fb, msk = s["stimOn_times"], s["feedback_times"], s["mask"]
        for t in range(len(so) - 1):
            if msk[t] and np.isfinite(fb[t]) and np.isfinite(so[t + 1]):
                a, b = int(np.ceil((fb[t] + ITI_START) / BIN)), int(np.floor((so[t + 1] - 0.5) / BIN))
                if b > a:
                    ep[a:b] = True
    w = np.nan_to_num(s["wheel_speed"].astype(float))
    lk = s["licks"].astype(float)
    lk[lk == 255] = 0
    me = s["motion_energy"].astype(float)
    pu = s["pupil"].astype(float)
    if STILL:
        mov = (w > 0.05) | (lk > 0)
        if np.isfinite(me).mean() > .5:
            mov |= np.nan_to_num(me) > np.nanpercentile(me, 60)
        ep &= ~mov
    valid = coarse(ep.astype(float), k) == 1
    if valid.sum() < MIN_BINS:
        return None
    Xs, regs, xyz = [], [], []
    for f in sorted(FULL.glob(f"{eid}_probe*.npz")):
        p = np.load(f, allow_pickle=True)
        c = p["counts"][:, :nb].T.astype(np.float32)  # bins x neurons
        Xs.append(coarse(c, k, "sum")[valid])
        regs.append(np.char.add(beryl(p["atlas_id"]).astype(str), ""))
        xyz.append(p["xyz"])
        probe_names = [str(p["probe"])] * c.shape[1]
        regs[-1] = np.array([f"{r}|{str(p['probe'])}" for r in regs[-1]])
    X = np.concatenate(Xs, 1)
    reg = np.concatenate(regs)
    xyz = np.concatenate(xyz)
    keep = (X.mean(0) / BIN_B >= 0.5) & ~np.char.startswith(reg, "root|") & ~np.char.startswith(reg, "void|")
    X, reg, xyz = X[:, keep], reg[keep], xyz[keep]
    if X.shape[1] < 2 * MIN_NEU:
        return None
    nus = {"wheel": coarse(w, k)[valid], "licks": coarse(lk, k)[valid], "me": coarse(me, k)[valid],
           "pupil": coarse(pu, k)[valid]}
    cols = []
    if VARIANT in ("motion", "motion_pupil", "gsr"):
        cols += ["wheel", "licks", "me"]
    if VARIANT in ("motion_pupil", "gsr"):
        cols += ["pupil"]
    D = [np.ones(valid.sum())]
    for c in cols:
        v = nus[c]
        if np.isfinite(v).mean() < .5:
            continue
        v = np.where(np.isfinite(v), v, np.nanmean(v))
        v = (v - v.mean()) / (v.std() + 1e-9)
        D += [v, np.r_[v[0], v[:-1]]]
    Z = (X - X.mean(0)) / (X.std(0) + 1e-9)
    raw_Z = Z.copy()
    if VARIANT == "gsr":
        D.append(Z.mean(1))
    D = np.array(D).T
    if D.shape[1] > 1:
        beta = np.linalg.lstsq(D, Z, rcond=None)[0]
        Z = Z - D @ beta
        Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    sig, meta = [], []
    for r in np.unique(reg):
        sel = np.flatnonzero(reg == r)
        if len(sel) < MIN_NEU:
            continue
        rh = []
        for _ in range(10):
            pm = rng.permutation(sel)
            rh.append(np.corrcoef(Z[:, pm[: len(pm) // 2]].mean(1), Z[:, pm[len(pm) // 2:]].mean(1))[0, 1])
        rh = np.nanmean(rh)
        sig.append(Z[:, sel].mean(1))
        rs = raw_Z[:, sel].mean(1)
        ar = {}
        for c in ("pupil", "me"):
            v = nus[c]
            ok = np.isfinite(v)
            ar[c] = np.corrcoef(rs[ok], v[ok])[0, 1] if ok.mean() > .5 and np.std(v[ok]) > 0 else np.nan
        name, probe = r.split("|")
        meta.append(dict(region=name, probe=probe, n=len(sel), x=xyz[sel, 0].mean(), y=xyz[sel, 1].mean(),
                         z=xyz[sel, 2].mean(), rel=2 * rh / (1 + rh) if rh > -1 else np.nan,
                         r_pupil=ar["pupil"], r_me=ar["me"]))
    if len(sig) < 2:
        return None
    S = np.array(sig)
    C = np.corrcoef(S)
    M = pd.DataFrame(meta)
    M["eid"] = eid
    pairs = []
    for i in range(len(M)):
        for j in range(i + 1, len(M)):
            a, b = M.iloc[i], M.iloc[j]
            if a.region == b.region:
                continue
            d = float(np.linalg.norm(np.array([a.x - b.x, a.y - b.y, a.z - b.z])) * 1e3)
            pairs.append((eid, a.region, b.region, a.probe == b.probe, d, float(C[i, j]), a.rel, b.rel,
                          int(valid.sum())))
    return M, pairs


if __name__ == "__main__":
    eids = done_eids()
    with ProcessPoolExecutor(8) as ex:
        res = [r for r in ex.map(session, eids) if r is not None]
    M = pd.concat([r[0] for r in res], ignore_index=True)
    P = pd.DataFrame([p for r in res for p in r[1]],
                     columns=["eid", "r1", "r2", "same_probe", "dist_mm", "r", "rel1", "rel2", "n_bins"])
    sw = P.r1 > P.r2
    P.loc[sw, ["r1", "r2"]] = P.loc[sw, ["r2", "r1"]].values
    P.loc[sw, ["rel1", "rel2"]] = P.loc[sw, ["rel2", "rel1"]].values
    P["z"] = np.arctanh(P.r.clip(-.999, .999))
    ok = (P.rel1 >= .2) & (P.rel2 >= .2)
    P["r_dis"] = np.where(ok, P.r / np.sqrt(P.rel1.clip(lower=.2) * P.rel2.clip(lower=.2)), np.nan)
    P["z_dis"] = np.arctanh(P.r_dis.clip(-.99, .99))
    from iblatlas.regions import BrainRegions
    br = BrainRegions()
    acr = np.unique(np.r_[P.r1, P.r2, M.region])
    a2c = dict(zip(acr, cosmos(br.acronym2id(acr))))
    P["c1"], P["c2"] = P.r1.map(a2c), P.r2.map(a2c)
    M["cosmos"] = M.region.map(a2c)
    P.to_parquet(OUT / f"21_pairs_{TAG}.pqt")
    M.to_parquet(OUT / f"21_regions_{TAG}.pqt")
    summ = {"tag": TAG, "n_sessions": int(P.eid.nunique()), "n_pairs": int(len(P)),
            "median_bins": float(P.groupby("eid").n_bins.first().median()),
            "mean_r_same_probe": round(float(P[P.same_probe].r.mean()), 4),
            "mean_r_cross_probe": round(float(P[~P.same_probe].r.mean()), 4),
            "median_rel": round(float(M.rel.median()), 3)}
    print(json.dumps(summ, indent=1))
    json.dump(summ, open(OUT / f"21_full_B_{TAG}.json", "w"), indent=1)
