"""Mesoscope grey-screen (passive spontaneous) coupling, for comparison with BWM.

Sessions whose spontaneous period is >= MIN_DUR s (most canonical sessions only have
60 s). 1 s bins of the period (first 2 s skipped); still = wheel speed <= 0.05 rad/s
(when recorded), no lick, motion energy <= session 60th percentile. Nuisance regression
(wheel, licks, motion energy), region signals (Beryl, >= MIN_NEU ROIs, split-half
reliability >= 0.5), pair r -> Fisher z, residual from 0.5 mm distance bins.
Region sets: our DMN core / core+lateral and the Lu et al. 2012 rat-DMN mapping.

Usage: python 33_meso_spont.py [min_dur] [require_me]
With 'require_me', sessions without motion energy for the period (no way to check stillness) are dropped and
outputs get the suffix _me.
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bwm"))
from common import DMN_ALL, DMN_CORE, LU_DMN, LU_PARIETAL, LU_PFC_TEMPORAL, WHITESELL_DMN  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"
MIN_DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 200
REQ_ME = len(sys.argv) > 2 and sys.argv[2] == "require_me"
SUF = "_me" if REQ_ME else ""
MIN_NEU, MIN_BINS = 10, 100
rng = np.random.default_rng(0)


def session(f):
    from iblatlas.regions import BrainRegions
    s = dict(np.load(f, allow_pickle=True))
    sp = s["spont"]
    if not np.all(np.isfinite(sp)) or sp[1] - sp[0] < MIN_DUR:
        return None
    BIN, nb = float(s["BIN"]), s["Xbin"].shape[1]
    ep = np.zeros(nb, bool)
    ep[int(np.ceil(sp[0] / BIN)) + 2: min(nb, int(sp[1] / BIN))] = True
    w, lk, me = (s[k].astype(float) for k in ("wheel_speed", "licks", "motion_energy"))
    info = {"eid": str(s["eid"]), "dur": float(sp[1] - sp[0]), "n_bins": int(ep.sum()),
            "wheel_cov": float(np.isfinite(w[ep]).mean()), "me_cov": float(np.isfinite(me[ep]).mean()),
            "frac_wheel": float(np.nanmean(w[ep] > 0.05)) if np.isfinite(w[ep]).any() else np.nan}
    if REQ_ME and info["me_cov"] < .5:
        return info, None
    mov = (np.nan_to_num(w) > 0.05) | (np.nan_to_num(lk) > 0)
    if np.isfinite(me).mean() > .5:
        hi = np.nan_to_num(me) > np.nanpercentile(me, 60)
        info["frac_me_high"] = float(hi[ep].mean())
        mov |= hi
    ep &= ~mov
    info["n_still"] = int(ep.sum())
    X = s["Xbin"][:, ep].astype(np.float32).T
    good = np.all(np.isfinite(X), 0) & (X.std(0) > 0)
    if ep.sum() < MIN_BINS or good.sum() < 50:
        return info, None
    reg = np.array(BrainRegions().id2acronym(s["atlas_id"], mapping="Beryl"))[good]
    xyz = s["xyz"][good]
    Z = (X[:, good] - X[:, good].mean(0)) / X[:, good].std(0)
    D = [np.ones(len(Z))]
    for v in (w[ep], lk[ep], me[ep]):
        if np.isfinite(v).mean() > .5 and np.nanstd(v) > 0:
            v = np.where(np.isfinite(v), v, np.nanmean(v))
            D.append((v - v.mean()) / v.std())
    D = np.array(D).T
    Z = Z - D @ np.linalg.lstsq(D, Z, rcond=None)[0]
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-9)
    sig, meta = [], []
    for r in np.unique(reg):
        sel = np.flatnonzero(reg == r)
        if len(sel) < MIN_NEU or r in ("root", "void"):
            continue
        rh = np.nanmean([np.corrcoef(Z[:, pm[: len(pm) // 2]].mean(1), Z[:, pm[len(pm) // 2:]].mean(1))[0, 1]
                         for pm in (rng.permutation(sel) for _ in range(10))])
        sig.append(Z[:, sel].mean(1))
        meta.append((r, len(sel), *np.abs(xyz[sel, :2]).mean(0), 2 * rh / (1 + rh)))
    if len(sig) < 2:
        return info, None
    C = np.corrcoef(np.array(sig))
    pairs = [(info["eid"], meta[i][0], meta[j][0], float(np.hypot(meta[i][2] - meta[j][2], meta[i][3] - meta[j][3])),
              float(C[i, j]), meta[i][4], meta[j][4]) for i in range(len(meta)) for j in range(i + 1, len(meta))]
    return info, pairs


if __name__ == "__main__":
    with ProcessPoolExecutor(4) as ex:
        res = [r for r in ex.map(session, sorted((OUT / "sessions").glob("*.npz"))) if r is not None]
    subj = pd.read_csv(OUT / "session_subjects.csv").drop_duplicates("session").set_index("session").subject
    S = pd.DataFrame([r[0] for r in res])
    S["subject"] = S.eid.map(subj)
    S.to_csv(OUT / f"33_spont_sessions{SUF}.csv", index=False)
    P = pd.DataFrame([p for r in res if r[1] for p in r[1]], columns=["eid", "r1", "r2", "dist_mm", "r", "rel1", "rel2"])
    sw = P.r1 > P.r2
    P.loc[sw, ["r1", "r2"]] = P.loc[sw, ["r2", "r1"]].values
    P.loc[sw, ["rel1", "rel2"]] = P.loc[sw, ["rel2", "rel1"]].values
    P["subject"] = P.eid.map(subj)
    P = P[(P.rel1 >= .5) & (P.rel2 >= .5)].copy()
    P["z"] = np.arctanh(P.r.clip(-.999, .999))
    P["dbin"] = pd.cut(P.dist_mm, [0, .5, 1, 1.5, 2, 2.5, 3, 4, 10], labels=False)
    P["resid"] = P.z - P.groupby("dbin").z.transform("mean")
    P.to_parquet(OUT / f"33_spont_pairs{SUF}.pqt")
    out = {"min_dur": MIN_DUR, "require_me": REQ_ME, "n_sessions_long_spont": int(len(S)), "n_mice_long_spont": int(S.subject.nunique()),
           "n_sessions_with_pairs": int(P.eid.nunique()), "n_mice_with_pairs": int(P.subject.nunique()),
           "n_pairs": int(len(P)), "sessions": S.round(3).to_dict("records"),
           "regions_imaged": P[["r1", "r2"]].stack().value_counts().to_dict()}

    def mboot(Q, cl, a, b, n=2000):
        mice = Q.subject.unique()
        f = lambda w: (np.sum(w[cl == a] * Q.resid.values[cl == a]) / max(w[cl == a].sum(), 1e-9)
                       - np.sum(w[cl == b] * Q.resid.values[cl == b]) / max(w[cl == b].sum(), 1e-9))
        bs = []
        for _ in range(n):
            cnt = pd.Series(rng.choice(mice, len(mice))).value_counts()
            bs.append(f(Q.subject.map(cnt).fillna(0).values))
        return [round(float(f(np.ones(len(Q)))), 4)] + np.nanquantile(bs, [.025, .975]).round(4).tolist()

    out["classes"] = {}
    for sname, mem in (("ours_core", DMN_CORE), ("ours_core+lateral", DMN_ALL), ("lu_all", LU_DMN),
                       ("lu_pfc_temporal", LU_PFC_TEMPORAL), ("lu_parietal", LU_PARIETAL), ("whitesell", WHITESELL_DMN)):
        d1, d2 = P.r1.isin(mem), P.r2.isin(mem)
        cl = np.where(d1 & d2, "dmn-dmn", np.where(d1 | d2, "dmn-other", "other-other"))
        o = {"n": pd.Series(cl).value_counts().to_dict(),
             "means": {c: round(float(P.resid[cl == c].mean()), 4) for c in np.unique(cl)}}
        for a, b in (("dmn-dmn", "dmn-other"), ("dmn-dmn", "other-other"), ("dmn-other", "other-other")):
            if (cl == a).sum() >= 5 and (cl == b).sum() >= 5:
                o[f"{a} minus {b}"] = mboot(P, cl, a, b)
        out["classes"][sname] = o
    E = P.groupby(["r1", "r2"]).agg(n_sess=("eid", "nunique"), n_mice=("subject", "nunique"), resid=("resid", "mean"),
                                    z=("z", "mean"), dist=("dist_mm", "mean")).reset_index()
    E["lu_edge"] = E.r1.isin(LU_DMN) & E.r2.isin(LU_DMN)
    E["ours_edge"] = E.r1.isin(DMN_ALL) & E.r2.isin(DMN_ALL)
    E["whitesell_edge"] = E.r1.isin(WHITESELL_DMN) & E.r2.isin(WHITESELL_DMN)
    E.to_csv(OUT / f"33_spont_edges{SUF}.csv", index=False)
    print(json.dumps({k: v for k, v in out.items() if k not in ("sessions",)}, indent=1, default=str))
    json.dump(out, open(OUT / f"33_spont_meso{SUF}.json", "w"), indent=1, default=str)
