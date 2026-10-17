"""Full data: DMN tests on B pairs, A-B alignment, arousal-coupling maps.

Usage: python 22_compare_full.py <B tag> <A tag> [RI column] [z column]
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import DMN_ALL, DMN_CORE, OUT
from dmnstats import PRIMARY, perm_diff

BTAG = sys.argv[1]
ATAG = sys.argv[2] if len(sys.argv) > 2 else "A3"
RI = sys.argv[3] if len(sys.argv) > 3 else "RI_iti"
ZCOL = sys.argv[4] if len(sys.argv) > 4 else "z"
rng = np.random.default_rng(0)
SUBJ = pd.read_csv(OUT / "bwm_insertions.csv").groupby("eid").subject.first()

P = pd.read_parquet(OUT / f"21_pairs_{BTAG}.pqt")
M = pd.read_parquet(OUT / f"21_regions_{BTAG}.pqt")
R = pd.read_csv(OUT / f"20_regions_{ATAG}.csv", index_col=0)
P["subject"] = P.eid.map(SUBJ)
P["zz"] = P[ZCOL]
P = P[np.isfinite(P.zz)].copy()
P["dbin"] = pd.cut(P.dist_mm, [0, .5, 1, 1.5, 2, 3, 4, 6, 20], labels=False)
P["resid"] = P.zz - P.groupby(["dbin", "same_probe"]).zz.transform("mean")


def wmean(Q, w, sel):
    ww = w[sel]
    return np.sum(ww * Q.resid.values[sel]) / ww.sum() if ww.sum() > 0 else np.nan


def boot(Q, fn, n=1000):
    subj = Q.subject.unique()
    out = []
    for _ in range(n):
        cnt = pd.Series(rng.choice(subj, len(subj))).value_counts()
        out.append(fn(Q, Q.subject.map(cnt).fillna(0).values))
    return np.nanquantile(out, [.025, .975]).round(4).tolist()


res = {"btag": BTAG, "atag": ATAG, "ri": RI, "zcol": ZCOL, "n_pairs": int(len(P)),
       "n_sessions": int(P.eid.nunique()), "n_subjects": int(P.subject.nunique())}

# 1. literature DMN classes among cortical pairs
Cp = P[(P.c1 == "Isocortex") & (P.c2 == "Isocortex")].copy()
for setname, mem in (("core", DMN_CORE), ("core+lateral", DMN_ALL)):
    d1, d2 = Cp.r1.isin(mem), Cp.r2.isin(mem)
    Cp["cls"] = np.where(d1 & d2, "dmn-dmn", np.where(d1 | d2, "dmn-other", "other-other"))
    cl = Cp.cls.values

    def f1(Q, w):
        return wmean(Q, w, cl == "dmn-dmn") - wmean(Q, w, cl == "dmn-other")

    def f2(Q, w):
        return wmean(Q, w, cl == "dmn-dmn") - wmean(Q, w, cl == "other-other")

    res[f"dmn_pairs|{setname}"] = {
        "n": Cp.cls.value_counts().to_dict(), "n_cross_probe": Cp[~Cp.same_probe].cls.value_counts().to_dict(),
        "resid": Cp.groupby("cls").resid.mean().round(4).to_dict(),
        "raw_cross_probe": Cp[~Cp.same_probe].groupby("cls").zz.mean().round(4).to_dict(),
        "ci_dmn-dmn_minus_dmn-other": boot(Cp, f1), "ci_dmn-dmn_minus_other-other": boot(Cp, f2)}
dm = Cp[Cp.r1.isin(DMN_CORE) & Cp.r2.isin(DMN_CORE)]
res["dmn_core_pairs_covered"] = int(dm[["r1", "r2"]].drop_duplicates().shape[0])
res["dmn_core_pairs_cross_probe"] = int(dm[~dm.same_probe][["r1", "r2"]].drop_duplicates().shape[0])
ant = ["ACAd", "ACAv", "PL", "ILA", "ORBm", "ORBl", "ORBvl"]
post = ["RSPagl", "RSPd", "RSPv", "VISa", "VISam"]
ap = dm[(dm.r1.isin(ant) & dm.r2.isin(post)) | (dm.r1.isin(post) & dm.r2.isin(ant))]
res["dmn_anterior_posterior_pairs"] = {"n_pairs": int(len(ap)), "n_sessions": int(ap.eid.nunique()),
                                       "mean_resid": round(float(ap.resid.mean()), 4) if len(ap) else None}

# 2. alignment with A
Q = P[P.r1.isin(R.index) & P.r2.isin(R.index)].copy()
Q["ri_sum"] = Q.r1.map(R[RI]) + Q.r2.map(R[RI])


def slope(Q, w):
    x, y = Q.ri_sum.values, Q.resid.values
    xm, ym = np.average(x, weights=w), np.average(y, weights=w)
    return np.sum(w * (x - xm) * (y - ym)) / np.sum(w * (x - xm) ** 2)


for name, S in (("brainwide", Q), ("cross_probe", Q[~Q.same_probe]),
                ("isocortex", Q[(Q.c1 == "Isocortex") & (Q.c2 == "Isocortex")])):
    res[f"align_slope|{name}"] = {"n": int(len(S)), "spearman": round(float(spearmanr(S.ri_sum, S.resid).statistic), 4),
                                  "slope": round(float(slope(S, np.ones(len(S)))), 4), "ci": boot(S, slope)}
q = pd.qcut(R[RI], 3, labels=["taskpos", "mid", "tneg"])
Q["cls"] = ["-".join(sorted([str(a), str(b)])) for a, b in zip(Q.r1.map(q), Q.r2.map(q))]
X = Q[~Q.same_probe]
res["terciles_cross_probe"] = X.groupby("cls").agg(n=("r", "size"), r=("r", "mean"), zz=("zz", "mean"),
                                                   frac_neg=("r", lambda x: float((x < 0).mean()))).round(4).to_dict("index")

# 3. arousal coupling per region (pooled across animals)
M["subject"] = M.eid.map(SUBJ)
A = M.groupby("region").agg(r_pupil=("r_pupil", "mean"), r_me=("r_me", "mean"), n_sess=("eid", "nunique"),
                            n_subj=("subject", "nunique"), cosmos=("cosmos", "first"))
A = A[A.n_sess >= 3]
A.to_csv(OUT / f"22_arousal_{BTAG}.csv")
common = A.index.intersection(R.index)
for c in ("r_pupil", "r_me"):
    ok = np.isfinite(A.loc[common, c])
    res[f"arousal|{c}_vs_{RI}"] = {"n": int(ok.sum()), "spearman": round(float(spearmanr(A.loc[common, c][ok], R.loc[common, RI][ok]).statistic), 4)}
    Cx = A[A.cosmos == "Isocortex"]
    for setname, mem in (("core", DMN_CORE), ("core+lateral", DMN_ALL)):
        isd = Cx.index.isin(mem)
        for oname, oth in (("other_ctx", ~isd), ("higher_order", ~isd & ~Cx.index.isin(PRIMARY))):
            v = Cx[c].values
            ok = np.isfinite(v)
            if (isd & ok).sum() > 1 and (oth & ok).sum() > 1:
                obs, p = perm_diff(v[isd & ok], v[oth & ok])
                res[f"arousal|{c}|{setname}|vs_{oname}"] = {"dmn": round(float(v[isd & ok].mean()), 4),
                                                            "other": round(float(v[oth & ok].mean()), 4),
                                                            "p_perm": round(p, 5)}
    res[f"arousal|{c}|ctx_ranking"] = A[A.cosmos == "Isocortex"][c].sort_values().round(3).to_dict()
print(json.dumps(res, indent=1, default=str))
json.dump(res, open(OUT / f"22_compare_{BTAG}_{ATAG}_{RI}_{ZCOL}.json", "w"), indent=1, default=str)
