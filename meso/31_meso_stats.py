"""Statistics for approach C (mesoscope) and cross-modal comparison with BWM A.

Usage: python 31_meso_stats.py <C tag> [BWM A tag]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "bwm"))
from common import DMN_ALL, DMN_CORE  # noqa: E402
from dmnstats import PRIMARY, perm_diff  # noqa: E402

OUT = HERE / "output"
TAG = sys.argv[1] if len(sys.argv) > 1 else "motion_iti2"
ATAG = sys.argv[2] if len(sys.argv) > 2 else "A3"
rng = np.random.default_rng(0)
subj = pd.read_csv(OUT / "session_subjects.csv", index_col=0).subject if (OUT / "session_subjects.csv").exists() else None
res = {"tag": TAG}

# ---- C-A
A = pd.read_parquet(OUT / f"30_neurons_{TAG}.pqt")
A["suppressed"] = ((A.p < 0.01) & (A.RI > 0)).astype(float)
A["enhanced"] = ((A.p < 0.01) & (A.RI < 0)).astype(float)
g = A.groupby("region")
RA = pd.DataFrame({"n": g.size(), "n_sess": g.eid.nunique(), "RI": g.RI.mean(),
                   "suppressed": g.suppressed.mean(), "enhanced": g.enhanced.mean()})
RA = RA[(RA.n >= 100) & (RA.n_sess >= 3) & ~RA.index.isin(["root", "void"])]
RA.to_csv(OUT / f"31_regions_A_{TAG}.csv")
res["A_regions"] = RA.round(3).to_dict("index")
for setname, mem in (("core", DMN_CORE), ("core+lateral", DMN_ALL)):
    isd = RA.index.isin(mem)
    for oname, oth in (("other", ~isd), ("higher_order", ~isd & ~RA.index.isin(PRIMARY))):
        for c in ("RI", "suppressed"):
            if isd.sum() > 1 and oth.sum() > 1:
                obs, p = perm_diff(RA[c].values[isd], RA[c].values[oth])
                res[f"A|{c}|{setname}|vs_{oname}"] = {"dmn": round(float(RA[c].values[isd].mean()), 4),
                                                       "other": round(float(RA[c].values[oth].mean()), 4),
                                                       "p_perm": round(p, 5), "n": [int(isd.sum()), int(oth.sum())]}
# cross-modal: BWM A vs meso A over shared cortical regions
bwm_f = HERE.parent / "bwm" / "output" / f"20_regions_{ATAG}.csv"
if bwm_f.exists():
    RB = pd.read_csv(bwm_f, index_col=0)
    cm = RA.index.intersection(RB.index)
    for cb, cmz in (("RI_iti", "RI"), ("suppressed", "suppressed"), ("RI_spont", "RI")):
        if cb in RB:
            rho = spearmanr(RB.loc[cm, cb], RA.loc[cm, cmz])
            res[f"crossmodal|bwm_{cb}_vs_meso_{cmz}"] = {"n": int(len(cm)), "spearman": round(float(rho.statistic), 3),
                                                          "p": round(float(rho.pvalue), 4)}

# ---- C-B pairs
P = pd.read_parquet(OUT / f"30_pairs_{TAG}.pqt")
P = P[(P.rel1 >= .5) & (P.rel2 >= .5)].copy()
P["z"] = np.arctanh(P.r.clip(-.999, .999))
P["dbin"] = pd.cut(P.dist_mm, [0, .5, 1, 1.5, 2, 2.5, 3, 4, 10], labels=False)
P["resid"] = P.z - P.groupby("dbin").z.transform("mean")
eids = P.eid.unique()


def boot(Q, fn, n=1000):
    e = Q.eid.unique()
    out = []
    for _ in range(n):
        cnt = pd.Series(rng.choice(e, len(e))).value_counts()
        out.append(fn(Q, Q.eid.map(cnt).fillna(0).values))
    return np.nanquantile(out, [.025, .975]).round(4).tolist()


def wm(Q, w, sel):
    return np.sum(w[sel] * Q.resid.values[sel]) / max(w[sel].sum(), 1e-9)


for setname, mem in (("core", DMN_CORE), ("core+lateral", DMN_ALL)):
    d1, d2 = P.r1.isin(mem), P.r2.isin(mem)
    cl = np.where(d1 & d2, "dmn-dmn", np.where(d1 | d2, "dmn-other", "other-other"))
    res[f"B|{setname}"] = {
        "n": pd.Series(cl).value_counts().to_dict(), "n_sessions": int(P.eid.nunique()),
        "resid": P.groupby(cl).resid.mean().round(4).to_dict(),
        "raw_z": P.groupby(cl).z.mean().round(4).to_dict(),
        "ci_dmn-dmn_minus_dmn-other": boot(P, lambda Q, w: wm(Q, w, cl == "dmn-dmn") - wm(Q, w, cl == "dmn-other")),
        "ci_dmn-dmn_minus_other-other": boot(P, lambda Q, w: wm(Q, w, cl == "dmn-dmn") - wm(Q, w, cl == "other-other"))}
# pair-level region matrix (mean resid per region pair)
Mx = P.groupby(["r1", "r2"]).agg(resid=("resid", "mean"), z=("z", "mean"), n=("eid", "nunique")).reset_index()
Mx.to_csv(OUT / f"31_pairmatrix_{TAG}.csv", index=False)
# alignment with meso A
Q = P[P.r1.isin(RA.index) & P.r2.isin(RA.index)].copy()
Q["ri_sum"] = Q.r1.map(RA.RI) + Q.r2.map(RA.RI)
Q["sup_sum"] = Q.r1.map(RA.suppressed) + Q.r2.map(RA.suppressed)
for c in ("ri_sum", "sup_sum"):
    rho = spearmanr(Q[c], Q.resid)
    res[f"align|{c}"] = {"n": int(len(Q)), "spearman": round(float(rho.statistic), 4)}

# ---- seed maps
for sname in ("RSP", "SSp"):
    f = OUT / f"30_seed_{sname}_{TAG}.pqt"
    if not f.exists():
        continue
    S = pd.read_parquet(f)
    # distance-controlled: residual over 0.5 mm distance-to-seed bins
    S["dbin"] = pd.cut(S.dist_seed, np.arange(0, 8, .5), labels=False)
    S["resid"] = S.r - S.groupby("dbin").r.transform("mean")
    byreg = S.groupby("region").agg(r=("r", "mean"), resid=("resid", "mean"), n=("r", "size"), n_sess=("eid", "nunique"))
    byreg = byreg[(byreg.n >= 100) & (byreg.n_sess >= 3)]
    byreg.to_csv(OUT / f"31_seed_{sname}_regions_{TAG}.csv")
    grid = S.groupby(["gx", "gy"]).agg(r=("r", "mean"), n_sess=("eid", "nunique"), n=("r", "size")).reset_index()
    grid.to_csv(OUT / f"31_seed_{sname}_grid_{TAG}.csv", index=False)
    isd = byreg.index.isin(DMN_ALL)
    if isd.sum() > 1 and (~isd).sum() > 1:
        for c in ("r", "resid"):
            obs, p = perm_diff(byreg[c].values[isd], byreg[c].values[~isd])
            res[f"seed_{sname}|{c}|dmn_vs_other"] = {"dmn": round(float(byreg[c].values[isd].mean()), 4),
                                                    "other": round(float(byreg[c].values[~isd].mean()), 4),
                                                    "p_perm": round(p, 5), "n": [int(isd.sum()), int((~isd).sum())]}
    res[f"seed_{sname}|regions"] = byreg.resid.sort_values(ascending=False).round(3).to_dict()
print(json.dumps(res, indent=1, default=str))
json.dump(res, open(OUT / f"31_meso_stats_{TAG}.json", "w"), indent=1, default=str)
