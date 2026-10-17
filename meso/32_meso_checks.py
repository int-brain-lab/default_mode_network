"""Checks on C results: which DMN pairs drive B, mouse-level bootstrap,
leave-one-region-out, fine distance control, and a session-relative rest index
(RI minus session median, removing the global task-evoked rise of calcium signals).

Usage: python 32_meso_checks.py [tag]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bwm"))
from common import DMN_CORE, DMN_ALL  # noqa: E402

OUT = Path(__file__).resolve().parent / "output"
TAG = sys.argv[1] if len(sys.argv) > 1 else "motion_iti1"
rng = np.random.default_rng(0)
subj = pd.read_csv(OUT / "session_subjects.csv").drop_duplicates("session").set_index("session").subject
res = {"tag": TAG}

P = pd.read_parquet(OUT / f"30_pairs_{TAG}.pqt")
P = P[(P.rel1 >= .5) & (P.rel2 >= .5)].copy()
P["z"] = np.arctanh(P.r.clip(-.999, .999))
P["mouse"] = P.eid.map(subj)
res["n_sessions"], res["n_mice"] = int(P.eid.nunique()), int(P.mouse.nunique())


def classify(Q, mem):
    d1, d2 = Q.r1.isin(mem), Q.r2.isin(mem)
    return np.where(d1 & d2, "dmn-dmn", np.where(d1 | d2, "dmn-other", "other-other"))


def resid(Q, edges):
    db = pd.cut(Q.dist_mm, edges, labels=False)
    return Q.z - Q.groupby(db).z.transform("mean")


def stat(Q, w, cl, a, b):
    m = lambda s: np.sum(w[cl == s] * Q.resid.values[cl == s]) / max(w[cl == s].sum(), 1e-9)
    return m(a) - m(b)


def mouse_boot(Q, cl, a, b, n=2000):
    mice = Q.mouse.unique()
    out = []
    for _ in range(n):
        cnt = pd.Series(rng.choice(mice, len(mice))).value_counts()
        out.append(stat(Q, Q.mouse.map(cnt).fillna(0).values, cl, a, b))
    return [round(float(stat(Q, np.ones(len(Q)), cl, a, b)), 4)] + np.nanquantile(out, [.025, .975]).round(4).tolist()


for ename, edges in (("coarse_0.5mm", [0, .5, 1, 1.5, 2, 2.5, 3, 4, 10]), ("fine_0.25mm", np.r_[np.arange(0, 4, .25), 10])):
    P["resid"] = resid(P, edges)
    for setname, mem in (("core", DMN_CORE), ("core+lateral", DMN_ALL)):
        cl = classify(P, mem)
        res[f"B|{setname}|{ename}"] = {
            "n": pd.Series(cl).value_counts().to_dict(),
            "dd_minus_do_mouseCI": mouse_boot(P, cl, "dmn-dmn", "dmn-other"),
            "dd_minus_oo_mouseCI": mouse_boot(P, cl, "dmn-dmn", "other-other"),
            "do_minus_oo_mouseCI": mouse_boot(P, cl, "dmn-other", "other-other")}

P["resid"] = resid(P, [0, .5, 1, 1.5, 2, 2.5, 3, 4, 10])
cl = classify(P, DMN_CORE)
dd = P[cl == "dmn-dmn"]
res["dmn_pairs"] = dd.groupby(["r1", "r2"]).agg(n=("eid", "size"), mice=("mouse", "nunique"),
                                                resid=("resid", "mean"), dist=("dist_mm", "mean")).round(3).reset_index().to_dict("records")
# leave one DMN region out
loo = {}
for r in sorted(set(dd.r1) | set(dd.r2)):
    Q = P[(P.r1 != r) & (P.r2 != r)]
    c = classify(Q, DMN_CORE)
    if (c == "dmn-dmn").sum() >= 5:
        loo[r] = {"dd_minus_do": mouse_boot(Q, c, "dmn-dmn", "dmn-other", 500),
                  "dd_minus_oo": mouse_boot(Q, c, "dmn-dmn", "other-other", 500)}
res["leave_one_region_out"] = loo

# session-relative rest index
A = pd.read_parquet(OUT / f"30_neurons_{TAG}.pqt")
A["RIrel"] = A.RI - A.groupby("eid").RI.transform("median")
g = A.groupby("region")
RA = pd.DataFrame({"n": g.size(), "n_sess": g.eid.nunique(), "RIrel": g.RIrel.mean()})
RA = RA[(RA.n >= 100) & (RA.n_sess >= 3) & ~RA.index.isin(["root", "void"])]
res["A_RIrel"] = RA.RIrel.sort_values(ascending=False).round(3).to_dict()
isd = RA.index.isin(DMN_CORE)
obs = RA.RIrel[isd].mean() - RA.RIrel[~isd].mean()
null = [RA.RIrel.values[p][:isd.sum()].mean() - RA.RIrel.values[p][isd.sum():].mean()
        for p in (rng.permutation(len(RA)) for _ in range(20000))]
res["A_RIrel|core_vs_other"] = {"diff": round(float(obs), 4), "p_perm": float(np.mean(np.abs(null) >= abs(obs))),
                                 "n": [int(isd.sum()), int((~isd).sum())]}
B = pd.read_csv(OUT.parents[1] / "bwm" / "output" / "20_regions_A_full.csv", index_col=0)
cm = RA.index.intersection(B.index)
for c in ("RI_iti", "RI_spont", "suppressed"):
    if c in B:
        rho = spearmanr(B.loc[cm, c], RA.loc[cm, "RIrel"])
        res[f"crossmodal|bwm_{c}_vs_meso_RIrel"] = {"n": int(len(cm)), "spearman": round(float(rho.statistic), 3),
                                                    "p": round(float(rho.pvalue), 4)}
print(json.dumps(res, indent=1, default=str))
json.dump(res, open(OUT / f"32_meso_checks_{TAG}.json", "w"), indent=1, default=str)
