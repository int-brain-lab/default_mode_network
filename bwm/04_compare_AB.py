"""Do pilots A (rest preference) and B (spontaneous co-fluctuation) agree?

1. Pair level: distance/probe-residual co-fluctuation z of a region pair vs
   the rest preference of the two regions (RI_i + RI_j). A task-negative
   network predicts a positive slope: regions that both prefer rest
   co-fluctuate more at rest. Brain-wide and isocortex-only, with a
   session-cluster bootstrap.
2. Data-driven 'task-negative set' = top quartile of regions by RI: is
   within-set coupling higher than set-other and other-other?
3. Region level: B 'DMN affinity' (mean residual z with DMN-core partners
   minus with non-DMN cortical partners) vs A RI_task across cortical regions.
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import DMN_CORE, OUT

TAG = sys.argv[1] if len(sys.argv) > 1 else "dt25"
RI = sys.argv[2] if len(sys.argv) > 2 else "RI_task"
ZCOL = sys.argv[3] if len(sys.argv) > 3 else "z"
rng = np.random.default_rng(0)
P = pd.read_parquet(OUT / f"03_pairs_{TAG}.pqt")
R = pd.read_csv(OUT / "02_regions.csv", index_col=0)

P = P[P.r1.isin(R.index) & P.r2.isin(R.index)].copy()
P["z"] = P[ZCOL]
P = P[np.isfinite(P.z)]
P["dbin"] = pd.cut(P.dist_mm, [0, .5, 1, 1.5, 2, 3, 4, 6, 20], labels=False)
P["resid"] = P.z - P.groupby(["dbin", "same_probe"]).z.transform("mean")
P["ri1"], P["ri2"] = P.r1.map(R[RI]), P.r2.map(R[RI])
P["ri_sum"] = P.ri1 + P.ri2
eids = P.eid.unique()


def slope(Q, w=None):
    x, y = Q.ri_sum.values, Q.resid.values
    w = np.ones(len(x)) if w is None else w
    xm, ym = np.average(x, weights=w), np.average(y, weights=w)
    return np.sum(w * (x - xm) * (y - ym)) / np.sum(w * (x - xm) ** 2)


def boot(Q, fn, n=1000):
    e = Q.eid.unique()
    out = []
    for _ in range(n):
        cnt = pd.Series(rng.choice(e, len(e))).value_counts()
        w = Q.eid.map(cnt).fillna(0).values
        out.append(fn(Q, w))
    return np.nanquantile(out, [.025, .975]).round(4).tolist()


res = {"tag": TAG, "ri": RI, "zcol": ZCOL}
for name, Q in (("brainwide", P), ("isocortex", P[(P.c1 == "Isocortex") & (P.c2 == "Isocortex")]),
                ("brainwide_cross_probe", P[~P.same_probe])):
    rho = spearmanr(Q.ri_sum, Q.resid)
    res[f"pair_{name}"] = {"n_pairs": int(len(Q)), "n_sessions": int(Q.eid.nunique()),
                           "spearman": round(float(rho.statistic), 4), "p_naive": float(rho.pvalue),
                           "slope": round(float(slope(Q)), 4), "slope_boot_ci95": boot(Q, slope)}

# data-driven task-negative set
q75 = R[RI].quantile(.75)
tn = set(R.index[R[RI] >= q75])
P["cls"] = np.where(P.r1.isin(tn) & P.r2.isin(tn), "tn-tn", np.where(P.r1.isin(tn) | P.r2.isin(tn), "tn-other", "other-other"))


def cls_diff(Q, w):
    m = {c: np.average(Q.resid.values[(Q.cls == c).values], weights=w[(Q.cls == c).values]) if w[(Q.cls == c).values].sum() > 0 else np.nan
         for c in ("tn-tn", "tn-other", "other-other")}
    return m["tn-tn"] - m["tn-other"]


res["task_negative_set"] = {"n_regions": len(tn), "regions": sorted(tn),
                            "dmn_core_in_set": sorted(tn & set(DMN_CORE)),
                            "n_pairs": P.cls.value_counts().to_dict(),
                            "resid_mean": P.groupby("cls").resid.mean().round(4).to_dict(),
                            "boot_ci95_tn-tn_minus_tn-other": boot(P, cls_diff)}

# region-level DMN affinity among cortical regions
C = P[(P.c1 == "Isocortex") & (P.c2 == "Isocortex")]
aff = {}
for r in R.index[R.cosmos == "Isocortex"]:
    s = C[(C.r1 == r) | (C.r2 == r)]
    partner = np.where(s.r1 == r, s.r2, s.r1)
    isd = np.isin(partner, DMN_CORE)
    if isd.sum() >= 2 and (~isd).sum() >= 2:
        aff[r] = s.resid.values[isd].mean() - s.resid.values[~isd].mean()
aff = pd.Series(aff)
rho = spearmanr(aff.values, R.loc[aff.index, RI].values)
res["region_dmn_affinity_vs_RI"] = {"n_regions": int(len(aff)), "spearman": round(float(rho.statistic), 4),
                                    "p": float(rho.pvalue), "affinity": aff.round(3).to_dict()}
print(json.dumps(res, indent=1, default=str))
json.dump(res, open(OUT / f"04_compare_AB_{TAG}_{RI}_{ZCOL}.json", "w"), indent=1, default=str)
