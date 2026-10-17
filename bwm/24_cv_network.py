"""Cross-validated test of the 'task-negative network' (B4).

Mice are split at random into halves. In half 1, regions are ranked by rest
preference (A: RI_iti or suppressed fraction) and split into terciles. In
half 2, rest co-fluctuation (B pairs) is averaged per tercile class
(cross-probe pairs only, so neighbours do not drive it, plus all pairs with
the distance/probe residual). Swap halves, repeat over random splits.
Literature DMN regions are tested the same way (no fitting involved).

Usage: python 24_cv_network.py <B tag> <A tag> [A column]
"""
import json
import sys

import numpy as np
import pandas as pd

from common import DMN_ALL, OUT

BTAG, ATAG = sys.argv[1], sys.argv[2]
ACOL = sys.argv[3] if len(sys.argv) > 3 else "RI_iti"
N_SPLIT = 200
rng = np.random.default_rng(0)
SUBJ = pd.read_csv(OUT / "bwm_insertions.csv").groupby("eid").subject.first()
N = pd.read_parquet(OUT / f"20_neurons_{ATAG}.pqt")
P = pd.read_parquet(OUT / f"21_pairs_{BTAG}.pqt")
P["subject"] = P.eid.map(SUBJ)
P["dbin"] = pd.cut(P.dist_mm, [0, .5, 1, 1.5, 2, 3, 4, 6, 20], labels=False)
P["resid"] = P.z - P.groupby(["dbin", "same_probe"]).z.transform("mean")
subj = np.intersect1d(N.subject.unique(), P.subject.unique())
CLS = ["neg-neg", "neg-pos", "pos-pos"]


def tercile_map(Nh):
    g = Nh.groupby("region")[ACOL].agg(["mean", "size"])
    g = g[g["size"] >= 20]["mean"]
    q = pd.qcut(g, 3, labels=["pos", "mid", "neg"])  # high RI = task-negative
    return q


def classify(Q, lab):
    a, b = Q.r1.map(lab), Q.r2.map(lab)
    s = ["-".join(sorted([str(x), str(y)])) for x, y in zip(a, b)]
    return pd.Series(s, index=Q.index)


out = {c: {"xprobe_z": [], "resid": []} for c in CLS}
dmn = {c: {"xprobe_z": [], "resid": []} for c in ("dmn-dmn", "dmn-other", "other-other")}
for _ in range(N_SPLIT):
    h = set(rng.permutation(subj)[: len(subj) // 2])
    for train, test in ((True, False), (False, True)):
        lab = tercile_map(N[N.subject.isin(h) == train])
        Q = P[(P.subject.isin(h) == test)]
        Q = Q[Q.r1.isin(lab.index) & Q.r2.isin(lab.index)]
        cl = classify(Q, lab)
        for c in CLS:
            m = cl == c
            out[c]["xprobe_z"].append(Q.z[m & ~Q.same_probe].mean())
            out[c]["resid"].append(Q.resid[m].mean())
# literature DMN (all cortical regions present), bootstrap over mice
C = P[(P.c1 == "Isocortex") & (P.c2 == "Isocortex")]
d1, d2 = C.r1.isin(DMN_ALL), C.r2.isin(DMN_ALL)
lit = pd.Series(np.where(d1 & d2, "dmn-dmn", np.where(d1 | d2, "dmn-other", "other-other")), index=C.index)
for _ in range(N_SPLIT * 2):
    s = rng.choice(subj, len(subj))
    cnt = pd.Series(s).value_counts()
    w = C.subject.map(cnt).fillna(0).values
    for c in dmn:
        m = (lit == c).values
        mx = m & ~C.same_probe.values
        dmn[c]["xprobe_z"].append(np.sum(w[mx] * C.z.values[mx]) / max(w[mx].sum(), 1e-9))
        dmn[c]["resid"].append(np.sum(w[m] * C.resid.values[m]) / max(w[m].sum(), 1e-9))


def summ(d):
    return {c: {k: [round(float(np.nanmean(v)), 4), np.nanquantile(v, [.025, .975]).round(4).tolist()]
                for k, v in dd.items()} for c, dd in d.items()}


res = {"btag": BTAG, "atag": ATAG, "acol": ACOL, "n_subjects": int(len(subj)),
       "cv_terciles": summ(out), "literature_dmn": summ(dmn)}
diff = np.array(out["neg-neg"]["xprobe_z"]) - np.array(out["neg-pos"]["xprobe_z"])
res["cv_negneg_minus_negpos_xprobe"] = [round(float(np.nanmean(diff)), 4), np.nanquantile(diff, [.025, .975]).round(4).tolist(),
                                        "frac_splits_>0", round(float(np.nanmean(diff > 0)), 3)]
diff2 = np.array(out["neg-pos"]["xprobe_z"]) - np.array(out["pos-pos"]["xprobe_z"])
res["cv_negpos_minus_pospos_xprobe"] = [round(float(np.nanmean(diff2)), 4), np.nanquantile(diff2, [.025, .975]).round(4).tolist()]
print(json.dumps(res, indent=1))
json.dump(res, open(OUT / f"24_cv_{BTAG}_{ATAG}_{ACOL}.json", "w"), indent=1)
