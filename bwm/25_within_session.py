"""B5: within-session DMN contrast (removes session-level coupling offsets).

For every session that has >= 1 DMN-DMN cortical pair and >= 1 DMN-other
cortical pair: contrast = mean distance/probe residual of DMN-DMN pairs minus
that of DMN-other pairs (same session). Mouse-level mean (sessions of a
mouse averaged), sign test and bootstrap over mice. Same for the
data-free control 'non-DMN cortical region pairs': other-other minus
other-DMN within session. Also a distance-matched version: each DMN-DMN pair
is compared with DMN-other pairs from the same session whose distance is
within 0.5 mm.

Usage: python 25_within_session.py <B tag> [set: core|all]
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from common import DMN_ALL, DMN_CORE, OUT

BTAG = sys.argv[1]
MEM = DMN_CORE if (len(sys.argv) > 2 and sys.argv[2] == "core") else DMN_ALL
rng = np.random.default_rng(0)
SUBJ = pd.read_csv(OUT / "bwm_insertions.csv").groupby("eid").subject.first()
P = pd.read_parquet(OUT / f"21_pairs_{BTAG}.pqt")
P["subject"] = P.eid.map(SUBJ)
P["dbin"] = pd.cut(P.dist_mm, [0, .5, 1, 1.5, 2, 3, 4, 6, 20], labels=False)
P["resid"] = P.z - P.groupby(["dbin", "same_probe"]).z.transform("mean")
C = P[(P.c1 == "Isocortex") & (P.c2 == "Isocortex")].copy()
d1, d2 = C.r1.isin(MEM), C.r2.isin(MEM)
C["cls"] = np.where(d1 & d2, "dd", np.where(d1 | d2, "do", "oo"))

rows, matched = [], []
for eid, g in C.groupby("eid"):
    dd, do = g[g.cls == "dd"], g[g.cls == "do"]
    if len(dd) and len(do):
        rows.append(dict(eid=eid, subject=g.subject.iat[0], n_dd=len(dd), n_do=len(do),
                         contrast=dd.resid.mean() - do.resid.mean(),
                         contrast_raw=dd.z.mean() - do.z.mean()))
        for _, p in dd.iterrows():
            m = do[(np.abs(do.dist_mm - p.dist_mm) <= 0.5) & (do.same_probe == p.same_probe)]
            if len(m):
                matched.append(dict(eid=eid, subject=p.subject, diff=p.z - m.z.mean()))
S = pd.DataFrame(rows)
Mt = pd.DataFrame(matched)
res = {"btag": BTAG, "set": "core" if MEM is DMN_CORE else "core+lateral", "n_sessions": int(len(S)),
       "n_subjects": int(S.subject.nunique()) if len(S) else 0}
for name, D, col in (("resid", S, "contrast"), ("raw", S, "contrast_raw"), ("dist_matched", Mt, "diff")):
    if not len(D):
        continue
    m = D.groupby("subject")[col].mean()
    bs = [m.sample(len(m), replace=True, random_state=int(rng.integers(1e9))).mean() for _ in range(5000)]
    res[name] = {"n_subjects": int(len(m)), "mean": round(float(m.mean()), 4),
                 "boot_ci95": np.quantile(bs, [.025, .975]).round(4).tolist(),
                 "frac_subjects_pos": round(float((m > 0).mean()), 3),
                 "wilcoxon_p": float(wilcoxon(m).pvalue) if len(m) >= 6 else None}
print(json.dumps(res, indent=1))
json.dump(res, open(OUT / f"25_within_{BTAG}_{res['set']}.json", "w"), indent=1)
