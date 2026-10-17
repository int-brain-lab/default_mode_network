"""A5: is the DMN rest-preference effect just 'prefrontal'?

Matched comparisons within the isocortex:
  anterior : DMN (ACAd, ACAv, PL, ILA, ORBm, ORBl, ORBvl) vs non-DMN frontal
             (MOs, MOp, FRP, AId, AIv, AIp)
  posterior: DMN (RSPagl, RSPd, RSPv, VISa, VISam) vs non-DMN posterior
             (VISp, VISl, VISpm, VISal, VISrl, VISli, VISpl, VISpor, SSp-*, SSs, AUD*)
  lateral  : TEa, ECT, PERI vs AUDp, AUDd, AUDv, AUDpo, VISC, GU
Per contrast: region-level permutation p and a mouse bootstrap (neurons of
the resampled mice pooled per region).

Usage: python 26_A_subsets.py <A tag> [columns...]
"""
import json
import sys

import numpy as np
import pandas as pd

from common import OUT
from dmnstats import perm_diff, subject_boot

ATAG = sys.argv[1]
COLS = sys.argv[2:] or ["suppressed", "RI_iti", "RI_spont"]
N = pd.read_parquet(OUT / f"20_neurons_{ATAG}.pqt")
R = pd.read_csv(OUT / f"20_regions_{ATAG}.csv", index_col=0)
C = R[R.cosmos == "Isocortex"]
POST_OTHER = [r for r in C.index if r.startswith(("VISp", "VISl", "VISal", "VISrl", "VISli", "VISpl", "VISpor", "SSp", "SSs", "AUD"))
              and r not in ("VISa", "VISam")]
GROUPS = {
    "anterior": (["ACAd", "ACAv", "PL", "ILA", "ORBm", "ORBl", "ORBvl"], ["MOs", "MOp", "FRP", "AId", "AIv", "AIp"]),
    "posterior": (["RSPagl", "RSPd", "RSPv", "VISa", "VISam"], POST_OTHER),
    "lateral": (["TEa", "ECT", "PERI"], ["AUDp", "AUDd", "AUDv", "AUDpo", "VISC", "GU"]),
}
res = {"atag": ATAG}
for g, (dmn, oth) in GROUPS.items():
    dm = [r for r in dmn if r in C.index]
    ot = [r for r in oth if r in C.index]
    for col in COLS:
        if col not in C:
            continue
        a, b = C.loc[dm, col].dropna(), C.loc[ot, col].dropna()
        if len(a) < 2 or len(b) < 2:
            continue
        obs, p = perm_diff(a.values, b.values)
        res[f"{g}|{col}"] = {"dmn": dict(a.round(3)), "other": dict(b.round(3)),
                             "dmn_mean": round(float(a.mean()), 4), "other_mean": round(float(b.mean()), 4),
                             "diff": round(obs, 4), "p_perm": round(p, 4),
                             "mouse_boot_ci95": subject_boot(N[np.isfinite(N[col])], R, col, a.index, b.index)}
print(json.dumps(res, indent=1))
json.dump(res, open(OUT / f"26_A_subsets_{ATAG}.json", "w"), indent=1)
