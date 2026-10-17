"""Iteration A2 (pilot data): deactivation and the 'less sensorimotor' control.

1. Per neuron, Welch t-test of per-trial rates in still rest vs task windows
   (motor_init, move, feedback averaged per trial); fraction of neurons per
   region that are significantly suppressed during the task (task-negative
   neurons, p < 0.01 and rate lower in task) and significantly enhanced.
2. DMN vs other cortex again, but 'other' restricted to higher-order cortex
   (primary sensory/motor areas removed) and with a regression on the
   region's mean absolute task modulation |RI| (is the DMN effect more than
   weaker task drive?).
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

from common import DMN_ALL, DMN_CORE, OUT, beryl, cosmos, insertions, load_cache

PRIMARY = ["VISp", "SSp-bfd", "SSp-ll", "SSp-ul", "SSp-m", "SSp-n", "SSp-tr", "SSp-un", "MOp", "AUDp", "GU", "VISC"]
mv = pd.read_parquet(OUT / "01b_movement_trials.pqt")
rng = np.random.default_rng(1)


def rates(c, key):
    return pd.DataFrame(c[f"{key}__counts"] / c[f"{key}__dur"], index=pd.Index(c[f"{key}__trials"], name="trial"))


rows = []
for eid, probe in zip(insertions().eid, insertions().probe):
    c = load_cache(eid, probe)
    rest, mi = rates(c, "inter_trial"), rates(c, "motor_init")
    mvt = pd.concat([rates(c, "choiceL"), rates(c, "choiceR")])
    fb = pd.concat([rates(c, "fback1"), rates(c, "fback0")])
    tr = mi.index.intersection(mvt.index).intersection(fb.index)
    m = mv[mv.eid == eid].set_index("trial")
    rtr = rest.index[((m.loc[rest.index].rest_wheel == 0) & (m.loc[rest.index].rest_licks.fillna(0) == 0)).values]
    if len(rtr) < 10 or len(tr) < 20:
        continue
    Rr = rest.loc[rtr].values
    T = (mi.loc[tr].values + mvt.loc[tr].values + fb.loc[tr].values) / 3
    with np.errstate(invalid="ignore", divide="ignore"):
        t, p = ttest_ind(Rr, T, axis=0, equal_var=False)
    rs, rt = Rr.mean(0), T.mean(0)
    rows.append(pd.DataFrame({"eid": eid, "atlas_id": c["ids"], "r_rest": rs, "r_task": rt,
                              "RI": (rs - rt) / (rs + rt + 1e-9), "p": np.nan_to_num(p, nan=1.)}))
N = pd.concat(rows, ignore_index=True)
N["region"] = beryl(N.atlas_id)
N["cosmos"] = cosmos(N.atlas_id)
N = N[(~N.region.isin(["root", "void"])) & ((N.r_rest + N.r_task) / 2 >= 0.5)]
N["suppressed"] = (N.p < 0.01) & (N.RI > 0)
N["enhanced"] = (N.p < 0.01) & (N.RI < 0)
g = N.groupby("region")
R = pd.DataFrame({"n": g.size(), "n_sess": g.eid.nunique(), "cosmos": g.cosmos.agg(lambda s: s.mode().iat[0]),
                  "RI": g.RI.mean(), "absRI": g.RI.apply(lambda x: np.abs(x).mean()),
                  "frac_suppressed": g.suppressed.mean(), "frac_enhanced": g.enhanced.mean()})
R = R[(R.n >= 20) & (R.n_sess >= 2)]
R["dmn"] = np.where(R.index.isin(DMN_CORE), "core", np.where(R.index.isin(DMN_ALL), "lateral", "no"))
R.to_csv(OUT / "05_regions_A2.csv")

res = {"brainwide_frac_suppressed": round(float(N.suppressed.mean()), 4),
       "brainwide_frac_enhanced": round(float(N.enhanced.mean()), 4)}
C = R[R.cosmos == "Isocortex"]
for setname, mem in (("core", DMN_CORE), ("core+lateral", DMN_ALL)):
    for other_name, other in (("all_other_ctx", ~C.index.isin(mem)),
                              ("higher_order_ctx", ~C.index.isin(mem) & ~C.index.isin(PRIMARY))):
        isd = C.index.isin(mem)
        for col in ("RI", "frac_suppressed", "frac_enhanced"):
            v = C[col].values
            obs = v[isd].mean() - v[other].mean()
            lab = np.r_[np.ones(isd.sum()), np.zeros(other.sum())].astype(bool)
            vv = np.r_[v[isd], v[other]]
            null = np.array([vv[q].mean() - vv[~q].mean() for q in (rng.permutation(lab) for _ in range(20000))])
            res[f"{col}|{setname}|vs_{other_name}"] = {
                "dmn": round(float(v[isd].mean()), 4), "other": round(float(v[other].mean()), 4),
                "n_other": int(other.sum()), "p_perm": round(float((np.sum(np.abs(null) >= abs(obs)) + 1) / 20001), 4)}
    # regression: RI ~ dmn + absRI  (cortex)
    X = np.c_[np.ones(len(C)), C.index.isin(mem), C.absRI.values]
    beta = np.linalg.lstsq(X, C.RI.values, rcond=None)[0]
    nb = []
    for _ in range(5000):
        Xp = X.copy()
        Xp[:, 1] = rng.permutation(Xp[:, 1])
        nb.append(np.linalg.lstsq(Xp, C.RI.values, rcond=None)[0][1])
    res[f"RI~dmn+absRI|{setname}"] = {"beta_dmn": round(float(beta[1]), 4),
                                      "p_perm": round(float((np.sum(np.abs(nb) >= abs(beta[1])) + 1) / 5001), 4)}
res["frac_suppressed_top_ctx"] = C.frac_suppressed.sort_values(ascending=False).head(12).round(3).to_dict()
res["frac_suppressed_top_brain"] = R.frac_suppressed.sort_values(ascending=False).head(12).round(3).to_dict()
print(json.dumps(res, indent=1))
json.dump(res, open(OUT / "05_pilot_A2.json", "w"), indent=1)
