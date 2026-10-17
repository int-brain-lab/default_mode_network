"""Pilot B (local data): spontaneous co-fluctuation between regions, sampled
once per trial, pooled across sessions into a sparse region x region matrix.

Per session (both probes pooled), per Beryl region with >= MIN_NEU neurons:
population rate per trial in the pre-stimulus still period = 'inter_trial'
(stimOn -1.15..-1.0) + quiescence (-0.4..-0.1), only trials without wheel
move or lick in the rest window. Each neuron is z-scored over trials first,
the region signal is the mean over its neurons. Optional detrending: subtract
a running median over DETREND trials (removes slow drift; set 0 to keep it).
Pair correlation -> Fisher z. Pairs are labelled same-probe / cross-probe, with
centroid distance.

DMN test (cortical pairs only): z ~ distance spline + same_probe + class,
class in {DMN-DMN, DMN-other, other-other}; session-cluster bootstrap.
Coverage table: how many sessions contribute each DMN pair.
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

from common import DMN_ALL, DMN_CORE, OUT, beryl, cosmos, insertions, load_cache

MIN_NEU = 5
MIN_TR = 40
DETREND = int(sys.argv[1]) if len(sys.argv) > 1 else 25
TAG = sys.argv[2] if len(sys.argv) > 2 else f"dt{DETREND}"
WINDOW = sys.argv[3] if len(sys.argv) > 3 else "rest+quies"
mv = pd.read_parquet(OUT / "01b_movement_trials.pqt")
ins = insertions()
rng = np.random.default_rng(0)


def per_trial(c, key):
    return pd.DataFrame(c[f"{key}__counts"], index=pd.Index(c[f"{key}__trials"], name="trial"))


pairs = []
regsig = {}
for eid, g in ins.groupby("eid"):
    sigs, meta = [], []
    caches = {p: load_cache(eid, p) for p in g.probe}
    # common trials across probes
    trs = None
    for p, c in caches.items():
        t = per_trial(c, "inter_trial").index.intersection(per_trial(c, "quiescence").index)
        trs = t if trs is None else trs.intersection(t)
    m = mv[mv.eid == eid].set_index("trial").loc[trs]
    trs = trs[((m.rest_wheel == 0) & (m.rest_licks.fillna(0) == 0)).values]
    if len(trs) < MIN_TR:
        continue
    for p, c in caches.items():
        parts = []
        if "rest" in WINDOW:
            parts.append(per_trial(c, "inter_trial").loc[trs].values)
        if "quies" in WINDOW:
            parts.append(per_trial(c, "quiescence").loc[trs].values)
        X = np.sum(parts, axis=0)  # trials x neurons
        reg = beryl(c["ids"])
        keep = (X.mean(0) > 0.05) & ~np.isin(reg, ["root", "void"])
        X = X[:, keep]
        reg = reg[keep]
        xyz = c["xyz"][keep]
        if DETREND:
            X = X - median_filter(X, size=(DETREND, 1), mode="nearest")
        sd = X.std(0)
        ok = sd > 0
        X, reg, xyz = X[:, ok], reg[ok], xyz[ok]
        X = (X - X.mean(0)) / X.std(0)
        for r in np.unique(reg):
            sel = reg == r
            if sel.sum() >= MIN_NEU:
                sigs.append(X[:, sel].mean(1))
                # split-half reliability of the region signal (random neuron halves, Spearman-Brown)
                idx = np.flatnonzero(sel)
                rh = []
                for _ in range(10):
                    pm = rng.permutation(idx)
                    h1, h2 = pm[: len(pm) // 2], pm[len(pm) // 2:]
                    rh.append(np.corrcoef(X[:, h1].mean(1), X[:, h2].mean(1))[0, 1])
                rh = np.nanmean(rh)
                meta.append((r, p, sel.sum(), *xyz[sel].mean(0), 2 * rh / (1 + rh) if rh > -1 else np.nan))
    if len(sigs) < 2:
        continue
    S = np.array(sigs)
    C = np.corrcoef(S)
    M = pd.DataFrame(meta, columns=["region", "probe", "n", "x", "y", "z", "rel"])
    regsig[eid] = (M, S)
    for i in range(len(M)):
        for j in range(i + 1, len(M)):
            a, b = M.iloc[i], M.iloc[j]
            if a.region == b.region:
                continue
            pairs.append((eid, a.region, b.region, a.probe == b.probe,
                          float(np.linalg.norm(a[["x", "y", "z"]].values.astype(float) - b[["x", "y", "z"]].values.astype(float)) * 1e3),
                          float(np.arctanh(np.clip(C[i, j], -0.999, 0.999))), len(trs), a.n, b.n,
                          float(C[i, j]), a.rel, b.rel))

P = pd.DataFrame(pairs, columns=["eid", "r1", "r2", "same_probe", "dist_mm", "z", "n_tr", "n1", "n2",
                                 "r", "rel1", "rel2"])
# disattenuated correlation (only for pairs where both signals are reliable)
REL_MIN = 0.2
okrel = (P.rel1 >= REL_MIN) & (P.rel2 >= REL_MIN)
P["r_dis"] = np.where(okrel, P.r / np.sqrt(P.rel1.clip(lower=REL_MIN) * P.rel2.clip(lower=REL_MIN)), np.nan)
P["z_dis"] = np.arctanh(P.r_dis.clip(-0.99, 0.99))
# order pair names
sw = P.r1 > P.r2
P.loc[sw, ["r1", "r2"]] = P.loc[sw, ["r2", "r1"]].values
P.loc[sw, ["rel1", "rel2"]] = P.loc[sw, ["rel2", "rel1"]].values
P["n_a"], P["n_b"] = P.n1, P.n2
from iblatlas.regions import BrainRegions  # noqa: E402
br = BrainRegions()
acr2cos = {a: cosmos(br.acronym2id(a))[0] for a in np.unique(np.r_[P.r1, P.r2])}
P["c1"], P["c2"] = P.r1.map(acr2cos), P.r2.map(acr2cos)
P["d1"], P["d2"] = P.r1.isin(DMN_CORE), P.r2.isin(DMN_CORE)
P["dl1"], P["dl2"] = P.r1.isin(DMN_ALL), P.r2.isin(DMN_ALL)
P.to_parquet(OUT / f"03_pairs_{TAG}.pqt")
pd.to_pickle(regsig, OUT / f"03_regsig_{TAG}.pkl")


def dmn_test(P, d1, d2, label):
    Q = P[(P.c1 == "Isocortex") & (P.c2 == "Isocortex")].copy()
    Q["cls"] = np.where(Q[d1] & Q[d2], "dmn-dmn", np.where(Q[d1] | Q[d2], "dmn-other", "other-other"))
    # distance-binned residual: subtract mean z of pairs in the same distance
    # bin and same/cross-probe stratum (non-parametric distance control)
    Q["dbin"] = pd.cut(Q.dist_mm, [0, .5, 1, 1.5, 2, 3, 4, 6, 20], labels=False)
    Q["resid"] = Q.z - Q.groupby(["dbin", "same_probe"]).z.transform("mean")
    out = {"n_pairs": Q.cls.value_counts().to_dict(),
           "n_sessions": Q.groupby("cls").eid.nunique().to_dict(),
           "raw_mean_z": Q.groupby("cls").z.mean().round(4).to_dict(),
           "resid_mean_z": Q.groupby("cls").resid.mean().round(4).to_dict(),
           "cross_probe_raw_mean_z": Q[~Q.same_probe].groupby("cls").z.mean().round(4).to_dict(),
           "cross_probe_n": Q[~Q.same_probe].cls.value_counts().to_dict()}
    eids = Q.eid.unique()
    bs = []
    for _ in range(1000):
        s = rng.choice(eids, len(eids))
        cnt = pd.Series(s).value_counts()
        w = Q.eid.map(cnt).fillna(0).values
        m = {}
        for cl in ("dmn-dmn", "dmn-other", "other-other"):
            sel = (Q.cls == cl).values
            m[cl] = np.sum(w[sel] * Q.resid.values[sel]) / max(np.sum(w[sel]), 1e-9)
        bs.append((m["dmn-dmn"] - m["dmn-other"], m["dmn-dmn"] - m["other-other"]))
    bs = np.array(bs)
    out["boot_ci95_dmn-dmn_minus_dmn-other"] = np.quantile(bs[:, 0], [.025, .975]).round(4).tolist()
    out["boot_ci95_dmn-dmn_minus_other-other"] = np.quantile(bs[:, 1], [.025, .975]).round(4).tolist()
    return out


res = {"tag": TAG, "detrend": DETREND, "window": WINDOW, "n_sessions": int(P.eid.nunique()),
       "n_pairs": int(len(P)), "n_unique_region_pairs": int(P[["r1", "r2"]].drop_duplicates().shape[0]),
       "frac_pairs_same_probe": round(float(P.same_probe.mean()), 3),
       "mean_z_same_probe": round(float(P[P.same_probe].z.mean()), 4),
       "mean_z_cross_probe": round(float(P[~P.same_probe].z.mean()), 4),
       "core": dmn_test(P, "d1", "d2", "core"),
       "core+lateral": dmn_test(P, "dl1", "dl2", "core+lateral")}
dm = P[P.d1 & P.d2]
res["dmn_core_pair_coverage"] = {f"{a}-{b}": int(v) for (a, b), v in dm.groupby(["r1", "r2"]).eid.nunique().sort_values(ascending=False).head(30).items()}
res["dmn_core_pairs_covered_of_possible"] = f"{dm[['r1', 'r2']].drop_duplicates().shape[0]} / {len(DMN_CORE) * (len(DMN_CORE) - 1) // 2}"
res["dmn_core_pairs_cross_probe"] = int(dm[~dm.same_probe][["r1", "r2"]].drop_duplicates().shape[0])
res["dmn_core_pairs_cross_probe_sessions"] = int(dm[~dm.same_probe].eid.nunique())
res = json.loads(json.dumps(res, default=str))
print(json.dumps(res, indent=1))
json.dump(res, open(OUT / f"03_cofluct_{TAG}.json", "w"), indent=1)
