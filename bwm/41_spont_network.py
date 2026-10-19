"""Grey-screen (passive spontaneous) co-fluctuation: which region pairs exist, and do
DMN regions couple preferentially?

Input: 21_pairs_<BTAG>.pqt (21_full_B.py spont ... still): per session, region signals
(mean of z-scored neurons per region and probe, >= 5 neurons, nuisance-regressed for
wheel, licks, motion energy + 1-bin lags) on still bins of the grey-screen period;
pair r -> Fisher z. Coupling metric: 'resid' = z minus the mean z of pairs with the
same probe relation (same/different probe) and 3-D distance bin.

Outputs (output/41_*_<BTAG>.*):
  coverage: per region pair, sessions / mice / cross-probe sessions
  edges: Lu et al. 2012 and our DMN edges with mean resid and mouse-bootstrap CI
  classes: DMN-DMN vs DMN-other vs other-other (mouse bootstrap) for each region set
  cv: cross-validated rest-preference network from 42_spont_A (RI_grey)
  within: within-session DMN-DMN minus DMN-other contrast

Usage: python 41_spont_network.py <BTAG> [ATAG]
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from common import DMN_ALL, DMN_CORE, LU_DMN, LU_PARIETAL, LU_PFC_TEMPORAL, OUT, WHITESELL_DMN, CONS_DMN

BTAG = sys.argv[1] if len(sys.argv) > 1 else "spont_motion_1s_still"
ATAG = sys.argv[2] if len(sys.argv) > 2 else "G_full"
rng = np.random.default_rng(0)
SUBJ = pd.read_csv(OUT / "bwm_insertions.csv").groupby("eid").subject.first()
P = pd.read_parquet(OUT / f"21_pairs_{BTAG}.pqt")
P["subject"] = P.eid.map(SUBJ)
CTX = (P.c1 == "Isocortex") & (P.c2 == "Isocortex")
CTX_CA1 = (P.c1.eq("Isocortex") | P.r1.eq("CA1")) & (P.c2.eq("Isocortex") | P.r2.eq("CA1"))
# distance/probe baseline from cortex(+CA1) pairs only, 0.5 mm bins up to 4 mm
P = P[CTX_CA1].reset_index(drop=True)
CTX = (P.c1 == "Isocortex") & (P.c2 == "Isocortex")
CTX_CA1 = pd.Series(True, index=P.index)
P["dbin"] = pd.cut(P.dist_mm, [0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 20], labels=False)
P["resid"] = P.z - P.groupby(["dbin", "same_probe"]).z.transform("mean")
res = {"btag": BTAG, "baseline": "cortex+CA1 pairs, 0.5 mm bins x probe relation",
       "n_sessions": int(P.eid.nunique()), "n_mice": int(P.subject.nunique()),
       "n_pairs": int(len(P)), "n_pairs_cortex": int(CTX.sum()), "n_sessions_cortex_pair": int(P[CTX].eid.nunique()),
       "n_pairs_cross_probe_cortex": int((CTX & ~P.same_probe).sum())}


def mboot(Q, fn, n=2000):
    """Mouse bootstrap of fn(Q, weights)."""
    mice = Q.subject.unique()
    out = []
    for _ in range(n):
        cnt = pd.Series(rng.choice(mice, len(mice))).value_counts()
        out.append(fn(Q, Q.subject.map(cnt).fillna(0).values))
    return [round(float(fn(Q, np.ones(len(Q)))), 4)] + np.nanquantile(out, [.025, .975]).round(4).tolist()


def wmean(Q, w, sel):
    return np.sum(w[sel] * Q.resid.values[sel]) / max(w[sel].sum(), 1e-9)


# ---- coverage
cov = P.groupby(["r1", "r2"]).agg(n_sess=("eid", "nunique"), n_mice=("subject", "nunique"),
                                  n_cross=("same_probe", lambda s: int((~s).sum())),
                                  dist_mm=("dist_mm", "median"), resid=("resid", "mean"), z=("z", "mean"),
                                  c1=("c1", "first"), c2=("c2", "first")).reset_index()
xm = P[~P.same_probe].groupby(["r1", "r2"]).subject.nunique()
cov["n_cross_mice"] = [int(xm.get((a, b), 0)) for a, b in zip(cov.r1, cov.r2)]
cov.to_csv(OUT / f"41_coverage_{BTAG}.csv", index=False)


def edge_table(mem, name):
    E = cov[cov.r1.isin(mem) & cov.r2.isin(mem)].copy()
    rows = []
    for _, e in E.iterrows():
        Q = P[(P.r1 == e.r1) & (P.r2 == e.r2)]
        m = Q.groupby("subject").resid.mean()
        ci = np.quantile([rng.choice(m.values, len(m)).mean() for _ in range(2000)], [.025, .975]) if len(m) >= 3 else [np.nan] * 2
        rows.append({**e.to_dict(), "mouse_mean": m.mean(), "ci_lo": ci[0], "ci_hi": ci[1]})
    E = pd.DataFrame(rows)
    E.to_csv(OUT / f"41_edges_{name}_{BTAG}.csv", index=False)
    n_possible = len(mem) * (len(mem) - 1) // 2
    return {"n_possible_edges": n_possible, "n_edges_any": int(len(E)), "n_edges_ge3_mice": int((E.n_mice >= 3).sum()),
            "n_edges_cross_probe_ge3_mice": int((E.n_cross_mice >= 3).sum()) if len(E) else 0}


res["coverage"] = {"cons": edge_table(CONS_DMN, "cons"), "ours_core": edge_table(DMN_CORE, "ours_core"), "ours_all": edge_table(DMN_ALL, "ours_all"),
                   "lu": edge_table(LU_DMN, "lu"), "whitesell": edge_table(WHITESELL_DMN, "whitesell")}
# Lu hub edges: prefrontal (ACA, PL, ORB) x RSP
pfc = ["ACAd", "ACAv", "PL", "ORBm", "ORBl", "ORBvl", "ILA"]
rsp = ["RSPagl", "RSPd", "RSPv"]
hub = P[(P.r1.isin(pfc) & P.r2.isin(rsp)) | (P.r1.isin(rsp) & P.r2.isin(pfc))]
pfc_all = pfc + ["ILA"]
res["any_frontal_x_rsp_any_variant_note"] = "checked in review: 0 pairs in all grey/ITI variants"
res["lu_hub_pfc_rsp"] = {"n_pairs": int(len(hub)), "n_sessions": int(hub.eid.nunique()), "n_mice": int(hub.subject.nunique())}
mod = P[(P.r1.isin(LU_PFC_TEMPORAL) & P.r2.isin(LU_PARIETAL)) | (P.r1.isin(LU_PARIETAL) & P.r2.isin(LU_PFC_TEMPORAL))]
res["lu_between_module"] = {"n_pairs": int(len(mod)), "n_sessions": int(mod.eid.nunique()), "n_mice": int(mod.subject.nunique()),
                            "edges": mod.groupby(["r1", "r2"]).subject.nunique().to_dict().__repr__()}


# ---- classes
def classes(Q, mem):
    d1, d2 = Q.r1.isin(mem), Q.r2.isin(mem)
    return np.where(d1 & d2, "dmn-dmn", np.where(d1 | d2, "dmn-other", "other-other"))


SETS = {"cons": CONS_DMN, "ours_core": DMN_CORE, "ours_core+lateral": DMN_ALL, "lu_all": LU_DMN,
        "lu_pfc_temporal": LU_PFC_TEMPORAL, "lu_parietal": LU_PARIETAL, "whitesell": WHITESELL_DMN}
res["classes"] = {}
for sname, mem in SETS.items():
    base = CTX_CA1 if sname in ("lu_all", "lu_parietal") else CTX
    for scope, sel in (("cortex", base), ("cortex_cross_probe", base & ~P.same_probe)):
        Q = P[sel].reset_index(drop=True)
        cl = classes(Q, mem)
        out = {"n": pd.Series(cl).value_counts().to_dict(), "n_mice_dd": int(Q[cl == "dmn-dmn"].subject.nunique())}
        for a, b in (("dmn-dmn", "dmn-other"), ("dmn-dmn", "other-other"), ("dmn-other", "other-other")):
            if (cl == a).sum() >= 5 and (cl == b).sum() >= 5:
                out[f"{a} minus {b}"] = mboot(Q, lambda Q, w: wmean(Q, w, cl == a) - wmean(Q, w, cl == b))
        out["means"] = {c: round(float(Q.resid[cl == c].mean()), 4) for c in np.unique(cl)}
        v = out.get("dmn-dmn minus dmn-other")
        if v:
            out["mde_80pct_power"] = round(2.8 * (v[2] - v[1]) / 3.92, 4)
            # robustness: leave one session out, drop the strongest DMN-DMN pair
            loo = []
            for e in Q.eid[cl == "dmn-dmn"].unique():
                k = (Q.eid != e).values
                Qk, ck = Q[k].reset_index(drop=True), cl[k]
                if (ck == "dmn-dmn").sum() >= 2:
                    loo.append(wmean(Qk, np.ones(len(Qk)), ck == "dmn-dmn") - wmean(Qk, np.ones(len(Qk)), ck == "dmn-other"))
            out["loo_session_min_max"] = [round(float(min(loo)), 4), round(float(max(loo)), 4)] if loo else None
            top = Q.resid.where(cl == "dmn-dmn").idxmax()
            k = (Q.index != top)
            Qk, ck = Q[k].reset_index(drop=True), cl[k]
            out["drop_top_pair"] = round(float(wmean(Qk, np.ones(len(Qk)), ck == "dmn-dmn") - wmean(Qk, np.ones(len(Qk)), ck == "dmn-other")), 4)
            dd = Q[cl == "dmn-dmn"]
            out["dd_sessions_top2_share"] = round(float(dd.eid.value_counts().iloc[:2].sum() / len(dd)), 3)
            out["dd_median_dist"] = round(float(dd.dist_mm.median()), 2)
            out["do_median_dist"] = round(float(Q[cl == "dmn-other"].dist_mm.median()), 2)
        res["classes"][f"{sname}|{scope}"] = out

# ---- cross-validated rest-preference network (RI_grey from 42_spont_A)
N = pd.read_parquet(OUT / f"42_neurons_{ATAG}.pqt")
subj = np.intersect1d(N.subject.unique(), P.subject.unique())
ctx_regions = set(P.r1[P.c1 == "Isocortex"]) | set(P.r2[P.c2 == "Isocortex"])
cvs = {"rest-rest": [], "rest-task": [], "task-task": []}
cvx = {"rest-rest": [], "rest-task": [], "task-task": []}
for _ in range(200):
    h = rng.permutation(subj)
    halves = (set(h[: len(h) // 2]), set(h[len(h) // 2:]))
    for a, b in ((0, 1), (1, 0)):
        g = N[N.subject.isin(halves[a])].groupby("region").RI_grey.agg(["mean", "size"])
        g = g[(g["size"] >= 20) & g.index.isin(ctx_regions)]["mean"]
        t1, t2 = g.quantile([1 / 3, 2 / 3])
        lab = pd.Series(np.where(g >= t2, "rest", np.where(g <= t1, "task", "mid")), index=g.index)
        Q = P[P.subject.isin(halves[b]) & CTX]
        l1, l2 = Q.r1.map(lab), Q.r2.map(lab)
        for key, c in (("rest-rest", (l1 == "rest") & (l2 == "rest")),
                       ("rest-task", ((l1 == "rest") & (l2 == "task")) | ((l1 == "task") & (l2 == "rest"))),
                       ("task-task", (l1 == "task") & (l2 == "task"))):
            cvs[key].append(Q.resid[c].mean())
            cvx[key].append(Q.z[c & ~Q.same_probe].mean())
cvs = {k: np.array(v) for k, v in cvs.items()}
cvx = {k: np.array(v) for k, v in cvx.items()}
res["cv_restpref"] = {
    "resid_means": {k: round(float(np.nanmean(v)), 4) for k, v in cvs.items()},
    "resid_q025_q975": {k: np.nanquantile(v, [.025, .975]).round(4).tolist() for k, v in cvs.items()},
    "rest-rest minus rest-task": [round(float(np.nanmean(cvs["rest-rest"] - cvs["rest-task"])), 4),
                                  float(np.nanmean((cvs["rest-rest"] - cvs["rest-task"]) > 0))],
    "task-task minus rest-task": [round(float(np.nanmean(cvs["task-task"] - cvs["rest-task"])), 4),
                                  float(np.nanmean((cvs["task-task"] - cvs["rest-task"]) > 0))],
    "cross_probe_z_means": {k: round(float(np.nanmean(v)), 4) for k, v in cvx.items()},
    "note": "second entry = fraction of 400 half-splits with difference > 0"}

# ---- within-session contrast
res["within"] = {}
for sname, mem in SETS.items():
    sel = CTX if not sname.startswith("lu") else CTX_CA1
    Q = P[sel].copy()
    Q["cl"] = classes(Q, mem)
    g = Q.groupby(["eid", "cl"]).resid.mean().unstack()
    if {"dmn-dmn", "dmn-other"} <= set(g.columns):
        g = g.dropna(subset=["dmn-dmn", "dmn-other"])
        d = (g["dmn-dmn"] - g["dmn-other"]).groupby(g.index.map(SUBJ)).mean()
        res["within"][sname] = {"n_sessions": int(len(g)), "n_mice": int(len(d)), "mean": round(float(d.mean()), 4),
                                "frac_pos": round(float((d > 0).mean()), 3),
                                "wilcoxon_p": round(float(wilcoxon(d).pvalue), 4) if len(d) >= 6 else None}
# state invariance: same session + region pair, grey screen vs task ITI (0.5 s still bins)
try:
    A_ = pd.read_parquet(OUT / "21_pairs_spont_motion_0.5s_still.pqt")
    B_ = pd.read_parquet(OUT / "21_pairs_iti_motion_0.5s_still.pqt")
    m = A_.merge(B_, on=["eid", "r1", "r2", "same_probe"], suffixes=("_g", "_i"))
    from scipy.stats import spearmanr
    mc = m[(m.c1_g == "Isocortex") & (m.c2_g == "Isocortex")]
    res["grey_vs_iti_pair_z"] = {"n_pairs": int(len(m)), "n_sessions": int(m.eid.nunique()),
                                 "spearman_all": round(float(spearmanr(m.z_g, m.z_i).statistic), 3),
                                 "n_cortex": int(len(mc)),
                                 "spearman_cortex": round(float(spearmanr(mc.z_g, mc.z_i).statistic), 3) if len(mc) > 10 else None,
                                 "mean_z_grey": round(float(m.z_g.mean()), 4), "mean_z_iti": round(float(m.z_i.mean()), 4)}
except FileNotFoundError:
    pass
print(json.dumps(res, indent=1, default=str))
json.dump(res, open(OUT / f"41_spont_network_{BTAG}.json", "w"), indent=1, default=str)
