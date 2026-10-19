"""Statistics on the window rates of 44_windows.py for the conservative DMN core.

DMN-motivated contrasts per neuron, log2 ratio of session-averaged rates (+0.1 Hz):
  task_vs_iti     movefb / iti        task-negative: deactivation during movement and feedback vs inter-trial rest
  stim_vs_iti     stim / iti          deactivation at stimulus onset (before movement in most trials)
  quies_vs_iti    quies / iti         engagement without movement: waiting for the stimulus vs between trials
  grey_vs_iti     grey / iti          long passive rest vs the short rest between trials
  move_grey       grey_move / grey    movement without any task
  move_iti        iti_move / iti      movement between trials
  task_vs_grey    movefb / grey       the grey-screen version of task_vs_iti (post-task timing)
For each: (1) core alone (mean over mice of the core-neuron mean, bootstrap over mice); (2) core vs all other
cortex and vs all other regions brain-wide, at region level (permutation over regions) and as a mouse
bootstrap; (3) within session, core neurons minus non-core neurons recorded together (cortex / all), averaged per
mouse, bootstrap and Wilcoxon over mice. Suppressed fractions for the quies and iti references.
Trial-by-trial co-fluctuation (44_pairs): distance/probe residual, core-core vs core-other vs other-other cortical pairs.

Usage: python 45_window_stats.py
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from common import CONS_DMN, OUT
from dmnstats import perm_diff

rng = np.random.default_rng(0)
N = pd.read_parquet(OUT / "44_neurons.pqt")
WIN = ["stim", "movefb", "quies", "iti", "iti_move", "grey", "grey_move"]
CON = {"task_vs_iti": ("movefb", "iti"), "stim_vs_iti": ("stim", "iti"), "quies_vs_iti": ("quies", "iti"),
       "grey_vs_iti": ("grey", "iti"), "move_grey": ("grey_move", "grey"), "move_iti": ("iti_move", "iti"),
       "task_vs_grey": ("movefb", "grey")}
for k, (a, b) in CON.items():
    N[k] = np.log2((N[a] + 0.1) / (N[b] + 0.1))
N["core"] = N.region.isin(CONS_DMN)
N["ctx"] = N.cosmos == "Isocortex"
for ref in ("quies", "iti"):
    N[f"supp_{ref}"] = np.where(N[f"p_{ref}"].notna(), ((N[f"p_{ref}"] < .01) & (N[ref] > N.movefb)).astype(float), np.nan)


def mouse_mean_ci(v, subj, n=5000):
    m = pd.Series(v).groupby(subj).mean().dropna()
    bs = [rng.choice(m.values, len(m)).mean() for _ in range(n)]
    return [round(float(m.mean()), 4)] + np.quantile(bs, [.025, .975]).round(4).tolist() + [int(len(m))]


def mouse_diff(A, B, col, n=5000):
    """Mouse bootstrap of mean(A) - mean(B), neurons pooled per mouse."""
    a = A.groupby("subject")[col].mean()
    b = B.groupby("subject")[col].mean()
    mice = np.union1d(a.index, b.index)
    out = []
    for _ in range(n):
        s = rng.choice(mice, len(mice))
        out.append(np.nanmean(a.reindex(s).values) - np.nanmean(b.reindex(s).values))
    return [round(float(a.mean() - b.mean()), 4)] + np.nanquantile(out, [.025, .975]).round(4).tolist()


def within_session(D, col, other_mask):
    g = D[D.core | other_mask].assign(grp=lambda x: np.where(x.core, "core", "other"))
    t = g.groupby(["eid", "grp"])[col].mean().unstack().dropna()
    if not {"core", "other"} <= set(t.columns) or len(t) < 3:
        return None
    d = (t["core"] - t["other"]).groupby(t.index.map(D.groupby("eid").subject.first())).mean()
    bs = [rng.choice(d.values, len(d)).mean() for _ in range(5000)]
    return {"n_sessions": int(len(t)), "n_mice": int(len(d)), "mean": round(float(d.mean()), 4),
            "ci": np.quantile(bs, [.025, .975]).round(4).tolist(), "frac_mice_pos": round(float((d > 0).mean()), 3),
            "wilcoxon_p": round(float(wilcoxon(d).pvalue), 4) if len(d) >= 6 else None}


res = {"n_neurons": int(len(N)), "n_sessions": int(N.eid.nunique()), "n_mice": int(N.subject.nunique()),
       "n_core_neurons": int(N.core.sum()), "n_core_mice": int(N[N.core].subject.nunique()),
       "contrasts": {}, "rates": {}, "suppressed": {}}
# region table (regions with >= 20 neurons in >= 2 sessions)
g = N.groupby("region")
R = pd.DataFrame({"n": g.size(), "n_sess": g.eid.nunique(), "n_subj": g.subject.nunique(),
                  "cosmos": g.cosmos.agg(lambda s: s.mode().iat[0]),
                  **{c: g[c].mean() for c in list(CON) + WIN + ["supp_quies", "supp_iti"]}})
R = R[(R.n >= 20) & (R.n_sess >= 2)]
R.to_csv(OUT / "45_regions.csv")
core_r = R.index.isin(CONS_DMN)
ctx_r = R.cosmos == "Isocortex"
for k in CON:
    v = N[k]
    ok = np.isfinite(v)
    D = N[ok]
    out = {"core_alone": mouse_mean_ci(D[D.core][k].values, D[D.core].subject.values),
           "other_ctx_alone": mouse_mean_ci(D[~D.core & D.ctx][k].values, D[~D.core & D.ctx].subject.values),
           "other_all_alone": mouse_mean_ci(D[~D.core][k].values, D[~D.core].subject.values)}
    for oname, om_r, om_n in (("vs_other_ctx", ~core_r & ctx_r, ~D.core & D.ctx), ("vs_all_other", ~core_r, ~D.core)):
        obs, p = perm_diff(R[k].values[core_r], R[k].values[om_r])
        out[oname] = {"region_diff": round(obs, 4), "region_perm_p": round(p, 4), "n_regions": [int(core_r.sum()), int(om_r.sum())],
                      "mouse_boot": mouse_diff(D[D.core], D[om_n], k),
                      "within_session": within_session(D, k, om_n)}
    res["contrasts"][k] = out
for w in WIN:
    res["rates"][w] = {"core": mouse_mean_ci(N[N.core][w].values, N[N.core].subject.values),
                       "other_ctx": mouse_mean_ci(N[~N.core & N.ctx][w].values, N[~N.core & N.ctx].subject.values),
                       "other_all": mouse_mean_ci(N[~N.core][w].values, N[~N.core].subject.values),
                       **{r: mouse_mean_ci(N[N.region == r][w].values, N[N.region == r].subject.values) for r in CONS_DMN}}
for ref in ("quies", "iti"):
    c = f"supp_{ref}"
    obs, p = perm_diff(R[c].dropna().values[R[c].dropna().index.isin(CONS_DMN)],
                       R[c].dropna().values[~R[c].dropna().index.isin(CONS_DMN) & (R.loc[R[c].dropna().index, "cosmos"] == "Isocortex").values])
    D = N[N[c].notna()]
    res["suppressed"][ref] = {"core": round(float(R.loc[core_r, c].mean()), 4),
                              "other_ctx": round(float(R.loc[~core_r & ctx_r, c].mean()), 4),
                              "region_perm_p_vs_ctx": round(p, 4),
                              "mouse_boot_vs_ctx": mouse_diff(D[D.core], D[~D.core & D.ctx], c),
                              "within_session_vs_ctx": within_session(D, c, ~D.core & D.ctx)}
# coupling across trials
P = pd.read_parquet(OUT / "44_pairs.pqt")
P = P[(P.c1.eq("Isocortex")) & (P.c2.eq("Isocortex"))].copy()
P["z"] = np.arctanh(P.r.clip(-.999, .999))
res["coupling"] = {}
for wname, Q in P.groupby("window"):
    Q = Q.copy()
    Q["dbin"] = pd.cut(Q.dist_mm, [0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 20], labels=False)
    Q["resid"] = Q.z - Q.groupby(["dbin", "same_probe"]).z.transform("mean")
    d1, d2 = Q.r1.isin(CONS_DMN), Q.r2.isin(CONS_DMN)
    cl = np.where(d1 & d2, "core-core", np.where(d1 | d2, "core-other", "other-other"))
    Q["cl"] = cl
    o = {"n": pd.Series(cl).value_counts().to_dict(), "n_mice_cc": int(Q[cl == "core-core"].subject.nunique()),
         "means": Q.groupby("cl").resid.mean().round(4).to_dict()}
    mice = Q.subject.unique()
    for a, b in (("core-core", "core-other"), ("core-other", "other-other")):
        if (cl == a).sum() >= 5 and (cl == b).sum() >= 5:
            f = lambda w: (np.sum(w[cl == a] * Q.resid.values[cl == a]) / max(w[cl == a].sum(), 1e-9)
                           - np.sum(w[cl == b] * Q.resid.values[cl == b]) / max(w[cl == b].sum(), 1e-9))
            bs = []
            for _ in range(2000):
                cnt = pd.Series(rng.choice(mice, len(mice))).value_counts()
                bs.append(f(Q.subject.map(cnt).fillna(0).values))
            o[f"{a} minus {b}"] = [round(float(f(np.ones(len(Q)))), 4)] + np.nanquantile(bs, [.025, .975]).round(4).tolist()
    res["coupling"][wname] = o
print(json.dumps(res, indent=1, default=str)[:12000])
json.dump(res, open(OUT / "45_window_stats.json", "w"), indent=1, default=str)
