"""Single-region resting-state metrics on the grey screen (no co-recording needed).

fMRI analogues that are computed for one region at a time, so they pool across animals:
  fALFF  per neuron: fraction of low-frequency power, P(0.01-0.1 Hz) / P(0.01-0.5 Hz), of the
         1 s binned rate over the whole grey-screen period after regressing wheel, licks and
         motion energy (+1 bin lags). The fMRI fALFF measures slow spontaneous fluctuation.
  CV     per neuron: std / mean of that 1 s rate (amplitude of fluctuation, ALFF-like).
  sync   per region and probe (ReHo-like local synchrony): mean pairwise correlation between
         its neurons in fully still 1 s bins, after the same nuisance regression.
All three depend on firing rate, so each is also given as the residual after regressing on
log rate across all units ('_adj'). Region sets are tested against other cortex as in
42_spont_A (region permutation + mouse bootstrap).

Usage: python 43_spont_single_region.py
"""
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import spearmanr

from common import (DMN_ALL, DMN_CORE, LU_DMN, LU_PARIETAL, LU_PFC_TEMPORAL, OUT, WHITESELL_DMN, CONS_DMN, beryl, cosmos,
                    done_eids)
from dmnstats import PRIMARY, perm_diff, region_table, split_half, subject_boot

FULL = OUT / "full"
SUBJ = pd.read_csv(OUT / "bwm_insertions.csv").groupby("eid").subject.first()
K = 20  # 50 ms -> 1 s
MIN_NEU, MIN_STILL = 5, 120


def coarse(x, how="mean"):
    n = len(x) // K
    x = x[: n * K].reshape(n, K, *x.shape[1:])
    return np.nanmean(x, 1) if how == "mean" else x.sum(1)


def design(cols):
    D = [np.ones(len(cols[0]))]
    for v in cols:
        if np.isfinite(v).mean() < .5 or np.nanstd(v) == 0:
            continue
        v = np.where(np.isfinite(v), v, np.nanmean(v))
        v = (v - v.mean()) / (v.std() + 1e-9)
        D += [v, np.r_[v[0], v[:-1]]]
    return np.array(D).T


def resid(Y, D):
    return Y - D @ np.linalg.lstsq(D, Y, rcond=None)[0]


def session(eid):
    s = dict(np.load(FULL / f"{eid}.npz", allow_pickle=True))
    sp = s["spont"]
    if not np.all(np.isfinite(sp)):
        return None
    BIN, nb = float(s["BIN"]), len(s["licks"])
    a, b = int(sp[0] / BIN) + 20, min(nb, int(sp[1] / BIN))
    w = s["wheel_speed"][a:b].astype(float)
    lk = s["licks"][a:b].astype(float)
    lk[lk == 255] = np.nan
    me = s["motion_energy"].astype(float)
    thr = np.nanpercentile(me, 60) if np.isfinite(me).mean() > .5 else np.inf
    me = me[a:b]
    mov = (np.nan_to_num(w) > 0.05) | (np.nan_to_num(lk) > 0) | (np.nan_to_num(me) > thr)
    still1 = coarse((~mov).astype(float)) == 1
    D = design([coarse(w), coarse(lk), coarse(me)])
    if len(D) < 256:
        return None
    neu, reg = [], []
    for f in sorted(FULL.glob(f"{eid}_probe*.npz")):
        p = np.load(f, allow_pickle=True)
        X = coarse(p["counts"][:, a:b].T.astype(np.float32), "sum")  # 1 s bins x neurons
        rate = X.mean(0)
        keep = rate >= 0.5
        X, rate = X[:, keep], rate[keep]
        if X.shape[1] == 0:
            continue
        R = resid(X, D)
        fr, P = welch(R, fs=1.0, nperseg=128, axis=0)
        lo = (fr >= 0.01) & (fr <= 0.1)
        al = (fr >= 0.01) & (fr <= 0.5)
        falff = P[lo].sum(0) / (P[al].sum(0) + 1e-12)
        cv = X.std(0) / (rate + 1e-9)
        regs = beryl(p["atlas_id"][keep])
        neu.append(pd.DataFrame({"eid": eid, "probe": str(p["probe"]), "atlas_id": p["atlas_id"][keep],
                                 "rate": rate, "fALFF": falff, "CV": cv}))
        if still1.sum() >= MIN_STILL:
            Xs = resid(X[still1], D[still1])
            Z = (Xs - Xs.mean(0)) / (Xs.std(0) + 1e-9)
            for r in np.unique(regs):
                sel = np.flatnonzero(regs == r)
                if len(sel) < MIN_NEU or r in ("root", "void"):
                    continue
                C = np.corrcoef(Z[:, sel].T)
                iu = np.triu_indices(len(sel), 1)
                reg.append({"eid": eid, "probe": str(p["probe"]), "region": r, "n": len(sel),
                            "sync": float(np.nanmean(C[iu])), "rate": float(rate[sel].mean()),
                            "n_still": int(still1.sum())})
    return (pd.concat(neu, ignore_index=True) if neu else None), pd.DataFrame(reg)


SETS = {"cons": CONS_DMN, "ours_core": DMN_CORE, "ours_core+lateral": DMN_ALL, "whitesell": WHITESELL_DMN, "lu_all": LU_DMN,
        "lu_pfc_temporal": LU_PFC_TEMPORAL, "lu_parietal": LU_PARIETAL}


def adjust(df, col, xcol="rate"):
    x = np.log(df[xcol].values)
    ok = np.isfinite(x) & np.isfinite(df[col].values)
    c = np.polyfit(x[ok], df[col].values[ok], 1)
    return df[col] - np.polyval(c, x)


def set_tests(T, R, cols):
    C = R[R.cosmos == "Isocortex"]
    out = {}
    for sname, mem in SETS.items():
        isd = C.index.isin(mem)
        for oname, oth in (("other_ctx", ~isd), ("higher_order_ctx", ~isd & ~C.index.isin(PRIMARY))):
            for c in cols:
                obs, p = perm_diff(C[c].values[isd], C[c].values[oth])
                out[f"{c}|{sname}|vs_{oname}"] = {
                    "dmn": round(float(C[c].values[isd].mean()), 4), "other": round(float(C[c].values[oth].mean()), 4),
                    "diff": round(obs, 4), "p_perm": round(p, 5), "n": [int(isd.sum()), int(oth.sum())],
                    "subject_boot_ci95": subject_boot(T, R, c, C.index[isd], C.index[oth])}
    return out


if __name__ == "__main__":
    with ProcessPoolExecutor(6) as ex:
        res = [r for r in ex.map(session, done_eids()) if r is not None]
    N = pd.concat([r[0] for r in res if r[0] is not None], ignore_index=True)
    N["region"], N["cosmos"] = beryl(N.atlas_id), cosmos(N.atlas_id)
    N["subject"] = N.eid.map(SUBJ)
    N = N[~N.region.isin(["root", "void"])].reset_index(drop=True)
    for c in ("fALFF", "CV"):
        N[f"{c}_adj"] = adjust(N, c)
    S = pd.concat([r[1] for r in res], ignore_index=True)
    from iblatlas.regions import BrainRegions
    br = BrainRegions()
    S["cosmos"] = cosmos(br.acronym2id(S.region.values))
    S["subject"] = S.eid.map(SUBJ)
    S["sync_adj"] = adjust(S, "sync")
    N.to_parquet(OUT / "43_neurons.pqt")
    S.to_parquet(OUT / "43_region_sessions.pqt")
    RN = region_table(N, ["fALFF", "CV", "fALFF_adj", "CV_adj", "rate"])
    RS = region_table(S, ["sync", "sync_adj", "rate"], min_n=3, min_sess=3)
    RN.to_csv(OUT / "43_regions_neuron_metrics.csv")
    RS.to_csv(OUT / "43_regions_sync.csv")
    G = pd.read_csv(OUT / "42_regions_G_full.csv", index_col=0)
    out = {"n_sessions": int(N.eid.nunique()), "n_mice": int(N.subject.nunique()), "n_neurons": int(len(N)),
           "n_region_sessions_sync": int(len(S)), "n_mice_sync": int(S.subject.nunique()),
           "rho_rate": {"fALFF": round(float(spearmanr(N.rate, N.fALFF).statistic), 3),
                        "CV": round(float(spearmanr(N.rate, N.CV).statistic), 3),
                        "sync": round(float(spearmanr(S.rate, S.sync).statistic), 3)},
           "reliability": {"fALFF": split_half(N, RN, "fALFF"), "CV": split_half(N, RN, "CV"),
                           "fALFF_adj": split_half(N, RN, "fALFF_adj")},
           "tests": {**set_tests(N, RN, ["fALFF", "CV", "fALFF_adj", "CV_adj"]),
                     **set_tests(S, RS, ["sync", "sync_adj"])}}
    for name, R, cols in (("neuron", RN, ["fALFF", "fALFF_adj", "CV", "CV_adj"]), ("sync", RS, ["sync", "sync_adj"])):
        C = R[R.cosmos == "Isocortex"]
        cm = C.index.intersection(G.index)
        out[f"vs_RI_grey_{name}"] = {c: round(float(spearmanr(C.loc[cm, c], G.loc[cm, "RI_grey"]).statistic), 3)
                                     for c in cols}
        out[f"ctx_ranking_{name}"] = {c: C[c].sort_values(ascending=False).round(4).to_dict() for c in cols}
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("ctx_")}, indent=1)[:5000])
    json.dump(out, open(OUT / "43_single_region.json", "w"), indent=1)
