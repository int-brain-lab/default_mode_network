"""Region-level statistics shared by the full-data analyses."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import DMN_ALL, DMN_CORE

PRIMARY = ["VISp", "SSp-bfd", "SSp-ll", "SSp-ul", "SSp-m", "SSp-n", "SSp-tr", "SSp-un", "MOp", "AUDp", "GU", "VISC"]
SETS = {"core": DMN_CORE, "core+lateral": DMN_ALL}


def region_table(N, cols, min_n=20, min_sess=2):
    g = N.groupby("region")
    R = pd.DataFrame({"n": g.size(), "n_sess": g.eid.nunique(), "n_subj": g.subject.nunique(),
                      "cosmos": g.cosmos.agg(lambda s: s.mode().iat[0])})
    for c in cols:
        R[c] = g[c].mean()
    R = R[(R.n >= min_n) & (R.n_sess >= min_sess)].copy()
    R["dmn"] = np.where(R.index.isin(DMN_CORE), "core", np.where(R.index.isin(DMN_ALL), "lateral", "no"))
    return R


def split_half(N, R, col, unit="subject", n=300, seed=0):
    """Median Spearman r between region maps from two random halves of
    subjects (or sessions), and Spearman-Brown."""
    rng = np.random.default_rng(seed)
    N = N[N.region.isin(R.index)]
    units = N[unit].unique()
    rs = []
    for _ in range(n):
        h = set(rng.permutation(units)[: len(units) // 2])
        m = N[unit].isin(h)
        a = N[m].groupby("region")[col].agg(["mean", "size"])
        b = N[~m].groupby("region")[col].agg(["mean", "size"])
        idx = a.index[(a["size"] >= 5)].intersection(b.index[b["size"] >= 5])
        rs.append(spearmanr(a.loc[idx, "mean"], b.loc[idx, "mean"]).statistic)
    r = float(np.nanmedian(rs))
    return {"r": round(r, 3), "spearman_brown": round(2 * r / (1 + r), 3),
            "q05_q95": np.nanquantile(rs, [.05, .95]).round(3).tolist()}


def perm_diff(v_in, v_out, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    obs = v_in.mean() - v_out.mean()
    vv = np.r_[v_in, v_out]
    lab = np.r_[np.ones(len(v_in)), np.zeros(len(v_out))].astype(bool)
    null = np.array([vv[q].mean() - vv[~q].mean() for q in (rng.permutation(lab) for _ in range(n))])
    return float(obs), float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n + 1))


def subject_boot(N, R, col, members, others, n=1000, seed=0):
    """Bootstrap over subjects of (mean over DMN regions - mean over other
    regions) of region means."""
    rng = np.random.default_rng(seed)
    N = N[N.region.isin(R.index)]
    subj = np.sort(N.subject.unique())
    regs = R.index.values
    si = pd.Index(subj).get_indexer(N.subject)
    rj = pd.Index(regs).get_indexer(N.region)
    S = np.zeros((len(subj), len(regs)))
    C = np.zeros_like(S)
    np.add.at(S, (si, rj), N[col].values)
    np.add.at(C, (si, rj), 1)
    W = rng.multinomial(len(subj), np.ones(len(subj)) / len(subj), size=n).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        M = (W @ S) / (W @ C)
    a = np.isin(regs, members)
    b = np.isin(regs, others)
    d = np.nanmean(M[:, a], 1) - np.nanmean(M[:, b], 1)
    return np.nanquantile(d, [.025, .975]).round(4).tolist()


def dmn_tests(N, R, cols):
    out = {}
    C = R[R.cosmos == "Isocortex"]
    for setname, mem in SETS.items():
        isd = C.index.isin(mem)
        for oname, other in (("other_ctx", ~isd), ("higher_order_ctx", ~isd & ~C.index.isin(PRIMARY))):
            for col in cols:
                obs, p = perm_diff(C[col].values[isd], C[col].values[other])
                out[f"{col}|{setname}|vs_{oname}"] = {
                    "dmn": round(float(C[col].values[isd].mean()), 4),
                    "other": round(float(C[col].values[other].mean()), 4),
                    "diff": round(obs, 4), "p_perm": round(p, 5), "n_dmn": int(isd.sum()), "n_other": int(other.sum()),
                    "subject_boot_ci95": subject_boot(N, R, col, C.index[isd], C.index[other])}
    return out
