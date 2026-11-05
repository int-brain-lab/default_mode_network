"""Pilot, step 2: the same test as 10_meso_repertoire.py with task subspaces that are reliable across trials.

In step 1 the evoked subspace (top 5 dims of 16 trial-averaged patterns over ~7,600 ROIs) was noisy: odd- and
even-trial subspaces shared only 22% of their variance, which deflates the alignment with rest. Two cleaner task
subspaces:
  evoked_cv   cross-validated evoked dims: the top K eigenvectors of P_odd P_even' + P_even P_odd' (patterns from
              odd and even trials), i.e. only the dims in which both halves agree.
  trials      dominant dims of single-trial task activity: top K PCs of the evoked single-trial window values
              (window - pre, 4 windows x trials, odd trials), whose reproduction in even trials is reported.
Rest = still bins of the passive block, and still bins of the grey screen alone; the uniform direction is projected
out. Nulls as in step 1: ROI permutation (same permutation for both halves) and covariance-matched random subspaces.

Outputs: output/11_meso_repertoire_denoised.csv.
Usage: python 11_meso_repertoire_denoised.py
"""
import importlib.util
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("r10", HERE / "10_meso_repertoire.py")
r10 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r10)
DEV, T, orth, align, states, WINS = r10.DEV, r10.T, r10.orth, r10.align, r10.states, r10.WINS
K, NPERM, NCOV, SEED = 5, 100, 200, 20261104


def cv_dims(A, B, k):
    """Top-k eigenvectors of the symmetrised cross-product A B' + B A' (dims shared by two noisy halves)."""
    Q, _ = torch.linalg.qr(torch.cat([A, B], 1))
    a, b = Q.T @ A, Q.T @ B
    M = a @ b.T
    M = M + M.T
    ev, V = torch.linalg.eigh(M)
    return Q @ V[:, torch.argsort(ev, descending=True)[:k]]


def session(s, p0, p1, gen):
    Xb = s["Xbin"].astype(np.float32)
    fin = np.isfinite(Xb)
    S, M, G = states(s, p0, p1)
    S &= fin[:20].all(0)
    G &= fin[:20].all(0)
    tm = s["mask"].astype(bool) & np.isfinite(s["feedbackType"])
    side = np.where(np.isfinite(s["contrastLeft"]), 0, 1)[tm]
    fb = s["feedbackType"][tm]
    Xw = s["Xwin"][:, tm]
    good = np.isfinite(Xb[:, S | M]).all(1) & np.isfinite(Xw).all((1, 2))
    mu = np.nanmean(np.where(fin, Xb, np.nan), 1)
    sd = np.nanstd(np.where(fin, Xb, np.nan), 1)
    good &= sd > 0
    if S.sum() < 40 or good.sum() < 50:
        return None
    mu, sd = mu[good, None], sd[good, None]
    Z = lambda A: (A - mu) / sd
    n = int(good.sum())
    wn = list(s["win_names"])
    W = (Xw[good] - mu[..., None]) / sd[..., None]
    E = np.stack([W[:, :, wn.index(w)] - W[:, :, wn.index("pre")] for w in WINS], 2)  # ROI x trials x 4
    odd = np.arange(tm.sum()) % 2 == 1

    def patterns(tr):
        cols = []
        for j in range(len(WINS)):
            for a in (0, 1):
                for b in (-1, 1):
                    t = tr & (side == a) & (fb == b)
                    if t.sum() >= 3:
                        cols.append(E[:, t, j].mean(1))
        return cols

    po, pe = patterns(odd), patterns(~odd)
    m = min(len(po), len(pe))
    Po, Pe = T(np.stack(po[:m], 1)), T(np.stack(pe[:m], 1))
    So = T(E[:, odd].reshape(n, -1))
    Se = T(E[:, ~odd].reshape(n, -1))
    g = torch.ones(n, 1, device=DEV) / n ** .5
    R = lambda A: A - g @ (g.T @ A)
    Po, Pe, So, Se = R(Po), R(Pe), R(So - So.mean(1, keepdim=True)), R(Se - Se.mean(1, keepdim=True))
    D = T(Z(Xb[good][:, fin[good].all(0)]))
    D = R(D - D.mean(1, keepdim=True))
    Ud, Sd, _ = torch.linalg.svd(D, full_matrices=False)
    Ud, Sd = Ud[:, :500], Sd[:500]
    cov = [orth(Ud @ (Sd[:, None] * torch.randn(len(Sd), K, device=DEV, generator=gen)), K) for _ in range(NCOV)]
    subs = {"evoked_cv": cv_dims(Po, Pe, K), "trials": orth(So, K)}
    perms = [torch.randperm(n, device=DEV, generator=gen) for _ in range(NPERM)]
    nulls = {"evoked_cv": [cv_dims(Po[p], Pe[p], K) for p in perms], "trials": [orth(So[p], K) for p in perms]}
    # reproduction of the single-trial PCs in even trials: variance share captured, relative to even trials' own PCs
    tr_rep = align(subs["trials"], Se, K)[0]
    rest = {"still": T(Z(Xb[good][:, S]))}
    if G.sum() >= 30:
        rest["grey_still"] = T(Z(Xb[good][:, G]))
    rows = []
    for key, X in rest.items():
        X = R(X - X.mean(1, keepdim=True))
        ac = np.array([align(U, X, K)[0] for U in cov])
        for sub, U in subs.items():
            a, v = align(U, X, K)
            ap = np.array([align(Un, X, K)[0] for Un in nulls[sub]])
            rows.append(dict(eid=str(s["eid"]), rest=key, subspace=sub, n_rois=n, n_bins=X.shape[1], A=a, V=v,
                             A_perm=ap.mean(), A_perm95=np.quantile(ap, .95), A_cov=ac.mean(),
                             A_cov95=np.quantile(ac, .95), trials_reproduction=tr_rep))
    return rows


if __name__ == "__main__":
    MESO = r10.MESO
    SE = pd.read_csv(MESO / "output" / "34_still_sessions.csv").dropna(subset=["obs_still"])
    qc = pd.read_json(MESO / "output" / "35_meso_still_stats.json", typ="series")["qc_excluded"]
    SE = SE[~SE.eid.isin(qc)]
    subj = pd.read_csv(MESO / "output" / "session_subjects.csv").drop_duplicates("session").set_index("session").subject
    PP = r10.m34.passive_periods(SE.eid.tolist())
    gen = torch.Generator(device=DEV).manual_seed(SEED)
    load = lambda e: dict(np.load(MESO / "output" / "sessions" / f"{e}.npz", allow_pickle=True))
    rows = []
    with ThreadPoolExecutor(3) as ex:
        for i, (eid, s) in enumerate(zip(SE.eid, ex.map(load, SE.eid))):
            r = session(s, *PP.loc[eid, ["t0", "t1"]], gen)
            if r:
                rows += r
                print(i, eid[:8], " ".join(f"{q['subspace']}:{q['A']:.3f}/{q['A_perm']:.3f}/{q['A_cov']:.3f}"
                                           for q in r if q["rest"] == "still"), flush=True)
    out = pd.DataFrame(rows)
    out["subject"] = out.eid.map(subj)
    out.to_csv(r10.OUT / "11_meso_repertoire_denoised.csv", index=False)
    print(len(out.eid.unique()), "sessions,", out.subject.nunique(), "mice")
