"""Pilot: do resting fluctuations sample the task-evoked repertoire? (IBL mesoscope)

Test of the central prediction of Deco, Jirsa & McIntosh 2013 (Trends Neurosci 36:268): at rest the network sits in
a low-activity state from which noise drives excursions towards the configurations that tasks recruit, so resting
fluctuations should be concentrated in the task-evoked dimensions.

Per session (the 48 sessions of the comprehensive report, section 12; meso/output/sessions/<eid>.npz):
  units     each ROI z-scored over all 1 s bins of the recording (task + passive)
  evoked    task-evoked patterns: per-trial window means (Xwin; trials passing the BWM trial mask) minus the
            pre-stimulus window, averaged per condition (stimulus side L/R x outcome correct/error, >= 5 trials) in the
            stim, choice, fback and post_fb windows -> up to 16 ROI-space patterns. Task subspace U = top K left
            singular vectors.
  rest      still 1 s bins of the passive block (grey screen + receptive-field mapping; states as section 12),
            centred on their mean. Also the moving bins of the block, and still bins of the grey screen alone.
  alignment A = ||U' X||^2 / (sum of the top K squared singular values of X): the share of rest variance in the task
            subspace, relative to the most that any K-dim subspace could capture (Elsayed et al. 2016).
            Also V = ||U' X||^2 / ||X||^2, the plain share of variance.
  nulls     (1) ROI permutation of the evoked patterns (keeps their spectrum, breaks the ROI correspondence);
            (2) covariance-matched random subspaces: K random vectors drawn from the covariance of all recording bins
            (Elsayed & Cunningham 2017), i.e. how aligned an arbitrary high-variance subspace would be.
  global    every analysis is repeated after projecting out the uniform direction (all ROIs up together), which
            otherwise dominates both evoked patterns and rest fluctuations.
  windows   alignment of rest with each window's own 2-dim subspace (4 conditions per window).
  halves    evoked subspace from odd against even trials: the subspace overlap measures how reliable U is.
Runs on the GPU (torch) where available.

Outputs: output/10_meso_repertoire.csv (one row per session x variant).
Usage: python 10_meso_repertoire.py
"""
import importlib.util
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
MESO = HERE.parent / "meso"
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)
_spec = importlib.util.spec_from_file_location("m34", MESO / "34_meso_still.py")
m34 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m34)

K, NPERM, NCOV, MIN_TR, SEED = 5, 200, 200, 5, 20261104
WINS = ["stim", "choice", "fback", "post_fb"]
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T = lambda a: torch.as_tensor(np.asarray(a, np.float32), device=DEV)


def states(s, p0, p1):
    """Passive block (grey + RFM) still / moving masks over the recording's 1 s bins, as 34_meso_still.py."""
    nb = s["Xbin"].shape[1]
    blk = np.zeros(nb, bool)
    blk[int(np.ceil(p0)) + 2: min(nb, int(p1))] = True
    w, lk, me = (s[k].astype(float) for k in ("wheel_speed", "licks", "motion_energy"))
    mov = (np.nan_to_num(w) > .05) | (np.nan_to_num(lk) > 0) | (np.nan_to_num(me) > np.nanpercentile(me, 60))
    still = ~(mov | np.r_[False, mov[:-1]] | np.r_[mov[1:], False]) & (m34.run_lengths(mov) <= 30) & np.isfinite(me)
    grey = np.zeros(nb, bool)
    g0, g1 = s["spont"]
    if np.isfinite(g0) and np.isfinite(g1):
        grey[int(np.ceil(g0)) + 2: min(nb, int(g1))] = True
    return blk & still, blk & mov & np.isfinite(me), blk & still & grey


def orth(A, k):
    U, _, _ = torch.linalg.svd(A, full_matrices=False)
    return U[:, :k]


def align(U, X, k):
    """Alignment index and variance share of the columns of X (centred) in subspace U."""
    proj = (U.T @ X).pow(2).sum()
    sv = torch.linalg.svdvals(X)
    return float(proj / sv[:k].pow(2).sum()), float(proj / sv.pow(2).sum())


def session(s, p0, p1, gen):
    Xb = s["Xbin"].astype(np.float32)
    fin = np.isfinite(Xb)
    S, M, G = states(s, p0, p1)
    S &= fin[:20].all(0)
    M &= fin[:20].all(0)
    G &= fin[:20].all(0)
    tm = s["mask"].astype(bool) & np.isfinite(s["feedbackType"])
    side = np.where(np.isfinite(s["contrastLeft"]), 0, 1)
    Xw = s["Xwin"][:, tm]  # ROI x trials x windows
    good = (np.isfinite(Xb[:, S | M]).all(1) & np.isfinite(Xw).all((1, 2)))
    mu = np.nanmean(np.where(fin, Xb, np.nan), 1)
    sd = np.nanstd(np.where(fin, Xb, np.nan), 1)
    good &= sd > 0
    if S.sum() < 40 or good.sum() < 50:
        return None
    Z = lambda A: (A - mu[good, None]) / sd[good, None]
    Xs = T(Z(Xb[good][:, S]))
    Xm = T(Z(Xb[good][:, M]))
    Xg = T(Z(Xb[good][:, G])) if G.sum() >= 30 else None
    D = Xb[good][:, fin[good].all(0)]
    D = T(Z(D))
    W = (Xw[good] - mu[good, None, None]) / sd[good, None, None]
    wn = list(s["win_names"])
    fb, sd_ = s["feedbackType"][tm], side[tm]
    conds = [(a, b) for a in (0, 1) for b in (-1, 1)]

    def patterns(trials):
        P = {}
        for w in WINS:
            ev = W[:, :, wn.index(w)] - W[:, :, wn.index("pre")]
            cols = []
            for a, b in conds:
                t = trials & (sd_ == a) & (fb == b)
                if t.sum() >= MIN_TR:
                    cols.append(ev[:, t].mean(1))
            if cols:
                P[w] = np.stack(cols, 1)
        return P

    allt = np.ones(tm.sum(), bool)
    P = patterns(allt)
    if not P:
        return None
    Pall = T(np.concatenate([P[w] for w in WINS if w in P], 1))
    odd = np.arange(tm.sum()) % 2 == 1
    Po, Pe = patterns(odd), patterns(~odd)
    Po = T(np.concatenate([Po[w] for w in WINS if w in Po], 1))
    Pe = T(np.concatenate([Pe[w] for w in WINS if w in Pe], 1))
    n = int(good.sum())
    rows = []
    for glob in (False, True):
        g = torch.ones(n, 1, device=DEV) / n ** .5
        R = (lambda A: A - g @ (g.T @ A)) if glob else (lambda A: A)
        k = min(K, Pall.shape[1])
        U = orth(R(Pall), k)
        Uo, Ue = orth(R(Po), k), orth(R(Pe), k)
        half_overlap = float((Uo.T @ Ue).pow(2).sum() / k)
        rest = {"still": Xs, "moving": Xm}
        if Xg is not None:
            rest["grey_still"] = Xg
        rest = {key: R(X - X.mean(1, keepdim=True)) for key, X in rest.items()}
        # covariance-matched random subspaces from all recording bins
        Dc = R(D - D.mean(1, keepdim=True))
        Ud, Sd, _ = torch.linalg.svd(Dc, full_matrices=False)
        Ud, Sd = Ud[:, :500], Sd[:500]
        nulls_cov = [orth(Ud @ (Sd[:, None] * torch.randn(len(Sd), k, device=DEV, generator=gen)), k)
                     for _ in range(NCOV)]
        nulls_perm = [orth(R(Pall[torch.randperm(n, device=DEV, generator=gen)]), k) for _ in range(NPERM)]
        for key, X in rest.items():
            a, v = align(U, X, k)
            ap = np.array([align(Un, X, k)[0] for Un in nulls_perm])
            ac = np.array([align(Un, X, k)[0] for Un in nulls_cov])
            row = dict(eid=str(s["eid"]), rest=key, no_global=glob, k=k, n_rois=n, n_bins=X.shape[1],
                       n_patterns=Pall.shape[1], A=a, V=v, A_perm=ap.mean(), A_perm95=np.quantile(ap, .95),
                       A_cov=ac.mean(), A_cov95=np.quantile(ac, .95), half_overlap=half_overlap,
                       half_overlap_chance=k / n)
            for w in WINS:
                if w in P and P[w].shape[1] >= 2:
                    Uw = orth(R(T(P[w])), 2)
                    row[f"A_{w}"] = align(Uw, X, 2)[0]
                    row[f"A_{w}_perm"] = np.mean([align(orth(R(T(P[w])[
                        torch.randperm(n, device=DEV, generator=gen)]), 2), X, 2)[0] for _ in range(50)])
            rows.append(row)
    return rows


if __name__ == "__main__":
    SE = pd.read_csv(MESO / "output" / "34_still_sessions.csv").dropna(subset=["obs_still"])
    qc = pd.read_json(MESO / "output" / "35_meso_still_stats.json", typ="series")["qc_excluded"]
    SE = SE[~SE.eid.isin(qc)]
    subj = pd.read_csv(MESO / "output" / "session_subjects.csv").drop_duplicates("session").set_index("session").subject
    PP = m34.passive_periods(SE.eid.tolist())
    gen = torch.Generator(device=DEV).manual_seed(SEED)
    load = lambda e: dict(np.load(MESO / "output" / "sessions" / f"{e}.npz", allow_pickle=True))
    rows = []
    with ThreadPoolExecutor(3) as ex:
        for i, (eid, s) in enumerate(zip(SE.eid, ex.map(load, SE.eid))):
            r = session(s, *PP.loc[eid, ["t0", "t1"]], gen)
            if r:
                rows += r
                x = [q for q in r if q["rest"] == "still"]
                print(i, eid[:8], " ".join(f"A={q['A']:.3f} perm={q['A_perm']:.3f} cov={q['A_cov']:.3f}" for q in x),
                      flush=True)
    R = pd.DataFrame(rows)
    R["subject"] = R.eid.map(subj)
    R.to_csv(OUT / "10_meso_repertoire.csv", index=False)
    print(len(R.eid.unique()), "sessions,", R.subject.nunique(), "mice")
