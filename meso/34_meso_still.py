"""Correlates of stillness in the IBL mesoscope (comprehensive report, IBL mesoscope chapter).

The mesoscope counterpart of neural_correlates_stillness/10_neurons.py. Most canonical sessions have only a 60 s
grey screen, and during the blk the mice turn the wheel almost continuously (median 1 s wheel speed 0.2-0.3 rad/s,
inter-trial gaps 1.6 s), so the contrast uses the passive block before the blk replay: grey screen plus
receptive-field mapping (sparse noise, not locked to movement), from _ibl_passivePeriods (cached in
output/34_passive_periods.csv). The replay is excluded because its tones and valve clicks evoke face movements.
  moving bin   wheel speed > 0.05 rad/s (where recorded; mostly not in the passive block), a lick, or motion energy
               above the session's 60th percentile (whole recording)
  still bin    no movement in the bin or the neighbouring bins (+-1 s), <= 30 s since the last movement
Per ROI (deconvolved activity, 1 s bin means, finite and non-constant in the used bins):
  ratio        log2((still + e) / (moving + e)), e = 2% of the ROI's mean (the activity scale is arbitrary per ROI)
  label        Welch test of still against moving bins, at most one bin of each kind per 2 s, p < 0.01; +1 still-,
               -1 movement-preferring
Circular-shift null: the activity is rotated against the fixed labels (100 offsets >= 60 s, seed 20261030).
Sessions need motion energy in the block, >= 20 spaced bins of each state and >= 50 ROIs.
Outputs: output/34_still_rois.pqt, output/34_still_sessions.csv.

Usage: python 34_meso_still.py [n_workers]
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

OUT = Path(__file__).resolve().parent / "output"
SPACE, NSHIFT, K0, SEED = 2, 100, 60, 20261030


def run_lengths(x):
    """Length of the current run of False (bins since the last True), per bin."""
    out, c = np.zeros(len(x), int), 0
    for i, v in enumerate(x):
        c = 0 if v else c + 1
        out[i] = c
    return out


def spaced(idx):
    if len(idx) == 0:
        return idx
    w = idx // SPACE
    return idx[np.r_[0, np.flatnonzero(np.diff(w)) + 1]]


def labels(X, a, b, n, o=0):
    A, B = X[:, (a - o) % n], X[:, (b - o) % n]
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.nan_to_num(ttest_ind(A, B, axis=1, equal_var=False).pvalue, nan=1.)
    return np.where(p < .01, np.sign(A.mean(1) - B.mean(1)), 0)


def passive_periods(eids):
    """Grey-screen start and RFM stop per session, from ONE; cached."""
    f = OUT / "34_passive_periods.csv"
    P = pd.read_csv(f) if f.exists() else pd.DataFrame(columns=["eid", "t0", "t1"])
    todo = sorted(set(eids) - set(P.eid))
    if todo:
        from one.api import ONE
        one, rows = ONE(), []
        for eid in todo:
            try:
                d = [x for x in one.list_datasets(eid) if x.endswith("_ibl_passivePeriods.intervalsTable.csv")]
                t = one.load_dataset(eid, d[0]).set_index("Unnamed: 0")
                rows.append(dict(eid=eid, t0=t.loc["start", "spontaneousActivity"], t1=t.loc["stop", "RFM"]))
            except Exception:  # no passive data
                rows.append(dict(eid=eid, t0=np.nan, t1=np.nan))
        P = pd.concat([P, pd.DataFrame(rows)], ignore_index=True)
        P.to_csv(f, index=False)
    return P.set_index("eid")


def session(args):
    from iblatlas.regions import BrainRegions
    f, (p0, p1) = args
    s = np.load(f, allow_pickle=True)
    eid = str(s["eid"])
    nb = s["Xbin"].shape[1]
    info = dict(eid=eid, block_s=float(p1 - p0) if np.isfinite(p1 - p0) else 0.)
    if not np.isfinite(p1 - p0) or p1 - p0 < 60:
        return info, None
    blk = np.zeros(nb, bool)  # grey screen + RFM
    blk[int(np.ceil(p0)) + 2: min(nb, int(p1))] = True
    blk &= np.isfinite(s["Xbin"][:20].astype(np.float32)).all(0)
    w, lk, me = (s[k].astype(float) for k in ("wheel_speed", "licks", "motion_energy"))
    info["me_cov"] = float(np.isfinite(me[blk]).mean()) if blk.any() else 0.
    if info["me_cov"] < .5:
        return info, None
    mov = (np.nan_to_num(w) > .05) | (np.nan_to_num(lk) > 0) | (np.nan_to_num(me) > np.nanpercentile(me, 60))
    still = ~(mov | np.r_[False, mov[:-1]] | np.r_[mov[1:], False]) & (run_lengths(mov) <= 30) & np.isfinite(me)
    S, M = blk & still, blk & mov & np.isfinite(me)
    g0, g1 = np.flatnonzero(blk)[[0, -1]]
    X = s["Xbin"][:, g0:g1 + 1].astype(np.float32)
    n = X.shape[1]
    iS, iM = np.flatnonzero(S[g0:g1 + 1]), np.flatnonzero(M[g0:g1 + 1])
    aS, aM = spaced(iS), spaced(iM)
    good = np.isfinite(X[:, np.r_[iS, iM]]).all(1) & (X[:, np.r_[iS, iM]].std(1) > 0)
    info.update(n_still=len(iS), n_move=len(iM), n_still_sp=len(aS), n_move_sp=len(aM), n_rois=int(good.sum()),
                wheel_cov=float(np.isfinite(w[blk]).mean()))
    if len(aS) < 20 or len(aM) < 20 or good.sum() < 50:
        return info, None
    X = np.nan_to_num(X[good])
    rs, rm = X[:, iS].mean(1), X[:, iM].mean(1)
    e = .02 * X[:, np.r_[iS, iM]].mean(1)
    lab = labels(X, aS, aM, n)
    rng = np.random.default_rng(SEED)
    null = np.array([labels(X, aS, aM, n, o) for o in rng.integers(K0, n - K0, NSHIFT)])
    fs, fm = (null == 1).mean(1), (null == -1).mean(1)
    info.update(obs_still=float((lab == 1).mean()), obs_move=float((lab == -1).mean()),
                null_still=float(fs.mean()), null_move=float(fm.mean()),
                p_still=float((1 + (fs >= (lab == 1).mean()).sum()) / (1 + NSHIFT)),
                p_move=float((1 + (fm >= (lab == -1).mean()).sum()) / (1 + NSHIFT)))
    br = BrainRegions()
    aid = s["atlas_id"][good]
    R = pd.DataFrame({"eid": eid, "beryl": br.id2acronym(aid, mapping="Beryl"),
                      "cosmos": br.id2acronym(aid, mapping="Cosmos"),
                      "ratio": np.log2((rs + e) / (rm + e)).astype(np.float32), "label": lab.astype(np.int8)})
    return info, R


if __name__ == "__main__":
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    subj = pd.read_csv(OUT / "session_subjects.csv").drop_duplicates("session").set_index("session").subject
    files = sorted((OUT / "sessions").glob("*.npz"))
    PP = passive_periods([f.stem for f in files])
    infos, rois = [], []
    with ProcessPoolExecutor(nw) as ex:
        for i, (info, R) in enumerate(ex.map(session, [(f, tuple(PP.loc[f.stem, ["t0", "t1"]])) for f in files])):
            infos.append(info)
            if R is not None:
                rois.append(R)
            if i % 20 == 0:
                print(i, len(rois), flush=True)
    S = pd.DataFrame(infos)
    S["subject"] = S.eid.map(subj)
    S.to_csv(OUT / "34_still_sessions.csv", index=False)
    R = pd.concat(rois, ignore_index=True)
    R["subject"] = R.eid.map(subj)
    R.to_parquet(OUT / "34_still_rois.pqt")
    print(len(rois), "sessions,", R.subject.nunique(), "mice,", len(R), "ROIs")
