"""Reproducibility of stillness preference within mesoscope recordings (comprehensive report, section 12).

The mesoscope counterpart of the within-recording tests of section 10.3 (neural_correlates_stillness/10_neurons.py,
20_stats.py). Same sessions, block, states and ROI inclusion as 34_meso_still.py. Per ROI, the stillness contrast
is computed separately in two parts of the passive block (grey screen + receptive-field mapping):
  halves       first against second half of the block (split at the middle bin)
  chunks       alternating 30 s chunks (even against odd), which spreads slow drift over both parts
Metrics per part: ratio = log2((still + e)/(moving + e)), e = 2% of the ROI's block mean (as section 12), and
SI = (still - moving)/(still + moving) (as section 10.3; deconvolved activity is non-negative).
Null per session: the activity of the second part is rotated against its fixed state labels (NROT offsets
>= 15 s within the part), which keeps ROI properties and breaks only the alignment with behaviour.
Parts need >= 10 still and >= 10 moving bins. Section 10.3's other two comparisons are not possible here: the task
has no still bins outside stimulus-to-feedback windows (median 0 per session), and the wheel is not recorded in
the passive block.
Outputs: output/36_repro_rois.pqt, output/36_repro_sessions.csv.

Usage: python 36_meso_still_repro.py [n_workers]
"""
import importlib.util
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT = Path(__file__).resolve().parent / "output"
_spec = importlib.util.spec_from_file_location("m34", Path(__file__).resolve().parent / "34_meso_still.py")
m34 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m34)
MIN_BINS, CHUNK, NROT, K0, SEED = 10, 30, 20, 15, 20261104


def contrast(X, S, M, e):
    rs, rm = X[:, S].mean(1), X[:, M].mean(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        si = (rs - rm) / (rs + rm)
    return np.log2((rs + e) / (rm + e)), si


def session(args):
    from iblatlas.regions import BrainRegions
    f, (p0, p1) = args
    s = np.load(f, allow_pickle=True)
    eid = str(s["eid"])
    nb = s["Xbin"].shape[1]
    info = dict(eid=eid)
    if not np.isfinite(p1 - p0) or p1 - p0 < 60:
        return info, None
    blk = np.zeros(nb, bool)
    blk[int(np.ceil(p0)) + 2: min(nb, int(p1))] = True
    blk &= np.isfinite(s["Xbin"][:20].astype(np.float32)).all(0)
    w, lk, me = (s[k].astype(float) for k in ("wheel_speed", "licks", "motion_energy"))
    if not blk.any() or np.isfinite(me[blk]).mean() < .5:
        return info, None
    mov = (np.nan_to_num(w) > .05) | (np.nan_to_num(lk) > 0) | (np.nan_to_num(me) > np.nanpercentile(me, 60))
    still = ~(mov | np.r_[False, mov[:-1]] | np.r_[mov[1:], False]) & (m34.run_lengths(mov) <= 30) & np.isfinite(me)
    g0, g1 = np.flatnonzero(blk)[[0, -1]]
    sl = slice(g0, g1 + 1)
    S, M = (blk & still)[sl], (blk & mov & np.isfinite(me))[sl]
    X = s["Xbin"][:, sl].astype(np.float32)
    n = X.shape[1]
    iSM = np.flatnonzero(S | M)
    good = np.isfinite(X[:, iSM]).all(1) & (X[:, iSM].std(1) > 0)
    # same inclusion as 34_meso_still.py
    if len(m34.spaced(np.flatnonzero(S))) < 20 or len(m34.spaced(np.flatnonzero(M))) < 20 or good.sum() < 50:
        return info, None
    X = np.nan_to_num(X[good])
    e = .02 * X[:, iSM].mean(1)
    t = np.arange(n)
    parts = {"halves": t < n // 2, "chunks": (t // CHUNK) % 2 == 0}
    br = BrainRegions()
    aid = s["atlas_id"][good]
    R = pd.DataFrame({"eid": eid, "beryl": br.id2acronym(aid, mapping="Beryl")})
    rng = np.random.default_rng(SEED)
    for name, a in parts.items():
        nS = [(S & a).sum(), (S & ~a).sum()]
        nM = [(M & a).sum(), (M & ~a).sum()]
        info.update({f"{name}_still": min(nS), f"{name}_move": min(nM)})
        if min(nS + nM) < MIN_BINS:
            continue
        r1, s1 = contrast(X, S & a, M & a, e)
        r2, s2 = contrast(X, S & ~a, M & ~a, e)
        R[f"ratio_{name}_1"], R[f"ratio_{name}_2"] = r1.astype(np.float32), r2.astype(np.float32)
        R[f"si_{name}_1"], R[f"si_{name}_2"] = s1.astype(np.float32), s2.astype(np.float32)
        info[f"r_ratio_{name}"] = spearmanr(r1, r2, nan_policy="omit").statistic
        info[f"r_si_{name}"] = spearmanr(s1, s2, nan_policy="omit").statistic
        # null: rotate the activity of part 2 against its labels (bins of part 2 in time order)
        i2 = np.flatnonzero(~a)
        S2, M2 = S[i2], M[i2]
        X2 = X[:, i2]
        nr, ns = [], []
        for o in rng.integers(K0, len(i2) - K0, NROT):
            Xr = np.roll(X2, o, axis=1)
            rr, sr = contrast(Xr, S2, M2, e)
            nr.append(spearmanr(r1, rr, nan_policy="omit").statistic)
            ns.append(spearmanr(s1, sr, nan_policy="omit").statistic)
        info[f"null_ratio_{name}"], info[f"null_si_{name}"] = float(np.mean(nr)), float(np.mean(ns))
        info[f"null95_ratio_{name}"] = float(np.quantile(nr, .95))
    info["n_rois"] = int(good.sum())
    return info, R


if __name__ == "__main__":
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    subj = pd.read_csv(OUT / "session_subjects.csv").drop_duplicates("session").set_index("session").subject
    S34 = pd.read_csv(OUT / "34_still_sessions.csv").dropna(subset=["obs_still"])
    PP = m34.passive_periods(S34.eid.tolist())
    files = [OUT / "sessions" / f"{e}.npz" for e in S34.eid]
    infos, rois = [], []
    with ProcessPoolExecutor(nw) as ex:
        for i, (info, R) in enumerate(ex.map(session, [(f, tuple(PP.loc[f.stem, ["t0", "t1"]])) for f in files])):
            infos.append(info)
            if R is not None:
                rois.append(R)
            print(i, info.get("r_ratio_halves"), info.get("null_ratio_halves"), flush=True)
    S = pd.DataFrame(infos)
    S["subject"] = S.eid.map(subj)
    S.to_csv(OUT / "36_repro_sessions.csv", index=False)
    R = pd.concat(rois, ignore_index=True)
    R["subject"] = R.eid.map(subj)
    R.to_parquet(OUT / "36_repro_rois.pqt")
    print(len(rois), "sessions,", R.subject.nunique(), "mice,", len(R), "ROIs")
