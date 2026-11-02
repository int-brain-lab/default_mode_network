"""IBL stillness contrasts recomputed with the Allen replication's definitions (comprehensive report, section 13).

The Allen Neuropixels replication (comprehensive_report/allen) defines movement from running speed only, because
those datasets have no face video and no grey-screen licks. Here the IBL grey screen is re-labelled the same way,
with the wheel standing in for the running disc:
  moving_w     wheel speed > 0.05 rad/s (licks and face motion are allowed in still bins)
  awake_w      not wheel-moving (+-1 bin margin), <= 30 s since the last wheel movement, pupil gate as common_still
A coarse variant (suffix _w2) uses the Allen running threshold, 2 cm/s, converted to the IBL wheel (radius 3.1 cm):
  moving_w2    wheel speed > 0.645 rad/s, so slow wheel turns count as still, as slow running does in the Allen data
Per neuron (the 53,136 neurons of 20_neurons.pqt, grey screen):
  r_still_w, r_move_w   rates in awake_w and moving_w bins
  p_w                   signed Welch p (negative: higher in movement) of 0.5 s epochs entirely inside one
                        state, at most one epoch per 2 s (Allen rule)
  p_ibl_s{k}, p_w_s{k}  the IBL-definition epoch test (as 10_neurons.py) and p_w with still epochs randomly
                        subsampled to k = 12, 24, 48, 96 (Allen median 24; moving epochs all, IBL median 134 vs
                        Allen 125): the effect of short still time
Per session: population rate around onsets (first moving bin after >= 1 s of awake stillness, Allen rule), -2..+1 s,
for both definitions (wheel onsets; any-movement onsets), mean over neurons.
Sessions need wheel data on >= 50% of the grey screen and >= 10 still and >= 10 moving epochs (Allen inclusion).

Usage: python 30_allen_matched.py [n_workers]
"""
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

from common_still import BIN, MAX_STILL_S, OUT, load_session, probes, run_lengths, states

EP, SPACE = 10, 40
SUB = (12, 24, 48, 96)             # still epochs kept (Allen median 24)
W2 = 2.0 / 3.1                     # rad/s: 2 cm/s at the wheel rim
PRE, WIN = 20, np.arange(-40, 20)  # >= 1 s still before onset; -2..+1 s
NEUR = pd.read_parquet(OUT / "20_neurons.pqt", columns=["eid", "uuid"])
KEEP = set(NEUR.uuid)


def epochs(mask):
    """Starts of 10-bin runs fully inside mask, non-overlapping, at most one per 2 s window."""
    out, run, last = [], 0, -1
    for j, m in enumerate(mask):
        run = run + 1 if m else 0
        if run >= EP:
            s = j - EP + 1
            if (not out or s >= out[-1] + EP) and s // SPACE != last:
                out.append(s)
                last = s // SPACE
                run = 0
    return np.array(out, int)


def spaced(idx):
    if len(idx) == 0:
        return idx
    w = idx // SPACE
    return idx[np.r_[0, np.flatnonzero(np.diff(w)) + 1]]


def welch(X, a, b):
    if len(a) < 3 or len(b) < 3:
        return np.full(len(X), np.nan)
    RS = np.stack([X[:, s:s + EP].sum(1) for s in a], 1)
    RM = np.stack([X[:, s:s + EP].sum(1) for s in b], 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        pv = np.nan_to_num(ttest_ind(RS, RM, axis=1, equal_var=False).pvalue, nan=1.)
    return np.copysign(pv, RS.mean(1) - RM.mean(1))  # signed: negative = higher when moving


def rate(X, m):
    n = int(m.sum())
    return X[:, m].sum(1, dtype=np.float64) / (n * BIN) if n >= 20 else np.full(len(X), np.nan)


def onsets(mov, awake):
    on = np.flatnonzero(mov[1:] & ~mov[:-1]) + 1
    on = on[(on - PRE - 1 >= 0) & (on + WIN[0] >= 0) & (on + WIN[-1] < len(mov))]
    return np.array([i for i in on if awake[i - PRE:i - 1].all()], int)


def session(eid):
    s = load_session(eid)
    st = states(s)
    nb, grey = st["nb"], st["grey"]
    w = s["wheel_speed"].astype(float)
    if not grey.any() or np.isfinite(w[grey]).mean() < 0.5:
        return None
    mw = st["wheel"]
    still_w = ~(mw | np.r_[False, mw[:-1]] | np.r_[mw[1:], False]) & np.isfinite(w)
    awake_w = still_w & (run_lengths(mw) * BIN <= MAX_STILL_S)
    # pupil gate at the task 10th percentile; bins without pupil are not gated (Allen rule)
    gate = np.ones(nb, bool)
    pu, task = st["pupil"], st["task"]
    if st["pupil_ok"] and np.isfinite(pu[task]).sum() > 100:
        gate = ~(pu < np.nanpercentile(pu[task], 10))
    Sw, Mw = grey & awake_w & gate, grey & mw
    mw2 = np.isfinite(w) & (w > W2)
    still_w2 = ~(mw2 | np.r_[False, mw2[:-1]] | np.r_[mw2[1:], False]) & np.isfinite(w)
    Sw2, Mw2 = grey & still_w2 & (run_lengths(mw2) * BIN <= MAX_STILL_S) & gate, grey & mw2
    eS2, eM2 = epochs(Sw2), epochs(Mw2)
    eS, eM = epochs(Sw), epochs(Mw)
    if len(eS) < 10 or len(eM) < 10:
        return None
    # IBL-definition epochs as in 10_neurons.py
    S, M = grey & st["awake"], grey & st["mov"]
    ga = np.flatnonzero(grey)
    e0 = np.arange(ga[0], ga[-1] - EP + 1, EP)
    iS = spaced(e0[np.array([S[a:a + EP].all() for a in e0])])
    iM = spaced(e0[np.array([M[a:a + EP].sum() >= EP // 2 for a in e0])])
    rng = np.random.default_rng(int(eid[:8], 16))
    sub = lambda e, k: np.sort(rng.choice(e, min(k, len(e)), replace=False)) if len(e) else e  # noqa: E731
    subs = {k: (sub(iS, k), sub(eS, k)) for k in SUB}
    on_w = onsets(Mw, Sw)
    on_i = onsets(M, S)
    on_w2 = onsets(Mw2, Sw2)
    e24 = sub(eS2, 24)
    rows, P = [], {"w": [], "ibl": [], "w2": []}
    for f in probes(eid):
        p = np.load(f, allow_pickle=True)
        X = p["counts"][:, :nb]
        k = np.isin(p["uuids"], list(KEEP))
        if not k.any():
            continue
        X = X[k]
        rows.append(pd.DataFrame({"uuid": p["uuids"][k], "r_still_w": rate(X, Sw), "r_move_w": rate(X, Mw),
                                  "p_w": welch(X, eS, eM),
                                  **{f"p_ibl_s{k}": welch(X, a, iM) for k, (a, _) in subs.items()},
                                  **{f"p_w_s{k}": welch(X, b, eM) for k, (_, b) in subs.items()},
                                  "r_still_w2": rate(X, Sw2), "r_move_w2": rate(X, Mw2),
                                  "p_w2": welch(X, eS2, eM2), "p_w2_s24": welch(X, e24, eM2)}))
        for key, on in (("w", on_w), ("ibl", on_i), ("w2", on_w2)):
            if len(on) >= 5:
                P[key].append(np.stack([X[:, a + WIN] for a in on], 0).mean(0).sum(0) / BIN)  # summed over neurons
    if not rows:
        return None
    D = pd.concat(rows, ignore_index=True)
    D["eid"] = eid
    n = len(D)
    meta = dict(eid=eid, n_still_w=int(Sw.sum()), n_move_w=int(Mw.sum()), n_epS_w=len(eS), n_epM_w=len(eM),
                n_epS=len(iS), n_epM=len(iM), n_on_w=len(on_w), n_on_ibl=len(on_i), n_neurons=n,
                n_still_w2=int(Sw2.sum()), n_move_w2=int(Mw2.sum()), n_epS_w2=len(eS2), n_epM_w2=len(eM2),
                n_on_w2=len(on_w2),
                still_face_frac=float((Sw & (st["face"] | st["lick"])).sum() / max(Sw.sum(), 1)))
    curves = {k: (np.sum(v, 0) / n if v else None) for k, v in P.items()}
    return D, meta, curves


if __name__ == "__main__":
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    eids = sorted(NEUR.eid.unique())
    res = []
    with ProcessPoolExecutor(nw) as ex:
        for i, r in enumerate(ex.map(session, eids)):
            if r is not None:
                res.append(r)
            if i % 50 == 0:
                print(i, len(res), flush=True)
    D = pd.concat([r[0] for r in res], ignore_index=True)
    D.to_parquet(OUT / "30_allen_matched_neurons.pqt")
    pd.DataFrame([r[1] for r in res]).to_csv(OUT / "30_allen_matched_sessions.csv", index=False)
    C = {k: np.array([r[2][k] if r[2][k] is not None else np.full(len(WIN), np.nan) for r in res])
         for k in ("w", "ibl", "w2")}
    np.savez_compressed(OUT / "30_allen_matched_onsets.npz", eid=np.array([r[1]["eid"] for r in res]),
                        t=WIN * BIN, **C)
    print(len(D), D.eid.nunique())
