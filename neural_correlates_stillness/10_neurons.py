"""Per-neuron correlates of awake stillness (states as in common_still.py).

Per neuron, on the passive grey screen (first 1 s skipped) unless stated:
  r_still, r_move        rate in awake-still bins and in moving bins
  *_h1, *_h2             the same in the first and second half of the grey screen (cross-validation)
  p_epoch                Welch test of 0.5 s epochs: still epochs (all 10 bins awake-still) vs moving epochs
                         (>= 5 of 10 bins moving), at most one epoch of each kind per 2 s window, to reduce dense sampling; residual temporal dependence can remain
  r_iti_still, r_iti_move  the same contrast in the task's inter-trial gaps (second context)
  r_face, r_wheel        moving bins split into face/body-only (motion energy, no wheel, no lick) and wheel turns
  r_early, r_late        within awake-still bouts lasting >= 4 s: first 0.5 s vs 2-4 s into the bout;
                         p_build = Wilcoxon over bouts
  tsm_k / ttn_k          rate in awake-still bins by time since the last movement / to the next movement,
                         TPROF edges (s)
  r_pupil                within awake-still 1 s bins: partial correlation of the spike count with pupil diameter,
                         controlling for time (quadratic) and motion energy
  r_still_nogate         still bins without the anti-sleep gate; r_lowpup still (<= 30 s) with pupil below the gate;
                         r_long still for > 30 s
Movement-offset (-1..+4 s) and movement-onset (-3..+1 s) aligned rates are saved separately (10_peths.npz):
offsets = first bin after the last moving bin of awake-still bouts lasting >= 4 s; onsets = first moving bin after
>= 3 s of awake stillness.

Usage: python 10_neurons.py [n_workers]
"""
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, wilcoxon

from common_still import BIN, OUT, SUBJ, beryl, bouts, cosmos, done_eids, load_session, probes, states

EP = 10                              # 0.5 s epochs
SPACE = 40                           # one epoch of each kind per 2 s window
BOUT_MIN, EARLY, LATE = 80, (0, 10), (40, 80)
TPROF = np.array([0.1, 0.5, 1, 2, 4, 8, 30])
OFF_WIN, ON_WIN = (-20, 80), (-60, 20)
MIN_S, MIN_M = 600, 200              # 30 s still, 10 s moving on the grey screen


def rate(X, m):
    n = int(m.sum())
    return X[:, m].sum(1, dtype=np.float64) / (n * BIN) if n >= 20 else np.full(len(X), np.nan)


def spaced(idx, space=SPACE):
    """Keep the first epoch start of each `space`-bin window."""
    if len(idx) == 0:
        return idx
    w = idx // space
    return idx[np.r_[0, np.flatnonzero(np.diff(w)) + 1]]


def movement_events(st, S):
    """True movement offsets and onsets, with complete grey-screen windows."""
    nb = st["nb"]
    grey = st["grey"]
    ga = np.flatnonzero(grey)
    b0, b1 = bouts(S)
    long_ = (b1 - b0) >= BOUT_MIN
    # Pupil-gate crossings and grey boundaries are not movement offsets.
    true_offset = (b0 >= 2) & (st["tsm"][b0] == 2)
    lb0 = b0[long_ & true_offset]
    off = lb0 - 1  # first bin after the last moving bin (bin b0-1 is the margin bin)
    off = off[(off + OFF_WIN[0] >= ga[0]) & (off + OFF_WIN[1] <= ga[-1] + 1)]
    # onsets: moving bin right after an awake-still bout (bout ends at the margin bin) of >= 3 s
    on = b1[(b1 - b0) >= 60] + 1
    on = on[(on + ON_WIN[0] >= ga[0]) & (on + ON_WIN[1] <= ga[-1] + 1)]
    on = on[st["mov"][on] & grey[on]]
    return lb0, off, on


def session(eid):
    s = load_session(eid)
    st = states(s)
    nb, grey = st["nb"], st["grey"]
    S, M = grey & st["awake"], grey & st["mov"]
    if S.sum() < MIN_S or M.sum() < MIN_M:
        return None
    ga = np.flatnonzero(grey)
    mid = (ga[0] + ga[-1]) // 2
    h1 = np.zeros(nb, bool)
    h1[:mid] = True
    # epochs
    e0 = np.arange(ga[0], ga[-1] - EP + 1, EP)
    sfrac = np.array([S[a:a + EP].all() for a in e0])
    mfrac = np.array([M[a:a + EP].sum() >= EP // 2 for a in e0])
    eS, eM = spaced(e0[sfrac]), spaced(e0[mfrac])
    lb0, off, on = movement_events(st, S)
    # time-since / time-to profiles
    tb = np.digitize(st["tsm"] * BIN, TPROF) - 1
    nb_ = np.digitize(st["ttn"] * BIN, TPROF) - 1
    # 1 s bins fully awake-still for the pupil correlation
    K = 20
    c0 = np.arange(ga[0], ga[-1] - K + 1, K)
    c0 = c0[np.array([S[a:a + K].all() for a in c0])] if len(c0) else c0
    pu, me = st["pupil"], st["me"]
    pup_ok = st["pupil_ok"] and len(c0) >= 60
    if pup_ok:
        P = np.array([np.nanmean(pu[a:a + K]) for a in c0])
        Me = np.array([np.nanmean(me[a:a + K]) for a in c0]) if st["me_ok"] else np.zeros(len(c0))
        tt = (c0 - c0.mean()) / (c0.std() + 1e-9)
        D = np.c_[np.ones(len(c0)), tt, tt ** 2, np.nan_to_num((Me - np.nanmean(Me)) / (np.nanstd(Me) + 1e-9))]
        okp = np.isfinite(P)
        pup_ok = okp.sum() >= 60
    nogate = grey & st["still"]
    lowpup = grey & st["still"] & ~st["awake"] & (st["tsm"] * BIN <= 30)
    longs = grey & st["still"] & (st["tsm"] * BIN > 30)
    rows, offp, onp = [], [], []
    for f in probes(eid):
        p = np.load(f, allow_pickle=True)
        X = p["counts"][:, :nb]
        n = len(X)
        out = {"r_still": rate(X, S), "r_move": rate(X, M),
               "r_still_h1": rate(X, S & h1), "r_move_h1": rate(X, M & h1),
               "r_still_h2": rate(X, S & ~h1), "r_move_h2": rate(X, M & ~h1),
               "r_iti_still": rate(X, st["iti"] & st["awake"]), "r_iti_move": rate(X, st["iti"] & st["mov"]),
               "r_face": rate(X, M & st["face"] & ~st["wheel"] & ~st["lick"]), "r_wheel": rate(X, M & st["wheel"]),
               "r_still_nogate": rate(X, nogate), "r_lowpup": rate(X, lowpup), "r_long": rate(X, longs)}
        for k in range(len(TPROF) - 1):
            out[f"tsm_{k}"] = rate(X, S & (tb == k))
            out[f"ttn_{k}"] = rate(X, S & (nb_ == k))
        # epoch test
        if len(eS) >= 20 and len(eM) >= 20:
            RS = np.stack([X[:, a:a + EP].sum(1) for a in eS], 1) / (EP * BIN)
            RM = np.stack([X[:, a:a + EP].sum(1) for a in eM], 1) / (EP * BIN)
            with np.errstate(invalid="ignore", divide="ignore"):
                out["p_epoch"] = np.nan_to_num(ttest_ind(RS, RM, axis=1, equal_var=False).pvalue, nan=1.)
        else:
            out["p_epoch"] = np.full(n, np.nan)
        # build-up within long bouts
        if len(lb0) >= 8:
            E = np.stack([X[:, a + EARLY[0]:a + EARLY[1]].sum(1) / ((EARLY[1] - EARLY[0]) * BIN) for a in lb0], 1)
            L = np.stack([X[:, a + LATE[0]:a + LATE[1]].sum(1) / ((LATE[1] - LATE[0]) * BIN) for a in lb0], 1)
            out["r_early"], out["r_late"] = E.mean(1), L.mean(1)
            pb = np.ones(n)
            for i in range(n):
                d = L[i] - E[i]
                if np.any(d != 0):
                    pb[i] = wilcoxon(d).pvalue
            out["p_build"] = pb
        else:
            out["r_early"] = out["r_late"] = out["p_build"] = np.full(n, np.nan)
        # pupil within stillness
        if pup_ok:
            C = np.stack([X[:, a:a + K].sum(1) for a in c0], 1).astype(float)[:, okp]
            Dd = D[okp]
            Cr = C.T - Dd @ np.linalg.lstsq(Dd, C.T, rcond=None)[0]
            Pr = P[okp] - Dd @ np.linalg.lstsq(Dd, P[okp], rcond=None)[0]
            with np.errstate(invalid="ignore", divide="ignore"):
                out["r_pupil"] = (Cr * Pr[:, None]).sum(0) / np.sqrt((Cr ** 2).sum(0) * (Pr ** 2).sum())
        else:
            out["r_pupil"] = np.full(n, np.nan)
        # aligned rates
        offp.append(np.stack([X[:, a + OFF_WIN[0]:a + OFF_WIN[1]] for a in off], 0).mean(0, dtype=np.float32) / BIN
                    if len(off) >= 5 else np.full((n, OFF_WIN[1] - OFF_WIN[0]), np.nan, np.float32))
        onp.append(np.stack([X[:, a + ON_WIN[0]:a + ON_WIN[1]] for a in on], 0).mean(0, dtype=np.float32) / BIN
                   if len(on) >= 5 else np.full((n, ON_WIN[1] - ON_WIN[0]), np.nan, np.float32))
        rows.append(pd.DataFrame({"eid": eid, "probe": str(p["probe"]), "uuid": p["uuids"], "atlas_id": p["atlas_id"],
                                  "x": p["xyz"][:, 0], "y": p["xyz"][:, 1], "z": p["xyz"][:, 2],
                                  "rate_all": X.sum(1, dtype=np.float64) / (nb * BIN), **out}))
    D_ = pd.concat(rows, ignore_index=True)
    meta = dict(eid=eid, n_still=int(S.sum()), n_move=int(M.sum()), n_epS=len(eS), n_epM=len(eM),
                n_bouts4=int(len(lb0)), n_off=len(off), n_on=len(on), pupil_ok=st["pupil_ok"], me_ok=st["me_ok"],
                n_pupil_bins=int(len(c0)), n_nogate=int(nogate.sum()), n_lowpup=int(lowpup.sum()),
                n_long=int(longs.sum()), n_grey=int(grey.sum()), n_iti_still=int((st["iti"] & st["awake"]).sum()),
                n_iti_move=int((st["iti"] & st["mov"]).sum()))
    for k in ("n_still", "n_move", "pupil_ok", "me_ok", "n_off", "n_on", "n_bouts4"):
        D_[k] = meta[k]
    return D_, np.concatenate(offp), np.concatenate(onp), meta


if __name__ == "__main__":
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    eids = done_eids()
    res = []
    with ProcessPoolExecutor(nw) as ex:
        for i, r in enumerate(ex.map(session, eids)):
            if r is not None:
                res.append(r)
            if i % 50 == 0:
                print(i, len(res), flush=True)
    N = pd.concat([r[0] for r in res], ignore_index=True)
    OFF = np.concatenate([r[1] for r in res])
    ON = np.concatenate([r[2] for r in res])
    N["region"], N["cosmos"] = beryl(N.atlas_id), cosmos(N.atlas_id)
    N["subject"] = N.eid.map(SUBJ)
    keep = (~N.region.isin(["root", "void"]) & (N.rate_all >= 0.5)).values
    N, OFF, ON = N[keep].reset_index(drop=True), OFF[keep], ON[keep]
    N.to_parquet(OUT / "10_neurons.pqt")
    np.savez_compressed(OUT / "10_peths.npz", off=OFF.astype(np.float32), on=ON.astype(np.float32),
                        uuid=N.uuid.values.astype(str), t_off=np.arange(*OFF_WIN) * BIN, t_on=np.arange(*ON_WIN) * BIN)
    pd.DataFrame([r[3] for r in res]).to_csv(OUT / "10_sessions.csv", index=False)
    print(len(N), N.eid.nunique(), N.subject.nunique())
