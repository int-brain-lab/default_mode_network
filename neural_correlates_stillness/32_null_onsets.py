"""IBL counterparts of the Allen follow-up analyses (comprehensive report, Allen Neuropixels chapter).

1. Circular-shift null for the still-/movement-preferring labels. Per session, the grey-screen 50 ms counts of all
   kept neurons (20_neurons.pqt) are rotated together by one random offset (>= 10 s from either end) against the fixed
   behavioural labels, and the epoch test of 10_neurons.py is recomputed (Welch, p < 0.01, still epochs all 10 bins
   awake-still, moving epochs >= 5 of 10 bins moving, first epoch of each kind per 2 s). 200 rotations per session,
   seed 20261030, as in the Allen follow-up. Also with still epochs subsampled to 24 (the Allen median), which shows
   how the false-positive rate grows when one state has few epochs.
2. Pre-movement rise by onset rule: population rate (mean over kept neurons) from -2 to +1 s around onsets preceded
   by >= 1 s or >= 3 s of awake stillness, for any-movement onsets (report definition) and wheel onsets (Allen-like,
   wheel-only states of 30_allen_matched.py). Rise = mean fractional change over -0.5..-0.05 s against -2..-1 s;
   sessions need >= 3 onsets (Allen follow-up rule).
Outputs: output/32_null_sessions.csv, output/32_premove_sessions.csv, output/32_premove_curves.npz.

Usage: python 32_null_onsets.py [n_workers]
"""
import importlib
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

from common_still import BIN, MAX_STILL_S, OUT, SUBJ, load_session, probes, run_lengths, states

am = importlib.import_module("30_allen_matched")
EP, NSHIFT, SEED, K0 = 10, 200, 20261030, 200      # 0.5 s epochs; 200 rotations; >= 10 s offset
WIN = np.arange(-40, 21)                           # -2..+1 s
T = WIN * BIN
PRE_RULES = {"1s": 20, "3s": 60}
KEEP = set(pd.read_parquet(OUT / "20_neurons.pqt", columns=["uuid"]).uuid)


def labels(cs, nb, es, em, o=0):
    """+1 still-preferring, -1 movement-preferring, 0 neither; epoch sums from the doubled cumsum, counts rotated by o."""
    a, b = (es - o) % nb, (em - o) % nb
    RS, RM = cs[:, a + EP] - cs[:, a], cs[:, b + EP] - cs[:, b]
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.nan_to_num(ttest_ind(RS, RM, axis=1, equal_var=False).pvalue, nan=1.)
    d = RS.mean(1) - RM.mean(1)
    return np.where(p < .01, np.sign(d), 0)


def onsets(mov, awake, pre):
    on = np.flatnonzero(mov[1:] & ~mov[:-1]) + 1
    on = on[(on - pre - 1 >= 0) & (on + WIN[0] >= 0) & (on + WIN[-1] < len(mov))]
    return np.array([i for i in on if awake[i - pre:i - 1].all()], int)


def session(eid):
    s = load_session(eid)
    st = states(s)
    nb, grey = st["nb"], st["grey"]
    Xs = []
    for f in probes(eid):
        p = np.load(f, allow_pickle=True)
        k = np.isin(p["uuids"], list(KEEP))
        if k.any():
            Xs.append(p["counts"][k, :nb].astype(np.float32))
    if not Xs or not grey.any():
        return None
    X = np.concatenate(Xs)
    S, M = grey & st["awake"], grey & st["mov"]
    ga = np.flatnonzero(grey)
    g0, g1 = ga[0], ga[-1] + 1
    e0 = np.arange(g0, g1 - EP + 1, EP)
    iS = am.spaced(e0[np.array([S[a:a + EP].all() for a in e0])]) - g0
    iM = am.spaced(e0[np.array([M[a:a + EP].sum() >= EP // 2 for a in e0])]) - g0
    out = None
    if len(iS) >= 10 and len(iM) >= 10 and g1 - g0 > 2 * K0:
        Xg = X[:, g0:g1]
        n = Xg.shape[1]
        cs = np.concatenate([np.zeros((len(X), 1), np.float64), np.cumsum(np.concatenate([Xg, Xg], 1), 1, dtype=np.float64)], 1)
        rng = np.random.default_rng(SEED)
        i24 = np.sort(rng.choice(iS, min(24, len(iS)), replace=False))
        offs = rng.integers(K0, n - K0, NSHIFT)
        row = dict(eid=eid, subject=SUBJ.get(eid, ""), n_neurons=len(X), n_ep_still=len(iS), n_ep_move=len(iM))
        for tag, es in (("", iS), ("_s24", i24)):
            obs = labels(cs, n, es, iM)
            null = np.array([labels(cs, n, es, iM, o) for o in offs])
            fs, fm = (null == 1).mean(1), (null == -1).mean(1)
            row.update({f"obs_still{tag}": (obs == 1).mean(), f"obs_move{tag}": (obs == -1).mean(),
                        f"null_still{tag}": fs.mean(), f"null_move{tag}": fm.mean(),
                        f"null_still_p95{tag}": np.percentile(fs, 95), f"null_move_p95{tag}": np.percentile(fm, 95),
                        f"p_still{tag}": (1 + (fs >= (obs == 1).mean()).sum()) / (1 + NSHIFT),
                        f"p_move{tag}": (1 + (fm >= (obs == -1).mean()).sum()) / (1 + NSHIFT)})
        out = row
    # onsets: any movement (report states) and wheel only (30_allen_matched states)
    w = s["wheel_speed"].astype(float)
    mw = st["wheel"]
    still_w = ~(mw | np.r_[False, mw[:-1]] | np.r_[mw[1:], False]) & np.isfinite(w)
    gate = np.ones(nb, bool)
    pu, task = st["pupil"], st["task"]
    if st["pupil_ok"] and np.isfinite(pu[task]).sum() > 100:
        gate = ~(pu < np.nanpercentile(pu[task], 10))
    Sw = grey & still_w & (run_lengths(mw) * BIN <= MAX_STILL_S) & gate
    Mw = grey & mw & np.isfinite(w)
    curves, pm = {}, []
    for kind, (mov, aw) in {"any": (M, S), "wheel": (Mw, Sw)}.items():
        if kind == "wheel" and np.isfinite(w[grey]).mean() < .5:
            continue
        for rule, pre in PRE_RULES.items():
            on = onsets(mov, aw, pre)
            if len(on) < 3:
                continue
            P = X[:, on[:, None] + WIN].mean((0, 1)) / BIN
            base = P[(T >= -2) & (T < -1)].mean()
            if base <= 0:
                continue
            c = P / base - 1
            curves[(kind, rule)] = c
            pm.append(dict(eid=eid, subject=SUBJ.get(eid, ""), kind=kind, rule=rule, n_onsets=len(on),
                           rise=c[(T >= -.5) & (T < -.05)].mean(), early=c[(T >= -1) & (T < -.5)].mean()))
    return out, pm, curves


if __name__ == "__main__":
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    eids = sorted(pd.read_parquet(OUT / "20_neurons.pqt", columns=["eid"]).eid.unique())
    rows, pms, C = [], [], []
    with ProcessPoolExecutor(nw) as ex:
        for i, r in enumerate(ex.map(session, eids)):
            if r is None:
                continue
            if r[0] is not None:
                rows.append(r[0])
            pms += r[1]
            C += [(r[1][0]["eid"], k, v) for k, v in r[2].items()] if r[1] else []
            if i % 50 == 0:
                print(i, len(rows), flush=True)
    pd.DataFrame(rows).to_csv(OUT / "32_null_sessions.csv", index=False)
    pd.DataFrame(pms).to_csv(OUT / "32_premove_sessions.csv", index=False)
    np.savez_compressed(OUT / "32_premove_curves.npz", t=T, eid=np.array([c[0] for c in C]),
                        kind=np.array([c[1][0] for c in C]), rule=np.array([c[1][1] for c in C]),
                        curve=np.array([c[2] for c in C]))
    print(len(rows), "null sessions,", len(pms), "onset rows")
