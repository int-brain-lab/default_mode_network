"""Population structure within a region during awake stillness vs movement (grey screen).

Per region and probe (>= MIN_NEU neurons >= 0.5 Hz), from 100 ms bins (two 50 ms bins of one state):
  states   still = both bins awake-still; move = both bins moving; still_early / still_late = awake-still bins
           < 1 s / >= 2 s after the last movement
  samples  equalised: the same random number of 100 ms bins per state (the smallest state, >= MIN_SAMP)
  corr     mean pairwise Pearson correlation of spike counts
  pr       participation ratio of the correlation matrix, divided by the number of neurons
  burst    fraction of bins with a population count > 3 SD above its mean within the state (synchronous events)
  rate     mean rate of the population
  matched  the same after binomial thinning of each neuron's counts in its higher-rate state down to its rate in
           the other state (equal rates per neuron, as a sensitivity analysis for rate dependence; thinning does not remove shared drift)
Usage: python 11_population.py
"""
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from common_still import BIN, OUT, SUBJ, beryl, cosmos, done_eids, load_session, probes, states

MIN_NEU, MIN_SAMP = 8, 200


def stats(Y):
    """Y neurons x samples (counts)."""
    sd = Y.std(1)
    Y = Y[sd > 0]
    if len(Y) < MIN_NEU:
        return dict(corr=np.nan, pr=np.nan, burst=np.nan, rate=np.nan, n_used=len(Y))
    C = np.corrcoef(Y)
    iu = np.triu_indices(len(C), 1)
    lam = np.linalg.eigvalsh(C)
    pop = Y.sum(0)
    z = (pop - pop.mean()) / (pop.std() + 1e-9)
    return dict(corr=float(C[iu].mean()), pr=float(lam.sum() ** 2 / (lam ** 2).sum() / len(C)),
                burst=float((z > 3).mean()), rate=float(Y.mean() / (2 * BIN)), n_used=len(Y))


def session(eid):
    s = load_session(eid)
    st = states(s)
    nb, grey = st["nb"], st["grey"]
    nb2 = nb // 2
    pair = lambda m: m[: 2 * nb2].reshape(nb2, 2).all(1)
    tsm = st["tsm"] * BIN
    M2 = {"still": pair(grey & st["awake"]), "move": pair(grey & st["mov"]),
          "still_early": pair(grey & st["awake"] & (tsm < 1)), "still_late": pair(grey & st["awake"] & (tsm >= 2))}
    rng = np.random.default_rng(int(eid[:8], 16))
    rows = []
    for f in probes(eid):
        p = np.load(f, allow_pickle=True)
        X = p["counts"][:, : 2 * nb2]
        rate_all = X.sum(1, dtype=np.float64) / (nb * BIN)
        regs = beryl(p["atlas_id"])
        X2 = X.reshape(len(X), nb2, 2).sum(2, dtype=np.int16)
        for r in np.unique(regs):
            ids = np.flatnonzero((regs == r) & (rate_all >= 0.5))
            if len(ids) < MIN_NEU or r in ("root", "void"):
                continue
            Xr = X2[ids].astype(np.float32)
            for comp in (("still", "move"), ("still_early", "still_late")):
                n = min(M2[c].sum() for c in comp)
                if n < MIN_SAMP:
                    continue
                Y = {c: Xr[:, rng.choice(np.flatnonzero(M2[c]), n, replace=False)] for c in comp}
                mu = np.stack([Y[c].mean(1) for c in comp], 1)
                lo = mu.min(1)
                # Use identical nonconstant neuron sets in both states, separately
                # for raw and thinned comparisons; thinning may silence units.
                thinned = {}
                for c, k in zip(comp, (0, 1)):
                    pth = np.where(mu[:, k] > 0, lo / np.maximum(mu[:, k], 1e-9), 1.0)
                    thinned[c] = rng.binomial(Y[c].astype(np.int64), pth[:, None]).astype(np.float32)
                common_raw = np.logical_and.reduce([Y[c].std(1) > 0 for c in comp])
                common_matched = np.logical_and.reduce([thinned[c].std(1) > 0 for c in comp])
                for c, k in zip(comp, (0, 1)):
                    for variant, Yv in (("raw", Y[c][common_raw]),
                                        ("matched", thinned[c][common_matched])):
                        rows.append(dict(eid=eid, probe=str(p["probe"]), region=r, comp="/".join(comp), state=c,
                                         variant=variant, n_neu=len(ids), n_samp=int(n), **stats(Yv)))
    return rows


if __name__ == "__main__":
    with ProcessPoolExecutor(8) as ex:
        res = [r for rr in ex.map(session, done_eids()) for r in rr]
    P = pd.DataFrame(res)
    P["subject"] = P.eid.map(SUBJ)
    from iblatlas.regions import BrainRegions
    br = BrainRegions()
    acr = P.region.unique()
    P["cosmos"] = P.region.map(dict(zip(acr, cosmos(br.acronym2id(acr)))))
    P.to_parquet(OUT / "11_population.pqt")
    print(P.groupby(["variant", "comp", "state"])[["corr", "pr", "burst", "rate"]].median().round(4))
    print(len(P), P.eid.nunique(), P.subject.nunique())
