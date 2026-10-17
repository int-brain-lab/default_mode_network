"""Pilot A: task-negative (rest-preference) index per neuron and region.

Per neuron, on trials that have all windows:
  rest     = 'inter_trial' window (stimOn -1.15..-1.0 s); split into
             still (no wheel move, no lick in the window) and moving
  quies    = quiescence window (stimOn -0.4..-0.1 s; wheel still by design)
  task     = mean of motor_init (firstMove -0.15..0), move (firstMove
             0..0.15) and feedback (0..0.3 s)
Indices, (a - b) / (a + b) on trial-averaged rates:
  RI_task   rest_still vs task
  RI_quies  rest_still vs quiescence (both still; pure engagement contrast)
  MI_move   rest_moving vs rest_still (movement effect inside 'rest')
Region = Beryl; keep regions with >= MIN_N neurons from >= MIN_SESS sessions.

Feasibility: split-half (by session) reliability of the region maps.
DMN test (within isocortex): DMN core / core+lateral vs other cortical
regions, region-label permutation p and session-bootstrap CI.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import (DMN_ALL, DMN_CORE, FIGS, OUT, beryl, cosmos, insertions,
                    load_cache)

MIN_RATE = 0.5
MIN_N = 20
MIN_SESS = 2
MIN_TR = 10
rng = np.random.default_rng(0)
mv = pd.read_parquet(OUT / "01b_movement_trials.pqt")


def rates(c, key):
    tr = c[f"{key}__trials"]
    return pd.DataFrame(c[f"{key}__counts"] / c[f"{key}__dur"], index=pd.Index(tr, name="trial"))


def ri(a, b):
    return (a - b) / (a + b + 1e-9)


rows = []
for eid, probe in zip(insertions().eid, insertions().probe):
    c = load_cache(eid, probe)
    rest, quies, mi = rates(c, "inter_trial"), rates(c, "quiescence"), rates(c, "motor_init")
    mvt = pd.concat([rates(c, "choiceL"), rates(c, "choiceR")])
    fb = pd.concat([rates(c, "fback1"), rates(c, "fback0")])
    tr = rest.index.intersection(quies.index).intersection(mi.index).intersection(mvt.index).intersection(fb.index)
    m = mv[mv.eid == eid].set_index("trial").loc[tr]
    still = ((m.rest_wheel == 0) & (m.rest_licks.fillna(0) == 0)).values
    if still.sum() < MIN_TR or (~still).sum() < MIN_TR:
        continue
    R, Q = rest.loc[tr].values, quies.loc[tr].values
    T = (mi.loc[tr].values + mvt.loc[tr].values + fb.loc[tr].values) / 3
    rs, rm = R[still].mean(0), R[~still].mean(0)
    rq, rt = Q.mean(0), T.mean(0)
    rows.append(pd.DataFrame({
        "eid": eid, "probe": probe, "uuid": c["uuids"], "atlas_id": c["ids"],
        "x": c["xyz"][:, 0], "y": c["xyz"][:, 1], "z": c["xyz"][:, 2],
        "n_still": still.sum(), "n_moving": (~still).sum(),
        "r_rest_still": rs, "r_rest_moving": rm, "r_quies": rq, "r_task": rt,
        "RI_task": ri(rs, rt), "RI_quies": ri(rs, rq), "MI_move": ri(rm, rs),
        "RI_task_allrest": ri(R.mean(0), rt), "RI_quies_allrest": ri(R.mean(0), rq)}))

N = pd.concat(rows, ignore_index=True)
N["region"] = beryl(N.atlas_id)
N["cosmos"] = cosmos(N.atlas_id)
N = N[~N.region.isin(["root", "void"])]
N["mean_rate"] = (N.r_rest_still + N.r_quies + N.r_task) / 3
N = N[N.mean_rate >= MIN_RATE].reset_index(drop=True)
N.to_parquet(OUT / "02_neurons.pqt")
print("still/moving trials per insertion (median):", N.groupby(["eid", "probe"]).n_still.first().median(),
      N.groupby(["eid", "probe"]).n_moving.first().median())

COLS = ["RI_task", "RI_quies", "MI_move", "RI_task_allrest", "RI_quies_allrest"]
g = N.groupby("region")
R = pd.DataFrame({"n": g.size(), "n_sess": g.eid.nunique(),
                  "cosmos": g.cosmos.agg(lambda s: s.mode().iat[0])})
for col in COLS:
    R[col] = g[col].mean()
R = R[(R.n >= MIN_N) & (R.n_sess >= MIN_SESS)].copy()
R["dmn"] = np.where(R.index.isin(DMN_CORE), "core", np.where(R.index.isin(DMN_ALL), "lateral", "no"))
N = N[N.region.isin(R.index)].reset_index(drop=True)

# session x region sums for fast resampling
sess = np.sort(N.eid.unique())
regs = R.index.values
si = pd.Index(sess).get_indexer(N.eid)
rj = pd.Index(regs).get_indexer(N.region)
CNT = np.zeros((len(sess), len(regs)))
np.add.at(CNT, (si, rj), 1)
SUM = {}
for col in COLS:
    S = np.zeros((len(sess), len(regs)))
    np.add.at(S, (si, rj), N[col].values)
    SUM[col] = S


def region_mean(col, w):
    """w: weights per session (bootstrap counts or split mask)."""
    num = w @ SUM[col]
    den = w @ CNT
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den, den


rel = {}
for col in COLS:
    rs = []
    for _ in range(500):
        h = rng.permutation(len(sess)) < len(sess) // 2
        a, na = region_mean(col, h.astype(float))
        b, nb = region_mean(col, (~h).astype(float))
        ok = (na >= 5) & (nb >= 5)
        rs.append(spearmanr(a[ok], b[ok]).statistic)
    r = float(np.median(rs))
    rel[col] = {"split_half_r": round(r, 3), "spearman_brown": round(2 * r / (1 + r), 3),
                "q05_q95": np.quantile(rs, [.05, .95]).round(3).tolist()}

ctx = R.cosmos.values == "Isocortex"
res = {"n_neurons": int(len(N)), "n_sessions": int(len(sess)), "n_regions": int(len(R)),
       "n_cortical_regions": int(ctx.sum()),
       "dmn_core_present": R.index[R.dmn == "core"].tolist(),
       "dmn_lateral_present": R.index[R.dmn == "lateral"].tolist(),
       "reliability": rel, "tests": {}}
B = rng.multinomial(len(sess), np.ones(len(sess)) / len(sess), size=1000).astype(float)
for col in COLS:
    v = R[col].values
    for setname, members in (("core", ["core"]), ("core+lateral", ["core", "lateral"])):
        isd = R.dmn.isin(members).values & ctx
        oth = (~R.dmn.isin(members).values) & ctx
        obs = v[isd].mean() - v[oth].mean()
        lab = isd[ctx]
        vc = v[ctx]
        null = np.array([vc[p].mean() - vc[~p].mean() for p in (rng.permutation(lab) for _ in range(20000))])
        p_perm = (np.sum(np.abs(null) >= abs(obs)) + 1) / (len(null) + 1)
        bm, _ = region_mean(col, B)
        boots = np.nanmean(bm[:, isd], 1) - np.nanmean(bm[:, oth], 1)
        res["tests"][f"{col}|{setname}"] = {
            "dmn": round(float(v[isd].mean()), 4), "other_ctx": round(float(v[oth].mean()), 4),
            "diff": round(float(obs), 4), "p_perm": round(float(p_perm), 4),
            "boot_ci95": np.nanquantile(boots, [.025, .975]).round(4).tolist()}
    res[f"{col}|by_cosmos"] = R.groupby("cosmos")[col].mean().round(3).to_dict()
    res[f"{col}|top10"] = R[col].sort_values(ascending=False).head(10).round(3).to_dict()
R.to_csv(OUT / "02_regions.csv")
print(json.dumps(res, indent=1))
json.dump(res, open(OUT / "02_rest_index.json", "w"), indent=1)

C = R[ctx]
fig, axs = plt.subplots(3, 1, figsize=(10, 7.5))
for ax, col, lab in zip(axs, ("RI_task", "RI_quies", "MI_move"),
                        ("rest(still) vs task", "rest(still) vs quiescence", "rest moving vs still")):
    s = C.sort_values(col)
    ax.bar(range(len(s)), s[col], color=s.dmn.map({"core": "C3", "lateral": "C1", "no": "0.6"}))
    ax.set_xticks(range(len(s)))
    ax.set_xticklabels(s.index, rotation=90, fontsize=6)
    ax.set_ylabel(lab, fontsize=8)
    ax.axhline(0, color="k", lw=.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.locator_params(axis="y", nbins=3)
axs[0].set_title("Isocortex; red = DMN core, orange = DMN lateral", fontsize=8)
fig.tight_layout()
fig.savefig(FIGS / "02_rest_index_cortex.pdf")
fig.savefig(FIGS / "02_rest_index_cortex.png", dpi=150)
