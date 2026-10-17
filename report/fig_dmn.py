"""Report figure: BWM task suppression of DMN cortex + mesoscope region coupling."""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "bwm"))
from plotstyle import NATURE_DOUBLE_COL_IN, panel_label, save_with_caption, style_axes  # noqa: E402
from common import DMN_CORE, DMN_LATERAL  # noqa: E402

BWM = HERE.parent / "bwm" / "output"
MESO = HERE.parent / "meso" / "output"
C_CORE, C_LAT, C_OTH = "#2a78d6", "#eb6834", "#a3a29b"
rng = np.random.default_rng(0)


def col(r):
    return C_CORE if r in DMN_CORE else C_LAT if r in DMN_LATERAL else C_OTH


fig = plt.figure(figsize=(NATURE_DOUBLE_COL_IN, 3.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], hspace=0.75, wspace=0.65)

# a: suppressed fraction per cortical region
A = json.load(open(BWM / "20_full_A_A_full.json"))
rk = pd.Series(A["ctx_ranking_suppressed"]).sort_values(ascending=False)
ax = fig.add_subplot(gs[0, :])
ax.bar(np.arange(len(rk)), rk.values * 100, color=[col(r) for r in rk.index], width=0.75)
ax.set_xticks(np.arange(len(rk)))
ax.set_xticklabels(rk.index, rotation=60, ha="right", fontsize=5)
ax.set_xlim(-0.7, len(rk) - 0.3)
ax.set_ylabel("task-suppressed\nneurons (%)")
style_axes(ax, categorical_x=True)
for i, (lab, c) in enumerate((("DMN core", C_CORE), ("DMN lateral", C_LAT), ("other cortex", C_OTH))):
    ax.text(0.99, 0.95 - 0.14 * i, lab, color=c, transform=ax.transAxes, ha="right", va="top", fontsize=6)
panel_label(ax, "a", x=-0.05)

# b: subdivision tests (strict stillness): DMN vs matched non-DMN per region
S = json.load(open(BWM / "26_A_subsets_A_full_strict.json"))
ax = fig.add_subplot(gs[1, 0])
for i, part in enumerate(("anterior", "posterior", "lateral")):
    v = S[f"{part}|suppressed"]
    for j, (grp, c) in enumerate((("other", C_OTH), ("dmn", C_LAT if part == "lateral" else C_CORE))):
        y = np.array(list(v[grp].values())) * 100
        x = i * 2.6 + j + rng.uniform(-0.18, 0.18, len(y))
        ax.scatter(x, y, s=6, color=c, lw=0)
        ax.plot([i * 2.6 + j - 0.3, i * 2.6 + j + 0.3], [y.mean()] * 2, color="k", lw=1)
    ax.text(i * 2.6 + 0.5, 47, f"p={v['p_perm']:.3f}", ha="center", fontsize=5)
ax.set_xticks([0.5, 3.1, 5.7])
ax.set_xticklabels(["frontal", "posterior", "lateral"])
ax.set_ylim(0, 50)
ax.set_ylabel("task-suppressed (%)")
style_axes(ax, categorical_x=True)
panel_label(ax, "b", x=-0.3)

# c: mesoscope pair coupling by class, per mouse
P = pd.read_parquet(MESO / "30_pairs_motion_iti1.pqt")
P = P[(P.rel1 >= .5) & (P.rel2 >= .5)].copy()
subj = pd.read_csv(MESO / "session_subjects.csv").drop_duplicates("session").set_index("session").subject
P["mouse"] = P.eid.map(subj)
P["z"] = np.arctanh(P.r.clip(-.999, .999))
db = pd.cut(P.dist_mm, [0, .5, 1, 1.5, 2, 2.5, 3, 4, 10], labels=False)
P["resid"] = P.z - P.groupby(db).z.transform("mean")
d1, d2 = P.r1.isin(DMN_CORE), P.r2.isin(DMN_CORE)
P["cl"] = np.where(d1 & d2, "DMN–DMN", np.where(d1 | d2, "DMN–other", "other–other"))
M = P.groupby(["mouse", "cl"]).resid.mean().unstack()
ax = fig.add_subplot(gs[1, 1])
order = ["other–other", "DMN–other", "DMN–DMN"]
for i, c in enumerate(order):
    y = M[c].dropna().values
    ax.scatter(i + rng.uniform(-0.15, 0.15, len(y)), y, s=6, color=(C_OTH, "#86b6ef", C_CORE)[i], lw=0)
    bs = [rng.choice(y, len(y)).mean() for _ in range(2000)]
    ax.errorbar(i + 0.3, y.mean(), yerr=[[y.mean() - np.quantile(bs, .025)], [np.quantile(bs, .975) - y.mean()]],
                fmt="o", color="k", ms=2.5, lw=0.8, capsize=0)
ax.axhline(0, color="k", lw=0.4, ls=":")
ax.set_xticks(range(3))
ax.set_xticklabels(order, rotation=30, ha="right")
ax.set_ylabel("ITI coupling\n(distance residual, z)")
style_axes(ax, categorical_x=True)
panel_label(ax, "c", x=-0.3)

# d: which DMN pairs
ck = json.load(open(MESO / "32_meso_checks_motion_iti1.json"))
dp = pd.DataFrame(ck["dmn_pairs"]).sort_values("resid")
ax = fig.add_subplot(gs[1, 2])
ax.barh(np.arange(len(dp)), dp.resid, color=C_CORE, height=0.6)
ax.set_yticks(np.arange(len(dp)))
ax.set_yticklabels([f"{a}–{b} ({m})" for a, b, m in zip(dp.r1, dp.r2, dp.mice)], fontsize=5)
ax.axvline(0, color="k", lw=0.4)
ax.set_xlabel("distance residual (z)")
style_axes(ax, categorical_y=True)
panel_label(ax, "d", x=-0.55)

caps = [
    "BWM Neuropixels, 419 sessions, 137 mice: percentage of neurons per cortical region (Beryl) significantly "
    "suppressed during the task (stimulus, movement and feedback windows) relative to still inter-trial rest. "
    "Blue, DMN core (ACA, PL, ILA, ORB, RSP, VISa, VISam); orange, DMN lateral (TEa, ECT, PERI); grey, other cortex.",
    "Same measure under strict stillness, DMN regions vs non-DMN neighbours within frontal (vs MOs, MOp, FRP, AI), "
    "posterior (vs SSp, AUD, VIS) and lateral (vs AUD, VISC, GU) cortex; dots are regions, bars means, "
    "region-permutation p.",
    "Mesoscope (45 sessions, 11 mice): correlation of region-averaged calcium activity in still inter-trial "
    "bins (1 s, wheel/lick/motion-energy regressed), as residual from the distance-matched mean; dots are mice, "
    "black mean and 95\\% mouse-bootstrap CI.",
    "Mean residual coupling per DMN region pair (number of mice in brackets); the DMN--DMN excess is carried by "
    "VISa--VISam and RSP pairs.",
]
save_with_caption(fig, HERE / "fig_dmn", caps, label="dmn")
