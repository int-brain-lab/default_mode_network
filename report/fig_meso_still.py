"""Figure 17 of the comprehensive report: correlates of stillness in the IBL mesoscope (meso/34_meso_still.py).

Writes comprehensive_report/fig17_meso_still.* and meso/output/35_meso_still_stats.json.
Usage: python fig_meso_still.py
"""
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from plotstyle import NATURE_DOUBLE_COL_IN, allen_color, panel_label, save_with_caption, style_axes  # noqa: E402

MESO = HERE.parent / "meso" / "output"
OUTD = HERE.parent / "comprehensive_report"
STEM = OUTD / "fig17_meso_still"
rng = np.random.default_rng(0)
GREY = "#8a8880"
GROUPS = [("all", "all ROIs", "Isocortex"), ("VIS", "visual", "VIS"), ("SS", "somatosensory", "SSp"),
          ("MO", "motor", "MO"), ("RSP", "RSP (core)", "RSP")]
MIN_ROIS = 20
MAX_NULL = .05  # QC: sessions whose circular-shift null exceeds 5% (slow drift dominates) are dropped


def boot(v, n=2000):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    b = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)])
    return v.mean(), *np.quantile(b, [.025, .975])


def grp(a):
    for p in ("VIS", "SS", "MO", "RSP"):
        if a.startswith(p):
            return p
    return "other"


SE = pd.read_csv(MESO / "34_still_sessions.csv")
R = pd.read_parquet(MESO / "34_still_rois.pqt")
ok = SE.dropna(subset=["obs_still"])
bad = ok[(ok.null_still > MAX_NULL) | (ok.null_move > MAX_NULL)]
ok = ok[~ok.eid.isin(bad.eid)]
R = R[R.eid.isin(ok.eid)].copy()
R["grp"] = R.beryl.astype(str).map(grp)
R["SP"], R["MP"] = (R.label == 1).astype(float), (R.label == -1).astype(float)


def per_mouse(D, col):
    g = D.groupby(["subject", "eid"])[col].agg(["mean", "count"])
    return g[g["count"] >= MIN_ROIS]["mean"].groupby("subject").mean()


S = dict(sessions=int(len(ok)), mice=int(ok.subject.nunique()), rois=int(len(R)), qc_excluded=bad.eid.tolist(),
         sessions_screened=int(len(SE)), sessions_with_block=int((SE.block_s > 0).sum()),
         block_s=float(ok.block_s.median()), still_bins=float(ok.n_still.median()), move_bins=float(ok.n_move.median()),
         still_sp=float(ok.n_still_sp.median()), move_sp=float(ok.n_move_sp.median()),
         rois_per_session=float(ok.n_rois.median()), wheel_cov=float(ok.wheel_cov.median()))
S["ratio"] = {}
for key, lab, _ in GROUPS:
    v = per_mouse(R if key == "all" else R[R.grp == key], "ratio")
    mu, lo, hi = boot(v)
    S["ratio"][lab] = dict(mean=mu, lo=lo, hi=hi, n=int(len(v)), n_neg=int((v < 0).sum()))
NM = ok.groupby("subject")[["obs_still", "null_still", "obs_move", "null_move"]].mean() * 100
S["pref"] = {c: dict(zip(["mean", "lo", "hi"], boot(NM[c]))) for c in NM}
S["pref"]["sessions_p05"] = dict(still=int((ok.p_still < .05).sum()), move=int((ok.p_move < .05).sum()))
# RSP against the other ROIs of the same session
rs = R[R.grp == "RSP"].groupby(["subject", "eid"]).agg(n=("ratio", "size"), rsp=("ratio", "mean"), rsp_sp=("SP", "mean"))
rs = rs[rs.n >= MIN_ROIS]
oth = R[R.grp != "RSP"].groupby("eid").agg(oth=("ratio", "mean"), oth_sp=("SP", "mean"))
rs = rs.join(oth, on="eid").reset_index()
rm = rs.groupby("subject")[["rsp", "oth", "rsp_sp", "oth_sp"]].mean()
S["rsp"] = dict(sessions=int(len(rs)), mice=int(len(rm)), rsp=float(rm.rsp.mean()), other=float(rm.oth.mean()),
                diff=float((rm.rsp - rm.oth).mean()), sessions_rsp_less_suppressed=int((rs.rsp > rs.oth).sum()),
                rsp_sp=float(100 * rm.rsp_sp.mean()), other_sp=float(100 * rm.oth_sp.mean()),
                regions=R[R.grp == "RSP"].beryl.value_counts().to_dict())

fig = plt.figure(figsize=(NATURE_DOUBLE_COL_IN, 2.3))
gs = fig.add_gridspec(1, 3, wspace=.5, left=.08, right=.98, top=.9, bottom=.25, width_ratios=[1.4, 1, .8])

ax = fig.add_subplot(gs[0])
for i, (key, lab, col) in enumerate(GROUPS):
    v = per_mouse(R if key == "all" else R[R.grp == key], "ratio")
    c = allen_color(col)
    ax.scatter(i + rng.uniform(-.15, .15, len(v)), v, s=5, color=c, lw=0, alpha=.8)
    mu, lo, hi = boot(v)
    ax.plot([i, i], [lo, hi], color="k", lw=1)
    ax.plot([i - .25, i + .25], [mu, mu], color="k", lw=1)
    ax.text(i, -1.35, f"n={len(v)}", ha="center", fontsize=4.5, color=GREY)
ax.axhline(0, color=GREY, lw=.5, ls=":")
ax.set_xticks(range(len(GROUPS)))
ax.set_xticklabels([g[1] for g in GROUPS], rotation=30, ha="right", rotation_mode="anchor", fontsize=5.5)
ax.set_ylim(-1.4, .4)
ax.set_ylabel("log$_2$ ratio (still / moving)")
style_axes(ax, categorical_x=True)
panel_label(ax, "a")

ax = fig.add_subplot(gs[1])
for j, (o, n, lab) in enumerate((("obs_still", "null_still", "still-pref."), ("obs_move", "null_move", "movement-pref."))):
    for x, col, c in ((2 * j, n, GREY), (2 * j + .8, o, "#222222")):
        ax.scatter(x + rng.uniform(-.08, .08, len(NM)), NM[col], s=5, color=c, lw=0, alpha=.8)
        ax.plot([x - .2, x + .2], [NM[col].mean()] * 2, color="k", lw=1)
    for _, r in NM.iterrows():
        ax.plot([2 * j, 2 * j + .8], [r[n], r[o]], color=GREY, lw=.3, alpha=.6)
ax.set_xticks([0, .8, 2, 2.8])
ax.set_xticklabels(["null", "obs.", "null", "obs."], fontsize=5.5)
ax.text(.4, -.3, "still-pref.", transform=ax.get_xaxis_transform(), ha="center", fontsize=5.5)
ax.text(2.4, -.3, "movement-pref.", transform=ax.get_xaxis_transform(), ha="center", fontsize=5.5)
ax.set_ylabel("ROIs per mouse (%)")
style_axes(ax, categorical_x=True)
panel_label(ax, "b")

ax = fig.add_subplot(gs[2])
for s, d in rs.groupby("subject"):
    for _, r in d.iterrows():
        ax.plot([0, 1], [r.oth, r.rsp], color=GREY, lw=.4)
    ax.scatter(np.zeros(len(d)), d.oth, s=5, color=GREY, lw=0)
    ax.scatter(np.ones(len(d)), d.rsp, s=5, color=allen_color("RSP"), lw=0)
ax.axhline(0, color=GREY, lw=.5, ls=":")
ax.set_xticks([0, 1])
ax.set_xticklabels(["other\ncortex", "RSP"], fontsize=5.5)
ax.set_xlim(-.4, 1.4)
ax.set_ylabel("log$_2$ ratio per session")
style_axes(ax, categorical_x=True)
panel_label(ax, "c")

save_with_caption(fig, STEM, [
    f"Activity in still against moving 1\\,s bins of the passive block before the task replay (grey screen and "
    f"receptive-field mapping; median {S['block_s'] / 60:.1f}\\,min per session), log$_2$((still\\,+\\,$e$)/"
    "(moving\\,+\\,$e$)) of each ROI's deconvolved activity with $e$ = 2\\% of its mean; ROIs averaged per session "
    f"(sessions with $\\geq${MIN_ROIS} ROIs in the group), sessions per mouse. Moving: lick or face motion energy above "
    "the session's 60th percentile (the wheel is not recorded in this block); still: no movement within $\\pm$1\\,s "
    "and $\\leq$30\\,s after the last movement. Groups by Beryl region (Allen colours); black, mean and 95\\% "
    "bootstrap CI over mice; n, mice.",
    "Still- and movement-preferring ROIs per mouse (Welch test of still against moving bins, at most one of each "
    "per 2\\,s, $p<0.01$), observed against a circular-shift null (activity rotated against the behavioural labels, "
    "100 offsets $\\geq$60\\,s). Bars: means over mice. One session whose null exceeded 5\\% (slow drift; "
    "a quarter of its ROIs silent in all still bins) is excluded from all panels.",
    "RSP against the other imaged ROIs of the same session (lines: sessions)."],
    label="meso_still", number=17,
    intro="Correlates of stillness in the IBL mesoscope: lower activity in stillness across dorsal cortex.")
subprocess.run(["pdftoppm", "-png", "-r", "200", "-singlefile", f"{STEM}.pdf", str(STEM)], check=True)
plt.close(fig)


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


json.dump(clean(S), open(MESO / "35_meso_still_stats.json", "w"), indent=1)
print(json.dumps(clean(S), indent=1))
