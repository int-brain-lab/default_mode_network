"""Figures for the grey-screen (passive spontaneous period) report.

Reads outputs of bwm/40_spont_characterize.py, 42_spont_A.py, 41_spont_network.py,
21_full_B.py (spont ... still) and meso/33_meso_spont.py. Writes report/grey/fig*.pdf
(+ .png for the web page).
"""
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "bwm"))
from plotstyle import NATURE_DOUBLE_COL_IN, panel_label, save_with_caption, style_axes  # noqa: E402
from common import DMN_ALL, DMN_CORE, LU_DMN, LU_PARIETAL, LU_PFC_TEMPORAL  # noqa: E402

BWM = HERE.parent / "bwm" / "output"
MESO = HERE.parent / "meso" / "output"
OUTD = HERE / "grey"
OUTD.mkdir(exist_ok=True)
BTAG = "spont_motion_1s_still"
# categorical slots 1-3 of the dataviz reference palette + neutral
C_BOTH, C_OURS, C_LU, C_OTH = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29b"
C_GREY, C_ITI = "#2a78d6", "#a3a29b"
rng = np.random.default_rng(0)
SUBJ = pd.read_csv(BWM / "bwm_insertions.csv").groupby("eid").subject.first()


def rcol(r):
    a, b = r in DMN_ALL, r in LU_DMN
    return C_BOTH if a and b else C_OURS if a else C_LU if b else C_OTH


def legend_sets(ax, x=0.99, y=0.97, dy=0.12):
    for i, (lab, c) in enumerate((("our DMN and Lu et al.", C_BOTH), ("our DMN only", C_OURS),
                                  ("Lu et al. only", C_LU), ("neither", C_OTH))):
        ax.text(x, y - dy * i, lab, color=c, transform=ax.transAxes, ha="right", va="top", fontsize=6)


def png(stem):
    subprocess.run(["pdftoppm", "-png", "-r", "200", "-singlefile", f"{stem}.pdf", str(stem)], check=True)


def mboot_mean(v, n=2000):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    b = [rng.choice(v, len(v)).mean() for _ in range(n)]
    return v.mean(), np.quantile(b, .025), np.quantile(b, .975)


# ======================================================================= Fig 1: protocol and timing
D = pd.read_csv(BWM / "40_spont_sessions.csv")
H = D[D.has_spont].copy()
fig = plt.figure(figsize=(NATURE_DOUBLE_COL_IN, 4.0))
gs = fig.add_gridspec(3, 3, height_ratios=[0.4, 1, 1.15], hspace=0.75, wspace=0.45)
ax = fig.add_subplot(gs[0, :])
blocks = [("task", 0, 38, "#d9d7cf", "k"), ("grey screen, 10 min", 42, 66, C_GREY, "white"),
          ("RF map, ~5 min", 67, 80, "#d9d7cf", "k"), ("task replay", 81, 96, "#d9d7cf", "k")]
for lab, a, b, c, tc in blocks:
    ax.add_patch(Rectangle((a, 0), b - a, 1, color=c, lw=0))
    ax.text((a + b) / 2, 0.5, lab, ha="center", va="center", fontsize=5.5, color=tc)
for xs in (41.4, 66.5, 80.5):
    ax.add_patch(Rectangle((xs - .35, -0.1), .7, 1.2, color="k", lw=0))
ax.text(41.4, 1.25, "spacer", fontsize=5, ha="center", va="bottom")
ax.annotate("", xy=(41, -0.25), xytext=(38, -0.25), arrowprops=dict(arrowstyle="<->", lw=0.6))
ax.text(39.5, -0.6, "median 1.1 min", fontsize=5, ha="center", va="top")
ax.set_xlim(0, 96.5)
ax.set_ylim(-1.0, 1.6)
ax.axis("off")
ax.text(96, 1.25, "not to scale", fontsize=5, ha="right", va="bottom", style="italic")
panel_label(ax, "a", x=-0.03, y=0.75)

ax = fig.add_subplot(gs[1, 0])
ax.hist(H.dur_s / 60, bins=np.arange(0, 12, 0.5), color=C_GREY)
ax.set_xlabel("grey-screen duration (min)")
ax.set_ylabel("sessions")
style_axes(ax)
panel_label(ax, "b", x=-0.3)
ax = fig.add_subplot(gs[1, 1])
ax.hist(H.gap_after_task_min.clip(upper=8), bins=np.arange(0, 8.25, 0.25), color=C_GREY)
ax.set_xlabel("start after last trial (min)")
ax.set_ylabel("sessions")
style_axes(ax)
panel_label(ax, "c", x=-0.3)
ax = fig.add_subplot(gs[1, 2])
ax.hist(H.frac_still, bins=np.arange(0.2, 1.01, 0.04), color=C_GREY)
ax.set_xlabel("still fraction of the period")
ax.set_ylabel("sessions")
style_axes(ax)
panel_label(ax, "d", x=-0.3)

# example session: median still fraction, full coverage
E = H[(H.cov_wheel > .99) & (H.cov_me > .99) & (H.dur_s > 590)].copy()
E["d"] = (E.frac_still - H.frac_still.median()).abs()
ex = E.sort_values("d").eid.iloc[0]
s = dict(np.load(BWM / "full" / f"{ex}.npz", allow_pickle=True))
BIN = float(s["BIN"])
sp = s["spont"]
a, b = int(sp[0] / BIN), int(sp[1] / BIN)
t = (np.arange(a, b) * BIN - sp[0]) / 60
w, me, lk = (s[k][a:b].astype(float) for k in ("wheel_speed", "motion_energy", "licks"))
thr = np.nanpercentile(s["motion_energy"].astype(float), 60)
mov = (w > 0.05) | ((lk > 0) & (lk < 255)) | (me > thr)
mov = mov | np.r_[False, mov[:-1]] | np.r_[mov[1:], False]
sub = gs[2, :].subgridspec(3, 1, hspace=0.15)
for i, (y, lab, c) in enumerate(((w, "wheel", "k"), (me / np.nanmedian(s["motion_energy"]), "motion\nenergy", "k"),
                                  (None, "still", None))):
    ax = fig.add_subplot(sub[i])
    if y is not None:
        ax.plot(t, y, lw=0.4, color=c)
        if i == 1:
            ax.axhline(thr / np.nanmedian(s["motion_energy"]), color=C_OURS, lw=0.6, ls="--")
        style_axes(ax, max_yticks=2)
        ax.set_ylabel(lab, fontsize=6, rotation=0, ha="right", va="center")
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
    else:
        ax.fill_between(t, 0, (~mov).astype(float), step="mid", color=C_GREY, lw=0)
        style_axes(ax)
        ax.set_yticks([])
        ax.set_ylabel(lab, fontsize=6, rotation=0, ha="right", va="center")
        ax.set_xlabel("time in grey-screen period (min)")
        ax.spines["left"].set_visible(False)
    ax.set_xlim(0, t[-1])
    if i == 0:
        panel_label(ax, "e", x=-0.085, y=1.0)
caps = [
    "IBL passive protocol after the task (iblrig \\texttt{passiveChoiceWorld}): frame2ttl spacers separate the spontaneous "
    "period, during which Bonsai clears the screen to grey with the sync square off and shows no stimulus "
    "(\\texttt{SPONTANEOUS\\_ACTIVITY\\_SECONDS} = 600), receptive-field mapping (sparse noise) and the replay of task "
    "stimuli, valve clicks and tones. Task length is schematic. The interval comes from "
    "\\texttt{\\_ibl\\_passivePeriods.intervalsTable}.",
    f"Grey-screen duration over the {len(H)} BWM sessions ({H.subject.nunique()} mice) that have it; 600 s in "
    f"{(H.dur_s.round(-1) == 600).mean() * 100:.0f}\\%.",
    "Delay between the end of the last task trial and the start of the grey screen (values above 8 min clipped).",
    "Fraction of the period that is still: 50 ms bins without wheel movement ($>$0.05 rad/s), licks or motion energy above the "
    "session's 60th percentile, with a one-bin margin (the first second is skipped); median 6.5 min of 10. In the 35 sessions without "
    "motion energy, stillness rests on wheel and licks only (median fraction 0.81 vs 0.67).",
    f"Example session with median stillness: wheel speed, left-camera motion energy (normalised to "
    "the session median; dashed, the stillness threshold; wheel in rad/s) and the resulting still bins.",
]
save_with_caption(fig, OUTD / "fig1_protocol", caps, label="grey_protocol", number=1)
png(OUTD / "fig1_protocol")

# ======================================================================= Fig 2: movement vs ITI
T = pd.read_csv(BWM / "40_spont_timecourse.csv")
fig = plt.figure(figsize=(NATURE_DOUBLE_COL_IN, 2.0))
gs = fig.add_gridspec(1, 4, wspace=0.6)
for k, (x, y, lab) in enumerate((("iti_frac_wheel_moving", "frac_wheel_moving", "wheel moving"),
                                 ("iti_frac_me_high", "frac_me_high", "motion energy high"))):
    ax = fig.add_subplot(gs[0, k])
    ax.scatter(H[x], H[y], s=3, color=C_GREY, lw=0, alpha=.6)
    ax.plot([0, 1], [0, 1], color="k", lw=0.5, ls=":")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel(f"task ITI, fraction")
    ax.set_ylabel(f"grey screen, fraction")
    ax.set_title(lab, fontsize=7)
    style_axes(ax)
    panel_label(ax, "ab"[k], x=-0.35)
ax = fig.add_subplot(gs[0, 2])
for i, (c, lab) in enumerate((("me_spont_over_iti", "motion\nenergy"), ("pupil_spont_over_iti", "pupil\ndiameter"))):
    v = H[c].dropna().values
    ax.scatter(i + rng.uniform(-.18, .18, len(v)), v, s=3, color=C_GREY, lw=0, alpha=.6)
    ax.plot([i - .25, i + .25], [np.median(v)] * 2, color="k", lw=1)
ax.axhline(1, color="k", lw=0.5, ls=":")
ax.set_yscale("log")
ax.set_ylim(0.1, 4)
from matplotlib.ticker import NullFormatter, NullLocator
ax.yaxis.set_minor_locator(NullLocator())
ax.yaxis.set_minor_formatter(NullFormatter())
ax.set_yticks([0.25, 0.5, 1, 2])
ax.set_yticklabels(["0.25", "0.5", "1", "2"])
ax.set_xticks([0, 1])
ax.set_xticklabels(["motion\nenergy", "pupil\ndiameter"])
ax.set_ylabel("grey screen / task ITI\n(median ratio)")
style_axes(ax, categorical_x=True, categorical_y=True)
panel_label(ax, "c", x=-0.45)
ax = fig.add_subplot(gs[0, 3])
g = T.groupby("minute")
for c, lab, col in (("still", "still", C_GREY), ("wheel", "wheel moving", C_OURS)):
    m = g[c].mean()
    se = g[c].sem()
    ax.fill_between(m.index + .5, m - 1.96 * se, m + 1.96 * se, color=col, alpha=.25, lw=0)
    ax.plot(m.index + .5, m, color=col, lw=1)
    ax.text(10.2, m.iloc[-1], lab, color=col, fontsize=6, va="center")
ax.set_xlim(0, 10)
ax.set_ylim(0, 1)
ax.set_xlabel("minute of grey screen")
ax.set_ylabel("fraction of bins")
style_axes(ax)
panel_label(ax, "d", x=-0.35)
caps = [
    "Fraction of 50 ms bins with wheel movement in the grey-screen period against task inter-trial intervals "
    "(feedback + 1.5 s to next stimulus onset $-$ 0.5 s) of the same session; dots are the 338 sessions with wheel "
    "data in both.",
    "Same for motion energy above the session's 60th percentile (361 sessions with motion energy).",
    "Median motion energy and pupil diameter in the grey-screen period relative to task ITIs; bars are medians over "
    "sessions.",
    "Time course over the grey-screen period in whole minutes (the first second is skipped, so a 600 s period gives "
    "9 full minutes; 300 s sessions contribute to the first 4): fraction of still bins and of bins with wheel "
    "movement, mean over sessions with 95\\% CI.",
]
save_with_caption(fig, OUTD / "fig2_movement", caps, label="grey_movement", number=2)
png(OUTD / "fig2_movement")

# ======================================================================= Fig 3: rest preference vs grey screen
G = json.load(open(BWM / "42_spont_A_G_full.json"))
RG = pd.read_csv(BWM / "42_regions_G_full.csv", index_col=0)
RA = pd.read_csv(BWM / "20_regions_A_full.csv", index_col=0)
C = RG[RG.cosmos == "Isocortex"].sort_values("suppressed_grey", ascending=False)
fig = plt.figure(figsize=(NATURE_DOUBLE_COL_IN, 3.9))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], hspace=0.8, wspace=0.55)
ax = fig.add_subplot(gs[0, :])
ax.bar(np.arange(len(C)), C.suppressed_grey * 100, color=[rcol(r) for r in C.index], width=0.75)
ax.set_xticks(np.arange(len(C)))
ax.set_xticklabels(C.index, rotation=60, ha="right", fontsize=5)
ax.set_xlim(-0.7, len(C) - 0.3)
ax.set_ylabel("task-suppressed\nneurons (%)")
style_axes(ax, categorical_x=True)
legend_sets(ax)
panel_label(ax, "a", x=-0.05)
sub3 = gs[1, :].subgridspec(1, 2, width_ratios=[1, 2.1], wspace=1.0)
ax = fig.add_subplot(sub3[0])
cm = C.index.intersection(RA.index)
ax.scatter(RA.loc[cm, "suppressed"] * 100, C.loc[cm, "suppressed_grey"] * 100, s=8, c=[rcol(r) for r in cm], lw=0)
xmax = RA.loc[cm, "suppressed"].idxmax()
for r, dx, ha in (("ILA", 1.2, "left"), ("RSPv", 1.2, "left"), ("VISl", 1.2, "left"), ("AUDd", 1.2, "left"),
                  ("ECT", -1.2, "right"), (xmax, -1.2, "right")):
    if r in cm:
        ax.text(RA.loc[r, "suppressed"] * 100 + dx, C.loc[r, "suppressed_grey"] * 100 + 1.5, r, fontsize=5,
                va="bottom", ha=ha)
ax.set_xlabel("suppressed vs task ITI (%)")
ax.set_ylabel("suppressed vs grey (%)")
style_axes(ax)
panel_label(ax, "b", x=-0.35)
sub3b = sub3[1].subgridspec(1, 2, wspace=0.12)
rows = [("ours_core", "our DMN core"), ("ours_core+lateral", "our DMN core + lateral"),
        ("lu_all", "Lu et al. all"), ("lu_without_auditory", "Lu et al. without AUD"),
        ("lu_pfc_temporal", "Lu prefrontal-temporal"), ("lu_parietal", "Lu parietal")]
for jj, (met, scale, xl) in enumerate((("suppressed_grey", 100, "suppressed (% points)"), ("RI_grey", 1, "rest index RI"))):
    ax = fig.add_subplot(sub3b[jj])
    for i, (k, lab) in enumerate(rows):
        for (suf, col, off) in (("", "k", -.15), ("_late", C_GREY, .15)):
            v = G["tests"][f"{met}{suf}|{k}|vs_other_ctx"]
            lo, hi = v["subject_boot_ci95"]
            ax.plot([lo * scale, hi * scale], [i + off] * 2, color=col, lw=1)
            ax.plot(v["diff"] * scale, i + off, "o", color=col, ms=2.6, mfc="white" if v["p_perm"] >= 0.05 else col)
    ax.axvline(0, color="k", lw=0.5, ls=":")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[1] for r in rows] if jj == 0 else [])
    ax.invert_yaxis()
    ax.set_xlabel(f"set $-$ other cortex,\n{xl}")
    style_axes(ax, categorical_y=True)
    if jj == 0:
        panel_label(ax, "c", x=-1.0)
        ax.text(0, 1.1, "all trials", color="k", fontsize=6, transform=ax.transAxes)
        ax.text(0, 1.02, "last third of trials", color=C_GREY, fontsize=6, transform=ax.transAxes)
caps = [
    f"Percentage of neurons per cortical region (Beryl) significantly suppressed during the task (movement onset, "
    f"movement and feedback windows) relative to still 0.3 s chunks of the grey-screen period (Welch test with one "
    f"chunk per 3 s, p $<$ 0.01; {G['n_neurons']:,} neurons, {G['n_sessions']} sessions, {G['n_subjects']} mice). Colour: membership in "
    "our mouse DMN definition (ACA, PL, ILA, ORB, RSP, VISa, VISam + TEa, ECT, PERI) and in the rat DMN of Lu et al. "
    "(2012) mapped to mouse regions (ORB, PL, ACA, AUDp/d/v, TEa, VISa, VISrl, RSP, VISam, VISpm; CA1 not shown).",
    f"Per region, the same measure referenced to task ITIs (earlier report) against the grey-screen reference; "
    f"Spearman $\\rho$ = {G['vs_iti_A']['suppressed']:.2f} over {G['vs_iti_A']['n']} regions.",
    "Difference between each region set and all other cortex in the suppressed fraction (left) and in the "
    "continuous rest index RI = (rest $-$ task)/(rest + task) (right), against all trials (black) or only the last "
    "third of trials, closest in time to the grey screen (blue): 95\\% mouse-bootstrap CI; filled markers, "
    "region-permutation p $<$ 0.05. Note that 81\\% of neurons differ significantly between grey screen and task, so "
    "the suppressed fraction is close to the sign of RI.",
]
save_with_caption(fig, OUTD / "fig3_restpref", caps, label="grey_restpref", number=3)
png(OUTD / "fig3_restpref")

# ======================================================================= Fig 4: coverage
P = pd.read_parquet(BWM / f"21_pairs_{BTAG}.pqt")
P["subject"] = P.eid.map(SUBJ)
M = pd.read_parquet(BWM / f"21_regions_{BTAG}.pqt")
keep = (P.c1.eq("Isocortex") | P.r1.eq("CA1")) & (P.c2.eq("Isocortex") | P.r2.eq("CA1"))
Q = P[keep]
order_groups = [("Lu prefrontal-temporal", LU_PFC_TEMPORAL), ("Lu parietal", LU_PARIETAL),
                ("our DMN only", [r for r in DMN_ALL if r not in LU_DMN])]
present = set(Q.r1) | set(Q.r2) | set(M[M.cosmos.eq("Isocortex") | M.region.eq("CA1")].region)
order, bounds = [], []
for lab, mem in order_groups:
    m = [r for r in mem if r in present]
    bounds.append((lab, len(order), len(order) + len(m)))
    order += m
others = sorted(present - set(order), key=lambda r: (r[:3], r))
bounds.append(("other cortex", len(order), len(order) + len(others)))
order += others
idx = {r: i for i, r in enumerate(order)}
n = len(order)
MM = np.zeros((n, n))
XX = np.zeros((n, n))
for (r1, r2), g in Q.groupby(["r1", "r2"]):
    i, j = idx[r1], idx[r2]
    MM[i, j] = MM[j, i] = g.subject.nunique()
    XX[i, j] = XX[j, i] = g[~g.same_probe].subject.nunique()
fig = plt.figure(figsize=(NATURE_DOUBLE_COL_IN, 8.2))
gs = fig.add_gridspec(2, 3, height_ratios=[3.2, 1], hspace=0.28, wspace=0.75)
ax = fig.add_subplot(gs[0, :])
Mshow = np.where(MM > 0, MM, np.nan)
im = ax.imshow(Mshow, cmap="Blues", vmin=0, vmax=max(5, np.nanmax(Mshow)), interpolation="nearest")
for i in range(n):
    for j in range(n):
        if XX[i, j] > 0:
            ax.plot(j, i, marker="x", ms=3, color=C_OURS, mew=0.8)
for lab, a, b in bounds[:-1]:
    ax.add_patch(Rectangle((a - .5, a - .5), b - a, b - a, fill=False, ec="k", lw=0.8))
ax.add_patch(Rectangle((-.5, -.5), bounds[1][2], bounds[1][2], fill=False, ec=C_LU, lw=1.2, ls="--"))
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(order, rotation=90, fontsize=5)
ax.set_yticklabels(order, fontsize=5)
for t_, r in zip(ax.get_yticklabels(), order):
    t_.set_color(rcol(r) if rcol(r) != C_OTH else "k")
for t_, r in zip(ax.get_xticklabels(), order):
    t_.set_color(rcol(r) if rcol(r) != C_OTH else "k")
cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, shrink=0.6)
cb.set_label("mice with the pair", fontsize=6)
cb.ax.tick_params(labelsize=5)
panel_label(ax, "a", x=-0.03, y=1.0)
ax = fig.add_subplot(gs[1, 0])
k_ours = M[M.region.isin(DMN_ALL)].groupby("eid").region.nunique().reindex(M.eid.unique(), fill_value=0)
k_lu = M[M.region.isin(LU_DMN)].groupby("eid").region.nunique().reindex(M.eid.unique(), fill_value=0)
xs = np.arange(0, 6)
ax.bar(xs - .2, [(k_ours == x).sum() for x in xs], width=.4, color=C_OURS, label="ours")
ax.bar(xs + .2, [(k_lu == x).sum() for x in xs], width=.4, color=C_LU, label="Lu et al.")
ax.set_xlabel("DMN regions per session")
ax.set_ylabel("sessions")
ax.text(0.98, 0.95, "our DMN (core + lateral)", color=C_OURS, transform=ax.transAxes, ha="right", fontsize=6, va="top")
ax.text(0.98, 0.83, "Lu et al.", color=C_LU, transform=ax.transAxes, ha="right", fontsize=6, va="top")
style_axes(ax)
ax.set_xticks(range(0, 6))
panel_label(ax, "b", x=-0.4)
ax = fig.add_subplot(gs[1, 1])
kc = M[M.cosmos == "Isocortex"].groupby("eid").region.nunique().reindex(M.eid.unique(), fill_value=0)
ax.bar(range(0, 8), [(kc == x).sum() for x in range(8)], color=C_OTH)
ax.set_xlabel("cortical regions\nper session")
ax.set_ylabel("sessions")
style_axes(ax)
panel_label(ax, "c", x=-0.4)
ax = fig.add_subplot(gs[1, 2])
J = json.load(open(BWM / f"41_spont_network_{BTAG}.json"))
labels = ["our DMN core", "our core + lateral", "Lu et al."]
keys = ["ours_core", "ours_all", "lu"]
poss = [J["coverage"][k]["n_possible_edges"] for k in keys]
anyp = [J["coverage"][k]["n_edges_any"] for k in keys]
ge3 = [J["coverage"][k]["n_edges_ge3_mice"] for k in keys]
y = np.arange(3)
cx3 = [J["coverage"][k]["n_edges_cross_probe_ge3_mice"] for k in keys]
for j, (vals, col) in enumerate(((poss, "#cfccc2"), (anyp, "#86b6ef"), (ge3, C_BOTH), (cx3, C_OURS))):
    ax.barh(y - 0.3 + j * 0.2, vals, color=col, height=0.18)
    for i in range(3):
        ax.text(vals[i] + 2, y[i] - 0.3 + j * 0.2, str(vals[i]), va="center", fontsize=4.5)
ax.set_yticks(y)
ax.set_yticklabels(labels)
ax.invert_yaxis()
ax.set_xlabel("region pairs (edges)")
ax.set_xlim(0, max(poss) * 1.2)
for j, (lab, col) in enumerate((("possible", "#a9a597"), ("recorded", "#5b9be6"), ("$\\geq$3 mice", C_BOTH),
                                ("$\\geq$3 mice, 2 probes", C_OURS))):
    ax.text(1.0, 1.12 - j * 0.08, lab, color=col, fontsize=5, transform=ax.transAxes, ha="right")
style_axes(ax, categorical_y=True)
panel_label(ax, "d", x=-0.35)
caps = [
    f"Which cortical (and CA1) region pairs have grey-screen co-fluctuation data: number of mice with both regions "
    f"recorded ($\\geq$5 neurons each) in the same session with $\\geq$120 still 1 s bins. Of 319 sessions (116 mice) "
    f"with enough still bins, 308 (114 mice) yield region signals and {J['n_sessions']} ({J['n_mice']} mice) record at "
    f"least one such pair. Orange crosses: at least one session records the pair on two different probes. Boxes: "
    "the two modules of Lu et al. (prefrontal-temporal, parietal) and the regions only in our DMN (ILA, ECT, PERI; "
    "no pair among them is recorded, so that box is empty); dashed green: the full Lu et al. network. Tick colours as in Fig. 3.",
    "Number of DMN regions recorded per session, by definition.",
    "Number of cortical regions recorded per session; most sessions record none or one.",
    "Edges (region pairs) within each DMN definition: possible (light grey), recorded at least once (light blue), "
    f"recorded in $\\geq$3 mice (blue) and recorded on two different probes in $\\geq$3 mice (orange), top to bottom. The hub edge of Lu et al., prefrontal (ACA/PL/ORB) to RSP, is recorded "
    f"in {J['lu_hub_pfc_rsp']['n_sessions']} sessions.",
]
save_with_caption(fig, OUTD / "fig4_coverage", caps, label="grey_coverage", number=4)
png(OUTD / "fig4_coverage")

# ======================================================================= Fig 5: BWM coupling
J5 = json.load(open(BWM / f"41_spont_network_spont_motion_0.5s_still.json"))
fig = plt.figure(figsize=(NATURE_DOUBLE_COL_IN, 3.6))
gs = fig.add_gridspec(2, 3, width_ratios=[1.3, 1.1, 0.9], hspace=0.7, wspace=0.85)
ax = fig.add_subplot(gs[:, 0])
sets = [("ours_core", "our core"), ("ours_core+lateral", "our core+lat."), ("lu_all", "Lu all"),
        ("lu_pfc_temporal", "Lu pf-temporal"), ("lu_parietal", "Lu parietal")]
yl, yt = [], []
yy = 0
for k, lab in sets:
    for scope, sl, mk in (("cortex", "all pairs", "o"), ("cortex_cross_probe", "cross-probe", "s")):
        for jj, (JJ, col, off) in enumerate(((J, "k", -.12), (J5, C_OTH, .12))):
            v = JJ["classes"][f"{k}|{scope}"].get("dmn-dmn minus dmn-other")
            if v is None:
                continue
            ax.plot([v[1], v[2]], [yy + off] * 2, color=col, lw=0.9)
            ax.plot(v[0], yy + off, mk, color=col, ms=2.8)
            dt = JJ["classes"][f"{k}|{scope}"].get("drop_top_pair")
            if dt is not None and jj == 0:
                ax.plot(dt, yy + off, "|", color=C_OURS, ms=5, mew=1.2)
        nm = JJ["classes"][f"{k}|{scope}"]["n_mice_dd"]
        yt.append(f"{lab}, {sl} ({J['classes'][f'{k}|{scope}']['n_mice_dd']})")
        yl.append(yy)
        yy += 1
    yy += 0.4
ax.axvline(0, color="k", lw=0.5, ls=":")
ax.set_yticks(yl)
ax.set_yticklabels(yt, fontsize=5)
ax.invert_yaxis()
ax.set_xlabel("DMN-DMN minus DMN-other\n(distance residual, z)")
style_axes(ax, categorical_y=True)
panel_label(ax, "a", x=-1.05, y=1.02)
# edges with >= 3 mice
EO = pd.read_csv(BWM / f"41_edges_ours_all_{BTAG}.csv")
EL = pd.read_csv(BWM / f"41_edges_lu_{BTAG}.csv")
EE = pd.concat([EO, EL]).drop_duplicates(["r1", "r2"])
EE = EE[EE.n_mice >= 3].sort_values("mouse_mean")
ax = fig.add_subplot(gs[:, 1])
for i, (_, e) in enumerate(EE.iterrows()):
    a1, b1 = e.r1 in DMN_ALL and e.r2 in DMN_ALL, e.r1 in LU_DMN and e.r2 in LU_DMN
    col = C_BOTH if a1 and b1 else C_OURS if a1 else C_LU
    ax.plot([e.ci_lo, e.ci_hi], [i, i], color=col, lw=0.9)
    ax.plot(e.mouse_mean, i, "o", color=col, ms=2.8)
ax.axvline(0, color="k", lw=0.5, ls=":")
ax.set_yticks(range(len(EE)))
ax.set_yticklabels([f"{a}–{b} ({m}, {d:.1f} mm)" for a, b, m, d in zip(EE.r1, EE.r2, EE.n_mice, EE.dist_mm / 1)],
                   fontsize=5)
ax.set_xlabel("residual coupling (z)")
style_axes(ax, categorical_y=True)
panel_label(ax, "b", x=-1.0, y=1.02)
# CV
ax = fig.add_subplot(gs[0, 2])
cv = J["cv_restpref"]["resid_means"]
cq = J["cv_restpref"]["resid_q025_q975"]
kk = ["rest-rest", "rest-task", "task-task"]
ax.bar(range(3), [cv[k] for k in kk], color=[C_BOTH, "#86b6ef", C_OTH])
ax.errorbar(range(3), [cv[k] for k in kk], yerr=[[cv[k] - cq[k][0] for k in kk], [cq[k][1] - cv[k] for k in kk]],
            fmt="none", color="k", lw=0.7)
ax.set_xticks(range(3))
ax.set_xticklabels(["rest–rest", "rest–task", "task–task"], rotation=30, ha="right")
ax.set_ylabel("residual coupling (z)")
ax.axhline(0, color="k", lw=0.5)
style_axes(ax, categorical_x=True)
panel_label(ax, "c", x=-0.6)
ax = fig.add_subplot(gs[1, 2])
W = J["within"]
ks = [k for k, _ in sets if k in W and W[k]["n_mice"] >= 5]
ax.bar(range(len(ks)), [W[k]["mean"] for k in ks], color=[C_BOTH if k.startswith("ours") else C_LU for k in ks])
for i, k in enumerate(ks):
    ax.text(i, max(W[k]["mean"], 0) + .01, f"{W[k]['n_mice']}", ha="center", fontsize=5)
ax.set_xticks(range(len(ks)))
ax.set_xticklabels([dict(sets)[k] for k in ks], rotation=30, ha="right", fontsize=5)
ax.set_ylabel("within session,\nDMN-DMN $-$ DMN-other")
ax.axhline(0, color="k", lw=0.5)
style_axes(ax, categorical_x=True)
panel_label(ax, "d", x=-0.6)
caps = [
    "Grey-screen co-fluctuation of region signals (still bins only; wheel, lick and motion-energy regression), as "
    "residual from pairs at the same distance and probe relation. Difference DMN-DMN minus DMN-other cortical pairs "
    "for each region set, all pairs (circles) and pairs on different probes only (squares), 1 s (black) and 0.5 s "
    "(grey) bins; 95\\% mouse-bootstrap CI; number of mice with a DMN-DMN pair in brackets. Orange tick: the 1 s "
    "estimate after dropping the single most coupled DMN-DMN pair. The distance baseline uses cortical and CA1 pairs only. "
    "With about 20 such contrasts, CIs that just exclude 0 would not survive correction.",
    "Every DMN edge (either definition) recorded in $\\geq$3 mice: mean over mice of the residual coupling, 95\\% "
    "bootstrap CI; brackets give the number of mice and the median distance between the two regions. Blue: edge in both definitions; orange: our definition only; green: Lu et "
    "al. only.",
    "Cross-validated rest-preference network: cortical regions ranked by grey-screen rest preference (RI) in one half of the "
    "mice (terciles), coupling measured in the other half; mean and 2.5--97.5\\% range over 400 half-splits; rest--rest exceeds rest--task in 52\\% of splits.",
    "Within-session contrast DMN-DMN minus DMN-other, mean over mice (number of mice above bars; Lu parietal has only "
    "2 mice and is not shown).",
]
save_with_caption(fig, OUTD / "fig5_coupling", caps, label="grey_coupling", number=5)
png(OUTD / "fig5_coupling")

# ======================================================================= Fig 6: mesoscope grey screen
MJ = json.load(open(MESO / "33_spont_meso_me.json"))
MJA = json.load(open(MESO / "33_spont_meso.json"))
MP = pd.read_parquet(MESO / "33_spont_pairs_me.pqt")
fig = plt.figure(figsize=(NATURE_DOUBLE_COL_IN, 3.0))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.25, 1.0], wspace=0.7)
ax = fig.add_subplot(gs[0, 0])
for i, (k, lab, mem) in enumerate((("ours_core", "our DMN core", DMN_CORE), ("lu_all", "Lu et al. all", LU_DMN))):
    d1, d2 = MP.r1.isin(mem), MP.r2.isin(mem)
    cl = np.where(d1 & d2, "dmn-dmn", np.where(d1 | d2, "dmn-other", "other-other"))
    mm = MP.assign(cl=cl).groupby(["subject", "cl"]).resid.mean().unstack()
    for j, c in enumerate(["other-other", "dmn-other", "dmn-dmn"]):
        if c not in mm:
            continue
        v = mm[c].dropna().values
        x0 = i * 3.6 + j
        col = (C_OTH, "#86b6ef" if i == 0 else "#8fd6b8", C_BOTH if i == 0 else C_LU)[j]
        ax.scatter(x0 + rng.uniform(-.15, .15, len(v)), v, s=5, color=col, lw=0)
        m, lo, hi = mboot_mean(v)
        ax.errorbar(x0 + .3, m, yerr=[[m - lo], [hi - m]], fmt="o", color="k", ms=2, lw=.7)
        ax.text(x0, -0.56, f"{len(v)}", ha="center", fontsize=5)
    ax.text(i * 3.6 + 1.1, 0.62, lab, ha="center", fontsize=6)
ax.axhline(0, color="k", lw=.4, ls=":")
ax.set_xlim(-0.6, 6.2)
ax.set_xticks([0, 1, 2, 3.6, 4.6, 5.6])
ax.set_xticklabels(["oth–oth", "DMN–oth", "DMN–DMN"] * 2, rotation=45, ha="right", fontsize=5)
ax.set_ylim(-0.6, 0.7)
ax.set_ylabel("residual coupling (z)")
style_axes(ax, categorical_x=True)
panel_label(ax, "a", x=-0.3)
ME = pd.read_csv(MESO / "33_spont_edges_me.csv")
regs = [r for r in ["RSPagl", "RSPd", "VISam", "VISa", "VISpm", "VISrl", "TEa", "AUDp", "AUDd", "AUDpo", "VISp", "VISl",
                    "VISal", "VISli", "SSp-bfd", "SSp-tr", "SSp-ll", "SSp-ul", "SSp-un", "SSp-n", "SSp-m", "SSs", "MOp",
                    "MOs"] if r in set(ME.r1) | set(ME.r2)]
ix = {r: i for i, r in enumerate(regs)}
A = np.full((len(regs), len(regs)), np.nan)
ME2 = ME[ME.n_mice >= 2]
regs = [r for r in regs if r in set(ME2.r1) | set(ME2.r2)]
ix = {r: i for i, r in enumerate(regs)}
A = np.full((len(regs), len(regs)), np.nan)
for _, e in ME2.iterrows():
    if e.r1 in ix and e.r2 in ix:
        A[ix[e.r1], ix[e.r2]] = A[ix[e.r2], ix[e.r1]] = e.resid
ax = fig.add_subplot(gs[0, 1])
lim = np.nanmax(np.abs(A))
cmap6 = plt.get_cmap("RdBu_r").copy()
cmap6.set_bad("#d9d7cf")
im = ax.imshow(np.ma.masked_invalid(A), cmap=cmap6, vmin=-lim, vmax=lim, interpolation="nearest")
ax.set_xticks(range(len(regs)))
ax.set_yticks(range(len(regs)))
ax.set_xticklabels(regs, rotation=90, fontsize=4.5)
ax.set_yticklabels(regs, fontsize=4.5)
for tl in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
    r = tl.get_text()
    tl.set_color(rcol(r) if rcol(r) != C_OTH else "k")
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
cb.set_label("residual coupling (z)", fontsize=6)
cb.ax.tick_params(labelsize=5)
panel_label(ax, "b", x=-0.25, y=1.01)
ax = fig.add_subplot(gs[0, 2])
rr = []
for r in regs:
    v = ME[(ME.r1 == r) | (ME.r2 == r)]
    v = v[v.n_mice >= 2]
    if len(v) >= 3:
        rr.append((r, v.resid.mean()))
rr = sorted(rr, key=lambda x: x[1])
ax.barh(range(len(rr)), [x[1] for x in rr], color=[rcol(x[0]) for x in rr], height=.7)
ax.set_yticks(range(len(rr)))
ax.set_yticklabels([x[0] for x in rr], fontsize=5)
ax.axvline(0, color="k", lw=.5)
ax.set_xlabel("mean residual coupling\nwith all other regions (z)")
style_axes(ax, categorical_y=True)
panel_label(ax, "c", x=-0.45)
caps = [
    f"Mesoscope grey-screen period: sessions with $\\geq$200 s (most canonical sessions have only 60 s) and motion "
    f"energy to verify stillness, {MJ['n_sessions_with_pairs']} sessions, {MJ['n_mice_with_pairs']} mice "
    f"(all {MJA['n_sessions_long_spont']} long sessions, without the motion-energy requirement, give the same ordering). Region-averaged calcium activity in "
    "still 1 s bins with nuisance regression; coupling as residual from the distance-matched mean; dots are mice, black "
    "mean and 95\\% mouse-bootstrap CI; numbers at the bottom give the mice per class (DMN-DMN pairs come from 2 mice, "
    "mostly adjacent VISa-VISam). No frontal DMN region is imaged in these sessions; the wheel is logged in only 8 of "
    "the 23 long sessions.",
    "Residual coupling of every imaged region pair present in $\\geq$2 mice (grey: pair not imaged in $\\geq$2 mice; regions without any such pair omitted), ordered posterior DMN, Lu temporal "
    "(TEa, AUD), visual, somatosensory, motor; tick colours as in Fig. 3.",
    "Mean residual coupling of each region with all other imaged regions (pairs in $\\geq$2 mice). VISa and VISam are "
    "the most coupled regions together with trunk somatosensory cortex (SSp-tr); RSP has too few pairs in this subset; "
    "the auditory areas and TEa (Lu et al. temporal module) are among the least coupled.",
]
save_with_caption(fig, OUTD / "fig6_meso", caps, label="grey_meso", number=6)
png(OUTD / "fig6_meso")
print("done", ex)
