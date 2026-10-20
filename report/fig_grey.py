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
from common import beryl, CONS_DMN, GRANDJEAN_DMN, MANDINO_DMN, DMN_ALL, DMN_CORE, DMN_LATERAL, LU_DMN, LU_PARIETAL, LU_PFC_TEMPORAL, WHITESELL_DMN  # noqa: E402

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


LIT_DMN = sorted(set(WHITESELL_DMN) | set(LU_DMN) | set(GRANDJEAN_DMN) | set(MANDINO_DMN))


def rcol(r):
    return C_BOTH if r in CONS_DMN else C_OURS if r in LIT_DMN else C_OTH


def legend_sets(ax, x=0.99, y=0.97, dy=0.12):
    for i, (lab, c) in enumerate((("conservative DMN core", C_BOTH), ("other literature DMN", C_OURS),
                                  ("not DMN in any paper", C_OTH))):
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
fig = plt.figure(figsize=(NATURE_DOUBLE_COL_IN, 7.2))
gs = fig.add_gridspec(4, 4, height_ratios=[0.4, 1, 1.25, 2.9], hspace=0.55, wspace=0.55)
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
# e: durations of uninterrupted still bouts on the grey screen (same stillness definition as d), all sessions
bouts = []
for eid_ in H.eid:
    s_ = dict(np.load(BWM / "full" / f"{eid_}.npz", allow_pickle=True))
    B_ = float(s_["BIN"])
    a_, b_ = int(s_["spont"][0] / B_) + 20, min(len(s_["licks"]), int(s_["spont"][1] / B_))
    w_, lk_, me_ = (s_[k].astype(float) for k in ("wheel_speed", "licks", "motion_energy"))
    mv = (np.isfinite(w_) & (w_ > 0.05)) | ((lk_ > 0) & (lk_ < 255))
    if np.isfinite(me_).mean() > .5:
        mv |= np.isfinite(me_) & (me_ > np.nanpercentile(me_, 60))
    bad_ = mv | np.r_[False, mv[:-1]] | np.r_[mv[1:], False]
    st_ = (~bad_[a_:b_]).astype(int)
    dd = np.diff(np.r_[0, st_, 0])
    bouts.append((np.flatnonzero(dd == -1) - np.flatnonzero(dd == 1)) * B_)
bouts = np.concatenate(bouts)
ax = fig.add_subplot(gs[1, 3])
# log-spaced bins snapped to the 50 ms grid, so every bin contains at least one possible duration
edges_ = np.unique(np.round(np.logspace(np.log10(0.05), np.log10(120), 40) / 0.05)) * 0.05 - 0.025
ax.hist(bouts, bins=edges_, color=C_GREY)
ax.set_xscale("log")
ax.set_xticks([0.1, 1, 10, 100])
ax.set_xticklabels(["0.1", "1", "10", "100"])
ax.set_xlabel("still bout duration (s)")
ax.set_ylabel("bouts")
style_axes(ax, categorical_x=True)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k" if v >= 1000 else f"{v:.0f}"))
f_ge1 = bouts[bouts >= 1].sum() / bouts.sum()
f_ge5 = bouts[bouts >= 5].sum() / bouts.sum()
panel_label(ax, "e", x=-0.3)

# example session: most conservative-core DMN neurons (3 RSP subregions) among sessions with full wheel and camera
# coverage, 600 s, and median stillness (0.70 vs median 0.69); no session records frontal and RSP core together
ex = "49e0ab27-827a-4c91-bcaa-97eea27a1b8d"
s = dict(np.load(BWM / "full" / f"{ex}.npz", allow_pickle=True))
BIN = float(s["BIN"])
sp = s["spont"]
a, b = int(sp[0] / BIN), int(sp[1] / BIN)
t = (np.arange(a, b) * BIN - sp[0]) / 60
w, me, lk = (s[k][a:b].astype(float) for k in ("wheel_speed", "motion_energy", "licks"))
thr = np.nanpercentile(s["motion_energy"].astype(float), 60)
mov = (w > 0.05) | ((lk > 0) & (lk < 255)) | (me > thr)
mov = mov | np.r_[False, mov[:-1]] | np.r_[mov[1:], False]
# raster: 20 neurons with the highest firing rate in the window, 0.5 s bins
RB = 10  # 50 ms bins per raster bin
Xs, regs_ = [], []
for f in sorted((BWM / "full").glob(f"{ex}_probe*.npz")):
    p_ = np.load(f, allow_pickle=True)
    Xs.append(p_["counts"][:, a:b].astype(np.float32))
    regs_.append(np.char.add(beryl(p_["atlas_id"]).astype(str), np.full(len(p_["atlas_id"]), f" ({str(p_['probe'])[-2:]})")))
Xs, regs_ = np.concatenate(Xs), np.concatenate(regs_)
rates = Xs.sum(1) / ((b - a) * BIN)
in_core = np.array([r.split(" ")[0] in CONS_DMN for r in regs_])
top = np.flatnonzero(in_core)[np.argsort(rates[in_core])[::-1][:20]]
n_core_ex = int(in_core.sum())
nr = (b - a) // RB
Rr = Xs[top, : nr * RB].reshape(len(top), nr, RB).sum(2) / (RB * BIN)
lo_, hi_ = np.percentile(Rr, 1, axis=1, keepdims=True), np.percentile(Rr, 99, axis=1, keepdims=True)
Rr = (Rr - lo_) / (hi_ - lo_).clip(1e-9)
n_neurons_ex = len(rates)
# 10 s zoom window: still fraction closest to 0.5 with some wheel movement, after the first 30 s
ZW = 200  # 10 s of 50 ms bins
best, zs = 9, None
for st_ in range(600, len(t) - ZW, 20):
    sf = (~mov[st_:st_ + ZW]).mean()
    if (w[st_:st_ + ZW] > 0.05).sum() >= 4 and abs(sf - 0.5) < best:
        best, zs = abs(sf - 0.5), st_
z0, z1 = t[zs], t[zs + ZW - 1]
axes_e = []
sub = gs[2, :].subgridspec(3, 1, hspace=0.15, height_ratios=[1, 1, 0.45])
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
        ax.spines["left"].set_visible(False)
        ax.set_xlabel("time in grey-screen period (min)")
    ax.set_xlim(0, t[-1])
    axes_e.append(ax)
    if i == 0:
        panel_label(ax, "f", x=-0.085, y=1.0)
for ax_ in axes_e:
    ax_.axvspan(z0, z1, color=C_OURS, alpha=0.25, lw=0)
# f: 10 s zoom at the native 50 ms resolution
tz = (np.arange(zs, zs + ZW) * BIN)  # seconds from grey-screen start (t is in minutes from a)
tz = tz - tz[0] + t[zs] * 60
subf = gs[3, :].subgridspec(4, 1, hspace=0.15, height_ratios=[1, 1, 0.45, 4.2])
for i, (y, lab) in enumerate(((w[zs:zs + ZW], "wheel"), (me[zs:zs + ZW] / np.nanmedian(s["motion_energy"]), "motion\nenergy"),
                              (None, "still"))):
    ax = fig.add_subplot(subf[i])
    if y is not None:
        ax.plot(tz, y, lw=0.8, color="k", drawstyle="steps-mid")
        if i == 0:
            ax.axhline(0.05, color=C_OURS, lw=0.6, ls="--")
        if i == 1:
            ax.axhline(thr / np.nanmedian(s["motion_energy"]), color=C_OURS, lw=0.6, ls="--")
        style_axes(ax, max_yticks=2)
        ax.set_ylabel(lab, fontsize=6, rotation=0, ha="right", va="center")
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
    else:
        ax.fill_between(tz, 0, (~mov[zs:zs + ZW]).astype(float), step="mid", color=C_GREY, lw=0)
        style_axes(ax)
        ax.set_yticks([])
        ax.set_ylabel(lab, fontsize=6, rotation=0, ha="right", va="center")
        ax.set_xticks([])
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
    ax.set_xlim(tz[0] - BIN / 2, tz[-1] + BIN / 2)
    if i == 0:
        panel_label(ax, "g", x=-0.085, y=1.0)
ax = fig.add_subplot(subf[3])
Z = Xs[top, zs:zs + ZW] / BIN
zhi = np.percentile(Xs[top] / BIN, 99, axis=1, keepdims=True).clip(1e-9)
ax.imshow(np.clip(Z / zhi, 0, 1), aspect="auto", cmap="Greys", interpolation="nearest",
          extent=[tz[0] - BIN / 2, tz[-1] + BIN / 2, len(top) - 0.5, -0.5])
ax.set_yticks(range(len(top)))
ax.set_yticklabels([f"{regs_[i]} {rates[i]:.0f} Hz" for i in top], fontsize=4.2)
style_axes(ax, categorical_y=True)
ax.set_xlabel("time in grey-screen period (s)")
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
    f"Durations of uninterrupted still bouts on the grey screen (same stillness definition as d), all sessions pooled "
    f"({len(bouts):,} bouts, log axis): median {np.median(bouts):.2f} s; {f_ge1 * 100:.0f}\\% of still time lies in bouts "
    f"of $\\geq$1 s and {f_ge5 * 100:.0f}\\% in bouts of $\\geq$5 s.",
    f"Example session with neurons in the conservative DMN core (CSH\\_ZAD\\_019, {n_core_ex} of {n_neurons_ex} good units "
    "in RSPagl, RSPd and RSPv; still fraction 0.70, close to the median): wheel speed, left-camera motion energy "
    "(normalised to the session median; dashed, the stillness threshold; wheel in rad/s) and the resulting still bins. "
    "Orange band: the 10 s shown in g.",
    "Zoom into 10 s of f at the native 50 ms resolution: wheel speed (dashed, the 0.05 rad/s movement threshold), motion "
    "energy (dashed, stillness threshold), still bins and spike counts per 50 ms bin of the 20 most active core-DMN neurons of the session (each row "
    "scaled to its 99th percentile over the whole period, black = high; labels give region, probe and mean rate). The window was chosen automatically as the one after the "
    "first 30 s whose still fraction is closest to 0.5 and that contains wheel movement.",
]
save_with_caption(fig, OUTD / "fig2_protocol", caps, label="grey_protocol", number=2)
png(OUTD / "fig2_protocol")

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
save_with_caption(fig, OUTD / "fig3_movement", caps, label="grey_movement", number=3)
png(OUTD / "fig3_movement")

# ======================================================================= Fig 4: rest preference vs grey screen, core only
G = json.load(open(BWM / "42_spont_A_G_full.json"))
GM = json.load(open(BWM / "42_spont_A_G_me.json"))
RG = pd.read_csv(BWM / "42_regions_G_full.csv", index_col=0)
NG = pd.read_parquet(BWM / "42_neurons_G_full.pqt")
NG = NG[NG.cosmos == "Isocortex"]
fig = plt.figure(figsize=(NATURE_DOUBLE_COL_IN, 2.5))
gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.5, 1.05, 0.85], wspace=0.85)
for k, (col, scale, lab) in enumerate((("suppressed_grey", 100, "task-suppressed\nneurons (%)"), ("RI_grey", 1, "rest index RI"))):
    ax = fig.add_subplot(gs[0, k])
    oth = NG[~NG.region.isin(CONS_DMN)]
    mo, lo_o, hi_o = mboot_mean(oth.groupby("subject")[col].mean().values * scale)
    ax.axhspan(lo_o, hi_o, color=C_OTH, alpha=0.3, lw=0)
    ax.axhline(mo, color=C_OTH, lw=0.8)
    for i, r in enumerate(CONS_DMN):
        v = NG[NG.region == r].groupby("subject")[col].mean().dropna().values * scale
        m, lo, hi = mboot_mean(v)
        ax.plot([i, i], [lo, hi], color=C_BOTH, lw=0.9)
        ax.plot(i, m, "o", color=C_BOTH, ms=3)
        ax.text(i, ax.get_ylim()[0], "", fontsize=4)
    if col == "RI_grey":
        ax.axhline(0, color="k", lw=0.4, ls=":")
    ax.set_xticks(range(len(CONS_DMN)))
    ax.set_xticklabels(CONS_DMN, rotation=60, ha="right", fontsize=5)
    ax.set_ylabel(lab)
    style_axes(ax, categorical_x=True)
    panel_label(ax, "ab"[k], x=-0.45, y=1.02)
rows4 = [("all trials", G, "", "other_ctx", "k"), ("last third of trials", G, "_late", "other_ctx", C_GREY),
         ("vs higher-order cortex", G, "", "higher_order_ctx", C_OTH), ("motion-energy sessions", GM, "", "other_ctx", "#6250d6")]
for jj, (met, scale, xl) in enumerate((("suppressed_grey", 100, "suppressed (% points)"), ("RI_grey", 1, "rest index RI"))):
    ax = fig.add_subplot(gs[0, 3 + jj])
    for i, (lab, src, suf, oth, colr) in enumerate(rows4):
        v = src["tests"].get(f"{met}{suf}|cons|vs_{oth}")
        if v is None:
            continue
        lo, hi = v["subject_boot_ci95"]
        ax.plot([lo * scale, hi * scale], [i, i], color=colr, lw=1)
        ax.plot(v["diff"] * scale, i, "o", color=colr, ms=3, mfc=colr if v["p_perm"] < 0.05 else "white")
    ax.axvline(0, color="k", lw=0.5, ls=":")
    ax.set_yticks(range(len(rows4)))
    ax.set_yticklabels([r[0] for r in rows4] if jj == 0 else [], fontsize=5.5)
    ax.invert_yaxis()
    ax.set_xlabel(f"core $-$ other cortex,\n{xl}")
    style_axes(ax, categorical_y=True)
    panel_label(ax, "cd"[jj], x=-0.75 if jj == 0 else -0.1, y=1.02)
caps = [
    f"Percentage of neurons in each core region significantly suppressed during the task (movement onset, movement "
    f"and feedback windows) relative to still 0.3 s chunks of the grey screen (Welch test with one chunk per 3 s, "
    f"p $<$ 0.01): mean over mice with 95\\% bootstrap CI; grey line and band, all other cortex ({G['n_neurons']:,} "
    f"neurons, {G['n_sessions']} sessions, {G['n_subjects']} mice in total).",
    "The continuous rest index RI = (rest $-$ task)/(rest + task) for the same neurons.",
    "Core minus other cortex in the suppressed fraction (regions averaged): against all trials, against only the last "
    "third of trials (closest in time to the grey screen), against higher-order cortex only, and in the 361 sessions "
    "with motion energy; 95\\% mouse-bootstrap CI; filled markers, region-permutation p $<$ 0.05.",
    "The same for the rest index. 81\\% of all neurons differ significantly between grey screen and task, so the "
    "suppressed fraction is close to the sign of RI.",
]
save_with_caption(fig, OUTD / "fig4_restpref", caps, label="grey_restpref", number=4)
png(OUTD / "fig4_restpref")

# ======================================================================= Fig 4: coverage
P = pd.read_parquet(BWM / f"21_pairs_{BTAG}.pqt")
P["subject"] = P.eid.map(SUBJ)
M = pd.read_parquet(BWM / f"21_regions_{BTAG}.pqt")
keep = (P.c1.eq("Isocortex") | P.r1.eq("CA1")) & (P.c2.eq("Isocortex") | P.r2.eq("CA1"))
Q = P[keep]
# a: core region pairs only (7 x 7): mice with the pair, of which on two different probes
order = list(CONS_DMN)
idx = {r: i for i, r in enumerate(order)}
n = len(order)
MM = np.zeros((n, n))
XX = np.zeros((n, n))
for (r1, r2), g in Q[Q.r1.isin(order) & Q.r2.isin(order)].groupby(["r1", "r2"]):
    i, j = idx[r1], idx[r2]
    MM[i, j] = MM[j, i] = g.subject.nunique()
    XX[i, j] = XX[j, i] = g[~g.same_probe].subject.nunique()
fig = plt.figure(figsize=(NATURE_DOUBLE_COL_IN, 2.5))
gs = fig.add_gridspec(1, 4, width_ratios=[1.25, 1, 1, 1.25], wspace=0.75)
ax = fig.add_subplot(gs[0, 0])
cmW7 = plt.get_cmap("Blues").copy()
cmW7.set_bad("#f0efea")
Mshow = np.where(MM > 0, MM, np.nan)
np.fill_diagonal(Mshow, np.nan)
im = ax.imshow(np.ma.masked_invalid(Mshow), cmap=cmW7, vmin=0, vmax=max(8, np.nanmax(Mshow)), interpolation="nearest")
for i in range(n):
    for j in range(n):
        if i != j and MM[i, j] > 0:
            ax.text(j, i, f"{MM[i, j]:.0f}" + (f"\n({XX[i, j]:.0f})" if XX[i, j] else ""), ha="center", va="center",
                    fontsize=4.3, color="white" if MM[i, j] >= 5 else "k", linespacing=0.9)
ax.add_patch(Rectangle((-.5, -.5), 4, 4, fill=False, ec="k", lw=0.6))
ax.add_patch(Rectangle((3.5, 3.5), 3, 3, fill=False, ec="k", lw=0.6))
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(order, rotation=90, fontsize=5.5)
ax.set_yticklabels(order, fontsize=5.5)
ax.tick_params(length=0)
for sp_ in ax.spines.values():
    sp_.set_visible(False)
panel_label(ax, "a", x=-0.45, y=1.02)
ax = fig.add_subplot(gs[0, 1])
k_ours = M[M.region.isin(CONS_DMN)].groupby("eid").region.nunique().reindex(M.eid.unique(), fill_value=0)
k_lu = M[M.region.isin(LU_DMN)].groupby("eid").region.nunique().reindex(M.eid.unique(), fill_value=0)
xs = np.arange(0, 6)
ax.bar(xs, [(k_ours == x).sum() for x in xs], width=.7, color=C_BOTH)
ax.set_xlabel("core regions per session")
ax.set_ylabel("sessions")
style_axes(ax)
ax.set_xticks(range(0, 6))
panel_label(ax, "b", x=-0.4)
ax = fig.add_subplot(gs[0, 2])
kc = M[M.cosmos == "Isocortex"].groupby("eid").region.nunique().reindex(M.eid.unique(), fill_value=0)
ax.bar(range(0, 8), [(kc == x).sum() for x in range(8)], color=C_OTH)
ax.set_xlabel("cortical regions\nper session")
ax.set_ylabel("sessions")
style_axes(ax)
panel_label(ax, "c", x=-0.4)
ax = fig.add_subplot(gs[0, 3])
J = json.load(open(BWM / f"41_spont_network_{BTAG}.json"))
labels = ["conservative core"]
keys = ["cons"]
poss = [J["coverage"][k]["n_possible_edges"] for k in keys]
anyp = [J["coverage"][k]["n_edges_any"] for k in keys]
ge3 = [J["coverage"][k]["n_edges_ge3_mice"] for k in keys]
y = np.arange(1)
cx3 = [J["coverage"][k]["n_edges_cross_probe_ge3_mice"] for k in keys]
for j, (vals, col) in enumerate(((poss, "#cfccc2"), (anyp, "#86b6ef"), (ge3, C_BOTH), (cx3, C_OURS))):
    ax.barh(y - 0.3 + j * 0.2, vals, color=col, height=0.18)
    for i in range(len(keys)):
        ax.text(vals[i] + 0.4, y[i] - 0.3 + j * 0.2, str(vals[i]), va="center", fontsize=5)
ax.set_yticks(y - 0.3 + np.arange(4) * 0.2 if False else [])
for j, lab in enumerate(["possible", "recorded", "$\\geq$3 mice", "$\\geq$3 mice, 2 probes"]):
    ax.text(-0.6, -0.3 + j * 0.2, lab, ha="right", va="center", fontsize=5.5)
ax.invert_yaxis()
ax.set_xlabel("core region pairs")
ax.set_xlim(0, max(poss) * 1.25)
style_axes(ax, categorical_y=True)
panel_label(ax, "d", x=-0.35)
caps = [
    "Core DMN region pairs with grey-screen co-fluctuation data: number of mice with both regions recorded ($\\geq$5 "
    "neurons each) in the same session with $\\geq$120 still 1 s bins; in brackets, mice in which the pair is recorded on "
    "two different probes; grey, never recorded. Boxes: frontal core (ACA, PL, ORBm) and RSP. No frontal-RSP pair is "
    "ever recorded.",
    "Number of core regions recorded per session, out of 308 sessions with region signals.",
    "Number of cortical regions recorded per session; most sessions record none or one.",
    "Core region pairs: possible, recorded at least once, recorded in $\\geq$3 mice and recorded on two different "
    f"probes in $\\geq$3 mice. The frontal-to-RSP edge at the centre of every DMN map is recorded in "
    f"{J['lu_hub_pfc_rsp']['n_sessions']} sessions.",
]
save_with_caption(fig, OUTD / "fig7_coverage", caps, label="grey_coverage", number=7)
png(OUTD / "fig7_coverage")

# ======================================================================= Fig 8: BWM coupling, core only
J5 = json.load(open(BWM / f"41_spont_network_spont_motion_0.5s_still.json"))
# residual coupling per pair as in 41_spont_network.py (cortex + CA1 baseline, 0.5 mm distance bins x probe relation)
P8 = pd.read_parquet(BWM / f"21_pairs_{BTAG}.pqt")
P8["subject"] = P8.eid.map(SUBJ)
ctxca1 = (P8.c1.eq("Isocortex") | P8.r1.eq("CA1")) & (P8.c2.eq("Isocortex") | P8.r2.eq("CA1"))
P8 = P8[ctxca1].copy()
P8["dbin"] = pd.cut(P8.dist_mm, [0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 20], labels=False)
P8["resid"] = P8.z - P8.groupby(["dbin", "same_probe"]).z.transform("mean")
P8 = P8[(P8.c1 == "Isocortex") & (P8.c2 == "Isocortex")]
d1, d2 = P8.r1.isin(CONS_DMN), P8.r2.isin(CONS_DMN)
P8["cl"] = np.where(d1 & d2, "core–core", np.where(d1 | d2, "core–other", "other–other"))
fig = plt.figure(figsize=(NATURE_DOUBLE_COL_IN, 2.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 0.9, 1.0], wspace=1.25)
ax = fig.add_subplot(gs[0, 0])
rows8 = [("dmn-dmn minus dmn-other", "core–core − core–other"), ("dmn-dmn minus other-other", "core–core − other–other"),
         ("dmn-other minus other-other", "core–other − other–other")]
for i, (k, lab) in enumerate(rows8):
    for JJ, col, off in ((J, "k", -0.12), (J5, C_OTH, 0.12)):
        v = JJ["classes"]["cons|cortex"].get(k)
        if v:
            ax.plot([v[1], v[2]], [i + off] * 2, color=col, lw=1)
            ax.plot(v[0], i + off, "o", color=col, ms=2.8)
ax.axvline(0, color="k", lw=0.5, ls=":")
ax.set_yticks(range(len(rows8)))
ax.set_yticklabels([r[1] for r in rows8], fontsize=5.5)
ax.invert_yaxis()
ax.set_xlabel("difference in residual\ncoupling (z)")
ax.text(1.0, 1.1, "1 s bins", color="k", fontsize=5.5, transform=ax.transAxes, ha="right")
ax.text(1.0, 1.03, "0.5 s bins", color=C_OTH, fontsize=5.5, transform=ax.transAxes, ha="right")
style_axes(ax, categorical_y=True)
panel_label(ax, "a", x=-1.0, y=1.02)
ax = fig.add_subplot(gs[0, 1])
mm = P8.groupby(["subject", "cl"]).resid.mean().unstack()
for i, (c, colc) in enumerate((("other–other", C_OTH), ("core–other", "#86b6ef"), ("core–core", C_BOTH))):
    v = mm[c].dropna().values
    ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=5, color=colc, lw=0)
    m, lo, hi = mboot_mean(v)
    ax.errorbar(i + 0.3, m, yerr=[[m - lo], [hi - m]], fmt="o", color="k", ms=2, lw=0.7)
    ax.text(i, -0.95, f"{len(v)}", ha="center", fontsize=5)
ax.axhline(0, color="k", lw=0.4, ls=":")
ax.set_ylim(-1.0, 1.0)
ax.set_xticks(range(3))
ax.set_xticklabels(["other–other", "core–other", "core–core"], rotation=30, ha="right", fontsize=5.5)
ax.set_ylabel("residual coupling (z)")
style_axes(ax, categorical_x=True)
panel_label(ax, "b", x=-0.45, y=1.02)
EC = pd.read_csv(BWM / f"41_edges_cons_{BTAG}.csv").sort_values("mouse_mean")
ax = fig.add_subplot(gs[0, 2])
for i, (_, e) in enumerate(EC.iterrows()):
    if np.isfinite(e.ci_lo):
        ax.plot([e.ci_lo, e.ci_hi], [i, i], color=C_BOTH, lw=0.9)
    ax.plot(e.mouse_mean, i, "o", color=C_BOTH, ms=2.8, mfc=C_BOTH if np.isfinite(e.ci_lo) else "white")
ax.axvline(0, color="k", lw=0.5, ls=":")
ax.set_yticks(range(len(EC)))
ax.set_yticklabels([f"{a}–{b} ({m} mice, {d:.1f} mm)" for a, b, m, d in zip(EC.r1, EC.r2, EC.n_mice, EC.dist_mm)],
                   fontsize=5)
ax.set_xlabel("residual coupling (z)")
style_axes(ax, categorical_y=True)
panel_label(ax, "c", x=-1.05, y=1.02)
W8 = J["within"]["cons"]
caps = [
    "Grey-screen co-fluctuation of region signals for the conservative core (still bins only; wheel, lick and "
    "motion-energy regression), as residual from cortical and CA1 pairs at the same distance and probe relation. "
    "Differences between pair classes (cortical pairs only), 1 s (black) and 0.5 s (grey) bins, 95\\% mouse-bootstrap "
    f"CI ({J['classes']['cons|cortex']['n_mice_dd']} mice with a core--core pair). Only 4 core--core pairs come from "
    "two probes, too few to test separately.",
    "Residual coupling per pair class, averaged per mouse (dots; number of mice at the bottom), black: mean and 95\\% "
    "bootstrap CI.",
    "Every recorded core pair: mean residual coupling over mice, with 95\\% bootstrap CI where $\\geq$3 mice (open "
    "markers: fewer mice). Within session, core--core minus core--other coupling is "
    f"{W8['mean']:+.2f} ({W8['n_mice']} mice, Wilcoxon p = {W8['wilcoxon_p']:.2f}).",
]
save_with_caption(fig, OUTD / "fig8_coupling", caps, label="grey_coupling", number=8)
png(OUTD / "fig8_coupling")

# ======================================================================= Fig 9: mesoscope, core (RSP) only
msubj = pd.read_csv(MESO / "session_subjects.csv").drop_duplicates("session").set_index("session").subject
MI = pd.read_parquet(MESO / "30_pairs_motion_iti1.pqt")
MI["subject"] = MI.eid.map(msubj)
MI = MI[(MI.rel1 >= .5) & (MI.rel2 >= .5)].copy()
MI["z"] = np.arctanh(MI.r.clip(-.999, .999))
MI["dbin"] = pd.cut(MI.dist_mm, [0, .5, 1, 1.5, 2, 2.5, 3, 4, 10], labels=False)
MI["resid"] = MI.z - MI.groupby("dbin").z.transform("mean")
m1, m2 = MI.r1.isin(CONS_DMN), MI.r2.isin(CONS_DMN)
MI["cl"] = np.where(m1 & m2, "core–core", np.where(m1 | m2, "core–other", "other–other"))
RS = MI[MI.cl == "core–other"].copy()
RS["partner"] = np.where(RS.r1.isin(CONS_DMN), RS.r2, RS.r1)
fig = plt.figure(figsize=(NATURE_DOUBLE_COL_IN, 2.5))
gs = fig.add_gridspec(1, 2, width_ratios=[0.8, 1.6], wspace=0.45)
ax = fig.add_subplot(gs[0, 0])
mm = MI.groupby(["subject", "cl"]).resid.mean().unstack()
for i, (c, colc) in enumerate((("other–other", C_OTH), ("core–other", "#86b6ef"), ("core–core", C_BOTH))):
    v = mm[c].dropna().values if c in mm else np.array([])
    ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=5, color=colc, lw=0)
    if len(v) >= 3:
        m, lo, hi = mboot_mean(v)
        ax.errorbar(i + 0.3, m, yerr=[[m - lo], [hi - m]], fmt="o", color="k", ms=2, lw=0.7)
    ax.text(i, ax.get_ylim()[0] if False else -0.55, f"{len(v)}", ha="center", fontsize=5)
ax.axhline(0, color="k", lw=0.4, ls=":")
ax.set_ylim(-0.6, 0.6)
ax.set_xticks(range(3))
ax.set_xticklabels(["other–other", "RSP–other", "RSP–RSP"], rotation=30, ha="right", fontsize=5.5)
ax.set_ylabel("residual coupling (z)")
style_axes(ax, categorical_x=True)
panel_label(ax, "a", x=-0.45, y=1.02)
ax = fig.add_subplot(gs[0, 1])
pm = RS.groupby(["partner", "subject"]).resid.mean().groupby("partner").agg(["mean", "size"])
pm = pm[pm["size"] >= 2].sort_values("mean")
ax.bar(range(len(pm)), pm["mean"], color=[rcol(r) for r in pm.index], width=0.7)
for i, (r, row) in enumerate(pm.iterrows()):
    ax.text(i, row["mean"] + (0.01 if row["mean"] >= 0 else -0.01), f"{int(row['size'])}", ha="center",
            va="bottom" if row["mean"] >= 0 else "top", fontsize=4.5)
ax.axhline(0, color="k", lw=0.5)
ax.set_ylim(min(pm["mean"].min() * 1.25, -0.05), max(pm["mean"].max() * 1.25, 0.05))
ax.set_xticks(range(len(pm)))
ax.set_xticklabels(pm.index, rotation=60, ha="right", fontsize=5)
ax.set_ylabel("RSP coupling with region\n(residual z, mean over mice)")
style_axes(ax, categorical_x=True)
panel_label(ax, "b", x=-0.12, y=1.02)
caps = [
    f"Mesoscope, inter-trial rest (the grey screen images RSP in only 1--2 mice): still 1 s bins from 1 s after "
    f"feedback to 0.2 s before the next stimulus, nuisance-regressed; RSP pairs from {RS.eid.nunique()} sessions, "
    f"{RS.subject.nunique()} mice. The mesoscope images no frontal core region, so the core is RSPagl and RSPd. Residual "
    "coupling (distance-matched) per pair class, averaged per mouse (dots; number of mice at the bottom); black, mean "
    "and 95\\% bootstrap CI where $\\geq$3 mice.",
    "Mean residual coupling of RSP with each imaged region (pairs in $\\geq$2 mice; numbers above bars: mice), "
    "orange: regions assigned to the DMN by at least one paper (Fig. 1) but not in the core; grey: in none.",
]
save_with_caption(fig, OUTD / "fig9_meso", caps, label="grey_meso", number=9)
png(OUTD / "fig9_meso")

# ======================================================================= Fig 10: single-region metrics, core only
SR = json.load(open(BWM / "43_single_region.json"))
N43 = pd.read_parquet(BWM / "43_neurons.pqt")
S43 = pd.read_parquet(BWM / "43_region_sessions.pqt")
fig = plt.figure(figsize=(NATURE_DOUBLE_COL_IN, 2.4))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.15], wspace=0.75)
for k, (D_, col, lab) in enumerate(((N43, "fALFF_adj", "fALFF (rate-adj.)"), (N43, "CV_adj", "CV (rate-adj.)"),
                                     (S43, "sync_adj", "local synchrony\n(rate-adj.)"))):
    ax = fig.add_subplot(gs[0, k])
    oth = D_[(D_.cosmos == "Isocortex") & ~D_.region.isin(CONS_DMN)]
    mo, lo_o, hi_o = mboot_mean(oth.groupby("subject")[col].mean().values)
    ax.axhspan(lo_o, hi_o, color=C_OTH, alpha=0.3, lw=0)
    ax.axhline(mo, color=C_OTH, lw=0.8)
    for i, r in enumerate(CONS_DMN):
        v = D_[D_.region == r].groupby("subject")[col].mean().dropna().values
        if len(v) >= 2:
            m, lo, hi = mboot_mean(v)
            ax.plot([i, i], [lo, hi], color=C_BOTH, lw=0.9)
        elif len(v):
            m = v.mean()
        else:
            continue
        ax.plot(i, m, "o", color=C_BOTH, ms=3)
    ax.set_xticks(range(len(CONS_DMN)))
    ax.set_xticklabels(CONS_DMN, rotation=60, ha="right", fontsize=5)
    ax.set_ylabel(lab)
    style_axes(ax, categorical_x=True)
    panel_label(ax, "abc"[k], x=-0.45, y=1.02)
ax = fig.add_subplot(gs[0, 3])
for i, (m_, lab, sc) in enumerate((("fALFF_adj", "fALFF", 1), ("CV_adj", "CV × 0.1", 0.1), ("sync_adj", "synchrony", 1))):
    v = SR["tests"][f"{m_}|cons|vs_other_ctx"]
    lo, hi = v["subject_boot_ci95"]
    ax.plot([lo * sc, hi * sc], [i, i], color=C_BOTH, lw=1)
    ax.plot(v["diff"] * sc, i, "o", color=C_BOTH, ms=3, mfc=C_BOTH if v["p_perm"] < 0.05 else "white")
    ax.text(max(hi * sc, 0) + 0.003, i, f"p={v['p_perm']:.2f}", va="center", fontsize=5)
ax.axvline(0, color="k", lw=0.5, ls=":")
ax.set_yticks(range(3))
ax.set_yticklabels(["fALFF", "CV × 0.1", "synchrony"], fontsize=5.5)
ax.invert_yaxis()
ax.set_xlabel("core − other cortex")
style_axes(ax, categorical_y=True)
panel_label(ax, "d", x=-0.5, y=1.02)
caps = [
    f"Single-region resting metrics of the grey screen for each core region: fALFF per neuron (fraction of "
    f"0.01--0.1 Hz power in the 1 s binned rate after regressing wheel, licks and motion energy, residualised on log "
    f"firing rate), mean over mice with 95\\% bootstrap CI. Grey line and band: all other cortex.",
    "Fluctuation amplitude (coefficient of variation of the 1 s rate, rate-adjusted).",
    "Local synchrony (ReHo analogue): mean pairwise correlation between neurons of a region on one probe in still "
    "1 s bins, rate-adjusted.",
    "Core minus other cortex for the three metrics: 95\\% mouse-bootstrap CI and region-permutation p (open markers: "
    "p $\\geq$ 0.05).",
]
save_with_caption(fig, OUTD / "fig10_single_region", caps, label="grey_single", number=10)
png(OUTD / "fig10_single_region")

# ======================================================================= Fig S1: three DMN definitions, rest preference
GS = json.load(open(BWM / "42_spont_A_G_full.json"))
CS = RG[RG.cosmos == "Isocortex"].sort_values("suppressed_grey", ascending=False)
fig = plt.figure(figsize=(NATURE_DOUBLE_COL_IN, 3.9))
gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1], width_ratios=[1, 1], hspace=0.9, wspace=0.6)
sub = gs[0, :].subgridspec(2, 1, height_ratios=[1, 0.55], hspace=0.05)
ax = fig.add_subplot(sub[0])
ax.bar(np.arange(len(CS)), CS.suppressed_grey * 100, color=C_OTH, width=0.75)
ax.set_xlim(-0.7, len(CS) - 0.3)
ax.set_xticks([])
ax.set_ylabel("task-suppressed\nneurons (%)")
style_axes(ax, categorical_x=True)
ax.spines["bottom"].set_visible(False)
panel_label(ax, "a", x=-0.05)
ax = fig.add_subplot(sub[1])
defs = [("cons. core", CONS_DMN, C_BOTH), ("Whitesell", WHITESELL_DMN, "#6250d6"), ("Grandjean", GRANDJEAN_DMN, "#eda100"),
        ("Mandino", MANDINO_DMN, "#e34948"), ("Lu et al.", LU_DMN, C_LU)]
for i, (lab, mem, col) in enumerate(defs):
    for j, r in enumerate(CS.index):
        if r in mem:
            ax.add_patch(Rectangle((j - .4, i - .4), .8, .8, color=col, lw=0))
ax.set_xlim(-0.7, len(CS) - 0.3)
ax.set_ylim(len(defs) - .5, -.5)
ax.set_yticks(range(len(defs)))
ax.set_yticklabels([d[0] for d in defs], fontsize=5.5)
ax.set_xticks(np.arange(len(CS)))
ax.set_xticklabels(CS.index, rotation=60, ha="right", fontsize=5)
for sp_ in ("top", "right", "left", "bottom"):
    ax.spines[sp_].set_visible(False)
ax.tick_params(length=0)
rowsS = [("cons", "conservative core"), ("whitesell", "Whitesell et al."), ("lu_all", "Lu et al.")]
for jj, (met, scale, xl) in enumerate((("suppressed_grey", 100, "suppressed (% points)"), ("RI_grey", 1, "rest index RI"))):
    ax = fig.add_subplot(gs[1, jj])
    for i, (k, lab) in enumerate(rowsS):
        for (suf, oth, col, off) in (("", "other_ctx", "k", -.22), ("_late", "other_ctx", C_GREY, 0),
                                     ("", "higher_order_ctx", C_OTH, .22)):
            v = GS["tests"][f"{met}{suf}|{k}|vs_{oth}"]
            lo, hi = v["subject_boot_ci95"]
            ax.plot([lo * scale, hi * scale], [i + off] * 2, color=col, lw=1)
            ax.plot(v["diff"] * scale, i + off, "o", color=col, ms=2.4, mfc=col if v["p_perm"] < 0.05 else "white")
    ax.axvline(0, color="k", lw=0.5, ls=":")
    ax.set_yticks(range(len(rowsS)))
    ax.set_yticklabels([r[1] for r in rowsS] if jj == 0 else [])
    ax.invert_yaxis()
    ax.set_xlabel(f"set $-$ comparison, {xl}")
    style_axes(ax, categorical_y=True)
    if jj == 0:
        panel_label(ax, "b", x=-0.75)
    else:
        for t_, (lab, col) in enumerate((("vs other cortex", "k"), ("vs other cortex, last third", C_GREY),
                                         ("vs higher-order cortex", C_OTH))):
            ax.text(1.0, 1.25 - t_ * 0.1, lab, color=col, fontsize=5.5, transform=ax.transAxes, ha="right")
caps = [
    "Task-suppressed neurons per cortical region relative to the grey screen (the measure of Fig. 4a, here for all cortical regions), with region "
    "membership in the conservative core and the four published DMN maps underneath (Fig. 1). Whitesell et al. (2021): "
    "cortical regions with $>$50\\% of voxels in the core DMN mask of mouse rsfMRI group ICA registered to the Allen "
    "CCFv3; they place TEa, ECT and PERI in a separate lateral cortical network.",
    "The conservative core and the full Whitesell et al. and Lu et al. sets minus other cortex (black), the same against only the last third of trials (blue), and minus "
    "higher-order cortex only (grey): 95\\% mouse-bootstrap CI; filled markers, region-permutation p $<$ 0.05. The "
    "Whitesell set is weaker than the core because it adds the task-driven SSp-tr, SSp-ll and MOs.",
]
save_with_caption(fig, OUTD / "figS1_whitesell_restpref", caps, label="grey_s1", number=1, prefix="S")
png(OUTD / "figS1_whitesell_restpref")

# ======================================================================= Fig S2: Whitesell coupling
fig = plt.figure(figsize=(NATURE_DOUBLE_COL_IN, 3.4))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.1, 1.0], wspace=1.05)
ax = fig.add_subplot(gs[0])
wreg = [r for r in WHITESELL_DMN]
iw = {r: i for i, r in enumerate(wreg)}
MW = np.full((len(wreg), len(wreg)), np.nan)
XW = np.zeros((len(wreg), len(wreg)))
for (r1, r2), g in P[P.r1.isin(wreg) & P.r2.isin(wreg)].groupby(["r1", "r2"]):
    i, j = iw[r1], iw[r2]
    MW[i, j] = MW[j, i] = g.subject.nunique()
    XW[i, j] = XW[j, i] = g[~g.same_probe].subject.nunique()
cmW = plt.get_cmap("Blues").copy()
cmW.set_bad("#f0efea")
im = ax.imshow(np.ma.masked_invalid(MW), cmap=cmW, vmin=0, vmax=np.nanmax(MW), interpolation="nearest")
for i in range(len(wreg)):
    for j in range(len(wreg)):
        if XW[i, j] > 0:
            ax.plot(j, i, marker="x", ms=3, color="k", mew=0.8)
ax.set_xticks(range(len(wreg)))
ax.set_yticks(range(len(wreg)))
ax.set_xticklabels(wreg, rotation=90, fontsize=5)
ax.set_yticklabels(wreg, fontsize=5)
cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02, location="top")
cb.set_label("mice with the pair", fontsize=6)
cb.ax.tick_params(labelsize=5)
panel_label(ax, "a", x=-0.3, y=1.25)
ax = fig.add_subplot(gs[1])
M5 = json.load(open(MESO / "33_spont_meso_me.json"))
rowsC = [("cons", "cons. core"), ("whitesell", "Whitesell"), ("lu_all", "Lu et al.")]
yt, yy = [], 0
for k, lab in rowsC:
    for nm, v, mk in (("BWM all pairs", J["classes"][f"{k}|cortex"].get("dmn-dmn minus dmn-other"), "o"),
                      ("BWM 2 probes", J["classes"][f"{k}|cortex_cross_probe"].get("dmn-dmn minus dmn-other"), "s"),
                      ("mesoscope", M5["classes"][k].get("dmn-dmn minus dmn-other"), "^")):
        if v:
            ax.plot([v[1], v[2]], [yy, yy], color="k", lw=0.9)
            ax.plot(v[0], yy, mk, color="k", ms=2.8)
        yt.append(f"{lab}, {nm}")
        yy += 1
    yy += 0.4
ax.axvline(0, color="k", lw=0.5, ls=":")
ax.set_yticks([i + (i // 3) * 0.4 for i in range(len(yt))])
ax.set_yticklabels(yt, fontsize=5)
ax.invert_yaxis()
ax.set_xlabel("DMN-DMN $-$ DMN-other (z)")
style_axes(ax, categorical_y=True)
panel_label(ax, "b", x=-0.8, y=1.1)
EW = pd.read_csv(BWM / f"41_edges_whitesell_{BTAG}.csv")
EW = EW[EW.n_mice >= 3].sort_values("mouse_mean")
ax = fig.add_subplot(gs[2])
for i, (_, e) in enumerate(EW.iterrows()):
    ax.plot([e.ci_lo, e.ci_hi], [i, i], color="#6250d6", lw=0.9)
    ax.plot(e.mouse_mean, i, "o", color="#6250d6", ms=2.8)
ax.axvline(0, color="k", lw=0.5, ls=":")
ax.set_yticks(range(len(EW)))
ax.set_yticklabels([f"{a}–{b} ({m})" for a, b, m in zip(EW.r1, EW.r2, EW.n_mice)], fontsize=5)
ax.set_xlabel("residual coupling (z)")
style_axes(ax, categorical_y=True)
panel_label(ax, "c", x=-0.7, y=1.02)
JW = J["classes"]["whitesell|cortex_cross_probe"]
caps = [
    f"Grey-screen region pairs within the Whitesell et al. DMN: mice with the pair (grey: never recorded); orange "
    f"crosses, recorded on two probes in at least one session. {J['coverage']['whitesell']['n_edges_any']} of "
    f"{J['coverage']['whitesell']['n_possible_edges']} edges are recorded, {J['coverage']['whitesell']['n_edges_ge3_mice']} in "
    "$\\geq$3 mice; those are all frontal-frontal (ACA, PL, ILA, ORB, MOs) or RSPd-RSPv, and no frontal-posterior "
    "edge is recorded at all.",
    "DMN-DMN minus DMN-other coupling for three definitions: BWM cortical pairs (circles), BWM pairs on two probes "
    f"(squares) and mesoscope (triangles; 13 sessions, 7 mice): 95\\% mouse-bootstrap CI. For Whitesell, the two-probe "
    f"excess (+{JW['dmn-dmn minus dmn-other'][0]:.2f}) comes {JW['dd_sessions_top2_share'] * 100:.0f}\\% from two "
    "frontal multi-probe sessions (KS046, SWC\\_038; 6 mice in all), and the mesoscope excess mainly from one mouse "
    "(SP054, SSp-tr/VISa/VISam pairs).",
    "Every Whitesell DMN edge recorded in $\\geq$3 mice (number of mice in brackets): mean residual coupling over mice "
    "and 95\\% bootstrap CI.",
]
save_with_caption(fig, OUTD / "figS2_whitesell_coupling", caps, label="grey_s2", number=2, prefix="S")
png(OUTD / "figS2_whitesell_coupling")
print("done", ex)
