"""Figures 5-6 of the grey-screen report: firing rates of the conservative DMN core across behavioural windows,
and the DMN-motivated contrasts (outputs of bwm/44_windows.py and bwm/45_window_stats.py)."""
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "bwm"))
from matplotlib.patches import Rectangle  # noqa: E402
from plotstyle import NATURE_DOUBLE_COL_IN, panel_label, save_with_caption, style_axes  # noqa: E402
from common import CONS_DMN  # noqa: E402

BWM = HERE.parent / "bwm" / "output"
OUTD = HERE / "grey"
S = json.load(open(BWM / "45_window_stats.json"))
WIN = [("stim", "stimulus", "#eb6834"), ("movefb", "movement +\nfeedback", "#c4451c"), ("quies", "quiescence", "#6250d6"),
       ("iti", "ITI rest", "#2a78d6"), ("iti_move", "ITI moving", "#86b6ef"), ("grey", "grey still", "#5f5e58"),
       ("grey_move", "grey moving", "#b9b7ae")]
C_CORE, C_CTX, C_ALL = "#2a78d6", "#5f5e58", "#b9b7ae"


def png(stem):
    subprocess.run(["pdftoppm", "-png", "-r", "200", "-singlefile", f"{stem}.pdf", str(stem)], check=True)


# ======================================================================= Fig 5: rates per window
groups = [(r, r) for r in ["ACAd", "ACAv", "PL", "ORBm", "RSPagl", "RSPd", "RSPv"]] + \
         [("core", "core (all 7)"), ("other_ctx", "other cortex"), ("other_all", "all other regions")]
fig = plt.figure(figsize=(NATURE_DOUBLE_COL_IN, 4.6))
gs5 = fig.add_gridspec(3, 5, height_ratios=[0.62, 1, 1], hspace=0.5, wspace=0.45)
axs = np.array([[fig.add_subplot(gs5[r + 1, c]) for c in range(5)] for r in range(2)])
# a: windows on a median trial, relative to stimulus onset
T = json.load(open(BWM / "44_trial_timing.json"))
rt, fbt, fbend, gap, qd = (T[k][1] for k in ("rt", "fb", "fb_end", "gap", "quies"))
t_start, t_end = -qd, fbt + fbend
t_next = t_end + gap
col = {w: c for w, _, c in WIN}
ax = fig.add_subplot(gs5[0, :])
# lane y-ranges: main lane for rest/quiescence/movement/feedback; a lower lane for the stimulus window,
# which overlaps the movement window in time on a median trial
UP, LO = (0.42, 0.7), (0.3, 0.4)
blocks = [(-qd, -0.1, "quies", UP, "quiescence (still bins)", (-qd + -0.1) / 2, 0.76, "center"),
          (0, 0.15, "stim", LO, "stimulus 0–0.15 s", -0.03, 1.12, "right"),
          (rt - 0.15, rt + 0.15, "movefb", UP, "movement ±0.15 s", rt, 1.0, "left"),
          (fbt, fbt + 0.3, "movefb", UP, "feedback 0–0.3 s", fbt + 0.32, 0.76, "left"),
          (t_end, t_next, "iti", UP, "ITI rest (still bins)\nITI moving (moving bins)", (t_end + t_next) / 2, 0.76, "center"),
          (-qd - gap, -qd, "iti", UP, "", 0, 0, "center")]
for a0, a1, w, (y0, y1), lab, lx, ly, ha in blocks:
    ax.add_patch(Rectangle((a0, y0), a1 - a0, y1 - y0, color=col[w], lw=0, alpha=1 if lab else 0.45))
    if lab:
        ax.text(lx, ly, lab, ha=ha, va="bottom", fontsize=4.8, color=col[w], linespacing=1.0)
        if ly >= 1.0:
            ax.plot([(a0 + a1) / 2, (a0 + a1) / 2 if ha == "left" else lx], [y1, ly - 0.02], color=col[w], lw=0.4)
ax.add_patch(Rectangle((fbt + 0.32, 0.5), t_end - fbt - 0.32, 0.12, color="#e6e4dc", lw=0))
ax.text((fbt + 0.32 + t_end) / 2, 0.56, "reward / error tone (not used)", ha="center", va="center", fontsize=4.0, color="#77756c")
ax.plot([-qd - gap - 0.2, t_next + 0.5], [0.3, 0.3], color="k", lw=0.6)
for x, lab, ly, ha in ((-qd - gap, "prev. trial end", 0.17, "center"), (-qd, "trial start", 0.17, "center"),
                       (0, "stimulus onset", -0.36, "right"), (rt, "1st movement", -0.36, "left"),
                       (fbt, "feedback", 0.17, "left"), (t_end, "trial end", 0.17, "center"),
                       (t_next, "next trial start", 0.17, "center")):
    ax.plot([x, x], [ly + 0.04, 0.3], color="k", lw=0.5)
    ax.text(x, ly, f"{lab}\n{x:+.2f} s", ha=ha, va="top", fontsize=4.3, linespacing=1.0)
# grey screen block after an axis break
gx = t_next + 1.0
ax.text(t_next + 0.62, 0.3, "//", ha="center", va="center", fontsize=6)
ax.add_patch(Rectangle((gx, 0.3), 1.6, 0.38, color=col["grey"], lw=0))
ax.text(gx + 0.8, 0.76, "grey screen (after the task, 10 min)\nstill bins / moving bins", ha="center", va="bottom",
        fontsize=4.8, color=col["grey"], linespacing=1.0)
ax.set_xlim(-qd - gap - 0.25, gx + 1.7)
ax.set_ylim(-0.68, 1.3)
ax.axis("off")
panel_label(ax, "a", x=-0.02, y=0.8)
for ax, (key, lab) in zip(axs.flat, groups):
    for i, (w, wl, col) in enumerate(WIN):
        m, lo, hi, nm = S["rates"][w][key]
        ax.bar(i, m, color=col, width=0.8)
        ax.plot([i, i], [lo, hi], color="k", lw=0.6)
    nmice = S["rates"]["movefb"][key][3]
    ax.set_title(f"{lab} ({nmice})", fontsize=6.5, color=C_CORE if key in CONS_DMN or key == "core" else "k")
    style_axes(ax)
    ax.set_xticks([])
    if ax in axs[:, 0]:
        ax.set_ylabel("firing rate (Hz)")
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=col, label=wl.replace("\n", " ")) for w, wl, col in WIN], loc="lower center", ncol=7,
           fontsize=5.5, frameon=False, handlelength=1, columnspacing=1.2, bbox_to_anchor=(0.5, -0.01))
fig.subplots_adjust(bottom=0.08)
panel_label(axs[0, 0], "b", x=-0.45, y=1.12)
caps = [
    "The windows on a median trial, aligned to stimulus onset (median event times from all BWM trials; quiescence "
    f"lasts {T['quies'][0]:.2f}--{T['quies'][2]:.2f} s (IQR) because it restarts when the wheel moves). Window "
    "lengths are to scale; the faded block is the previous inter-trial gap. The period between feedback and trial end "
    "(reward consumption or error tone) is not used. The grey screen follows the task.",
    "Session-averaged firing rate in seven behavioural windows for each region of the conservative DMN core, the core "
    "pooled, all other cortex and all other recorded regions (bars, left to right as in the legend): stimulus "
    "(0--0.15 s after onset), movement + feedback (movement onset $-$0.15--0 s, 0--0.15 s and feedback 0--0.3 s; the task "
    "windows of Fig. 4), quiescence (still bins from trial start to 0.1 s before the stimulus: engaged, waiting, not "
    "moving), ITI rest (still bins between the end of a trial and the start of the next), ITI moving (moving bins in the "
    "same gaps), and the grey screen still and moving. Mean over mice of the neuron-averaged rate, 95\\% bootstrap CI "
    f"over mice (number of mice in brackets); {S['n_neurons']:,} neurons with $\\geq$0.5 Hz, {S['n_sessions']} "
    f"sessions, {S['n_mice']} mice.",
]
save_with_caption(fig, OUTD / "fig5_windows", caps, label="grey_windows", number=5)
png(OUTD / "fig5_windows")

# ======================================================================= Fig 6: contrasts
CON = [("task_vs_iti", "movement+feedback / ITI rest"), ("stim_vs_iti", "stimulus / ITI rest"),
       ("quies_vs_iti", "quiescence / ITI rest"), ("grey_vs_iti", "grey still / ITI rest"),
       ("move_iti", "ITI moving / ITI still"), ("move_grey", "grey moving / grey still"),
       ("task_vs_grey", "movement+feedback / grey still")]
fig = plt.figure(figsize=(NATURE_DOUBLE_COL_IN, 3.6))
gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 0.9], wspace=0.35)
ax = fig.add_subplot(gs[0])
for i, (k, lab) in enumerate(CON):
    for j, (key, col, off) in enumerate((("core_alone", C_CORE, -0.22), ("other_ctx_alone", C_CTX, 0), ("other_all_alone", C_ALL, 0.22))):
        m, lo, hi, n = S["contrasts"][k][key]
        ax.plot([lo, hi], [i + off] * 2, color=col, lw=1)
        ax.plot(m, i + off, "o", color=col, ms=2.6)
ax.axvline(0, color="k", lw=0.5, ls=":")
ax.set_yticks(range(len(CON)))
ax.set_yticklabels([c[1] for c in CON], fontsize=5.5)
ax.invert_yaxis()
ax.set_xlabel("log$_2$ rate ratio\n(> 0: higher in first window)")
for t_, (lab, col) in enumerate((("core", C_CORE), ("other cortex", C_CTX), ("all other regions", C_ALL))):
    ax.text(1.0, 1.13 - t_ * 0.055, lab, color=col, fontsize=5.5, transform=ax.transAxes, ha="right")
style_axes(ax, categorical_y=True)
panel_label(ax, "a", x=-0.9)
ax = fig.add_subplot(gs[1])
for i, (k, lab) in enumerate(CON):
    for j, (o, col, off) in enumerate((("vs_other_ctx", C_CTX, -0.15), ("vs_all_other", C_ALL, 0.15))):
        ws = S["contrasts"][k][o]["within_session"]
        if ws is None:
            continue
        ax.plot(ws["ci"], [i + off] * 2, color=col, lw=1)
        ax.plot(ws["mean"], i + off, "s", color=col, ms=2.6, mfc=col if (ws["wilcoxon_p"] or 1) < 0.05 else "white")
ax.axvline(0, color="k", lw=0.5, ls=":")
ax.set_yticks(range(len(CON)))
ax.set_yticklabels([])
ax.invert_yaxis()
ax.set_xlabel("core $-$ other, same session\n(log$_2$ ratio)")
ax.text(1.0, 1.13, "vs other cortex", color=C_CTX, fontsize=5.5, transform=ax.transAxes, ha="right")
ax.text(1.0, 1.075, "vs all other regions", color=C_ALL, fontsize=5.5, transform=ax.transAxes, ha="right")
style_axes(ax, categorical_y=True)
panel_label(ax, "b", x=-0.08)
sub = gs[2].subgridspec(2, 1, hspace=0.75)
ax = fig.add_subplot(sub[0])
refs = [("iti", "ITI rest"), ("quies", "quiescence")]
for i, (r, lab) in enumerate(refs):
    v = S["suppressed"][r]
    ax.bar(i - 0.18, v["core"] * 100, width=0.34, color=C_CORE)
    ax.bar(i + 0.18, v["other_ctx"] * 100, width=0.34, color=C_CTX)
    ax.text(i, max(v["core"], v["other_ctx"]) * 100 + 2, f"p={v['region_perm_p_vs_ctx']:.2f}", ha="center", fontsize=5)
ax.set_xticks(range(2))
ax.set_xticklabels([r[1] for r in refs], fontsize=5.5)
ax.set_ylabel("suppressed (%)")
ax.set_ylim(0, 38)
style_axes(ax, categorical_x=True)
panel_label(ax, "c", x=-0.45)
ax = fig.add_subplot(sub[1])
for i, (w, lab) in enumerate((("iti", "ITI rest"), ("quies", "quiescence"))):
    v = S["coupling"][w]
    for j, (k, col, off) in enumerate((("core-core minus core-other", C_CORE, -0.12), ("core-other minus other-other", C_CTX, 0.12))):
        x = v.get(k)
        if x:
            ax.plot([x[1], x[2]], [i + off] * 2, color=col, lw=1)
            ax.plot(x[0], i + off, "o", color=col, ms=2.6)
ax.axvline(0, color="k", lw=0.5, ls=":")
ax.set_yticks(range(2))
ax.set_yticklabels(["ITI rest", "quiescence"], fontsize=5.5)
ax.invert_yaxis()
ax.set_xlabel("trial-wise coupling\ndifference (z)")
style_axes(ax, categorical_y=True)
panel_label(ax, "d", x=-0.45)
cc = S["coupling"]["iti"]
caps = [
    "DMN-motivated contrasts, per neuron log$_2$ ratio of session-averaged rates (+0.1 Hz), mean over mice with 95\\% "
    "bootstrap CI, for the conservative core, all other cortex and all other recorded regions. A task-negative region "
    "should be below zero in the first three rows; the quiescence row separates engagement from movement, the "
    "movement rows separate movement from the task.",
    "The most conservative version: core neurons minus non-core neurons recorded in the same session, averaged per "
    "mouse (31 mice vs other cortex, 49 vs all other regions); filled markers, Wilcoxon p $<$ 0.05 over mice.",
    "Fraction of neurons suppressed during movement + feedback relative to ITI rest or quiescence (per-trial Welch "
    "test, p $<$ 0.01), core vs other cortex; region-permutation p.",
    f"Trial-by-trial co-fluctuation of region signals in ITI rest and quiescence windows (distance and probe residual; "
    f"{cc['n']['core-core']} core-core pairs, {cc['n_mice_cc']} mice): core-core minus core-other (blue) and core-other "
    "minus other-other (grey), 95\\% mouse-bootstrap CI. The core-core excess comes from same-probe neighbours "
    "($<$0.5 mm, mainly RSPd--RSPv); distance-matched within session it is +0.06 (CI $-$0.02 to 0.13).",
]
save_with_caption(fig, OUTD / "fig6_contrasts", caps, label="grey_contrasts", number=6)
png(OUTD / "fig6_contrasts")
print("ok")
