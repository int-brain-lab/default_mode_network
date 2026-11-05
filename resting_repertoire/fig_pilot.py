"""Pilot figure: alignment of resting fluctuations with task subspaces in the IBL mesoscope (10_, 11_ outputs).

Writes fig_repertoire_pilot.{pdf,tex,png}. Usage: python fig_pilot.py
"""
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "report"))
from plotstyle import panel_label, save_with_caption, style_axes  # noqa: E402

GREY = "#8a8880"
rng = np.random.default_rng(0)
A = pd.read_csv(HERE / "output" / "10_meso_repertoire.csv")
B = pd.read_csv(HERE / "output" / "11_meso_repertoire_denoised.csv")
A = A[A.no_global].assign(subspace="evoked")
D = pd.concat([A[["eid", "subject", "rest", "subspace", "A", "A_perm", "A_cov"]],
               B[["eid", "subject", "rest", "subspace", "A", "A_perm", "A_cov"]]])
SUBS = [("evoked", "evoked\n(trial means)"), ("evoked_cv", "evoked,\ncross-validated"), ("trials", "single-trial\ntask PCs")]
fig, axs = plt.subplots(1, 2, figsize=(6.0, 2.3), gridspec_kw=dict(wspace=.45))
for ax, (rest, title), L in zip(axs, (("still", "still bins, passive block"), ("grey_still", "still bins, grey screen")), "ab"):
    for i, (sub, lab) in enumerate(SUBS):
        m = D[(D.rest == rest) & (D.subspace == sub)].groupby("subject")[["A_perm", "A", "A_cov"]].mean()
        for j, (col, c) in enumerate((("A_perm", GREY), ("A", "#222222"), ("A_cov", "#2a78d6"))):
            x = i * 4 + j
            ax.scatter(x + rng.uniform(-.12, .12, len(m)), m[col], s=5, color=c, lw=0, alpha=.8)
            ax.plot([x - .3, x + .3], [m[col].mean()] * 2, color="k", lw=1)
        for _, r in m.iterrows():
            ax.plot([i * 4, i * 4 + 1, i * 4 + 2], r[["A_perm", "A", "A_cov"]], color=GREY, lw=.3, alpha=.5)
    ax.set_xticks([i * 4 + 1 for i in range(3)])
    ax.set_xticklabels([s[1] for s in SUBS], fontsize=5.5)
    ax.set_ylim(0, .3)
    ax.set_yticks([0, .1, .2, .3])
    ax.set_ylabel("alignment index")
    ax.set_title(title, fontsize=6.5)
    style_axes(ax, categorical_x=True)
    panel_label(ax, L)
fig.text(.5, -.08, "grey: ROI-permuted null    black: observed    blue: covariance-matched random", ha="center", fontsize=5.5, color=GREY)
STEM = HERE / "fig_repertoire_pilot"
save_with_caption(fig, STEM, [
    "Alignment of resting fluctuations with task subspaces (K = 5 dims; uniform direction projected out), per mouse "
    "(dots; bars, means over 9 mice; lines connect a mouse). Rest: still 1\\,s bins of the passive block (grey "
    "screen and receptive-field mapping), z-scored deconvolved activity of all ROIs, centred. Alignment index: rest "
    "variance in the subspace relative to the most any 5-dim subspace captures. Task subspaces: top dims of 16 "
    "trial-averaged evoked patterns (stim, choice, feedback, post-feedback windows minus pre-stimulus; side $\\times$ "
    "outcome); the same restricted to dims shared by odd and even trials; top PCs of single-trial evoked activity "
    "(odd trials). Nulls: ROI permutation of the task patterns (grey) and random subspaces drawn from the covariance "
    "of all recording bins (blue).",
    "The same for still bins of the grey screen alone (34 sessions)."],
    label="repertoire_pilot",
    intro="Resting fluctuations in the mesoscope share the dims of task-period variability, but are not concentrated "
          "in the task-evoked patterns.")
subprocess.run(["pdftoppm", "-png", "-r", "200", "-singlefile", f"{STEM}.pdf", str(STEM)], check=True)
