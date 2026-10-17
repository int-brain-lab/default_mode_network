"""Where do the 'rest' windows fall in the BWM inter-trial interval?

Uses the BWM aggregate trials table. For each trial t>0 of the 345 sessions in
the ~/dmn insertion list:
  gap_end = stimOn[t] - intervals_1[t-1]   (time since previous trial end)
  gap_fb  = stimOn[t] - feedback[t-1]      (time since previous feedback)
The ~/dmn 'inter_trial' window is stimOn -1.15..-1.0 s and keeps trials with
gap_end > 1.15 s. Also reports how long a rest window could be if defined
from feedback + X to stimOn - 0.5 s (before the quiescence period).
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import FIGS, OUT, TRIALS_PQT, insertions

t = pd.read_parquet(TRIALS_PQT)
eids = set(insertions().eid)
t = t[t.eid.isin(eids)].copy()

rows = []
for eid, g in t.groupby("eid", sort=False):
    g = g.sort_values("intervals_0")
    so = g.stimOn_times.values
    fb = g.feedback_times.values
    i1 = g.intervals_1.values
    i0 = g.intervals_0.values
    ft = g.feedbackType.values
    inc = g.bwm_include.values.astype(bool)
    rows.append(pd.DataFrame({
        "eid": eid,
        "gap_end": so[1:] - i1[:-1],
        "gap_fb": so[1:] - fb[:-1],
        "start_to_stim": so[1:] - i0[1:],
        "end_minus_fb_prev": i1[:-1] - fb[:-1],
        "prev_correct": ft[:-1] == 1,
        "include": inc[1:],
    }))
df = pd.concat(rows, ignore_index=True)
df = df[df.include & np.isfinite(df.gap_fb)]

kept = df.gap_end > 1.15
# window start relative to previous feedback
win_after_fb = df.gap_fb - 1.15
summ = {
    "n_sessions": int(df.eid.nunique()),
    "n_trials": int(len(df)),
    "frac_trials_kept_by_iti_mask": float(kept.mean()),
    "prev_trial_end_minus_feedback_s_median_correct": float(df.loc[df.prev_correct, "end_minus_fb_prev"].median()),
    "prev_trial_end_minus_feedback_s_median_error": float(df.loc[~df.prev_correct, "end_minus_fb_prev"].median()),
    "gap_fb_s_quantiles_5_25_50_75_95": np.nanquantile(df.gap_fb, [.05, .25, .5, .75, .95]).round(3).tolist(),
    "gap_end_s_quantiles_5_25_50_75_95": np.nanquantile(df.gap_end, [.05, .25, .5, .75, .95]).round(3).tolist(),
    "rest_window_start_after_prev_feedback_s_quantiles_kept": np.nanquantile(win_after_fb[kept], [.05, .25, .5, .75, .95]).round(3).tolist(),
    "frac_kept_windows_within_1s_of_prev_feedback": float((win_after_fb[kept] < 1.0).mean()),
    "frac_kept_windows_within_1s_of_prev_feedback_correct_prev": float((win_after_fb[kept & df.prev_correct] < 1.0).mean()),
}
# possible longer rest window: feedback + X .. stimOn - 0.5 (before quiescence starts)
for x in (1.0, 1.5, 2.0):
    L = df.gap_fb - x - 0.5
    summ[f"alt_window_fb+{x}_to_stim-0.5_median_len_s"] = float(np.median(L.clip(lower=0)))
    summ[f"alt_window_fb+{x}_to_stim-0.5_frac_trials_len_ge_0.5s"] = float((L >= 0.5).mean())
    summ[f"alt_window_fb+{x}_to_stim-0.5_total_hours"] = float(L.clip(lower=0).sum() / 3600)
print(json.dumps(summ, indent=1))
json.dump(summ, open(OUT / "01_iti_timing.json", "w"), indent=1)

fig, axs = plt.subplots(1, 3, figsize=(10, 2.8))
bins = np.linspace(0, 6, 121)
for ax, (lab, x) in zip(axs, [("stimOn - prev feedback (s)", df.gap_fb),
                              ("stimOn - prev trial end (s)", df.gap_end),
                              ("rest window start - prev feedback (s)", win_after_fb[kept])]):
    for pc, c in [(True, "C0"), (False, "C3")]:
        sel = df.prev_correct if lab != "rest window start - prev feedback (s)" else df.prev_correct[kept]
        ax.hist(x[sel == pc], bins=bins, color=c, alpha=.6, density=True,
                label="prev correct" if pc else "prev error")
    ax.set_xlabel(lab)
    ax.spines[["top", "right"]].set_visible(False)
    ax.locator_params(nbins=3)
axs[1].axvline(1.15, color="k", lw=.8, ls="--")
axs[0].legend(frameon=False, fontsize=7)
axs[0].set_ylabel("density")
fig.tight_layout()
fig.savefig(FIGS / "01_iti_timing.pdf")
fig.savefig(FIGS / "01_iti_timing.png", dpi=150)
