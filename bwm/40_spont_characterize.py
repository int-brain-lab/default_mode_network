"""Passive spontaneous period (grey screen) of each BWM session: timing and movement.

iblrig passiveChoiceWorld: spacer -> spontaneous activity (SPONTANEOUS_ACTIVITY_SECONDS,
default 600 s; Bonsai clears the screen to Gray, sync square off, no stimuli) -> spacer
-> receptive-field mapping -> spacer -> task replay. The interval comes from
_ibl_passivePeriods.intervalsTable (frame2ttl spacers), stored as s['spont'].

Per session: start relative to the end of the task, duration, data coverage, and
movement inside the period on the 50 ms grid (first 1 s skipped as in A/B):
wheel speed > 0.05 rad/s, licks, left-camera motion energy above the session's
60th percentile (the 'still' criterion of 20_full_A), pupil vs task ITIs.
Writes output/40_spont_sessions.csv, output/40_spont_timecourse.csv, 40_spont_summary.json.
"""
import json

import numpy as np
import pandas as pd

from common import OUT, done_eids

FULL = OUT / "full"
SUBJ = pd.read_csv(OUT / "bwm_insertions.csv").groupby("eid").subject.first()
ME_PCT = 60


def runs(b):
    """Lengths of consecutive True runs."""
    d = np.diff(np.r_[0, b.astype(int), 0])
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


rows, tc = [], []
for eid in done_eids():
    s = dict(np.load(FULL / f"{eid}.npz", allow_pickle=True))
    BIN = float(s["BIN"])
    nb = len(s["licks"])
    sp = s["spont"]
    r = {"eid": eid, "subject": SUBJ.get(eid), "has_spont": bool(np.all(np.isfinite(sp)))}
    t_first, t_last = np.nanmin(s["intervals_0"]), np.nanmax(s["intervals_1"])
    r.update(task_start=t_first, task_end=t_last, task_dur_min=(t_last - t_first) / 60)
    if not r["has_spont"]:
        rows.append(r)
        continue
    a, b = int(sp[0] / BIN) + 20, min(nb, int(sp[1] / BIN))
    r.update(spont_start=sp[0], spont_stop=sp[1], dur_s=sp[1] - sp[0], gap_after_task_min=(sp[0] - t_last) / 60,
             truncated=b < int(sp[1] / BIN))
    w = s["wheel_speed"].astype(float)
    lk = s["licks"].astype(float)
    me = s["motion_energy"].astype(float)
    pu = s["pupil"].astype(float)
    seg = slice(a, b)
    r["cov_wheel"] = np.isfinite(w[seg]).mean()
    r["cov_me"] = np.isfinite(me[seg]).mean()
    r["cov_pupil"] = np.isfinite(pu[seg]).mean()
    r["licks_ok"] = not np.all(lk == 255)
    wm = np.isfinite(w[seg]) & (w[seg] > 0.05)
    lm = (lk[seg] > 0) & (lk[seg] < 255)
    # movement fractions only where the signal was recorded (NaN otherwise, not 0)
    r["frac_wheel_moving"] = wm[np.isfinite(w[seg])].mean() if r["cov_wheel"] > .5 else np.nan
    r["licks_per_min"] = lk[seg][lk[seg] < 255].sum() / ((b - a) * BIN / 60)
    # task ITIs for comparison (feedback + 1.5 s .. next stimOn - 0.5 s)
    iti = np.zeros(nb, bool)
    so, fb = s["stimOn_times"], s["feedback_times"]
    for t in range(len(so) - 1):
        if np.isfinite(fb[t]) and np.isfinite(so[t + 1]):
            i0, i1 = int(np.ceil((fb[t] + 1.5) / BIN)), int(np.floor((so[t + 1] - 0.5) / BIN))
            if i1 > i0:
                iti[i0:i1] = True
    r["iti_frac_wheel_moving"] = (w[iti][np.isfinite(w[iti])] > 0.05).mean() if np.isfinite(w[iti]).mean() > .5 else np.nan
    mov = wm | lm
    if np.isfinite(me).mean() > .5:
        thr = np.nanpercentile(me, ME_PCT)
        mm = np.isfinite(me[seg]) & (me[seg] > thr)
        r["frac_me_high"] = mm[np.isfinite(me[seg])].mean() if r["cov_me"] > .5 else np.nan
        r["iti_frac_me_high"] = (me[iti][np.isfinite(me[iti])] > thr).mean() if np.isfinite(me[iti]).mean() > .5 else np.nan
        r["me_spont_over_iti"] = np.nanmedian(me[seg]) / np.nanmedian(me[iti])
        mov = mov | mm
    else:
        r["frac_me_high"] = r["iti_frac_me_high"] = r["me_spont_over_iti"] = np.nan
    if np.isfinite(pu[seg]).mean() > .5 and np.isfinite(pu[iti]).mean() > .5:
        r["pupil_spont_over_iti"] = np.nanmedian(pu[seg]) / np.nanmedian(pu[iti])
    still = ~mov
    bad = mov | np.r_[False, mov[:-1]] | np.r_[mov[1:], False]  # one-bin margin as in 20_full_A
    r["me_verified"] = bool(r["cov_me"] > .5)
    r["frac_still"] = (~bad).mean()
    r["still_s"] = (~bad).sum() * BIN
    ru = runs(~bad) * BIN
    r["still_bout_median_s"] = np.median(ru) if len(ru) else 0
    r["still_in_bouts_ge_5s"] = ru[ru >= 5].sum() / max(ru.sum(), 1e-9)
    # full 1 s bins that are entirely still (what 21_full_B uses with 'still')
    k = int(round(1 / BIN))
    n1 = (b - a) // k
    r["n_still_1s_bins"] = int(((~bad)[: n1 * k].reshape(n1, k).all(1)).sum())
    rows.append(r)
    # minute-by-minute
    for m in range(int((b - a) * BIN // 60)):
        sl = slice(m * 1200, (m + 1) * 1200)
        tc.append({"eid": eid, "minute": m, "wheel": wm[sl].mean(), "still": (~bad)[sl].mean(),
                   "me_high": (np.isfinite(me[seg][sl]) & (me[seg][sl] > np.nanpercentile(me, ME_PCT))).mean()
                   if np.isfinite(me).mean() > .5 else np.nan})

D = pd.DataFrame(rows)
D.to_csv(OUT / "40_spont_sessions.csv", index=False)
pd.DataFrame(tc).to_csv(OUT / "40_spont_timecourse.csv", index=False)
H = D[D.has_spont]
q = lambda c: [round(float(x), 3) for x in np.nanquantile(H[c], [.05, .25, .5, .75, .95])]
summ = {
    "n_sessions": len(D), "n_with_spont": int(len(H)), "n_mice_with_spont": int(H.subject.nunique()),
    "quantiles": "5/25/50/75/95%",
    "dur_s": q("dur_s"), "gap_after_task_min": q("gap_after_task_min"),
    "n_before_task": int((H.gap_after_task_min < 0).sum()), "n_truncated": int(H.truncated.sum()),
    "cov_wheel": q("cov_wheel"), "cov_me": q("cov_me"), "cov_pupil": q("cov_pupil"),
    "frac_wheel_moving": q("frac_wheel_moving"), "iti_frac_wheel_moving": q("iti_frac_wheel_moving"),
    "frac_me_high": q("frac_me_high"), "iti_frac_me_high": q("iti_frac_me_high"),
    "me_spont_over_iti": q("me_spont_over_iti"), "pupil_spont_over_iti": q("pupil_spont_over_iti"),
    "licks_per_min": q("licks_per_min"), "frac_still": q("frac_still"), "still_s": q("still_s"),
    "still_bout_median_s": q("still_bout_median_s"), "n_still_1s_bins": q("n_still_1s_bins"),
    "n_sessions_ge_120_still_1s_bins": int((H.n_still_1s_bins >= 120).sum()),
    "n_wheel_ok": int(H.frac_wheel_moving.notna().sum()), "n_me_ok": int(H.frac_me_high.notna().sum()),
    "frac_still_me_verified": q("frac_still") if False else
        [round(float(x), 3) for x in np.nanquantile(H[H.me_verified == True].frac_still, [.05, .25, .5, .75, .95])],
    "frac_still_no_me": [round(float(x), 3) for x in np.nanquantile(H[H.me_verified != True].frac_still, [.05, .25, .5, .75, .95])],
}
print(json.dumps(summ, indent=1))
json.dump(summ, open(OUT / "40_spont_summary.json", "w"), indent=1)
