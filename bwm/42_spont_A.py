"""Approach A referenced to the passive grey-screen period instead of task ITIs.

Rest = still 50 ms bins of the passive spontaneous period (grey screen, after the task;
still as in 20_full_A: no wheel > 0.05 rad/s, no lick, motion energy <= session 60th
percentile, one-bin margin). Rest samples = 0.3 s chunks of consecutive still bins; for the
test, only the first chunk in every 3 s window is kept, so samples are roughly independent
and about as many as trials (all chunks would make the test anti-conservative).
Task = per-trial mean rate of the TASK windows (motor_init, move, feedback).
Per neuron: RI_grey = (rest - task) / (rest + task) (all still chunks); Welch t-test
spaced chunks vs trials; suppressed_grey / enhanced_grey at p < 0.01.
Drift controls: the same against the last third of trials only (closest in time to the
grey screen; *_late), and a drift index DI = (late - early) / (late + early) of the task-window
rate between the last and first third of trials.
Region tests for our DMN sets and for the Lu et al. 2012 rat-DMN mapping.

Usage: python 42_spont_A.py [tag]
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

from common import (DMN_ALL, DMN_CORE, LU_DMN, LU_PARIETAL, LU_PFC_TEMPORAL, OUT, WHITESELL_DMN, beryl, cosmos,
                    done_eids)
from dmnstats import PRIMARY, perm_diff, region_table, split_half, subject_boot

TAG = sys.argv[1] if len(sys.argv) > 1 else "G_full"  # second arg 'me_only': sessions with motion energy
FULL = OUT / "full"
SUBJ = pd.read_csv(OUT / "bwm_insertions.csv").groupby("eid").subject.first()
TASK = ("motor_init", "move", "fback")
CHUNK = 6  # 0.3 s
SPACING = 60  # one test chunk per 3 s
ME_OK = None


def still_mask(s):
    nb = len(s["licks"])
    w, lk, me = (s[k].astype(float) for k in ("wheel_speed", "licks", "motion_energy"))
    ok = ~(np.isfinite(w) & (w > 0.05)) & ~((lk > 0) & (lk < 255))
    if np.isfinite(me).mean() > 0.5:
        ok &= ~(np.isfinite(me) & (me > np.nanpercentile(me, 60)))
    bad = ~ok
    return ~(bad | np.r_[False, bad[:-1]] | np.r_[bad[1:], False])


def session(eid):
    s = dict(np.load(FULL / f"{eid}.npz", allow_pickle=True))
    sp = s["spont"]
    if not np.all(np.isfinite(sp)):
        return []
    BIN, nb = float(s["BIN"]), len(s["licks"])
    ep = np.zeros(nb, bool)
    ep[int(sp[0] / BIN) + 20: min(nb, int(sp[1] / BIN))] = True
    ep &= still_mask(s)
    # chunk ids: consecutive still bins, cut every CHUNK bins
    d = np.diff(np.r_[0, ep.astype(int), 0])
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    chunks = [np.arange(a + i * CHUNK, a + (i + 1) * CHUNK) for a, b in zip(starts, ends)
              for i in range((b - a) // CHUNK)]
    if len(chunks) < 50:
        return []
    ci = np.array(chunks)
    win = ci[:, 0] // SPACING
    sp_idx = np.r_[0, np.flatnonzero(np.diff(win)) + 1]  # first chunk of every 3 s window
    me_ok = bool(np.isfinite(s["motion_energy"][ep]).mean() > .5)
    msk = s["mask"].astype(bool)
    ntr = len(msk)
    third = np.zeros(ntr, int)
    third[: ntr // 3], third[ntr - ntr // 3:] = -1, 1
    rows = []
    for f in sorted(FULL.glob(f"{eid}_probe*.npz")):
        p = np.load(f, allow_pickle=True)
        X = p["counts"][:, :nb].astype(np.float32) / BIN
        rest = X[:, ci].mean(2).T  # chunks x neurons
        W = {k: p[f"win_{k}"] / dur for k, dur in (("motor_init", .15), ("move", .15), ("fback", .3))}
        task_all = np.mean([W[k] for k in TASK], axis=0)
        ok = msk & np.all(np.isfinite(task_all), 1)
        task, late, early = task_all[ok], task_all[ok & (third == 1)], task_all[ok & (third == -1)]
        with np.errstate(invalid="ignore", divide="ignore"):
            _, pv = ttest_ind(rest[sp_idx], task, axis=0, equal_var=False)
            _, pvl = ttest_ind(rest[sp_idx], late, axis=0, equal_var=False)
        rows.append(pd.DataFrame({"eid": eid, "probe": str(p["probe"]), "atlas_id": p["atlas_id"],
                                  "r_grey": rest.mean(0), "r_task": task.mean(0), "r_late": late.mean(0),
                                  "r_early": early.mean(0), "p_grey_task": np.nan_to_num(pv, nan=1.),
                                  "p_grey_late": np.nan_to_num(pvl, nan=1.), "n_chunks": len(ci),
                                  "n_test_chunks": len(sp_idx), "n_trials": len(task), "me_ok": me_ok}))
    return rows


SETS = {"ours_core": DMN_CORE, "ours_core+lateral": DMN_ALL, "lu_all": LU_DMN,
        "lu_pfc_temporal": LU_PFC_TEMPORAL, "lu_parietal": LU_PARIETAL,
        "lu_without_auditory": [r for r in LU_DMN if not r.startswith("AUD")], "whitesell": WHITESELL_DMN}

if __name__ == "__main__":
    with ProcessPoolExecutor(8) as ex:
        rows = [r for rr in ex.map(session, done_eids()) for r in rr]
    N = pd.concat(rows, ignore_index=True)
    N["region"], N["cosmos"] = beryl(N.atlas_id), cosmos(N.atlas_id)
    N["subject"] = N.eid.map(SUBJ)
    N = N[~N.region.isin(["root", "void"])]
    N = N[(N.r_grey + N.r_task) / 2 >= 0.5].reset_index(drop=True)
    N["RI_grey"] = (N.r_grey - N.r_task) / (N.r_grey + N.r_task + 1e-9)
    N["suppressed_grey"] = ((N.p_grey_task < .01) & (N.RI_grey > 0)).astype(float)
    N["enhanced_grey"] = ((N.p_grey_task < .01) & (N.RI_grey < 0)).astype(float)
    N["RI_grey_late"] = (N.r_grey - N.r_late) / (N.r_grey + N.r_late + 1e-9)
    N["suppressed_grey_late"] = ((N.p_grey_late < .01) & (N.RI_grey_late > 0)).astype(float)
    N["DI"] = (N.r_late - N.r_early) / (N.r_late + N.r_early + 1e-9)
    if len(sys.argv) > 2 and sys.argv[2] == "me_only":
        N = N[N.me_ok].reset_index(drop=True)
    N.to_parquet(OUT / f"42_neurons_{TAG}.pqt")
    cols = ["RI_grey", "suppressed_grey", "enhanced_grey", "RI_grey_late", "suppressed_grey_late", "DI"]
    R = region_table(N, cols)
    R.to_csv(OUT / f"42_regions_{TAG}.csv")
    res = {"tag": TAG, "n_sessions": int(N.eid.nunique()), "n_subjects": int(N.subject.nunique()),
           "n_neurons": int(len(N)), "median_grey_chunks": float(N.groupby("eid").n_chunks.first().median()),
           "median_test_chunks": float(N.groupby("eid").n_test_chunks.first().median()),
           "median_trials": float(N.groupby("eid").n_trials.first().median()),
           "frac_sig_any": round(float(((N.suppressed_grey + N.enhanced_grey) > 0).mean()), 3),
           "reliability": {c: split_half(N, R, c) for c in ("RI_grey", "suppressed_grey")}, "tests": {}}
    # cortex tests (CA1 of the Lu set is tested separately below, as it is not cortex)
    C = R[R.cosmos == "Isocortex"]
    for sname, mem in SETS.items():
        isd = C.index.isin(mem)
        for oname, oth in (("other_ctx", ~isd), ("higher_order_ctx", ~isd & ~C.index.isin(PRIMARY))):
            for c in cols:
                obs, p = perm_diff(C[c].values[isd], C[c].values[oth])
                res["tests"][f"{c}|{sname}|vs_{oname}"] = {
                    "dmn": round(float(C[c].values[isd].mean()), 4), "other": round(float(C[c].values[oth].mean()), 4),
                    "diff": round(obs, 4), "p_perm": round(p, 5), "n": [int(isd.sum()), int(oth.sum())],
                    "subject_boot_ci95": subject_boot(N, R, c, C.index[isd], C.index[oth])}
    for c in cols:
        res[f"ctx_ranking_{c}"] = C[c].sort_values(ascending=False).round(3).to_dict()
    res["CA1"] = R.loc["CA1", cols].round(3).to_dict() if "CA1" in R.index else None
    from scipy.stats import spearmanr as _sp
    res["drift"] = {"ctx_median_DI": round(float(C.DI.median()), 4),
                    "rho_DI_vs_RI_grey": round(float(_sp(C.DI, C.RI_grey).statistic), 3),
                    "rho_DI_vs_suppressed_grey": round(float(_sp(C.DI, C.suppressed_grey).statistic), 3),
                    "rho_late_vs_all_suppressed": round(float(_sp(C.suppressed_grey_late, C.suppressed_grey).statistic), 3)}
    # agreement with the ITI-referenced A (same neurons' regions)
    A = pd.read_csv(OUT / "20_regions_A_full.csv", index_col=0)
    cm = C.index.intersection(A.index)
    from scipy.stats import spearmanr
    res["vs_iti_A"] = {"RI": round(float(spearmanr(C.loc[cm, "RI_grey"], A.loc[cm, "RI_iti"]).statistic), 3),
                       "suppressed": round(float(spearmanr(C.loc[cm, "suppressed_grey"], A.loc[cm, "suppressed"]).statistic), 3),
                       "n": int(len(cm))}
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("ctx_")}, indent=1)[:6000])
    json.dump(res, open(OUT / f"42_spont_A_{TAG}.json", "w"), indent=1)
