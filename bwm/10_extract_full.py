"""Full-BWM extraction for the DMN analyses (all insertions of bwm_query).

Loaders follow ~/dmn/dmn_bwm.py: brainwidemap bwm_query, load_good_units,
load_trials_and_mask (with dmn_bwm's 2024_Q2 trials-tag fix).

Per session -> output/full/<eid>.npz:
  t0, BIN, trials columns (+ BWM mask), passive spontaneous interval,
  behaviour on the same 50 ms grid: wheel speed, licks, left-camera motion
  energy, pupil diameter (NaN where unavailable).
Per insertion -> output/full/<eid>_<probe>.npz:
  uuids, atlas_id, xyz, counts (neurons x bins, uint8, 50 ms bins over the
  whole recording) and exact per-trial counts for the task windows in WIN.
Spike files downloaded by this run are deleted afterwards (disk space);
files already in the ONE cache are kept.
"""
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import os
# dmn_bwm.py comes from int-brain-lab/bwm_dmn; point DMN_CODE at a checkout of it (default ~/dmn)
sys.path.insert(0, os.environ.get("DMN_CODE", str(Path.home() / "dmn")))
HERE = Path(__file__).resolve().parent
FULL = HERE / "output" / "full"
BIN = 0.05
# name: (event, t0, t1) relative to event
WIN = {
    "rest_bwm": ("stimOn_times", -1.15, -1.0),
    "quies": ("stimOn_times", -0.4, -0.1),
    "stim": ("stimOn_times", 0.0, 0.15),
    "motor_init": ("firstMovement_times", -0.15, 0.0),
    "move": ("firstMovement_times", 0.0, 0.15),
    "fback": ("feedback_times", 0.0, 0.3),
    "post_fb": ("feedback_times", 1.0, 1.5),
}
TRIAL_COLS = ["stimOn_times", "firstMovement_times", "feedback_times", "intervals_0", "intervals_1",
              "feedbackType", "choice", "probabilityLeft", "contrastLeft", "contrastRight"]


def _binned_mean(t, x, edges):
    """Mean of samples x(t) in each bin; NaN for empty bins."""
    ok = np.isfinite(t) & np.isfinite(x)
    t, x = t[ok], x[ok]
    i = np.searchsorted(edges, t, side="right") - 1
    v = (i >= 0) & (i < len(edges) - 1)
    s = np.bincount(i[v], weights=x[v], minlength=len(edges) - 1)
    n = np.bincount(i[v], minlength=len(edges) - 1)
    with np.errstate(invalid="ignore"):
        return (s / n).astype(np.float32)


def behaviour(one, eid, edges, task_coll):
    out = {}
    try:
        w = one.load_object(eid, "wheel", collection=task_coll)
        t, p = w["timestamps"], w["position"]
        tt = np.arange(t[0], t[-1], 0.01)
        pp = np.interp(tt, t, p)
        sp = np.abs(np.gradient(pp, tt))
        out["wheel_speed"] = _binned_mean(tt, sp, edges)
    except Exception:
        out["wheel_speed"] = np.full(len(edges) - 1, np.nan, np.float32)
    try:
        lk = one.load_dataset(eid, "licks.times.npy", collection="alf")
        out["licks"] = np.histogram(lk, edges)[0].astype(np.uint8)
    except Exception:
        out["licks"] = np.full(len(edges) - 1, 255, np.uint8)  # 255 = missing
    try:
        me = one.load_dataset(eid, "leftCamera.ROIMotionEnergy.npy", collection="alf")
        ct = one.load_dataset(eid, "_ibl_leftCamera.times.npy", collection="alf")
        n = min(len(me), len(ct))
        out["motion_energy"] = _binned_mean(ct[:n], me[:n], edges)
    except Exception:
        out["motion_energy"] = np.full(len(edges) - 1, np.nan, np.float32)
    try:
        f = one.load_dataset(eid, "_ibl_leftCamera.features.pqt", collection="alf")
        ct = one.load_dataset(eid, "_ibl_leftCamera.times.npy", collection="alf")
        col = "pupilDiameter_smooth" if "pupilDiameter_smooth" in f else "pupilDiameter_raw"
        n = min(len(f), len(ct))
        out["pupil"] = _binned_mean(ct[:n], f[col].values[:n].astype(float), edges)
    except Exception:
        out["pupil"] = np.full(len(edges) - 1, np.nan, np.float32)
    return out


def passive_spont(one, eid):
    for coll in ("alf", "alf/task_00", "alf/task_01"):
        try:
            pp = one.load_dataset(eid, "_ibl_passivePeriods.intervalsTable.csv", collection=coll)
            if "spontaneousActivity" in pp:
                return np.array(pp["spontaneousActivity"].values[:2], float)
        except Exception:
            continue
    return np.array([np.nan, np.nan])


def session(args):
    eid, probes = args
    from brainwidemap import load_good_units, load_trials_and_mask
    import dmn_bwm
    one = dmn_bwm.use_online_one()
    from iblatlas.regions import BrainRegions  # noqa: F401
    fs = FULL / f"{eid}.npz"
    if fs.exists() and all((FULL / f"{eid}_{p}.npz").exists() for p, _ in probes):
        return eid, "exists"
    sess_path = Path(one.eid2path(eid))
    before = set(sess_path.rglob("spikes.*"))
    msgs = []
    try:
        trials, mask = load_trials_and_mask(one, str(eid), saturation_intervals=[
            'saturation_stim_plus04', 'saturation_feedback_plus04', 'saturation_move_minus02',
            'saturation_stim_minus04_minus01', 'saturation_stim_plus06', 'saturation_stim_minus06_plus06'])
        units = {}
        tmax = float(np.nanmax(trials["intervals_1"]))
        for probe, pid in probes:
            spikes, clusters = load_good_units(one, pid)
            units[probe] = (spikes, clusters)
            tmax = max(tmax, float(spikes["times"].max()))
        spont = passive_spont(one, eid)
        if np.isfinite(spont[1]):
            tmax = max(tmax, spont[1])
        edges = np.arange(0, tmax + BIN, BIN)
        nb = len(edges) - 1
        task_coll = "alf"
        try:
            cols = sorted({d.rsplit("/", 1)[0] for d in one.list_datasets(eid) if "_ibl_wheel.position" in d})
            task_coll = cols[0] if cols else "alf"
        except Exception:
            pass
        beh = behaviour(one, eid, edges, task_coll)
        tr = {c: trials[c].values.astype(float) for c in TRIAL_COLS}
        np.savez_compressed(fs, eid=eid, BIN=BIN, mask=np.asarray(mask, bool), spont=spont, **tr, **beh)
        for probe, pid in probes:
            spikes, clusters = units[probe]
            nc = len(clusters)
            bi = np.floor(spikes["times"] / BIN).astype(np.int64)
            ok = (bi >= 0) & (bi < nb)
            cnt = np.bincount(spikes["clusters"][ok].astype(np.int64) * nb + bi[ok], minlength=nc * nb)
            cnt = np.minimum(cnt, 255).astype(np.uint8).reshape(nc, nb)
            st = spikes["times"]
            order = np.argsort(st, kind="stable")
            st_s, cl_s = st[order], spikes["clusters"][order]
            wc = {}
            for name, (ev, a, b) in WIN.items():
                e = trials[ev].values.astype(float)
                res = np.full((len(e), nc), np.nan, np.float32)
                fin = np.isfinite(e)
                i0 = np.searchsorted(st_s, e[fin] + a)
                i1 = np.searchsorted(st_s, e[fin] + b)
                rows = np.flatnonzero(fin)
                for r, j0, j1 in zip(rows, i0, i1):
                    res[r] = np.bincount(cl_s[j0:j1], minlength=nc)
                wc[f"win_{name}"] = res
            np.savez_compressed(
                FULL / f"{eid}_{probe}.npz", eid=eid, probe=probe, pid=pid,
                uuids=np.asarray(clusters["uuids"]).astype(str),
                atlas_id=np.asarray(clusters["atlas_id"]),
                xyz=np.c_[clusters["x"], clusters["y"], clusters["z"]].astype(np.float32),
                counts=cnt, **wc)
            msgs.append(f"{probe}:{nc}")
        return eid, "ok " + " ".join(msgs) + f" spont={spont.round(1).tolist()}"
    except Exception:
        return eid, "FAILED\n" + traceback.format_exc()
    finally:
        for f in set(sess_path.rglob("spikes.*")) - before:
            f.unlink(missing_ok=True)


def main():
    FULL.mkdir(parents=True, exist_ok=True)
    import dmn_bwm
    from brainwidemap import bwm_query
    one = dmn_bwm.use_online_one()
    df = bwm_query(one)
    groups = [(eid, list(zip(g.probe_name, g.pid))) for eid, g in df.groupby("eid")]
    if len(sys.argv) > 2:
        groups = groups[: int(sys.argv[2])]
    print(f"{len(df)} insertions, {len(groups)} sessions", flush=True)
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    with ProcessPoolExecutor(nw) as ex:
        for eid, st in ex.map(session, groups):
            print(eid, st, flush=True)


if __name__ == "__main__":
    main()
