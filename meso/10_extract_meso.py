"""Mesoscope extraction for the DMN analyses (approach C).

For the canonical mesoscope sessions (as in mesoscope/pilot_analyses/dorsal_maps),
per session -> output/sessions/<eid>.npz:
  xyz (mm), atlas_id, xyz_source, signal name
  Xbin   (neurons x bins) float16: 1 s bin means of the deconvolved trace over
         the whole recording (task + passive), for co-fluctuation analyses
  Xwin   (neurons x trials x windows) float32: per-trial window means for WIN
  trials columns, wheel speed / licks / motion energy / pupil on the 1 s grid
  spont: passive spontaneous interval if present
Signal files downloaded by this run are deleted afterwards.
"""
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import os
# loaders from int-brain-lab/mesoscope (pilot_analyses/dorsal_maps); MESOSCOPE_REPO overrides ~/mesoscope
DM = Path(os.environ.get("MESOSCOPE_REPO", Path.home() / "mesoscope")) / "pilot_analyses" / "dorsal_maps"
sys.path.insert(0, str(DM.parents[1]))
sys.path.insert(0, str(DM))
from common import interp_window_means, isocortex_mask, remove_wheel_jumps, trial_mask  # noqa: E402
from extract_meso import SkipFOV, canonical_eids, frame_times_for, task_collection  # noqa: E402
from meso_loader import _REGION_KEYS, _SIGNAL_DATASETS  # noqa: E402

OUT = Path(__file__).resolve().parent / "output" / "sessions"
BIN = 1.0
ROI_CHUNK = 256
WIN = {  # event, t0, t1
    "pre": ("stimOn_times", -0.8, -0.1),
    "stim": ("stimOn_times", 0.0, 0.4),
    "choice": ("firstMovement_times", -0.3, 0.1),
    "fback": ("feedback_times", 0.0, 0.4),
    "post_fb": ("feedback_times", 1.0, 1.6),
}
TRIAL_COLS = ["stimOn_times", "firstMovement_times", "feedback_times", "intervals_0", "intervals_1",
              "feedbackType", "choice", "probabilityLeft", "contrastLeft", "contrastRight"]


def _binned_mean(t, x, edges):
    ok = np.isfinite(t) & np.isfinite(x)
    t, x = t[ok], x[ok]
    i = np.searchsorted(edges, t, side="right") - 1
    v = (i >= 0) & (i < len(edges) - 1)
    s = np.bincount(i[v], weights=x[v], minlength=len(edges) - 1)
    n = np.bincount(i[v], minlength=len(edges) - 1)
    with np.errstate(invalid="ignore"):
        return (s / n).astype(np.float32)


def fov(one, eid, f, names, regions, xyz_key, iv_trials, iv_bins):
    load = lambda d: one.load_dataset(eid, d, collection=f)  # noqa: E731
    region_key = next((f"mpciROIs.{k}.npy" for k in _REGION_KEYS if f"mpciROIs.{k}.npy" in names), None)
    if region_key is None or xyz_key not in names:
        raise SkipFOV("no region / position dataset")
    sig_name = next((s for s in _SIGNAL_DATASETS if s in names), None)
    if sig_name is None:
        raise SkipFOV("no signal dataset")
    sig = np.load(one.load_dataset(eid, sig_name, collection=f, download_only=True), mmap_mode="r")
    region_ids = load(region_key)
    is_neuron = load("mpciROIs.mpciROITypes.npy").astype(bool)
    xyz = load(xyz_key)
    stack_pos = load("mpciROIs.stackPos.npy")
    timeshift = load("mpciStack.timeshift.npy")
    offsets = timeshift[stack_pos[:, len(timeshift.shape)]].astype(np.float64)
    ft = frame_times_for(one, eid, f, sig.shape[0])
    if not len(region_ids) == len(is_neuron) == len(xyz) == sig.shape[1]:
        raise SkipFOV("ROI count mismatch")
    keep = np.flatnonzero(is_neuron & isocortex_mask(region_ids, regions) & np.all(np.isfinite(xyz), 1))
    Xw = np.empty((keep.size, iv_trials.shape[0], iv_trials.shape[1]), np.float32)
    Xb = np.empty((keep.size, iv_bins.shape[0]), np.float16)
    inside = (iv_bins[:, 0] >= ft[0]) & (iv_bins[:, 1] <= ft[-1])
    for i0 in range(0, keep.size, ROI_CHUNK):
        idx = keep[i0:i0 + ROI_CHUNK]
        chunk = np.ascontiguousarray(sig[:, idx].T, dtype=np.float32)
        Xw[i0:i0 + ROI_CHUNK] = interp_window_means(chunk, ft, offsets[idx], iv_trials)
        b = interp_window_means(chunk, ft, offsets[idx], iv_bins, n_sub=5)
        b[:, ~inside] = np.nan
        Xb[i0:i0 + ROI_CHUNK] = b.astype(np.float16)
    del sig
    return Xw, Xb, xyz[keep], region_ids[keep], sig_name


def behaviour(one, eid, task, edges):
    out = {}
    try:
        w = one.load_object(eid, "wheel", collection=task)
        t, p = w["timestamps"], remove_wheel_jumps(w["position"])
        tt = np.arange(t[0], t[-1], 0.01)
        out["wheel_speed"] = _binned_mean(tt, np.abs(np.gradient(np.interp(tt, t, p), tt)), edges)
    except Exception:
        out["wheel_speed"] = np.full(len(edges) - 1, np.nan, np.float32)
    for key, ds in (("licks", "licks.times.npy"),):
        try:
            lk = one.load_dataset(eid, ds, collection="alf")
            out[key] = np.histogram(lk, edges)[0].astype(np.float32)
        except Exception:
            out[key] = np.full(len(edges) - 1, np.nan, np.float32)
    for cam in ("left", "right", "body"):
        try:
            me = one.load_dataset(eid, f"{cam}Camera.ROIMotionEnergy.npy", collection="alf")
            ct = one.load_dataset(eid, f"_ibl_{cam}Camera.times.npy", collection="alf")
            n = min(len(me), len(ct))
            out["motion_energy"] = _binned_mean(ct[:n], me[:n], edges)
            out["me_camera"] = cam
            break
        except Exception:
            continue
    out.setdefault("motion_energy", np.full(len(edges) - 1, np.nan, np.float32))
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
    for d in one.list_datasets(eid):
        if "passivePeriods.intervalsTable" in d:
            try:
                pp = one.load_dataset(eid, d.split("/")[-1], collection=d.rsplit("/", 1)[0])
                if "spontaneousActivity" in pp:
                    return np.array(pp["spontaneousActivity"].values[:2], float)
            except Exception:
                pass
    return np.array([np.nan, np.nan])


def extract(eid):
    out = OUT / f"{eid}.npz"
    if out.exists():
        return eid, "exists"
    from one.api import ONE
    from iblatlas.regions import BrainRegions
    one = ONE()
    regions = BrainRegions()
    sess_path = Path(one.eid2path(eid))
    before = set(sess_path.rglob("mpci.ROIActivity*.npy"))
    try:
        task = task_collection(one, eid)
        trials = one.load_object(eid, "trials", collection=task)
        if "intervals" in trials and "intervals_1" not in trials:
            trials["intervals_0"] = np.asarray(trials["intervals"])[:, 0]
            trials["intervals_1"] = np.asarray(trials["intervals"])[:, 1]
        mask = trial_mask(trials)
        iv_trials = np.stack([np.c_[np.asarray(trials[ev], float) + a, np.asarray(trials[ev], float) + b]
                              for ev, a, b in WIN.values()], axis=1)
        iv_trials = np.nan_to_num(iv_trials, nan=-1e6)
        spont = passive_spont(one, eid)
        fovs = sorted(one.list_collections(eid, collection="alf/FOV_*"), key=lambda c: int(c[-2:]))
        names = {f: {d.split("/")[-1] for d in one.list_datasets(eid, collection=f)} for f in fovs}
        xyz_key = ("mpciROIs.mlapdv.npy" if all("mpciROIs.mlapdv.npy" in n for n in names.values())
                   else "mpciROIs.mlapdv_estimate.npy")
        t_end = np.nanmax([np.nanmax(np.asarray(trials["intervals_1"], float)), np.nan_to_num(spont[1])])
        ft0 = frame_times_for(one, eid, fovs[0], np.load(one.load_dataset(
            eid, "mpci.ROIActivityDeconvolved.npy", collection=fovs[0], download_only=True), mmap_mode="r").shape[0])
        t_end = max(t_end, ft0[-1])
        edges = np.arange(0, t_end + BIN, BIN)
        iv_bins = np.c_[edges[:-1], edges[1:]]
        parts, skipped = [], []
        for f in fovs:
            try:
                parts.append(fov(one, eid, f, names[f], regions, xyz_key, iv_trials, iv_bins))
            except SkipFOV as e:
                skipped.append(f"{f.split('/')[-1]}: {e}")
        if not parts:
            raise ValueError(f"no usable FOV ({skipped})")
        beh = behaviour(one, eid, task, edges)
        tr = {c: np.asarray(trials[c], float) for c in TRIAL_COLS if c in trials}
        np.savez_compressed(
            out, eid=eid, BIN=BIN, mask=mask, spont=spont, win_names=np.array(list(WIN)),
            Xwin=np.concatenate([p[0] for p in parts]), Xbin=np.concatenate([p[1] for p in parts]),
            xyz=np.concatenate([p[2] for p in parts]) / 1e3, atlas_id=np.concatenate([p[3] for p in parts]),
            xyz_source=xyz_key.split(".")[1], signal=parts[0][4], skipped=np.array(skipped, dtype=str),
            **tr, **beh)
        n = sum(len(p[2]) for p in parts)
        return eid, f"ok n={n} bins={len(edges) - 1} spont={spont.round(0).tolist()} me={beh.get('me_camera')}"
    except Exception:
        return eid, "FAILED\n" + traceback.format_exc()
    finally:
        for f in set(sess_path.rglob("mpci.ROIActivity*.npy")) - before:
            f.unlink(missing_ok=True)


if __name__ == "__main__":
    from one.api import ONE
    OUT.mkdir(parents=True, exist_ok=True)
    eids = canonical_eids(ONE())
    if len(sys.argv) > 2:
        eids = eids[: int(sys.argv[2])]
    with ProcessPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 2) as ex:
        for eid, status in ex.map(extract, eids):
            print(eid, status, flush=True)
