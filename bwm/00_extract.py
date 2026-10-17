"""Reduce each ~/dmn PETH bundle to per-trial spike counts per neuron for the
windows in common.WINDOWS (non-overlapping 12.5 ms bins summed over the window).

Output: output/cache/<eid>_<probe>.npz with uuids, atlas ids, xyz and, per
window, '<key>__trials' (trial indices) and '<key>__counts' (trials x neurons),
'<key>__dur' (window length in s).
"""
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common import BUNDLES, CACHE, N_SHIFT, T_BIN, WINDOWS, insertions


def extract(args):
    eid, probe = args
    out = CACHE / f"{eid}_{probe}.npz"
    if out.exists():
        return eid, probe, "skip"
    f = BUNDLES / f"{eid}_{probe}.npy"
    if not f.exists():
        return eid, probe, "missing"
    d = np.load(f, allow_pickle=True).flat[0]
    names = d["trial_names"]
    res = {"uuids": np.asarray(d["uuids"]).astype(str), "ids": np.asarray(d["ids"]),
           "xyz": np.asarray(d["xyz"], dtype=np.float32)}
    for key in WINDOWS:
        w = d["ws"][names.index(key)]
        if w is None or len(np.shape(w)) != 3:
            res[f"{key}__trials"] = np.zeros(0, int)
            res[f"{key}__counts"] = np.zeros((0, len(res["uuids"])), np.float32)
            res[f"{key}__dur"] = np.float32(0)
            continue
        base = w[:, :, 0::N_SHIFT]  # shift 0 = non-overlapping bins
        res[f"{key}__trials"] = np.asarray(d["trial_indices"][key])
        res[f"{key}__counts"] = base.sum(-1).astype(np.float32)
        res[f"{key}__dur"] = np.float32(base.shape[-1] * T_BIN)
    np.savez_compressed(out, **res)
    return eid, probe, "ok"


def main():
    ins = insertions()
    todo = list(zip(ins.eid, ins.probe))
    with ProcessPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 6) as ex:
        for i, (eid, probe, st) in enumerate(ex.map(extract, todo)):
            if st != "skip":
                print(i, eid, probe, st, flush=True)


if __name__ == "__main__":
    main()
