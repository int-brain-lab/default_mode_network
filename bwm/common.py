"""Shared paths, region sets and loaders for the BWM default-mode-network pilots."""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"
CACHE = OUT / "cache"
FIGS = OUT / "figs"
for p in (OUT, CACHE, FIGS):
    p.mkdir(parents=True, exist_ok=True)

import os

# ~/dmn holds the PETH bundles and pipeline tables of the bwm_dmn project; override with DMN_DATA.
DMN_DATA = Path(os.environ.get("DMN_DATA", Path.home() / "dmn"))
BUNDLES = DMN_DATA / "concat"
INSERTIONS = DMN_DATA / "pipeline" / "insertions.csv"
ONE_CACHE = Path(os.environ.get("ONE_CACHE", Path.home() / "Downloads" / "ONE"))
TRIALS_PQT = ONE_CACHE / "bwm_tables" / "trials.pqt"

# PETH binning of the ~/dmn bundles: 12.5 ms bins, 6 interleaved 2 ms shifts.
T_BIN = 0.0125
N_SHIFT = 6

# Windows kept from the bundles (bundle key -> role).
WINDOWS = {
    "inter_trial": "rest",          # stimOn -1.15 .. -1.0 s, gap to previous trial end > 1.15 s
    "quiescence": "quiescence",     # stimOn -0.4 .. -0.1 s
    "stimLbLcL": "stim", "stimLbRcL": "stim", "stimRbRcR": "stim", "stimRbLcR": "stim",
    "motor_init": "motor_init",     # firstMovement -0.15 .. 0
    "choiceL": "move", "choiceR": "move",  # firstMovement 0 .. 0.15
    "fback1": "fback1",             # feedback 0 .. 0.3, correct
    "fback0": "fback0",
}

# Mouse fMRI default-mode network at Beryl level. The core is the midline
# prefrontal + retrosplenial + posterior parietal set that the rsfMRI studies
# agree on (Stafford 2014, Sforazzini 2014, Gozzi & Schwarz 2016, Grandjean
# 2020); it is the prefrontal + medial part of the Whitesell 2021 DMN.
# Lateral (TEa, ECT, PERI): temporal association areas sometimes grouped with the
# rodent DMN (Lu et al. 2012 include TeA). NOTE: Whitesell 2021 (Fig. 1E) put them in the
# separate lateral cortical network (~0% of their voxels in the DMN mask), so this
# extension is not supported by the mouse rsfMRI map.
DMN_CORE = ["ACAd", "ACAv", "PL", "ILA", "ORBm", "ORBl", "ORBvl",
            "RSPagl", "RSPd", "RSPv", "VISa", "VISam"]
DMN_LATERAL = ["TEa", "ECT", "PERI"]
DMN_ALL = DMN_CORE + DMN_LATERAL


def insertions():
    return pd.read_csv(INSERTIONS)


_br = None


def regions():
    global _br
    if _br is None:
        from iblatlas.regions import BrainRegions
        _br = BrainRegions()
    return _br


def beryl(ids):
    return np.array(regions().id2acronym(np.asarray(ids), mapping="Beryl"))


def cosmos(ids):
    return np.array(regions().id2acronym(np.asarray(ids), mapping="Cosmos"))


def load_cache(eid, probe):
    f = CACHE / f"{eid}_{probe}.npz"
    if not f.exists():
        return None
    with np.load(f, allow_pickle=True) as d:
        return {k: d[k] for k in d.files}


def done_eids():
    """Sessions whose full extraction finished (listed as ok in the log)."""
    log = OUT / "extract_full.log"
    out = []
    for line in log.read_text().splitlines():
        parts = line.split()
        if len(parts) > 1 and parts[1] in ("ok", "exists"):
            out.append(parts[0])
    return sorted(set(out) | {"004d8fd5-41e7-4f1b-a45b-0d4ad76fe446"})

# Rat DMN of Lu et al. 2012 (PNAS 109:3979, rsfMRI, group ICA), mapped to mouse
# Beryl regions. Orbital VO/LO/rMO -> ORBvl/ORBl/ORBm; PrL -> PL; Cg1/Cg2 -> ACAd/ACAv;
# Au1/AuD/AuV/TeA -> AUDp/AUDd/AUDv/TEa; PPC (PtPR/PtPD) -> VISa, VISrl; RSG/RSD ->
# RSPv/RSPd (+RSPagl); V2M -> VISam, VISpm; dorsal CA1 -> CA1 (tentative in the paper).
# Their modularity analysis splits it into a prefrontal-temporal and a parietal module.
LU_PFC_TEMPORAL = ["ORBm", "ORBl", "ORBvl", "PL", "ACAd", "ACAv", "AUDp", "AUDd", "AUDv", "TEa"]
LU_PARIETAL = ["RSPagl", "RSPd", "RSPv", "VISa", "VISrl", "VISam", "VISpm", "CA1"]
LU_DMN = LU_PFC_TEMPORAL + LU_PARIETAL

# Whitesell et al. 2021 (Neuron 109:545, mouse rsfMRI, group ICA, CCFv3-registered):
# cortical regions with >50% of voxels in the core DMN mask (z > 1.7), Fig. 1E.
WHITESELL_DMN = ["ACAd", "ACAv", "PL", "ILA", "ORBl", "ORBm", "ORBvl", "VISa", "VISam", "RSPagl", "RSPd",
                 "RSPv", "SSp-tr", "SSp-ll", "MOs"]
