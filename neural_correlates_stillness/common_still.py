"""Behavioural state masks for the stillness analyses (50 ms grid of bwm/output/full).

moving       wheel speed > 0.05 rad/s, or a lick, or left-camera motion energy above the session's 60th
             percentile (the definitions of the grey-screen report, bwm/44_windows.py)
still        not moving, with one bin of margin on either side of any moving bin (as in the report)
awake-still  still, at most MAX_STILL_S since the last moving bin, and pupil diameter at or above the 10th
             percentile of the session's task-period pupil (anti-sleep gate: head-fixed mice can enter NREM
             with eyes open after ~40 s of immobility, with a constricted pupil). Bins with missing pupil are
             excluded; sessions with < 50% pupil coverage on the grey screen are flagged pupil_ok = False and
             gated by time since movement only.
grey         passive grey-screen period, first second skipped
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BWM = HERE.parent / "bwm"
sys.path.insert(0, str(BWM))
from common import CONS_DMN, beryl, cosmos, done_eids  # noqa: E402,F401

FULL = BWM / "output" / "full"
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)
SUBJ = pd.read_csv(BWM / "output" / "bwm_insertions.csv").groupby("eid").subject.first()
BIN = 0.05
MAX_STILL_S = 30.0
PUPIL_PCT = 10


def load_session(eid):
    return dict(np.load(FULL / f"{eid}.npz", allow_pickle=True))


def probes(eid):
    return sorted(FULL.glob(f"{eid}_probe*.npz"))


def run_lengths(x):
    """Bins since the last True (inclusive: 0 at True bins); large where never True."""
    idx = np.arange(len(x))
    last = np.maximum.accumulate(np.where(x, idx, -10 ** 9))
    return idx - last


def states(s):
    """Dict of boolean bin masks and helper arrays for one session."""
    nb = len(s["licks"])
    w, lk, me, pu = (s[k].astype(float) for k in ("wheel_speed", "licks", "motion_energy", "pupil"))
    wheel = np.isfinite(w) & (w > 0.05)
    lick = (lk > 0) & (lk < 255)
    me_ok = np.isfinite(me).mean() > 0.5
    face = np.isfinite(me) & (me > np.nanpercentile(me, 60)) if me_ok else np.zeros(nb, bool)
    mov = wheel | lick | face
    still = ~(mov | np.r_[False, mov[:-1]] | np.r_[mov[1:], False])
    tsm = run_lengths(mov)                      # bins since last moving bin
    ttn = run_lengths(mov[::-1])[::-1]          # bins to next moving bin
    sp = s["spont"]
    grey = np.zeros(nb, bool)
    if np.all(np.isfinite(sp)):
        grey[int(sp[0] / BIN) + 20: min(nb, int(sp[1] / BIN))] = True
    i0, i1 = s["intervals_0"], s["intervals_1"]
    task = np.zeros(nb, bool)
    task[int(np.nanmin(i0) / BIN): min(nb, int(np.nanmax(i1) / BIN))] = True
    pupil_ok = bool(np.isfinite(pu[grey]).mean() > 0.5) if grey.any() else False
    awake = still & (tsm * BIN <= MAX_STILL_S)
    if pupil_ok and np.isfinite(pu[task]).sum() > 100:
        thr = np.nanpercentile(pu[task], PUPIL_PCT)
        awake &= np.isfinite(pu) & (pu >= thr)
    # inter-trial gaps (trial end to next trial start), BWM trial mask
    msk = s["mask"].astype(bool)
    iti = np.zeros(nb, bool)
    for t in range(len(i0) - 1):
        if msk[t] and np.isfinite(i1[t]) and np.isfinite(i0[t + 1]):
            a, b = int(np.ceil(i1[t] / BIN)), int(np.floor(i0[t + 1] / BIN))
            if b > a:
                iti[a:min(b, nb)] = True
    return dict(nb=nb, mov=mov, wheel=wheel, lick=lick, face=face, still=still, awake=awake, tsm=tsm, ttn=ttn,
                grey=grey, task=task, iti=iti, me_ok=me_ok, pupil_ok=pupil_ok, pupil=pu, me=me)


def bouts(mask):
    """(start, end) of runs of True."""
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)
