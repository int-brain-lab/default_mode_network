"""Figure 1 of the grey-screen report: which cortical (Beryl) regions each rodent DMN paper assigns to the DMN.

Four-set Venn diagram (ellipse layout as in pyvenn's venn4). Region labels are placed in the zone of their
membership pattern; a geometric check asserts that every label lies inside exactly the right ellipses.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "bwm"))
from plotstyle import NATURE_ONEHALF_COL_IN, save_with_caption  # noqa: E402
from common import CONS_DMN, GRANDJEAN_DMN, GRANDJEAN_PFC_UNSPECIFIED, LU_DMN, MANDINO_DMN, WHITESELL_DMN  # noqa: E402

OUTD = HERE / "grey"
SETS = [("Lu et al. 2012\nrat, rsfMRI ICA", LU_DMN, "#1baf7a"),
        ("Whitesell et al. 2021\nmouse, rsfMRI ICA + tracing", WHITESELL_DMN, "#6250d6"),
        ("Grandjean et al. 2020\nmouse, 17-centre rsfMRI", GRANDJEAN_DMN, "#eda100"),
        ("Mandino et al. 2022\nmouse, rsfMRI CAPs + tracing", MANDINO_DMN, "#e34948")]
# pyvenn venn4 geometry
ELL = [((0.350, 0.400), 140), ((0.450, 0.500), 140), ((0.544, 0.500), 40), ((0.644, 0.400), 40)]
W, Hh = 0.72, 0.45
POS = {"1000": (0.14, 0.42), "0100": (0.32, 0.72), "0010": (0.68, 0.72), "0001": (0.85, 0.42),
       "1100": (0.23, 0.59), "1010": (0.29, 0.30), "1001": (0.50, 0.17), "0110": (0.50, 0.66),
       "0101": (0.71, 0.30), "0011": (0.77, 0.59), "1110": (0.35, 0.50), "1101": (0.61, 0.24),
       "1011": (0.39, 0.24), "0111": (0.65, 0.50), "1111": (0.50, 0.38)}


def inside(pt, centre, ang):
    th = np.deg2rad(ang)
    dx, dy = pt[0] - centre[0], pt[1] - centre[1]
    x = dx * np.cos(th) + dy * np.sin(th)
    y = -dx * np.sin(th) + dy * np.cos(th)
    return (x / (W / 2)) ** 2 + (y / (Hh / 2)) ** 2 <= 1


for key, pt in POS.items():  # geometric check of the zone positions
    assert "".join("1" if inside(pt, c, a) else "0" for c, a in ELL) == key, key

regions = sorted(set().union(*[s[1] for s in SETS]))
order = ["ACAd", "ACAv", "PL", "ILA", "ORBm", "ORBl", "ORBvl", "RSPagl", "RSPd", "RSPv", "VISa", "VISam", "VISpm",
         "VISrl", "TEa", "AUDp", "AUDd", "AUDv", "SSp-tr", "SSp-ll", "MOs", "CA1"]
zones = {}
for r in sorted(regions, key=order.index):
    key = "".join("1" if r in s[1] else "0" for s in SETS)
    zones.setdefault(key, []).append(r)

fig, ax = plt.subplots(figsize=(NATURE_ONEHALF_COL_IN, NATURE_ONEHALF_COL_IN * 0.78))
for (c, a), (lab, mem, col) in zip(ELL, SETS):
    ax.add_patch(Ellipse(c, W, Hh, angle=a, fc=col, alpha=0.18, ec=col, lw=1.2))
for key, regs in zones.items():
    x, y = POS[key]
    dy = 0.034
    for i, r in enumerate(regs):
        lab = f"{r}{'†' if (r in GRANDJEAN_PFC_UNSPECIFIED and key[2] == '1') else ''}"
        ax.text(x, y + ((len(regs) - 1) / 2 - i) * dy, lab, ha="center", va="center", fontsize=6.2,
                fontweight="bold" if r in CONS_DMN else "normal", color="k" if r in CONS_DMN else "#333")
for (lab, mem, col), (tx, ty, ha) in zip(SETS, [(0.0, 0.70, "left"), (0.08, 0.97, "left"), (0.92, 0.97, "right"),
                                                 (1.0, 0.70, "right")]):
    name, sub = lab.split("\n")
    ax.text(tx, ty + 0.035, name, color=col, fontsize=6.8, ha=ha, va="bottom", fontweight="bold")
    ax.text(tx, ty, sub, color=col, fontsize=5.6, ha=ha, va="bottom")
ax.text(0.5, 0.03, "in no paper: ECT, PERI (our earlier 'lateral' extension)", ha="center", fontsize=5.8, color="#555")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.06)
ax.set_aspect("equal")
ax.axis("off")

caps = [
    "Cortical regions (Allen CCF, Beryl level) assigned to the default mode network by four rodent resting-state "
    "studies. Lu et al. (2012): rat, BOLD rsfMRI under dexmedetomidine/isoflurane, group ICA (30 components); rat areas "
    "mapped to mouse (PtP $\\rightarrow$ VISa/VISrl, V2M $\\rightarrow$ VISam/VISpm; dorsal CA1 tentative). Whitesell et al. "
    "(2021): mouse, BOLD rsfMRI under halothane, group ICA (5 components) registered to CCFv3; regions with $>$50\\% of "
    "voxels in the core DMN mask. Grandjean et al. (2020): 17 mouse rsfMRI datasets, ACA seed and group ICA; DMN "
    "components named prefrontal (PFC), cingulate/RSP and temporal association areas; \\textdagger{} prefrontal subregions not "
    "specified, mapped here to PL, ILA and ORB. Mandino et al. (2022): mouse rsfMRI co-activation patterns and viral-"
    "tracer projection similarity to an ACA seed; cortical DMN labelled ACA and RSP (a medial visual extension is not "
    "labelled). Subcortical DMN elements (dorsal striatum, dorsal and lateral thalamus, zona incerta, periaqueductal "
    "grey, peri-hippocampal areas) are not shown. Bold: the conservative core used in this report (ACA, RSP, PL, "
    "ORBm). ACA and RSP are in all four maps; PL and ORB in three (Mandino et al. name only cingulate regions). ORBl "
    "and ORBvl have the same support as ORBm and could be added.",
]
save_with_caption(fig, OUTD / "fig1_regions", caps, label="grey_regions", number=1)
import subprocess  # noqa: E402
subprocess.run(["pdftoppm", "-png", "-r", "200", "-singlefile", str(OUTD / "fig1_regions.pdf"), str(OUTD / "fig1_regions")],
               check=True)
print({k: v for k, v in zones.items()})
