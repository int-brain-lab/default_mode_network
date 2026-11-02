"""Figure 18 of the comprehensive report: stillness correlates in Allen Neuropixels data against IBL (section 13).
Allen values come from the exported replication (comprehensive_report/allen/data, per-neuron and per-session CSVs);
IBL values from 20_neurons.pqt and the Allen-matched rerun 30_allen_matched.py.
Writes comprehensive_report/fig18_still_allen.* and output/31_allen_compare.json.
Usage: python fig_allen.py
"""
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "report"))
from plotstyle import ALLEN_EDGE, NATURE_DOUBLE_COL_IN, allen_color, panel_label, save_with_caption, style_axes  # noqa: E402
from common_still import OUT  # noqa: E402

REP = ROOT.parent / "comprehensive_report"
AD = REP / "allen" / "data"
STEM = REP / "fig18_still_allen"
rng = np.random.default_rng(0)
GREY = "#8a8880"
C_IBL, C_VBN, C_VC = "#222222", "#2a78d6", "#9b59b6"
SUB = (12, 24, 48, 96)


def boot(v, n=2000):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    b = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)])
    return v.mean(), *np.quantile(b, [.025, .975])


def lr(a, b):
    return np.log2((a + 0.1) / (b + 0.1))


def per_mouse(D, col, by="subject"):
    """Mean over neurons per session, then over sessions per mouse (the Allen aggregation)."""
    return D.groupby([by, "eid"])[col].mean().groupby(by).mean()


# ------------------------------------------------------------------ IBL
N = pd.read_parquet(OUT / "20_neurons.pqt")
W = pd.read_parquet(OUT / "30_allen_matched_neurons.pqt")
WS = pd.read_csv(OUT / "30_allen_matched_sessions.csv")
I = N.merge(W, on=["uuid", "eid"])  # sessions with grey-screen wheel that pass the Allen inclusion rule
I["lr_ibl"] = lr(I.r_still, I.r_move)
I["lr_w"] = lr(I.r_still_w, I.r_move_w)
I["lr_w2"] = lr(I.r_still_w2, I.r_move_w2)
I["SP_ibl"], I["MP_ibl"] = I.SP.astype(float), I.MP.astype(float)
for c in ["p_w", "p_w2"] + [f"p_{d}_s{k}" for d in ("ibl", "w") for k in SUB] + ["p_w2_s24"]:
    tag = c[2:]
    I[f"SP_{tag}"] = ((I[c].abs() < .01) & (I[c] > 0)).astype(float)
    I[f"MP_{tag}"] = ((I[c].abs() < .01) & (I[c] < 0)).astype(float)
I["vis"] = I.region.str.startswith("VIS")

# ------------------------------------------------------------------ Allen
A = pd.read_csv(AD / "data_neurons.csv", dtype={"subject": str})
A = A[A.keep == 1].copy()
A["eid"] = A.session.astype(str) + "_" + A.subject
A["SP"], A["MP"] = (A.pref == 1).astype(float), (A.pref == -1).astype(float)
AM = pd.read_csv(AD / "data_mice.csv", dtype={"subject": str})
AS = pd.read_csv(AD / "data_sessions.csv", dtype={"subject": str})
ADIV = pd.read_csv(AD / "data_divisions_per_mouse.csv", dtype={"subject": str})
APRE = pd.read_csv(AD / "data_premovement_curves_pct.csv")
try:
    from iblatlas.regions import BrainRegions
    _br = BrainRegions()
    A["beryl"] = _br.acronym2acronym(A.region.values, mapping="Beryl")
except Exception:  # noqa: BLE001
    A["beryl"] = A.region

S = {}  # numbers for the text
# a: mouse-level rate ratio
groups_a = [("IBL\nfull", per_mouse(I, "lr_ibl"), C_IBL), ("IBL\nwheel", per_mouse(I, "lr_w"), C_IBL),
            ("IBL\n2 cm/s", per_mouse(I, "lr_w2"), C_IBL),
            ("Allen\nVBN", AM[AM.dataset == "VBN"].set_index("subject").mean_lr, C_VBN),
            ("Allen\nVC", AM[AM.dataset == "VC"].set_index("subject").mean_lr, C_VC)]
S["rate_ratio"] = {}
for lab, v, _ in groups_a:
    mu, lo, hi = boot(v)
    S["rate_ratio"][lab.replace("\n", " ")] = dict(mean=mu, lo=lo, hi=hi, n_mice=int(v.notna().sum()),
                                                  n_neg=int((v < 0).sum()))
S["ibl_n"] = dict(sessions=int(I.eid.nunique()), mice=int(I.subject.nunique()), neurons=int(len(I)),
                  sessions_all=int(N.eid.nunique()))
S["ibl_sessions"] = {k: float(WS[k].median()) for k in ["n_epS", "n_epM", "n_epS_w", "n_epM_w", "n_epS_w2",
                                                         "n_epM_w2", "still_face_frac"]}
S["ibl_sessions"]["still_min_w"] = float((WS.n_still_w * 0.05 / 60).median())
S["allen_sessions"] = {k: float(AS[k].median()) for k in ["n_ep_still", "n_ep_move", "still_min"]}

# b, c: preference fractions against the number of still epochs
pref = {}
for d, lab in (("ibl", "IBL, full definition"), ("w", "IBL, wheel only")):
    allk = S["ibl_sessions"]["n_epS" if d == "ibl" else "n_epS_w"]
    xs = list(SUB) + [allk]
    for kind in ("SP", "MP"):
        cols = [f"{kind}_{d}_s{k}" for k in SUB] + [f"{kind}_{d}" if d == "ibl" else f"{kind}_w"]
        pref[(d, kind)] = (xs, [boot(per_mouse(I, c) * 100) for c in cols], lab)
S["pref"] = {f"{d}_{kind}": dict(zip([str(x) for x in v[0]], [m[0] for m in v[1]])) for (d, kind), v in pref.items()}
S["pref"]["w2_SP_s24"], S["pref"]["w2_MP_s24"] = per_mouse(I, "SP_w2_s24").mean() * 100, \
    per_mouse(I, "MP_w2_s24").mean() * 100
for kind in ("SP", "MP"):
    v = AM.set_index("subject")["still_pref" if kind == "SP" else "move_pref"] * 100
    S["pref"][f"allen_{kind}"] = boot(v)

# d: divisions (per session >= 20 neurons, then mouse)
DIVS = [("MB", "midbrain"), ("TH", "thalamus"), ("HPF", "hippocampal f."), ("VIS", "visual cortex"),
        ("Isocortex", "all cortex")]


def ibl_div(col, d):
    D = I[I.vis] if d == "VIS" else I[I.cosmos == d]
    g = D.groupby(["subject", "eid"])[col].agg(["mean", "count"])
    return g[g["count"] >= 20]["mean"].groupby("subject").mean()


def allen_div(ds, d):
    if d == "all cortex" or d == "Isocortex":
        return pd.Series(dtype=float)
    dd = "Isocortex" if d == "VIS" else d  # Allen cortex units are 99% visual areas
    x = ADIV[(ADIV.dataset == ds) & (ADIV.cosmos == dd)]
    return x.set_index("subject").mean_log2rr


S["divisions"] = {}
for d, lab in DIVS:
    row = {}
    for name, v in (("IBL wheel", ibl_div("lr_w", d)), ("IBL full", ibl_div("lr_ibl", d)),
                    ("VBN", allen_div("VBN", d)), ("VC", allen_div("VC", d))):
        if len(v):
            mu, lo, hi = boot(v)
            row[name] = dict(mean=mu, lo=lo, hi=hi, n=int(len(v)), n_neg=int((v < 0).sum()))
    S["divisions"][lab] = row

# short-bout check: IBL still rate 0.1-2 s after a movement (most Allen stillness is this close to running)
N["lr_early"] = lr(N[["tsm_0", "tsm_1", "tsm_2"]].mean(axis=1), N.r_move)
S["early_still"] = {lab: N[m].groupby("subject").lr_early.mean().mean() for lab, m in
                    (("all", N.index == N.index), ("visual cortex", N.region.str.startswith("VIS")),
                     ("thalamus", N.cosmos == "TH"), ("hippocampal f.", N.cosmos == "HPF"))}

# e: region-level agreement (Beryl regions, mouse means, >= 3 mice with >= 5 neurons in both)


def region_means(D, col, reg):
    g = D.groupby([reg, "subject"])[col].agg(["mean", "count"])
    g = g[g["count"] >= 5]["mean"].groupby(level=0).agg(["mean", "count"])
    return g[g["count"] >= 3]["mean"]


RI = region_means(I, "lr_w", "region")
RA = region_means(A, "log2rr", "beryl")
RR = pd.concat([RI.rename("ibl"), RA.rename("allen")], axis=1).dropna()
RR = RR[~RR.index.isin(["root", "void"])]
rho = spearmanr(RR.ibl, RR.allen)
S["regions"] = dict(n=int(len(RR)), spearman=float(rho.correlation), p=float(rho.pvalue),
                    allen_minus_ibl=float((RR.allen - RR.ibl).mean()),
                    allen_above_ibl=int((RR.allen > RR.ibl).sum()))

# f: pre-movement firing
O = np.load(OUT / "30_allen_matched_onsets.npz")
t = O["t"]
base = (t >= -2) & (t < -1)
pre = (t >= -0.5) & (t < -0.05)
subj = WS.set_index("eid").reindex(O["eid"]).index.map(N.groupby("eid").subject.first())


def pct_curves(C):
    P = 100 * (C / np.nanmean(C[:, base], 1, keepdims=True) - 1)
    ok = np.isfinite(P).all(1)
    return pd.DataFrame(P[ok]).groupby(np.asarray(subj)[ok]).mean()


PI = pct_curves(O["w"])
ta = APRE.t_s.values
PA = APRE.drop(columns="t_s").T.values
S["premove"] = {}
for name, P, tt in (("IBL wheel onsets", PI.values, t), ("Allen running onsets", PA, ta)):
    m = (tt >= -0.5) & (tt < -0.05)
    v = P[:, m].mean(1)
    mu, lo, hi = boot(v)
    S["premove"][name] = dict(mean=mu, lo=lo, hi=hi, n_mice=int(len(v)), n_pos=int((v > 0).sum()),
                              p=float(wilcoxon(v).pvalue))
PIB = pct_curves(O["ibl"])
v = PIB.values[:, pre].mean(1)
S["premove"]["IBL any-movement onsets"] = dict(zip(["mean", "lo", "hi"], boot(v)), n_mice=int(len(v)),
                                              n_pos=int((v > 0).sum()))
# the onsets of section 12.4 (any movement after >= 3 s of awake stillness, 10_neurons.py), same window and baseline
PE = np.load(OUT / "10_peths.npz", allow_pickle=True)
assert (PE["uuid"].astype(str) == N.uuid.values).all()
t3, ok3 = PE["t_on"], np.isfinite(PE["on"]).all(1)
pop3 = pd.DataFrame(PE["on"][ok3]).groupby([N.subject.values[ok3], N.eid.values[ok3]]).mean()
b3, w3 = (t3 >= -2) & (t3 < -1), (t3 >= -0.5) & (t3 < -0.05)
pct3 = 100 * (pop3 / pop3.loc[:, b3].mean(1).values[:, None] - 1)
v = pct3.groupby(level=0).mean().loc[:, w3].mean(1).values
v = v[np.isfinite(v)]
S["premove"]["IBL any-movement onsets, >=3 s still"] = dict(zip(["mean", "lo", "hi"], boot(v)), n_mice=int(len(v)),
                                                           n_pos=int((v > 0).sum()))

# ------------------------------------------------------------------ figure
fig = plt.figure(figsize=(NATURE_DOUBLE_COL_IN, 4.9))
gs = fig.add_gridspec(2, 3, hspace=.6, wspace=.45, left=.08, right=.98, top=.95, bottom=.13)

ax = fig.add_subplot(gs[0, 0])
for i, (lab, v, c) in enumerate(groups_a):
    v = v.dropna()
    ax.scatter(i + rng.uniform(-.18, .18, len(v)), v, s=3, color=c, alpha=.45, lw=0)
    mu, lo, hi = boot(v)
    ax.plot([i, i], [lo, hi], color="k", lw=1)
    ax.plot([i - .25, i + .25], [mu, mu], color="k", lw=1)
ax.axhline(0, color=GREY, lw=.5, ls=":")
ax.set_xticks(range(len(groups_a)))
ax.set_xticklabels([g[0] for g in groups_a], fontsize=5)
ax.set_ylabel("log$_2$ rate ratio (still / moving)")
ax.set_ylim(-1.5, .6)
style_axes(ax, categorical_x=True)
panel_label(ax, "a")

for j, kind in enumerate(("SP", "MP")):
    ax = fig.add_subplot(gs[0, 1 + j])
    for d, ls in (("ibl", "-"), ("w", "--")):
        xs, vals, lab = pref[(d, kind)]
        mu = [v[0] for v in vals]
        ax.plot(xs, mu, ls, color=C_IBL, lw=1, marker="o", ms=2, label=lab)
        ax.fill_between(xs, [v[1] for v in vals], [v[2] for v in vals], color=C_IBL, alpha=.12, lw=0)
    for ds, c in (("VBN", C_VBN), ("VC", C_VC)):
        x = AS[(AS.dataset == ds) & AS.n_ep_still.notna()]
        ax.scatter(x.n_ep_still, x["still_pref" if kind == "SP" else "move_pref"] * 100, s=5, color=c, lw=0,
                   alpha=.8, label=f"Allen {ds} sessions")
    ax.set_xscale("log")
    ax.set_xticks([10, 30, 100, 300])
    ax.set_xticklabels(["10", "30", "100", "300"])
    ax.minorticks_off()
    ax.set_xlabel("still epochs per session (0.5 s)")
    ax.set_ylabel("still-preferring (%)" if kind == "SP" else "movement-preferring (%)")
    style_axes(ax, categorical_x=True)
    if kind == "SP":
        ax.legend(frameon=False, fontsize=4.5, loc="upper left", handlelength=1.5)
    panel_label(ax, "bc"[j])

ax = fig.add_subplot(gs[1, 0])
for i, (d, lab) in enumerate(DIVS):
    for k, (v, c, off) in enumerate(((ibl_div("lr_w", d), C_IBL, -.25), (allen_div("VBN", d), C_VBN, 0),
                                     (allen_div("VC", d), C_VC, .25))):
        if not len(v):
            continue
        ax.scatter(i + off + rng.uniform(-.06, .06, len(v)), v, s=2, color=c, alpha=.4, lw=0)
        mu, lo, hi = boot(v)
        ax.plot([i + off] * 2, [lo, hi], color=c, lw=1)
        ax.scatter([i + off], [mu], s=10, color=c, edgecolor="white", lw=.4, zorder=3)
ax.axhline(0, color=GREY, lw=.5, ls=":")
ax.set_xticks(range(len(DIVS)))
ax.set_xticklabels([lab for _, lab in DIVS], rotation=35, ha="right", rotation_mode="anchor", fontsize=5.5)
ax.set_ylabel("log$_2$ rate ratio per mouse")
ax.set_ylim(-1.6, .8)
style_axes(ax, categorical_x=True)
panel_label(ax, "d")

ax = fig.add_subplot(gs[1, 1])
lim = (-1.2, .5)
ax.plot(lim, lim, color=GREY, lw=.5, ls="--")
ax.axhline(0, color=GREY, lw=.4, ls=":")
ax.axvline(0, color=GREY, lw=.4, ls=":")
for r, row in RR.iterrows():
    ax.scatter(row.ibl, row.allen, s=9, color=allen_color(r), edgecolor=ALLEN_EDGE, lw=.2)
    ax.annotate(r, (row.ibl, row.allen), fontsize=3, xytext=(1.5, 1.5), textcoords="offset points")
ax.text(.04, .96, f"ρ = {rho.correlation:.2f}, {len(RR)} regions", transform=ax.transAxes, va="top", fontsize=5.5)
ax.set_xlim(*lim)
ax.set_ylim(*lim)
ax.set_xlabel("IBL, wheel only (log$_2$ ratio)")
ax.set_ylabel("Allen (log$_2$ ratio)")
style_axes(ax)
panel_label(ax, "e")

ax = fig.add_subplot(gs[1, 2])
for P, tt, c, lab in ((PI.values, t, C_IBL, f"IBL wheel onsets ({len(PI)} mice)"),
                      (PA, ta, C_VBN, f"Allen running onsets ({len(PA)} mice)")):
    mu = np.nanmean(P, 0)
    B = np.array([np.nanmean(P[rng.integers(0, len(P), len(P))], 0) for _ in range(1000)])
    ax.fill_between(tt, *np.nanquantile(B, [.025, .975], 0), color=c, alpha=.18, lw=0)
    ax.plot(tt, mu, color=c, lw=1, label=lab)
ax.axvspan(-.5, -.05, color=GREY, alpha=.12, lw=0)
ax.axhline(0, color=GREY, lw=.4, ls=":")
ax.axvline(0, color=GREY, lw=.4)
ax.set_xlim(-2, 1)
ax.set_xlabel("time from movement onset (s)")
ax.set_ylabel("population rate change (%)")
ax.legend(frameon=False, fontsize=4.5, loc="upper left")
style_axes(ax)
panel_label(ax, "f")

R = S["rate_ratio"]
save_with_caption(fig, STEM, [
    f"Mouse-level log$_2$ rate ratio (still/moving; neurons averaged per session, sessions per mouse). IBL: the "
    f"{S['ibl_n']['sessions']} sessions ({S['ibl_n']['mice']} mice) with a grey-screen wheel trace, with the report's "
    "movement definition (wheel, licks, face motion), wheel movement only ($>$0.05\\,rad/s), and wheel movement "
    "above the Allen running threshold (2\\,cm/s at the wheel rim). Allen: Visual Behavior (VBN) and Visual Coding "
    "(VC) mice, running only. Black: mean and 95\\% bootstrap CI over mice.",
    "Still-preferring neurons (Welch $p<0.01$, 0.5\\,s epochs) against the number of still epochs per session. "
    "IBL lines: still epochs randomly subsampled to 12--96, and all (session median); mean over mice and 95\\% CI. "
    "Dots: Allen sessions.",
    "Movement-preferring neurons, as in b.",
    "Rate ratio by division (sessions with $\\geq$20 neurons in the division, averaged per mouse): IBL wheel only "
    "(black), Allen VBN (blue), Allen VC (purple); large dots, mean and 95\\% CI over mice. Allen cortical units "
    "are in visual areas; IBL is shown for visual areas and for all cortex.",
    "Region-level agreement: mean over mice of the region's log$_2$ ratio (Allen CCF labels mapped to Beryl; "
    "regions with $\\geq$3 mice of $\\geq$5 neurons in both datasets; Allen colours). Dashed: identity.",
    "Population rate around movement onsets after $\\geq$1\\,s of awake stillness, percent change from "
    "$-2$ to $-1$\\,s; mean over mice and 95\\% bootstrap CI. IBL onsets are wheel onsets (wheel-only definition), "
    "Allen onsets running onsets (curves from the Allen export). Shaded: $-0.5$ to $-0.05$\\,s test window."],
    label="still_allen", number=18,
    intro="Neural correlates of stillness in Allen Institute Neuropixels data against IBL.")
subprocess.run(["pdftoppm", "-png", "-r", "200", "-singlefile", f"{STEM}.pdf", str(STEM)], check=True)
plt.close(fig)


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


json.dump(clean(S), open(OUT / "31_allen_compare.json", "w"), indent=1)
print(json.dumps(clean(S), indent=1))
