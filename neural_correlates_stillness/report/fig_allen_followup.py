"""Figure 19 of the comprehensive report: the Allen follow-up analyses next to their IBL counterparts.
Allen values come from the exported follow-up tables (comprehensive_report/allen/data/fu_*.csv: circular-shift null,
onset rules, frontal core - RSP coupling in OpenScope); IBL values from 32_null_onsets.py.
Writes comprehensive_report/fig19_allen_followup.* and output/33_allen_followup.json.
Usage: python fig_allen_followup.py
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
from plotstyle import NATURE_DOUBLE_COL_IN, panel_label, save_with_caption, style_axes  # noqa: E402
from common_still import OUT  # noqa: E402

REP = ROOT.parent / "comprehensive_report"
AD = REP / "allen" / "data"
STEM = REP / "fig19_allen_followup"
rng = np.random.default_rng(0)
GREY = "#8a8880"
C_IBL, C_VBN, C_FR = "#222222", "#2a78d6", "#d0663a"


def boot(v, n=2000):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    b = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)])
    return v.mean(), *np.quantile(b, [.025, .975])


def summ(v):
    mu, lo, hi = boot(v)
    return dict(mean=mu, lo=lo, hi=hi, n=int(np.isfinite(v).sum()), n_pos=int((np.asarray(v) > 0).sum()))


S = {}
# ------------------------------------------------------------------ null calibration
NI = pd.read_csv(OUT / "32_null_sessions.csv")
NA = pd.read_csv(AD / "fu_null_sessions.csv", dtype={"subject": str})
NA = NA.rename(columns={"null_still_mean": "null_still", "null_move_mean": "null_move"})
MI = NI.groupby("subject").mean(numeric_only=True)
MA = NA.groupby("subject").mean(numeric_only=True)
S["null"] = {}
for name, M, tag in (("IBL", MI, ""), ("IBL 24 still epochs", MI, "_s24"), ("Allen VBN", MA, "")):
    o = {}
    for k in ("still", "move"):
        o[f"obs_{k}"] = summ(100 * M[f"obs_{k}{tag}"])
        o[f"null_{k}"] = summ(100 * M[f"null_{k}{tag}"])
        o[f"corr_{k}"] = summ(100 * (M[f"obs_{k}{tag}"] - M[f"null_{k}{tag}"]))
    S["null"][name] = o
S["null"]["IBL_sessions"] = dict(
    n=len(NI), mice=int(NI.subject.nunique()),
    above_p95_still=int((NI.obs_still > NI.null_still_p95).sum()), above_p95_move=int((NI.obs_move > NI.null_move_p95).sum()),
    rho_still_null_vs_move_epochs=float(spearmanr(NI.n_ep_move, NI.null_still).correlation),
    rho_move_null_vs_still_epochs=float(spearmanr(NI.n_ep_still, NI.null_move).correlation),
    med_ep_still=float(NI.n_ep_still.median()), med_ep_move=float(NI.n_ep_move.median()))
S["null"]["Allen_sessions"] = dict(
    n=len(NA), mice=int(NA.subject.nunique()),
    above_p95_still=int((NA.obs_still > NA.null_still_p95).sum()), above_p95_move=int((NA.obs_move > NA.null_move_p95).sum()))

# ------------------------------------------------------------------ onset rules
PI = pd.read_csv(OUT / "32_premove_sessions.csv")
PA = pd.read_csv(AD / "fu_premove_sessions.csv", dtype={"subject": str}).rename(columns={"delta": "rise"})
PA["kind"] = "running"
RISE = [("IBL any", PI[PI.kind == "any"], C_IBL), ("IBL wheel", PI[PI.kind == "wheel"], C_IBL),
        ("Allen running", PA, C_VBN)]
S["onsets"] = {}
for name, P, _ in RISE:
    o = {}
    for rule in ("1s", "3s"):
        c = P[P.rule == rule]
        m = 100 * c.groupby("subject").rise.mean()
        e = 100 * c.groupby("subject").early.mean()
        o[rule] = dict(**summ(m), p=float(wilcoxon(m).pvalue), sessions=int(len(c)), early=float(e.mean()),
                       p_early=float(wilcoxon(e).pvalue), med_onsets=float(c.n_onsets.median()))
    key = "eid" if "eid" in P else ["subject", "date"]
    W = P.pivot_table(index=key, columns="rule", values="rise").dropna()
    d = 100 * (W["3s"] - W["1s"])
    o["paired"] = dict(n=int(len(d)), diff=float(d.mean()), n_pos=int((d > 0).sum()), p=float(wilcoxon(d).pvalue))
    S["onsets"][name] = o

CI = np.load(OUT / "32_premove_curves.npz")
subj = PI.drop_duplicates("eid").set_index("eid").subject
CL = pd.read_csv(AD / "fu_premove_curves_long.csv", dtype={"subject": str})


def ibl_curves(kind, rule):
    m = (CI["kind"] == kind) & (CI["rule"] == rule)
    return CI["t"], 100 * pd.DataFrame(CI["curve"][m]).groupby(subj.reindex(CI["eid"][m]).values).mean().values


def allen_curves(rule):
    d = CL[CL.rule == rule]
    Y = d.pivot_table(index=["subject", "date"], columns="t_s", values="frac_change").groupby(level=0).mean()
    return Y.columns.values.astype(float), 100 * Y.values


# ------------------------------------------------------------------ frontal core - RSP coupling (Allen OpenScope)
FR = pd.read_csv(AD / "fu_coupling_front_rsp.csv", dtype={"subject": str})
CP = pd.read_csv(AD / "fu_coupling_pairs.csv", dtype={"subject": str})
g = FR.groupby(["mask", "subject"]).agg(pr=("pct_raw", "mean"), pc=("pct_resid", "mean"), rz=("resid", "mean")).reset_index()
S["coupling"] = dict(sessions=int(CP.session.nunique()), sessions_front_rsp=int(FR.session.nunique()),
                     mice=int(FR.subject.nunique()), pairs=int((CP["mask"] == "all").sum()))
for mk, h in g.groupby("mask"):
    S["coupling"][mk] = dict(pct_raw=float(h.pr.mean()), pct_corr=float(h.pc.mean()), n_above50=int((h.pr > 50).sum()),
                             n_above90=int((h.pr > 90).sum()), resid_z=float(h.rz.mean()),
                             n_resid_pos=int((h.rz > 0).sum()), p=float(wilcoxon(h.rz).pvalue))
sm = g.pivot(index="subject", columns="mask", values="rz")
dd = sm["still"] - sm["moving"]
a = CP[CP["mask"] == "all"]
S["coupling"]["still_minus_moving"] = dict(mean=float(dd.mean()), n_pos=int((dd > 0).sum()), p=float(wilcoxon(dd).pvalue))
S["coupling"]["distance"] = dict(slope_per_mm=float(np.polyfit(a.dist_um / 1000, a.z, 1)[0]),
                                 rho=float(spearmanr(a.dist_um, a.z).correlation))
S["coupling"]["front_rsp_pairs"] = sorted(FR[FR["mask"] == "all"].pair.unique().tolist())

# ------------------------------------------------------------------ figure
fig = plt.figure(figsize=(NATURE_DOUBLE_COL_IN, 4.9))
gs = fig.add_gridspec(2, 3, hspace=.62, wspace=.5, left=.08, right=.98, top=.95, bottom=.13)

ax = fig.add_subplot(gs[0, 0])
cols = [("IBL", MI, "", C_IBL), ("Allen", MA, "", C_VBN)]
xt, xl = [], []
for j, k in enumerate(("still", "move")):
    for i, (lab, M, tag, c) in enumerate(cols):
        x = 3 * j + i
        o, nl = 100 * M[f"obs_{k}{tag}"], 100 * M[f"null_{k}{tag}"]
        ax.scatter(x - .18 + rng.uniform(-.06, .06, len(nl)), nl, s=2, color=GREY, alpha=.5, lw=0)
        ax.scatter(x + .18 + rng.uniform(-.06, .06, len(o)), o, s=2, color=c, alpha=.5, lw=0)
        for xx, v, cc in ((x - .18, nl, GREY), (x + .18, o, c)):
            mu, lo, hi = boot(v)
            ax.plot([xx, xx], [lo, hi], color="k", lw=1)
            ax.plot([xx - .12, xx + .12], [mu, mu], color="k", lw=1)
        xt.append(x)
        xl.append(lab)
ax.set_xticks(xt)
ax.set_xticklabels(xl, fontsize=5.5)
ax.text(.5, -.25, "still-preferring", transform=ax.get_xaxis_transform(), ha="center", fontsize=5.5)
ax.text(3.5, -.25, "movement-preferring", transform=ax.get_xaxis_transform(), ha="center", fontsize=5.5)
ax.set_ylabel("neurons per mouse (%)")
style_axes(ax, categorical_x=True)
panel_label(ax, "a")

ax = fig.add_subplot(gs[0, 1])
ax.scatter(NI.n_ep_move, 100 * NI.null_still, s=2, color=C_IBL, alpha=.4, lw=0, label="IBL, still-pref. null")
ax.scatter(NI.n_ep_still, 100 * NI.null_move, s=2, color=C_IBL, alpha=.4, lw=0, marker="^", label="IBL, movement-pref. null")
ax.scatter(np.full(len(NI), 24), 100 * NI.null_move_s24, s=2, color=GREY, alpha=.3, lw=0, marker="^",
           label="IBL, 24 still epochs")
ax.scatter(NA.n_ep_move, 100 * NA.null_still, s=4, color=C_VBN, lw=0, label="Allen, still-pref. null")
ax.scatter(NA.n_ep_still, 100 * NA.null_move, s=4, color=C_VBN, lw=0, marker="^", label="Allen, movement-pref. null")
ax.axhline(1, color=C_FR, lw=.6, ls="--")
ax.set_xscale("log")
ax.set_xticks([10, 100, 1000])
ax.set_xticklabels(["10", "100", "1000"])
ax.minorticks_off()
ax.set_xlabel("epochs in the other state (0.5 s)")
ax.set_ylabel("null false-positive rate (%)")
ax.legend(frameon=False, fontsize=4, loc="upper right", handletextpad=.2, markerscale=2)
style_axes(ax, categorical_x=True)
panel_label(ax, "b")

ax = fig.add_subplot(gs[0, 2])
for (t, Y), c, ls, lab in ((ibl_curves("wheel", "3s"), C_IBL, "-", "IBL wheel, ≥3 s"),
                           (ibl_curves("wheel", "1s"), C_IBL, "--", "IBL wheel, ≥1 s"),
                           (allen_curves("3s"), C_VBN, "-", "Allen running, ≥3 s"),
                           (allen_curves("1s"), C_VBN, "--", "Allen running, ≥1 s")):
    B = np.array([np.nanmean(Y[rng.integers(0, len(Y), len(Y))], 0) for _ in range(1000)])
    ax.fill_between(t, *np.nanquantile(B, [.025, .975], 0), color=c, alpha=.12, lw=0)
    ax.plot(t, np.nanmean(Y, 0), ls, color=c, lw=1, label=f"{lab} ({len(Y)})")
ax.axvspan(-.5, -.05, color=GREY, alpha=.12, lw=0)
ax.axhline(0, color=GREY, lw=.4, ls=":")
ax.axvline(0, color=GREY, lw=.4)
ax.set_xlim(-2, 1)
ax.set_xlabel("time from movement onset (s)")
ax.set_ylabel("population rate change (%)")
ax.legend(frameon=False, fontsize=4.2, loc="upper left")
style_axes(ax)
panel_label(ax, "c")

ax = fig.add_subplot(gs[1, 0])
xt = []
for i, (name, P, c) in enumerate(RISE):
    for j, rule in enumerate(("1s", "3s")):
        x = 2.6 * i + j
        v = 100 * P[P.rule == rule].groupby("subject").rise.mean()
        ax.scatter(x + rng.uniform(-.15, .15, len(v)), v, s=2, color=c, alpha=.35, lw=0)
        mu, lo, hi = boot(v)
        ax.plot([x, x], [lo, hi], color="k", lw=1)
        ax.plot([x - .25, x + .25], [mu, mu], color="k", lw=1)
        xt.append((x, f"≥{rule[0]} s"))
    ax.text(2.6 * i + .5, 1.0, name, transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=5.5)
ax.axhline(0, color=GREY, lw=.5, ls=":")
ax.set_xticks([x for x, _ in xt])
ax.set_xticklabels([lab for _, lab in xt], fontsize=5.5)
ax.set_ylabel("pre-movement rise per mouse (%)")
ax.set_ylim(-25, 40)
style_axes(ax, categorical_x=True)
panel_label(ax, "d", y=1.1)

ax = fig.add_subplot(gs[1, 1])
for i, mk in enumerate(("all", "still", "moving")):
    h = g[g["mask"] == mk]
    for j, (col, c) in enumerate((("pr", C_FR), ("pc", GREY))):
        x = i + (j - .5) * .36
        ax.scatter(x + rng.uniform(-.05, .05, len(h)), h[col], s=5, color=c, lw=0, alpha=.9,
                   label=(["raw", "distance-corrected"][j] if i == 0 else None))
        ax.plot([x - .12, x + .12], [h[col].mean()] * 2, color="k", lw=1)
ax.axhline(50, color=GREY, lw=.5, ls=":")
ax.axhline(90, color=GREY, lw=.5, ls="--")
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(["all bins", "still", "moving"], fontsize=5.5)
ax.set_ylim(0, 100)
ax.set_ylabel("frontal core–RSP percentile\namong cross-probe pairs")
ax.legend(frameon=False, fontsize=4.5, loc="lower right")
style_axes(ax, categorical_x=True)
panel_label(ax, "e")

ax = fig.add_subplot(gs[1, 2])
o, f = a[a.kind != "FRONT-RSP"], a[a.kind == "FRONT-RSP"]
ax.scatter(o.dist_um / 1000, o.z, s=2, color=GREY, alpha=.5, lw=0, label="other cross-probe pairs")
ax.scatter(f.dist_um / 1000, f.z, s=6, color=C_FR, lw=0, label="frontal core–RSP")
b1, b0 = np.polyfit(a.dist_um / 1000, a.z, 1)
xx = np.array([a.dist_um.min(), a.dist_um.max()]) / 1000
ax.plot(xx, b0 + b1 * xx, color="k", lw=.8)
ax.set_xlabel("distance between region centres (mm)")
ax.set_ylabel("coupling (Fisher z, 1 s bins)")
ax.legend(frameon=False, fontsize=4.5, loc="lower center", bbox_to_anchor=(.5, .98), ncol=2, handletextpad=.1)
style_axes(ax)
panel_label(ax, "f")

C = S["coupling"]
save_with_caption(fig, STEM, [
    "Still- and movement-preferring neurons per mouse (colour) against their circular-shift null (grey): all "
    "neurons' grey-screen 50\\,ms counts rotated together against the fixed behavioural labels (200 offsets $\\geq$10\\,s "
    f"per session) and the epoch test repeated. IBL: {S['null']['IBL_sessions']['n']} sessions, "
    f"{S['null']['IBL_sessions']['mice']} mice, report definition; Allen: Visual Behavior, "
    f"{S['null']['Allen_sessions']['n']} sessions, {S['null']['Allen_sessions']['mice']} mice (follow-up export). "
    "Black: mean and 95\\% bootstrap CI over mice.",
    "Null false-positive rate per session against the number of epochs in the opposite state. Grey triangles: "
    "IBL movement-preferring null with still epochs subsampled to 24, the Allen median. Dashed: nominal 1\\%.",
    "Population rate around movement onsets preceded by $\\geq$3\\,s (solid) or $\\geq$1\\,s (dashed) of awake "
    "stillness, percent change from the $-2$ to $-1$\\,s mean; mean over mice (number in brackets) and 95\\% "
    "bootstrap CI. IBL: wheel onsets (wheel-only states); Allen: running onsets. Shaded: $-0.5$ to $-0.05$\\,s.",
    "Rise ($-0.5$ to $-0.05$\\,s against $-2$ to $-1$\\,s) per mouse for each onset rule: IBL any-movement onsets "
    "(report definition), IBL wheel onsets and Allen running onsets; sessions with $\\geq$3 onsets.",
    f"Allen OpenScope ({C['sessions_front_rsp']} sessions, {C['mice']} mice with a frontal core and an RSP region on "
    "different probes): percentile of each mouse's frontal core--RSP pairs among all cross-probe region pairs of the "
    "session, from Fisher $z$ of 1\\,s regional rates (60\\,s running mean removed), in all, still and moving bins "
    "(passive viewing; running $>$2\\,cm/s; still = no running within $\\pm$1\\,s); raw (orange) and after removing "
    "a per-session linear fit on distance (grey). Bars: mean over mice. Dotted: 50th, dashed: 90th percentile.",
    f"Coupling against the distance between Allen CCF region centres for all {C['pairs']} cross-probe pairs of the "
    f"{C['sessions']} sessions; orange, frontal core--RSP pairs; line, linear fit."],
    label="allen_followup", number=19,
    intro="Allen follow-up analyses and their IBL counterparts: label calibration, onset rule and frontal--retrosplenial coupling.")
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


json.dump(clean(S), open(OUT / "33_allen_followup.json", "w"), indent=1)
print(json.dumps(clean(S), indent=1))
