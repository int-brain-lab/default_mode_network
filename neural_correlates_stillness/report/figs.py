"""Figures for the stillness report (outputs of 10_neurons, 11_population, 20_stats, 21_population_stats).
Each figure -> <name>_panels.pdf, <name>.tex, <name>.pdf (captioned, tectonic) and <name>.png for the web page.
Usage: python figs.py          # this folder, Figures 1-6
       python figs.py main     # ../../comprehensive_report/fig1N_still_*, Figures 11-16 of the grey-screen DMN report
"""
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "report"))
from plotstyle import ALLEN_EDGE, NATURE_DOUBLE_COL_IN, allen_color, panel_label, save_with_caption, style_axes  # noqa: E402
from common_still import BIN, CONS_DMN, OUT, bouts, load_session, states  # noqa: E402

N = pd.read_parquet(OUT / "20_neurons.pqt")
ST = json.load(open(OUT / "20_stats.json"))
PS = json.load(open(OUT / "21_population_stats.json"))
R = pd.read_csv(OUT / "20_regions.csv", index_col=0)
SESS = pd.read_csv(OUT / "10_sessions.csv")
DIV = ["Isocortex", "HPF", "TH", "CNU", "MB", "HB", "CB"]
DIVLAB = {"Isocortex": "cortex", "HPF": "hippocampal f.", "TH": "thalamus", "CNU": "striatum/pallidum",
          "MB": "midbrain", "HB": "hindbrain", "CB": "cerebellum"}
COL = {d: allen_color(d) for d in DIV}  # Allen atlas division colours (cerebellum brown)
GREY = "#8a8880"
rng = np.random.default_rng(0)
MAIN = len(sys.argv) > 1 and sys.argv[1] == "main"
OFF = 9 if MAIN else 0
OUTD = ROOT.parent / "comprehensive_report" if MAIN else HERE


def stem(k, name):
    return OUTD / (f"fig{k + OFF}_still_{name}" if MAIN else f"fig{k}_{name}")


def png(stem):
    subprocess.run(["pdftoppm", "-png", "-r", "200", "-singlefile", f"{stem}.pdf", str(stem)], check=True)


def boot(v, n=2000):
    v = np.asarray(v)[np.isfinite(v)]
    b = np.array([v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)])
    return v.mean(), *np.quantile(b, [.025, .975])


def mouse_means(D, col, min_n=5):
    g = D.groupby("subject")[col]
    m = g.mean()[g.count() >= min_n]
    return m.dropna()


def div_dots(ax, col, ylab, data=N, hline=0, min_n=5):
    for i, d in enumerate(DIV):
        m = mouse_means(data[data.cosmos == d], col, min_n)
        x = i + rng.uniform(-.18, .18, len(m))
        ax.scatter(x, m, s=4, color=COL[d], alpha=.8, edgecolor=ALLEN_EDGE, lw=.15)
        mu, lo, hi = boot(m)
        ax.plot([i, i], [lo, hi], color="k", lw=1)
        ax.plot([i - .25, i + .25], [mu, mu], color="k", lw=1)
    if hline is not None:
        ax.axhline(hline, color=GREY, lw=.5, ls=":")
    ax.set_xticks(range(len(DIV)))
    ax.set_xticklabels([DIVLAB[d] for d in DIV], rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylabel(ylab)
    style_axes(ax, categorical_x=True)


def hexpanel(ax, a, b, xl, yl, lim=1):
    m = N[[a, b]].dropna()
    ax.hexbin(m[a].values, m[b].values, gridsize=45, extent=(-lim, lim, -lim, lim), bins="log", cmap="Greys", mincnt=1,
              lw=0, rasterized=True)
    ax.plot([-lim, lim], [-lim, lim], color="#2a78d6", lw=.5, ls="--")
    rho = ST["neuron_consistency_spearman"].get(f"{a}~{b}")
    ax.text(.04, .96, f"ρ = {rho:.2f}\n{len(m):,} neurons", transform=ax.transAxes, va="top", fontsize=5.5)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    style_axes(ax)


def fmt(x, d=2):
    return f"{x:.{d}f}"


G = ST["global"]

# ======================================================================= Fig 1: behaviour and state definition
# example: the session closest to median stillness with pupil, motion energy and wheel on the grey screen
cand = SESS[SESS.pupil_ok & SESS.me_ok]
for eid in cand.eid.values[(cand.n_still - cand.n_still.median()).abs().argsort()]:
    s = load_session(eid)
    st = states(s)
    if np.isfinite(s["wheel_speed"][st["grey"]].astype(float)).mean() > .95:
        break
print("Fig 1 example session", eid)
ga = np.flatnonzero(st["grey"])
# Choose a grey-screen minute with tracking coverage in this excerpt.
candidates = np.arange(ga[0], ga[-1] - 1199, 200)
scores = [min(np.isfinite(st[k][a:a+1200]).mean() for k in ("pupil", "me")) for a in candidates]
t0 = int(candidates[np.argmax(scores)])
win = np.arange(t0, t0 + 1200)  # 60 s
tt = (win - t0) * BIN
fig = plt.figure(figsize=(NATURE_DOUBLE_COL_IN, 3.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1], hspace=.65, wspace=.45)
gtop = gs[0, :].subgridspec(3, 1, hspace=.15)
axw, axm, axp = (fig.add_subplot(gtop[i]) for i in range(3))
for ax, key, lab in ((axw, "wheel_speed", "wheel\n(rad/s)"), (axm, "me", "motion\nenergy"), (axp, "pupil", "pupil\n(px)")):
    v = s[key][win].astype(float) if key in s else st[key][win]
    ax.plot(tt, v, color="k", lw=.6)
    for mask, c in ((st["awake"], "#1baf7a"), (st["mov"], "#eb6834"), (st["still"] & ~st["awake"], "#6250d6")):
        b0, b1 = bouts(mask[win])
        for a, b in zip(b0, b1):
            ax.axvspan(tt[a], tt[min(b, len(tt) - 1)], color=c, alpha=.25, lw=0)
    ax.set_ylabel(lab, fontsize=6)
    style_axes(ax)
    ax.set_xlim(0, 60)
    if ax is not axp:
        ax.set_xticklabels([])
axp.set_xlabel("time on the grey screen (s, 60 s excerpt)")
for lab, c, x in (("awake-still", "#1baf7a", .55), ("moving", "#eb6834", .7), ("still, gated out", "#6250d6", .82)):
    axw.text(x, 1.15, "■ " + lab, color=c, transform=axw.transAxes, fontsize=5.5, va="bottom")
panel_label(axw, "a", x=-.06, y=1.0)
ax = fig.add_subplot(gs[1, 0])
ax.hist(SESS.n_still * BIN / 60, bins=25, color="#1baf7a", alpha=.8, label="awake-still")
ax.hist(SESS.n_move * BIN / 60, bins=25, color="#eb6834", alpha=.6, label="moving")
ax.set_xlabel("time per session (min)")
ax.set_ylabel("sessions")
ax.legend(frameon=False, fontsize=5.5)
style_axes(ax)
panel_label(ax, "b")
# bout durations (awake-still, starting after movement) pooled over sessions
dur = []
for e in SESS.eid.values[::3]:
    s2 = load_session(e)
    st2 = states(s2)
    b0, b1 = bouts(st2["grey"] & st2["awake"])
    dur.append((b1 - b0) * BIN)
dur = np.concatenate(dur)
ax = fig.add_subplot(gs[1, 1])
ax.hist(dur, bins=np.logspace(np.log10(.05), np.log10(40), 40), color="#1baf7a")
ax.set_xscale("log")
ax.axvline(4, color="k", lw=.5, ls=":")
ax.text(4.3, ax.get_ylim()[1] * .9, f"≥ 4 s bouts\n(Fig. {4 + OFF})", fontsize=5, va="top")
ax.set_xlabel("awake-still bout (s)")
ax.set_ylabel("bouts (1/3 of sessions)")
style_axes(ax)
ax.set_xticks([.1, 1, 10])
ax.set_xticklabels(["0.1", "1", "10"])
panel_label(ax, "c")
ax = fig.add_subplot(gs[1, 2])
gated = 1 - SESS.n_still / SESS.n_nogate.clip(lower=1)
ax.hist(gated[SESS.pupil_ok], bins=25, color="#6250d6", alpha=.85, label="pupil available")
ax.hist(gated[~SESS.pupil_ok], bins=25, color=GREY, alpha=.8, label="no pupil")
ax.set_xlabel("fraction of still time gated out")
ax.set_ylabel("sessions")
ax.legend(frameon=False, fontsize=5.5)
style_axes(ax)
panel_label(ax, "d")
save_with_caption(fig, stem(1, "states"), [
    f"Example grey-screen minute (session {eid[:8]}): wheel speed, left-camera motion energy and pupil diameter, "
    "with awake-still (green), moving (orange) and still-but-gated-out (violet) 50\\,ms bins.",
    f"Awake-still and moving time per session ({len(SESS)} sessions).",
    "Durations of awake-still bouts (log axis); true-offset bouts of at least 4\\,s enter the within-bout analyses.",
    "Fraction of still time removed by the anti-sleep gate (pupil below the 10th percentile of the task-period "
    "pupil, or more than 30\\,s since the last movement)."],
    label="states", number=1 + OFF, intro="Defining awake stillness on the passive grey screen.")
png(stem(1, "states"))
plt.close(fig)

# ======================================================================= Fig 2: firing rate
fig = plt.figure(figsize=(NATURE_DOUBLE_COL_IN, 4.4))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.15], hspace=.75, wspace=.45)
ax = fig.add_subplot(gs[0, 0])
div_dots(ax, "lrate", "log$_2$ rate still / moving")
panel_label(ax, "a")
ax = fig.add_subplot(gs[0, 1])
div_dots(ax, "SP", "still-preferring neurons", hline=None)
style_axes(ax, categorical_x=True)
panel_label(ax, "b")
ax = fig.add_subplot(gs[0, 2])
div_dots(ax, "MP", "movement-preferring neurons", hline=None)
panel_label(ax, "c")
ax = fig.add_subplot(gs[1, :])
Rr = R[R.n_subj >= 5].sort_values("lrate")
k = 22
sel = pd.concat([Rr.head(k), Rr.tail(k)])
x = np.r_[np.arange(k), np.arange(k) + k + 1.5]
for xi, (r, row) in zip(x, sel.iterrows()):
    c = allen_color(r)
    ax.plot([xi, xi], [row.lrate_lo, row.lrate_hi], color=ALLEN_EDGE, lw=.5)
    ax.scatter([xi], [row.lrate], color=c, s=12, zorder=3, edgecolor=ALLEN_EDGE, lw=.3)
ax.axhline(0, color=GREY, lw=.5, ls=":")
ax.axhline(Rr.lrate.median(), color=GREY, lw=.5, ls="--")
ax.text(x[-1] + .8, Rr.lrate.median(), f"median,\n{len(Rr)} regions", fontsize=5, ha="left", va="center", color=GREY)
ax.set_xticks(x)
ax.set_xticklabels([f"{r}*" if r in CONS_DMN else r for r in sel.index], rotation=60, ha="right",
                   rotation_mode="anchor", fontsize=5)
ax.set_ylabel("log$_2$ rate still / moving")
ax.set_xlim(-1, x[-1] + 1)
style_axes(ax, categorical_x=True)
panel_label(ax, "d", x=-.04)
gl = G["lrate"]
save_with_caption(fig, stem(2, "rates"), [
    "Population firing in awake stillness relative to movement, log$_2$ of the rate ratio, one dot per mouse "
    "(neurons averaged within mouse and division, mice with $\\geq$5 neurons); black: mean over mice and 95\\% "
    f"bootstrap CI. Over all neurons {gl['frac_mice_neg']*100:.0f}\\% of {gl['n_mice']} mice are below zero.",
    "Fraction of still-preferring neurons (0.5\\,s epoch Welch test, $p<0.01$, higher in stillness).",
    "Fraction of movement-preferring neurons.",
    f"The {k} regions with the strongest and the {k} with the weakest suppression in stillness, among the "
    f"{len(Rr)} regions recorded in $\\geq$5 mice (mean over mice, mouse-bootstrap 95\\% CI; dots in Allen atlas region colours; * conservative DMN core)."],
    label="rates", number=2 + OFF, intro="Stillness lowers firing across the brain; orofacial brainstem and thalamus are suppressed most, hippocampus and cerebellum least.")
png(stem(2, "rates"))
plt.close(fig)

# ======================================================================= Fig 3: stability of the neuron property; region specificity
fig = plt.figure(figsize=(NATURE_DOUBLE_COL_IN, 2.5))
gs = fig.add_gridspec(1, 4, wspace=.6, width_ratios=[1, 1, 1, 1.5])
ax = fig.add_subplot(gs[0])
hexpanel(ax, "SI_h1", "SI_h2", "SI, 1st half", "SI, 2nd half")
panel_label(ax, "a", x=-.3)
ax = fig.add_subplot(gs[1])
hexpanel(ax, "SI", "SI_iti", "SI, grey screen", "SI, task inter-trial gaps")
panel_label(ax, "b", x=-.3)
ax = fig.add_subplot(gs[2])
hexpanel(ax, "SI_face", "SI_wheel", "SI vs face/body only", "SI vs wheel turns")
panel_label(ax, "c", x=-.3)
ax = fig.add_subplot(gs[3])
REL = ST["region_reliability_over_mice"]
items = [("lrate", "rate ratio"), ("SP", "still-pref."), ("MP", "move-pref."), ("BI", "bout build-up"),
         ("l_tsm_late_early", "late vs early"), ("r_pupil", "pupil corr.")]
for i, (k_, lab) in enumerate(items):
    for j, (pre, c) in enumerate((("", "#2a78d6"), ("ws_", "#eb6834"))):
        key = pre + k_
        if key not in REL:
            continue
        v = REL[key]
        ax.bar(i + (j - .5) * .38, v["r"], .36, color=c, label=["all", "within session"][j] if i == 0 else None)
        ax.plot([i + (j - .5) * .38] * 2, v["q05_q95"], color="k", lw=.6)
ax.set_xticks(range(len(items)))
ax.set_xticklabels([l for _, l in items], rotation=45, ha="right", rotation_mode="anchor")
ax.set_ylabel("region-map split-half r\n(halves of mice)")
ax.legend(frameon=False, fontsize=5.5, loc="upper right")
style_axes(ax, categorical_x=True)
panel_label(ax, "d", x=-.2)
VE = ST["variance_explained"]["lrate"]
save_with_caption(fig, stem(3, "stability"), [
    "Stillness index SI = (still $-$ moving)/(still $+$ moving) of each neuron in the first vs second half of "
    "the grey screen.",
    "SI on the grey screen vs in the inter-trial gaps of the task, a second behavioural context recorded earlier in "
    "the session.",
    "SI computed against face/body movements without wheel turns vs against wheel turns.",
    "Reliability of region maps over animals: Spearman r between region means from two random halves of the mice "
    "(200 splits, median and 5--95\\% range), from raw neuron values (blue) and after subtracting each "
    "session's mean (orange). Regions with $\\geq$20 neurons from $\\geq$3 mice. Region within session explains "
    f"{VE['region_within_session']*100:.0f}\\% of neuron variance in the rate ratio (label shuffle "
    f"{VE['region_within_session_shuffle']*100:.0f}\\%), session {VE['session']*100:.0f}\\%."],
    label="stability", number=3 + OFF, intro="Stillness preference is reproducible within recordings; region differences are reproducible but modest.")
png(stem(3, "stability"))
plt.close(fig)

# ======================================================================= Fig 4: dynamics
P = np.load(OUT / "10_peths.npz", allow_pickle=True)
assert (P["uuid"].astype(str) == N.uuid.values.astype(str)).all()


def pop_curve(A, d, smooth=5):
    m = (N.cosmos == d).values & np.isfinite(A).all(1)
    D = pd.DataFrame(A[m])
    D["s"] = N.subject.values[m]
    g = D.groupby("s")
    cnt = g.size()
    mean = g.mean()[cnt >= 5]
    base = N[m].groupby("subject").r_still.mean().loc[mean.index]
    L = np.log2((mean.values + .1) / (base.values[:, None] + .1))
    k_ = np.ones(smooth) / smooth
    L = np.array([np.convolve(np.pad(x, smooth // 2, mode="edge"), k_, "valid") for x in L])
    b = np.array([L[rng.integers(0, len(L), len(L))].mean(0) for _ in range(500)])
    return L.mean(0), *np.quantile(b, [.025, .975], 0), len(L)


fig = plt.figure(figsize=(NATURE_DOUBLE_COL_IN, 4.3))
gs = fig.add_gridspec(2, 3, hspace=.7, wspace=.45)
SHOW = ["Isocortex", "HPF", "TH", "MB", "CB"]
for j, (key, tk, xl, lab) in enumerate((("off", "t_off", "time from movement offset (s)", "a"),
                                        ("on", "t_on", "time from movement onset (s)", "b"))):
    ax = fig.add_subplot(gs[0, j])
    t = P[tk] + BIN / 2
    for d in SHOW:
        mu, lo, hi, n = pop_curve(P[key], d)
        ax.fill_between(t, lo, hi, color=COL[d], alpha=.18, lw=0)
        ax.plot(t, mu, color=COL[d], lw=.9, label=f"{DIVLAB[d]} ({n})")
    ax.axvline(0, color="k", lw=.5)
    ax.axhline(0, color=GREY, lw=.5, ls=":")
    ax.set_xlabel(xl)
    ax.set_ylabel("log$_2$ rate / awake-still rate")
    style_axes(ax)
    if j == 0:
        ax.legend(frameon=False, fontsize=5, loc="upper right", handlelength=1)
    panel_label(ax, lab)
# c: time since movement profile
ax = fig.add_subplot(gs[0, 2])
TP = np.array([0.1, 0.5, 1, 2, 4, 8, 30])
xc = np.sqrt(TP[:-1] * TP[1:])
for d in DIV:
    D = N[N.cosmos == d]
    rows = []
    for sbj, g in D.groupby("subject"):
        if len(g) < 5:
            continue
        base = g.r_still.mean()
        rows.append([np.log2((g[f"tsm_{i}"].mean() + .1) / (base + .1)) for i in range(6)])
    L = np.array(rows)
    mu = np.nanmean(L, 0)
    se = np.nanstd(L, 0) / np.sqrt(np.isfinite(L).sum(0))
    ax.errorbar(xc, mu, yerr=1.96 * se, color=COL[d], lw=.9, capsize=0, marker="o", ms=2.5, label=DIVLAB[d])
ax.set_xscale("log")
ax.axhline(0, color=GREY, lw=.5, ls=":")
ax.set_xticks([.3, 1, 3, 10])
ax.set_xticklabels(["0.3", "1", "3", "10"])
ax.set_xlabel("time since last movement (s)")
ax.set_ylabel("log$_2$ rate / awake-still rate")
style_axes(ax)
ax.xaxis.set_major_locator(plt.FixedLocator([.3, 1, 3, 10]))
panel_label(ax, "c")
# d,e: build-up / build-down fractions
for j, (col_, lab, l_) in enumerate((("BU", "neurons rising within bouts", "d"), ("BD", "neurons falling within bouts", "e"))):
    ax = fig.add_subplot(gs[1, j])
    div_dots(ax, col_, lab, hline=None)
    panel_label(ax, l_)
ax = fig.add_subplot(gs[1, 2])
div_dots(ax, "l_tsm_late_early", "log$_2$ rate 2–4 s / 0.5–1 s\nafter movement")
panel_label(ax, "f")
save_with_caption(fig, stem(4, "dynamics"), [
    "Population rate around the end of a movement that starts an awake-still bout of $\\geq$4\\,s, relative to "
    "the awake-still rate; mean over mice (number in legend) and bootstrap 95\\% CI, 250\\,ms smoothing.",
    "The same around movement onset after $\\geq$3\\,s of awake stillness.",
    "Rate in awake-still bins by time since the last movement (all bouts), mean over mice $\\pm$ 1.96 s.e.m.; "
    "colours as in a, plus striatum/pallidum and hindbrain.",
    "Fraction of neurons whose rate rises from the first 0.5\\,s to 2--4\\,s within $\\geq$4\\,s bouts "
    "(Wilcoxon over bouts, $p<0.01$); dots are mice.",
    "Fraction of neurons whose rate falls.",
    "Rate 2--4\\,s vs 0.5--1\\,s after the last movement, per mouse."],
    label="dynamics", number=4 + OFF, intro="Stillness is graded: firing dips after movement ends, then thalamus keeps falling while cortex and hippocampus recover.")
png(stem(4, "dynamics"))
plt.close(fig)

# ======================================================================= Fig 5: population structure
T = pd.read_parquet(OUT / "21_population_diffs.pqt")
fig = plt.figure(figsize=(NATURE_DOUBLE_COL_IN, 2.5))
gs = fig.add_gridspec(1, 3, wspace=.45)
for j, (col_, lab, l_) in enumerate((("corr", "Δ mean pairwise correlation\n(still − moving)", "a"),
                                     ("pr", "Δ participation ratio / N\n(still − moving)", "b"),
                                     ("burst", "Δ fraction of synchronous bins\n(still − moving)", "c"))):
    ax = fig.add_subplot(gs[j])
    for i, d in enumerate(DIV + ["all"]):
        for k_, (var, mk, off) in enumerate((("raw", "o", -.17), ("matched", "s", .17))):
            D = T[(T.comp == "still/move") & (T.variant == var)]
            if d != "all":
                D = D[D.cosmos == d]
            m = D.groupby("subject")[col_].mean()
            mu, lo, hi = boot(m)
            c = COL.get(d, "k")
            ax.plot([i + off] * 2, [lo, hi], color=c, lw=.8)
            ax.scatter([i + off], [mu], marker=mk, s=10, color=c if var == "matched" else "white", edgecolor=c,
                       lw=.7, zorder=3, label=("raw" if var == "raw" else "rate-matched") if i == 0 and j == 0 else None)
    ax.axhline(0, color=GREY, lw=.5, ls=":")
    ax.set_xticks(range(len(DIV) + 1))
    ax.set_xticklabels([DIVLAB[d] for d in DIV] + ["all"], rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylabel(lab)
    style_axes(ax, categorical_x=True)
    if j == 0:
        ax.legend(frameon=False, fontsize=5.5, loc="upper left")
    panel_label(ax, l_)
pm = PS["still/move|matched"]
save_with_caption(fig, stem(5, "population"), [
    "Mean pairwise spike-count correlation within a region (100\\,ms bins, equal numbers of still and moving "
    "bins), still minus moving, per region and probe averaged within mouse; mean over mice and bootstrap 95\\% CI. "
    "Open: raw counts; filled: each neuron's counts in its higher-rate state binomially thinned to its rate in the "
    f"other state. Rate-matched, {pm['all']['corr']['frac_mice_pos']*100:.0f}\\% of "
    f"{pm['all']['corr']['n_mice']} mice have higher correlation in stillness "
    f"(cortex {pm['Isocortex']['corr']['frac_mice_pos']*100:.0f}\\%).",
    "Participation ratio of the correlation matrix divided by the number of neurons (dimensionality). The raw "
    "increase in stillness disappears after rate matching.",
    "Fraction of 100\\,ms bins in which the summed population count exceeds its within-state mean by 3 SD "
    "(synchronous events)."],
    label="population", number=5 + OFF, intro="In stillness, neurons within a region fluctuate together more (rate-matched), most clearly in cortex and hippocampus.")
png(stem(5, "population"))
plt.close(fig)

# ======================================================================= Fig 6: arousal and the sleep gate
fig = plt.figure(figsize=(NATURE_DOUBLE_COL_IN, 2.5))
gs = fig.add_gridspec(1, 4, wspace=.6, width_ratios=[1.3, 1.3, 1, 1])
ax = fig.add_subplot(gs[0])
div_dots(ax, "r_pupil", "partial r with pupil\nwithin awake stillness")
panel_label(ax, "a")
ax = fig.add_subplot(gs[1])
div_dots(ax, "l_lowpup", "log$_2$ rate still, small pupil\n/ awake-still")
panel_label(ax, "b")
ax = fig.add_subplot(gs[2])
hexpanel(ax, "SI", "SI_nogate", "SI, awake-still", "SI, all still bins")
panel_label(ax, "c", x=-.3)
ax = fig.add_subplot(gs[3])
C = R[R.cosmos == "Isocortex"]
isd = C.index.isin(CONS_DMN)
ax.scatter(rng.uniform(-.15, .15, (~isd).sum()), C.lrate[~isd], s=7, color=[allen_color(r_) for r_ in C.index[~isd]],
           edgecolor=ALLEN_EDGE, lw=.2)
ax.scatter(1 + rng.uniform(-.15, .15, isd.sum()), C.lrate[isd], s=9, color=[allen_color(r_) for r_ in C.index[isd]],
           edgecolor=ALLEN_EDGE, lw=.2)
cv = C.lrate[isd].sort_values()
ly = cv.values.copy()
for i in range(1, len(ly)):
    ly[i] = max(ly[i], ly[i - 1] + .055)
for (r_, v), y_ in zip(cv.items(), ly):
    ax.plot([1.08, 1.18], [v, y_], color=GREY, lw=.3)
    ax.text(1.2, y_, r_, fontsize=4.5, va="center")
ax.set_xticks([0, 1])
ax.set_xticklabels(["other cortex", "DMN core"], rotation=45, ha="right", rotation_mode="anchor")
ax.set_ylabel("log$_2$ rate still / moving")
ax.set_xlim(-.5, 1.9)
style_axes(ax, categorical_x=True)
panel_label(ax, "d", x=-.4)
dm = ST["dmn_core_vs_other_ctx"]["lrate"]
save_with_caption(fig, stem(6, "arousal"), [
    "Partial correlation of each neuron's 1\\,s spike count with pupil diameter within awake stillness (time and "
    "motion energy regressed out); dots are mice.",
    "Rate in still bins with a pupil below the gate (but $\\leq$30\\,s since movement) relative to awake "
    "stillness; lower in thalamus, midbrain and hindbrain, unchanged in cortex and hippocampus.",
    "Stillness index with and without the anti-sleep gate.",
    f"Cortical regions: conservative DMN core vs other cortex (core {dm['core']:.2f}, other {dm['other']:.2f}, "
    f"region permutation p = {dm['p_perm']:.2f}); dots in Allen atlas region colours."],
    label="arousal", number=6 + OFF, intro="Arousal within stillness, the sleep gate and the DMN core.")
png(stem(6, "arousal"))
plt.close(fig)
print("done")
