# Scripts: Allen stillness replication

Three Python scripts. Copy each block into a `.py` file of the given name (the download format here cannot hold `.py` files directly).
Run order to rebuild the figure from the exported tables (no data download needed):

    python stillness_stats.py   ./data      # prints every number in the report
    python plot_stillness_figures.py ./data ./figures   # writes SVG/PDF/PNG figures

Script 3 is the per-session analysis pipeline that produced the tables from the raw Allen spikes, running speed and pupil.
It is included so the methods are exact; it needs the spike trains (SpikeLab SpikeData) and behaviour streams, which were
loaded from the OCS platform's copies of DANDI 000713 (Visual Behavior) and the Allen Visual Coding Neuropixels release.

## 1. `stillness_stats.py` (statistics from the tables)

```python
"""Recompute the summary statistics of the Allen stillness replication from data_neurons.csv.

Usage: python stillness_stats.py [DATA_DIR]
Mouse-level: sessions are averaged within mouse, each mouse weighted once.
95% CIs: 10,000 bootstrap resamples over mice (seed 20261030). Tests: two-sided Wilcoxon signed-rank over mice.
"""
import sys, os, numpy as np, pandas as pd
from scipy import stats
DATA = sys.argv[1] if len(sys.argv) > 1 else "."
rng = np.random.default_rng(20261030)
def boot(x, n=10000):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return tuple(np.percentile(rng.choice(x, (n, len(x))).mean(1), [2.5, 97.5])) if len(x) > 1 else (np.nan, np.nan)

N = pd.read_csv(os.path.join(DATA, "data_neurons.csv"), dtype={"subject": str, "session": str}, low_memory=False)
N = N[N.keep.astype(str).str.lower().isin(["true", "1", "1.0"])]            # neurons >= 0.5 Hz on the grey screen
sess = N.groupby(["dataset", "subject", "session"]).agg(mean_lr=("log2rr", "mean"),
        still_pref=("pref", lambda p: (p == 1).mean()), move_pref=("pref", lambda p: (p == -1).mean()), n=("log2rr", "size")).reset_index()
mice = sess.groupby(["dataset", "subject"])[["mean_lr", "still_pref", "move_pref"]].mean().reset_index()
print(f"{len(N)} neurons, {len(sess)} sessions, {mice.subject.nunique()} mice")
for ds, g in list(mice.groupby("dataset")) + [("ALL", mice)]:
    print(f"\n== {ds} (n={len(g)} mice)")
    for col in ["mean_lr", "still_pref", "move_pref"]:
        lo, hi = boot(g[col]); print(f"  {col:10s} mean {g[col].mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")
    print(f"  mice with mean_lr < 0: {(g.mean_lr < 0).sum()}/{len(g)};  Wilcoxon p = {stats.wilcoxon(g.mean_lr).pvalue:.2g}")
    print(f"  mice with move_pref > still_pref: {(g.move_pref > g.still_pref).sum()}/{len(g)}")
# divisions: per session, Cosmos divisions with >= 20 kept neurons; then mean per mouse
D = N.groupby(["dataset", "subject", "session", "cosmos"]).log2rr.agg(["mean", "size"]).reset_index()
D = D[D["size"] >= 20].groupby(["dataset", "subject", "cosmos"])["mean"].mean().reset_index()
print("\n== Divisions (mean log2 ratio per mouse)")
for (ds, c), g in D[D.cosmos.isin(["MB", "TH", "HPF", "Isocortex"])].groupby(["dataset", "cosmos"]):
    lo, hi = boot(g["mean"]); p = stats.wilcoxon(g["mean"]).pvalue if len(g) >= 5 else float("nan")
    print(f"  {ds:3s} {c:10s} n={len(g):2d}  {g['mean'].mean():+.3f} [{lo:+.3f}, {hi:+.3f}]  below 0: {(g['mean']<0).sum()}  p={p:.2g}")
# pre-movement rise
PC = pd.read_csv(os.path.join(DATA, "data_premovement_curves_pct.csv")); t = PC.t_s.values; Y = PC.drop(columns="t_s").values.T
w = (t >= -0.5) & (t < -0.05); pre = Y[:, w].mean(1)
print(f"\n== Pre-movement: change -0.5..-0.05 s = {pre.mean():+.1f}%, positive in {(pre>0).sum()}/{len(pre)} mice, Wilcoxon p = {stats.wilcoxon(pre).pvalue:.2g}")
S = pd.read_csv(os.path.join(DATA, "data_sessions.csv"), dtype={"subject": str}); S = S[S.n_ep_still.notna()]
r = stats.spearmanr(S.n_ep_still, S.still_pref); print(f"== still-pref vs still epochs: Spearman rho = {r.correlation:.2f}, p = {r.pvalue:.2g}, {len(S)} sessions")
```

## 2. `plot_stillness_figures.py` (all figure panels)

```python
"""Reproduce the Allen stillness-replication figure from the exported CSV tables.

Usage:  python plot_stillness_figures.py [DATA_DIR] [OUT_DIR]
Needs:  numpy, pandas, scipy, matplotlib. Writes stillness_allen_summary.svg/.pdf/.png
        and one SVG per panel (panel_A.svg ... panel_F.svg) into OUT_DIR.
Bootstrap: 10,000 resamples over mice, seed 20261030 (CIs may differ in the 3rd decimal from the report
if the resampling order differs).
"""
import sys, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

DATA = sys.argv[1] if len(sys.argv) > 1 else "."
OUT = sys.argv[2] if len(sys.argv) > 2 else "."
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.15, "svg.fonttype": "none", "pdf.fonttype": 42,
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})
C_VBN, C_VC, C_REF, C_MAIN, C_THIN, C_BAND = "#3381D5", "#9B60BC", "#E28E5B", "#4B6733", "#C4CBB1", "#EAC5FF"
IBL_LR, IBL_STILL, IBL_MOVE = -0.43, 15, 40          # values from the IBL manuscript
rng = np.random.default_rng(20261030)

def boot_ci(x, n=10000):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 2: return (np.nan, np.nan)
    return tuple(np.percentile(rng.choice(x, (n, len(x))).mean(1), [2.5, 97.5]))

def jit(n, w=0.12): return rng.uniform(-w, w, n)

M  = pd.read_csv(os.path.join(DATA, "data_mice.csv"), dtype={"subject": str})
S  = pd.read_csv(os.path.join(DATA, "data_sessions.csv"), dtype={"subject": str})
Dm = pd.read_csv(os.path.join(DATA, "data_divisions_per_mouse.csv"), dtype={"subject": str})
PC = pd.read_csv(os.path.join(DATA, "data_premovement_curves_pct.csv"))
SC = pd.read_csv(os.path.join(DATA, "data_screening.csv"))

def panel_A(ax):
    for i, (ds, c) in enumerate([("VBN", C_VBN), ("VC", C_VC)]):
        x = M.loc[M.dataset == ds, "mean_lr"].values
        ax.scatter(i + jit(len(x)), x, s=28, color=c, alpha=.8, zorder=3)
        lo, hi = boot_ci(x)
        ax.errorbar(i + .3, x.mean(), yerr=[[x.mean() - lo], [hi - x.mean()]], fmt="o", color="k", capsize=4, zorder=4)
    ax.axhline(0, color="k", lw=.8); ax.axhline(IBL_LR, color=C_REF, ls="--", lw=1.2)
    ax.text(1.55, IBL_LR + .015, "IBL (-0.43)", ha="right", fontsize=10)
    ax.set_xticks([0, 1]); ax.set_xlim(-.5, 1.6)
    ax.set_xticklabels([f"Visual Behavior\n(n={(M.dataset=='VBN').sum()} mice)", f"Visual Coding\n(n={(M.dataset=='VC').sum()})"])
    ax.set_ylabel("Mean log$_2$ rate ratio\n(still / moving)"); ax.set_title("A  Lower firing in stillness", loc="left")

def panel_B(ax):
    for i, col in enumerate(["still_pref", "move_pref"]):
        for ds, c, off in [("VBN", C_VBN, -.12), ("VC", C_VC, .12)]:
            x = M.loc[M.dataset == ds, col].values * 100
            ax.scatter(i + off + jit(len(x), .06), x, s=24, color=c, alpha=.8, label=(["Visual Behavior", "Visual Coding"][ds == "VC"] if i == 0 else None), zorder=3)
        xa = M[col].values * 100; lo, hi = boot_ci(xa)
        ax.errorbar(i + .32, xa.mean(), yerr=[[xa.mean() - lo], [hi - xa.mean()]], fmt="o", color="k", capsize=4)
    for i, v in [(0, IBL_STILL), (1, IBL_MOVE)]: ax.hlines(v, i - .3, i + .3, color=C_REF, ls="--", lw=1.2)
    ax.text(1.3, IBL_MOVE + .8, "IBL", fontsize=10)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Still-preferring", "Movement-preferring"]); ax.set_xlim(-.5, 1.6)
    ax.set_ylabel("% of neurons per mouse"); ax.set_title("B  Stable still-preferring minority", loc="left")
    ax.legend(loc="upper left", fontsize=10, frameon=False)

def panel_C(ax):
    ss = S[S.n_ep_still.notna()]
    for ds, c in [("VBN", C_VBN), ("VC", C_VC)]:
        g = ss[ss.dataset == ds]; ax.scatter(g.n_ep_still, g.still_pref * 100, s=26, color=c, alpha=.8)
    rho = stats.spearmanr(ss.n_ep_still, ss.still_pref)
    ax.axhline(IBL_STILL, color=C_REF, ls="--", lw=1.2); ax.set_xscale("log")
    ax.set_xlabel("Still epochs in session (0.5 s)"); ax.set_ylabel("% still-preferring")
    ax.set_title("C  Detection scales with still time", loc="left")
    ax.text(.03, .93, f"Spearman rho = {rho.correlation:.2f}\n{len(ss)} sessions", transform=ax.transAxes, fontsize=10, va="top")

def panel_D(ax):
    order = ["MB", "TH", "HPF", "Isocortex"]; names = ["Midbrain", "Thalamus", "Hippocampal\nformation", "Isocortex"]
    for i, cn in enumerate(order):
        for ds, c, off in [("VBN", C_VBN, -.15), ("VC", C_VC, .15)]:
            x = Dm.loc[(Dm.dataset == ds) & (Dm.cosmos == cn), "mean_log2rr"].values
            if len(x) == 0: continue
            ax.scatter(i + off + jit(len(x), .06), x, s=16, color=c, alpha=.45)
            lo, hi = boot_ci(x) if len(x) > 1 else (x.mean(), x.mean())
            ax.errorbar(i + off, x.mean(), yerr=[[x.mean() - lo], [hi - x.mean()]], fmt="o", color=c, mec="k", ms=8, capsize=4, zorder=4)
    ax.axhline(0, color="k", lw=.8); ax.set_xticks(range(4)); ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Mean log$_2$ rate ratio per mouse"); ax.set_title("D  By brain division", loc="left")

def panel_E(ax):
    t = PC.t_s.values; Y = PC.drop(columns="t_s").values.T          # mice x time, % change
    for y in Y: ax.plot(t, y, color=C_THIN, lw=.8)
    bs = np.stack([Y[rng.integers(0, len(Y), len(Y))].mean(0) for _ in range(5000)])
    ax.fill_between(t, np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0), color=C_BAND, alpha=.7, lw=0)
    ax.plot(t, Y.mean(0), color=C_MAIN, lw=2.2); ax.axvline(0, color="k", lw=.8); ax.axhline(0, color="k", lw=.8, alpha=.5)
    ax.axvspan(-.4, 0, color="#EEE8DA", lw=0, zorder=0)
    ax.set_xlim(-2, 1); ax.set_ylim(-20, 60); ax.set_xlabel("Time from running onset (s)"); ax.set_ylabel("Population rate change (%)")
    ax.set_title(f"E  Firing rises before movement (n={len(Y)})", loc="left")

def panel_F(ax):
    g = SC.groupby("dataset")[["screened", "included"]].sum()
    for y, ds, c, lab in [(1, "VBN", C_VBN, "Visual Behavior"), (0, "VC", C_VC, "Visual Coding")]:
        ax.barh(y, g.loc[ds, "screened"], color="#DBD1BD"); ax.barh(y, g.loc[ds, "included"], color=c)
        ax.text(g.loc[ds, "screened"] + 2, y, f"{g.loc[ds,'included']}/{g.loc[ds,'screened']} sessions", va="center", fontsize=10)
    ax.set_yticks([1, 0]); ax.set_yticklabels(["Visual Behavior", "Visual Coding"]); ax.set_xlim(0, 200)
    ax.set_xlabel("Sessions"); ax.grid(axis="y", visible=False); ax.set_title("F  Sessions screened / included", loc="left")

PANELS = [panel_A, panel_B, panel_C, panel_D, panel_E, panel_F]
fig, axs = plt.subplots(2, 3, figsize=(16, 9.5))
for f, ax in zip(PANELS, axs.ravel()): f(ax)
fig.tight_layout()
for ext in ("svg", "pdf", "png"):
    fig.savefig(os.path.join(OUT, f"stillness_allen_summary.{ext}"), dpi=300, bbox_inches="tight")
plt.close(fig)
for f, L in zip(PANELS, "ABCDEF"):
    fig, ax = plt.subplots(figsize=(5.5, 4.5)); f(ax); fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"panel_{L}.svg"), bbox_inches="tight"); plt.close(fig)
print("figures written to", OUT)
```

## 3. `stillness_pipeline.py` (per-session analysis)

Part a defines the state labelling, binning, epochs and per-neuron statistics; part b runs one session (all probes) and computes
the pre-movement curves. Times are milliseconds; bins are 50 ms. For Visual Coding the pupil arrays were neutral (no pupil gate).

```python
# ---- part a: state labelling and per-neuron statistics ----
import numpy as np
from scipy import stats
BIN = 50.0
def state_bins(G0, G1, run_t, run_v, pup_t, pup_v, pup_blink, task_win, run_thr=2.0, max_since=30000.0, pup_pct=10):
    edges = np.arange(G0, G1 + 1e-9, BIN); nb = len(edges) - 1
    idx = np.clip(((run_t - G0) // BIN).astype(int), -1, nb)
    ok = (idx >= 0) & (idx < nb) & np.isfinite(run_v)
    mx = np.zeros(nb); np.maximum.at(mx, idx[ok], np.abs(run_v[ok]))
    cnt = np.bincount(idx[ok], minlength=nb)[:nb]
    moving = mx > run_thr
    # bins with no running sample inherit nearest-neighbour (running ~60 Hz so rare)
    nos = cnt == 0
    if nos.any():
        good = np.flatnonzero(~nos); moving[nos] = moving[good[np.searchsorted(good, np.flatnonzero(nos)).clip(0, len(good)-1)]]
    still = ~moving
    margin = moving.copy(); margin[1:] |= moving[:-1]; margin[:-1] |= moving[1:]
    still &= ~margin
    # time since last movement
    centers = edges[:-1] + BIN/2
    mv_idx = np.flatnonzero(moving)
    last = np.full(nb, -np.inf)
    if len(mv_idx):
        pos = np.searchsorted(mv_idx, np.arange(nb), side='right') - 1
        has = pos >= 0; last[has] = centers[mv_idx[pos[has]]]
    since = centers - last  # inf if none yet (treated as > max_since only if beyond; start-of-block: count from G0)
    since[~np.isfinite(since)] = centers[~np.isfinite(since)] - G0
    # pupil gate
    tw = (pup_t >= task_win[0]) & (pup_t < task_win[1]) & (pup_blink == 0) & np.isfinite(pup_v)
    thr = np.nanpercentile(pup_v[tw], pup_pct)
    pidx = ((pup_t - G0) // BIN).astype(int); pok = (pidx >= 0) & (pidx < nb) & (pup_blink == 0) & np.isfinite(pup_v)
    psum = np.bincount(pidx[pok], weights=pup_v[pok], minlength=nb)[:nb]; pc = np.bincount(pidx[pok], minlength=nb)[:nb]
    pmean = np.where(pc > 0, psum / np.maximum(pc, 1), np.nan)
    pupil_ok = ~(pmean < thr)  # NaN pupil -> not gated by pupil
    awake_still = still & (since <= max_since) & pupil_ok
    gated_out = still & ~awake_still
    return dict(edges=edges, moving=moving, awake_still=awake_still, gated_out=gated_out, pmean=pmean, pup_thr=thr, since=since)

def bin_counts(trains, nb):
    C = np.zeros((len(trains), nb), np.int32)
    for i, tr in enumerate(trains):
        b = (np.asarray(tr) // BIN).astype(int); b = b[(b >= 0) & (b < nb)]
        np.add.at(C[i], b, 1)
    return C

def epochs(mask, L=10, spacing=40):
    """non-overlapping runs of L bins fully in mask; at most one epoch per `spacing` bins (2 s)."""
    out = []; i = 0; n = len(mask); last = -10**9
    run = 0
    for j in range(n):
        run = run + 1 if mask[j] else 0
        if run >= L:
            s = j - L + 1
            if s // spacing != last // spacing if last >= 0 else True:
                if not out or s >= out[-1] + L:
                    if not out or (s // spacing) != (out[-1] // spacing):
                        out.append(s); last = s
            run = 0 if (out and out[-1] == j - L + 1) else run
    return np.array(out, int)

def neuron_stats(C, st, min_rate=0.5):
    nb = C.shape[1]; dur = nb * BIN / 1000
    rate_all = C.sum(1) / dur
    a, m = st['awake_still'], st['moving']
    r_s = C[:, a].sum(1) / (a.sum() * BIN / 1000); r_m = C[:, m].sum(1) / (m.sum() * BIN / 1000)
    lr = np.log2((r_s + 0.1) / (r_m + 0.1)); si = (r_s - r_m) / np.maximum(r_s + r_m, 1e-12)
    es, em = epochs(a), epochs(m)
    Xs = np.stack([C[:, s:s+10].sum(1) for s in es], 1) * 2.0 if len(es) else np.zeros((C.shape[0], 0))
    Xm = np.stack([C[:, s:s+10].sum(1) for s in em], 1) * 2.0 if len(em) else np.zeros((C.shape[0], 0))
    p = np.full(C.shape[0], np.nan); d = np.zeros(C.shape[0])
    if Xs.shape[1] >= 3 and Xm.shape[1] >= 3:
        with np.errstate(all='ignore'):
            tt = stats.ttest_ind(Xs, Xm, axis=1, equal_var=False)
        p = tt.pvalue; d = Xs.mean(1) - Xm.mean(1)
    pref = np.where((p < 0.01) & (d > 0), 1, np.where((p < 0.01) & (d < 0), -1, 0))
    keep = rate_all >= min_rate
    return dict(rate=rate_all, r_still=r_s, r_move=r_m, log2rr=lr, si=si, p=p, pref=pref, keep=keep, n_ep_still=len(es), n_ep_move=len(em))

# ---- part b: one session ----
from iblatlas.regions import BrainRegions
_br = BrainRegions()
def premove(C, st, pre_still_s=1.0):
    """population-normalised rate around still->moving onsets after >= pre_still_s of awake-still."""
    a, m = st['awake_still'], st['moving']; nb = len(a); k = int(pre_still_s / 0.05)
    on = [i for i in range(k + 1, nb - 20) if m[i] and not m[i - 1] and a[i - 1 - k + 1:i - 1].all()]
    if len(on) < 3: return None, 0
    W = np.arange(-40, 21)
    on = [i for i in on if i + W[0] >= 0 and i + W[-1] < nb]
    P = np.stack([C[:, np.array(on)[:, None] + W].mean(1)], 0)[0] / 0.05  # neurons x time (Hz)
    return P, len(on)

def run_mouse(subject, sd_by_rec, beh, task_end, G0, G1):
    rs = beh.members["running_speed__running"]; pu = beh.members["pupil_area__pupil"]
    st = state_bins(G0, G1, rs.times_ms, rs.values[0], pu.times_ms, pu.values[0], pu.values[1], (0.0, task_end), run_thr=2.0)
    nb = len(st['edges']) - 1
    rows = []; P_all = []; n_on = 0
    for rid, s in sd_by_rec.items():
        g = s.subtime(G0, G1); C = bin_counts([np.asarray(t) for t in g.train], nb)
        ns = neuron_stats(C, st); regs = np.array(unit_regions_canonical(s))
        _valid = set(_br.acronym)
        acr = np.array([r if (r and r in _valid) else 'root' for r in regs])  # v2: unknown labels (e.g. 'none') -> root
        cos = np.asarray(_br.acronym2acronym(acr, mapping='Cosmos'))
        assert len(cos) == len(acr) == len(regs), 'region/cosmos length mismatch'
        P, n_on = premove(C, st)
        for i in range(len(regs)):
            rows.append(dict(subject=subject, rec=rid, region=acr[i], cosmos=cos[i], rate=ns['rate'][i], r_still=ns['r_still'][i],
                             r_move=ns['r_move'][i], log2rr=ns['log2rr'][i], p=ns['p'][i], pref=int(ns['pref'][i]), keep=bool(ns['keep'][i])))
        if P is not None: P_all.append(P)
    P_all = np.concatenate(P_all, 0) if P_all else None
    info = dict(still_min=st['awake_still'].sum()*0.05/60, move_min=st['moving'].sum()*0.05/60,
                gated_min=st['gated_out'].sum()*0.05/60, n_ep_still=len(epochs(st['awake_still'])), n_ep_move=len(epochs(st['moving'])), n_onsets=n_on)
    return rows, P_all, info
```

Pre-movement curves in the exported table are the mean over kept neurons, expressed as % change from the -2 to -1 s baseline
(the first 4 mice were stored already in this normalised form). Sessions with fewer than 5 qualifying onsets were excluded from panel E.
