# Scripts: follow-up analyses (stillness report, Allen data)

Copy each block into a `.py` file of the given name. Rebuild figure and numbers from the exported tables (no data download):

    python stats_followup.py ./data        # every number in followup_report_section.md
    python plot_followup.py ./data ./figs  # followup_summary.svg/.pdf/.png and fu_panel_A..F.svg

Scripts 3-5 are the analysis code that produced the tables; they need the spike trains and behaviour streams (Allen DANDI 000713, 000563, 000690, 001637 as held on the OCS platform) plus the per-session pipeline from the earlier package (state_bins, epochs, bin_counts).

## 1. `stats_followup.py`
```python
"""Recompute every number in followup_report_section.md from the exported CSVs.
Usage: python stats_followup.py [DATA_DIR]   Bootstrap: 10,000 resamples over mice, seed 20261030."""
import sys, os, numpy as np, pandas as pd
from scipy import stats
D = sys.argv[1] if len(sys.argv) > 1 else "."
rng = np.random.default_rng(20261030)
def boot(x, n=10000):
    x = np.asarray(x, float); return np.percentile(rng.choice(x, (n, len(x))).mean(1), [2.5, 97.5])
NS = pd.read_csv(os.path.join(D, "fu_null_sessions.csv"), dtype={"subject": str})
M = NS.groupby("subject")[["obs_still", "null_still_mean", "still_corr", "obs_move", "null_move_mean", "move_corr"]].mean()
print(f"== 1. Null calibration: {len(NS)} sessions, {NS.subject.nunique()} mice")
for c in M.columns: lo, hi = boot(M[c]); print(f"  {c:16s} {100*M[c].mean():6.2f}%  [{100*lo:.2f}, {100*hi:.2f}]")
print(f"  sessions obs > null p95: still {(NS.obs_still > NS.null_still_p95).sum()}, move {(NS.obs_move > NS.null_move_p95).sum()}; max perm p still {NS.p_still.max():.3f}, move {NS.p_move.max():.3f}")
print(f"  rho(moving epochs, still null) = {stats.spearmanr(NS.n_ep_move, NS.null_still_mean).correlation:.2f}; rho(still epochs, move null) = {stats.spearmanr(NS.n_ep_still, NS.null_move_mean).correlation:.2f}")
w = NS[NS.min_epochs >= 50]; print(f"  >=50 epochs both states: n={len(w)}, null still {100*w.null_still_mean.mean():.2f}%, move {100*w.null_move_mean.mean():.2f}%, corrected {100*w.still_corr.mean():.1f}% / {100*w.move_corr.mean():.1f}%")
PS = pd.read_csv(os.path.join(D, "fu_premove_sessions.csv"), dtype={"subject": str})
print("== 2. Pre-movement rise (-0.5..-0.05 s vs -2..-1 s)")
for rule in ["3s", "1s"]:
    c = PS[PS.rule == rule]; m = c.groupby("subject").delta.mean(); e = c.groupby("subject").early.mean(); lo, hi = boot(m)
    print(f"  {rule}: {len(c)} sessions, {len(m)} mice, {100*m.mean():.1f}% [{100*lo:.1f}, {100*hi:.1f}], positive {(m>0).sum()}/{len(m)}, p={stats.wilcoxon(m).pvalue:.2g}; early -1..-0.5 s {100*e.mean():.1f}% p={stats.wilcoxon(e).pvalue:.2f}; onsets {c.n_onsets.min()}-{c.n_onsets.max()}")
W = PS.pivot_table(index=["subject", "date"], columns="rule", values="delta").dropna(); d = W["3s"] - W["1s"]
print(f"  paired {len(W)}: 3s-1s {100*d.mean():.1f} pts, 3s>1s {(d>0).sum()}, p={stats.wilcoxon(d).pvalue:.2g}")
F = pd.read_csv(os.path.join(D, "fu_coupling_front_rsp.csv"), dtype={"subject": str}); CP = pd.read_csv(os.path.join(D, "fu_coupling_pairs.csv"), dtype={"subject": str})
print(f"== 3. Coupling: {CP.session.nunique()} sessions, {F.session.nunique()} with frontal-RSP pairs, {F.subject.nunique()} mice")
g = F.groupby(["mask", "subject"]).agg(pr=("pct_raw", "mean"), pc=("pct_resid", "mean"), rz=("resid", "mean")).reset_index()
for mk, h in g.groupby("mask"):
    print(f"  {mk:6s}: pct raw {h.pr.mean():.1f} (>50: {(h.pr>50).sum()}, >90: {(h.pr>90).sum()}), dist-corrected {h.pc.mean():.1f}, resid z {h.rz.mean():.3f} (>0 in {(h.rz>0).sum()}/{len(h)}, p={stats.wilcoxon(h.rz).pvalue:.2f})")
s = g.pivot(index="subject", columns="mask", values="rz"); dd = s["still"] - s["moving"]
print(f"  still-moving resid z {dd.mean():.3f}, >0 in {(dd>0).sum()}/{len(dd)}, p={stats.wilcoxon(dd).pvalue:.2f}")
a = CP[CP["mask"] == "all"]; b, _ = np.polyfit(a.dist_um/1000, a.z, 1); print(f"  distance slope {b:.3f} z/mm, rho={stats.spearmanr(a.dist_um, a.z).correlation:.2f}, {len(a)} pairs")
```

## 2. `plot_followup.py`
```python
"""Follow-up figure for the stillness report: null calibration, onset-rule comparison, frontal-RSP coupling.
Usage: python plot_followup.py [DATA_DIR] [OUT_DIR]   (needs numpy, pandas, scipy, matplotlib)
Writes followup_summary.svg/.pdf/.png and fu_panel_A.svg ... fu_panel_F.svg. Bootstrap seed 20261030."""
import sys, os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
DATA = sys.argv[1] if len(sys.argv) > 1 else "."; OUT = sys.argv[2] if len(sys.argv) > 2 else "."; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .15,
                     "svg.fonttype": "none", "pdf.fonttype": 42, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"})
CS, CM, CN, CMAIN, CTHIN, CBAND, CREF = "#3381D5", "#9B60BC", "#DBD1BD", "#4B6733", "#C4CBB1", "#EAC5FF", "#E28E5B"
rng = np.random.default_rng(20261030)
def boot(Y, n=5000):
    Y = np.asarray(Y, float); idx = rng.integers(0, len(Y), (n, len(Y)))
    return np.percentile(Y[idx].mean(1), [2.5, 97.5], axis=0)
NS = pd.read_csv(os.path.join(DATA, "fu_null_sessions.csv"), dtype={"subject": str})
NM = pd.read_csv(os.path.join(DATA, "fu_null_mice.csv"), dtype={"subject": str})
PC = pd.read_csv(os.path.join(DATA, "fu_premove_curves_long.csv"), dtype={"subject": str})
PS = pd.read_csv(os.path.join(DATA, "fu_premove_sessions.csv"), dtype={"subject": str})
CP = pd.read_csv(os.path.join(DATA, "fu_coupling_pairs.csv"), dtype={"subject": str})
FR = pd.read_csv(os.path.join(DATA, "fu_coupling_front_rsp.csv"), dtype={"subject": str})

def panel_A(ax):
    for i, (o, n, c, lab) in enumerate([("obs_still", "null_still_mean", CS, "Still-preferring"), ("obs_move", "null_move_mean", CM, "Movement-preferring")]):
        x0 = 2*i
        for _, r in NM.iterrows(): ax.plot([x0, x0+1], [r[n]*100, r[o]*100], color="#bbbbbb", lw=.6, zorder=1)
        ax.scatter(np.full(len(NM), x0), NM[n]*100, color=CN, s=18, zorder=2); ax.scatter(np.full(len(NM), x0+1), NM[o]*100, color=c, s=18, zorder=2)
    ax.set_xticks([0, 1, 2, 3]); ax.set_xticklabels(["null", "observed", "null", "observed"]); ax.set_ylabel("% of neurons per mouse")
    ax.text(.5, 1.0, "still-pref.", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=10)
    ax.text(2.5, 1.0, "movement-pref.", transform=ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=10)
    ax.set_title(f"A  Labels vs circular-shift null (n={len(NM)} mice)", loc="left", pad=16)

def panel_B(ax):
    ax.scatter(NS.n_ep_move, NS.null_still_mean*100, s=20, color=CS, label="still-pref. null vs moving epochs")
    ax.scatter(NS.n_ep_still, NS.null_move_mean*100, s=20, color=CM, label="movement-pref. null vs still epochs")
    ax.axhline(1, color=CREF, ls="--", lw=1.2); ax.set_xscale("log"); ax.set_xlabel("Epochs in the other state (0.5 s)")
    ax.set_ylabel("Null false-positive rate (%)"); ax.legend(fontsize=9, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    ax.set_title("B  Label is liberal when epochs are few", loc="left")

def mouse_curves(rule):
    d = PC[PC.rule == rule]; Y = d.pivot_table(index=["subject", "date"], columns="t_s", values="frac_change")
    Y = Y.groupby(level=0).mean(); return Y.columns.values.astype(float), Y.values*100
def panel_C(ax):
    for rule, c, lab in [("3s", CMAIN, "≥3 s prior stillness"), ("1s", CM, "≥1 s prior stillness")]:
        t, Y = mouse_curves(rule); lo, hi = boot(Y)
        ax.fill_between(t, lo, hi, color=c, alpha=.18, lw=0); ax.plot(t, Y.mean(0), color=c, lw=2, label=f"{lab} (n={len(Y)} mice)")
    ax.axvline(0, color="k", lw=.8); ax.axhline(0, color="k", lw=.8, alpha=.5); ax.axvspan(-.5, -.05, color="#EEE8DA", lw=0, zorder=0)
    ax.set_xlim(-2, 1); ax.set_xlabel("Time from running onset (s)"); ax.set_ylabel("Population rate change (%)")
    ax.legend(fontsize=9, frameon=False, loc="upper left"); ax.set_title("C  Pre-movement rise by onset rule", loc="left")

def panel_D(ax):
    W = PS.pivot_table(index=["subject", "date"], columns="rule", values="delta").dropna()*100
    ax.scatter(W["1s"], W["3s"], s=30, color=CMAIN, zorder=3); lim = [min(W.min().min(), -5)-2, W.max().max()+3]
    ax.plot(lim, lim, color="k", lw=.8); ax.axhline(0, color="k", lw=.5, alpha=.5); ax.axvline(0, color="k", lw=.5, alpha=.5)
    p = stats.wilcoxon(W["3s"]-W["1s"]).pvalue
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_xlabel("Rise, ≥1 s rule (%)"); ax.set_ylabel("Rise, ≥3 s rule (%)")
    ax.text(.04, .94, f"{(W['3s']>W['1s']).sum()}/{len(W)} sessions above diagonal\nWilcoxon p = {p:.1g}", transform=ax.transAxes, fontsize=10, va="top")
    ax.set_title("D  Stricter rule, larger rise", loc="left")

def panel_E(ax):
    g = FR.groupby(["mask", "subject"])[["pct_raw", "pct_resid"]].mean().reset_index()
    for i, mk in enumerate(["all", "still", "moving"]):
        d = g[g["mask"] == mk]
        for j, (col, c) in enumerate([("pct_raw", CS), ("pct_resid", CM)]):
            x = i + (j - .5)*.3; ax.scatter(x + rng.uniform(-.05, .05, len(d)), d[col], color=c, s=22, alpha=.8, label=(["raw", "distance-corrected"][j] if i == 0 else None))
            ax.errorbar(x + .12, d[col].mean(), yerr=[[d[col].mean()-np.percentile(d[col], 25)], [np.percentile(d[col], 75)-d[col].mean()]], fmt="o", color="k", capsize=3, ms=4)
    ax.axhline(50, color="k", lw=.8); ax.axhline(90, color=CREF, ls="--", lw=1)
    ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["all bins", "still", "moving"]); ax.set_ylim(0, 110)
    ax.set_ylabel("Frontal–RSP percentile among\ncross-probe region pairs"); ax.legend(fontsize=9, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax.set_title(f"E  Frontal–RSP coupling (n={FR.subject.nunique()} mice)", loc="left")

def panel_F(ax):
    d = CP[CP["mask"] == "all"]
    o = d[d.kind != "FRONT-RSP"]; f = d[d.kind == "FRONT-RSP"]
    ax.scatter(o.dist_um/1000, o.z, s=10, color=CN, label="other cross-probe pairs")
    ax.scatter(f.dist_um/1000, f.z, s=26, color=CMAIN, label="frontal core – RSP")
    b, a = np.polyfit(d.dist_um/1000, d.z, 1); xx = np.linspace(d.dist_um.min()/1000, d.dist_um.max()/1000, 10)
    ax.plot(xx, a+b*xx, color="k", lw=1); ax.set_xlabel("Distance between region centres (mm)"); ax.set_ylabel("Coupling (Fisher z, 1 s bins)")
    ax.legend(fontsize=9, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2); ax.set_title("F  Coupling barely depends on distance", loc="left")

PANELS = [panel_A, panel_B, panel_C, panel_D, panel_E, panel_F]
fig, axs = plt.subplots(2, 3, figsize=(16.5, 10.5))
for fn, ax in zip(PANELS, axs.ravel()): fn(ax)
fig.tight_layout()
for ext in ("svg", "pdf", "png"): fig.savefig(os.path.join(OUT, f"followup_summary.{ext}"), dpi=300, bbox_inches="tight")
plt.close(fig)
for fn, L in zip(PANELS, "ABCDEF"):
    fig, ax = plt.subplots(figsize=(5.8, 4.6)); fn(ax); fig.tight_layout(); fig.savefig(os.path.join(OUT, f"fu_panel_{L}.svg"), bbox_inches="tight"); plt.close(fig)
print("figures written to", OUT)
```

## 3. `null_onset.py` (circular-shift null; onset rules)
```python
import numpy as np
from scipy import stats
def _pref_from_counts(C, es, em, alpha=0.01):
    """still/moving preference labels from 50 ms counts C (neurons x bins) and epoch starts (10-bin epochs)."""
    if len(es) < 3 or len(em) < 3: return np.zeros(C.shape[0], int)
    cs = np.cumsum(np.concatenate([np.zeros((C.shape[0],1)), C], 1), 1)
    Xs = (cs[:, es+10] - cs[:, es]) * 2.0; Xm = (cs[:, em+10] - cs[:, em]) * 2.0
    with np.errstate(all='ignore'):
        tt = stats.ttest_ind(Xs, Xm, axis=1, equal_var=False)
    d = Xs.mean(1) - Xm.mean(1); p = tt.pvalue
    return np.where((p < alpha) & (d > 0), 1, np.where((p < alpha) & (d < 0), -1, 0))

def shift_null(C, st, keep, n_shift=200, min_shift_s=10.0, seed=20261030):
    """Circular-shift null for the still/moving preference label.
    Each shift rotates ALL neurons' 50 ms count rows together by the same random offset (>= min_shift_s from 0 and
    from the end), keeping the behavioural labels fixed. This preserves each neuron's autocorrelation and the
    population co-fluctuations, and breaks only the alignment to behaviour. Returns observed and null fractions."""
    a, m = st['awake_still'], st['moving']
    es, em = epochs(a), epochs(m)
    Ck = C[keep]
    obs = _pref_from_counts(Ck, es, em)
    nb = C.shape[1]; k0 = int(min_shift_s / 0.05)
    rng = np.random.default_rng(seed)
    if nb <= 2 * k0: return None
    offs = rng.integers(k0, nb - k0, n_shift)
    fs, fm = np.empty(n_shift), np.empty(n_shift)
    for i, o in enumerate(offs):
        pr = _pref_from_counts(np.roll(Ck, o, axis=1), es, em)
        fs[i] = np.mean(pr == 1); fm[i] = np.mean(pr == -1)
    return dict(obs_still=float(np.mean(obs == 1)), obs_move=float(np.mean(obs == -1)), null_still=fs, null_move=fm,
                n_neurons=int(keep.sum()), n_ep_still=len(es), n_ep_move=len(em), n_shift=n_shift, seed=seed)

def premove_rule(C, st, keep, pre_still_s=3.0):
    """Population rate (Hz, mean over kept neurons) around still->moving onsets preceded by >= pre_still_s of
    awake stillness; window -2.0..+1.0 s in 50 ms bins. Returns (curve_Hz [61], n_onsets) or (None, n)."""
    a, m = st['awake_still'], st['moving']; nb = len(a); k = int(round(pre_still_s / 0.05))
    W = np.arange(-40, 21)
    on = [i for i in range(max(k, 40) + 1, nb - 21) if m[i] and not m[i - 1] and a[i - k:i - 1].all()]
    if len(on) < 1: return None, 0
    P = C[keep][:, np.array(on)[:, None] + W].mean(1).mean(0) / 0.05
    return P, len(on)
```

## 4. `coupling.py` (regional rates, cross-probe coupling, still/moving masks)
```python
import numpy as np
FRONT = ('ACA', 'PL', 'ORBm')        # frontal DMN core (prefix match on Allen acronym with layer stripped)
RSPP = ('RSP',)                       # RSPagl, RSPd, RSPv
def region_group(acr):
    import re
    a = re.sub(r'(\d.*)$', '', str(acr))          # strip layer suffix: RSPd5 -> RSPd, ACAd6a -> ACAd
    if a.startswith(('ACAd','ACAv','PL','ORBm')): return 'FRONT', a
    if a.startswith('RSP'): return 'RSP', a
    return 'OTHER', a

def regional_rates(trains_by_rec, regions_by_rec, t0, t1, bin_ms=1000.0, min_units=10, min_rate=0.5):
    """1 s population rates per (probe, region-without-layer). trains in ms. Only regions with >= min_units
    units at >= min_rate Hz. Returns dict key=(rec_id, region) -> rate vector, and edges."""
    edges = np.arange(t0, t1 + 1e-9, bin_ms); out = {}
    for rid, trains in trains_by_rec.items():
        regs = [region_group(r)[1] for r in regions_by_rec[rid]]
        groups = {}
        for tr, rg in zip(trains, regs):
            tr = np.asarray(tr); n = np.searchsorted(tr, [t0, t1]); r = (n[1]-n[0]) / ((t1-t0)/1000)
            if r >= min_rate and rg not in ('', 'root', 'none'): groups.setdefault(rg, []).append(tr)
        for rg, trs in groups.items():
            if len(trs) < min_units: continue
            cnt = np.zeros(len(edges)-1)
            for tr in trs: cnt += np.histogram(tr, edges)[0]
            out[(rid, rg)] = cnt / len(trs) / (bin_ms/1000)
    return out, edges

def detrend(x, win_bins=60):
    k = np.ones(win_bins)/win_bins
    pad = np.pad(x, (win_bins//2, win_bins - win_bins//2 - 1), mode='edge')
    return x - np.convolve(pad, k, mode='valid')

def pair_coupling(rates, mask, win_bins=60):
    """Fisher-z Pearson correlation of detrended regional rates, over bins in mask, for every CROSS-PROBE pair."""
    keys = list(rates); res = []
    X = {k: detrend(rates[k], win_bins) for k in keys}
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            (ri, gi), (rj, gj) = keys[i], keys[j]
            if ri == rj: continue                         # cross-probe only
            a, b = X[keys[i]][mask], X[keys[j]][mask]
            if a.std() == 0 or b.std() == 0 or mask.sum() < 30: continue
            r = np.corrcoef(a, b)[0, 1]
            ci, cj = region_group(gi)[0], region_group(gj)[0]
            kind = 'FRONT-RSP' if {ci, cj} == {'FRONT', 'RSP'} else ('core-other' if ('OTHER' in (ci, cj) and (ci != 'OTHER' or cj != 'OTHER')) else ('other-other' if ci == cj == 'OTHER' else 'core-core-same'))
            res.append(dict(rec_i=ri, reg_i=gi, rec_j=rj, reg_j=gj, kind=kind, r=r, z=np.arctanh(np.clip(r, -0.999, 0.999)), n_bins=int(mask.sum())))
    return res

def state_masks_1s(edges, run_t, run_v, thr=2.0, margin_s=1):
    """1 s bins: moving if max |speed| > thr in the bin; still if no movement in the bin or +-margin_s neighbours."""
    nb = len(edges) - 1; idx = np.clip(((run_t - edges[0]) // (edges[1]-edges[0])).astype(int), -1, nb)
    ok = (idx >= 0) & (idx < nb) & np.isfinite(run_v)
    mx = np.zeros(nb); np.maximum.at(mx, idx[ok], np.abs(run_v[ok])); mv = mx > thr
    near = mv.copy()
    for s in range(1, margin_s+1): near[s:] |= mv[:-s]; near[:-s] |= mv[s:]
    return ~near, mv
```

## 5. `combine_followup.py` (collects per-session outputs into the exported tables)
```python
import numpy as np, pandas as pd
from scipy import stats
def collect_null_premove(ws, prefixes):
    rows=[]; curves=[]; seen=set()
    for p in prefixes:
        for k in ws.list_keys(p):
            if not k.startswith('null__'): continue
            _,s,d = k.split('__')
            if (s,d) in seen: continue
            seen.add((s,d)); r = ws.get(p,k); pm = ws.get(p, f'premove_rules__{s}__{d}')
            ns, nm = np.asarray(r['null_still'],float), np.asarray(r['null_move'],float)
            rows.append(dict(subject=s, date=d, n_neurons=int(r['n_neurons']), n_ep_still=int(r['n_ep_still']), n_ep_move=int(r['n_ep_move']),
                obs_still=float(r['obs_still']), null_still_mean=ns.mean(), null_still_p95=np.percentile(ns,95), p_still=(1+(ns>=r['obs_still']).sum())/(1+len(ns)),
                obs_move=float(r['obs_move']), null_move_mean=nm.mean(), null_move_p95=np.percentile(nm,95), p_move=(1+(nm>=r['obs_move']).sum())/(1+len(nm)),
                n_onsets_3s=int(pm['n3']), n_onsets_1s=int(pm['n1'])))
            for rule,P,n in [('3s',pm['P3'],pm['n3']),('1s',pm['P1'],pm['n1'])]:
                P=np.asarray(P,float)
                if n>=3 and np.isfinite(P).all():
                    c=P/P[:20].mean()-1          # fractional change vs -2..-1 s baseline
                    curves.append(dict(subject=s,date=d,rule=rule,n_onsets=int(n),curve=c,delta=c[30:39].mean(),early=c[20:30].mean()))
    N=pd.DataFrame(rows); N['still_corr']=N.obs_still-N.null_still_mean; N['move_corr']=N.obs_move-N.null_move_mean
    N['min_epochs']=N[['n_ep_still','n_ep_move']].min(1)
    return N, pd.DataFrame(curves)
```

Distance correction (in the combining step): region centres = mean voxel position of each region (descendants included, right hemisphere) in the Allen CCF 25 um atlas (iblatlas AllenAtlas, BrainRegions); per session and mask, residual of z from a linear fit on centre distance (mm) across cross-probe pairs; percentile of each frontal-RSP pair's residual among that session's pairs.
