"""Statistics for the per-neuron stillness correlates (10_neurons.pqt).

Metrics per neuron (rates in Hz; contrasts are still minus moving unless stated):
  lrate      log2((still + 0.1) / (move + 0.1)) on the grey screen
  SI         (still - move) / (still + move); SI_h1 / SI_h2 halves of the grey screen; SI_iti in inter-trial gaps;
             SI_face / SI_wheel against face/body-only movement and against wheel turns
  SP / MP    still-preferring / movement-preferring: epoch Welch test p < 0.01 with SI > 0 / < 0
  BI         (late - early) / (late + early) within >= 4 s awake-still bouts (2-4 s vs first 0.5 s);
             BU / BD: Wilcoxon over bouts p < 0.01 with BI > 0 / < 0
  r_pupil    partial correlation with pupil within awake stillness (time, motion energy controlled)
  l_lowpup   log2((still, pupil below gate + 0.1) / (awake-still + 0.1)): drowsy-looking vs alert stillness
Robustness across animals: neuron means per mouse, then the fraction of mice with the majority sign and a mouse
bootstrap CI of the mean over mice. Region specificity: regions with >= 20 neurons from >= 3 mice; region-map
split-half reliability over mice (raw and with each neuron's session mean subtracted, i.e. within session);
variance of neuron metrics explained by session and by region within session.
Usage: python 20_stats.py
"""
import json

import numpy as np
import pandas as pd

from common_still import CONS_DMN, OUT
from dmnstats import perm_diff, split_half
from robust_stats import mouse_region_boot

N = pd.read_parquet(OUT / "10_neurons.pqt")
si = lambda a, b: (a - b) / (a + b + 1e-9)
lr = lambda a, b: np.log2((a + 0.1) / (b + 0.1))
N["lrate"] = lr(N.r_still, N.r_move)
N["SI"] = si(N.r_still, N.r_move)
N["SI_h1"], N["SI_h2"] = si(N.r_still_h1, N.r_move_h1), si(N.r_still_h2, N.r_move_h2)
N["SI_iti"] = si(N.r_iti_still, N.r_iti_move)
N["SI_face"], N["SI_wheel"] = si(N.r_still, N.r_face), si(N.r_still, N.r_wheel)
N["SI_nogate"] = si(N.r_still_nogate, N.r_move)
N["SP"] = ((N.p_epoch < .01) & (N.SI > 0)).astype(float)
N["MP"] = ((N.p_epoch < .01) & (N.SI < 0)).astype(float)
N.loc[N.p_epoch.isna(), ["SP", "MP"]] = np.nan
N["BI"] = si(N.r_late, N.r_early)
N["BU"] = ((N.p_build < .01) & (N.BI > 0)).astype(float)
N["BD"] = ((N.p_build < .01) & (N.BI < 0)).astype(float)
N.loc[N.p_build.isna(), ["BU", "BD"]] = np.nan
N["l_lowpup"] = lr(N.r_lowpup, N.r_still)
N.loc[N.n_still.isna(), "l_lowpup"] = np.nan
# time-since-movement profile relative to the neuron's awake-still rate
for k in range(6):
    N[f"ltsm_{k}"] = lr(N[f"tsm_{k}"], N.r_still)
    N[f"lttn_{k}"] = lr(N[f"ttn_{k}"], N.r_still)
N["l_tsm_late_early"] = lr(N.tsm_3, N.tsm_1)  # 2-4 s vs 0.5-1 s since last movement
N.to_parquet(OUT / "20_neurons.pqt")

METRICS = ["lrate", "SI", "SP", "MP", "SI_iti", "SI_face", "SI_wheel", "BI", "BU", "BD", "r_pupil", "l_lowpup",
           "l_tsm_late_early", "SI_nogate"]
rng = np.random.default_rng(0)


def mouse_summary(D, col):
    m = D.groupby("subject")[col].mean().dropna()
    if len(m) < 3:
        return None
    b = np.array([m.values[rng.integers(0, len(m), len(m))].mean() for _ in range(2000)])
    return {"mean_over_mice": round(float(m.mean()), 4), "ci95": np.quantile(b, [.025, .975]).round(4).tolist(),
            "frac_mice_pos": round(float((m > 0).mean()), 3), "frac_mice_neg": round(float((m < 0).mean()), 3),
            "n_mice": int(len(m))}


res = {"n_neurons": int(len(N)), "n_sessions": int(N.eid.nunique()), "n_mice": int(N.subject.nunique()),
       "n_regions": int(N.region.nunique())}
res["global"] = {c: mouse_summary(N, c) for c in METRICS}
res["global_fraction_neurons"] = {c: round(float(N[c].mean()), 4) for c in ("SP", "MP", "BU", "BD")}
from scipy.stats import spearmanr
res["neuron_consistency_spearman"] = {
    f"{a}~{b}": round(float(spearmanr(*N[[a, b]].dropna().values.T).statistic), 3)
    for a, b in (("SI_h1", "SI_h2"), ("SI", "SI_iti"), ("SI_face", "SI_wheel"), ("SI", "BI"), ("SI", "r_pupil"),
                 ("SI", "SI_nogate"), ("SI", "l_lowpup"))}
# per Cosmos
res["cosmos"] = {}
for c in sorted(N.cosmos.unique()):
    D = N[N.cosmos == c]
    res["cosmos"][c] = {"n": int(len(D)), "n_mice": int(D.subject.nunique()),
                        **{m: mouse_summary(D, m) for m in ("lrate", "SP", "MP", "BI", "BU", "BD", "r_pupil",
                                                            "l_lowpup", "l_tsm_late_early")}}

# region table
g = N.groupby("region")
R = pd.DataFrame({"n": g.size(), "n_sess": g.eid.nunique(), "n_subj": g.subject.nunique(),
                  "cosmos": g.cosmos.agg(lambda s: s.mode().iat[0])})
for c in METRICS:
    R[c] = N.groupby(["region", "subject"])[c].mean().groupby("region").mean()
R = R[(R.n >= 20) & (R.n_subj >= 3)].copy()
# within-session versions: neuron metric minus its session mean
for c in ("lrate", "SP", "MP", "BI", "r_pupil", "l_tsm_late_early"):
    N[f"ws_{c}"] = N[c] - N.groupby("eid")[c].transform("mean")
    R[f"ws_{c}"] = N.groupby(["region", "subject"])[f"ws_{c}"].mean().groupby("region").mean().reindex(R.index)


def region_boot(col, n=1000):
    lo, hi, fsign = [], [], []
    for r in R.index:
        m = N[N.region == r].groupby("subject")[col].mean().dropna().values
        if len(m) < 3:
            lo.append(np.nan)
            hi.append(np.nan)
            fsign.append(np.nan)
            continue
        b = np.array([m[rng.integers(0, len(m), len(m))].mean() for _ in range(n)])
        lo.append(np.quantile(b, .025))
        hi.append(np.quantile(b, .975))
        fsign.append((np.sign(m) == np.sign(np.median(m))).mean())
    return lo, hi, fsign


for c in ("lrate", "SP", "MP", "BI", "r_pupil", "l_tsm_late_early", "ws_lrate"):
    R[f"{c}_lo"], R[f"{c}_hi"], R[f"{c}_signcons"] = region_boot(c)
R.to_csv(OUT / "20_regions.csv")
res["n_regions_tested"] = int(len(R))
res["region_reliability_over_mice"] = {}
for c in ("lrate", "SP", "MP", "SI_iti", "BI", "BU", "BD", "r_pupil", "l_lowpup", "l_tsm_late_early",
          "ws_lrate", "ws_SP", "ws_MP", "ws_BI", "ws_r_pupil", "ws_l_tsm_late_early"):
    res["region_reliability_over_mice"][c] = split_half(N.dropna(subset=[c]), R, c, unit="subject", n=200)

# variance decomposition of neuron metrics
res["variance_explained"] = {}
for c in ("lrate", "SI", "BI", "r_pupil", "l_tsm_late_early"):
    D = N.dropna(subset=[c])
    tot = ((D[c] - D[c].mean()) ** 2).sum()
    ss_sess = ((D.groupby("eid")[c].transform("mean") - D[c].mean()) ** 2).sum()
    ss_reg = ((D.groupby("region")[c].transform("mean") - D[c].mean()) ** 2).sum()
    ss_sr = ((D.groupby(["eid", "region"])[c].transform("mean") - D.groupby("eid")[c].transform("mean")) ** 2).sum()
    # null for region-within-session: shuffle region labels within session
    nul = []
    for _ in range(20):
        sh = D.groupby("eid").region.transform(lambda s: s.sample(frac=1, random_state=int(rng.integers(1e9))).values)
        nul.append(((D.groupby([D.eid, sh])[c].transform("mean") - D.groupby("eid")[c].transform("mean")) ** 2).sum())
    res["variance_explained"][c] = {"session": round(float(ss_sess / tot), 4), "region": round(float(ss_reg / tot), 4),
                                    "region_within_session": round(float(ss_sr / tot), 4),
                                    "region_within_session_shuffle": round(float(np.mean(nul) / tot), 4)}

# DMN core vs other cortex (link to the grey-screen report)
C = R[R.cosmos == "Isocortex"]
isd = C.index.isin(CONS_DMN)
res["dmn_core_vs_other_ctx"] = {}
for c in ("lrate", "SP", "MP", "BI", "r_pupil"):
    obs, p = perm_diff(C[c].values[isd], C[c].values[~isd])
    res["dmn_core_vs_other_ctx"][c] = {"core": round(float(C[c].values[isd].mean()), 4),
                                       "other": round(float(C[c].values[~isd].mean()), 4), "p_perm": round(p, 4),
                                       "mouse_ci": mouse_region_boot(N, R.index, c, C.index[isd], C.index[~isd])}
json.dump(res, open(OUT / "20_stats.json", "w"), indent=1)
print(json.dumps({k: res[k] for k in ("n_neurons", "n_sessions", "n_mice", "n_regions_tested", "global",
                                      "global_fraction_neurons", "neuron_consistency_spearman",
                                      "region_reliability_over_mice", "variance_explained",
                                      "dmn_core_vs_other_ctx")}, indent=1))
