"""Statistics for 11_population.pqt: paired state differences per region and probe, summarised over mice.

For each comparison (still - move; still_late - still_early) and variant (raw; rate-matched by thinning), the
difference of corr, pr and burst per region-probe is averaged per mouse (all regions, or one Cosmos division);
reported: mean over mice, mouse-bootstrap CI, fraction of mice with a positive mean. Region-level means for
regions with >= 3 mice.
Usage: python 21_population_stats.py
"""
import json

import numpy as np
import pandas as pd

from common_still import OUT

P = pd.read_parquet(OUT / "11_population.pqt")
rng = np.random.default_rng(0)
KEY = ["eid", "probe", "region", "cosmos", "subject", "n_neu"]


def diffs(comp, a, b, variant):
    Q = P[(P.comp == comp) & (P.variant == variant)].pivot_table(index=KEY, columns="state",
                                                                 values=["corr", "pr", "burst", "rate"])
    Q = Q.dropna()
    D = pd.DataFrame({k: Q[k][a] - Q[k][b] for k in ("corr", "pr", "burst")})
    D["lrate"] = np.log2(Q["rate"][a] / Q["rate"][b])
    for k in ("corr", "pr"):
        D[f"{k}_{a}"], D[f"{k}_{b}"] = Q[k][a], Q[k][b]
    return D.reset_index()


def over_mice(D, col):
    m = D.groupby("subject")[col].mean().dropna()
    b = np.array([m.values[rng.integers(0, len(m), len(m))].mean() for _ in range(2000)])
    return {"mean": round(float(m.mean()), 5), "ci95": np.quantile(b, [.025, .975]).round(5).tolist(),
            "frac_mice_pos": round(float((m > 0).mean()), 3), "n_mice": int(len(m)), "n_regprobe": int(len(D))}


res, tabs = {}, []
for comp, a, b in (("still/move", "still", "move"), ("still_early/still_late", "still_late", "still_early")):
    for variant in ("raw", "matched"):
        D = diffs(comp, a, b, variant)
        D["comp"], D["variant"] = comp, variant
        tabs.append(D)
        key = f"{comp}|{variant}"
        res[key] = {"all": {c: over_mice(D, c) for c in ("corr", "pr", "burst", "lrate")}}
        for cz in sorted(D.cosmos.unique()):
            Dc = D[D.cosmos == cz]
            if Dc.subject.nunique() >= 5:
                res[key][cz] = {c: over_mice(Dc, c) for c in ("corr", "pr", "burst")}
T = pd.concat(tabs, ignore_index=True)
T.to_parquet(OUT / "21_population_diffs.pqt")
g = T.groupby(["comp", "variant", "region"])
R = pd.concat([g[["corr", "pr", "burst", "lrate"]].mean(), g.subject.nunique().rename("n_subj"),
               g.size().rename("n"), g.cosmos.first()], axis=1)
R[R.n_subj >= 3].to_csv(OUT / "21_population_regions.csv")
json.dump(res, open(OUT / "21_population_stats.json", "w"), indent=1)
for k, v in res.items():
    print(k)
    for cz, d in v.items():
        print(f"  {cz:10s}", {c: (d[c]["mean"], d[c]["frac_mice_pos"], d[c]["n_mice"]) for c in d})
