"""Cached-data sensitivity analyses; run after 20_stats.py and 21_population_stats.py.

BH corrections are separate across all finite neuron p-values for each test
family (epoch contrast and within-bout change). They do not cure temporal
autocorrelation. Session correlations require >=20 complete, varying neurons;
session coefficients are averaged within mouse before the mouse bootstrap.
"""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from common_still import OUT


def bh(p):
    p = np.asarray(p, float)
    q = np.full(p.shape, np.nan)
    good = np.flatnonzero(np.isfinite(p))
    order = good[np.argsort(p[good])]
    q[order] = np.minimum(1, np.minimum.accumulate(
        (p[order] * len(order) / np.arange(1, len(order) + 1))[::-1])[::-1])
    return q


def summary(D, col):
    m = D.groupby('subject')[col].mean().dropna()
    if len(m) < 3:
        return None
    rng = np.random.default_rng(22)
    boot = m.to_numpy()[rng.integers(len(m), size=(4000, len(m)))].mean(1)
    return dict(mean=float(m.mean()), ci95=np.quantile(boot, [.025, .975]).tolist(),
                n_mice=len(m), n_values=int(D[col].notna().sum()), frac_pos=float((m > 0).mean()))


def run():
    N = pd.read_parquet(OUT / '20_neurons.pqt')
    result = {'fdr': {}, 'subsets': {}, 'session_correlations': {}, 'dmn_paired': {}}
    for pcol, metric, name in [('p_epoch', 'SI', 'epoch'), ('p_build', 'BI', 'bout')]:
        q = bh(N[pcol])
        N['q_' + name] = q
        for alpha in [.01, .05]:
            for sign, label in [(1, 'positive'), (-1, 'negative')]:
                c = f'{name}_{label}_q{alpha}'
                N[c] = np.where(np.isfinite(q), ((q < alpha) & (sign*N[metric] > 0)).astype(float), np.nan)
                result['fdr'][c] = dict(pooled_fraction=float(N[c].mean()), **summary(N, c))
    N[['uuid', 'eid', 'q_epoch', 'q_bout']].to_parquet(OUT / '22_neuron_qvalues.pqt', index=False)
    for label, mask in [('all', np.ones(len(N), bool)),
                        ('pupil_and_motion_available', N.pupil_ok & N.me_ok),
                        ('rate_ge_1Hz', N.rate_all >= 1), ('rate_ge_2Hz', N.rate_all >= 2)]:
        D = N[mask]
        result['subsets'][label] = {c: summary(D, c) for c in ['lrate', 'SI', 'epoch_positive_q0.01']}
    for a, b in [('SI_h1', 'SI_h2'), ('SI', 'SI_iti'), ('SI_face', 'SI_wheel')]:
        rows = []
        for (subject, eid), D in N.groupby(['subject', 'eid']):
            D = D[[a, b]].dropna()
            if len(D) >= 20 and (D.nunique() > 1).all():
                rows.append(dict(subject=subject, eid=eid, rho=float(spearmanr(D[a], D[b]).statistic)))
        result['session_correlations'][a + '~' + b] = summary(pd.DataFrame(rows), 'rho')
    # Paired within-mouse comparison; equal weight per observed region within mouse.
    # Mice need both sets; anatomy remains incompletely sampled.
    from common_still import CONS_DMN
    C = N[N.cosmos == 'Isocortex'].copy()
    C['core'] = C.region.isin(CONS_DMN)
    for col in ['lrate', 'SP', 'BI', 'r_pupil']:
        M = C.groupby(['subject', 'core', 'region'])[col].mean().groupby(['subject', 'core']).mean().unstack('core')
        M = M.dropna()
        D = pd.DataFrame({'subject': M.index, 'difference': (M[True] - M[False]).values})
        result['dmn_paired'][col] = summary(D, 'difference')
    P = pd.read_parquet(OUT / '11_population.pqt')
    result['population_common_neurons'] = bool((P.groupby(['eid', 'probe', 'region', 'comp', 'variant']).n_used.nunique() == 1).all())
    result['population_matched_rate_residual'] = json.load(open(OUT / '21_population_stats.json'))['still/move|matched']['all']['lrate']
    (OUT / '22_robustness.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    run()
