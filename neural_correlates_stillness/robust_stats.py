"""Bootstrap region contrasts with equal weight per observed mouse-region mean."""
import numpy as np


def mouse_region_boot(N, regions, col, members, others, n=1000, seed=0):
    M = N[N.region.isin(regions)].groupby(['subject', 'region'])[col].mean().unstack('region').reindex(columns=regions)
    M = M.dropna(how='all')
    rng = np.random.default_rng(seed)
    W = rng.multinomial(len(M), np.full(len(M), 1/len(M)), size=n).astype(float)
    values = M.to_numpy()
    with np.errstate(invalid='ignore', divide='ignore'):
        means = (W @ np.nan_to_num(values)) / (W @ np.isfinite(values).astype(float))
    a, b = np.isin(M.columns, members), np.isin(M.columns, others)
    d = np.nanmean(means[:, a], axis=1) - np.nanmean(means[:, b], axis=1)
    return np.nanquantile(d, [.025, .975]).round(4).tolist()
