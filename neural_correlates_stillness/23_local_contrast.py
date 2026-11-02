"""Control slow drift using state contrasts inside shared 10 s / 30 s windows.

A window must contain >=0.5 s in each state; require >=5 qualifying
windows/session. Each window contributes equally to both state's mean rates.
This changes the sampled estimand; it is a drift sensitivity check, not a
causal effect or an autocorrelation-corrected neuron significance test.
"""
from concurrent.futures import ProcessPoolExecutor
import json
import numpy as np
import pandas as pd
from common_still import BIN, OUT, load_session, probes, states
from importlib import import_module
summary = import_module('22_robustness').summary


def session(eid):
    st = states(load_session(eid))
    S, M = st['grey'] & st['awake'], st['grey'] & st['mov']
    ga = np.flatnonzero(st['grey'])
    windows = {}
    for seconds in [10, 30]:
        pairs = []
        for a in range(ga[0], ga[-1]+1, int(seconds/BIN)):
            ix = np.arange(a, min(a+int(seconds/BIN), ga[-1]+1))
            si, mi = ix[S[ix]], ix[M[ix]]
            if len(si) >= 10 and len(mi) >= 10:
                pairs.append((si, mi))
        if len(pairs) >= 5:
            windows[seconds] = pairs
    rows = []
    for f in probes(eid):
        with np.load(f, allow_pickle=True) as p:
            X = p['counts']
            for seconds, pairs in windows.items():
                rs = np.mean([X[:, si].mean(1)/BIN for si, _ in pairs], axis=0)
                rm = np.mean([X[:, mi].mean(1)/BIN for _, mi in pairs], axis=0)
                rows.append(pd.DataFrame(dict(uuid=p['uuids'].astype(str), eid=eid,
                    window_s=seconds, n_windows=len(pairs), local_lrate=np.log2((rs+.1)/(rm+.1)))))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


if __name__ == '__main__':
    N = pd.read_parquet(OUT/'20_neurons.pqt')
    with ProcessPoolExecutor(6) as ex:
        T = pd.concat(ex.map(session, sorted(N.eid.unique())), ignore_index=True)
    N['uuid'] = N.uuid.astype(str)
    T = T.merge(N[['eid', 'uuid', 'subject', 'cosmos', 'lrate']], on=['eid', 'uuid'], validate='many_to_one')
    T.to_parquet(OUT/'23_local_contrast.pqt', index=False)
    res = {}
    for seconds, D in T.groupby('window_s'):
        res[str(seconds)] = {'all': summary(D, 'local_lrate'), 'original_same_neurons': summary(D, 'lrate'),
            'n_sessions': int(D.eid.nunique()), 'divisions': {c: summary(g, 'local_lrate') for c,g in D.groupby('cosmos')}}
    (OUT/'23_local_contrast.json').write_text(json.dumps(res, indent=2, allow_nan=False)+'\n')
    print(json.dumps(res, indent=2))
