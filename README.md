# Default mode network signatures in IBL data

Pilot analyses (started October 2026) asking whether IBL Neuropixels (Brain Wide
Map, BWM) and mesoscope data show signatures of the mouse default mode network
(DMN) known from resting-state fMRI.

**Reports**
- `comprehensive_report/dmn_grey.html`: the comprehensive report (also as `dmn_grey.tex`/`.pdf`). Open it in a browser; the figures sit next to it. It applies one set of tests (correlates of awake stillness; evidence for a DMN analogue) to each dataset in its own part: IBL Neuropixels (Part II: grey-screen period, task suppression of DMN regions, testable region pairs, resting coupling, single-region metrics, brain-wide stillness correlates), the IBL mesoscope (Part III: RSP coupling, stillness), Allen Institute Neuropixels (Part IV: stillness compared with IBL, label calibration and onset rule, frontal–retrosplenial coupling in OpenScope; exports in `comprehensive_report/allen/`) and Allen imaging (Part V, not analysed), then compares them (Part VI). Figures are written there by `report/fig_*.py` (incl. `fig_meso_still.py`), `neural_correlates_stillness/report/figs.py main`, `fig_allen.py` and `fig_allen_followup.py`; the PDF by `neural_correlates_stillness/report/html2tex.py ../../comprehensive_report/dmn_grey.html`.
- `report/REPORT.md` with `report/fig_dmn.pdf`: the earlier summary based on task inter-trial intervals (ITIs) as rest.
- `report/arousal/AROUSAL.md` with `figA1_cells.pdf`, `figA2_hemo.pdf`: the Landemard et al. 2026 Arousal+/Arousal− model applied to the conservative core.

**Short version** (report v5)
- **Region definition:** the main analyses use a conservative DMN core (ACA, RSP, PL, ORBm). ACA and RSP are in all four published rodent DMN maps (Lu 2012, Grandjean 2020, Whitesell 2021, Mandino 2022), and PL and ORB in three.
- **No deactivation:** across behavioural windows the core's population firing is never below its own rest level. It rises 19% during movement and feedback, against 39% in other cortex.
- **Weak:** the core is only modestly more task-suppressed than other cortex (42% vs 35%, permutation p = 0.07). The earlier, stronger effect came from ILA, ECT, PERI and TEa.
- **Untestable with BWM:** a resting-state DMN *network*. Frontal core and RSP are never recorded together, and the recorded core pairs show no preferential coupling.

DMN regions (Allen Beryl level, `bwm/common.py`):
- **Core:** ACAd, ACAv, PL, ILA, ORBm, ORBl, ORBvl, RSPagl, RSPd, RSPv, VISa, VISam.
- **Lateral:** TEa, ECT, PERI.
- **Conservative core (main):** `CONS_DMN` = ACAd, ACAv, PL, ORBm, RSPagl, RSPd, RSPv. `report/fig_regions.py` draws the Venn diagram that justifies it.
- **Published maps:** `common.py` also holds the Lu et al. 2012, Grandjean et al. 2020, Whitesell et al. 2021 and Mandino et al. 2022 DMN sets. Whitesell et al. do not include our lateral regions.

## Setup
- **Environment:** the IBL `iblenv` conda environment (ONE, ibllib, iblatlas, brainwidemap). Activate it before running anything.
- **Sibling checkouts** (paths overridable with environment variables):
  - [`int-brain-lab/bwm_dmn`](https://github.com/int-brain-lab/bwm_dmn) for `dmn_bwm.py` (BWM loaders with the trials-tag fix): `DMN_CODE`, default `~/dmn`.
  - The PETH bundles and pipeline tables used by the early pilots (`00`–`05`): `DMN_DATA`, default `~/dmn`.
  - [`int-brain-lab/mesoscope`](https://github.com/int-brain-lab/mesoscope) for the mesoscope loaders in `pilot_analyses/dorsal_maps`: `MESOSCOPE_REPO`, default `~/mesoscope`.
- **ONE cache:** `ONE_CACHE`, default `~/Downloads/ONE`.
- **Outputs:** they go to `bwm/output/` and `meso/output/`. Large intermediate files (per-session `.npz`, `.pqt` tables) are not in git. The small JSON/CSV result files that the reports quote are committed.

## Pipeline
`run_all.sh` runs everything:
1. BWM extraction (`bwm/10_extract_full.py`). It needs about 10 GB RAM per worker and runs 2 workers by default.
2. The BWM analysis battery (`bwm/run_full.sh`).
3. The mesoscope extraction and analyses.

The grey-screen analyses are run separately (see below).

### bwm/
| script | content |
|---|---|
| 00_extract.py – 05_pilot_A2.py | early pilots on the ~/dmn PETH bundles (515 insertions) |
| 10_extract_full.py | full BWM (699 insertions): 50 ms counts over the whole recording, exact task-window counts, wheel / licks / motion energy / pupil, passive grey-screen interval |
| 20_full_A.py | A: rest preference and task suppression vs still ITIs (and grey screen) |
| 21_full_B.py | B: region-region co-fluctuation (epoch × nuisance variant × bin size; `still` option) |
| 22_compare_full.py – 26_A_subsets.py | DMN pair tests, cross-validated network, within-session tests, ITI activity vs behaviour, frontal / posterior / lateral splits |
| 40_spont_characterize.py | grey-screen period per session: timing, duration, wheel / licks / motion energy / pupil, stillness |
| 41_spont_network.py | grey screen: region-pair coverage, DMN coupling (ours and Lu et al.), robustness, power, CV network, grey-vs-ITI consistency |
| 42_spont_A.py | rest preference vs the grey screen, with drift controls (`G_full`; `G_me me_only`) |
| 44_windows.py | session-averaged rates per neuron in stimulus, movement + feedback, quiescence, inter-trial rest/moving and grey still/moving windows; trial-wise coupling in quiescence and inter-trial rest |
| 45_window_stats.py | DMN-motivated contrasts (within region, core vs other cortex / all regions, within session), suppressed fractions, trial-wise coupling classes |
| 43_spont_single_region.py | single-region resting metrics on the grey screen: fALFF, fluctuation amplitude, local synchrony (ReHo analogue) |
| 50_arousal_model.py, 51_arousal_stats.py | Arousal+/Arousal− model of Landemard et al. 2026: whisking-defined classes, HRF-predicted blood volume, task vs grey contrasts, pair coupling (`51_arousal_stats.py meso` for the mesoscope) |
| common.py, dmnstats.py | region sets, loaders, region-level statistics |

### meso/
| script | content |
|---|---|
| 10_extract_meso.py | canonical mesoscope sessions: per-trial window means and 1 s binned traces |
| 30_meso_C.py, 31_meso_stats.py, 32_meso_checks.py | rest preference, ITI region coupling, seed maps, robustness checks (use ITI start 1 s: `30_meso_C.py motion 1.0`) |
| 33_meso_spont.py | grey-screen coupling (`200 require_me`: periods ≥200 s with motion energy) |
| 52_meso_arousal.py | Arousal+/Arousal− model on the mesoscope (classes from the task period, 1 s bins) |

### report/
- `fig_dmn.py`: figure for the ITI report.
- `fig_grey.py`: figures for the grey-screen report (PDF with LaTeX caption via `tectonic`, plus PNG).
- `plotstyle.py`: shared figure style.

### Grey-screen report, in order
```bash
cd bwm
python 40_spont_characterize.py
python 42_spont_A.py G_full && python 42_spont_A.py G_me me_only
python 21_full_B.py spont motion 1 still && python 21_full_B.py spont motion 0.5 still
python 21_full_B.py iti motion 0.5 still            # for the grey-vs-ITI comparison
python 41_spont_network.py spont_motion_0.5s_still && python 41_spont_network.py spont_motion_1s_still
python 43_spont_single_region.py
python 44_windows.py && python 45_window_stats.py
cd ../meso && python 33_meso_spont.py 200 && python 33_meso_spont.py 200 require_me
cd ../report && python fig_regions.py && python fig_grey.py && python fig_windows.py
```

## Contributing
Work on a branch and open a pull request. Please note new results in the relevant report or add a short markdown note in `report/`.
