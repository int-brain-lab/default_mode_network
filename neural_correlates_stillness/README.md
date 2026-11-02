# Neural correlates of stillness

Which changes in neural activity accompany awake idleness (no movement, not asleep) on the passive grey screen of
the BWM, which hold across mice, and which are area-specific. This is a follow-up to the grey-screen DMN report
(`../report/grey/dmn_grey.html`). It uses that report's movement thresholds and single-cell techniques: interleaved
still-vs-moving contrasts, spaced-sample tests, mouse bootstrap and within-session comparisons.

**Report:** `report/stillness.html`, typeset as `report/stillness.pdf` by `report/html2tex.py` (figures `report/fig*.pdf` with LaTeX captions, PNGs for the page).

## States (`common_still.py`)
- **moving:** wheel > 0.05 rad/s, a lick, or motion energy above the session's 60th percentile.
- **awake-still:** not moving (±1 bin margin), at most 30 s since the last movement, and pupil at or above the
  10th percentile of the task-period pupil. The pupil part is the anti-sleep gate.

## Pipeline
Runs on the existing extraction in `../bwm/output/full` (`../bwm/10_extract_full.py`); activate `iblenv`.
```bash
python 10_neurons.py 8          # per-neuron rates, tests, time profiles, pupil correlation, aligned rates (~6 min)
python 11_population.py         # within-region correlation, dimensionality, synchronous events; raw and rate-matched
PYTHONPATH=../bwm python 20_stats.py       # mouse-level robustness, region maps, reliability, variance partition
python 21_population_stats.py
python 22_robustness.py         # FDR, coverage/rate thresholds, within-session replication, paired DMN
python 23_local_contrast.py    # shared 10 s / 30 s windows to limit slow-drift confounding
python -m unittest test_robustness.py
cd report && PYTHONPATH=../../bwm python figs.py   # Figures 1-6; `figs.py main` writes Figures 11-16 of ../report/grey
python html2tex.py                                  # stillness.tex + stillness.pdf from stillness.html
python robustness.py           # Figure 7 and data-generated sensitivity section
```
Outputs go to `output/`; only the small JSON/CSV files are tracked by git.

## Short version (v2, 27 Oct 2026)
- **Robust across mice:** firing is lower in awake stillness (129/130 mice, every division). 15% of neurons are
  still-preferring (126/128 mice), and this preference is reproducible within recordings: ρ = 0.78 between halves, 0.55
  between grey screen and task ITIs, 0.74 between face and wheel movements. Rate-matched within-region synchrony is
  higher in stillness (82% of mice, cortex 91%), and firing rises 0.3–0.5 s before movement onset.
- **Area-specific:** thalamus is the most movement-driven, has lower firing at 2–4 s than at 0.5–1 s after movement, and is the only
  division whose firing follows pupil within stillness. The hippocampal formation dips after movement offset, then
  rises within bouts and shows the most synchronous events. Still-preferring neurons are enriched in cerebellar
  nuclei, lateral septum and BST. The cerebellum shows the opposite correlation change. These are state-conditioned spike-count correlations, not proof of excess coincidence above an independent-spike null.
- **DMN core:** no detected difference from other cortex in the original region test; this does not establish equivalence. The v2 report also provides a paired within-mouse comparison.

## Review and improvements (v2)

- Fixed the duration contrast: `tsm_3` is 2–4 s; `tsm_4` (previously used) is 4–8 s.
- Movement offsets must follow actual movement, excluding pupil-gate crossings. Aligned windows remain inside the grey block; within-bout tests use true-offset bouts.
- Population contrasts now use the same nonconstant neurons in both states, separately for raw and thinned counts.
- Region point estimates now average mouse means, matching their mouse-bootstrap intervals. Split-half reliability retains the original neuron-weighted maps and is labelled accordingly.
- Added separate global BH corrections for epoch and bout tests, coverage and rate-threshold sensitivity, within-session correlations summarised over mice, and a paired DMN comparison with equal region weighting within mouse.
- Added a local time-matched rate contrast: 10 s windows give a mean log₂ ratio of −0.372 (95% mouse CI −0.406 to −0.340), compared with −0.431 in the same retained neurons without matching. This retains 388 sessions and 130 mice.
- Clarified shared-baseline inflation, residual autocorrelation, putative wakefulness, null results versus equivalence, and the 100 ms population-event resolution.

The original p<0.01 neuron classifications remain explicitly labelled in Figures 2 and 4; Figure 7 provides the FDR sensitivity. The BH calculations assume valid input p-values and do not solve temporal autocorrelation. See `REVIEW.md` for scope and remaining limitations.
