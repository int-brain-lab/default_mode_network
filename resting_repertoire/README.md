# Resting repertoire pilot (2026-11-04)

Test of the central prediction of Deco, Jirsa & McIntosh 2013 (*Trends Neurosci* 36:268, "Resting brains never
rest"): resting activity sits in a low-firing state from which noise drives excursions towards the configurations
tasks recruit, so resting fluctuations should be concentrated in the task-evoked dimensions.

Data: IBL mesoscope, the 48 sessions / 9 mice of the comprehensive report section 12 (cached in
`../meso/output/sessions/`). Rest = still 1 s bins of the passive block (grey screen + receptive-field mapping), and
of the grey screen alone (34 sessions).

| Script | What |
|---|---|
| `10_meso_repertoire.py` | evoked subspace (top 5 dims of 16 trial-averaged patterns), alignment index (Elsayed et al. 2016) with ROI-permutation and covariance-matched nulls; with/without the uniform direction; per-window subspaces; moving bins |
| `11_meso_repertoire_denoised.py` | cross-validated evoked dims (shared by odd/even trials) and single-trial task PCs |
| `fig_pilot.py` | `fig_repertoire_pilot.pdf` (+ `.tex` caption, `.png`) |

## Result (uniform direction removed; means over 9 mice; still bins of the passive block)

| Task subspace | Alignment | ROI-permuted null | Covariance-matched null |
|---|---|---|---|
| Evoked (trial means) | 0.034 | 0.012 | 0.085 |
| Evoked, cross-validated | 0.033 | 0.012 | 0.085 |
| Single-trial task PCs | 0.079 (median over mice 0.048) | 0.012 | 0.085 |

- Rest is aligned with the task-evoked patterns above chance (46/48 sessions above the permutation null's 95th
  percentile, 9/9 mice), but far less than with arbitrary high-variance dims of the recording (0/48 sessions above
  the covariance-matched null). Grey-screen-only still bins give the same picture.
- Rest shares the dominant dims of single-trial task variability about as much as generic high-variance dims (mean
  pulled up by one session of SP037).
- Reading: resting fluctuations explore the same space as ongoing variability during the task, not preferentially
  the task-evoked configurations. This does not support the strong form of the prediction.
- Each window alone (stim, choice, feedback, post-feedback; 2 dims) is ~2.5-3x its permutation null, with no window
  standing out. Moving bins align slightly more than still bins.

## Caveats

- The evoked subspace is noisy: odd/even trial subspaces share only 22% (chance 0.07%); the cross-validated version
  did not change the result.
- Deconvolved calcium in 1 s bins at rest vs 0.4-0.7 s windows in the task; ~7,600 ROIs vs ~220 still bins.
- The passive block contains sparse-noise stimuli (grey screen alone agrees, 60 s in most sessions).
- The covariance-matched null draws from all recording bins, dominated by the task period and slow drift.

## Possible next steps

- BWM Neuropixels (spikes, 50 ms bins, 10 min grey screen): same test per probe.
- Ask the reverse question: are task-evoked patterns found in the dims of largest rest variance (rest PCs -> evoked)?
- Time-resolved: do excursions (high-variance rest moments) move towards evoked patterns?
