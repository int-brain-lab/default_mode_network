# Follow-up analyses for the stillness report (Allen Institute Neuropixels data)

Exploratory analyses (not pre-registered). Figure: `followup_summary.svg` (and `.pdf`, `.png`); each panel also as `fu_panel_A.svg` ... `fu_panel_F.svg`.

## 1. Calibrating the still- and movement-preferring labels

**Question.** The per-neuron label (Welch t-test on 0.5 s still vs moving epochs, p < 0.01) assumes independent epochs. Spike counts are autocorrelated, so the nominal 1% false-positive rate per direction may not hold.

**Method.** For each Allen Visual Behavior session (44 sessions, 30 mice; same grey-screen window, state labels, epochs and >= 0.5 Hz neuron filter as the main analysis), all neurons' 50 ms count rows were rotated together by a random offset (at least 10 s from either end) relative to the fixed behavioural labels, and the labels recomputed; 200 rotations per session (seed 20261030). Rotation preserves each neuron's autocorrelation and the population co-fluctuations and breaks only the alignment to behaviour. Null-corrected fraction = observed minus null mean. Mouse values average sessions within mouse; CIs are 95% bootstrap over mice (10,000 resamples).

**Results.**
- In all 44 sessions both observed fractions exceeded the null's 95th percentile (permutation p <= 0.045 for still-preferring, <= 0.005 for movement-preferring).
- Per mouse: still-preferring 11.8% observed vs 0.79% null, corrected **11.0% [9.6, 12.6]**; movement-preferring 23.6% vs 2.83%, corrected **20.8% [18.8, 22.8]**.
- The false-positive rate depends on sampling: the still-preferring null rises as moving epochs get fewer (Spearman rho = -0.74), the movement-preferring null as still epochs get fewer (rho = -0.84); it reaches 5-10% in sessions with fewer than about 20 epochs of one state. In the 10 sessions with at least 50 epochs of both states it is 0.88% and 1.05%, close to nominal; there the corrected fractions are 14.7% still- and 26.4% movement-preferring.

**Implication.** The still-preferring population is real, but raw fractions from thin sessions are inflated by a few percentage points, more so for the state with more epochs. The same check is advisable for the IBL fractions (15% / 40%).

## 2. Pre-movement rise under a stricter onset rule

**Question.** With onsets requiring only >= 1 s of prior stillness, the -2 to -1 s baseline can contain the previous movement. In IBL, any-movement onsets under that rule show no rise, while a >= 3 s rule shows one.

**Method.** Same sessions. Onsets = first moving 50 ms bin preceded by >= 3 s (strict) or >= 1 s (loose) of awake stillness. Population rate (mean over kept neurons) from -2.0 to +1.0 s, as fractional change from the -2 to -1 s mean; rise = mean change over -0.5 to -0.05 s. Sessions with < 3 onsets excluded; sessions averaged within mouse.

**Results.**
- Strict rule: rise **12.3% [9.5, 14.9]**, positive in **12/12 mice** (Wilcoxon p = 0.0005; 17 sessions).
- Loose rule: rise 5.0% [3.4, 6.8], positive in 24/27 mice (p = 2e-06; 38 sessions).
- Paired, in the 17 sessions measurable under both rules, the strict rule gives the larger rise in 15 (mean difference +5.0 percentage points, p = 0.0001).
- Under the strict rule a small change is already present at -1.0 to -0.5 s (2.6%, p = 0.03); under the loose rule there is none (-1.0%, p = 0.06).

**Implication.** The Allen pre-movement rise is not an artefact of the loose rule: it is larger after longer stillness, matching IBL. Only 12 of 30 mice have enough >= 3 s onsets, and running onsets (2 cm/s threshold) can lag the true start of movement, so part of the rise may still be early, sub-threshold movement.

## 3. Frontal core - retrosplenial coupling

**Question.** The BWM never records frontal DMN core (ACAd, ACAv, PL, ORBm) and RSP together. Allen OpenScope sessions do, on separate probes.

**Method.** 11 sessions (10 mice; DANDI 000563, 000690, 001637). Region population rates in 1 s bins (regions without layer suffix, >= 10 units at >= 0.5 Hz), minus a 60 s running mean; Fisher-z correlation for every cross-probe region pair (614 pairs), in all, still and moving bins (running > 2 cm/s; still = no movement within +-1 s). Window: first to last stimulus onset, excluding optotagging. **No stimulus-free block exists in these sessions; coupling is measured during passive viewing.** Distance correction: per session, residual from a linear fit of z on distance between Allen CCF region centres (25 um atlas). Flicker control in Temporal Barcode sessions: removal of the repeat-locked component of the 8 s repeated flicker.

**Results.** 7 mice (8 sessions) had at least one frontal-RSP pair; 3 sessions had no frontal or no RSP region passing the 10-unit cut.
- Mean frontal-RSP percentile among cross-probe pairs: all bins 59.7 (5/7 mice above the 50th, 0 above the 90th); still 61.7; moving 63.4.
- Distance-corrected: 58.4 / 60.2 / 62.4. Coupling barely depends on distance here (slope 0.007 z per mm, rho = 0.03), so the correction changes little. Residual z above zero in 5/7 mice (all bins, p = 0.22).
- Still minus moving residual z: 0.008 (larger in 5/7 mice, p = 0.69): no stillness-specific strengthening.
- Flicker control: frontal-RSP mean z 0.322 -> 0.321 and 0.076 -> 0.076 in the two testable mice.

**Implication.** Measured directly with single neurons, frontal-retrosplenial coupling during passive viewing is only slightly above that of other cross-probe pairs and does not strengthen in stillness. This does not test resting-state coupling, because no stimulus-free period exists; one session (DANDI 000690) has a spontaneous-presentation table and could support a true rest test in one mouse.

## LaTeX figure block

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\textwidth]{followup_summary.pdf}
  \caption{\textbf{Follow-up analyses in Allen Neuropixels data.}
  \textbf{A}, still- and movement-preferring fractions per mouse against their circular-shift null (200 rotations of all neurons' 50~ms counts relative to behaviour; 30 mice, 44 Visual Behavior sessions). Null-corrected: 11.0\% and 20.8\%.
  \textbf{B}, null false-positive rate per session against the number of epochs in the opposite state; dashed line, nominal 1\%.
  \textbf{C}, population rate around running onsets preceded by $\geq$3~s (green) or $\geq$1~s (violet) of awake stillness, \% change from the $-2$ to $-1$~s mean; mean over mice and 95\% bootstrap CI; shaded, the $-0.5$ to $-0.05$~s window.
  \textbf{D}, rise under the two rules in the 17 sessions measurable under both (15 above the diagonal; Wilcoxon $p=0.0001$).
  \textbf{E}, percentile of each mouse's frontal core--RSP pairs among all cross-probe region pairs (Allen OpenScope, 7 mice, passive viewing), raw and distance-corrected; black, mean and interquartile range; dashed, 90th percentile.
  \textbf{F}, coupling (Fisher $z$ of 1~s regional rates, 60~s detrended) against distance between Allen CCF region centres for all cross-probe pairs; line, linear fit.}
  \label{fig:stillness_followup}
\end{figure}
```

## Caveats
- Exploratory; thresholds were fixed before the runs. Visual Behavior pupil area is flagged by the platform as unconfirmed in meaning (used only for the drowsiness gate).
- Coupling: passive viewing only, 7 mice, 2-3 probes per session loaded; hippocampal fields merge to "CA" after layer stripping (does not affect frontal or RSP regions). The 000690 window was taken from the running record (which matched stimulus onsets within 0.4 s in every other session).
- Pre-movement: strict-rule onsets are few (3-18 per included session).
