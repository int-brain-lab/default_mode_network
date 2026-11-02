# Review, 27 October 2026

The principal rate result survives a new comparison using still and moving bins within shared short windows. The original analysis already used mice for uncertainty, within-session state contrasts, and useful replication across time and context. The revision strengthens those checks and corrects implementation/reporting errors.

## Implemented

1. **Wrong time bin:** the reported 2–4 s contrast used the 4–8 s column. Corrected and recomputed statistics and Figure 4.
2. **False offsets:** pupil-gate crossings and grey-block boundaries could masquerade as movement offsets. Require a true preceding movement and restrict the full event-aligned window to grey. Recomputed aligned rates and true-offset bout tests.
3. **Different population units:** independent removal of constant neurons after thinning could change the ensemble between states. Require their intersection in both states and validate equal `n_used`.
4. **Region weighting:** pooled-neuron region points and equal-mouse bootstrap intervals targeted different quantities. Region points and region-contrast bootstrap intervals now average within-mouse region means; missing metrics are excluded before aggregation. Region-map split-half reliability still uses neuron-weighted maps, explicitly distinguished in the report.
5. **Uncorrected classifications:** retain the original labelled p<0.01 results for comparison, and add global BH q<0.01 / q<0.05 separately for the epoch and bout test families. Preserve missing-test denominators and save neuron q-values.
6. **Drift sensitivity:** compare equally weighted windows with at least 0.5 s of both states, requiring five windows. Analyse both 10 s and 30 s windows. This controls coarse differences in temporal sampling without claiming causal identification.
7. **Pooling and coverage:** estimate correlations within sessions before averaging within mouse; repeat rate summaries with pupil plus motion coverage and with ≥1 / ≥2 Hz units. Add paired within-mouse DMN-minus-other contrasts with equal weights over observed region means.
8. **Interpretation:** qualify wakefulness, fixed neuron properties, shared stillness baselines, null comparisons and state-defined population bursts; correct population event bin duration to 100 ms. Link the cited quiet-wakefulness study to its primary publication.

## Validation and reproducibility

Run the README pipeline in order. All extraction-derived neuron outputs, population outputs, summaries and report figures were regenerated for the revision. `test_robustness.py` covers known BH outputs and missing values, rejection of gate crossings as offsets, real movement alignments, pupil/movement-margin masking, and invariance of mouse-weighted region contrasts to duplication of neurons within a mouse. Figure 7 and its HTML section are generated from JSON outputs by `report/robustness.py`.

## Remaining limits

The epoch and bout p-values have not been replaced by a temporally blocked null, so FDR results remain sensitivity analyses of these tests. Two-second windows do not guarantee independent observations. Shared stillness denominators can inflate face/wheel agreement; the session-wise analysis does not remove this algebraic coupling. State-conditioned correlations can include drift and state mixtures, and one deterministic thinning realization does not assess Monte Carlo variability. The local-window check changes the selected periods and does not match pupil, individual movement features, or fine-scale history. Region rankings and division comparisons are exploratory; regional multiplicity is not controlled. Missing anatomical coverage limits paired DMN comparisons. Sleep and sharp-wave ripples require independent physiological validation. Pre-movement timing is descriptive and threshold-dependent, with no formal onset-latency test.
