# DMN signatures in IBL data: full-data results (2026-10-17)

Figure: `fig_dmn.pdf` (built by `fig_dmn.py`).
Data: BWM Neuropixels, 419 sessions, 137 mice, 54,934 neurons, of which 389
sessions have a passive spontaneous period. Mesoscope, 153 canonical sessions;
45 sessions from 11 mice have enough still ITI bins for coupling.
DMN core is ACAd/v, PL, ILA, ORBm/l/vl, RSPagl/d/v, VISa and VISam; DMN lateral is
TEa, ECT and PERI (`bwm/common.py`).

## 1. DMN cortex is deactivated by the task (BWM): supported
Each neuron is compared between still ITI rest and the task windows (stimulus,
movement, feedback). Tests compare regions with a permutation test, and CIs come
from a mouse bootstrap.

| comparison | DMN | other | p | mouse CI of diff |
|---|---|---|---|---|
| suppressed, core vs other cortex | 29.4% | 23.6% | 0.039 | [0.027, 0.109] |
| suppressed, core+lateral vs other cortex | 30.7% | 22.2% | 0.0007 | [0.052, 0.153] |
| suppressed, core+lateral vs higher-order cortex only | 30.7% | 22.0% | 0.0025 | [0.049, 0.156] |
| rest preference (RI, still ITI), core+lateral | −0.04 | −0.15 | 0.0003 | [0.082, 0.169] |
| rest preference (passive spontaneous), core+lateral | −0.02 | −0.16 | 0.0001 | [0.064, 0.300] |
| movement modulation, core+lateral | 0.04 | 0.11 | 0.006 | [−0.129, −0.045] |

- **Robustness:** the effect survives strict stillness (core: 26.5% vs 18.4%,
  p = 0.001) and a task definition without the feedback window (p = 0.03).
- **Stimulus window:** it is largely absent when the task means the stimulus
  window alone (core p = 0.20). The deactivation is tied to movement, choice and
  feedback, not to sensory input.
- **Most suppressed:** TEa, ILA, RSPv, ACAv, ACAd, PERI, PL. **Least
  suppressed:** VISl, FRP, primary sensory areas.
- **Within-territory splits (strict stillness, Fig. b):**
  - posterior DMN vs sensorimotor: p = 0.015
  - frontal DMN vs MOs/MOp/FRP/AI: p = 0.065, mouse CI [0.03, 0.14]
  - lateral: n.s. for suppression, but rest preference p = 0.024
  - The frontal-only result is weaker than the partial-data estimate (p = 0.011).
- **Reliability:** the regional ordering is moderately reliable across mouse
  halves (Spearman-Brown 0.62 for suppression, 0.70 for RI).

## 2. DMN regions couple at rest
### BWM: underpowered
- **Rest-preference network:** in still ITIs and the passive spontaneous period,
  rest-preferring regions defined in one half of the mice co-fluctuate with each
  other more than with task-positive regions in the other half. This holds in
  89% of splits, but the CI includes 0.
- **Within-session DMN pairs:** with 12–13 mice, there is no effect (Wilcoxon
  p > 0.4).
- **Probe geometry:** frontal and retrosplenial DMN are never recorded in the
  same session.

### Mesoscope: partially supported, posterior DMN only
- **Coupling by pair class:** coupling is region-averaged ΔF correlation in still
  1 s ITI bins, with nuisance regression and minus the distance-matched mean.
  Both differences have mouse-bootstrap CIs that exclude 0:
  - DMN–DMN > other–other: +0.16 z, CI [0.09, 0.23]
  - DMN–other > other–other: +0.11 z, CI [0.06, 0.16]
- **Selectivity (DMN–DMN > DMN–other):** +0.05, CI [0.017, 0.075], but it is
  fragile:
  - it loses significance with 0.25 mm distance bins (CI [−0.01, 0.06]);
  - it disappears when VISa or VISam is left out;
  - it rests mainly on VISa–VISam (29 pairs, 5 mice) and RSP pairs (2–4 mice).
- **Interpretation:** posterior DMN regions (RSP, VISa, VISam) are more coupled
  at rest than sensorimotor cortex, both with each other and with everything
  else. That is a "hub" pattern; a segregated DMN block is not shown.
- **Rest preference does not replicate in calcium:** almost every ROI is higher
  during the task (median RI −0.10). With a session-relative index,
  DMN vs other is n.s. (p = 0.28). The cross-modal regional ordering is
  weakly positive (ρ = 0.35, p = 0.10, 23 regions).
- **No frontal DMN:** the canonical mesoscope sessions contain no frontal DMN,
  so the frontal–posterior DMN link stays untested in both datasets.

## 3. Not supported
- **Behaviour:** ITI activity does not predict next-trial reaction time or
  errors in a DMN-specific way (387 sessions; all DMN-vs-other p > 0.25).
- **Arousal:** pupil coupling is not DMN-specific.
- **Global-signal regression:** uninformative with the few regions per session
  in BWM.

## Bottom line
The defensible claim is a **task-negative profile of DMN cortex**: more
task-suppressed neurons and a stronger preference for still rest. It is
anatomically specific (vs higher-order cortex too), present in roughly
frontal and posterior subdivisions, and tied to action/outcome rather than
the stimulus. The network claim, coordinated DMN fluctuations at rest, is
only partly supported. Mesoscope posterior DMN is more coupled than
sensorimotor cortex, but a segregated block cannot be distinguished from
general hub-like coupling, and frontal–posterior coupling is untestable here.

## Run notes
- **OOM restart:** the first overnight run was OOM-killed (4 extraction workers
  at ~10 GB each). It was rerun with 2 workers via `run_all.sh`.
- **Failed sessions:** 3 BWM sessions failed (a trials-table mismatch, a probe
  with no good units, and one other); 12 mesoscope sessions failed extraction.
- **Mesoscope ITI start:** the pipeline's ITI start of 2 s left almost no still
  1 s bins (5 sessions). Results here use 1 s (`30_meso_C.py motion 1.0`);
  checks are in `meso/32_meso_checks.py`.
- **Mesoscope labels:** most mesoscope region labels are pre-histology
  estimates.
