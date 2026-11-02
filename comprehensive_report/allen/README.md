# Allen Institute Neuropixels analyses (input to Part IV, sections 13-14, of the comprehensive report)

Exported from an external analysis thread ("Model Base Inquiry", thread
analysis-20261028-125500-a7e1e470a8808b66) in two deliveries:
1. 2026-10-30, `open_culture_allen_expansion.zip`: the stillness replication (Visual Behavior + Visual Coding).
   Duplicates were removed on 2026-10-31; nothing unique was dropped.
2. 2026-11-02, `follow_up.zip`: the same files again (byte-identical) plus the follow-up analyses below.
   Only the new files were added here.

- `reports/`: the Allen analysis report (PDF, full methods)
- `data/`: per-neuron (`data_neurons.csv`, all 84,661 units), per-session, per-mouse, per-division,
  pre-movement and screening tables; `scripts.md` holds the Allen pipeline, the statistics script and the
  plotting script; `data/README.md` describes the columns
- `figures/stillness_allen_summary.pdf`: the Allen six-panel figure; the PNG/SVG versions and per-panel SVGs were
  deleted and can be regenerated from the CSVs with `plot_stillness_figures.py` in `scripts.md` (part 2).
  `stillness_pilot_allen_599294.png` and `stillness_allen_4mice.png` are pilot figures with no table behind them.
- `data/fu_*.csv`, `data/followup_*.md`, `figures/followup_summary.pdf`: the follow-up analyses (circular-shift
  null for the neuron labels and onset rules, Visual Behavior; frontal core - RSP coupling, OpenScope DANDI 000563,
  000690, 001637). `followup_report_section.md` is its write-up, `followup_scripts.md` holds its statistics,
  plotting and analysis code; the per-panel SVG/PNG versions of the summary figure were not copied (regenerate with
  `plot_followup.py` from `followup_scripts.md`)
- `provenance/`: the README, manifest and conversation transcript of the second export (a superset of the first:
  the transcript and manifest of the first export are contained in them)
- removed: `data/stillness_allen_per_mouse.csv` (identical to `data/data_mice.csv`, columns reordered)

Section 13 / Figure 18 are rebuilt from these tables plus the IBL outputs with
`neural_correlates_stillness/30_allen_matched.py` then `neural_correlates_stillness/report/fig_allen.py`;
sections 13.4 and 14 / Figure 19 with `neural_correlates_stillness/32_null_onsets.py` (IBL counterparts) then
`neural_correlates_stillness/report/fig_allen_followup.py`.
