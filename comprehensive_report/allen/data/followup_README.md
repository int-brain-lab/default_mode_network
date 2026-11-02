# Follow-up analyses: download contents

| File | What it is |
|---|---|
| followup_report_section.md | Report section: question, method, results, implication for each analysis; LaTeX figure block; caveats |
| followup_summary.svg / .pdf / .png | Six-panel figure (A-B null calibration, C-D onset rules, E-F frontal-RSP coupling) |
| fu_panel_A.svg ... fu_panel_F.svg | Each panel as its own SVG |
| followup_scripts.md | stats_followup.py, plot_followup.py and the analysis code (copy each block into a .py file) |
| fu_null_sessions.csv | Per session (44): neurons, still/moving epochs, observed still-/movement-preferring fraction, null mean, null 95th pct, permutation p, null-corrected fraction, onset counts |
| fu_null_mice.csv | Per mouse (30): session-averaged observed, null and corrected fractions (panel A) |
| fu_premove_sessions.csv | Per session and onset rule: n onsets, rise (-0.5..-0.05 s vs -2..-1 s), early change (-1..-0.5 s); fractions, not % |
| fu_premove_curves_long.csv | Pre-movement curves, long format: subject, date, rule, t_s, frac_change (panel C) |
| fu_coupling_pairs.csv | Every cross-probe region pair: session, subject, regions, kind (FRONT-RSP / core-other / other-other), r, Fisher z, n bins, percentile, mask (all/still/moving), centre distance (um) (panel F) |
| fu_coupling_front_rsp.csv | Frontal-RSP pairs: z, distance, distance-corrected residual and percentiles (panel E) |

Fractions are 0-1 in the tables and % in the text and figure. Datasets: Allen Visual Behavior Neuropixels (DANDI 000713) for analyses 1-2; Allen OpenScope (DANDI 000563, 000690, 001637) for analysis 3. Exploratory, not pre-registered.
Files from the earlier package (main report, data_*.csv, stillness_allen_summary.*) are in the same download.
