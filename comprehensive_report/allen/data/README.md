# Allen stillness replication: download contents

| File | What it is |
|---|---|
| stillness_allen_report.pdf | Full report with detailed methods |
| stillness_allen_summary.svg / .pdf / .png | Six-panel summary figure |
| panel_A.svg ... panel_F.svg | Each panel as its own SVG |
| stillness_allen_figure_latex.md | Ready-to-paste LaTeX figure block and caption |
| scripts.md | stillness_stats.py, plot_stillness_figures.py and the analysis pipeline |
| data_neurons.csv | Every unit (84,661): dataset, subject, session, probe recording id, region, Cosmos division, grey-screen rate (Hz), rate still / moving (Hz), log2 ratio, Welch p, preference (+1 still, -1 moving, 0 none), keep (rate >= 0.5 Hz) |
| data_sessions.csv | Per session (48): kept neurons, mean log2 ratio, fractions still-/movement-preferring, awake-still minutes, still/moving epochs, onsets |
| data_mice.csv | Per mouse (34): session-averaged values (panels A, B) |
| data_divisions_per_mouse.csv | Per mouse and Cosmos division, mean log2 ratio (panel D) |
| data_divisions_summary.csv | Division means, 95% bootstrap CIs, mice below zero, Wilcoxon p |
| data_premovement_curves_pct.csv | Population rate change (%) around running onset, one column per mouse, t in s (panel E) |
| data_screening.csv | Sessions screened and included per batch (panel F) |

Log2 ratio = log2((rate_still + 0.1) / (rate_moving + 0.1)); negative means lower firing in stillness.
Dataset codes: VBN = Allen Visual Behavior Neuropixels (DANDI 000713); VC = Allen Visual Coding Neuropixels.
Exploratory analysis, not pre-registered.
