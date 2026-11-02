# Stillness replication in Allen Neuropixels data: figure for the report

Exploratory replication (not pre-registered). Figure files: `stillness_allen_summary.pdf` (vector) and `stillness_allen_summary.png` (300 dpi), both in the Figures panel.

## LaTeX figure block

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\textwidth]{stillness_allen_summary.pdf}
  \caption{\textbf{Replication of the neural correlates of stillness in Allen Institute Neuropixels data.} Grey-screen (spontaneous) periods from Allen Visual Behavior (VBN; 30 mice, 44 sessions) and Visual Coding (VC; 4 mice, 4 sessions) Neuropixels recordings, analysed with the same pipeline as the IBL data (50~ms bins, one-bin margin around movement, stillness counted up to 30~s after the last movement, pupil area $\geq$ task 10th percentile for VBN only, 0.5~s epochs, Welch $p<0.01$, neurons $\geq$0.5~Hz). Movement is defined from running speed only ($>$2~cm\,s$^{-1}$): no face-motion video exists for these datasets, so `still' bins may include orofacial movement. Each mouse is weighted equally (sessions averaged within mouse); error bars are 95\% bootstrap confidence intervals over mice. \textbf{A}, mean log$_2$ firing-rate ratio (still/moving) per mouse; lower in stillness in all 34 mice (overall -0.13 [-0.16, -0.10]; Wilcoxon $p=1e-10$). Dashed line, IBL estimate ($-0.43$). \textbf{B}, percentage of still-preferring (11.9\%) and movement-preferring (24.1\%) neurons per mouse; movement-preferring neurons outnumber still-preferring ones in all 34 mice. Dashed lines, IBL values (15\%, 40\%). \textbf{C}, still-preferring fraction per session against the number of still epochs (Spearman $\rho=0.71$, $p=7e-08$, 44 sessions with epoch counts); sessions with $\geq$50 still epochs reach 15.4\%. \textbf{D}, mean log$_2$ ratio per mouse by Allen CCF (Cosmos) division (divisions with $\geq$20 neurons per session). \textbf{E}, population firing around still$\rightarrow$running onsets (preceded by $\geq$1~s of stillness), as percentage change from the $-2$ to $-1$~s baseline; thin lines, individual mice; shaded band, 95\% bootstrap CI (27 mice with $\geq$5 onsets). The rise begins about 0.4~s before onset (grey band); mean change $-0.5$ to $-0.05$~s: +6.6\%, positive in 25/27 mice. \textbf{F}, sessions screened and included (inclusion: $\geq$10 still and $\geq$10 moving epochs); most exclusions were mice that ran throughout the grey screen.}
  \label{fig:stillness_allen}
\end{figure}
```

## Key numbers (mouse-level, mean [95% bootstrap CI])

| Measure | Allen (34 mice) | IBL (manuscript) |
|---|---|---|
| Mean log2 rate ratio (still/moving) | -0.128 [-0.155, -0.101], 34/34 mice < 0 | -0.43, 129/130 mice |
| Still-preferring neurons | 11.9% [10.6, 13.4] | 15% |
| Movement-preferring neurons | 24.1% [22.5, 25.7] | 40% |
| Pre-movement rise (-0.5 to -0.05 s) | +6.6%, 25/27 mice | 0.3-0.5 s before onset |

By division (Visual Behavior): midbrain -0.47 [-0.60, -0.34], 26/28 mice below zero; thalamus -0.15 [-0.22, -0.09], 23/29 mice below zero; hippocampal formation -0.10 [-0.14, -0.06], 25/29 mice below zero; isocortex +0.04 [-0.01, +0.10], 12/30 mice below zero.
By division (Visual Coding, 4 mice): isocortex -0.18 [-0.23, -0.13], 4/4 mice below zero; thalamus -0.24 [-0.42, -0.06], 4/4 mice below zero.

## Caveats for the text
- Movement is running only (no face video, no post-task licks), so the contrast is diluted relative to IBL.
- Still periods are short (most sessions under 1 min of awake stillness); still-preferring detection depends on still time (panel C).
- Pre-movement curves for the first 4 mice were stored already normalised; the others were normalised here to the -2 to -1 s baseline.
- Bootstrap seed 20261030 (10,000 resamples per CI).
- Pupil area stream (VBN) is flagged by the platform as unconfirmed in meaning; it gates only the drowsiness filter.
