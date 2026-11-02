"""Generate the v2 sensitivity figure and report section from pipeline JSON outputs."""
import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
OUT = HERE.parent/'output'
R = json.loads((OUT/'22_robustness.json').read_text())
L = json.loads((OUT/'23_local_contrast.json').read_text())
S = json.loads((OUT/'20_stats.json').read_text())
P = json.loads((OUT/'21_population_stats.json').read_text())


def interval(d, digits=3):
    return f"{d['mean']:.{digits}f} [{d['ci95'][0]:.{digits}f}, {d['ci95'][1]:.{digits}f}]"


fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True)
items = [('Whole grey screen', R['subsets']['all']['lrate']),
         ('Within 10 s windows', L['10']['all']), ('Within 30 s windows', L['30']['all']),
         ('Pupil + motion coverage', R['subsets']['pupil_and_motion_available']['lrate']),
         ('Units ≥2 Hz', R['subsets']['rate_ge_2Hz']['lrate'])]
for i, (label, d) in enumerate(items):
    axes[0].errorbar(d['mean'], i, xerr=[[d['mean']-d['ci95'][0]], [d['ci95'][1]-d['mean']]], fmt='o', color='#13865d', capsize=3)
axes[0].set_yticks(range(len(items)), [x[0] for x in items]); axes[0].invert_yaxis()
axes[0].set_xlabel('Mean log₂ still / moving rate\n(mouse bootstrap 95% CI)')
axes[0].axvline(0, color='grey', linestyle=':')
f = R['fdr']
labels = ['Uncorrected p<.01', 'BH q<.01', 'BH q<.05']
for shift, metric, color in [(-.18, 'positive', '#13865d'), (.18, 'negative', '#e98044')]:
    vals = [S['global_fraction_neurons']['SP' if metric=='positive' else 'MP']]+[f[f'epoch_{metric}_q{a}']['pooled_fraction'] for a in [.01,.05]]
    axes[1].bar(np.arange(3)+shift, 100*np.array(vals), .34, color=color, label='Still-preferring' if metric=='positive' else 'Move-preferring')
axes[1].set_xticks(range(3), labels, rotation=20); axes[1].set_ylabel('Percent of tested neurons'); axes[1].legend(frameon=False, fontsize=8)
for i, (key, label) in enumerate(zip(R['session_correlations'], ['Grey halves', 'Grey / ITI', 'Face / wheel'])):
    d = R['session_correlations'][key]
    axes[2].errorbar(i, d['mean'], yerr=[[d['mean']-d['ci95'][0]], [d['ci95'][1]-d['mean']]], fmt='o', color='#13865d', capsize=3)
axes[2].set_xticks(range(3), ['Grey halves', 'Grey / ITI', 'Face / wheel'], rotation=20)
axes[2].set_ylabel('Within-session Spearman ρ\n(mean over mice, 95% CI)'); axes[2].set_ylim(0,1)
for ax, letter in zip(axes,'abc'):
    ax.spines[['top','right']].set_visible(False)
    ax.set_title(letter, loc='left', fontweight='bold')
fig.savefig(HERE/'fig7_robustness.png', dpi=180)
fig.savefig(HERE/'fig7_robustness.pdf')
plt.close(fig)
rows = ''.join(f'<tr><td>{label}</td><td class="num">{interval(d)}</td><td>{d["n_mice"]}</td></tr>' for label,d in items)
corr = '; '.join(f'{label}: {interval(R["session_correlations"][key])}' for key,label in zip(R['session_correlations'], ['grey halves', 'grey versus ITI', 'face versus wheel']))
sp = f['epoch_positive_q0.01']['pooled_fraction']*100
mp = f['epoch_negative_q0.01']['pooled_fraction']*100
section = f'''<!-- robustness:start -->
<section id="robustness" aria-labelledby="h-robustness">
<h2 id="h-robustness">Sensitivity analyses added in v2</h2>
<p><strong>Lower firing survives local time matching.</strong> Each 10&nbsp;s or 30&nbsp;s window must contain ≥0.5&nbsp;s in each state, with ≥5 qualifying windows per session. Rates are averaged equally over the same windows in both states, then neuron log ratios are averaged per mouse. The 10&nbsp;s analysis retains {L['10']['n_sessions']} sessions and {L['10']['all']['n_mice']} mice. It reduces the magnitude of the contrast but retains a negative estimate in every sampled division. This limits an explanation based only on slow drift, while changing which periods contribute; it is not a causal estimate.</p>
<div class="tablebox"><table><thead><tr><th>Rate sensitivity check</th><th>Mean [95% CI]</th><th>Mice</th></tr></thead><tbody>{rows}</tbody></table></div>
<p><strong>Multiplicity.</strong> Across all finite neuron epoch-test p-values, Benjamini–Hochberg q&lt;0.01 retains {sp:.1f}% still-preferring and {mp:.1f}% movement-preferring neurons. These are pooled fractions of tested neurons, not equal-mouse averages. Both q&lt;0.01 and q&lt;0.05, and a separate correction for within-bout tests, are saved in <code>22_robustness.json</code>. Original Figures 2 and 4 retain their explicitly labelled uncorrected p&lt;0.01 classifications for comparison. FDR correction does not repair miscalibrated p-values from temporal dependence.</p>
<p><strong>Within-session replication.</strong> Correlations computed separately in sessions with ≥20 complete neurons, then averaged within mouse and over mice: {corr}. CIs resample mice (4,000 draws). The face/wheel comparison still shares its stillness baseline; neither these estimates nor the pooled correlations establish a fixed tuning property.</p>
<p><strong>Paired DMN comparison.</strong> Within each mouse, region means receive equal weight within the core and other cortex; only mice with both sets contribute. Core-minus-other log₂ rate contrast: {interval(R['dmn_paired']['lrate'])}, {R['dmn_paired']['lrate']['n_mice']} mice. This targets a different estimand from the region permutation test; incomplete anatomical sampling remains a limitation.</p>
<p><strong>Population validation.</strong> Both states now use identical nonconstant neuron sets separately for raw and thinned comparisons. Residual matched log₂ rate contrast is {interval(R['population_matched_rate_residual'], 5)}. Thinning is stochastic and matches expected rates, not exact realised counts. It does not remove shared slow fluctuations or establish excess synchrony against an independent-spike null.</p>
<p><strong>Timing corrections.</strong> The 2–4&nbsp;s contrast previously indexed 4–8&nbsp;s; this is corrected. Movement-offset bouts must start two 50&nbsp;ms bins after an actual movement (allowing the margin), rather than at pupil-gate crossings. Aligned windows must lie entirely inside the analysed grey block. Figures and bout tests have been recomputed.</p>
<figure><img src="fig7_robustness.png" alt="Rate sensitivity checks, FDR-adjusted neuron fractions, and within-session replication with mouse bootstrap confidence intervals."><figcaption class="note">Figure 7. Sensitivity analyses. Rate effects and session correlations use equal mouse weighting; neuron classifications show pooled fractions. Face and wheel indices share their stillness baseline.</figcaption></figure>
</section>
<!-- robustness:end -->'''
p = HERE/'stillness.html'
text = p.read_text()
if '<!-- robustness:start -->' in text:
    a=text.index('<!-- robustness:start -->'); b=text.index('<!-- robustness:end -->')+len('<!-- robustness:end -->')
    text=text[:a]+section+text[b:]
else:
    text=text.replace('  <section id="summary"', section+'\n\n  <section id="summary"')
p.write_text(text)
print('Updated sensitivity section and Figure 7')
