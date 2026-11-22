# Verified Final Results

## Experimental integrity

- Dataset: NLTK Brown Corpus, split at the document level.
- Split: 400 training documents and 100 held-out testing documents.
- Split seed: 42.
- Within-word bigrams: 2,981,560 training and 740,924 testing.
- Independent matched seeds: 30, numbered 100 through 129.
- Budget: exactly 10,000 objective evaluations per method and run.
- Total stochastic runs: 150, with no missing values or invalid layouts.
- QWERTY training cost: 3.249138.
- QWERTY testing cost: 3.251401.

Costs are normalized corpus-weighted Euclidean movement distances. Lower is
better. Training and testing matrices were normalized independently.

## Held-out Brown Corpus performance

| Method | Mean train cost | Mean test cost ± SD | Mean test improvement vs. QWERTY | Median runtime (s) |
| --- | ---: | ---: | ---: | ---: |
| Random Search | 2.4044 | 2.4030 ± 0.0393 | 26.09% | 0.383 |
| Standard GA | 1.9866 | 1.9836 ± 0.0217 | 38.99% | 0.556 |
| Adaptive GA | 1.9871 | 1.9840 ± 0.0140 | 38.98% | 0.660 |
| Standard SA | 1.9467 | 1.9430 ± 0.0158 | 40.24% | 0.385 |
| Adaptive SA | **1.9383** | **1.9352 ± 0.0163** | **40.48%** | **0.383** |

## Critical adaptive comparison

Adaptive GA did not improve mean held-out quality relative to Standard GA.
Their mean test costs were 1.983985 and 1.983623. The paired Wilcoxon result was
not significant (p = 0.983834), and Cliff's delta was negligible
(-0.008889). Adaptive GA won 15 matched seeds and lost 15. Its useful observed
effect was lower test-cost variability (SD 0.013958 versus 0.021726), obtained
with approximately 13.8% more mean runtime.

Adaptive SA achieved a lower mean test cost than Standard SA: 1.935234 versus
1.943039. The absolute difference was 0.007805, approximately 0.40% relative to
Standard SA. However, the paired result was not statistically significant at
0.05 (p = 0.134735), and Cliff's delta indicated only a small effect
(-0.281111). Adaptive SA won 16 matched seeds and lost 14. The defensible claim
is therefore a modest but statistically inconclusive benefit—not significant
outperformance.

## Best training-selected layout

- Method: Adaptive SA.
- Seed: 103.
- Layout string: `zgdnalmyxjkwieropbqvchtsuf`.
- Keyboard rows:
  - Top: `Z G D N A L M Y X J`
  - Home: `K W I E R O P B Q`
  - Bottom: `V C H T S U F`
- Training cost: 1.899209.
- Held-out test cost: 1.896079.
- Held-out reduction relative to QWERTY: 41.6842%.

This layout was selected using training cost only. Its test result was computed
after selection, so the procedure does not use test-set leakage.

## Qualitative interpretation

Several frequent Brown Corpus bigrams became substantially closer under the
best layout. Modeled distances decreased from 3.400 to 1.118 for `he`, from
5.590 to 1.000 for `an`, from 4.250 to 1.031 for `en`, from 3.640 to 1.000 for
`nd`, and from 5.000 to 1.000 for `or`. The `ed` distance increased from 1.031
to 1.601, showing that the optimizer makes frequency-weighted trade-offs rather
than shortening every pair.

## Interpretation limits

These results demonstrate lower modeled movement cost for the selected corpus,
geometry, and objective. They do not demonstrate higher typing speed, improved
comfort, reduced fatigue, safety, or learnability. Those outcomes require a
more complete ergonomic model and longitudinal human testing.

