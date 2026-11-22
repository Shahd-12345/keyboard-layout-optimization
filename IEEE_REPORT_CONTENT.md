# IEEE Report Content Plan

The final report is limited to six IEEE pages, including references. Its main
contribution is the adaptive implementation and controlled experiment, so the
Assignment 3 literature review must be condensed.

## Recommended page allocation

1. Page 1: Abstract, Index Terms, Introduction, contributions, and research questions.
2. Page 2: Condensed Related Work and Problem Formulation.
3. Page 3: Standard GA/SA and the two feedback-adaptive mechanisms.
4. Pages 4–5: Experimental protocol, principal table, figures, and critical analysis.
5. Page 6: Limitations, Conclusion, GitHub link, and References.

## Main quantitative table

Use `report_assets/results_table.tex`. It gives QWERTY and all five stochastic
methods, with mean training cost, mean held-out cost and standard deviation,
mean held-out improvement, and median runtime. State that every stochastic
method received 10,000 evaluations for each of 30 matched seeds.

## Figures that belong in the paper

1. `results/figures/best_layout_comparison.pdf`
   - Show QWERTY and the best training-selected Adaptive-SA layout.
   - State that selection used training performance only.
2. `results/figures/convergence_comparison.pdf`
   - Show mean best-so-far training cost and 95% confidence intervals.

If space permits, combine the convergence plot with the relevant panels from
`adaptation_diagnostics.pdf` as one two-panel figure. Otherwise, leave full
adaptation diagnostics in the notebook and GitHub repository.

Do not use the standalone geometry figure because the layout-comparison figure
already communicates the geometry. Put runtime in the table instead of using
the runtime boxplot.

## Statistical wording to use

Adaptive GA did not improve held-out solution quality relative to Standard GA.
Their mean test costs were 1.9840 and 1.9836, respectively. The paired Wilcoxon
comparison was not significant (p = 0.9838), and the effect was negligible
(Cliff's delta = -0.0089). Adaptive GA nevertheless reduced run-to-run test-cost
variability from an SD of 0.0217 to 0.0140.

Adaptive SA produced a lower mean held-out cost than Standard SA, 1.9352 versus
1.9430. The difference was approximately 0.40% relative to Standard SA, but it
was not statistically significant (p = 0.1347); Cliff's delta of -0.2811
indicated a small effect. Describe this as a modest, statistically inconclusive
benefit—not as significant outperformance.

## Qualitative wording to use

The best layout shortened several frequent English bigrams. Relative to
QWERTY, modeled distance decreased from 3.400 to 1.118 for `he`, from 5.590 to
1.000 for `an`, from 4.250 to 1.031 for `en`, and from 5.000 to 1.000 for `or`.
The `ed` distance increased from 1.031 to 1.601, illustrating that the objective
makes frequency-weighted trade-offs rather than shortening every transition.

## What must be omitted

- All smoke-test numbers.
- Raw 30-run tables and document-ID lists.
- All five figures as separate paper figures.
- Long code extracts or notebook screenshots.
- Claims of statistically significant adaptive superiority.
- Claims about speed, comfort, fatigue, injury, safety, or learnability.
- Direct numerical comparison with papers using different corpora/objectives.
- Selecting or presenting a layout because it performed best on held-out data.

The raw runs, document IDs, full diagnostics, source code, and supplementary
figures belong in the GitHub repository and executed notebook.

## Defensible conclusion

The experiment shows that all heuristic methods reduced held-out modeled
movement relative to QWERTY, while both SA variants achieved lower mean costs
than the GA variants. Adaptive SA obtained the lowest mean test cost, 1.9352,
representing a 40.48% reduction relative to QWERTY. The best training-selected
Adaptive-SA layout achieved a 41.68% held-out reduction.

Adaptation did not help both method families equally. Adaptive GA reduced
variability without improving mean quality, while Adaptive SA produced a small
but statistically inconclusive improvement over Standard SA. Feedback control
therefore changed the search behavior but did not automatically guarantee
better solutions; its effect depended on the algorithm and controller
calibration.

The findings are limited by an older English corpus and a simplified Euclidean
movement proxy. Future work should tune controllers using a separate validation
protocol, evaluate modern/domain-specific corpora, add finger assignment, hand
alternation and learning cost, and validate layouts with longitudinal users.

