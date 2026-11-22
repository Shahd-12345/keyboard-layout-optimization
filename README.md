# Adaptive Heuristic Optimization of English Keyboard Layouts

Python implementation for the Heuristic Optimization term project.

## Project question

Under the same 26-key geometry, corpus, objective function, and evaluation
budget, which heuristic produces the lowest held-out average movement cost?

The implemented comparison includes:

1. QWERTY baseline
2. Random Search baseline
3. Standard Genetic Algorithm (GA)
4. Feedback-adaptive GA
5. Standard Simulated Annealing (SA)
6. Acceptance-ratio-adaptive SA

## Implemented phases

Phase 1 implements and tests the common benchmark:

- fixed 26-key staggered QWERTY geometry;
- Euclidean center-to-center key distances;
- normalized within-word letter bigram frequencies;
- average movement cost per letter transition;
- QWERTY evaluation;
- reproducible Random Search baseline;
- Brown Corpus train/test split support.

Phase 2 adds a permutation-safe Genetic Algorithm:

- order crossover (OX), which never duplicates or loses letters;
- swap mutation;
- tournament selection and elitism;
- an exact objective-evaluation budget;
- fixed-seed reproducibility and complete convergence history.

Phase 3 implements the rubric-aligned adaptive comparison:

- adaptive mutation responding to positional-entropy diversity and stagnation;
- adaptive SA reheating/cooling responding to measured acceptance ratio;
- standard-versus-adaptive ablations under identical budgets;
- 30 matched random seeds;
- held-out document-level testing;
- descriptive statistics, paired Wilcoxon tests, and Cliff's delta;
- convergence, adaptation, runtime, layout, and generalization figures;
- a 31-cell executable Jupyter Notebook.

The small built-in sample is only a smoke test. It must not be reported as a
final experiment.

## Verified Brown Corpus results

The preserved final experiment used 400 training documents, 100 held-out test
documents, 30 matched seeds, and exactly 10,000 objective evaluations per
stochastic method and run.

| Method | Mean held-out cost | SD | Mean reduction vs. QWERTY |
| --- | ---: | ---: | ---: |
| Random Search | 2.4030 | 0.0393 | 26.09% |
| Standard GA | 1.9836 | 0.0217 | 38.99% |
| Adaptive GA | 1.9840 | 0.0140 | 38.98% |
| Standard SA | 1.9430 | 0.0158 | 40.24% |
| Adaptive SA | **1.9352** | **0.0163** | **40.48%** |

Adaptive SA obtained the best mean result, but its paired advantage over
Standard SA was not statistically significant (Wilcoxon p = 0.1347; Cliff's
delta = -0.2811, small). Adaptive GA reduced variability but did not improve
mean held-out quality (p = 0.9838; negligible effect). See
`RESULTS_SUMMARY.md` for the critical interpretation and
`IEEE_REPORT_CONTENT.md` for the six-page paper plan.

## Objective

For a layout permutation pi, the objective is:

    C(pi) = sum_i sum_j f[i,j] * d[position(i), position(j)]

The frequency matrix is normalized so that the cost is the expected movement
distance per within-word letter transition. Lower is better.

## Installation

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
~~~

On Windows PowerShell, activate with:

~~~powershell
.venv\Scripts\Activate.ps1
~~~

## Run the tests

~~~bash
python -m unittest discover -s tests -v
~~~

The verified checkpoint contains 26 automated tests.

## Run the Phase 1 smoke test

~~~bash
python scripts/run_phase1.py --random-evaluations 1000 --seed 42
~~~

## Run the complete final experiment

The default command downloads the NLTK Brown Corpus, creates one reproducible
80/20 document split, and runs all five stochastic methods for 30 matched
seeds with 10,000 objective evaluations per method and seed:

~~~bash
python scripts/run_experiments.py
~~~

For a short software check only:

~~~bash
python scripts/run_experiments.py --smoke --runs 3 --evaluation-budget 1000
~~~

Smoke-test numbers are not valid evidence for the report.

## Jupyter Notebook

Open `notebooks/keyboard_layout_optimization.ipynb` in Jupyter or Google
Colab. When using Colab, upload and extract the complete project ZIP in the
same session first. Then use **Runtime > Restart session and run all**.

The notebook defaults to the final Brown Corpus experiment. To verify the
software quickly without producing reportable evidence, set the environment
variable `KEYBOARD_NOTEBOOK_FAST=1` before running it.

The notebook contains:

- project questions and assumptions;
- environment and corpus metadata;
- the mathematical objective;
- keyboard and adaptation diagrams;
- automatic unit tests;
- the complete matched-seed experiment;
- statistical and qualitative analysis;
- export of reproducibility tables, diagnostics, histories, and PDF figures.

At the end, the notebook also creates `final_results_bundle.zip` so the complete
tables, diagnostics, histories, and figures can be downloaded from Colab in one
file.

The committed notebook is the executed final Brown Corpus run. It contains all
30 matched seeds and no execution errors.

Every stochastic method receives the same objective-evaluation budget. GA
begins with QWERTY plus random layouts; SA begins with QWERTY. These
initialization decisions must be stated in the final report.

The Brown documents are split into 80% training and 20% testing using the
specified seed. The final experiments will store the exact document IDs so the
split can be reproduced.

## Important modeling decisions

- Layout strings list letters in physical-key order:
  qwertyuiopasdfghjklzxcvbnm.
- Each letter must occur exactly once.
- Key rows use horizontal offsets of 0.00, 0.25, and 0.75 key widths.
- Euclidean distance is the primary movement measure.
- Bigrams are counted only inside alphabetic words. Spaces and punctuation
  break transitions.
- Training and testing matrices are normalized separately.
- Every heuristic will receive the same number of objective evaluations.
- Final comparisons will use at least 30 independent runs.

## Repository structure

~~~text
src/keyboard_optimization/
  corpus.py       Corpus preprocessing and train/test loading
  keyboard.py     Keyboard geometry and layout validation
  objective.py    Common objective function
  baselines.py    QWERTY and Random Search baselines
  genetic_algorithm.py            Standard permutation GA
  adaptive_genetic_algorithm.py   Feedback-adaptive GA
  simulated_annealing.py          Standard and adaptive SA
  experiments.py                  Matched-budget repeated experiments
  analysis.py                     Statistics and effect sizes
notebooks/
  keyboard_layout_optimization.ipynb  Executable project narrative
scripts/
  run_phase1.py   Phase 1 verification and baseline runner
  run_ga.py       GA and baseline comparison runner
  run_experiments.py  Complete final experiment
  build_notebook.py  Reproducible notebook generator
tests/
  test_*.py       Unit and integration tests
results/
  final/          Final CSV, JSON, and compressed histories after execution
  figures/        Publication-ready PDF figures after execution
report_assets/
  results_table.tex  IEEE-ready principal results table
RESULTS_SUMMARY.md    Audited findings and interpretation limits
IEEE_REPORT_CONTENT.md  Six-page content and figure plan
REPRODUCIBILITY.md    Clean reproduction instructions
~~~

## Reproducibility checklist

- Record Python and dependency versions.
- Record all random seeds.
- Save train/test document identifiers.
- Use equal evaluation budgets.
- Use 30 matched seeds and retain all runs.
- Report mean, median, standard deviation, quartiles, confidence intervals,
  best, worst, and runtime.
- Use paired statistical tests and an effect size for adaptive ablations.
- Evaluate optimized layouts on unseen test text.
- Select the displayed final layout by training performance, never test cost.
- Preserve the executed final notebook and a permanent GitHub release/commit.

## Interpretation limit

The objective measures corpus-weighted center-to-center movement. It does not
directly measure typing speed, comfort, fatigue, injury risk, learning time, or
individual motor ability. Claims must therefore remain limited to modeled
movement cost unless human testing is added.

