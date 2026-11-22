# Reproducibility Guide

## Preserved final protocol

- Brown Corpus document split seed: 42.
- 400 training documents and 100 test documents.
- Optimization seeds: 100 through 129.
- Objective evaluations: 10,000 per stochastic method and run.
- Standard/adaptive GA initial mutation rate: 0.15.
- Standard/adaptive SA initial temperature: 1.0.
- The exact complete configurations are stored in
  `results/final/experiment_metadata.json`.

## Install and test

~~~bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m unittest discover -s tests -v
~~~

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

The final checkpoint has 26 automated tests. All must pass before running or
interpreting an experiment.

## Reproduce the full experiment

~~~bash
python scripts/run_experiments.py \
  --runs 30 \
  --evaluation-budget 10000 \
  --first-seed 100 \
  --output-dir results/reproduction
~~~

The first Brown Corpus use downloads the corpus through NLTK. Results should be
compared with the preserved files in `results/final`; small runtime differences
are expected across machines.

## Notebook

Open `notebooks/keyboard_layout_optimization.ipynb` and use **Restart and Run
All**. The committed notebook contains the verified final outputs. The setup
identifies Phase 3 ZIP archives from their contents rather than requiring one
specific filename, so renamed downloads do not change execution.

## Preserved artifacts

- `results/final/run_results.csv`: every run and best layout.
- `results/final/convergence_histories.npz`: complete best-so-far histories.
- `results/final/experiment_metadata.json`: configurations and document IDs.
- `results/final/adaptation_diagnostics.json`: controller trajectories.
- `results/final/*summary.csv`: compact report tables.
- `results/figures/*.pdf`: vector figures used by the notebook/report.

Do not tune parameters using the held-out test costs. Any future tuning should
use training-only pilots or a separately defined validation partition.

