# Colab, Jupyter, and GitHub Guide

## What each tool does

- **Jupyter Notebook** is the `.ipynb` document containing executable Python,
  explanations, tables, plots, and saved outputs.
- **Google Colab** is the browser-based environment used to run the notebook.
- **GitHub** stores and shares the complete repository through a permanent link.
- **Overleaf** is used separately to write and compile the six-page IEEE paper.

## Run the checkpoint in Google Colab

1. Download and extract the project ZIP on your computer.
2. Open <https://colab.research.google.com/>.
3. Select **File > Upload notebook**.
4. Choose `notebooks/keyboard_layout_optimization.ipynb` from the extracted
   project.
5. In Colab's left **Files** panel, upload the complete project ZIP as well.
6. Select **Runtime > Restart session and run all**.

The notebook setup detects the uploaded ZIP, extracts it, finds the project
source, and continues. The default run uses the Brown Corpus, 30 matched seeds,
and 10,000 evaluations per method. Keep the browser tab open until it finishes.

For a short software check only, insert and run this code cell before the
configuration cell:

    import os
    os.environ["KEYBOARD_NOTEBOOK_FAST"] = "1"

Fast-mode results must not be placed in the final paper.

## Create the GitHub repository after final results are ready

1. Sign in at <https://github.com/> and choose **New repository**.
2. Name it `adaptive-keyboard-layout-optimization`.
3. Make it public unless the instructor has explicitly approved a private link.
4. Do not initialize it with another README; this project already contains one.
5. Extract the final ZIP and upload its contents so that `README.md`, `src`,
   `notebooks`, `scripts`, `tests`, and `results` appear at the repository root.
6. Do not upload the outer ZIP, `.venv`, or `__pycache__` folders.
7. Commit the files.
8. Open the notebook on GitHub and verify that its headings, code, figures, and
   final saved outputs are visible.
9. Create a release or tag named `v1.0-final` after the submitted notebook,
   results, report, and code are frozen.
10. Put the release URL or exact commit permalink in the IEEE paper.

The repository page is a convenient project link, but the release or commit
permalink is the stronger permanent reference because it cannot silently change
when later edits are made.
