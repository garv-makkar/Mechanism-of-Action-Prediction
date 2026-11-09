# Mechanism of Action Prediction

Machine learning course project (IIIT Delhi, October 2023) by Rishav Raj, Parth Kaushal, Shubham Pal, and Garv Makkar. The experiments predict multiple drug mechanism-of-action labels from gene-expression and cell-viability measurements.

Start with the [project report](reports/ML%20Project%20Report.pdf), [presentation](reports/ML%20Project%20Presentation.pdf), and [review notes](docs/review.md). This repository contains research experiments, with console metrics and plots; it does not provide a deployed predictor or saved trained models.

## Quick setup

Use **Python 3.10 or 3.11** (3.11 preferred) and run the commands from the repository root. The original dependency versions were not recorded; the pinned requirements provide a conservative baseline, not a verified recreation of the 2023 environment.

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
python -m ipykernel install --sys-prefix --name python3 --display-name "Python (MoA)"
```

macOS / Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
python -m ipykernel install --sys-prefix --name python3 --display-name "Python (MoA)"
```

If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` in place of `python` in subsequent commands. No execution-policy change is needed.

For **CNN only**, also install:

```bash
python -m pip install -r requirements-cnn.txt
```

The documented path runs on CPU and does not require CUDA. TensorFlow installation depends on platform and Python wheel availability. If its pinned version cannot be installed, use Python 3.11 on a supported platform; the other seven experiments do not need TensorFlow.

## Get the data

Download and extract the **lish-moa** competition dataset from [Kaggle](https://www.kaggle.com/competitions/lish-moa/data), using your own account and accepting the competition terms as required. Data and credentials are not included in this repository.

Place these two files directly in `data/raw/`:

```text
data/raw/
  train_features.csv
  train_targets_scored.csv
```

Other competition files are not used by these experiments. Keep the original rows in matching order; `sig_id` connects each feature row to its targets. The scripts create local holdouts from the labeled training data, rather than score Kaggle's unlabeled test set.

To reuse an existing data directory, set its path before launching Python or Jupyter:

```powershell
$env:MOA_DATA_DIR = "C:\datasets\lish-moa"
```

```bash
export MOA_DATA_DIR="/path/to/lish-moa"
```

The precedence is `--data-dir`, then `MOA_DATA_DIR`, then the repository's `data/raw/`. Relative custom paths resolve from your working directory.

## Check and run

```bash
python -m moa list
python -m moa check
python -m moa run knn
```

`check` uses only Python's standard library. It checks required files and columns, row widths, unique IDs, binary targets, and matching row order without training. It is a structural check, not a full statistical or numeric-feature validation.

All experiment names:

| Command name | Source | Execution |
| --- | --- | --- |
| `naive-bayes` | [notebook](notebooks/naive_bayes.ipynb) | PCA sweep, 3-fold evaluation, final holdout |
| `logistic-regression` | [notebook](notebooks/logistic_regression.ipynb) | PCA sweep, 3-fold evaluation, final holdout |
| `svm` | [notebook](notebooks/svm.ipynb) | Original sweep and final holdout |
| `decision-tree` | [notebook](notebooks/decision_tree.ipynb) | Original split-parameter sweep and holdout |
| `random-forest` | [notebook](notebooks/random_forest.ipynb) | Original repeated configurations and holdout |
| `ann` | [script](experiments/ann.py) | Four MLP activation configurations |
| `knn` | [script](experiments/knn.py) | Five neighbor counts, 100 through 500 |
| `cnn` | [script](experiments/cnn.py) | Three activations, 50 epochs each; TensorFlow required |

Examples:

```bash
python -m moa run logistic-regression --data-dir /path/to/lish-moa
python -m moa run naive-bayes --output-dir outputs/review
python -m moa run cnn
```

A run executes the **entire original experiment**, including parameter sweeps. These are potentially long CPU jobs, not smoke tests. Several scripts fit a separate classifier per target. Dataset sampling, PCA dimensions, model settings, and evaluation behavior have been retained.

Notebook runs write `*.executed.ipynb` into `outputs/` by default and leave source notebooks intact. Reusing the same output directory overwrites the previous executed notebook for that experiment. Notebook cell timeouts are disabled by default; use `--timeout 3600` to set a one-hour limit per cell. Script runs print results and open Matplotlib plots; the output-directory and timeout options apply only to notebooks. For a terminal without a display, set `MPLBACKEND=Agg` before running scripts (plots will not be displayed or automatically saved).

## Explore interactively

```bash
python -m jupyterlab
```

Open a file under `notebooks/`, choose **Python (MoA)**, restart the kernel, and run cells from top to bottom. Historical saved outputs are retained for review; they are not evidence of a new run. Changing `MOA_DATA_DIR` after starting Jupyter requires restarting the server/kernel with the new environment.

## Repository layout

```text
moa/                    Shared data paths, input checks, and CLI
experiments/            ANN, CNN, and KNN scripts
notebooks/              Five original classical-model notebooks
reports/                Original report and presentation PDFs
  figures/              Original random-forest plots
docs/review.md          Scope, report/code differences, reproducibility notes
data/raw/               Local CSVs (ignored by Git)
outputs/                Executed notebooks (generated, ignored by Git)
tests/                  Lightweight setup and repository checks
requirements.txt        Core scientific Python and Jupyter baseline
requirements-cnn.txt    Optional TensorFlow dependency
pyproject.toml          Editable installation and moa console command
```

The editable installation is intentional: the launcher locates notebooks and scripts in this checkout. Keep the checkout in place; this is not a self-contained wheel distribution.

## Checks for contributors

```bash
python -m unittest discover -s tests -v
```

These checks need no dataset or ML libraries. CI runs them on Python 3.11 and checks CLI discovery. They validate setup utilities and source structure, not model quality or training reproducibility. Before reviewing changes, read [the known limitations](docs/review.md).

## Troubleshooting

- **Missing CSV:** extract the two files into the exact directory printed by `python -m moa check`, or pass `--data-dir`.
- **No module named moa:** run `python -m pip install -e .` inside the activated environment. Restart any existing notebook kernel.
- **Missing scientific library:** install `requirements.txt` with the same Python used to run the command. CNN also needs `requirements-cnn.txt`.
- **Notebook uses another Python:** rerun the kernel-registration command above inside `.venv` and select Python (MoA).
- **Python 3.12+ dependency failures:** create the documented Python 3.11 environment. The local default Python may differ from the project baseline.
- **A run is slow or results differ:** the preserved sweeps can train many classifiers, and some original random operations are unseeded. See the review notes before comparing results.
