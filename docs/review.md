# Review and reproducibility notes

## Scope of the repository cleanup

The October 2023 report and presentation are preserved under `reports/`; original figure files are under `reports/figures/`. All eight implemented experiments remain available. This cleanup changes file organization, CSV resolution, validation, launch tooling, import behavior, and documentation. It does not tune models, alter splits, change preprocessing, or claim better scores.

The three scripts now run through `main()` so importing a script does not start training. Notebook code changes are limited to the shared data loader; an introductory Markdown cell was added, and historical outputs were retained.

## How to review

1. Read the [report](../reports/ML%20Project%20Report.pdf) for the original motivation and methodology; the [presentation](../reports/ML%20Project%20Presentation.pdf) is also archived.
2. Inspect the selected notebook or script. Treat executable settings as the description of what a new run actually does.
3. Run `python -m moa check` against the original dataset before training.
4. Use a fresh kernel / process for each experiment. Keep executed notebooks in a separate output directory for each review run.
5. Record the Git revision, Python version, `python -m pip freeze`, command, dataset provenance, console metrics, and run date when comparing results.

## Existing report/code differences

These are observations retained for review, not changes made to the experiments:

- The report abstract mentions AdaBoost, but this checkout contains no AdaBoost implementation. The report body and repository include Decision Tree.
- The report's ANN discussion describes PCA with 30 components and no sampling; the ANN script uses PCA with 10 components and truncates each split to 5,000 rows.
- The KNN discussion says neighbors were checked up to 400; the saved script checks 100, 200, 300, 400, and 500.
- The SVM notebook labels a sweep with `penalty`, but the classifier's `C` stays at 0.5 while PCA changes through `pc[itr]`. Its final holdout uses an RBF kernel, while the sweep uses a linear kernel.
- The random-forest notebook loops over `max_depth` values but constructs the same estimator settings each time (`max_depth=None`, 80 trees, minimum split fraction 0.6). Its plot should not be treated as a clean max-depth comparison.
- The decision-tree notebook has a KFold loop, but the active assignments repeatedly use the same training and validation frames instead of the fold indices.
- The report abstract lists eight model names with seven loss values. Avoid deriving a model-to-score table from that sentence. The report concludes CNN performs best; no new ranking has been verified in this cleanup.

## Limits of reproducibility

- The original environment and complete dependency lockfile are unavailable. The new direct dependency pins are a setup baseline and have not been validated by a complete training run here.
- Several original sampling, PCA, tree, and neural-network operations have no fixed seed. Existing fixed seeds are retained, but results need not be identical across runs or environments.
- Train and holdout data are one-hot encoded separately in the original experiments. A subset missing a category can produce different columns. This modeling behavior is preserved; use the complete original data rather than small arbitrary CSV slices for training.
- Splits and sample sizes differ across experiments, so their scores do not necessarily form a controlled comparison. The data check deliberately does not reduce sample sizes or make synthetic data look like a valid training run.
- Dataset and trained models are not bundled. Metrics and plots come from each run, and historical notebook outputs are not a reproducibility guarantee.
- There is no inference/export pipeline, Kaggle submission generator, or automatic saving of script plots.

## Validation of the cleanup

The automated suite checks malformed/misaligned data rejection, path precedence, CLI behavior, notebook source syntax, and safe script imports. Full model execution requires the user-supplied Kaggle data and the documented scientific Python environment. CI intentionally performs no model training or dataset download.
