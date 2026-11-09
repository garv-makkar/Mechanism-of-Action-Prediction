"""Small launcher for the existing experiments; no training settings are overridden."""
import argparse
import importlib.util
import os
from pathlib import Path
import runpy
import subprocess
import sys

from moa.data import PROJECT_ROOT, data_directory, validate_data

EXPERIMENTS = {
    "ann": "experiments/ann.py",
    "cnn": "experiments/cnn.py",
    "knn": "experiments/knn.py",
    "decision-tree": "notebooks/decision_tree.ipynb",
    "logistic-regression": "notebooks/logistic_regression.ipynb",
    "naive-bayes": "notebooks/naive_bayes.ipynb",
    "random-forest": "notebooks/random_forest.ipynb",
    "svm": "notebooks/svm.ipynb",
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List the eight original experiments")
    check = commands.add_parser("check", help="Validate data; does not train a model")
    check.add_argument("--data-dir", type=Path)
    run = commands.add_parser("run", help="Execute one complete original experiment")
    run.add_argument("experiment", choices=EXPERIMENTS)
    run.add_argument("--data-dir", type=Path)
    run.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    run.add_argument("--timeout", type=int, default=-1, help="Notebook cell timeout in seconds; -1 disables it")
    args = parser.parse_args(argv)
    if args.command == "list":
        for name, path in EXPERIMENTS.items():
            print(f"{name:20} {path}")
        return 0
    try:
        directory = data_directory(args.data_dir)
        info = validate_data(directory)
        print(f"Data OK: {info['rows']} rows, {info['features']} features, {info['targets']} targets in {directory}")
        if args.command == "check":
            return 0
        for module in ["numpy", "pandas", "scipy", "sklearn", "matplotlib", "tqdm"] + (["tensorflow"] if args.experiment == "cnn" else []):
            if importlib.util.find_spec(module) is None:
                raise ValueError(f"Missing {module}. Install requirements.txt" + (" and requirements-cnn.txt" if args.experiment == "cnn" else "") + " in this Python environment.")
        os.environ["MOA_DATA_DIR"] = str(directory)
        path = PROJECT_ROOT / EXPERIMENTS[args.experiment]
        if path.suffix == ".py":
            runpy.run_path(str(path), run_name="__main__")
        else:
            if importlib.util.find_spec("nbconvert") is None:
                raise ValueError("Install requirements.txt to execute notebooks.")
            output = args.output_dir.resolve()
            output.mkdir(parents=True, exist_ok=True)
            subprocess.run([
                sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", str(path),
                "--output", f"{path.stem}.executed.ipynb", "--output-dir", str(output),
                f"--ExecutePreprocessor.timeout={args.timeout}",
                "--ExecutePreprocessor.kernel_name=python3",
            ], check=True)
        return 0
    except (ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        return exc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
