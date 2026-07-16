from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import nbformat

from SystemX.sca.leakage import LeakageClass

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class Stack:
    """One library/API combination the templates are instantiated over."""

    loader: str
    scaler_import: str
    scaler_ctor: str
    model_import: str
    model_ctor: str
    target: str

_STACKS = [
    Stack("pd.read_csv('data.csv')",
          "from sklearn.preprocessing import StandardScaler", "StandardScaler()",
          "from sklearn.linear_model import LogisticRegression", "LogisticRegression()", "target"),
    Stack("pd.read_parquet('train.parquet')",
          "from sklearn.preprocessing import MinMaxScaler", "MinMaxScaler()",
          "from sklearn.ensemble import RandomForestClassifier", "RandomForestClassifier()", "label"),
    Stack("pd.read_csv('churn.csv')",
          "from sklearn.preprocessing import RobustScaler", "RobustScaler()",
          "from sklearn.svm import SVC", "SVC()", "churn"),
    Stack("pd.read_csv('housing.csv')",
          "from sklearn.preprocessing import StandardScaler", "StandardScaler()",
          "from sklearn.ensemble import GradientBoostingClassifier", "GradientBoostingClassifier()", "y"),
]

def _header(s: Stack) -> list[str]:
    return [
        "import pandas as pd",
        "from sklearn.model_selection import train_test_split",
        s.scaler_import,
        s.model_import,
        "from sklearn.metrics import accuracy_score",
        f"df = {s.loader}",
        f"y = df['{s.target}']",
        f"X = df.drop('{s.target}', axis=1)",
    ]

def build_clean(s: Stack) -> list[str]:
    return [
        *_header(s),
        "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)",
        f"scaler = {s.scaler_ctor}",
        "X_train_s = scaler.fit_transform(X_train)",
        "X_test_s = scaler.transform(X_test)",
        f"clf = {s.model_ctor}",
        "clf.fit(X_train_s, y_train)",
        "pred = clf.predict(X_test_s)",
        "acc = accuracy_score(y_test, pred)",
    ]

def build_preprocessing_before_split(s: Stack) -> list[str]:
    return [
        *_header(s),
        f"scaler = {s.scaler_ctor}",
        "Xs = scaler.fit_transform(X)",
        "X_train, X_test, y_train, y_test = train_test_split(Xs, y, random_state=0)",
        f"clf = {s.model_ctor}",
        "clf.fit(X_train, y_train)",
        "pred = clf.predict(X_test)",
        "acc = accuracy_score(y_test, pred)",
    ]

def build_test_in_train(s: Stack) -> list[str]:
    return [
        *_header(s),
        "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)",
        f"scaler = {s.scaler_ctor}",
        "X_train_s = scaler.fit_transform(X_train)",
        "X_test_s = scaler.transform(X_test)",
        f"clf = {s.model_ctor}",
        "clf.fit(X_test_s, y_test)",
        "pred = clf.predict(X_test_s)",
    ]

def build_metric_on_train(s: Stack) -> list[str]:
    return [
        *_header(s),
        "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)",
        f"scaler = {s.scaler_ctor}",
        "X_train_s = scaler.fit_transform(X_train)",
        f"clf = {s.model_ctor}",
        "clf.fit(X_train_s, y_train)",
        "train_pred = clf.predict(X_train_s)",
        "acc = accuracy_score(y_train, train_pred)",
    ]

def build_target_leakage(s: Stack) -> list[str]:
    return [
        *_header(s),
        "leak = y * 2",
        "X = pd.concat([X, leak], axis=1)",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)",
        f"clf = {s.model_ctor}",
        "clf.fit(X_train, y_train)",
        "pred = clf.predict(X_test)",
    ]

def build_no_holdout_evaluation(s: Stack) -> list[str]:
    return [
        *_header(s),
        f"scaler = {s.scaler_ctor}",
        "Xs = scaler.fit_transform(X)",
        f"clf = {s.model_ctor}",
        "clf.fit(Xs, y)",
    ]

_BUILDERS = {
    frozenset(): build_clean,
    frozenset({LeakageClass.PREPROCESSING_BEFORE_SPLIT.value}): build_preprocessing_before_split,
    frozenset({LeakageClass.TEST_IN_TRAIN.value}): build_test_in_train,
    frozenset({LeakageClass.METRIC_ON_TRAIN.value}): build_metric_on_train,
    frozenset({LeakageClass.TARGET_LEAKAGE.value}): build_target_leakage,
    frozenset({LeakageClass.NO_HOLDOUT_EVALUATION.value}): build_no_holdout_evaluation,
}

def _write_notebook(path: Path, source_lines: list[str]) -> None:
    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_code_cell("\n".join(source_lines))]
    with open(path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

def generate(output_dir: Path, repeats: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    nb_dir = output_dir / "notebooks"
    nb_dir.mkdir(exist_ok=True)

    ground_truth: dict[str, list[str]] = {}
    idx = 0
    for r in range(repeats):
        stack = _STACKS[r % len(_STACKS)]
        for label, builder in _BUILDERS.items():
            name = f"nb_{idx:04d}_{'clean' if not label else '_'.join(sorted(label))}.ipynb"
            _write_notebook(nb_dir / name, builder(stack))
            ground_truth[name] = sorted(label)
            idx += 1

    manifest = {
        "n_notebooks": idx,
        "classes": [c.value for c in LeakageClass],
        "ground_truth": ground_truth,
    }
    with open(output_dir / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate the seeded leakage benchmark.")
    ap.add_argument("--output_dir", required=True, type=Path)
    ap.add_argument("--repeats", type=int, default=8,
                    help="How many times to cycle the library stacks (each cycle = 6 notebooks/stack).")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    manifest = generate(args.output_dir, args.repeats)
    n_pos = sum(1 for v in manifest["ground_truth"].values() if v)
    logger.info("Wrote %d notebooks (%d leaky / %d clean) + ground_truth.json to %s",
                manifest["n_notebooks"], n_pos, manifest["n_notebooks"] - n_pos, args.output_dir)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
