from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

ANN = Path("data/jetbrains_dataset/annotations")

NAMES = {
    0: "MODEL_OPERATION",
    1: "DATA_IMPORT_EXTRACTION",
    2: "DATA_TRANSFORM",
    3: "EDA",
    4: "DATA_EXPORT",
    5: "NOT_RELEVANT",
}
SHORT = {0: "ModelOp", 1: "DataLoad", 2: "Transform", 3: "EDA", 4: "Export", 5: "NotRel"}

EXPORT = {
    "to_csv", "to_parquet", "to_excel", "to_json", "to_pickle", "to_feather", "to_hdf",
    "to_sql", "savefig", "save_model", "save_weights", "to_markdown", "save", "dump",
    "imsave", "imwrite",
}
IMPORT = {
    "read_csv", "read_parquet", "read_excel", "read_json", "read_sql", "read_table",
    "read_pickle", "read_hdf", "read_html", "read_fwf", "load_dataset", "loadtxt",
    "genfromtxt", "imread", "load_files", "load_iris", "load_digits", "load_boston",
    "load_wine", "load_breast_cancer", "fetch_openml", "fetch_california_housing",
    "fetch_20newsgroups", "fetch_lfw_people",
}
MODEL = {
    "predict", "predict_proba", "predict_log_proba", "decision_function", "cross_val_score",
    "cross_validate", "accuracy_score", "f1_score", "precision_score", "recall_score",
    "roc_auc_score", "roc_curve", "mean_squared_error", "mean_absolute_error", "r2_score",
    "log_loss", "confusion_matrix", "classification_report", "evaluate", "load_model",
    "load_weights",
}
TRANSFORM = {
    "fit_transform", "train_test_split", "get_dummies", "dropna", "fillna", "standardscaler",
    "minmaxscaler", "robustscaler", "onehotencoder", "ordinalencoder", "labelencoder",
    "simpleimputer", "tfidfvectorizer", "countvectorizer", "pca", "drop_duplicates",
}
EDA = {
    "head", "tail", "describe", "info", "value_counts", "hist", "boxplot", "heatmap",
    "pairplot", "countplot", "scatterplot", "corr", "crosstab",
}
NOTREL = {
    "seed", "set_seed", "manual_seed", "filterwarnings", "getlogger", "basicconfig",
    "set_option", "makedirs", "mkdir", "getcwd", "chdir",
}

CODE_RULES_STRONG = [(1, [r"\bnp\.load\b", r"\bpickle\.load\b", r"\bjoblib\.load\b"])]
CODE_RULES_PLT = [(3, [r"\bplt\.", r"\bsns\.", r"\bseaborn\.", r"\.plot\(", r"\.imshow\("])]

def api_name(label: str, code: str) -> str:
    """Best-effort name of the called function from the node label/code."""
    parts = (label or "").strip().split(None, 1)
    if len(parts) == 2 and parts[0].lower() in ("call", "method"):
        return parts[1].strip().lower()
    m = re.search(r"([A-Za-z_]\w*)\s*\($", (code or "").strip())
    return m.group(1).lower() if m else ""

def rule_label(label: str, code: str, include_plt: bool) -> tuple[int | None, str | None]:
    api = api_name(label, code)
    for table, lid in ((EXPORT, 4), (IMPORT, 1), (MODEL, 0), (TRANSFORM, 2), (EDA, 3), (NOTREL, 5)):
        if api in table:
            return lid, "api"
    code_l = code or ""
    for lbl, pats in CODE_RULES_STRONG:
        if any(re.search(p, code_l) for p in pats):
            return lbl, "code"
    if include_plt:
        for lbl, pats in CODE_RULES_PLT:
            if any(re.search(p, code_l) for p in pats):
                return lbl, "plt"
    return None, None

def main() -> None:
    ap = argparse.ArgumentParser(description="High-precision CALL-label corrector")
    ap.add_argument("--apply", action="store_true", help="write changes back (default: dry-run)")
    ap.add_argument("--include-plt", action="store_true", help="also apply matplotlib/seaborn->EDA")
    args = ap.parse_args()

    changed: Counter = Counter()
    by_tier: Counter = Counter()
    before: Counter = Counter()
    after: Counter = Counter()
    n_files_changed = 0
    samples: list[tuple] = []

    for f in sorted(ANN.glob("*.json")):
        with open(f) as fh:
            raw = json.load(fh)
        els = raw if isinstance(raw, list) else raw.get("elements", [])
        file_changed = False
        for e in els:
            d = e.get("data", {})
            if "source" in d or int(d.get("node_type", -1)) != 9:
                continue
            cur = d.get("predicted_label")
            if cur is None:
                continue
            try:
                cur = int(cur)
            except (ValueError, TypeError):
                continue
            if not 0 <= cur <= 6:
                continue
            cur = 4 if cur == 6 else cur
            before[cur] += 1
            new, tier = rule_label(d.get("label", ""), d.get("code", ""), args.include_plt)
            if new is not None and new != cur:
                if tier != "plt" and len(samples) < 20:
                    snippet = (d.get("code") or d.get("label") or "")[:50]
                    samples.append((SHORT[cur], SHORT[new], tier, snippet))
                changed[(cur, new)] += 1
                by_tier[tier] += 1
                after[new] += 1
                file_changed = True
                if args.apply:
                    d["predicted_label"] = new
                    d["domain_label"] = NAMES[new]
            else:
                after[cur] += 1
        if file_changed:
            n_files_changed += 1
            if args.apply:
                with open(f, "w") as fh:
                    json.dump(raw, fh, indent=2)

    total = sum(before.values())
    nchg = sum(changed.values())
    hc = by_tier["api"] + by_tier["code"]
    mode = "APPLIED" if args.apply else "DRY-RUN"
    plt_tag = " (+plt)" if args.include_plt else ""
    print(f"{mode}{plt_tag}: {nchg}/{total} labels changed ({100 * nchg / max(total, 1):.1f}%) across {n_files_changed} files")
    print(f"  high-confidence (exact API + code): {hc}    matplotlib convention: {by_tier['plt']}\n")
    print("Top corrections (old -> new):")
    for (o, n), c in changed.most_common(12):
        print(f"  {SHORT[o]:>9} -> {SHORT[n]:<9} {c:>5}")
    print("\nLabel distribution before -> after:")
    for k in range(6):
        delta = after.get(k, 0) - before.get(k, 0)
        print(f"  {SHORT[k]:<10} {before.get(k, 0):>6} -> {after.get(k, 0):>6}  ({delta:+d})")
    print("\nSample high-confidence changes (old -> new [tier] : code):")
    for o, n, tier, code in samples:
        print(f"  {o:>9} -> {n:<9} [{tier}] | {code}")

if __name__ == "__main__":
    main()
