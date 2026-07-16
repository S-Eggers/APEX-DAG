#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
    }
)

C_BLUE = "#0072B2"
C_ORANGE = "#C77E00"
C_GREEN = "#009E73"
C_VERMILLION = "#D55E00"
C_PURPLE = "#AA4E86"
C_SKY = "#56B4E9"
C_GREY = "#6B7280"
C_TEAL = "#1B9E9E"
C_TEAL_DARK = "#0E6B6B"

from matplotlib.colors import LinearSegmentedColormap

SYSTEMX_SEQ = LinearSegmentedColormap.from_list(
    "systemx_seq", ["#08306b", "#2a5f9e", "#5a8fc7", C_SKY, "#dbeaf7"]
)

FIGURE_TITLES_PATH = Path(__file__).with_name("figure_titles.yaml")

def _load_figure_titles() -> dict:
    try:
        import yaml

        with open(FIGURE_TITLES_PATH, encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}
    except Exception as exc:
        print(f"  (figure titles: using built-in defaults - {exc})")
        return {}

FIGURE_TITLES = _load_figure_titles()

def fig_title(fig_id: str, field: str = "title", default: str = "", **fmt) -> str:
    """Return the configured title/subtitle for a figure, else default."""
    entry = FIGURE_TITLES.get(fig_id) or {}
    text = entry.get(field, default) if isinstance(entry, dict) else default
    if not isinstance(text, str):
        text = default
    if fmt:
        try:
            return text.format(**fmt)
        except (KeyError, IndexError, ValueError):
            return text
    return text

SYSTEMX_HGT, SYSTEMX_MLP, SYSTEMX_XGB = "v2_hgt_feat_api24", "v2_mlp_feat_api24", "v2_xgboost_feat_api24"

SYSNAME = os.environ.get("SYSTEMX_SYSNAME", "SystemX")

MODEL_STYLE: dict[str, tuple[str, str, bool]] = {
    SYSTEMX_HGT: (f"{SYSNAME} (HGT)", C_BLUE, False),
    SYSTEMX_MLP: (f"{SYSNAME} (MLP)", C_ORANGE, False),
    SYSTEMX_XGB: (f"{SYSNAME} (XGBoost)", C_GREEN, False),
    "standard": (f"{SYSNAME} (HGT)", C_BLUE, False),
    "v2_mlp": (f"{SYSNAME} (MLP)", C_ORANGE, False),
    "v2_xgboost": (f"{SYSNAME} (XGBoost)", C_GREEN, False),
    "vamsa_static_baseline": ("Vamsa", C_GREY, True),
}

LLM_BASELINE_KEYS: list[tuple[str, str]] = [
    ("llm_gemini_code", "Gemini-2.5 Pro (API)"),
    ("llm_gemma2_2b_local", "Gemma-2 2B (local)"),
    ("llm_gemma2_2b_local_strong", "Gemma-2 2B (local, strong)"),
    ("llm_mistral_7b_local", "Mistral 7B (local)"),
    ("llm_mistral_7b_local_strong", "Mistral 7B (local, strong)"),
]
for _k, _lbl in LLM_BASELINE_KEYS:
    MODEL_STYLE.setdefault(_k, (_lbl, C_PURPLE, True))

OUR_MODELS = [SYSTEMX_HGT, SYSTEMX_MLP, SYSTEMX_XGB]
STATIC_BASELINES = ["vamsa_static_baseline"]
MAIN_MODELS = OUR_MODELS + [k for k, _ in LLM_BASELINE_KEYS] + STATIC_BASELINES

BASELINE_KEYS = ["vamsa_static_baseline", "llm_gemini_code",
                 "llm_gemma2_2b_local", "llm_mistral_7b_local"]

BENCH_WORKERS: dict[str, int] = {
    "llm_gemini_code": 16,
    "llm_gemini_code_rich": 16,
    "llm_gemini_strong": 16,
}

def _bench_workers(key: str) -> int:
    return BENCH_WORKERS.get(key, 1)

def _serial_latency(results: dict, key: str) -> float:
    """Per-notebook latency normalized to a single worker (max_workers=1)."""
    workers = _bench_workers(key)
    mean_s, _ = _metric(results, key, "timing", "mean_s")
    if workers <= 1:
        return mean_s
    tpn = results.get(key, {}).get("timing_per_notebook") or []
    vals = [e["time_s"] * min(workers, e["n_nodes"])
            for e in tpn if e.get("time_s") is not None and e.get("n_nodes")]
    if vals:
        return float(np.mean(vals))
    return mean_s * workers if not np.isnan(mean_s) else mean_s

def _best_our_model(results: dict, path: tuple[str, ...] = ("raw", "global", "f1")) -> str:
    """The strongest present SystemX backend by path (default raw micro-F1)."""
    best, best_val = SYSTEMX_XGB, -1.0
    for key in (SYSTEMX_XGB, SYSTEMX_HGT, SYSTEMX_MLP):
        if not _present(results, key):
            continue
        val, _ = _metric(results, key, *path)
        if np.isnan(val):
            val, _ = _metric(results, key, "global", "f1")
        if not np.isnan(val) and val > best_val:
            best, best_val = key, val
    return best

def _emph(val: float, column: list[float], text: str) -> str:
    r"""Wrap text in \textbf if val is the column maximum, \underline if it is the second-highest distinct value; otherwise return it unchanged."""
    finite = sorted({v for v in column if v is not None and np.isfinite(v)}, reverse=True)
    if not finite or val is None or not np.isfinite(val):
        return text
    if abs(val - finite[0]) < 1e-9:
        return f"\\textbf{{{text}}}"
    if len(finite) > 1 and abs(val - finite[1]) < 1e-9:
        return f"\\underline{{{text}}}"
    return text

TUPLE_METRICS = [
    ("Global", ("global", "f1")),
    ("D, D", ("per_type", "<D, D>", "f1")),
    ("D, M", ("per_type", "<M, D>", "f1")),
    ("Empty", ("per_type", "<D, Empty>", "f1")),
]
TUPLE_METRIC_COLORS = [C_GREY, C_BLUE, C_ORANGE, C_GREEN]

REFINER_VARIANTS = [
    ("v2_hgt_feat_api24", "Full pipeline"),
    ("no_analysis", "Without Analysis"),
    ("no_resolution", "Without Resolution"),
    ("no_propagation", "Without Propagation"),
    ("extraction_and_propagation", "Extract and Propagate"),
    ("seeding_refiner", "Seeding only"),
    ("empty_refiner", "No refiner"),
]

FEATURE_PRESETS = [
    ("struct_only", "Struct (15-d)"),
    ("emb_only", "Code (300-d)"),
    ("api_lib", "API+Lib (600-d)"),
    ("api24", "Standard (324-d)"),
    ("all", "All (929-d)"),
]
FEATURE_VARIANT_KEY = {
    "HGT": {"struct_only": "v2_hgt_feat_struct_only", "emb_only": "v2_hgt_feat_emb_only", "api_lib": "v2_hgt_feat_api_lib", "api24": "v2_hgt_feat_api24", "api29": "v2_hgt_feat_api29", "standard": "standard", "all": "v2_hgt_feat_all"},
    "MLP": {"struct_only": "v2_mlp_feat_struct_only", "emb_only": "v2_mlp_feat_emb_only", "api_lib": "v2_mlp_feat_api_lib", "api24": "v2_mlp_feat_api24", "api29": "v2_mlp_feat_api29", "standard": "v2_mlp", "all": "v2_mlp_feat_all"},
    "XGBoost": {"struct_only": "v2_xgboost_feat_struct_only", "emb_only": "v2_xgboost_feat_emb_only", "api_lib": "v2_xgboost_feat_api_lib", "api24": "v2_xgboost_feat_api24", "api29": "v2_xgboost_feat_api29", "standard": "v2_xgboost", "all": "v2_xgboost_feat_all"},
}
FAMILY_COLOR = {"HGT": C_BLUE, "MLP": C_ORANGE, "XGBoost": C_GREEN}

SCALABILITY_AXES = [("n_nodes", "CDF-IR nodes")]

DATASET_STYLE: dict[str, tuple[str, str]] = {
    "jetbrains": ("JetBrains (2020-21)", C_BLUE),
    "github": ("GitHub (2026)", C_VERMILLION),
    "combined": ("Combined", C_GREEN),
}
DATASET_ORDER = ["jetbrains", "github", "combined"]
GEN_DATASETS = ["jetbrains", "github"]

CLASS_ORDER = [
    "Model Op (Train/Pred)",
    "Data Load",
    "Transformation",
    "EDA / Inspection",
    "Environment Export",
    "Artifact Export",
    "Not Relevant",
]
CLASS_SHORT = {
    "Model Op (Train/Pred)": "Model Op",
    "Data Load": "Data Load",
    "Transformation": "Transform",
    "EDA / Inspection": "EDA",
    "Environment Export": "Env Exp.",
    "Artifact Export": "Art. Exp.",
    "Not Relevant": "Not Rel.",
}

def _scalar(node: Any) -> float:
    if isinstance(node, dict):
        return node.get("mean", float("nan"))
    return float(node) if node is not None else float("nan")

def _spread(node: Any) -> float:
    if isinstance(node, dict):
        return node.get("std", 0.0)
    return 0.0

def _metric(results: dict, key: str, *path: str) -> tuple[float, float]:
    """Return (mean, std) for results[key][path...]; NaN if skipped/missing."""
    r = results.get(key, {})
    if not r or r.get("skipped"):
        return float("nan"), float("nan")
    cur: Any = r
    for p in path:
        if not isinstance(cur, dict):
            return float("nan"), float("nan")
        cur = cur.get(p, {})
    return _scalar(cur), _spread(cur)

def _present(results: dict, key: str) -> bool:
    return key in results and not results[key].get("skipped")

def _save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}")
    plt.close(fig)
    print(f"  Saved {stem}.pdf / .png")

def fig_main_f1(results: dict, out_dir: Path) -> None:
    """Headline domain-classification quality per model: micro-F1 and macro-F1."""
    rows = []
    for key in MAIN_MODELS:
        if not _present(results, key):
            continue
        label, color, _is_base = MODEL_STYLE[key]
        micro, micro_s = _metric(results, key, "raw", "global", "f1")
        macro, macro_s = _metric(results, key, "raw", "global", "macro_f1")
        if np.isnan(micro):
            micro, micro_s = _metric(results, key, "global", "f1")
        if np.isnan(macro):
            macro, macro_s = _metric(results, key, "global", "macro_f1")
        if np.isnan(micro):
            continue
        rows.append((label, color, micro, micro_s, macro, macro_s))
    if not rows:
        print("  Skipping fig1 - no data.")
        return

    rows.sort(key=lambda r: r[2], reverse=True)
    labels = [r[0] for r in rows]
    colors = [r[1] for r in rows]
    micro = np.array([r[2] for r in rows])
    micro_s = np.array([r[3] for r in rows])

    y = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(labels) + 1.4))
    ax.barh(y, micro, height=0.62, color=colors, edgecolor="white", linewidth=0.6,
            xerr=micro_s, capsize=2.5, error_kw={"elinewidth": 1, "ecolor": "#555"}, zorder=3)
    for yi, mi in zip(y, micro, strict=False):
        ax.text(mi + 0.016, yi, f"{mi:.2f}", va="center", ha="left", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Micro-F1 (held-out, mean ± std)")
    ax.grid(axis="y", visible=False)
    ax.set_title(fig_title("fig1_main_f1", default="Lineage Graph Extraction by Model"), fontweight="bold")
    fig.tight_layout()
    _save(fig, out_dir, "fig1_main_f1")

def fig_ablations(results: dict, out_dir: Path) -> None:
    """Feature-set ablation: classification F1 per feature preset, across families."""
    families = [f for f in ("HGT", "MLP", "XGBoost") if any(_present(results, FEATURE_VARIANT_KEY[f][p]) for p, _ in FEATURE_PRESETS)]
    if not families:
        print("  Skipping fig3 - no feature-ablation data.")
        return

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    preset_labels = [lbl for _, lbl in FEATURE_PRESETS]
    x = np.arange(len(FEATURE_PRESETS))
    width = 0.8 / max(len(families), 1)
    for i, fam in enumerate(families):
        means, stds = [], []
        for p, _ in FEATURE_PRESETS:
            k = FEATURE_VARIANT_KEY[fam][p]
            m, s = _metric(results, k, "raw", "global", "f1")
            if np.isnan(m):
                m, s = _metric(results, k, "global", "f1")
            means.append(m)
            stds.append(s)
        off = (i - len(families) / 2 + 0.5) * width
        ax.bar(x + off, [0 if np.isnan(m) else m for m in means],
               yerr=[0 if np.isnan(s) else s for s in stds], width=width, label=fam,
               color=FAMILY_COLOR[fam], capsize=2, error_kw={"elinewidth": 0.8, "ecolor": "#555"}, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(preset_labels, rotation=18, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Micro-F1 (held-out, mean ± std)")
    ax.set_title(fig_title("fig3_ablations", default="Feature-Set Ablation"), fontweight="bold")
    ax.legend(title="Model", framealpha=0.9)
    fig.tight_layout()
    _save(fig, out_dir, "fig3_ablations")

def fig_scalability(results: dict, out_dir: Path) -> None:
    """Scoring time per notebook vs."""
    llm_keys = [k for k, _ in LLM_BASELINE_KEYS]
    llm_set = set(llm_keys)
    order = [SYSTEMX_HGT, SYSTEMX_MLP, SYSTEMX_XGB, "vamsa_static_baseline", *llm_keys]

    def _collect(key: str, xk: str) -> list[tuple[float, float]]:
        r = results.get(key, {})
        if r.get("skipped"):
            return []
        out = []
        for e in r.get("timing_per_notebook", []):
            x, y = e.get(xk), e.get("time_s")
            if x is None or y is None:
                continue
            x, y = float(x), float(y) * 1000.0
            if np.isfinite(x) and np.isfinite(y) and x > 0 and y > 0:
                out.append((x, y))
        return out

    requested = [(xk, xl) for xk, xl in SCALABILITY_AXES
                 if len({x for key in order for x, _ in _collect(key, xk)}) > 1]
    if not requested:
        requested = [(xk, xl) for xk, xl in [("loc", "Lines of code"), ("n_nodes", "CDF-IR nodes")]
                     if len({x for key in order for x, _ in _collect(key, xk)}) > 1][:1]
    axes_keys = requested
    if not axes_keys:
        print("  Skipping fig4 - no varying complexity axis.")
        return

    def _power_fit(xs: np.ndarray, ys: np.ndarray, lo: float = 2.0, hi: float = 98.0):
        """Power-law fit (linear in log-log) over the inter-percentile range."""
        x_lo, x_hi = np.percentile(xs, [lo, hi])
        y_hi = np.percentile(ys, hi)
        mask = (xs >= x_lo) & (xs <= x_hi) & (ys <= y_hi) & (xs > 0) & (ys > 0)
        if mask.sum() < 3 or len(set(xs[mask])) < 2:
            return None
        b, a = np.polyfit(np.log10(xs[mask]), np.log10(ys[mask]), 1)
        xr = np.logspace(np.log10(x_lo), np.log10(x_hi), 60)
        return xr, 10.0 ** (a + b * np.log10(xr))

    llm_line_styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1)), (0, (5, 1))]
    llm_style_of = {k: llm_line_styles[i % len(llm_line_styles)] for i, k in enumerate(llm_keys)}

    n = len(axes_keys)
    fig, axes = plt.subplots(n, 1, figsize=(6.6, 4.4 * n), squeeze=False)
    for ax, (xk, xlabel) in zip(axes[:, 0], axes_keys, strict=False):
        for key in order:
            pts = _collect(key, xk)
            if len(pts) < 3:
                continue
            label, color, _ = MODEL_STYLE[key]
            xs, ys = (np.asarray(v, dtype=float) for v in zip(*pts, strict=False))
            is_llm = key in llm_set
            c, ls, lw = (C_PURPLE, llm_style_of[key], 2.6) if is_llm else (color, "-", 3.0)
            try:
                fit = _power_fit(xs, ys)
            except (np.linalg.LinAlgError, ValueError):
                fit = None
            if fit is None:
                continue
            xr, yr = fit
            ax.plot(xr, yr, color=c, lw=lw, ls=ls, label=label, solid_capstyle="round")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("End-to-end time (ms, log scale)")
        ax.legend(framealpha=0.92, ncol=2, loc="lower right")
    fig.suptitle(fig_title("fig4_scalability", default="End-to-End Time versus Notebook Complexity"), fontweight="bold")
    fig.tight_layout()
    _save(fig, out_dir, "fig4_scalability")

def dump_stats(results: dict, out_dir: Path) -> None:
    lines = [f"{SYSNAME} Ablation - Cross-Validated Statistics\n", "=" * 72 + "\n"]
    for variant, data in results.items():
        lines.append(f"\n[{variant}]  (folds={data.get('n_folds', '-')})\n")
        if data.get("skipped"):
            lines.append(f"  SKIPPED: {data.get('skip_reason', '')}\n")
            continue
        pm, ps = _metric(results, variant, "global", "precision")
        rm, rs = _metric(results, variant, "global", "recall")
        fm, fs = _metric(results, variant, "global", "f1")
        mm, ms = _metric(results, variant, "global", "macro_f1")
        tm, ts = _metric(results, variant, "timing", "mean_s")
        lines.append(f"  P={pm:.3f}±{ps:.3f}  R={rm:.3f}±{rs:.3f}  microF1={fm:.3f}±{fs:.3f}  macroF1={mm:.3f}±{ms:.3f}  lat={tm:.3f}±{ts:.3f}s\n")
        ra = data.get("refiner_annotations", {})
        if ra:
            lines.append(f"  refiner annotations: leakage={ra.get('leakage', 0)}  dead_code={ra.get('dead_code', 0)}\n")
        for cls in CLASS_ORDER:
            cm, cs = _metric(results, variant, "per_class", cls, "f1")
            if not np.isnan(cm):
                lines.append(f"    {cls:<26} F1={cm:.3f}±{cs:.3f}\n")
    (out_dir / "stats_summary.txt").write_text("".join(lines))
    print("  Saved stats_summary.txt")
    _dump_latex(results, out_dir)

def _dump_latex(results: dict, out_dir: Path) -> None:
    """Main classification table - raw labeler quality (no refinement)."""
    def _row(label: str, key: str) -> str:
        if not _present(results, key):
            return f"    {label} & --- & --- & --- \\\\"
        fm, fs = _metric(results, key, "raw", "global", "f1")
        mm, _ = _metric(results, key, "raw", "global", "macro_f1")
        if np.isnan(fm):
            fm, fs = _metric(results, key, "global", "f1")
        if np.isnan(mm):
            mm, _ = _metric(results, key, "global", "macro_f1")
        tm, _ = _metric(results, key, "timing", "mean_s")
        f_str = f"{fm:.3f}$\\pm${fs:.3f}" if fs and fs > 0 else f"{fm:.3f}"
        if not tm or tm <= 0:
            t_str = "---"
        elif tm >= 10:
            t_str = f"{tm:.1f}"
        else:
            t_str = f"{tm:.3f}"
        return f"    {label} & {f_str} & {mm:.3f} & {t_str} \\\\"

    our_keys = [SYSTEMX_HGT, SYSTEMX_MLP, SYSTEMX_XGB]
    base_keys = [k for k in BASELINE_KEYS if _present(results, k)]

    def _cls_scores(key):
        fm, fs = _metric(results, key, "raw", "global", "f1")
        mm, _ = _metric(results, key, "raw", "global", "macro_f1")
        if np.isnan(fm):
            fm, fs = _metric(results, key, "global", "f1")
        if np.isnan(mm):
            mm, _ = _metric(results, key, "global", "macro_f1")
        return fm, fs, mm

    scored = {k: _cls_scores(k) for k in (our_keys + base_keys) if _present(results, k)}
    micro_col = [v[0] for v in scored.values()]
    macro_col = [v[2] for v in scored.values()]

    def _main_row(label: str, key: str) -> str:
        if key not in scored:
            return f"    {label} & --- & --- & --- & --- \\\\"
        fm, fs, mm = scored[key]
        f_txt = f"{fm:.3f}$\\pm${fs:.3f}" if fs and fs > 0 else f"{fm:.3f}"
        f_cell = _emph(fm, micro_col, f_txt)
        m_cell = _emph(mm, macro_col, f"{mm:.3f}")
        lat, _ = _metric(results, key, "timing", "mean_s")
        if not lat or np.isnan(lat) or lat <= 0:
            t_str = "---"
        elif lat >= 10:
            t_str = f"{lat:.1f}"
        else:
            t_str = f"{lat:.3f}"
        return f"    {label} & {f_cell} & {m_cell} & {_bench_workers(key)} & {t_str} \\\\"

    our_rows = [_main_row(MODEL_STYLE[k][0], k) for k in our_keys]
    baseline_rows = [_main_row(MODEL_STYLE[k][0], k) for k in base_keys]
    main = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Lineage graph extraction (per-operation domain classification) on our corpus (5-fold "
        "cross-validation, held-out): the three \\sysname backends vs.\\ the baselines. Micro/macro-F1 are the "
        "raw labeler (classifier) metrics, mean$\\pm$std across folds; per column the best value is "
        "\\textbf{bold} and the second-best \\underline{underlined}. \\emph{Workers} is the request "
        "concurrency each latency was measured at --- the hosted Gemini API labels 16 nodes in parallel, "
        "whereas the local SLMs run serially (their MLX server cannot batch); \\emph{Lat.} is the measured "
        "per-notebook wall-clock at that concurrency (local-SLM latency is an 8-notebook sample). The "
        "feature-set comparison is in Section~\\ref{sec:eval:ablation}.}",
        "\\label{tab:classification}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Method & Micro-F1 & Macro-F1 & Workers & Lat. (s) \\\\",
        "\\midrule",
        "\\multicolumn{5}{l}{\\textit{SystemX (ours)}} \\\\",
        *our_rows,
        "\\midrule",
        "\\multicolumn{5}{l}{\\textit{Baselines}} \\\\",
        *baseline_rows,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    (out_dir / "table_classification.tex").write_text(("\n".join(main) + "\n").replace(SYSNAME, "\\sysname").replace("SystemX", "\\sysname"))
    print("  Saved table_classification.tex")

    feat_rows = []
    for fam in ("HGT", "MLP", "XGBoost"):
        for p, plbl in FEATURE_PRESETS:
            feat_rows.append(_row(f"{fam} $\\cdot$ {plbl}", FEATURE_VARIANT_KEY[fam][p]))
    abl = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Feature-set ablation (5-fold CV, held-out): micro/macro-F1 as the input "
        "feature groups are varied, for each SystemX model family.}",
        "\\label{tab:feat_ablation}",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Model $\\cdot$ features & Micro-F1 & Macro-F1 & Lat. (s) \\\\",
        "\\midrule",
        *feat_rows,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    (out_dir / "table_feature_ablation.tex").write_text(("\n".join(abl) + "\n").replace(SYSNAME, "\\sysname").replace("SystemX", "\\sysname"))
    print("  Saved table_feature_ablation.tex")

def fig_tuple_f1(results: dict, out_dir: Path) -> None:
    """Lineage-tuple extraction quality per method: global F1 + the hardest type."""
    rows = []
    for m in MAIN_MODELS:
        if not _present(results, m):
            continue
        g, gs = _metric(results, m, "global", "f1")
        if np.isnan(g):
            continue
        dm, _ = _metric(results, m, "per_type", "<M, D>", "f1")
        rows.append((MODEL_STYLE[m][0], MODEL_STYLE[m][1], g, gs, dm))
    if not rows:
        print("  Skipping fig_tuple_f1 - no tuple data.")
        return

    rows.sort(key=lambda r: r[2], reverse=True)
    labels = [r[0] for r in rows]
    colors = [r[1] for r in rows]
    glob = np.array([r[2] for r in rows])
    glob_s = np.array([r[3] for r in rows])

    y = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(labels) + 1.4))
    ax.barh(y, glob, height=0.62, color=colors, edgecolor="white", linewidth=0.6,
            xerr=glob_s, capsize=2.5, error_kw={"elinewidth": 1, "ecolor": "#555"}, zorder=3)
    for yi, gv in zip(y, glob, strict=False):
        ax.text(gv + 0.016, yi, f"{gv:.2f}", va="center", ha="left", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Lineage-tuple F1 (combined, held-out)")
    ax.grid(axis="y", visible=False)
    ax.set_title(fig_title("fig_tuple_f1", default="Lineage-Tuple Extraction F1 by Method"), fontweight="bold")
    fig.tight_layout()
    _save(fig, out_dir, "fig_tuple_f1")

def fig_tuple_refiner(results: dict, out_dir: Path) -> None:
    """Refinement-rule ablation on the lineage-tuple task."""
    rv = [(k, lbl) for k, lbl in REFINER_VARIANTS if _present(results, k)]
    if not rv:
        print("  Skipping fig_tuple_refiner - no refiner-ablation tuple data.")
        return

    labels = [lbl for _, lbl in rv]
    y = np.arange(len(labels))
    height = 0.8 / len(TUPLE_METRICS)
    fig, ax = plt.subplots(figsize=(8.2, 0.62 * len(labels) + 1.8))
    for mi, (mlabel, path) in enumerate(TUPLE_METRICS):
        pairs = [_metric(results, k, *path) for k, _ in rv]
        vals = [p[0] for p in pairs]
        errs = [p[1] for p in pairs]
        off = (mi - len(TUPLE_METRICS) / 2 + 0.5) * height
        ax.barh(y + off, [0 if np.isnan(v) else v for v in vals], height,
                xerr=[0 if np.isnan(e) else e for e in errs], label=mlabel,
                color=TUPLE_METRIC_COLORS[mi], capsize=2,
                error_kw={"elinewidth": 0.8, "ecolor": "#555"}, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Tuple F1 (held-out)")
    ax.set_title(fig_title("fig_tuple_refiner", default="Refinement-Rule Ablation on Lineage Tuples (HGT)"), fontweight="bold")
    ax.legend(title="Tuple type", framealpha=0.9, ncol=len(TUPLE_METRICS), fontsize=8,
              loc="lower right")
    fig.tight_layout()
    _save(fig, out_dir, "fig_tuple_refiner")

def dump_tuple_table(results: dict, out_dir: Path) -> None:
    """LaTeX table: global P/R/F1 + per-tuple-type F1 for all variants present."""
    def _row(label: str, key: str) -> str:
        if not _present(results, key):
            return f"    {label} & --- & --- & --- & --- & --- & --- \\\\"
        p, _ = _metric(results, key, "global", "precision")
        r, _ = _metric(results, key, "global", "recall")
        f, fs = _metric(results, key, "global", "f1")
        dd, _ = _metric(results, key, "per_type", "<D, D>", "f1")
        md, _ = _metric(results, key, "per_type", "<M, D>", "f1")
        de, _ = _metric(results, key, "per_type", "<D, Empty>", "f1")
        f_str = f"{f:.3f}$\\pm${fs:.3f}" if fs and fs > 0 else f"{f:.3f}"
        return f"    {label} & {p:.3f} & {r:.3f} & {f_str} & {dd:.3f} & {md:.3f} & {de:.3f} \\\\"

    hdr = "Method & P & R & F1 & $D, D$ & $D, M$ & Empty \\\\"

    our_keys = [SYSTEMX_HGT, SYSTEMX_MLP, SYSTEMX_XGB]
    base_keys = [k for k in BASELINE_KEYS if _present(results, k)]

    def _tuple_scores(key):
        p, _ = _metric(results, key, "global", "precision")
        r, _ = _metric(results, key, "global", "recall")
        f, fs = _metric(results, key, "global", "f1")
        dd, _ = _metric(results, key, "per_type", "<D, D>", "f1")
        md, _ = _metric(results, key, "per_type", "<M, D>", "f1")
        de, _ = _metric(results, key, "per_type", "<D, Empty>", "f1")
        return p, r, f, fs, dd, md, de

    tscored = {k: _tuple_scores(k) for k in (our_keys + base_keys) if _present(results, k)}
    col = {i: [v[i] for v in tscored.values()] for i in (0, 1, 2, 4, 5, 6)}

    def _main_trow(label: str, key: str) -> str:
        if key not in tscored:
            return f"    {label} & --- & --- & --- & --- & --- & --- \\\\"
        p, r, f, fs, dd, md, de = tscored[key]
        f_txt = f"{f:.3f}$\\pm${fs:.3f}" if fs and fs > 0 else f"{f:.3f}"
        return (f"    {label} & {_emph(p, col[0], f'{p:.3f}')} & {_emph(r, col[1], f'{r:.3f}')}"
                f" & {_emph(f, col[2], f_txt)} & {_emph(dd, col[4], f'{dd:.3f}')}"
                f" & {_emph(md, col[5], f'{md:.3f}')} & {_emph(de, col[6], f'{de:.3f}')} \\\\")

    our_rows = [_main_trow(MODEL_STYLE[k][0], k) for k in our_keys]
    baseline_rows = [_main_trow(MODEL_STYLE[k][0], k) for k in base_keys]
    main = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Lineage-tuple extraction on our corpus (held-out; F1 is "
        "mean$\\pm$std across CV folds): the three \\sysname backends vs.\\ the baselines. Tuples are compared "
        "as exact $(\\text{type}, \\text{subject}, \\text{object})$ sets against the gold-labeled graph "
        "projected through the standard refiner. The $D, D$, $D, M$, and Empty columns are per-tuple-type F1; "
        "per column the best value is \\textbf{bold} and the second-best \\underline{underlined}. The "
        "refinement-rule ablation is in Section~\\ref{sec:eval:ablation}.}",
        "\\label{tab:tuples}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        hdr,
        "\\midrule",
        "\\multicolumn{7}{l}{\\textit{SystemX (ours)}} \\\\",
        *our_rows,
        "\\midrule",
        "\\multicolumn{7}{l}{\\textit{Baselines}} \\\\",
        *baseline_rows,
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\end{table}",
    ]
    (out_dir / "table_tuples.tex").write_text(("\n".join(main) + "\n").replace(SYSNAME, "\\sysname").replace("SystemX", "\\sysname"))
    print("  Saved table_tuples.tex")

    refiner_rows = [_row(lbl, k) for k, lbl in REFINER_VARIANTS]
    abl = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Refinement-rule ablation on the lineage-tuple task (HGT, 5-fold CV, held-out): "
        "each rule group is removed and tuple P/R/F1 (+ per-type F1) is remeasured. Refinement "
        "does not affect per-call classification, so it is evaluated only here.}",
        "\\label{tab:refiner_ablation}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        hdr,
        "\\midrule",
        *refiner_rows,
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\end{table}",
    ]
    (out_dir / "table_refiner_ablation.tex").write_text(("\n".join(abl) + "\n").replace(SYSNAME, "\\sysname").replace("SystemX", "\\sysname"))
    print("  Saved table_refiner_ablation.tex")

def render_tuple_outputs(tuple_results_path: Path, out_dir: Path) -> None:
    """Load tuple results (if present) and emit the tuple figure + table."""
    if not tuple_results_path.exists():
        print(f"  (skip tuple figures: {tuple_results_path} not found)")
        return
    with open(tuple_results_path) as f:
        tuple_results = json.load(f)
    print(f"Generating lineage-tuple figures from {tuple_results_path.name}...")
    fig_tuple_f1(tuple_results, out_dir)
    fig_tuple_refiner(tuple_results, out_dir)
    dump_tuple_table(tuple_results, out_dir)

def _load_dataset_results(results_dir: Path, datasets: list[str]) -> dict[str, dict]:
    """Load cv_results_<ds>.json for each requested dataset that exists."""
    out: dict[str, dict] = {}
    for ds in datasets:
        path = results_dir / f"cv_results_{ds}.json"
        if path.exists():
            with open(path) as f:
                out[ds] = json.load(f)
        else:
            print(f"  (skip {ds}: {path.name} not found)")
    return out

def fig_dataset_f1(by_dataset: dict[str, dict], out_dir: Path) -> None:
    """Grouped bars: refined micro-F1 per main model, grouped by dataset."""
    datasets = [d for d in DATASET_ORDER if d in by_dataset]
    models = [m for m in MAIN_MODELS if any(_present(by_dataset[d], m) for d in datasets)]
    if not datasets or not models:
        print("  Skipping fig5 - no cross-dataset data.")
        return

    x = np.arange(len(models))
    width = 0.8 / max(len(datasets), 1)
    fig, ax = plt.subplots(figsize=(1.7 * len(models) + 2.0, 4.6))
    for i, ds in enumerate(datasets):
        res = by_dataset[ds]
        means = [_metric(res, m, "global", "f1")[0] for m in models]
        stds = [_metric(res, m, "global", "f1")[1] for m in models]
        off = (i - len(datasets) / 2 + 0.5) * width
        label, color = DATASET_STYLE[ds]
        ax.bar(x + off, [0 if np.isnan(m) else m for m in means],
               yerr=[0 if np.isnan(s) else s for s in stds], width=width, label=label,
               color=color, capsize=2, error_kw={"elinewidth": 0.8, "ecolor": "#555"}, edgecolor="white")
        for xi, m in zip(x + off, means, strict=False):
            if not np.isnan(m):
                ax.text(xi, m + 0.015, f"{m:.2f}", ha="center", va="bottom", fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_STYLE[m][0] for m in models], rotation=12, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Micro-F1 (full pipeline, mean ± std)")
    ax.set_title(fig_title("fig5_dataset_f1", default="Lineage Graph Extraction by Dataset"), fontweight="bold")
    ax.legend(title="Dataset", framealpha=0.9)
    fig.tight_layout()
    _save(fig, out_dir, "fig5_dataset_f1")

_LIBGEN_TEST_N = {"pytorch": 24, "plotly": 18, "xgboost": 31, "seaborn": 220, "pyspark": 12, "lightgbm": 12}
_LIBGEN_ORDER = ["seaborn", "xgboost", "pytorch", "plotly", "pyspark", "lightgbm"]
_LIBGEN_MODELS = [
    (f"{SYSNAME} (HGT)", "v2_hgt_feat_api24", C_BLUE),
    (f"{SYSNAME} (XGBoost)", "v2_xgboost_feat_api24", C_GREEN),
    (f"{SYSNAME} (MLP)", "v2_mlp_feat_api24", C_ORANGE),
    ("Vamsa", "vamsa_static_baseline", C_GREY),
]

def _libgen_f1(results_dir: Path, holdout: str, variant: str, tuple_metric: bool = True) -> float:
    """Tuple- (or node-macro-) F1 for one variant on one held-out library; NaN if absent."""
    p = results_dir / f"libgen_{'tuple_' if tuple_metric else ''}{holdout}.json"
    if not p.exists():
        return float("nan")
    try:
        g = json.loads(p.read_text())[variant]["global"]["f1" if tuple_metric else "macro_f1"]
        return float(g["mean"] if isinstance(g, dict) else g)
    except (KeyError, TypeError, ValueError):
        return float("nan")

def fig_library_generalization(results_dir: Path, out_dir: Path, tuple_metric: bool = True) -> None:
    """Grouped bars: learned models vs the rule-based baseline on held-out libraries."""
    holdouts = [h for h in _LIBGEN_ORDER if not np.isnan(_libgen_f1(results_dir, h, "vamsa_static_baseline", tuple_metric))
                or not np.isnan(_libgen_f1(results_dir, h, "standard", tuple_metric))]
    if not holdouts:
        print(f"  Skipping fig7 library-generalization - no libgen results in {results_dir}.")
        return

    x = np.arange(len(holdouts))
    width = 0.8 / len(_LIBGEN_MODELS)
    fig, ax = plt.subplots(figsize=(1.05 * len(holdouts) + 1.8, 5.0))
    for i, (label, key, color) in enumerate(_LIBGEN_MODELS):
        vals = [_libgen_f1(results_dir, h, key, tuple_metric) for h in holdouts]
        off = (i - len(_LIBGEN_MODELS) / 2 + 0.5) * width
        ax.bar(x + off, [0 if np.isnan(v) else v for v in vals], width=width, label=label,
               color=color, edgecolor="white", linewidth=0.5)
        for xi, v in zip(x + off, vals, strict=False):
            if not np.isnan(v):
                ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{h}\n(n={_LIBGEN_TEST_N.get(h, '?')})" for h in holdouts])
    for tick, h in zip(ax.get_xticklabels(), holdouts, strict=False):
        if _LIBGEN_TEST_N.get(h, 0) >= 100:
            tick.set_fontweight("bold")
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=13)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Tuple F1 (held-out library)" if tuple_metric else "Node macro-F1 (held-out library)", fontsize=15)
    ax.set_xlabel("Held-out library (removed from training, used only for test)", fontsize=13)
    ax.set_title(fig_title("fig_library_generalization", default="Generalization to Unseen Libraries"), fontweight="bold")
    ax.legend(framealpha=0.9, ncol=2, fontsize=11, loc="lower center")
    ax.margins(x=0.02)
    fig.tight_layout()
    _save(fig, out_dir, "fig_library_generalization")

def _best_model_key(by_dataset: dict[str, dict], datasets: list[str],
                    candidates: tuple[str, ...] = (SYSTEMX_HGT, SYSTEMX_XGB, SYSTEMX_MLP)) -> str:
    """Pick the strongest learned model by mean held-out (raw) micro-F1."""
    best, best_score = candidates[0], -1.0
    for key in candidates:
        scores = []
        for ds in datasets:
            if ds not in by_dataset:
                continue
            m, _ = _metric(by_dataset[ds], key, "raw", "global", "f1")
            if np.isnan(m):
                m, _ = _metric(by_dataset[ds], key, "global", "f1")
            if not np.isnan(m):
                scores.append(m)
        if scores:
            mean = float(np.mean(scores))
            if mean > best_score:
                best, best_score = key, mean
    return best

def render_generalization_matrix(
    matrix: np.ndarray,
    labels: list[str],
    out_dir: Path,
    stem: str,
    title: str,
    subtitle: str,
    *,
    show_diagonal: bool = True,
    show_deltas: bool = True,
    y_axis_label: str = "Train on",
    y_labels: list[str] | None = None,
) -> list[Path]:
    """Render a train/condition × evaluation matrix using the paper figure style."""
    if np.all(np.isnan(matrix)):
        return []

    y_labels = labels if y_labels is None else y_labels
    n_rows, n_columns = matrix.shape
    finite = matrix[np.isfinite(matrix)]
    vmin = max(0.0, float(np.floor(finite.min() * 20) / 20) - 0.05)
    vmax = min(1.0, float(np.ceil(finite.max() * 20) / 20) + 0.05)
    if vmax - vmin < 0.1:
        vmin, vmax = max(0.0, vmin - 0.05), min(1.0, vmax + 0.05)

    fig, ax = plt.subplots(figsize=(1.7 * n_columns + 2.4, 1.25 * n_rows + 2.0))
    im = ax.imshow(np.nan_to_num(matrix, nan=vmin), cmap=SYSTEMX_SEQ, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(n_columns))
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Evaluate on", fontweight="bold")
    ax.set_ylabel(y_axis_label, fontweight="bold")
    ax.set_xticks(np.arange(-0.5, n_columns, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)

    mid = (vmin + vmax) / 2
    for i in range(n_rows):
        diag = matrix[i, i] if i < n_columns else np.nan
        for j in range(n_columns):
            value = matrix[i, j]
            color = "white" if (np.isnan(value) or value < mid) else "black"
            if np.isnan(value):
                ax.text(j, i, "-", ha="center", va="center", fontsize=11, color=color)
                continue
            if i == j and show_diagonal:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="white", lw=2.5))
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=12, fontweight="bold", color=color)
                ax.text(j, i + 0.30, "in-domain", ha="center", va="center", fontsize=7.5, style="italic", color=color)
            else:
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=12, color=color)
                if show_deltas and not np.isnan(diag):
                    delta = value - diag
                    ax.text(j, i + 0.30, f"Δ {delta:+.3f}", ha="center", va="center", fontsize=8, color=color)

    colorbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    colorbar.set_label("Micro-F1 (held-out)", rotation=270, labelpad=14)
    ax.set_title(f"{title}\n{subtitle}", fontweight="bold")
    fig.tight_layout()
    _save(fig, out_dir, stem)
    return [out_dir / f"{stem}.pdf", out_dir / f"{stem}.png"]

def render_coverage_bars(
    dataset_labels: list[str],
    group_labels: list[str],
    recall: list[list[float]],
    support: list[list[int]],
    out_dir: Path,
    stem: str,
    title: str,
    subtitle: str,
    *,
    colors: list[str] | None = None,
    y_label: str = "Recall",
) -> list[Path]:
    """Grouped bar chart in the paper style (same visual language as the matrices)."""
    if not dataset_labels or not group_labels:
        return []
    palette = colors or ["#1f3a5f", C_BLUE, C_GREY][: len(group_labels)]

    fig, ax = plt.subplots(figsize=(2.2 * len(dataset_labels) + 3.2, 4.8))
    x = np.arange(len(dataset_labels))
    width = 0.8 / max(len(group_labels), 1)
    for g_index, group in enumerate(group_labels):
        offset = (g_index - len(group_labels) / 2 + 0.5) * width
        values = [float(recall[d][g_index]) for d in range(len(dataset_labels))]
        bars = ax.bar(x + offset, values, width=width, label=group, color=palette[g_index % len(palette)], edgecolor="white")
        for d_index, bar in enumerate(bars):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{values[d_index]:.2f}\nn={int(support[d_index][g_index])}",
                    ha="center", va="bottom", fontsize=8.5, linespacing=1.35, color="#333")
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_labels)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel(y_label)
    ax.set_title(f"{title}\n{subtitle}", fontweight="bold", fontsize=11)
    ax.legend(title=None, framealpha=0.9)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", length=0)
    fig.tight_layout()
    _save(fig, out_dir, stem)
    return [out_dir / f"{stem}.pdf", out_dir / f"{stem}.png"]

def fig_generalization(by_dataset: dict[str, dict], results_dir: Path, out_dir: Path,
                       model_key: str | None = None) -> None:
    """Train×eval micro-F1 heatmap for the best learned model."""
    grid = [d for d in GEN_DATASETS if d in by_dataset]
    if len(grid) < 2:
        print("  Skipping fig6 - need >=2 datasets for generalization grid.")
        return

    if model_key is None:
        model_key = _best_model_key(by_dataset, grid)

    matrix = np.full((len(grid), len(grid)), np.nan)
    for i, train_ds in enumerate(grid):
        for j, eval_ds in enumerate(grid):
            if train_ds == eval_ds:
                matrix[i, j] = _metric(by_dataset[train_ds], model_key, "global", "f1")[0]
            else:
                gen_path = results_dir / f"gen_{train_ds}__{eval_ds}.json"
                if gen_path.exists():
                    with open(gen_path) as f:
                        gen = json.load(f)
                    matrix[i, j] = _metric(gen, model_key, "global", "f1")[0]

    if np.all(np.isnan(matrix)):
        print("  Skipping fig6 - no generalization data.")
        return

    model_name = MODEL_STYLE[model_key][0]
    render_generalization_matrix(
        matrix,
        [DATASET_STYLE[d][0] for d in grid],
        out_dir,
        "fig6_generalization",
        fig_title("fig6_generalization", "title", f"Cross-Dataset Generalization for {model_name}", model=model_name),
        fig_title(
            "fig6_generalization",
            "subtitle",
            r"Diagonal cells are in-domain scores and $\Delta$ is the transfer gap versus the source dataset.",
            model=model_name,
        ),
    )

def dump_dataset_table(by_dataset: dict[str, dict], out_dir: Path) -> None:
    """Unified LaTeX table with a Dataset column (main models × datasets)."""
    datasets = [d for d in DATASET_ORDER if d in by_dataset]
    if not datasets:
        print("  Skipping table_datasets - no cross-dataset data.")
        return
    model_keys = [SYSTEMX_HGT, SYSTEMX_MLP, SYSTEMX_XGB]

    def _row(res: dict, label: str, key: str) -> str:
        if not _present(res, key):
            return f"    {label} & --- & --- & --- & --- \\\\"
        rawf, _ = _metric(res, key, "raw", "global", "f1")
        fm, fs = _metric(res, key, "global", "f1")
        if np.isnan(rawf):
            rawf = fm
        mm, _ = _metric(res, key, "global", "macro_f1")
        tm, _ = _metric(res, key, "timing", "mean_s")
        return f"    {label} & {rawf:.3f} & {fm:.3f}$\\pm${fs:.3f} & {mm:.3f} & {tm:.3f} \\\\"

    body: list[str] = []
    for di, ds in enumerate(datasets):
        res = by_dataset[ds]
        body.append(f"    \\multicolumn{{5}}{{l}}{{\\textit{{{DATASET_STYLE[ds][0]}}}}} \\\\")
        body += [_row(res, MODEL_STYLE[k][0], k) for k in model_keys]
        present_bases = [k for k in BASELINE_KEYS if _present(res, k)]
        if present_bases:
            body.append("    \\cmidrule(lr){1-5}")
            body += [_row(res, MODEL_STYLE[k][0], k) for k in present_bases]
        if di != len(datasets) - 1:
            body.append("    \\midrule")

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Per-dataset lineage graph extraction (5-fold cross-validation, held-out): "
        "the three \\sysname variants vs.\\ the rule-based (Vamsa) and LLM baselines on each "
        "source corpus and their union. F1 is mean$\\pm$std across folds; local SLM baselines "
        "were only run on the combined corpus.}",
        "\\label{tab:datasets}",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Model & Raw F1 & Refined F1 & Macro-F1 & Lat. (s) \\\\",
        "\\midrule",
        *body,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    (out_dir / "table_datasets.tex").write_text(("\n".join(lines) + "\n").replace(SYSNAME, "\\sysname").replace("SystemX", "\\sysname"))
    print("  Saved table_datasets.tex")

BALANCING_VARIANTS: dict[str, list[tuple[str, str]]] = {
    "HGT": [("v2_hgt_cw_none", "no weight"), ("standard", "sqrt-inv*"), ("v2_hgt_cw_balanced", "balanced")],
    "MLP": [("v2_mlp", "none"), ("v2_mlp_undersample", "undersamp."), ("v2_mlp_classweight", "class-wt")],
    "XGBoost": [("v2_xgboost", "none"), ("v2_xgboost_undersample", "undersamp."), ("v2_xgboost_classweight", "class-wt")],
}

def _minority_class(results: dict, ref_key: str = SYSTEMX_XGB) -> str | None:
    """Class with the smallest summed support under a reference variant."""
    pc = results.get(ref_key, {}).get("per_class", {})
    sup = {c: d.get("support", 0) for c, d in pc.items()}
    return min(sup, key=sup.get) if sup else None

def fig_balancing(results: dict, out_dir: Path) -> None:
    """Per family, how each balancing strategy moves micro-F1, macro-F1, and the minority-class F1 (the metrics the ablation targets)."""
    families = [f for f, vs in BALANCING_VARIANTS.items()
                if sum(_present(results, k) for k, _ in vs) >= 2]
    if not families:
        print("  Skipping fig7 - no balancing-ablation variants present.")
        return
    minc = _minority_class(results)
    metrics = [("global", "f1", "micro-F1", C_GREY),
               ("global", "macro_f1", "macro-F1", C_BLUE)]
    if minc:
        metrics.append(("per_class", "f1", f"{CLASS_SHORT.get(minc, minc)} F1", C_VERMILLION))

    fig, axes = plt.subplots(1, len(families), figsize=(4.4 * len(families) + 0.6, 4.6), squeeze=False)
    for ax, fam in zip(axes[0], families, strict=False):
        variants = BALANCING_VARIANTS[fam]
        x = np.arange(len(variants))
        width = 0.8 / len(metrics)
        for mi, (grp, metric, mlabel, color) in enumerate(metrics):
            vals, errs = [], []
            for vkey, _ in variants:
                if grp == "per_class":
                    m, s = _metric(results, vkey, "per_class", minc, "f1")
                else:
                    m, s = _metric(results, vkey, grp, metric)
                vals.append(0 if np.isnan(m) else m)
                errs.append(0 if np.isnan(s) else s)
            off = (mi - len(metrics) / 2 + 0.5) * width
            ax.bar(x + off, vals, width, yerr=errs, label=mlabel, color=color, capsize=2,
                   error_kw={"elinewidth": 0.8, "ecolor": "#555"}, edgecolor="white")
            for xi, v in zip(x + off, vals, strict=False):
                if v > 0:
                    ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, rotation=90)
        ax.set_xticks(x)
        ax.set_xticklabels([lbl for _, lbl in variants], rotation=12, ha="right")
        ax.set_ylim(0, 1.0)
        ax.set_title(fam, fontweight="bold")
        if fam == families[0]:
            ax.set_ylabel("F1 (combined, held-out)")
        ax.legend(fontsize=8, framealpha=0.9)
    fig.suptitle(fig_title("fig7_balancing", default="Class-Imbalance Ablation by Balancing Strategy per Model (asterisk marks the default)"), fontweight="bold")
    fig.tight_layout()
    _save(fig, out_dir, "fig7_balancing")

def dump_balancing_table(results: dict, out_dir: Path) -> None:
    """LaTeX table: micro / macro / minority-class F1 per (family, strategy)."""
    families = [f for f, vs in BALANCING_VARIANTS.items()
                if sum(_present(results, k) for k, _ in vs) >= 2]
    if not families:
        print("  Skipping table_balancing - no balancing variants present.")
        return
    minc = _minority_class(results)
    minc_lbl = CLASS_SHORT.get(minc, minc) if minc else "min-cls"

    def _row(label: str, key: str) -> str:
        if not _present(results, key):
            return f"    {label} & --- & --- & --- \\\\"
        mi, _ = _metric(results, key, "global", "f1")
        ma, _ = _metric(results, key, "global", "macro_f1")
        mc, _ = _metric(results, key, "per_class", minc, "f1") if minc else (float("nan"), 0)
        return f"    {label} & {mi:.3f} & {ma:.3f} & {mc:.3f} \\\\"

    body: list[str] = []
    for fi, fam in enumerate(families):
        body.append(f"    \\multicolumn{{4}}{{l}}{{\\textit{{{fam}}}}} \\\\")
        body += [_row(lbl, key) for key, lbl in BALANCING_VARIANTS[fam]]
        if fi != len(families) - 1:
            body.append("    \\midrule")

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\caption{Class-imbalance ablation on the combined dataset (5-fold CV, held-out). "
        f"Tabular models use node under-sampling / inverse-frequency class weights; HGT uses loss "
        f"reweighting. {minc_lbl} is the minority class.}}",
        "\\label{tab:balancing}",
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        f"Strategy & Micro-F1 & Macro-F1 & {minc_lbl} F1 \\\\",
        "\\midrule",
        *body,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ]
    (out_dir / "table_balancing.tex").write_text("\n".join(lines) + "\n")
    print("  Saved table_balancing.tex")

def main_compare(args) -> None:
    results_dir = Path(args.results_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_dataset = _load_dataset_results(results_dir, args.datasets)
    if not by_dataset:
        print(f"Error: no cv_results_<dataset>.json found in {results_dir}")
        raise SystemExit(1)
    print(f"Loaded results for datasets: {', '.join(by_dataset)}")

    default_ds = "combined" if "combined" in by_dataset else next(iter(by_dataset))
    default = by_dataset[default_ds]
    print(f"Generating default figure set from '{default_ds}'...")
    fig_main_f1(default, out_dir)
    fig_ablations(default, out_dir)
    fig_scalability(default, out_dir)
    dump_stats(default, out_dir)

    print("Generating cross-dataset comparison figures...")
    fig_dataset_f1(by_dataset, out_dir)
    fig_generalization(by_dataset, results_dir, out_dir)
    dump_dataset_table(by_dataset, out_dir)

    print("Generating class-imbalance (balancing) ablation figures...")
    fig_balancing(default, out_dir)
    dump_balancing_table(default, out_dir)

    tuple_path = results_dir / f"tuple_results_{default_ds}.json"
    if not tuple_path.exists():
        tuple_path = Path(args.tuple_results_path)
    render_tuple_outputs(tuple_path, out_dir)
    print(f"\nAll figures saved to {out_dir}/")

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper figures from CV results")
    parser.add_argument("--results_path", default="output/results/cv_results.json",
                        help="Single-dataset results (default mode).")
    parser.add_argument("--output_dir", default="output/figures")
    parser.add_argument("--compare", action="store_true",
                        help="Cross-dataset mode: load cv_results_<dataset>.json + gen_*.json.")
    parser.add_argument("--results_dir", default="output/results",
                        help="Directory holding per-dataset results (compare mode).")
    parser.add_argument("--datasets", nargs="*", default=["jetbrains", "github", "combined"],
                        help="Datasets to include in the comparison (compare mode).")
    parser.add_argument("--tuple_results_path", default="output/results/tuple_results_combined.json",
                        help="Lineage-tuple extraction results (evaluate_tuples.py). Rendered if present.")
    args = parser.parse_args()

    if args.compare:
        main_compare(args)
        return

    results_path = Path(args.results_path)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not results_path.exists():
        print(f"Error: results file not found: {results_path}")
        raise SystemExit(1)

    with open(results_path) as f:
        results = json.load(f)
    print(f"Loaded results for {len(results)} variants from {results_path}")

    print("Generating figures...")
    fig_main_f1(results, out_dir)
    fig_ablations(results, out_dir)
    fig_scalability(results, out_dir)
    dump_stats(results, out_dir)
    render_tuple_outputs(Path(args.tuple_results_path), out_dir)
    print(f"\nAll figures saved to {out_dir}/")

if __name__ == "__main__":
    main()
