from __future__ import annotations

import ast
import csv
import json
import logging
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from SystemX.experiment.ablation.datasets import GENERALIZATION_DATASETS, resolve_dataset
from SystemX.experiment.ablation.evaluate_all import _GraphWrapper
from SystemX.experiment.evaluation.metrics import ConfusionMatrix
from SystemX.labeler.vamsa_static_labeler import VamsaStaticLabeler
from SystemX.labeling.vamsa_loader import DomainEdgeId, IOSignatureMappingPolicy, VamsaEntry, VamsaKBLoader
from SystemX.nn.training.v2.data_utils import annotation_to_networkx
from SystemX.parser.sanitizer_mixin import IPythonSanitizerMixin
from SystemX.sca.constants import COMPUTE_HUBS, DOMAIN_EDGE_TYPES, DOMAIN_EDGES

logger = logging.getLogger(__name__)

DATASET_LABELS = {"jetbrains": "JetBrains (2020-21)", "github": "GitHub (2026)"}

_GROUP_SHARED = "shared"
_GROUP_UNIQUE = "unique"
_GROUP_UNRESOLVED = "unresolved"
_GROUP_ORDER = (_GROUP_SHARED, _GROUP_UNIQUE, _GROUP_UNRESOLVED)

@dataclass
class DatasetLibraryInventory:
    libraries: list[str]
    library_api_pairs: list[tuple[str, str]] | None = None
    notebooks_scanned: int = 0
    notebook_failures: int = 0
    code_cells_scanned: int = 0
    cell_parse_failures: int = 0
    total_calls: int = 0
    provenance_resolved_calls: int = 0
    annotation_calls: int = 0
    annotation_provenance_matches: int = 0

def extract_root_libraries(source: str) -> set[str]:
    """Extract normalized root packages from one parseable Python source string."""
    tree = ast.parse(source)
    libraries: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            libraries.update(alias.name.split(".", 1)[0].strip().casefold() for alias in node.names if alias.name.strip())
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            libraries.add(node.module.split(".", 1)[0].strip().casefold())
    return libraries

def discover_dataset_libraries(notebooks_dir: Path) -> DatasetLibraryInventory:
    """Return the dataset knowledge profile without its per-notebook lookup map."""
    inventory, _ = profile_dataset_knowledge(notebooks_dir)
    return inventory

def profile_dataset_knowledge(notebooks_dir: Path, annotations_dir: Path | None = None) -> tuple[DatasetLibraryInventory, dict[str, dict[str, str]]]:
    """Extract observed library/API pairs and resolvable call provenance from notebooks."""
    sanitizer = IPythonSanitizerMixin()
    libraries: set[str] = set()
    library_api_pairs: set[tuple[str, str]] = set()
    notebook_call_provenance: dict[str, dict[str, str]] = {}
    inventory = DatasetLibraryInventory(libraries=[], library_api_pairs=[])

    if annotations_dir is None:
        notebook_paths = sorted(notebooks_dir.glob("*.ipynb"))
    else:
        notebook_paths = [notebooks_dir / annotation.with_suffix(".ipynb").name for annotation in sorted(annotations_dir.glob("*.json"))]

    for notebook_path in notebook_paths:
        if not notebook_path.exists():
            inventory.notebook_failures += 1
            continue
        inventory.notebooks_scanned += 1
        try:
            notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
            cells = notebook.get("cells", []) if isinstance(notebook, dict) else []
        except Exception as exc:
            inventory.notebook_failures += 1
            logger.warning("Could not inspect imports in %s: %s", notebook_path, exc)
            continue

        code_cells = []
        for cell in cells:
            if not isinstance(cell, dict) or cell.get("cell_type") != "code":
                continue
            normalized_cell = dict(cell)
            source = normalized_cell.get("source", "")
            if isinstance(source, list):
                normalized_cell["source"] = "".join(str(line) for line in source)
            code_cells.append(normalized_cell)
        parsed_cells: list[tuple[str, ast.AST]] = []
        for cell in sanitizer.sanitize_ipython_cells(code_cells):
            inventory.code_cells_scanned += 1
            source = cell.get("source", "")
            try:
                parsed_cells.append((str(source), ast.parse(str(source))))
            except SyntaxError:
                inventory.cell_parse_failures += 1

        aliases: dict[str, str] = {}
        for _, tree in parsed_cells:
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        library = alias.name.split(".", 1)[0].strip().casefold()
                        if library:
                            aliases[alias.asname or alias.name.split(".", 1)[0]] = library
                            libraries.add(library)
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    library = node.module.split(".", 1)[0].strip().casefold()
                    libraries.add(library)
                    for alias in node.names:
                        aliases[alias.asname or alias.name] = library

        provenance_by_code: dict[str, str] = {}
        for source, tree in parsed_cells:
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                inventory.total_calls += 1
                library = _resolve_call_library(node.func, aliases)
                api_name = _call_api_name(node.func)
                if not (library and api_name):
                    continue
                inventory.provenance_resolved_calls += 1
                pair = (library, api_name.casefold())
                library_api_pairs.add(pair)
                code = ast.get_source_segment(source, node)
                if code:
                    provenance_by_code[code.strip()] = library
        notebook_call_provenance[notebook_path.with_suffix(".json").name] = provenance_by_code

    inventory.libraries = sorted(libraries)
    inventory.library_api_pairs = sorted(library_api_pairs)
    return inventory, notebook_call_provenance

def _resolve_call_library(node: ast.AST, aliases: dict[str, str]) -> str | None:
    if isinstance(node, ast.Name):
        return aliases.get(node.id)
    if isinstance(node, ast.Attribute):
        return _resolve_call_library(node.value, aliases)
    if isinstance(node, ast.Call):
        return _resolve_call_library(node.func, aliases)
    if isinstance(node, ast.Subscript):
        return _resolve_call_library(node.value, aliases)
    return None

def _call_api_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None

def _extract_api_name(code: str) -> str | None:
    """Extract the called API name from a CALL-node code snippet."""
    try:
        tree = ast.parse(code.strip(), mode="eval")
    except (SyntaxError, ValueError):
        return None
    body = tree.body
    if isinstance(body, ast.Call):
        if isinstance(body.func, ast.Name):
            return body.func.id
        if isinstance(body.func, ast.Attribute):
            return body.func.attr
    return None

def _gold_label_int(value: object) -> int | None:
    """Normalize an annotation domain_label (int or name) to a labelled class id."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if value is None:
        return None
    code = DOMAIN_EDGE_TYPES.get(str(value).upper())
    return code if code is not None and code >= 0 else None

def _record_annotation_provenance_coverage(
    inventory: DatasetLibraryInventory,
    annotations_dir: Path,
    notebook_call_provenance: dict[str, dict[str, str]],
) -> None:
    for annotation_path in annotations_dir.glob("*.json"):
        try:
            raw = json.loads(annotation_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        elements = raw if isinstance(raw, list) else raw.get("elements", [])
        provenance_by_code = notebook_call_provenance.get(annotation_path.name, {})
        for element in elements:
            data = element.get("data", {})
            if data.get("node_type") != 9:
                continue
            inventory.annotation_calls += 1
            if str(data.get("code", "")).strip() in provenance_by_code:
                inventory.annotation_provenance_matches += 1

def kb_libraries(kb_csv: Path) -> set[str]:
    """Distinct (case-folded) libraries covered by a Vamsa knowledge-base CSV."""
    libraries: set[str] = set()
    with kb_csv.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            library = str(row.get("Library", "")).strip().casefold()
            if library:
                libraries.add(library)
    return libraries

def build_learned_kb_from_annotations(
    annotations_dir: Path,
    notebook_call_provenance: dict[str, dict[str, str]],
) -> tuple[dict[VamsaEntry, DomainEdgeId], dict]:
    """Learn a (library, API) -> domain label KB from a corpus's gold annotations."""
    label_counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for annotation_path in sorted(annotations_dir.glob("*.json")):
        try:
            raw = json.loads(annotation_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        elements = raw if isinstance(raw, list) else raw.get("elements", [])
        nx_G = annotation_to_networkx(elements)
        provenance_by_code = notebook_call_provenance.get(annotation_path.name, {})
        for _, data in nx_G.nodes(data=True):
            if data.get("node_type") not in COMPUTE_HUBS:
                continue
            gold = _gold_label_int(data.get("domain_label"))
            if gold is None:
                continue
            code = str(data.get("code", "")).strip()
            library = provenance_by_code.get(code)
            api_name = _extract_api_name(code)
            if library and api_name:
                label_counts[(library.casefold(), api_name)][gold] += 1

    mapping: dict[VamsaEntry, DomainEdgeId] = {}
    for (library, api_name), counter in label_counts.items():
        label = counter.most_common(1)[0][0]
        entry = VamsaEntry(library=library, module="", caller="", api_name=api_name, inputs=(), outputs=())
        mapping[entry] = DomainEdgeId(label)

    summary = {
        "pairs": len(mapping),
        "libraries": len({library for library, _ in label_counts}),
        "apis": len({api_name for _, api_name in label_counts}),
        "labelled_call_observations": int(sum(sum(counter.values()) for counter in label_counts.values())),
    }
    return mapping, summary

def _library_group(library: str | None, shared: set[str], unique: set[str]) -> str:
    if not library:
        return _GROUP_UNRESOLVED
    folded = library.casefold()
    if folded in shared:
        return _GROUP_SHARED
    if folded in unique:
        return _GROUP_UNIQUE
    return _GROUP_UNRESOLVED

def _cm_metrics(cm: ConfusionMatrix) -> dict:
    return {
        "precision": cm.precision,
        "recall": cm.recall,
        "f1": cm.f1_score,
        "support": cm.tp + cm.fn,
        "tp": cm.tp,
        "fp": cm.fp,
        "fn": cm.fn,
    }

def evaluate_labeler(
    labeler: VamsaStaticLabeler,
    eval_paths: list[Path],
    notebook_call_provenance: dict[str, dict[str, str]] | None = None,
    inject_provenance: bool = False,
    library_groups: tuple[set[str], set[str]] | None = None,
) -> dict:
    """Run a Vamsa static labeler over annotation graphs (no refinement)."""
    global_cm = ConfusionMatrix()
    group_cm: dict[str, ConfusionMatrix] = {group: ConfusionMatrix() for group in _GROUP_ORDER}
    group_covered: dict[str, int] = dict.fromkeys(_GROUP_ORDER, 0)
    failures = 0

    for json_path in sorted(eval_paths):
        try:
            raw = json.loads(json_path.read_text(encoding="utf-8"))
            elements = raw if isinstance(raw, list) else raw.get("elements", [])
            nx_G = annotation_to_networkx(elements)
            if nx_G.number_of_nodes() == 0:
                continue

            provenance_by_code = (notebook_call_provenance or {}).get(json_path.name, {})
            if inject_provenance:
                for _, node_data in nx_G.nodes(data=True):
                    library = provenance_by_code.get(str(node_data.get("code", "")).strip())
                    if library:
                        node_data["base_inputs"] = library

            wrapper = _GraphWrapper(nx_G)
            golden = wrapper.golden_labels()
            if not golden:
                continue

            node_library = {
                node_id: provenance_by_code.get(str(nx_G.nodes[node_id].get("code", "")).strip())
                for node_id in golden
            }

            for node_id in list(nx_G.nodes):
                nx_G.nodes[node_id].pop("domain_label", None)
                nx_G.nodes[node_id].pop("predicted_label", None)

            labeler.apply_labels(wrapper)
            predicted = wrapper.predicted_labels()

            for node_id, gold in golden.items():
                pred = predicted.get(node_id)
                if pred == gold:
                    global_cm.tp += 1
                elif pred is None:
                    global_cm.fn += 1
                else:
                    global_cm.fp += 1
                    global_cm.fn += 1
                if library_groups is not None:
                    group = _library_group(node_library.get(node_id), *library_groups)
                    cm = group_cm[group]
                    if pred == gold:
                        cm.tp += 1
                    elif pred is None:
                        cm.fn += 1
                    else:
                        cm.fp += 1
                        cm.fn += 1
                    if pred is not None:
                        group_covered[group] += 1
        except Exception as exc:
            logger.warning("Vamsa generalization eval failed %s: %s", json_path.name, exc)
            failures += 1

    result = {"global": _cm_metrics(global_cm), "failures": failures}
    if library_groups is not None:
        result["per_group"] = {
            group: {**_cm_metrics(group_cm[group]), "predicted": group_covered[group]}
            for group in _GROUP_ORDER
        }
    return result

def _load_published_labeler(kb_csv: Path) -> VamsaStaticLabeler:
    mapping = VamsaKBLoader(IOSignatureMappingPolicy()).load_and_map(kb_csv)
    return VamsaStaticLabeler(mapping, strict_provenance=False)

def plot_generalization(results: dict, output_dir: Path) -> list[str]:
    """Render the KB-coverage bar chart and the learned-KB transfer matrix."""
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    outputs.extend(_plot_coverage(results, output_dir))
    outputs.extend(_plot_transfer_matrix(results, output_dir))
    return outputs

def _plot_transfer_matrix(results: dict, output_dir: Path) -> list[str]:
    """Render the 2×2 learned-KB cross-dataset transfer heatmap."""
    import numpy as np

    datasets = list(GENERALIZATION_DATASETS)
    runs = results.get("learned_kb_transfer", {}).get("runs", {})
    matrix = np.full((len(datasets), len(datasets)), np.nan)
    for row, learned_from in enumerate(datasets):
        for column, evaluated_on in enumerate(datasets):
            run = runs.get(f"{learned_from}__{evaluated_on}")
            if run is not None:
                matrix[row, column] = run["global"]["f1"]

    labels = [DATASET_LABELS[dataset] for dataset in datasets]
    title, subtitle = _vamsa_figure_text(
        "fig_vamsa_generalization",
        "Learned-KB Cross-Dataset Transfer",
        "KB learned from each corpus's gold annotations; raw static-labeler F1, no refinement.",
    )
    try:
        from SystemX.experiment.ablation.plot_results import render_generalization_matrix

        paths = render_generalization_matrix(
            matrix,
            labels,
            output_dir,
            "fig_vamsa_generalization",
            title,
            subtitle,
            show_diagonal=True,
            show_deltas=True,
            y_axis_label="KB learned from",
            y_labels=labels,
        )
        return [str(Path(path).resolve()) for path in paths]
    except Exception as exc:
        logger.warning("Ablation matrix renderer unavailable (%s); using the Pillow renderer.", exc)
        return _pillow_matrix(matrix.tolist(), labels, labels, title, subtitle, output_dir, "fig_vamsa_generalization")

def _pillow_matrix(matrix: list[list[float]], row_labels: list[str], col_labels: list[str], title: str, subtitle: str, output_dir: Path, stem: str) -> list[str]:
    """Dependency-safe heatmap renderer used when matplotlib is unavailable."""
    finite = [value for row in matrix for value in row if value is not None and value == value]
    finite = finite or [0.0]
    vmin = max(0.0, (min(finite) // 0.05) * 0.05 - 0.05)
    vmax = min(1.0, -(-max(finite) // 0.05) * 0.05 + 0.05)
    if vmax - vmin < 0.1:
        vmin, vmax = max(0.0, vmin - 0.05), min(1.0, vmax + 0.05)

    title_font = _font("DejaVuSerif-Bold.ttf", 44)
    subtitle_font = _font("DejaVuSerif.ttf", 24)
    label_font = _font("DejaVuSerif.ttf", 30)
    axis_font = _font("DejaVuSerif-Bold.ttf", 32)
    score_font = _font("DejaVuSerif.ttf", 38)

    n_rows, n_cols = len(matrix), len(col_labels)
    cell = 360
    left, top = 520, 260
    width = left + n_cols * cell + 220
    height = top + n_rows * cell + 260
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((width // 2, 50), title, fill="black", font=title_font, anchor="ma")
    draw.text((width // 2, 120), subtitle, fill="black", font=subtitle_font, anchor="ma")

    matrix_height = n_rows * cell
    for index, label in enumerate(col_labels):
        draw.text((left + index * cell + cell // 2, top + matrix_height + 30), label, fill="black", font=label_font, anchor="ma")
    draw.text((left + n_cols * cell // 2, top + matrix_height + 100), "Evaluate on", fill="black", font=axis_font, anchor="ma")
    for index, label in enumerate(row_labels):
        draw.text((left - 30, top + index * cell + cell // 2), label, fill="black", font=label_font, anchor="rm")

    for row in range(n_rows):
        for column in range(n_cols):
            value = matrix[row][column]
            x0, y0 = left + column * cell, top + row * cell
            missing = value is None or value != value
            color = _paper_heatmap_color(vmin if missing else value, vmin, vmax)
            draw.rectangle((x0, y0, x0 + cell, y0 + cell), fill=color, outline="white", width=6)
            if row == column and not missing:
                draw.rectangle((x0 + 4, y0 + 4, x0 + cell - 4, y0 + cell - 4), outline="white", width=4)
            text = "-" if missing else f"{value:.3f}"
            text_color = "white" if missing or value < (vmin + vmax) / 2 else "black"
            draw.text((x0 + cell // 2, y0 + cell // 2), text, fill=text_color, font=score_font, anchor="mm")

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [output_dir / f"{stem}.pdf", output_dir / f"{stem}.png"]
    canvas.save(outputs[0], "PDF", resolution=300)
    canvas.save(outputs[1], dpi=(300, 300))
    return [str(output.resolve()) for output in outputs]

def _vamsa_figure_text(fig_id: str, default_title: str, default_subtitle: str) -> tuple[str, str]:
    """Resolve a figure's (title, subtitle) from figure_titles.yaml, else defaults."""
    try:
        from SystemX.experiment.ablation.plot_results import fig_title

        return fig_title(fig_id, "title", default_title), fig_title(fig_id, "subtitle", default_subtitle)
    except Exception:
        return default_title, default_subtitle

def _plot_coverage(results: dict, output_dir: Path) -> list[str]:
    """Grouped bar chart (paper style): recall on shared vs."""
    coverage = results.get("coverage_analysis", {})
    datasets = [dataset for dataset in GENERALIZATION_DATASETS if dataset in coverage]
    if not datasets:
        return []

    title, subtitle = _vamsa_figure_text(
        "fig_vamsa_kb_coverage",
        "Vamsa KB Coverage of the Library Long Tail",
        "Raw static-labeler recall by library novelty; the published KB covers only mainstream shared libraries.",
    )
    groups = (_GROUP_SHARED, _GROUP_UNIQUE)
    group_labels = ["Shared libraries", "Corpus-unique libraries"]
    dataset_labels = [DATASET_LABELS[dataset] for dataset in datasets]

    def _stat(dataset: str, group: str, field: str, cast):
        return cast(coverage[dataset].get("per_group", {}).get(group, {}).get(field, 0) or 0)

    recall = [[_stat(dataset, group, "recall", float) for group in groups] for dataset in datasets]
    support = [[_stat(dataset, group, "support", int) for group in groups] for dataset in datasets]

    try:
        from SystemX.experiment.ablation.plot_results import render_coverage_bars

        paths = render_coverage_bars(
            dataset_labels,
            group_labels,
            recall,
            support,
            output_dir,
            "fig_vamsa_kb_coverage",
            title,
            subtitle,
            colors=["#1f3a5f", "#b0c4de"],
            y_label="Recall",
        )
        return [str(Path(path).resolve()) for path in paths]
    except Exception as exc:
        logger.warning("Ablation bar renderer unavailable (%s); using the Pillow renderer.", exc)
        return _plot_coverage_pillow(results, output_dir, title, subtitle)

def _plot_coverage_pillow(results: dict, output_dir: Path, title: str, subtitle: str) -> list[str]:
    """Dependency-safe grouped bar chart used when matplotlib is unavailable."""
    coverage = results.get("coverage_analysis", {})
    datasets = [dataset for dataset in GENERALIZATION_DATASETS if dataset in coverage]
    if not datasets:
        return []

    title_font = _font("DejaVuSerif-Bold.ttf", 44)
    subtitle_font = _font("DejaVuSerif.ttf", 24)
    label_font = _font("DejaVuSerif.ttf", 28)
    axis_font = _font("DejaVuSerif-Bold.ttf", 30)
    value_font = _font("DejaVuSerif.ttf", 26)
    legend_font = _font("DejaVuSerif.ttf", 26)

    group_colors = {_GROUP_SHARED: (31, 58, 95), _GROUP_UNIQUE: (176, 196, 222)}
    group_names = {_GROUP_SHARED: "Shared libraries", _GROUP_UNIQUE: "Corpus-unique libraries"}
    plotted_groups = (_GROUP_SHARED, _GROUP_UNIQUE)

    width, height = 1500, 950
    left, right, top, bottom = 150, 90, 230, 220
    plot_w = width - left - right
    plot_h = height - top - bottom
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((width // 2, 45), title, fill="black", font=title_font, anchor="ma")
    draw.text((width // 2, 110), subtitle, fill="black", font=subtitle_font, anchor="ma")

    for tick in range(0, 11, 2):
        value = tick / 10
        y = top + plot_h - value * plot_h
        draw.line((left, y, left + plot_w, y), fill=(220, 220, 220), width=2)
        draw.text((left - 15, y), f"{value:.1f}", fill="black", font=value_font, anchor="rm")
    draw.text((45, top + plot_h // 2), "Recall", fill="black", font=axis_font, anchor="mm")
    draw.line((left, top, left, top + plot_h), fill="black", width=3)
    draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill="black", width=3)

    slot = plot_w / len(datasets)
    bar_w = slot / (len(plotted_groups) + 1.5)
    for d_index, dataset in enumerate(datasets):
        per_group = coverage[dataset].get("per_group", {})
        base_x = left + d_index * slot + (slot - bar_w * len(plotted_groups)) / 2
        for g_index, group in enumerate(plotted_groups):
            stats = per_group.get(group, {})
            recall = float(stats.get("recall", 0.0) or 0.0)
            support = int(stats.get("support", 0) or 0)
            x0 = base_x + g_index * bar_w
            x1 = x0 + bar_w * 0.9
            y1 = top + plot_h
            y0 = y1 - recall * plot_h
            draw.rectangle((x0, y0, x1, y1), fill=group_colors[group])
            draw.text(((x0 + x1) / 2, y0 - 8), f"{recall:.2f}", fill="black", font=value_font, anchor="mb")
            draw.text(((x0 + x1) / 2, y1 + 10), f"n={support}", fill=(90, 90, 90), font=value_font, anchor="ma")
        draw.text((left + d_index * slot + slot / 2, top + plot_h + 60), DATASET_LABELS[dataset], fill="black", font=label_font, anchor="ma")

    legend_y = height - 55
    legend_x = left
    for group in plotted_groups:
        draw.rectangle((legend_x, legend_y, legend_x + 34, legend_y + 34), fill=group_colors[group])
        draw.text((legend_x + 46, legend_y + 17), group_names[group], fill="black", font=legend_font, anchor="lm")
        legend_x += 46 + draw.textlength(group_names[group], font=legend_font) + 70

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [output_dir / "fig_vamsa_kb_coverage.pdf", output_dir / "fig_vamsa_kb_coverage.png"]
    canvas.save(outputs[0], "PDF", resolution=300)
    canvas.save(outputs[1], dpi=(300, 300))
    return [str(output.resolve()) for output in outputs]

def _font(name: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()

def _paper_heatmap_color(value: float, vmin: float, vmax: float) -> tuple[int, int, int]:
    stops = [(31, 58, 95), (63, 99, 145), (127, 160, 196), (176, 196, 222), (220, 230, 241)]
    position = 0.0 if vmax <= vmin else max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))
    scaled = position * (len(stops) - 1)
    lower = min(int(scaled), len(stops) - 2)
    fraction = scaled - lower
    return tuple(round(stops[lower][channel] + fraction * (stops[lower + 1][channel] - stops[lower][channel])) for channel in range(3))

def render_saved_results(output_path: Path, figure_dir: Path) -> dict:
    """Render figures from an already completed experiment without reevaluation."""
    if not output_path.exists():
        raise FileNotFoundError(f"Vamsa generalization results not found: {output_path}")
    results = json.loads(output_path.read_text(encoding="utf-8"))
    results["figures"] = plot_generalization(results, figure_dir)
    results.pop("figure_error", None)
    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    logger.info("Rendered Vamsa generalization figures from saved results in %s", output_path)
    return results

def run_experiment(kb_csv: Path, output_path: Path, kb_output_dir: Path, figure_dir: Path) -> dict:
    """Run the Vamsa KB coverage analysis and the learned-KB transfer experiment."""
    kb_csv = kb_csv.resolve()
    inventories: dict[str, DatasetLibraryInventory] = {}
    dataset_call_provenance: dict[str, dict[str, dict[str, str]]] = {}
    dataset_paths: dict[str, tuple[Path, Path]] = {}
    for dataset in GENERALIZATION_DATASETS:
        annotations_dir, notebooks_dir = resolve_dataset(dataset)
        dataset_paths[dataset] = (annotations_dir, notebooks_dir)
        inventories[dataset], dataset_call_provenance[dataset] = profile_dataset_knowledge(notebooks_dir, annotations_dir)
        _record_annotation_provenance_coverage(inventories[dataset], annotations_dir, dataset_call_provenance[dataset])

    library_sets = {dataset: set(inventory.libraries) for dataset, inventory in inventories.items()}
    jetbrains_only = library_sets["jetbrains"] - library_sets["github"]
    github_only = library_sets["github"] - library_sets["jetbrains"]
    shared = library_sets["jetbrains"] & library_sets["github"]
    unique_by_dataset = {"jetbrains": jetbrains_only, "github": github_only}

    published_libraries = kb_libraries(kb_csv)

    results = {
        "experiment": "vamsa_static_cross_dataset_generalization",
        "graph_refinement": False,
        "vamsa_strict_provenance": False,
        "kb_source": str(kb_csv),
        "datasets": {dataset: asdict(inventory) for dataset, inventory in inventories.items()},
        "library_comparison": {
            "shared": sorted(shared),
            "github_only": sorted(github_only),
            "jetbrains_only": sorted(jetbrains_only),
        },
        "kb_library_coverage": {
            "n_kb_libraries": len(published_libraries),
            "kb_libraries": sorted(published_libraries),
            "per_dataset": {
                dataset: {
                    "shared_covered": len(shared & published_libraries),
                    "shared_total": len(shared),
                    "unique_covered": len(unique_by_dataset[dataset] & published_libraries),
                    "unique_total": len(unique_by_dataset[dataset]),
                }
                for dataset in GENERALIZATION_DATASETS
            },
        },
        "coverage_analysis": {},
        "learned_kb_transfer": {
            "note": "KB learned from each corpus's gold annotations; NOT the published Vamsa KB.",
            "learned_kbs": {},
            "runs": {},
        },
        "figures": [],
    }

    published_labeler = _load_published_labeler(kb_csv)
    for dataset in GENERALIZATION_DATASETS:
        annotations_dir, _ = dataset_paths[dataset]
        logger.info("Coverage analysis: published Vamsa KB (non-strict) on %s", dataset)
        evaluation = evaluate_labeler(
            published_labeler,
            sorted(annotations_dir.glob("*.json")),
            notebook_call_provenance=dataset_call_provenance[dataset],
            inject_provenance=False,
            library_groups=(shared, unique_by_dataset[dataset]),
        )
        evaluation["kb_library_coverage"] = results["kb_library_coverage"]["per_dataset"][dataset]
        results["coverage_analysis"][dataset] = evaluation

    kb_output_dir.mkdir(parents=True, exist_ok=True)
    learned_labelers: dict[str, VamsaStaticLabeler] = {}
    for source_dataset in GENERALIZATION_DATASETS:
        annotations_dir, _ = dataset_paths[source_dataset]
        mapping, summary = build_learned_kb_from_annotations(annotations_dir, dataset_call_provenance[source_dataset])
        learned_labelers[source_dataset] = VamsaStaticLabeler(mapping, strict_provenance=False)
        results["learned_kb_transfer"]["learned_kbs"][source_dataset] = summary
        _dump_learned_kb(mapping, kb_output_dir / f"learned_kb_{source_dataset}.json")

    for source_dataset in GENERALIZATION_DATASETS:
        for eval_dataset in GENERALIZATION_DATASETS:
            annotations_dir, _ = dataset_paths[eval_dataset]
            logger.info("Learned-KB transfer: learned_from=%s evaluate=%s", source_dataset, eval_dataset)
            evaluation = evaluate_labeler(
                learned_labelers[source_dataset],
                sorted(annotations_dir.glob("*.json")),
                notebook_call_provenance=dataset_call_provenance[eval_dataset],
                inject_provenance=True,
            )
            results["learned_kb_transfer"]["runs"][f"{source_dataset}__{eval_dataset}"] = {
                "learned_from": source_dataset,
                "evaluated_on": eval_dataset,
                "global": evaluation["global"],
                "failures": evaluation["failures"],
            }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    try:
        results["figures"] = plot_generalization(results, figure_dir)
    except Exception as exc:
        results["figure_error"] = str(exc)
        logger.exception("Could not render Vamsa generalization figures; metrics remain saved in %s", output_path)
    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote Vamsa generalization results to %s", output_path)
    return results

def _dump_learned_kb(mapping: dict[VamsaEntry, DomainEdgeId], output_path: Path) -> None:
    """Persist a learned KB as human-readable JSON for inspection/transparency."""
    id_to_name = {uid: data["name"] for uid, data in DOMAIN_EDGES.items()}
    rows = [
        {"library": entry.library, "api_name": entry.api_name, "domain_label": id_to_name.get(int(label), str(label))}
        for entry, label in sorted(mapping.items(), key=lambda item: (item[0].library, item[0].api_name))
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
