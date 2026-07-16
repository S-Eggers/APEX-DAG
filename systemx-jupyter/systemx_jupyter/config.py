from traitlets import Unicode
from traitlets.config import Configurable


class SystemXConfig(Configurable):
    """Configuration for the SystemX Jupyter Extension."""

    v2_checkpoints_dir = Unicode(
        "",
        help=(
            "Directory containing the trained lineage/labeling checkpoints.\n"
            "The extension exposes a (model family x feature preset) matrix and loads\n"
            "one checkpoint per combination. Discovery reads 'manifest.json' in this\n"
            "directory (as written by SystemX.experiment.ablation.train_all), whose\n"
            "keys are '<family>_<preset>' - e.g. hgt_standard, mlp_emb_only,\n"
            "xgboost_struct_only. Families: hgt (FastText), mlp, xgboost. Presets:\n"
            "standard, all, emb_only, api_lib, struct_only.\n"
            "When no manifest.json is present, the extension falls back to globbing\n"
            "'<family>_<preset>*.pt' (hgt/mlp) or '.json' (xgboost) in this directory.\n"
            "If unset, defaults to ./checkpoints/v2 when that directory exists."
        ),
    ).tag(config=True)
