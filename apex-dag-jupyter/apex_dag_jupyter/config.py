from traitlets.config import Configurable
from traitlets import Unicode


class APEXDAGConfig(Configurable):
    """
    Configuration for the APEX-DAG Jupyter Extension.
    """

    config_path = Unicode(
        "", help="The path to the APEX-DAG experiment configuration YAML file."
    ).tag(config=True)

    model_path = Unicode(
        "", help="The path to the APEX-DAG pre-trained model file (.pt)."
    ).tag(config=True)
