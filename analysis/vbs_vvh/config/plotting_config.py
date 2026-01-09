# config/plotting_config.py

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)

PROC_MAP_CSV = os.path.join(base_dir,"config/sample_names.csv")
REF_VAR = 'nGoodAK8' # var for getting yield and stuff

SIG_COLOR = "red"
DATA_COLOR = "blue"

DEFAULT_OUTPUT_DIRNAME = "plots"

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(message)s"

FIGURE_STYLE = {
    "figsize": (6, 5),
    "dpi": 120,
}

DEFAULT_PLOTTING_PRESET = "default"
