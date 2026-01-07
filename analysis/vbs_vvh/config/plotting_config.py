# config/plotting_config.py

PROC_MAP_CSV = "config/sample_names.csv"
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

SUBPLOTS = ("dataMC", "significance")

FIG_RATIO = {
    "main": 4,
    "dataMC": 1,
    "significance": 1,
}
