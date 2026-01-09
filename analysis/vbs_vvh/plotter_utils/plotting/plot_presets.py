from dataclasses import dataclass
from typing import Tuple, Dict


@dataclass(frozen=True)
class PlotConfig:
    subplots: Tuple[str, ...]
    fig_ratio: Dict[str, int]
    line_colors: Dict[str, str]


DEFAULT_COLORS = {
    "Signal": "red",
    "Background": "grey",
    "Data": "blue",

    "General": "black",
}

PLOT_PRESETS = { 
    "default": PlotConfig(
        subplots=("shape_plot", ),
        fig_ratio={
            "main": 3,
            "shape_plot": 1,
        },
        line_colors=DEFAULT_COLORS
    ),

    "datamc_comparison": PlotConfig(
        subplots=("dataMC","shape_plot",),
        fig_ratio={
            "main": 3,
            "dataMC": 1,
            "shape_plot": 1,
        },
        line_colors=DEFAULT_COLORS
    ),

    "cut_study": PlotConfig(
        subplots=("pass_rate", "significance"),
        fig_ratio={
            "main": 3,
            "pass_rate": 1,
            "significance": 1,
        },
        line_colors=DEFAULT_COLORS
    ),
}
preset_list = PLOT_PRESETS.keys()

