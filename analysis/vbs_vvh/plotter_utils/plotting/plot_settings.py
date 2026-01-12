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

def font_size(fig, frac):
    return frac * fig.get_figwidth() * 72

PLOT_SETTINGS = {
    "xsize": 7,
    "main_ysize":3,
    "sub_ysize":2,

    "show_title": False, #default to false because title is predefned to be the same as x-axis. change if necessary
    "rel_xfontsize": 3, #somehow looks good
    "rel_yfontsize": 2, #somehow looks good
    "rel_label_fontsize": 2,
    "title_fontsize": 12 #will be changed to use rel size. 
}

def rel_fontsize_calc(ratio,x_size):
    return ratio * x_size* 0.8
