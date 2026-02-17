# plotter_utils/plotting/draw.py

import matplotlib.pyplot as plt
from plotter_utils.metrics.significance import SignificanceMetric
from plotter_utils.metrics.dataMC import DataMCMetric
from plotter_utils.metrics.pass_rate import PassRateMetric
from plotter_utils.metrics.shape_plot import ShapePlotMetric
from plotter_utils.metrics.efficiency import EfficiencyMetric
from .main_plot import draw_main_plot
from .subplots import draw_metric_subplot
from plotter_utils.plotting.plot_settings import PLOT_SETTINGS, rel_fontsize_calc

import logging
logger = logging.getLogger(__name__)



METRIC_REGISTRY = {
    "significance": SignificanceMetric,
    "dataMC": DataMCMetric,
    "pass_rate": PassRateMetric,
    "shape_plot": ShapePlotMetric,
    "efficiency": EfficiencyMetric,
}


def draw(
    *,
    sig=None,
    bkg=None,
    data=None,
    proc_map=None,
    config=None,
    title=None,
):
    x_size = PLOT_SETTINGS['xsize']
    main_ysize = PLOT_SETTINGS['main_ysize']
    sub_ysize = PLOT_SETTINGS['sub_ysize']
    
    show_title = PLOT_SETTINGS["show_title"]
    rel_xfontsize = PLOT_SETTINGS["rel_xfontsize"]
    rel_yfontsize = PLOT_SETTINGS["rel_yfontsize"]
    rel_label_fontsize = PLOT_SETTINGS["rel_label_fontsize"]
    title_fontsize = PLOT_SETTINGS["title_fontsize"]

    subplots = config.subplots
    ratios = config.fig_ratio
    line_colors = config.line_colors
    logger.debug(f'subplots is {subplots}')

    heights = [ratios["main"]] + [ratios[s] for s in subplots]
    nrows = 1 + len(subplots)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=1,
        figsize=(x_size, main_ysize + sub_ysize * nrows),
        gridspec_kw={"height_ratios": heights},
        sharex=True,
    )

    ax_main = axes[0]
    draw_main_plot(ax_main, sig=sig, bkg=bkg, data=data, proc_map=proc_map)


    #need to move this into each ax later
    # xlabel = ax_main.xaxis.label
    # logger.debug(f'font size is {rel_fontsize_calc(rel_xfontsize,x_size)}')
    # xlabel.set_fontsize(rel_fontsize_calc(rel_xfontsize,x_size))

    ylabel = ax_main.yaxis.label
    logger.debug(f'font size is {rel_fontsize_calc(rel_yfontsize,x_size)}')
    ylabel.set_fontsize(rel_fontsize_calc(rel_yfontsize,x_size))
    ax_main.yaxis.get_offset_text().set_fontsize(rel_fontsize_calc(rel_label_fontsize,x_size))
    ax_main.tick_params(axis="y", labelsize=rel_fontsize_calc(rel_label_fontsize,x_size))

    
    #only adding title if necessary (usually it is the same as x-axis)
    if show_title and title is not None:
        if title_fontsize is not None:
            ax_main.set_title(title, fontsize=title_fontsize)
        else:
            ax_main.set_title(title)

    for ax, name in zip(axes[1:], subplots):
        metric_cls = METRIC_REGISTRY[name]
        metric = metric_cls(sig=sig, bkg=bkg, data=data)
        draw_metric_subplot(ax, metric, name,line_colors=line_colors)

    axes[-1].set_xlabel(title, fontsize=rel_fontsize_calc(rel_xfontsize, x_size))

    return fig
