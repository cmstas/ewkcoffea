# plotter_utils/plotting/draw.py

import matplotlib.pyplot as plt
from plotter_utils.metrics.significance import SignificanceMetric
from plotter_utils.metrics.dataMC import DataMCMetric
from .main_plot import draw_main_plot
from .subplots import draw_metric_subplot



METRIC_REGISTRY = {
    "significance": SignificanceMetric,
    "dataMC": DataMCMetric,
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
    subplots = config["SUBPLOTS"]
    ratios = config["FIG_RATIO"]

    heights = [ratios["main"]] + [ratios[s] for s in subplots]
    nrows = 1 + len(subplots)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=1,
        figsize=(7, 3 + 2 * nrows),
        gridspec_kw={"height_ratios": heights},
        sharex=True,
    )

    ax_main = axes[0]
    draw_main_plot(ax_main, sig=sig, bkg=bkg, data=data, proc_map=proc_map)
    ax_main.set_title(title)

    for ax, name in zip(axes[1:], subplots):
        metric_cls = METRIC_REGISTRY[name]
        metric = metric_cls(sig=sig, bkg=bkg, data=data)
        draw_metric_subplot(ax, metric, name)

    return fig
