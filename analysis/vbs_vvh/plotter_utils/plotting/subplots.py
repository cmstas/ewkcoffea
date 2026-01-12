# plotter_utils/plotting/subplots.py

import logging
logger = logging.getLogger(__name__)

def draw_metric_subplot(ax, metric, metric_name, line_colors):
    ok, missing = metric.available()
    if metric.y_range:
        ax.set_ylim(*metric.y_range)

    if not ok:
        msg = f"{metric_name}: {missing} not available"
        logger.warning(msg)
        ax.text(
            0.5, 0.5, msg,
            ha="center", va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return

    result = metric.compute()

    if isinstance(result.y, dict):
        for label, yvals in result.y.items():
            ax.step(result.x, yvals, where="mid", label=label,color=line_colors[label])
        ax.legend()
    else:
        if result.yerr is not None:
            ax.errorbar(result.x, result.y, yerr=result.yerr, fmt="o",color=line_colors["General"])
        else:
            ax.step(result.x, result.y,color=line_colors["General"])

    ax.set_ylabel(result.label)
