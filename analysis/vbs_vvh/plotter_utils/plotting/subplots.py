# plotter_utils/plotting/subplots.py

import logging

log = logging.getLogger(__name__)


def draw_metric_subplot(ax, metric, metric_name):
    ok, missing = metric.available()
    if metric.y_range:
        ax.set_ylim(*metric.y_range)

    logging.debug(f"in draw_metric_subplot we have\n {metric_name}: {metric.compute().x}\n{metric.compute().y}")

    if not ok:
        msg = f"{metric_name}: {missing} not available"
        log.warning(msg)
        ax.text(
            0.5, 0.5, msg,
            ha="center", va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        return

    result = metric.compute()

    if result.yerr is not None:
        ax.errorbar(
            result.x,
            result.y,
            yerr=result.yerr,
            fmt="o",
        )
    else:
        ax.step(result.x, result.y, where="mid")

    ax.axhline(1.0 if metric_name == "dataMC" else 0.0,
               color="gray", linestyle="--")

    ax.set_ylabel(result.label)
