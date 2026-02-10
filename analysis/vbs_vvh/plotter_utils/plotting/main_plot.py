# plotter_utils/plotting/main_plot.py

import numpy as np

import logging
logger = logging.getLogger(__name__)


def order_backgrounds(bkg, mode="yield"):
    """
    Order background process groups.
    """
    if mode == "yield":
        yields = {
            p: bkg.hist[{"process_grp": [p]}].values(flow=True).sum()
            for p in bkg.hist.axes["process_grp"]
        }
        return sorted(yields, key=yields.get)
    return list(bkg.hist.axes["process_grp"])


def get_scaled_signal(sig, bkg, mode="match_bkg"):
    """
    Scale signal to background yield for overlay.
    """
    if sig is None or bkg is None:
        return sig.hist, 1.0

    if mode == "match_bkg":
        s = sig.total_yield()
        b = bkg.total_yield()
        if s > 0:
            factor = b / s
            return sig.hist * factor, factor

    return sig.hist, 1.0


def draw_bkg_uncertainty(ax, bkg):
    """
    Draw MC statistical uncertainty band.
    """
    htot = bkg.hist_total
    logger.debug(f'bkg: {bkg.hist.shape}')
    logger.debug(f'htot: {htot.shape}')

    vals = htot.values(flow=False)
    errs = np.sqrt(htot.variances(flow=False))

    edges = htot.axes[-1].edges
    err_up = np.append(vals + errs, 0)
    err_dn = np.append(vals - errs, 0)
    logger.debug(f'vals: {vals.shape}\n errs {errs.shape}\n vals[-1] {vals[-1]}')
    
    logger.debug(f'edges: {edges.shape}')
    logger.debug(f'errup/dn: {err_up.shape} {err_dn.shape}')

    ax.fill_between(
        edges,
        err_dn,
        err_up,
        step="post",
        facecolor="none",
        edgecolor="gray",
        hatch="////",
        linewidth=0.0,
        alpha=0.5,
        label="MC stat",
    )


def draw_main_plot(ax, *, sig=None, bkg=None, data=None, proc_map=None):
    """
    Draw the main stacked distribution panel.
    """

    # ---- Background stack ----
    if bkg is not None:
        order = order_backgrounds(bkg)
        colors = [proc_map.background_colors[p] for p in order]

        bkg.hist[{"process_grp": order}].plot1d(
            ax=ax,
            stack=True,
            histtype="fill",
            color=colors,
            label=order,
            zorder=100,
        )

        draw_bkg_uncertainty(ax, bkg)

    # ---- Signal overlay ----
    if sig is not None:
        sig_plot, scale = get_scaled_signal(sig, bkg)
        sig_plot.plot1d(
            ax=ax,
            color="red",
            linewidth=2,
            label=f"Signal × {scale:.1f}",
            zorder=101,
        )

    # ---- Data ----
    if data is not None:
        data.hist.plot1d(
            ax=ax,
            color="blue",
            label="Data",
            zorder=102,
        )

    ax.legend()
    ax.set_ylabel("Events")
