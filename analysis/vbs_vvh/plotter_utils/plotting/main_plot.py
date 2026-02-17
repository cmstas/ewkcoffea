# plotter_utils/plotting/main_plot.py

import numpy as np
from plotter_utils.helpers.plotting_funcs import snap_to_decade,format_variable_label
from plotter_utils.plotting.plot_settings import PLOT_SETTINGS

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
    Scale signal histogram to background yield using a * 10^n factor.
    Returns (scaled_hist, scale_factor).
    """
    if sig is None or bkg is None:
        return sig.hist, 1.0

    if mode == "match_bkg":
        s = sig.total_yield()
        b = bkg.total_yield()

        if s > 0 and b > 0:
            raw_factor = b / s
            factor = snap_to_decade(raw_factor)
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
        sig_plot, sig_scale = get_scaled_signal(sig, bkg)
        if sig_scale==1: sig_label = 'Signal'
        elif sig_scale<1000 and sig_scale>0.01: sig_label = f'Signal × {sig_scale:.1f}'
        else: sig_label=f"Signal × {sig_scale:.1e}"


        sig_plot.plot1d(
            ax=ax,
            color="red",
            linewidth=2,
            label=sig_label,
            zorder=101,
        )

    # ---- Data ----
    if data is not None:
        data_plot, data_scale = get_scaled_signal(data, bkg)
        if data_scale==1: data_label = 'Data'
        elif data_scale<1000 and data_scale>0.01: data_label = f'Data × {data_scale:.1f}'
        else: data_label=f"Data × {data_scale:.1e}"
        data_plot.plot1d(
            ax=ax,
            color="blue",
            label=data_label,
            zorder=102,
        )
    ax.set_xlabel("")
    handles, labels = ax.get_legend_handles_labels()

    # split into two groups
    order_set = set(PLOT_SETTINGS.get("legend_order", []))
    group1 = [(h, l) for h, l in zip(handles, labels) if l.split()[0] in order_set]
    group2 = [(h, l) for h, l in zip(handles, labels) if l.split()[0] not in order_set]

    # unpack
    h1, l1 = zip(*group1) if group1 else ([], [])
    h2, l2 = zip(*group2) if group2 else ([], [])
    
    if "legend_order" in PLOT_SETTINGS:
        order_dict = {label: i for i, label in enumerate(PLOT_SETTINGS["legend_order"])}
        sorted_pairs = sorted(zip(handles, labels), key=lambda pair: order_dict.get(pair[1], 999))
        handles, labels = zip(*sorted_pairs)
    ax.legend(handles, labels, fontsize=PLOT_SETTINGS.get("legend_fontsize", 10), ncol=PLOT_SETTINGS.get("legend_ncol", PLOT_SETTINGS['ncol_legend']), frameon=False)

    ax.set_ylabel("Events")
    # -----------------------------
    # X-axis label (trying to fix)
    # -----------------------------
    # ref_hist = (
    #     sig.hist if sig is not None
    #     else bkg.hist if bkg is not None
    #     else data.hist
    # )

    # if ref_hist is not None:
    #     xlabel = format_variable_label(ref_hist)
    #     ax.set_xlabel(xlabel)
    
