# plotter_utils/plotting/main_plot.py

import logging
log = logging.getLogger(__name__)

def order_backgrounds(bkg, mode="yield"):
    """
    Get bkg proc type order by yield.
    """
    if mode == "yield":
        yields = {
            p: bkg.hist[{"process_grp": [p]}].values(flow=True).sum()
            for p in bkg.hist.axes["process_grp"]
        }
        logging.debug(f"in main_plot.order_bkg: {yields}")
        return sorted(yields, key=yields.get, reverse=False)
    
def get_scaled_signal(sig, bkg, mode="match_bkg"):
    """
    Scale signal to bkg 
    """
    if mode == "match_bkg":
        s = sig.total_yield()
        b = bkg.total_yield()
        if s > 0:
            factor = b / s
            return sig.hist * factor, factor
    return sig.hist, 1.0

def draw_main_plot(ax, *, sig=None, bkg=None, data=None, proc_map=None):
    """
    Draw the main kinematic distribution.
    """

    if bkg is not None:
        order = order_backgrounds(bkg)
        colors = [
            proc_map.background_colors[p]
            for p in order
        ]
        logging.debug(f"in draw_main_plot: order: {order}")
        logging.debug(f"in draw_main_plot: colors: {colors}")
        logging.debug(f"in draw_main_plot lebel is {list(bkg.hist.axes['process_grp'])}")
        bkg.hist[{"process_grp": order}].plot1d(
            ax=ax,
            stack=True,
            histtype="fill",
            color=colors,
            label=list(bkg.hist.axes["process_grp"]),
        )
    else:
        log.warning("Background histogram missing")

    if sig is not None:
        sig_plot, scale = get_scaled_signal(sig, bkg)
        sig_plot.plot1d(
            ax=ax,
            color="red",
            linewidth=2,
            label=f"Signal × {scale:.1f}",
        )
    logging.debug(f"in draw main plot data is {data.hist}")
    if data is not None:
        data.hist.plot1d(
            ax=ax,
            color="blue",
            linewidth=2,
            label="Data",
        )

    ax.legend()
