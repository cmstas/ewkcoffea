import numpy as np
import matplotlib.pyplot as plt
import os, sys
import logging

import config.plotting_config as CONFIG
from plotter_utils.histo_reader.loader import load_hist_collection, unpack_hist
from plotter_utils.histo_reader.sample_info import ProcessMap
from plotter_utils.helpers.plotting_funcs import snap_to_decade

import matplotlib as mpl

mpl.rcParams.update({
    # Base font size
    "font.size": 14,

    # Axes
    "axes.labelsize": 16,
    "axes.titlesize": 16,

    # Ticks
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,

    # Legend
    "legend.fontsize": 13,
    "legend.title_fontsize": 13,

    # Figure
    "figure.titlesize": 18,
})

# -------------------------------------------------------
# Logging
# -------------------------------------------------------

logging_level = "INFO"

logging.basicConfig(
    level=logging_level,
    format="%(message)s",
)
logger = logging.getLogger(__name__)

# -------------------------------------------------------
# Configuration
# -------------------------------------------------------

ROOT_FILE = [(
    "/home/users/pyli/projects/analysis_VVH/coffea/ewkcoffea/analysis/"
    "vbs_vvh/histos/metpt300_finalized/MET_met300_objsel/"
    "MET_met300_objsel_hists.pkl.gz"
)]

OUTDIR = "./histos/METtrigger_study"

# -------------------------------------------------------
# Helpers
# -------------------------------------------------------

def safe_ratio(num, den):
    out = np.zeros_like(num, dtype=float)
    mask = den > 0
    out[mask] = num[mask] / den[mask]
    return out


def safe_ratio_cumsum(num, den):
    num_s = np.cumsum(num[..., ::-1], axis=-1)[..., ::-1]
    den_s = np.cumsum(den[..., ::-1], axis=-1)[..., ::-1]
    out = np.zeros_like(num_s, dtype=float)
    mask = den_s > 0
    out[mask] = num_s[mask] / den_s[mask]
    return out


def get_scaled(base, target):
    if base is None or target is None:
        return base, 1.0

    s = np.sum(base)
    b = np.sum(target)

    if s > 0 and b > 0:
        factor = snap_to_decade(b / s)
        return base * factor, factor

    return base, 1.0


# -------------------------------------------------------
# Plot helpers
# -------------------------------------------------------

def plot_pertype_dist_ratio(
    x,
    trig_v,
    presel_v,
    all_v,
    outname,
    sample_label,
    var,
    color="gray",
):
    fig, ax = plt.subplots(figsize=(8, 6))

    # --- main distributions ---
    ax.errorbar(x, all_v, linestyle="-.", color=color,
                label=f"{sample_label} (all)")
    ax.errorbar(x, trig_v, linestyle="--", color=color,
                label=f"{sample_label} (triggered)")
    if presel_v is not None:
        ax.errorbar(x, presel_v, linestyle="-", color=color,
                    label=f"{sample_label} (presel)")

    ax.set_xlabel(var)
    if var =='Met_pt_low':
        ax.set_xlabel('Met_pt')
    ax.set_ylabel(f"{sample_label} Events")
    ax.set_yscale("log")

    # --- ratio axis ---
    axr = ax.twinx()

    ratio_trig = safe_ratio_cumsum(trig_v, all_v)
    axr.plot(x, ratio_trig, linestyle="--", color="red",
             linewidth=2, label="trigger / all")

    # if presel_v is not None:
    #     ratio_presel = safe_ratio_cumsum(presel_v, all_v)
    #     axr.plot(x, ratio_presel, linestyle="-", color="red",
    #              linewidth=2, label="presel / all")

    axr.set_ylim(0, 1.05)
    axr.set_ylabel("Efficiency")

    # --- combined legend ---
    l1, lab1 = ax.get_legend_handles_labels()
    l2, lab2 = axr.get_legend_handles_labels()
    ax.legend(l1 + l2, lab1 + lab2, loc="best",framealpha=0.3)

    plt.tight_layout()
    plt.savefig(outname)
    plt.close()


# -------------------------------------------------------
# Main plotting function
# -------------------------------------------------------

def plot_trigger_study(
    var,
    sig_all, sig_presel, sig_trig,
    bkg_all, bkg_presel, bkg_trig,
    data_all, data_presel, data_trig,
    outdir,
):
    os.makedirs(outdir, exist_ok=True)

    # --- binning ---
    if sig_all is not None:
        x = sig_all.bin_centers()
    elif bkg_all is not None:
        x = bkg_all.hist_total.axes[0].centers
    else:
        x = data_all.bin_centers()

    # --- values ---
    sig_all_v    = sig_all.values()[0][1:-1]    if sig_all    is not None else None
    sig_presel_v = sig_presel.values()[0][1:-1] if sig_presel is not None else None
    sig_trig_v   = sig_trig.values()[0][1:-1]   if sig_trig   is not None else None

    bkg_all_v    = bkg_all.values_total()[0][1:-1]    if bkg_all    is not None else None
    bkg_presel_v = bkg_presel.values_total()[0][1:-1] if bkg_presel is not None else None
    bkg_trig_v   = bkg_trig.values_total()[0][1:-1]   if bkg_trig   is not None else None

    data_all_v    = data_all.values()[0][1:-1]    if data_all    is not None else None
    data_presel_v = data_presel.values()[0][1:-1] if data_presel is not None else None
    data_trig_v   = data_trig.values()[0][1:-1]   if data_trig   is not None else None



    # --- scaling ---
    sig_all_s, sa = get_scaled(sig_all_v, bkg_all_v)
    sig_presel_s, sp = get_scaled(sig_presel_v, bkg_all_v)
    sig_trig_s, st = get_scaled(sig_trig_v, bkg_all_v)

    bkg_all_s, ba = get_scaled(bkg_all_v, bkg_all_v)
    bkg_presel_s, bp = get_scaled(bkg_presel_v, bkg_all_v)
    bkg_trig_s, bt = get_scaled(bkg_trig_v, bkg_all_v)

    data_all_s, da = get_scaled(data_all_v, bkg_all_v)
    data_presel_s, dp = get_scaled(data_presel_v, bkg_all_v)
    data_trig_s, dt = get_scaled(data_trig_v, bkg_all_v)

    # =====================================================
    # Main distribution plot
    # =====================================================

    plt.figure(figsize=(8, 6))

    if sig_all_v is not None:
        plt.errorbar(x, sig_all_s, linestyle="-.", color="red", label=f"Signal (all) x{sa:.1e}")
        plt.errorbar(x, sig_trig_s, linestyle="--", color="red", label=f"Signal (trig) x{st:.1e}")
        if sig_presel_v is not None:
            plt.errorbar(x, sig_presel_s, linestyle="-", color="red", label=f"Signal (presel) x{sp:.1e}")

    if bkg_all_v is not None:
        plt.errorbar(x, bkg_all_s, linestyle="-.", color="gray", label="Background (all)")
        plt.errorbar(x, bkg_trig_s, linestyle="--", color="gray", label=f"Background (trig) x{bt}")
        if bkg_presel_v is not None:
            plt.errorbar(x, bkg_presel_s, linestyle="-", color="gray", label=f"Background (presel) x{bp}")

    if data_all_v is not None:
        plt.errorbar(x, data_all_s, yerr=np.sqrt(data_all_v), fmt="-.", color="blue", label=f"Data (all) x{da}")
        plt.errorbar(x, data_trig_s, yerr=np.sqrt(data_trig_v), fmt="--", color="blue", label=f"Data (trig) x{dt}")
        if data_presel_v is not None:
            plt.errorbar(x, data_presel_s, yerr=np.sqrt(data_presel_v), fmt="-", color="blue", label=f"Data (presel) x{dp}")

    plt.xlabel(var)
    if var =='Met_pt_low':
        plt.xlabel('Met_pt')
    plt.ylabel("Events")
    plt.legend(framealpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{outdir}/dist_{var}.png")
    plt.close()

    # =====================================================
    # Per-type plots with ratio overlay
    # =====================================================

    plot_pertype_dist_ratio(x, sig_trig_v, sig_presel_v, sig_all_v,
                            f"{outdir}/sig_{var}.png", "Signal", var)

    plot_pertype_dist_ratio(x, bkg_trig_v, bkg_presel_v, bkg_all_v,
                            f"{outdir}/bkg_{var}.png", "Background", var)

    plot_pertype_dist_ratio(x, data_trig_v, data_presel_v, data_all_v,
                            f"{outdir}/data_{var}.png", "Data", var)


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():
    os.makedirs(OUTDIR, exist_ok=True)

    proc_map = ProcessMap.from_csv(CONFIG.PROC_MAP_CSV)

    all_events = load_hist_collection(ROOT_FILE, proc_map, cutname="all_events")["all_events"]
    triggered  = load_hist_collection(ROOT_FILE, proc_map, cutname="MET_trigger")["MET_trigger"]
    #presel  = load_hist_collection(ROOT_FILE, proc_map, cutname="AK4_Bveto")["AK4_Bveto"]


    variables = all_events.views.keys()

    for var in variables:
        sig_all, bkg_all, data_all = unpack_hist(all_events, var, proc_map)
        sig_trig, bkg_trig, data_trig = unpack_hist(triggered, var, proc_map)
        #sig_presel, bkg_presel, data_presel = unpack_hist(presel, var, proc_map)
        sig_presel, bkg_presel, data_presel =None,None,None 

        plot_trigger_study(
            var,
            sig_all, sig_presel, sig_trig,
            bkg_all, bkg_presel, bkg_trig,
            data_all, data_presel, data_trig,
            OUTDIR,
        )
        print(f'finished {var}')


if __name__ == "__main__":
    main()
