import argparse
import pickle
import gzip
import os
import shutil
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ewkcoffea.modules.plotting_tools as plt_tools

import check_vvh_hists as cvh

HTML_PC = "/home/users/kmohrman/ref_scripts/html_stuff/index.php"
#HTML_PC = "/home/k.mohrman/ref_scirpts/html_stuff/index.php"

def get_yield(h2d, score_slice, mjj_slice):
    return h2d[score_slice, mjj_slice].sum(flow=False).value

def plot_mjj_score_slices_optimized(histo_bkg, best_score_cut, output_dir="abcd_scan_plots"):
    bkg_h = histo_bkg[{"process_grp": sum}]
    score_edges = bkg_h.axes["dnn_score"].edges
    mjj_edges   = bkg_h.axes["mjj_max_any"].edges
    bkg_vals    = bkg_h.values(flow=False)
    n_score = len(score_edges) - 1

    # Find the bin index corresponding to best_score_cut
    best_score_bin = int(np.clip(np.searchsorted(score_edges, best_score_cut), 1, n_score - 1))

    def find_equal_yield_split(lo_bin, hi_bin):
        """Find the bin index that splits [lo_bin, hi_bin) into two equal-yield halves."""
        yields = bkg_vals[lo_bin:hi_bin, :].sum(axis=1)
        cumsum = np.cumsum(yields)
        split_idx = int(np.argmin(np.abs(cumsum - cumsum[-1] / 2)))
        return lo_bin + int(np.clip(split_idx + 1, 1, hi_bin - lo_bin - 1))

    lo_split = find_equal_yield_split(0, best_score_bin)
    hi_split = find_equal_yield_split(best_score_bin, n_score)

    # Check if the high side has enough bins to subdivide
    high_side_ok = (hi_split > best_score_bin) and (hi_split < n_score - 1)

    if not high_side_ok:
        print("WARNING: not enough bins on high score side to subdivide, using single slice")
        slices = [
            (0,               lo_split,       "tab:blue",   "low score, low yield"),
            (lo_split,        best_score_bin, "tab:cyan",   "low score, high yield"),
            (best_score_bin,  n_score,        "tab:red",    "high score (undivided)"),
        ]
    else:
        slices = [
            (0,               lo_split,       "tab:blue",   "low score, low yield"),
            (lo_split,        best_score_bin, "tab:cyan",   "low score, high yield"),
            (best_score_bin,  hi_split,       "tab:orange", "high score, low yield"),
            (hi_split,        n_score,        "tab:red",    "high score, high yield"),
        ]

    fig, ax = plt.subplots(figsize=(8, 6))
    for lo, hi, color, label in slices:
        mjj_proj = bkg_vals[lo:hi, :].sum(axis=0)
        total = mjj_proj.sum()
        if total > 0:
            mjj_proj = mjj_proj / total
        score_lo      = score_edges[lo]
        score_hi      = score_edges[hi]
        total_yield   = bkg_vals[lo:hi, :].sum()
        ax.stairs(mjj_proj, mjj_edges, linewidth=2, color=color,
                  label=f"{label} [{score_lo:.2f}, {score_hi:.2f}) yield={total_yield:.1f}")

    ax.set_xlabel("mjj_max_any [GeV]")
    ax.set_ylabel("Normalized yield")
    ax.set_title(
        f"mjj distribution in optimized score slices (background)\n"
        f"Split at best score cut ({best_score_cut:.3f}), then equal-yield subdivisions"
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/mjj_score_slices_optimized.png", dpi=150)
    plt.close()
    print(f"Saved {output_dir}/mjj_score_slices_optimized.png")


def plot_mjj_score_slices(histo_bkg, output_dir="abcd_scan_plots"):

    bkg_h = histo_bkg[{"process_grp": sum}]
    score_edges = bkg_h.axes["dnn_score"].edges
    mjj_edges   = bkg_h.axes["mjj_max_any"].edges
    mjj_centers = (mjj_edges[:-1] + mjj_edges[1:]) / 2
    bkg_vals    = bkg_h.values(flow=False)

    n_score = len(score_edges) - 1
    # Define a few score slices to compare
    slice_edges = np.linspace(0, n_score, 5, dtype=int)  # 4 equal slices
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    fig, ax = plt.subplots(figsize=(8, 6))
    for k in range(len(slice_edges) - 1):
        lo = slice_edges[k]
        hi = slice_edges[k + 1]
        mjj_proj = bkg_vals[lo:hi, :].sum(axis=0)
        total = mjj_proj.sum()
        if total > 0:
            mjj_proj = mjj_proj / total
        score_lo = score_edges[lo]
        score_hi = score_edges[hi]
        ax.stairs(mjj_proj, mjj_edges, linewidth=2, color=colors[k], label=f"score [{score_lo:.2f}, {score_hi:.2f})")

    ax.set_xlabel("mjj_max_any [GeV]")
    ax.set_ylabel("Normalized yield")
    ax.set_title("mjj distribution in score slices (background)\nShould be similar if decorrelated")
    #ax.set_xlim(mjj_edges[0], mjj_edges[-1])
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/mjj_score_slices.png", dpi=150)
    plt.close()
    print(f"Saved {output_dir}/mjj_score_slices.png")

def plot_best_working_point(histo_sig, histo_bkg, results, output_dir="abcd_scan_plots"):

    # Find the best working point (max significance)
    best_idx = np.nanargmax(results["significance"])
    best_i, best_j = np.unravel_index(best_idx, results["significance"].shape)

    best_score_cut = results["score_cuts"][best_i]
    best_mjj_cut   = results["mjj_cuts"][best_j]
    best_sig       = results["significance"][best_i, best_j]
    best_sig_true  = results["S"][best_i, best_j] / np.sqrt(results["B_true"][best_i, best_j]) if results["B_true"][best_i, best_j] > 0 else np.nan
    best_S         = results["S"][best_i, best_j]
    best_B_true    = results["B_true"][best_i, best_j]
    best_B_est     = results["B_est"][best_i, best_j]
    best_closure   = results["closure"][best_i, best_j]

    # Print summary
    print("\n" + "="*50)
    print("BEST WORKING POINT (max S/sqrt(B_est))")
    print("="*50)
    print(f"  Score cut       : {best_score_cut:.4f}  (scan index {best_i})")
    print(f"  mjj cut         : {best_mjj_cut:.1f} GeV  (scan index {best_j})")
    print(f"  S               : {best_S:.4f}")
    print(f"  B_true (MC)     : {best_B_true:.2f}")
    print(f"  B_est (B*C/D)   : {best_B_est:.2f}")
    print(f"  S/sqrt(B_est)   : {best_sig:.4f}")
    print(f"  S/sqrt(B_true)  : {best_sig_true:.4f}")
    print(f"  Closure         : {best_closure:.4f}")
    print("="*50 + "\n")

    # Make the 2D plot
    sig_h = histo_sig[{"process_grp": sum}]
    bkg_h = histo_bkg[{"process_grp": sum}]
    score_edges = sig_h.axes["dnn_score"].edges
    mjj_edges   = sig_h.axes["mjj_max_any"].edges
    bkg_vals    = bkg_h.values(flow=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(score_edges, mjj_edges, bkg_vals.T, cmap="Blues")
    plt.colorbar(im, ax=ax, label="Background yield")

    ax.axvline(best_score_cut, color="red",    linewidth=2, linestyle="--", label=f"score > {best_score_cut:.3f}")
    ax.axhline(best_mjj_cut,   color="orange", linewidth=2, linestyle="--", label=f"mjj > {best_mjj_cut:.0f} GeV")

    score_mid_lo = score_edges[0] + (best_score_cut - score_edges[0]) / 2
    score_mid_hi = best_score_cut + (score_edges[-1] - best_score_cut) / 2
    mjj_mid_lo   = mjj_edges[0]   + (best_mjj_cut - mjj_edges[0]) / 2
    mjj_mid_hi   = best_mjj_cut   + (mjj_edges[-1] - best_mjj_cut) / 2

    ax.text(score_mid_hi, mjj_mid_hi, "A (SR)", ha="center", va="center", color="red",   fontsize=12, fontweight="bold")
    ax.text(score_mid_lo, mjj_mid_hi, "B",      ha="center", va="center", color="black", fontsize=12, fontweight="bold")
    ax.text(score_mid_hi, mjj_mid_lo, "C",      ha="center", va="center", color="black", fontsize=12, fontweight="bold")
    ax.text(score_mid_lo, mjj_mid_lo, "D",      ha="center", va="center", color="black", fontsize=12, fontweight="bold")

    ax.set_xlim(score_edges[0], score_edges[-1])
    ax.set_ylim(mjj_edges[0],   mjj_edges[-1])
    ax.set_xlabel("DNN score")
    ax.set_ylabel("mjj_max_any [GeV]")
    ax.set_title(
        f"Best working point: score>{best_score_cut:.3f}, mjj>{best_mjj_cut:.0f} GeV\n"
        f"S={best_S:.3f}, B_true={best_B_true:.1f}, B_est={best_B_est:.1f}, "
        f"S/sqrt(B_est)={best_sig:.3f}, closure={best_closure:.3f}",
        fontsize=9,
    )
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/best_working_point.png", dpi=150)
    plt.close()
    print(f"Saved {output_dir}/best_working_point.png")


def plot_abcd_2d_snapshots(histo_sig, histo_bkg, results, output_dir="abcd_snapshots"):
    import os
    os.makedirs(output_dir, exist_ok=True)

    sig_h = histo_sig[{"process_grp": sum}]
    bkg_h = histo_bkg[{"process_grp": sum}]

    n_score = len(results["score_cuts"])
    n_mjj   = len(results["mjj_cuts"])

    score_edges = sig_h.axes["dnn_score"].edges
    mjj_edges   = sig_h.axes["mjj_max_any"].edges

    # Get the full 2D bkg array for plotting
    bkg_vals = bkg_h.values(flow=False)
    sig_vals = sig_h.values(flow=False)

    # Overview plots: background
    for scale, norm, suffix in [("linear", None, "lin"), ("log", matplotlib.colors.LogNorm(), "log")]:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.pcolormesh(score_edges, mjj_edges, bkg_vals.T, cmap="Blues", norm=norm)
        #im = ax.imshow(score_edges, mjj_edges, bkg_vals.T, cmap="Blues", norm=norm)
        plt.colorbar(im, ax=ax, label="Background yield")

        # Profile line
        score_centers = (score_edges[:-1] + score_edges[1:]) / 2
        mjj_centers   = (mjj_edges[:-1]   + mjj_edges[1:])   / 2
        profile = np.zeros(len(score_centers))
        for si in range(len(score_centers)):
            col = bkg_vals[si, :]
            total = col.sum()
            if total > 0:
                profile[si] = np.average(mjj_centers, weights=col)
            else:
                profile[si] = np.nan
        ax.plot(score_centers, profile, color="red", linewidth=2, marker="o", markersize=4, label="Mean mjj")
        ax.legend(loc="upper right", fontsize=8)

        ax.set_xlabel("DNN score")
        ax.set_ylabel("mjj_max_any [GeV]")
        ax.set_title(f"Background 2D histogram (no cuts, {scale})")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/scan_point_overview_bkg_{suffix}.png", dpi=150)
        plt.close()
        print(f"Saved {output_dir}/scan_point_overview_bkg_{suffix}.png")

    # Overview plots: signal
    for scale, norm, suffix in [("linear", None, "lin"), ("log", matplotlib.colors.LogNorm(), "log")]:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.pcolormesh(score_edges, mjj_edges, sig_vals.T, cmap="Reds", norm=norm)
        #im = ax.imshow(score_edges, mjj_edges, sig_vals.T, cmap="Reds", norm=norm)
        plt.colorbar(im, ax=ax, label="Signal yield")

        # Profile line
        score_centers = (score_edges[:-1] + score_edges[1:]) / 2
        mjj_centers   = (mjj_edges[:-1]   + mjj_edges[1:])   / 2
        profile = np.zeros(len(score_centers))
        for si in range(len(score_centers)):
            col = sig_vals[si, :]
            total = col.sum()
            if total > 0:
                profile[si] = np.average(mjj_centers, weights=col)
            else:
                profile[si] = np.nan
        ax.plot(score_centers, profile, color="red", linewidth=2, marker="o", markersize=4, label="Mean mjj")
        ax.legend(loc="upper right", fontsize=8)

        ax.set_xlabel("DNN score")
        ax.set_ylabel("mjj_max_any [GeV]")
        ax.set_title(f"Signal 2D histogram (no cuts, {scale})")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/scan_point_overview_sig_{suffix}.png", dpi=150)
        plt.close()
        print(f"Saved {output_dir}/scan_point_overview_sig_{suffix}.png")

    plot_idx = 0
    for i in range(n_score):
        score_cut = results["score_cuts"][i]

        fig, ax = plt.subplots(figsize=(8, 6))

        # Draw the score cut line
        ax.axvline(score_cut, color="red", linewidth=2, linestyle="--", label=f"score > {score_cut:.2f}")

        # Draw all 20 mjj cut lines
        for j in range(n_mjj):
            mjj_cut = results["mjj_cuts"][j]
            alpha = 0.2 + 0.8 * (j / (n_mjj - 1))
            ax.axhline(mjj_cut, color="orange", linewidth=1, linestyle="--", alpha=alpha)
            ax.text(score_edges[-1], mjj_cut, f"{j}", fontsize=7, va="bottom", ha="right", color="orange", alpha=alpha)

        ax.set_xlim(score_edges[0], score_edges[-1])
        ax.set_ylim(mjj_edges[0],   mjj_edges[-1])
        ax.set_xlabel("DNN score")
        ax.set_ylabel("mjj_max_any [GeV]")
        ax.set_title(f"Score cut block {i}: score > {score_cut:.2f}\nmjj cuts shown as horizontal lines")
        ax.legend(loc="upper right", fontsize=8)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/scan_block_{plot_idx:04d}.png", dpi=100)
        plt.close()
        print(f"Saved scan block {plot_idx}")
        plot_idx += 1

def do_abcd_scan(histo_sig, histo_bkg, score_axis_name="dnn_score", mjj_axis_name="mjj_max_any", n_scan=20):

    # Squeeze out the process_grp axis
    sig_h = histo_sig[{"process_grp": sum}]
    bkg_h = histo_bkg[{"process_grp": sum}]

    score_edges = sig_h.axes[score_axis_name].edges
    mjj_edges   = sig_h.axes[mjj_axis_name].edges
    n_score = len(score_edges) - 1
    n_mjj   = len(mjj_edges) - 1

    score_cut_bins = [int(x) for x in np.linspace(int(0.1*n_score), int(0.9*n_score), n_scan)]
    mjj_cut_bins   = [int(x) for x in np.linspace(int(0.1*n_mjj),   int(0.9*n_mjj),   n_scan)]

    closure      = np.zeros((n_scan, n_scan))
    significance = np.zeros((n_scan, n_scan))
    S_arr        = np.zeros((n_scan, n_scan))
    B_true_arr   = np.zeros((n_scan, n_scan))
    B_est_arr    = np.zeros((n_scan, n_scan))

    for i, si in enumerate(score_cut_bins):
        for j, mj in enumerate(mjj_cut_bins):
            A_sig = get_yield(sig_h, slice(si, None), slice(mj, None))
            A_bkg = get_yield(bkg_h, slice(si, None), slice(mj, None))
            B_bkg = get_yield(bkg_h, slice(None, si), slice(mj, None))
            C_bkg = get_yield(bkg_h, slice(si, None), slice(None, mj))
            D_bkg = get_yield(bkg_h, slice(None, si), slice(None, mj))

            B_est = (B_bkg * C_bkg / D_bkg) if D_bkg > 0 else np.nan

            S_arr[i, j]        = A_sig
            B_true_arr[i, j]   = A_bkg
            B_est_arr[i, j]    = B_est
            closure[i, j]      = (B_est / A_bkg) if (A_bkg > 0 and not np.isnan(B_est)) else np.nan
            significance[i, j] = (A_sig / np.sqrt(B_est)) if (not np.isnan(B_est) and B_est > 0) else np.nan

    return {
        "closure"      : closure,
        "significance" : significance,
        "S"            : S_arr,
        "B_true"       : B_true_arr,
        "B_est"        : B_est_arr,
        "score_cuts"   : score_edges[score_cut_bins],
        "mjj_cuts"     : mjj_edges[mjj_cut_bins],
    }

def plot_abcd_scan_panels(results, output_path):

    n_score = len(results["score_cuts"])
    n_mjj   = len(results["mjj_cuts"])

    scan_points       = []
    significance      = []
    significance_true = []
    B_true            = []
    B_est             = []
    closure           = []
    labels            = []

    for i in range(n_score):
        for j in range(n_mjj):
            scan_points.append(len(scan_points))
            significance.append(results["significance"][i, j])
            significance_true.append(
                results["S"][i, j] / np.sqrt(results["B_true"][i, j]) if results["B_true"][i, j] > 0 else np.nan
            )
            B_true.append(results["B_true"][i, j])
            B_est.append(results["B_est"][i, j])
            closure.append(results["closure"][i, j])
            labels.append(f"s>{results['score_cuts'][i]:.2f},mjj>{results['mjj_cuts'][j]:.0f}")

    scan_points  = np.array(scan_points)
    significance = np.array(significance)
    significance_true = np.array(significance_true)
    B_true       = np.array(B_true)
    B_est        = np.array(B_est)
    closure      = np.array(closure)

    fig, axes = plt.subplots(3, 1, figsize=(min(max(12, len(scan_points)//4), 80), 12), sharex=True)

    # Top panel: significance
    axes[0].plot(scan_points, significance,      marker="o", markersize=3, linewidth=1, color="tab:blue",   label="S/sqrt(B_est)")
    axes[0].plot(scan_points, significance_true, marker="o", markersize=3, linewidth=1, color="tab:orange", label="S/sqrt(B_true)")
    axes[0].set_ylabel("Significance")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Middle panel: B_true and B_est
    axes[1].plot(scan_points, B_true, marker="o", markersize=3, linewidth=1, color="tab:blue",   label="B true (MC)")
    axes[1].plot(scan_points, B_est,  marker="o", markersize=3, linewidth=1, color="tab:orange", label="B est (B*C/D)")
    axes[1].set_ylabel("Background yield in A")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Bottom panel: closure ratio
    axes[2].plot(scan_points, closure, marker="o", markersize=3, linewidth=1, color="tab:green")
    axes[2].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[2].axhline(1.2, color="red",   linestyle="--", linewidth=0.8, alpha=0.5)
    axes[2].axhline(0.8, color="red",   linestyle="--", linewidth=0.8, alpha=0.5)
    axes[2].set_ylabel("B_est / B_true")
    axes[2].set_xlabel("Scan point (score cut, mjj cut)")
    axes[2].set_ylim(0, 2)
    axes[2].grid(True, alpha=0.3)

    axes[0].set_xlim(-10, len(scan_points) + 10)

    tick_step = max(1, len(scan_points) // 20)
    tick_positions = [0] + list(scan_points[::tick_step])
    tick_labels = [labels[0]] + [labels[i] for i in range(0, len(scan_points), tick_step)]
    axes[2].set_xticks(tick_positions)
    axes[2].set_xticklabels(tick_labels, rotation=90, fontsize=6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Saved scan plot to {output_path}")


def main():

    # Set up the command line parser
    parser = argparse.ArgumentParser()
    parser.add_argument("pkl_file_path", help = "The path to the pkl file")
    args = parser.parse_args()

    cat_lst = cvh.CAT_LST_2l
    grp_dict = cvh.GRP_DICT_FULL_R2

    # Get the dictionary of histograms from the input pkl file
    histo_dict = pickle.load(gzip.open(args.pkl_file_path))

    # Get the hist for the categories of interest
    histo = histo_dict["abcd_histo"]
    histo = histo[{"category":"2lOSSF_1fjx_2j_mjj100", "lepflav":sum}]
    histo = plt_tools.group(histo,"process","process_grp",grp_dict)

    # Do the sample grouping
    sample_group_names_lst_bkg = []
    for grp_name in grp_dict:
        if grp_name not in ["Data", "Signal", 'VBSWWH_SS', 'VBSWWH_OS', 'VBSWZH', 'VBSZZH']:
            sample_group_names_lst_bkg.append(grp_name)

    # Final sig bkg and data hists
    histo_sig = histo[{"process_grp":["Signal"]}]
    histo_dat = histo[{"process_grp":["Data"]}]
    histo_dy  = histo[{"process_grp":["DY"]}]
    histo_allbkg = plt_tools.group(histo,"process_grp","process_grp",{"Background": sample_group_names_lst_bkg})

    print("s",histo_sig)
    print("d",histo_dat)
    print("b",histo_allbkg)
    print("dy",histo_dy)

    out_dir = "abcd_scan_plots"

    # Chek for just the bkg we trained on, and for all bkg together
    for tag,histo_bkg in [["dy",histo_dy],["allbkg",histo_allbkg]]:

        print(f"\n\n--- Running for {tag} ---\n")

        # Make  the out dir
        out_dir_with_tag = os.path.join(out_dir,tag)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        if not os.path.exists(out_dir_with_tag):
            os.mkdir(out_dir_with_tag)
            shutil.copyfile(HTML_PC, os.path.join(out_dir_with_tag,"index.php"))

        # Make a plot to check de-correlation
        plot_mjj_score_slices(histo_bkg, output_dir=out_dir_with_tag)

        # Do the scan and print results
        results = do_abcd_scan(histo_sig, histo_bkg, n_scan=50)
        plot_abcd_scan_panels(results, f"{out_dir_with_tag}/abcd_scan_panels.png")
        plot_abcd_2d_snapshots(histo_sig, histo_bkg, results, output_dir=out_dir_with_tag)
        plot_best_working_point(histo_sig, histo_bkg, results, output_dir=out_dir_with_tag)

        best_idx = np.nanargmax(results["significance"])
        best_i, best_j = np.unravel_index(best_idx, results["significance"].shape)
        best_score_cut = results["score_cuts"][best_i]
        plot_mjj_score_slices_optimized(histo_bkg, best_score_cut, output_dir=out_dir_with_tag)


main()
