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

def write_abcd_datacards(histo_sig, histo_dy, histo_otherbkg, results, output_dir="abcd_scan_plots", n_top=5):
    os.makedirs(os.path.join(output_dir, "datacards"), exist_ok=True)

    # Find the top n_top working points by significance
    sig_flat = results["significance"].flatten()
    top_indices = np.argsort(sig_flat)[::-1]
    top_indices = top_indices[~np.isnan(sig_flat[top_indices])][:n_top]

    sig_h   = histo_sig[{"process_grp": sum}]
    dy_h    = histo_dy[{"process_grp": sum}]
    other_h = histo_otherbkg[{"process_grp": sum}]

    score_edges = sig_h.axes["dnn_score"].edges
    mjj_edges   = sig_h.axes["mjj_max_any"].edges

    for rank, flat_idx in enumerate(top_indices):
        i, j = np.unravel_index(flat_idx, results["significance"].shape)
        score_cut = results["score_cuts"][i]
        mjj_cut   = results["mjj_cuts"][j]
        sig_val   = results["significance"][i, j]

        # Get the bin indices corresponding to the cuts
        si = int(np.searchsorted(score_edges, score_cut))
        mj = int(np.searchsorted(mjj_edges,   mjj_cut))

        # Get yields in region A
        A_sig   = get_yield(sig_h,   slice(si, None), slice(mj, None))
        A_dy    = get_yield(dy_h,    slice(si, None), slice(mj, None))
        B_dy    = get_yield(dy_h,    slice(None, si), slice(mj, None))
        C_dy    = get_yield(dy_h,    slice(si, None), slice(None, mj))
        D_dy    = get_yield(dy_h,    slice(None, si), slice(None, mj))
        A_other = get_yield(other_h, slice(si, None), slice(mj, None))

        # DY ABCD estimate in region A
        dy_est = (B_dy * C_dy / D_dy) if D_dy > 0 else 0.0
        A_bkg  = dy_est + A_other

        # Observed (MC pseudodata, blinded)
        A_obs = A_dy + A_other

        # Format the filename
        score_str = f"{score_cut:.2f}".replace(".", "p")
        mjj_str   = f"{mjj_cut:.0f}"
        sig_str   = f"{sig_val:.2f}".replace(".", "p")
        fpath = os.path.join(output_dir, "datacards", f"wp{rank}_sig{sig_str}_s{score_str}_mjj{mjj_str}.txt")

        with open(fpath, "w") as f:
            f.write(f"# Counting experiment datacard: rank={rank}, score>{score_cut:.4f}, mjj>{mjj_cut:.1f} GeV\n")
            f.write(f"# S/sqrt(B_total)={sig_val:.4f}\n")
            f.write(f"# A_dy_true={A_dy:.2f}, A_dy_est={dy_est:.2f}, A_other={A_other:.2f}\n\n")
            f.write("imax 1  number of channels\n")
            f.write("jmax 1  number of backgrounds\n")
            f.write("kmax 0  number of nuisance parameters\n")
            f.write("-" * 60 + "\n")
            f.write(f"bin          A\n")
            f.write(f"observation  {A_obs:.4f}\n")
            f.write("-" * 60 + "\n")
            f.write(f"bin          A       A\n")
            f.write(f"process      sig     bkg\n")
            f.write(f"process      0       1\n")
            f.write(f"rate         {A_sig:.4f}  {A_bkg:.4f}\n")
            f.write("-" * 60 + "\n")

        print(f"Wrote {fpath}")

def plot_mjj_score_slices_optimized(histo_dy, best_score_cut, output_dir="abcd_scan_plots"):
    bkg_h = histo_dy[{"process_grp": sum}]
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
            (0,               lo_split,       "tab:blue",   "low score, low half"),
            (lo_split,        best_score_bin, "tab:cyan",   "low score, high half"),
            (best_score_bin,  n_score,        "tab:red",    "high score (undivided)"),
        ]
    else:
        slices = [
            (0,               lo_split,       "tab:blue",   "low score, low half"),
            (lo_split,        best_score_bin, "tab:cyan",   "low score, high half"),
            (best_score_bin,  hi_split,       "tab:orange", "high score, low half"),
            (hi_split,        n_score,        "tab:red",    "high score, high half"),
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
        f"mjj distribution in optimized score slices (DY)\n"
        f"Split at best score cut ({best_score_cut:.3f}), then equal-yield subdivisions"
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/mjj_score_slices_optimized.png", dpi=150)
    plt.close()
    print(f"Saved {output_dir}/mjj_score_slices_optimized.png")

def plot_mjj_score_slices(histo_dy, output_dir="abcd_scan_plots"):
    bkg_h = histo_dy[{"process_grp": sum}]
    score_edges = bkg_h.axes["dnn_score"].edges
    mjj_edges   = bkg_h.axes["mjj_max_any"].edges
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
    ax.set_title("mjj distribution in score slices (DY)\nShould be similar if decorrelated")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/mjj_score_slices.png", dpi=150)
    plt.close()
    print(f"Saved {output_dir}/mjj_score_slices.png")

def plot_best_working_point(histo_sig, histo_dy, histo_otherbkg, results, output_dir="abcd_scan_plots"):
    # Find the best working point (max significance)
    best_idx = np.nanargmax(results["significance"])
    best_i, best_j = np.unravel_index(best_idx, results["significance"].shape)
    best_score_cut  = results["score_cuts"][best_i]
    best_mjj_cut    = results["mjj_cuts"][best_j]
    best_sig        = results["significance"][best_i, best_j]
    best_S          = results["S"][best_i, best_j]
    best_B_dy_true  = results["B_dy_true"][best_i, best_j]
    best_B_dy_est   = results["B_dy_est"][best_i, best_j]
    best_B_other    = results["B_other"][best_i, best_j]
    best_B_total    = results["B_total"][best_i, best_j]
    best_closure    = results["closure"][best_i, best_j]
    best_sig_true   = best_S / np.sqrt(best_B_dy_true + best_B_other) if (best_B_dy_true + best_B_other) > 0 else np.nan

    # Print summary
    print("\n" + "="*50)
    print("BEST WORKING POINT (max S/sqrt(B_total))")
    print("="*50)
    print(f"  Score cut         : {best_score_cut:.4f}  (scan index {best_i})")
    print(f"  mjj cut           : {best_mjj_cut:.1f} GeV  (scan index {best_j})")
    print(f"  S                 : {best_S:.4f}")
    print(f"  B_dy_true (MC)    : {best_B_dy_true:.2f}")
    print(f"  B_dy_est (B*C/D)  : {best_B_dy_est:.2f}")
    print(f"  B_other (MC)      : {best_B_other:.2f}")
    print(f"  B_total           : {best_B_total:.2f}")
    print(f"  S/sqrt(B_total)   : {best_sig:.4f}")
    print(f"  S/sqrt(B_true)    : {best_sig_true:.4f}")
    print(f"  DY Closure        : {best_closure:.4f}")
    print("="*50 + "\n")

    sig_h   = histo_sig[{"process_grp": sum}]
    dy_h    = histo_dy[{"process_grp": sum}]
    other_h = histo_otherbkg[{"process_grp": sum}]
    score_edges = sig_h.axes["dnn_score"].edges
    mjj_edges   = sig_h.axes["mjj_max_any"].edges
    dy_vals     = dy_h.values(flow=False)
    other_vals  = other_h.values(flow=False)
    allbkg_vals = dy_vals + other_vals

    score_mid_lo = score_edges[0] + (best_score_cut - score_edges[0]) / 2
    score_mid_hi = best_score_cut + (score_edges[-1] - best_score_cut) / 2
    mjj_mid_lo   = mjj_edges[0]   + (best_mjj_cut - mjj_edges[0]) / 2
    mjj_mid_hi   = best_mjj_cut   + (mjj_edges[-1] - best_mjj_cut) / 2

    def _add_cut_lines_and_labels(ax):
        ax.axvline(best_score_cut, color="red",    linewidth=2, linestyle="--", label=f"score > {best_score_cut:.3f}")
        ax.axhline(best_mjj_cut,   color="orange", linewidth=2, linestyle="--", label=f"mjj > {best_mjj_cut:.0f} GeV")
        ax.text(score_mid_hi, mjj_mid_hi, "A (SR)", ha="center", va="center", color="red",   fontsize=12, fontweight="bold")
        ax.text(score_mid_lo, mjj_mid_hi, "B",      ha="center", va="center", color="black", fontsize=12, fontweight="bold")
        ax.text(score_mid_hi, mjj_mid_lo, "C",      ha="center", va="center", color="black", fontsize=12, fontweight="bold")
        ax.text(score_mid_lo, mjj_mid_lo, "D",      ha="center", va="center", color="black", fontsize=12, fontweight="bold")
        ax.set_xlim(score_edges[0], score_edges[-1])
        ax.set_ylim(mjj_edges[0],   mjj_edges[-1])
        ax.set_xlabel("DNN score")
        ax.set_ylabel("mjj_max_any [GeV]")
        ax.legend(loc="upper right", fontsize=8)

    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: DY only
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(score_edges, mjj_edges, dy_vals.T, cmap="Blues")
    plt.colorbar(im, ax=ax, label="DY yield")
    _add_cut_lines_and_labels(ax)
    ax.set_title(
        f"Best working point (DY): score>{best_score_cut:.3f}, mjj>{best_mjj_cut:.0f} GeV\n"
        f"S={best_S:.3f}, B_dy_est={best_B_dy_est:.1f}, B_other={best_B_other:.1f}, "
        f"B_total={best_B_total:.1f}, S/sqrt(B_total)={best_sig:.3f}, closure={best_closure:.3f}",
        fontsize=9,
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/best_working_point_dy.png", dpi=150)
    plt.close()
    print(f"Saved {output_dir}/best_working_point_dy.png")

    # Plot 2: all backgrounds
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.pcolormesh(score_edges, mjj_edges, allbkg_vals.T, cmap="Blues")
    plt.colorbar(im, ax=ax, label="Total background yield (DY + other MC)")
    _add_cut_lines_and_labels(ax)
    ax.set_title(
        f"Best working point (all bkg): score>{best_score_cut:.3f}, mjj>{best_mjj_cut:.0f} GeV\n"
        f"S={best_S:.3f}, B_dy_est={best_B_dy_est:.1f}, B_other={best_B_other:.1f}, "
        f"B_total={best_B_total:.1f}, S/sqrt(B_total)={best_sig:.3f}, closure={best_closure:.3f}",
        fontsize=9,
    )
    plt.tight_layout()
    plt.savefig(f"{output_dir}/best_working_point_allbkg.png", dpi=150)
    plt.close()
    print(f"Saved {output_dir}/best_working_point_allbkg.png")


def plot_abcd_2d_snapshots(histo_sig, histo_dy, histo_otherbkg, results, output_dir="abcd_snapshots"):
    os.makedirs(output_dir, exist_ok=True)
    sig_h  = histo_sig[{"process_grp": sum}]
    dy_h   = histo_dy[{"process_grp": sum}]
    n_score = len(results["score_cuts"])
    n_mjj   = len(results["mjj_cuts"])
    score_edges = sig_h.axes["dnn_score"].edges
    mjj_edges   = sig_h.axes["mjj_max_any"].edges
    dy_vals  = dy_h.values(flow=False)
    sig_vals = sig_h.values(flow=False)

    # Overview plots: DY background
    for scale, norm, suffix in [("linear", None, "lin"), ("log", matplotlib.colors.LogNorm(), "log")]:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.pcolormesh(score_edges, mjj_edges, dy_vals.T, cmap="Blues", norm=norm)
        plt.colorbar(im, ax=ax, label="DY yield")
        score_centers = (score_edges[:-1] + score_edges[1:]) / 2
        mjj_centers   = (mjj_edges[:-1]   + mjj_edges[1:])   / 2
        profile = np.zeros(len(score_centers))
        for si in range(len(score_centers)):
            col = dy_vals[si, :]
            total = col.sum()
            if total > 0:
                profile[si] = np.average(mjj_centers, weights=col)
            else:
                profile[si] = np.nan
        ax.plot(score_centers, profile, color="red", linewidth=2, marker="o", markersize=4, label="Mean mjj")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_xlabel("DNN score")
        ax.set_ylabel("mjj_max_any [GeV]")
        ax.set_title(f"DY 2D histogram (no cuts, {scale})")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/scan_point_overview_bkg_{suffix}.png", dpi=150)
        plt.close()
        print(f"Saved {output_dir}/scan_point_overview_bkg_{suffix}.png")

    # Overview plots: signal
    for scale, norm, suffix in [("linear", None, "lin"), ("log", matplotlib.colors.LogNorm(), "log")]:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.pcolormesh(score_edges, mjj_edges, sig_vals.T, cmap="Reds", norm=norm)
        plt.colorbar(im, ax=ax, label="Signal yield")
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

    for i in range(n_score):
        score_cut = results["score_cuts"][i]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axvline(score_cut, color="red", linewidth=2, linestyle="--", label=f"score > {score_cut:.2f}")
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
        plt.savefig(f"{output_dir}/scan_block_{i:04d}.png", dpi=100)
        plt.close()
        print(f"Saved scan block {i}")

def do_abcd_scan(histo_sig, histo_dy, histo_otherbkg, score_axis_name="dnn_score", mjj_axis_name="mjj_max_any", n_scan=20):
    sig_h   = histo_sig[{"process_grp": sum}]
    dy_h    = histo_dy[{"process_grp": sum}]
    other_h = histo_otherbkg[{"process_grp": sum}]

    score_edges = sig_h.axes[score_axis_name].edges
    mjj_edges   = sig_h.axes[mjj_axis_name].edges
    n_score = len(score_edges) - 1
    n_mjj   = len(mjj_edges) - 1

    score_cut_bins = [int(x) for x in np.linspace(int(0.1*n_score), int(0.9*n_score), n_scan)]
    mjj_cut_bins   = [int(x) for x in np.linspace(int(0.1*n_mjj),   int(0.9*n_mjj),   n_scan)]

    closure      = np.zeros((n_scan, n_scan))
    significance = np.zeros((n_scan, n_scan))
    S_arr        = np.zeros((n_scan, n_scan))
    B_dy_true    = np.zeros((n_scan, n_scan))
    B_dy_est     = np.zeros((n_scan, n_scan))
    B_other      = np.zeros((n_scan, n_scan))
    B_total      = np.zeros((n_scan, n_scan))

    for i, si in enumerate(score_cut_bins):
        for j, mj in enumerate(mjj_cut_bins):
            A_sig       = get_yield(sig_h,   slice(si, None), slice(mj, None))
            A_dy        = get_yield(dy_h,    slice(si, None), slice(mj, None))
            B_dy        = get_yield(dy_h,    slice(None, si), slice(mj, None))
            C_dy        = get_yield(dy_h,    slice(si, None), slice(None, mj))
            D_dy        = get_yield(dy_h,    slice(None, si), slice(None, mj))
            A_other     = get_yield(other_h, slice(si, None), slice(mj, None))

            dy_est      = (B_dy * C_dy / D_dy) if D_dy > 0 else np.nan
            b_total     = (dy_est + A_other) if not np.isnan(dy_est) else np.nan

            S_arr[i, j]        = A_sig
            B_dy_true[i, j]    = A_dy
            B_dy_est[i, j]     = dy_est
            B_other[i, j]      = A_other
            B_total[i, j]      = b_total
            closure[i, j]      = (dy_est / A_dy) if (A_dy > 0 and not np.isnan(dy_est)) else np.nan
            significance[i, j] = (A_sig / np.sqrt(b_total)) if (not np.isnan(b_total) and b_total > 0) else np.nan

    return {
        "closure"      : closure,
        "significance" : significance,
        "S"            : S_arr,
        "B_dy_true"    : B_dy_true,
        "B_dy_est"     : B_dy_est,
        "B_other"      : B_other,
        "B_total"      : B_total,
        "score_cuts"   : score_edges[score_cut_bins],
        "mjj_cuts"     : mjj_edges[mjj_cut_bins],
    }

def plot_abcd_scan_panels(results, output_path):
    n_score = len(results["score_cuts"])
    n_mjj   = len(results["mjj_cuts"])

    scan_points       = []
    significance      = []
    significance_true = []
    B_dy_true         = []
    B_dy_est          = []
    B_other           = []
    B_total           = []
    closure           = []
    labels            = []

    for i in range(n_score):
        for j in range(n_mjj):
            scan_points.append(len(scan_points))
            significance.append(results["significance"][i, j])
            b_true_total = results["B_dy_true"][i, j] + results["B_other"][i, j]
            significance_true.append(
                results["S"][i, j] / np.sqrt(b_true_total) if b_true_total > 0 else np.nan
            )
            B_dy_true.append(results["B_dy_true"][i, j])
            B_dy_est.append(results["B_dy_est"][i, j])
            B_other.append(results["B_other"][i, j])
            B_total.append(results["B_total"][i, j])
            closure.append(results["closure"][i, j])
            labels.append(f"s>{results['score_cuts'][i]:.2f},mjj>{results['mjj_cuts'][j]:.0f}")

    scan_points       = np.array(scan_points)
    significance      = np.array(significance)
    significance_true = np.array(significance_true)
    B_dy_true         = np.array(B_dy_true)
    B_dy_est          = np.array(B_dy_est)
    B_other           = np.array(B_other)
    B_total           = np.array(B_total)
    closure           = np.array(closure)

    fig, axes = plt.subplots(3, 1, figsize=(min(max(12, len(scan_points)//4), 80), 12), sharex=True)

    # Top panel: significance
    axes[0].plot(scan_points, significance,      marker="o", markersize=3, linewidth=1, color="tab:blue",   label="S/sqrt(B_total)")
    axes[0].plot(scan_points, significance_true, marker="o", markersize=3, linewidth=1, color="tab:orange", label="S/sqrt(B_true_total)")
    axes[0].set_ylabel("Significance")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Middle panel: background yields
    axes[1].plot(scan_points, B_dy_true, marker="o", markersize=3, linewidth=1, color="tab:blue",   label="B_dy true (MC)")
    axes[1].plot(scan_points, B_dy_est,  marker="o", markersize=3, linewidth=1, color="tab:cyan",   label="B_dy est (B*C/D)")
    axes[1].plot(scan_points, B_other,   marker="o", markersize=3, linewidth=1, color="tab:green",  label="B_other (MC)")
    axes[1].plot(scan_points, B_total,   marker="o", markersize=3, linewidth=1, color="tab:orange", label="B_total")
    axes[1].set_ylabel("Background yield in A")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Bottom panel: DY closure ratio
    axes[2].plot(scan_points, closure, marker="o", markersize=3, linewidth=1, color="tab:green")
    axes[2].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[2].axhline(1.2, color="red",   linestyle="--", linewidth=0.8, alpha=0.5)
    axes[2].axhline(0.8, color="red",   linestyle="--", linewidth=0.8, alpha=0.5)
    axes[2].set_ylabel("B_dy_est / B_dy_true (DY closure)")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("pkl_file_path", help="The path to the pkl file")
    args = parser.parse_args()

    cat_lst  = cvh.CAT_LST_2l
    grp_dict = cvh.GRP_DICT_FULL_R2

    histo_dict = pickle.load(gzip.open(args.pkl_file_path))
    histo = histo_dict["abcd_histo"]
    histo = histo[{"category": "2lOSSF_1fjx_2j_mjj100", "lepflav": sum}]
    histo = plt_tools.group(histo, "process", "process_grp", grp_dict)

    # Build the list of non-DY background group names
    other_bkg_names = []
    for grp_name in grp_dict:
        if grp_name not in ["Data", "Signal", "DY", 'VBSWWH_SS', 'VBSWWH_OS', 'VBSWZH', 'VBSZZH']:
            other_bkg_names.append(grp_name)

    histo_sig      = histo[{"process_grp": ["Signal"]}]
    histo_dat      = histo[{"process_grp": ["Data"]}]
    histo_dy       = histo[{"process_grp": ["DY"]}]
    histo_otherbkg = plt_tools.group(histo, "process_grp", "process_grp", {"OtherBkg": other_bkg_names})

    print("sig",      histo_sig)
    print("data",     histo_dat)
    print("dy",       histo_dy)
    print("otherbkg", histo_otherbkg)

    out_dir = "abcd_scan_plots"
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(os.path.join(out_dir, "index.php")):
        shutil.copyfile(HTML_PC, os.path.join(out_dir, "index.php"))

    # Make a plot to check decorrelation (DY only)
    plot_mjj_score_slices(histo_dy, output_dir=out_dir)

    # Run the scan
    results = do_abcd_scan(histo_sig, histo_dy, histo_otherbkg, n_scan=50)
    plot_abcd_scan_panels(results, f"{out_dir}/abcd_scan_panels.png")
    plot_abcd_2d_snapshots(histo_sig, histo_dy, histo_otherbkg, results, output_dir=out_dir)
    plot_best_working_point(histo_sig, histo_dy, histo_otherbkg, results, output_dir=out_dir)

    # Make the slices plot based on best point
    best_idx = np.nanargmax(results["significance"])
    best_i, best_j = np.unravel_index(best_idx, results["significance"].shape)
    best_score_cut = results["score_cuts"][best_i]
    plot_mjj_score_slices_optimized(histo_dy, best_score_cut, output_dir=out_dir)

    # Write datacards for best scan points
    write_abcd_datacards(histo_sig, histo_dy, histo_otherbkg, results, output_dir=out_dir)

main()
