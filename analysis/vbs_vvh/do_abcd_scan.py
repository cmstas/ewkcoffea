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


def eval_at_fixed_cut(histo_sig, histo_abcdbkg, histo_otherbkg, score_cut, label="", score_axis_name="dnn_score"):
    """Evaluate S, B and significance at a fixed score cut, with stat uncertainties."""
    sig_h   = histo_sig[{"process_grp": sum}]
    abcd_h  = histo_abcdbkg[{"process_grp": sum}]
    other_h = histo_otherbkg[{"process_grp": sum}]
    score_edges = sig_h.axes[score_axis_name].edges
    si = int(np.searchsorted(score_edges, score_cut))
    S     = get_yield(sig_h,   slice(si, None), slice(None, None))
    B     = get_yield(abcd_h,  slice(si, None), slice(None, None)) + get_yield(other_h, slice(si, None), slice(None, None))
    S_err = np.sqrt(sig_h[slice(si, None),   slice(None, None)].sum(flow=False).variance)
    B_err = np.sqrt(
        abcd_h[slice(si, None),  slice(None, None)].sum(flow=False).variance +
        other_h[slice(si, None), slice(None, None)].sum(flow=False).variance
    )
    significance = S / np.sqrt(B) if B > 0 else np.nan
    tag = f" ({label})" if label else ""
    print("\n" + "="*50)
    print(f"FIXED CUT EVALUATION{tag}")
    print("="*50)
    print(f"  Score cut    : {score_cut:.4f}")
    print(f"  S            : {S:.4f} +/- {S_err:.4f}")
    print(f"  B            : {B:.2f} +/- {B_err:.2f}")
    print(f"  S/sqrt(B)    : {significance:.4f}")
    print("="*50 + "\n")
    return {"score_cut": score_cut, "S": S, "B": B, "S_err": S_err, "B_err": B_err, "significance": significance}


def write_single_datacard(A_sig, A_bkg, score_cut, mjj_cut, output_path):
    A_obs = A_sig + A_bkg
    with open(output_path, "w") as f:
        f.write(f"# Counting experiment datacard: score>{score_cut:.4f}, mjj>{mjj_cut:.1f} GeV\n")
        f.write( "imax 1  number of channels\n")
        f.write( "jmax 1  number of backgrounds\n")
        f.write( "kmax 0  number of nuisance parameters\n")
        f.write( "-" * 60 + "\n")
        f.write( "bin          A\n")
        f.write(f"observation  {A_obs:.4f}\n")
        f.write( "-" * 60 + "\n")
        f.write( "bin          A       A\n")
        f.write( "process      sig     bkg\n")
        f.write( "process      0       1\n")
        f.write(f"rate         {A_sig:.4f}  {A_bkg:.4f}\n")
        f.write( "-" * 60 + "\n")
    print(f"Wrote {output_path}")


def write_score_only_datacard(best, output_path):
    A_obs = best["S"] + best["B"]
    with open(output_path, "w") as f:
        f.write( "# Score-only counting experiment datacard\n")
        f.write(f"# score > {best['score_cut']:.4f}\n")
        f.write(f"# S/sqrt(B)={best['significance']:.4f}\n\n")
        f.write( "imax 1  number of channels\n")
        f.write( "jmax 1  number of backgrounds\n")
        f.write( "kmax 0  number of nuisance parameters\n")
        f.write( "-" * 60 + "\n")
        f.write( "bin          A\n")
        f.write(f"observation  {A_obs:.4f}\n")
        f.write( "-" * 60 + "\n")
        f.write( "bin          A       A\n")
        f.write( "process      sig     bkg\n")
        f.write( "process      0       1\n")
        f.write(f"rate         {best['S']:.4f}  {best['B']:.4f}\n")
        f.write( "-" * 60 + "\n")
    print(f"Wrote {output_path}")


def scan_score_only(histo_sig, histo_abcdbkg, histo_otherbkg, score_axis_name="dnn_score"):
    sig_h   = histo_sig[{"process_grp": sum}]
    abcd_h  = histo_abcdbkg[{"process_grp": sum}]
    other_h = histo_otherbkg[{"process_grp": sum}]
    score_edges = sig_h.axes[score_axis_name].edges
    n_score     = len(score_edges) - 1
    results = []
    for si in range(1, n_score):
        S = get_yield(sig_h,   slice(si, None), slice(None, None))
        B = get_yield(abcd_h,  slice(si, None), slice(None, None)) + \
            get_yield(other_h, slice(si, None), slice(None, None))
        S_err = np.sqrt(sig_h[slice(si, None), slice(None, None)].sum(flow=False).variance)
        B_err = np.sqrt(
            abcd_h[slice(si, None), slice(None, None)].sum(flow=False).variance +
            other_h[slice(si, None), slice(None, None)].sum(flow=False).variance
        )
        significance = S / np.sqrt(B) if B > 0 else np.nan
        results.append({"score_cut": score_edges[si], "S": S, "B": B, "significance": significance, "S_err": S_err, "B_err": B_err})
    results_sorted = sorted(results, key=lambda x: x["significance"] if not np.isnan(x["significance"]) else -np.inf, reverse=True)
    sig_rank = {r["score_cut"]: rank for rank, r in enumerate(results_sorted)}
    for r in list(reversed(results))[:30]:
        rank = sig_rank[r["score_cut"]]
        print(f"  sig_rank={rank:03d}  score>{r['score_cut']:.4f}  S={r['S']:.4f}±{r['S_err']:.4f}  B={r['B']:.2f}±{r['B_err']:.2f}  S/sqrt(B)={r['significance']:.4f}")
    best = results_sorted[0]
    print("\n" + "="*50)
    print("BEST SCORE-ONLY CUT")
    print("="*50)
    print(f"  Score cut    : {best['score_cut']:.4f}")
    print(f"  S            : {best['S']:.4f}")
    print(f"  B            : {best['B']:.2f}")
    print(f"  S/sqrt(B)    : {best['significance']:.4f}")
    print("="*50 + "\n")
    return results_sorted, best


def plot_score_only_scan(results, best, output_path):
    score_cuts   = [r["score_cut"]    for r in results]
    significance = [r["significance"] for r in results]
    S            = [r["S"]            for r in results]
    B            = [r["B"]            for r in results]
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(score_cuts, significance, marker="o", markersize=3, linewidth=1, color="tab:blue", label="S/sqrt(B)")
    axes[0].axvline(best["score_cut"], color="red", linewidth=2, linestyle="--", label=f"Best cut: {best['score_cut']:.3f}, S/sqrt(B)={best['significance']:.3f}")
    axes[0].set_ylabel("S/sqrt(B)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(score_cuts, S, marker="o", markersize=3, linewidth=1, color="tab:red",   label="S")
    axes[1].plot(score_cuts, B, marker="o", markersize=3, linewidth=1, color="tab:green", label="B")
    axes[1].axvline(best["score_cut"], color="red", linewidth=2, linestyle="--", label=f"Best cut: {best['score_cut']:.3f}")
    axes[1].set_ylabel("Yield")
    axes[1].set_xlabel("Score cut")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved {output_path}")


def plot_1d_stack(histo_sig, histo_dy, histo_ttbar, histo_otherbkg, axis_name, output_path):
    sig_h   = histo_sig[{"process_grp": sum}]
    dy_h    = histo_dy[{"process_grp": sum}]
    ttbar_h = histo_ttbar[{"process_grp": sum}]
    other_h = histo_otherbkg[{"process_grp": sum}]
    sig_proj   = sig_h.project(axis_name)
    dy_proj    = dy_h.project(axis_name)
    ttbar_proj = ttbar_h.project(axis_name)
    other_proj = other_h.project(axis_name)
    edges = sig_proj.axes[axis_name].edges
    sig_vals   = sig_proj.values(flow=False)
    dy_vals    = dy_proj.values(flow=False)
    ttbar_vals = ttbar_proj.values(flow=False)
    other_vals = other_proj.values(flow=False)
    total_bkg  = dy_vals.sum() + ttbar_vals.sum() + other_vals.sum()
    sig_scaled = sig_vals * (total_bkg / sig_vals.sum()) if sig_vals.sum() > 0 else sig_vals
    scale_factor = total_bkg / sig_vals.sum() if sig_vals.sum() > 0 else 1.0
    stack_dy    = other_vals + ttbar_vals + dy_vals
    stack_ttbar = other_vals + ttbar_vals
    stack_other = other_vals
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.stairs(stack_dy,    edges, fill=True, color="tab:blue",   label="DY",        baseline=stack_ttbar)
    ax.stairs(stack_ttbar, edges, fill=True, color="tab:orange", label="ttbar",     baseline=stack_other)
    ax.stairs(stack_other, edges, fill=True, color="tab:green",  label="Other bkg", baseline=0)
    ax.stairs(sig_scaled,  edges, linewidth=2, color="red", linestyle="--", label=f"Signal (x{scale_factor:.1f})")
    ax.set_xlabel(axis_name)
    ax.set_ylabel("Yield")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved {output_path}")


def plot_abcd_regions(score_edges, mjj_edges, bkg_vals, score_cut, mjj_cut, constrain_var, title, cbar_label, output_path, extra_text="", histo_sig=None, histo_abcdbkg=None, histo_otherbkg=None, histo_dat=None):
    """Plot the 2D ABCD regions for a given set of cuts and background values."""
    score_mid_lo = score_edges[0] + (score_cut - score_edges[0]) / 2
    score_mid_hi = score_cut      + (score_edges[-1] - score_cut) / 2
    mjj_mid_lo   = mjj_edges[0]   + (mjj_cut - mjj_edges[0]) / 2
    mjj_mid_hi   = mjj_cut        + (mjj_edges[-1] - mjj_cut) / 2

    si = int(np.searchsorted(score_edges, score_cut))
    mj = int(np.searchsorted(mjj_edges,   mjj_cut))

    # Compute region yields with uncertainties
    region_text = ""
    if histo_sig is not None and histo_abcdbkg is not None and histo_otherbkg is not None:
        sig_h   = histo_sig[{"process_grp": sum}]
        abcd_h  = histo_abcdbkg[{"process_grp": sum}]
        other_h = histo_otherbkg[{"process_grp": sum}]

        def get_val_err(h, s_slice, m_slice):
            sub = h[s_slice, m_slice]
            val = sub.sum(flow=False).value
            err = np.sqrt(sub.sum(flow=False).variance)
            return val, err

        region_text += "MC:\n"
        for region, s_slice, m_slice in [
            ("A", slice(si, None), slice(mj, None)),
            ("B", slice(None, si), slice(mj, None)),
            ("C", slice(si, None), slice(None, mj)),
            ("D", slice(None, si), slice(None, mj)),
        ]:
            s,   s_err   = get_val_err(sig_h,   s_slice, m_slice)
            ab,  ab_err  = get_val_err(abcd_h,  s_slice, m_slice)
            ot,  ot_err  = get_val_err(other_h, s_slice, m_slice)
            tot, tot_err = ab + ot, np.sqrt(ab_err**2 + ot_err**2)
            region_text += (
                f"{region}: sig={s:.4f}+-{s_err:.4f}  abcdbkg={ab:.4f}+-{ab_err:.4f}"
                f"  otherbkg={ot:.4f}+-{ot_err:.4f}  totbkg={tot:.4f}+-{tot_err:.4f}\n"
            )
        # Total across all regions
        total_sig,     total_sig_err = get_val_err(sig_h,   slice(None), slice(None))
        total_ab,      total_ab_err  = get_val_err(abcd_h,  slice(None), slice(None))
        total_ot,      total_ot_err  = get_val_err(other_h, slice(None), slice(None))
        total_tot     = total_ab + total_ot
        total_tot_err = np.sqrt(total_ab_err**2 + total_ot_err**2)
        region_text += (
            f"A+B+C+D: sig={total_sig:.4f}+-{total_sig_err:.4f}  abcdbkg={total_ab:.4f}+-{total_ab_err:.4f}"
            f"  otherbkg={total_ot:.4f}+-{total_ot_err:.4f}  totbkg={total_tot:.4f}+-{total_tot_err:.4f}\n"
        )
        if histo_dat is not None:
            region_text += "\nData:\n"
            dat_h = histo_dat[{"process_grp": sum}]
            for region, s_slice, m_slice in [
                ("B", slice(None, si), slice(mj, None)),
                ("C", slice(si, None), slice(None, mj)),
                ("D", slice(None, si), slice(None, mj)),
            ]:
                d,   d_err   = get_val_err(dat_h,   s_slice, m_slice)
                ab,  ab_err  = get_val_err(abcd_h,  s_slice, m_slice)
                ot,  ot_err  = get_val_err(other_h, s_slice, m_slice)
                tot, tot_err = ab + ot, np.sqrt(ab_err**2 + ot_err**2)
                ratio = d / tot if tot > 0 else np.nan
                ratio_err = (d_err / tot) * np.sqrt(1 + (ratio * tot_err / tot)**2) if tot > 0 else np.nan
                region_text += (
                    f"{region}: data={d:.1f}+-{d_err:.1f}  totbkg={tot:.4f}+-{tot_err:.4f}  data/totbkg={ratio:.3f}+-{ratio_err:.3f}\n"
                )
            B_dat, B_dat_err = get_val_err(dat_h, slice(None, si), slice(mj, None))
            C_dat, C_dat_err = get_val_err(dat_h, slice(si, None), slice(None, mj))
            D_dat, D_dat_err = get_val_err(dat_h, slice(None, si), slice(None, mj))

            if D_dat > 0:
                A_est_dat = B_dat * C_dat / D_dat
                A_est_dat_err = A_est_dat * np.sqrt(
                    (B_dat_err / B_dat)**2 + (C_dat_err / C_dat)**2 + (D_dat_err / D_dat)**2
                ) if (B_dat > 0 and C_dat > 0) else 0.0
            else:
                A_est_dat = np.nan
                A_est_dat_err = np.nan

            A_mc_tot, A_mc_tot_err = get_val_err(abcd_h, slice(si, None), slice(mj, None))
            A_ot,     A_ot_err     = get_val_err(other_h, slice(si, None), slice(mj, None))
            A_mc_total     = A_mc_tot + A_ot
            A_mc_total_err = np.sqrt(A_mc_tot_err**2 + A_ot_err**2)

            region_text += (
                f"A (data est B*C/D): {A_est_dat:.4f}+-{A_est_dat_err:.4f}\n"
                f"A (MC truth total): {A_mc_total:.4f}+-{A_mc_total_err:.4f}\n"
            )

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.pcolormesh(score_edges, mjj_edges, bkg_vals.T, cmap="Blues")
    plt.colorbar(im, ax=ax, label=cbar_label)
    ax.axvline(score_cut, color="red",    linewidth=2, linestyle="--", label=f"score > {score_cut:.3f}")
    ax.axhline(mjj_cut,   color="orange", linewidth=2, linestyle="--", label=f"{constrain_var} > {mjj_cut:.0f} GeV")
    ax.text(0.02, 0.98, extra_text,  fontsize=9, transform=ax.transAxes, va="top")
    ax.text(0.02, 0.22, region_text, fontsize=7, transform=ax.transAxes, va="top", family="monospace")
    ax.text(score_mid_hi, mjj_mid_hi, "A (SR)", ha="center", va="center", color="red",   fontsize=12, fontweight="bold")
    ax.text(score_mid_lo, mjj_mid_hi, "B",      ha="center", va="center", color="black", fontsize=12, fontweight="bold")
    ax.text(score_mid_hi, mjj_mid_lo, "C",      ha="center", va="center", color="black", fontsize=12, fontweight="bold")
    ax.text(score_mid_lo, mjj_mid_lo, "D",      ha="center", va="center", color="black", fontsize=12, fontweight="bold")
    ax.set_xlim(score_edges[0], score_edges[-1])
    ax.set_ylim(mjj_edges[0],   mjj_edges[-1])
    ax.set_xlabel("DNN score")
    ax.set_ylabel(f"{constrain_var}")
    ax.set_title(title, fontsize=9)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved {output_path}")


def plot_mjj_score_slices(histo, tag, constrain_var, output_dir="abcd_scan_plots"):
    """Plot mjj distribution in equal score slices for a single sample."""
    bkg_h = histo[{"process_grp": sum}]
    score_edges = bkg_h.axes["dnn_score"].edges
    mjj_edges   = bkg_h.axes[constrain_var].edges
    bkg_vals    = bkg_h.values(flow=False)
    n_score = len(score_edges) - 1
    slice_edges = np.linspace(0, n_score, 5, dtype=int)
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
    ax.set_xlabel(f"{constrain_var}")
    ax.set_ylabel("Normalized yield")
    ax.set_title(f"mjj distribution in score slices ({tag})\nShould be similar if decorrelated")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/mjj_score_slices_{tag}.png", dpi=150)
    plt.close()
    print(f"Saved {output_dir}/mjj_score_slices_{tag}.png")


def plot_mjj_score_slices_optimized(histo, tag, best_score_cut, constrain_var, output_dir="abcd_scan_plots"):
    """Plot mjj distribution in equal-yield score slices around the best cut for a single sample."""
    bkg_h = histo[{"process_grp": sum}]
    score_edges = bkg_h.axes["dnn_score"].edges
    mjj_edges   = bkg_h.axes[constrain_var].edges
    bkg_vals    = bkg_h.values(flow=False)
    n_score = len(score_edges) - 1
    best_score_bin = int(np.clip(np.searchsorted(score_edges, best_score_cut), 1, n_score - 1))

    def find_equal_yield_split(lo_bin, hi_bin):
        yields = bkg_vals[lo_bin:hi_bin, :].sum(axis=1)
        cumsum = np.cumsum(yields)
        split_idx = int(np.argmin(np.abs(cumsum - cumsum[-1] / 2)))
        return lo_bin + int(np.clip(split_idx + 1, 1, hi_bin - lo_bin - 1))
    lo_split = find_equal_yield_split(0, best_score_bin)
    hi_split = find_equal_yield_split(best_score_bin, n_score)
    high_side_ok = (hi_split > best_score_bin) and (hi_split < n_score - 1)
    if not high_side_ok:
        print(f"WARNING ({tag}): not enough bins on high score side to subdivide, using single slice")
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
        score_lo    = score_edges[lo]
        score_hi    = score_edges[hi]
        total_yield = bkg_vals[lo:hi, :].sum()
        ax.stairs(mjj_proj, mjj_edges, linewidth=2, color=color,
                  label=f"{label} [{score_lo:.2f}, {score_hi:.2f}) yield={total_yield:.1f}")
    ax.set_xlabel(f"{constrain_var}")
    ax.set_ylabel("Normalized yield")
    ax.set_title(
        f"mjj distribution in optimized score slices ({tag})\n"
        f"Split at best score cut ({best_score_cut:.3f}), then equal-yield subdivisions"
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/mjj_score_slices_optimized_{tag}.png", dpi=150)
    plt.close()
    print(f"Saved {output_dir}/mjj_score_slices_optimized_{tag}.png")


def plot_best_working_point(histo_sig, histo_dy, histo_ttbar, histo_abcdbkg, histo_otherbkg, histo_dat, results, constrain_var, output_dir="abcd_scan_plots", guardrails={}):
    top_indices = get_top_scan_indices(results, n_top=1, **guardrails)
    best_idx = top_indices[0]
    best_i, best_j = np.unravel_index(best_idx, results["significance"].shape)
    best_score_cut    = results["score_cuts"][best_i]
    best_mjj_cut      = results["mjj_cuts"][best_j]
    best_sig          = results["significance"][best_i, best_j]
    best_S            = results["S"][best_i, best_j]
    best_B_abcd_true  = results["B_abcd_true"][best_i, best_j]
    best_B_abcd_est   = results["B_abcd_est"][best_i, best_j]
    best_B_other      = results["B_other"][best_i, best_j]
    best_B_total      = results["B_total"][best_i, best_j]
    best_closure      = results["closure"][best_i, best_j]
    best_sig_true     = best_S / np.sqrt(best_B_abcd_true + best_B_other) if (best_B_abcd_true + best_B_other) > 0 else np.nan
    print("\n" + "="*50)
    print("BEST WORKING POINT (max S/sqrt(B_total))")
    print("="*50)
    print(f"  Score cut             : {best_score_cut:.4f}  (scan index {best_i})")
    print(f"  mjj cut               : {best_mjj_cut:.1f} GeV  (scan index {best_j})")
    print(f"  S                     : {best_S:.4f}")
    print(f"  B_abcd_true (MC)      : {best_B_abcd_true:.2f}")
    print(f"  B_abcd_est (B*C/D)    : {best_B_abcd_est:.2f}")
    print(f"  B_other (MC)          : {best_B_other:.2f}")
    print(f"  B_total               : {best_B_total:.2f}")
    print(f"  S/sqrt(B_total)       : {best_sig:.4f}")
    print(f"  S/sqrt(B_true)        : {best_sig_true:.4f}")
    print(f"  ABCD bkg Closure      : {best_closure:.4f}")
    print("="*50 + "\n")
    sig_h     = histo_sig[{"process_grp": sum}]
    abcd_h    = histo_abcdbkg[{"process_grp": sum}]
    other_h   = histo_otherbkg[{"process_grp": sum}]
    score_edges  = sig_h.axes["dnn_score"].edges
    mjj_edges    = sig_h.axes[constrain_var].edges
    abcdbkg_vals = abcd_h.values(flow=False)
    allbkg_vals  = abcdbkg_vals + other_h.values(flow=False)
    best_S_err           = results["S_err"][best_i, best_j]
    best_B_abcd_true_err = results["B_abcd_true_err"][best_i, best_j]
    best_B_abcd_est_err  = results["B_abcd_est_err"][best_i, best_j]
    best_B_other_err     = results["B_other_err"][best_i, best_j]
    best_B_total_err     = results["B_total_err"][best_i, best_j]
    abcd_closure_sigma = (best_B_abcd_est - best_B_abcd_true) / np.sqrt(best_B_abcd_true_err**2 + best_B_abcd_est_err**2) if (best_B_abcd_true_err**2 + best_B_abcd_est_err**2) > 0 else np.nan
    extra_text = (
        f"Sig in A: {best_S:.4f} +/- {best_S_err:.4f}\n"
        f"Est ABCD bkg in A: {best_B_abcd_est:.4f} +/- {best_B_abcd_est_err:.4f}\n"
        f"Truth ABCD bkg in A: {best_B_abcd_true:.4f} +/- {best_B_abcd_true_err:.4f}\n"
        f"Other bkg in A: {best_B_other:.4f} +/- {best_B_other_err:.4f}\n"
        f"Tot bkg in A: {best_B_total:.4f} +/- {best_B_total_err:.4f}\n"
        f"Closure of ABCD bkgs: {abcd_closure_sigma:.4f} s.d."
    )
    os.makedirs(output_dir, exist_ok=True)
    for vals, tag, cbar_label in [
        (allbkg_vals, "allbkg", "Total background yield (DY + ttbar + other MC)"),
    ]:
        plot_abcd_regions(
            score_edges, mjj_edges, vals,
            best_score_cut, best_mjj_cut,
            constrain_var,
            title=f"Best working point ({tag}): score>{best_score_cut:.3f}, mjj>{best_mjj_cut:.0f} GeV",
            cbar_label=cbar_label,
            output_path=f"{output_dir}/best_working_point_{tag}.png",
            extra_text=extra_text,
            histo_sig=histo_sig,
            histo_abcdbkg=histo_abcdbkg,
            histo_otherbkg=histo_otherbkg,
            histo_dat=histo_dat,
        )
    write_single_datacard(
        best_S, best_B_total,
        best_score_cut, best_mjj_cut,
        output_path=f"{output_dir}/best_working_point_allbkg.txt",
    )


def write_abcd_datacards(histo_sig, histo_abcdbkg, histo_otherbkg, histo_dat, results, constrain_var, output_dir="abcd_scan_plots", n_top=5, min_significance=0, guardrails={}):
    os.makedirs(output_dir, exist_ok=True)
    top_indices = get_top_scan_indices(results, n_top=n_top, **guardrails)
    sig_h     = histo_sig[{"process_grp": sum}]
    abcd_h    = histo_abcdbkg[{"process_grp": sum}]
    other_h   = histo_otherbkg[{"process_grp": sum}]
    score_edges  = sig_h.axes["dnn_score"].edges
    mjj_edges    = sig_h.axes[constrain_var].edges
    abcdbkg_vals = abcd_h.values(flow=False)
    allbkg_vals  = abcdbkg_vals + other_h.values(flow=False)
    for rank, flat_idx in enumerate(top_indices):
        i, j = np.unravel_index(flat_idx, results["significance"].shape)
        score_cut = results["score_cuts"][i]
        mjj_cut   = results["mjj_cuts"][j]
        sig_val   = results["significance"][i, j]
        si = int(np.searchsorted(score_edges, score_cut))
        mj = int(np.searchsorted(mjj_edges,   mjj_cut))
        A_sig        = get_yield(sig_h,   slice(si, None), slice(mj, None))
        A_abcd       = get_yield(abcd_h,  slice(si, None), slice(mj, None))
        B_abcd       = get_yield(abcd_h,  slice(None, si), slice(mj, None))
        C_abcd       = get_yield(abcd_h,  slice(si, None), slice(None, mj))
        D_abcd       = get_yield(abcd_h,  slice(None, si), slice(None, mj))
        A_other      = get_yield(other_h, slice(si, None), slice(mj, None))
        abcd_est = (B_abcd * C_abcd / D_abcd) if D_abcd > 0 else 0.0
        A_bkg    = abcd_est + A_other
        A_obs    = A_sig + A_bkg
        fname_base = f"dc_wp{rank}"
        A_sig_err    = results["S_err"][i, j]
        A_abcd_err   = results["B_abcd_true_err"][i, j]
        abcd_est_err = results["B_abcd_est_err"][i, j]
        A_other_err  = results["B_other_err"][i, j]
        A_bkg_err    = results["B_total_err"][i, j]
        A_abcd       = results["B_abcd_true"][i, j]
        denom        = np.sqrt(A_abcd_err**2 + abcd_est_err**2)
        closure_sd   = (abcd_est - A_abcd) / denom if denom > 0 else np.nan
        fpath = os.path.join(output_dir, f"{fname_base}.txt")
        with open(fpath, "w") as f:
            f.write(f"# Counting experiment datacard: rank={rank}, score>{score_cut:.4f}, mjj>{mjj_cut:.1f} GeV\n")
            f.write(f"# S/sqrt(B_total)={sig_val:.4f}\n")
            f.write(f"# A_abcd_true={A_abcd:.2f}, A_abcd_est={abcd_est:.2f}, A_other={A_other:.2f}\n\n")
            f.write( "# Details:\n")
            f.write(f"#   Sig in A: {A_sig:.4f} +/- {A_sig_err:.4f}\n")
            f.write(f"#   Est ABCD bkg in A: {abcd_est:.4f} +/- {abcd_est_err:.4f}\n")
            f.write(f"#   Truth ABCD bkg in A: {A_abcd:.4f} +/- {A_abcd_err:.4f}\n")
            f.write(f"#   Other bkg in A: {A_other:.4f} +/- {A_other_err:.4f}\n")
            f.write(f"#   Tot bkg in A: {A_bkg:.4f} +/- {A_bkg_err:.4f}\n")
            f.write(f"#   Closure of ABCD bkgs: {closure_sd:.4f} s.d.\n\n")
            f.write( "imax 1  number of channels\n")
            f.write( "jmax 1  number of backgrounds\n")
            f.write( "kmax 0  number of nuisance parameters\n")
            f.write( "-" * 60 + "\n")
            f.write( "bin          A\n")
            f.write(f"observation  {A_obs:.4f}\n")
            f.write( "-" * 60 + "\n")
            f.write( "bin          A       A\n")
            f.write( "process      sig     bkg\n")
            f.write( "process      0       1\n")
            f.write(f"rate         {A_sig:.4f}  {A_bkg:.4f}\n")
            f.write( "-" * 60 + "\n")
        print(f"Wrote {fpath}")
        extra_text = (
            f"Sig in A: {A_sig:.4f} +/- {A_sig_err:.4f}\n"
            f"Est ABCD bkg in A: {abcd_est:.4f} +/- {abcd_est_err:.4f}\n"
            f"Truth ABCD bkg in A: {A_abcd:.4f} +/- {A_abcd_err:.4f}\n"
            f"Other bkg in A: {A_other:.4f} +/- {A_other_err:.4f}\n"
            f"Tot bkg in A: {A_bkg:.4f} +/- {A_bkg_err:.4f}\n"
            f"Closure of ABCD bkgs: {closure_sd:.4f} s.d."
        )
        plot_abcd_regions(
            score_edges, mjj_edges, allbkg_vals,
            score_cut, mjj_cut,
            constrain_var,
            title=(
                f"rank={rank}: score>{score_cut:.3f}, mjj>{mjj_cut:.0f} GeV\n"
                f"S={A_sig:.3f}, B_abcd_est={abcd_est:.1f}, B_other={A_other:.1f}, "
                f"B_total={A_bkg:.1f}, S/sqrt(B_total)={sig_val:.3f}"
            ),
            cbar_label="Total background yield (DY + ttbar + other MC)",
            output_path=os.path.join(output_dir, f"{fname_base}.png"),
            extra_text=extra_text,
            histo_sig=histo_sig,
            histo_abcdbkg=histo_abcdbkg,
            histo_otherbkg=histo_otherbkg,
        )


def plot_abcd_2d_snapshots(histo_sig, histo_dy, histo_ttbar, histo_abcdbkg, histo_otherbkg, results, constrain_var, output_dir="abcd_snapshots", make_scan_blocks=True):
    os.makedirs(output_dir, exist_ok=True)
    sig_h   = histo_sig[{"process_grp": sum}]
    dy_h    = histo_dy[{"process_grp": sum}]
    ttbar_h = histo_ttbar[{"process_grp": sum}]
    abcd_h  = histo_abcdbkg[{"process_grp": sum}]
    other_h = histo_otherbkg[{"process_grp": sum}]
    n_score = len(results["score_cuts"])
    n_mjj   = len(results["mjj_cuts"])
    score_edges  = sig_h.axes["dnn_score"].edges
    mjj_edges    = sig_h.axes[constrain_var].edges
    dy_vals      = dy_h.values(flow=False)
    ttbar_vals   = ttbar_h.values(flow=False)
    abcdbkg_vals = abcd_h.values(flow=False)
    allbkg_vals  = abcdbkg_vals + other_h.values(flow=False)
    sig_vals     = sig_h.values(flow=False)
    other_vals   = other_h.values(flow=False)

    def _make_overview_plots(vals, cbar_label, fname_prefix,cmap="Blues"):
        for scale, norm, suffix in [("linear", None, "lin"), ("log", matplotlib.colors.LogNorm(), "log")]:
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.pcolormesh(score_edges, mjj_edges, vals.T, cmap=cmap, norm=norm)
            plt.colorbar(im, ax=ax, label=cbar_label)
            score_centers = (score_edges[:-1] + score_edges[1:]) / 2
            mjj_centers   = (mjj_edges[:-1]   + mjj_edges[1:])   / 2
            profile = np.zeros(len(score_centers))
            profile_err = np.zeros(len(score_centers))
            for si in range(len(score_centers)):
                col = vals[si, :]
                total = col.sum()
                if total > 0:
                    mean = np.average(mjj_centers, weights=col)
                    variance = np.average((mjj_centers - mean) ** 2, weights=col)
                    n_eff = (total ** 2 / np.sum(col ** 2)) if np.sum(col ** 2) > 0 else 1.0
                    profile[si] = mean
                    profile_err[si] = np.sqrt(variance / n_eff)
                else:
                    profile[si] = np.nan
                    profile_err[si] = np.nan

            ax.errorbar(score_centers, profile, yerr=profile_err, color="red", linewidth=2, marker="o", markersize=4, label="Mean", capsize=2)
            ax.legend(loc="upper right", fontsize=8)
            ax.set_xlabel("DNN score")
            ax.set_ylabel(f"{constrain_var}")
            ax.set_title(f"{fname_prefix} 2D histogram (no cuts, {scale})")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{fname_prefix}_{suffix}.png", dpi=150)
            plt.close()
            print(f"Saved {output_dir}/{fname_prefix}_{suffix}.png")
    _make_overview_plots(dy_vals,      "DY yield",                            "scan_point_overview_dy")
    _make_overview_plots(ttbar_vals,   "ttbar yield",                         "scan_point_overview_ttbar")
    _make_overview_plots(abcdbkg_vals, "ABCD background yield (DY + ttbar)",  "scan_point_overview_abcdbkg")
    _make_overview_plots(allbkg_vals,  "Total background yield",              "scan_point_overview_allbkg")
    _make_overview_plots(sig_vals,     "Signal yield",                        "scan_point_overview_sig", cmap="Greens")
    _make_overview_plots(other_vals, "Other background yield", "scan_point_overview_otherbkg")

    if make_scan_blocks:
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
            ax.set_ylabel(f"{constrain_var}")
            ax.set_title(f"Score cut block {i}: score > {score_cut:.2f}\nmjj cuts shown as horizontal lines")
            ax.legend(loc="upper right", fontsize=8)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/scan_block_{i:04d}.png", dpi=100)
            plt.close()
            print(f"Saved scan block {i}")


def get_top_scan_indices(results, n_top=1, min_significance=0.0, max_closure_sd=np.inf, min_S=0.0, max_B_total=np.inf, min_bcd_yield=0.0):
    sig_flat             = results["significance"].flatten()
    S_flat               = results["S"].flatten()
    B_total_flat         = results["B_total"].flatten()
    B_abcd_true_err_flat = results["B_abcd_true_err"].flatten()
    B_abcd_est_err_flat  = results["B_abcd_est_err"].flatten()
    B_abcd_true_flat     = results["B_abcd_true"].flatten()
    B_abcd_est_flat      = results["B_abcd_est"].flatten()
    B_flat               = results["B_abcd_yield"].flatten()
    C_flat               = results["C_abcd_yield"].flatten()
    D_flat               = results["D_abcd_yield"].flatten()

    denom = np.sqrt(B_abcd_true_err_flat**2 + B_abcd_est_err_flat**2)
    closure_sd_flat = np.where(denom > 0, np.abs(B_abcd_est_flat - B_abcd_true_flat) / denom, np.nan)

    valid_mask = (
        ~np.isnan(sig_flat)                 &
        (sig_flat     >= min_significance)  &
        (S_flat       >= min_S)             &
        (B_total_flat <= max_B_total)       &
        (~np.isnan(closure_sd_flat))        &
        (closure_sd_flat <= max_closure_sd) &
        (B_flat >= min_bcd_yield)           &
        (C_flat >= min_bcd_yield)           &
        (D_flat >= min_bcd_yield)
    )

    valid_indices = np.where(valid_mask)[0]
    top_indices = valid_indices[np.argsort(sig_flat[valid_indices])[::-1]][:n_top]
    print(f"  {len(valid_indices)} scan points pass guard rails, returning top {len(top_indices)}")
    return top_indices

def do_abcd_scan(histo_sig, histo_abcdbkg, histo_otherbkg, constrain_axis_name, score_axis_name="dnn_score"):
    sig_h   = histo_sig[{"process_grp": sum}]
    abcd_h  = histo_abcdbkg[{"process_grp": sum}]
    other_h = histo_otherbkg[{"process_grp": sum}]
    score_edges = sig_h.axes[score_axis_name].edges
    mjj_edges   = sig_h.axes[constrain_axis_name].edges
    n_score = len(score_edges) - 1
    n_mjj   = len(mjj_edges) - 1
    score_cut_bins = list(range(1, n_score))
    mjj_cut_bins   = list(range(1, n_mjj))
    n_scan_score = len(score_cut_bins)
    n_scan_mjj   = len(mjj_cut_bins)
    closure          = np.zeros((n_scan_score, n_scan_mjj))
    significance     = np.zeros((n_scan_score, n_scan_mjj))
    S_arr            = np.zeros((n_scan_score, n_scan_mjj))
    B_abcd_true      = np.zeros((n_scan_score, n_scan_mjj))
    B_abcd_est       = np.zeros((n_scan_score, n_scan_mjj))
    B_other          = np.zeros((n_scan_score, n_scan_mjj))
    B_total          = np.zeros((n_scan_score, n_scan_mjj))
    S_err_arr        = np.zeros((n_scan_score, n_scan_mjj))
    B_abcd_true_err  = np.zeros((n_scan_score, n_scan_mjj))
    B_abcd_est_err   = np.zeros((n_scan_score, n_scan_mjj))
    B_other_err      = np.zeros((n_scan_score, n_scan_mjj))
    B_total_err      = np.zeros((n_scan_score, n_scan_mjj))
    B_total_err      = np.zeros((n_scan_score, n_scan_mjj))
    B_abcd_yield     = np.zeros((n_scan_score, n_scan_mjj))
    C_abcd_yield     = np.zeros((n_scan_score, n_scan_mjj))
    D_abcd_yield     = np.zeros((n_scan_score, n_scan_mjj))
    for i, si in enumerate(score_cut_bins):
        for j, mj in enumerate(mjj_cut_bins):
            A_sig   = get_yield(sig_h,   slice(si, None), slice(mj, None))
            A_abcd  = get_yield(abcd_h,  slice(si, None), slice(mj, None))
            B_abcd  = get_yield(abcd_h,  slice(None, si), slice(mj, None))
            C_abcd  = get_yield(abcd_h,  slice(si, None), slice(None, mj))
            D_abcd  = get_yield(abcd_h,  slice(None, si), slice(None, mj))
            A_other = get_yield(other_h, slice(si, None), slice(mj, None))
            A_other    = get_yield(other_h, slice(si, None), slice(mj, None))
            B_otherbkg = get_yield(other_h, slice(None, si), slice(mj, None))
            C_otherbkg = get_yield(other_h, slice(si, None), slice(None, mj))
            D_otherbkg = get_yield(other_h, slice(None, si), slice(None, mj))
            A_sig_err   = np.sqrt(sig_h[slice(si, None),   slice(mj, None)].sum(flow=False).variance)
            A_abcd_err  = np.sqrt(abcd_h[slice(si, None),  slice(mj, None)].sum(flow=False).variance)
            B_abcd_err  = np.sqrt(abcd_h[slice(None, si),  slice(mj, None)].sum(flow=False).variance)
            C_abcd_err  = np.sqrt(abcd_h[slice(si, None),  slice(None, mj)].sum(flow=False).variance)
            D_abcd_err  = np.sqrt(abcd_h[slice(None, si),  slice(None, mj)].sum(flow=False).variance)
            A_other_err = np.sqrt(other_h[slice(si, None),  slice(mj, None)].sum(flow=False).variance)
            if D_abcd == 0:
                closure[i, j]      = np.nan
                significance[i, j] = np.nan
                S_arr[i, j]        = np.nan
                B_abcd_true[i, j]  = np.nan
                B_abcd_est[i, j]   = np.nan
                B_other[i, j]      = np.nan
                B_total[i, j]      = np.nan
                continue
            abcd_est = B_abcd * C_abcd / D_abcd
            abcd_est_err = abcd_est * np.sqrt(
                (B_abcd_err / B_abcd)**2 + (C_abcd_err / C_abcd)**2 + (D_abcd_err / D_abcd)**2
            ) if (B_abcd > 0 and C_abcd > 0) else 0.0
            b_total     = abcd_est + A_other
            b_total_err = np.sqrt(abcd_est_err**2 + A_other_err**2)
            S_arr[i, j]        = A_sig
            B_abcd_true[i, j]  = A_abcd
            B_abcd_est[i, j]   = abcd_est
            B_other[i, j]      = A_other
            B_total[i, j]      = b_total
            closure[i, j]      = (abcd_est / A_abcd) if A_abcd > 0 else np.nan
            significance[i, j] = (A_sig / np.sqrt(b_total)) if b_total > 0 else np.nan
            S_err_arr[i, j]       = A_sig_err
            B_abcd_true_err[i, j] = A_abcd_err
            B_abcd_est_err[i, j]  = abcd_est_err
            B_other_err[i, j]     = A_other_err
            B_total_err[i, j]     = b_total_err
            B_total_err[i, j]     = b_total_err
            B_abcd_yield[i, j]    = B_abcd + B_otherbkg
            C_abcd_yield[i, j]    = C_abcd + C_otherbkg
            D_abcd_yield[i, j]    = D_abcd + D_otherbkg
    return {
        "closure"         : closure,
        "significance"    : significance,
        "S"               : S_arr,
        "B_abcd_true"     : B_abcd_true,
        "B_abcd_est"      : B_abcd_est,
        "B_other"         : B_other,
        "B_total"         : B_total,
        "score_cuts"      : score_edges[score_cut_bins],
        "mjj_cuts"        : mjj_edges[mjj_cut_bins],
        "S_err"           : S_err_arr,
        "B_abcd_true_err" : B_abcd_true_err,
        "B_abcd_est_err"  : B_abcd_est_err,
        "B_other_err"     : B_other_err,
        "B_total_err"     : B_total_err,
        "B_total_err"  : B_total_err,
        "B_abcd_yield" : B_abcd_yield,
        "C_abcd_yield" : C_abcd_yield,
        "D_abcd_yield" : D_abcd_yield,
    }


def plot_abcd_scan_panels(results, output_path):
    n_score = len(results["score_cuts"])
    n_mjj   = len(results["mjj_cuts"])
    scan_points       = []
    significance      = []
    significance_true = []
    B_abcd_true       = []
    B_abcd_est        = []
    B_other           = []
    B_total           = []
    closure           = []
    labels            = []
    for i in range(n_score):
        for j in range(n_mjj):
            scan_points.append(len(scan_points))
            significance.append(results["significance"][i, j])
            b_true_total = results["B_abcd_true"][i, j] + results["B_other"][i, j]
            significance_true.append(
                results["S"][i, j] / np.sqrt(b_true_total) if b_true_total > 0 else np.nan
            )
            B_abcd_true.append(results["B_abcd_true"][i, j])
            B_abcd_est.append(results["B_abcd_est"][i, j])
            B_other.append(results["B_other"][i, j])
            B_total.append(results["B_total"][i, j])
            closure.append(results["closure"][i, j])
            labels.append(f"s>{results['score_cuts'][i]:.2f},mjj>{results['mjj_cuts'][j]:.0f}")
    scan_points       = np.array(scan_points)
    significance      = np.array(significance)
    significance_true = np.array(significance_true)
    B_abcd_true       = np.array(B_abcd_true)
    B_abcd_est        = np.array(B_abcd_est)
    B_other           = np.array(B_other)
    B_total           = np.array(B_total)
    closure           = np.array(closure)
    fig, axes = plt.subplots(3, 1, figsize=(min(max(12, len(scan_points)//4), 80), 12), sharex=True)
    axes[0].plot(scan_points, significance,      marker="o", markersize=3, linewidth=1, color="tab:blue",   label="S/sqrt(B_total)")
    axes[0].plot(scan_points, significance_true, marker="o", markersize=3, linewidth=1, color="tab:orange", label="S/sqrt(B_true_total)")
    axes[0].set_ylabel("Significance")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(scan_points, B_abcd_true, marker="o", markersize=3, linewidth=1, color="tab:blue",   label="B_abcd true (MC)")
    axes[1].plot(scan_points, B_abcd_est,  marker="o", markersize=3, linewidth=1, color="tab:cyan",   label="B_abcd est (B*C/D)")
    axes[1].plot(scan_points, B_other,     marker="o", markersize=3, linewidth=1, color="tab:green",  label="B_other (MC)")
    axes[1].plot(scan_points, B_total,     marker="o", markersize=3, linewidth=1, color="tab:orange", label="B_total")
    axes[1].set_ylabel("Background yield in A")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(scan_points, closure, marker="o", markersize=3, linewidth=1, color="tab:green")
    axes[2].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[2].axhline(1.2, color="red",   linestyle="--", linewidth=0.8, alpha=0.5)
    axes[2].axhline(0.8, color="red",   linestyle="--", linewidth=0.8, alpha=0.5)
    axes[2].set_ylabel("B_abcd_est / B_abcd_true (ABCD bkg closure)")
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
    grp_dict = cvh.GRP_DICT_FULL_R2
    #grp_dict = cvh.GRP_DICT_FULL_R3

    # Set the options for this run (NOTE these are hard coded)
    cat_for_dnn    = "2lOSSF_nFJ1_massHi_Zp5Hp5VBSp5"
    abcd_hist_name = "abcd2d_2lH"
    #constrain_var  = "vbs_score"
    constrain_var  = "vbs_mjj"
    guardrails = {
        "min_significance" : 0.0,
        "max_closure_sd"   : 1.2,
        "min_S"            : 0.03,
        "max_B_total"      : 1e99,
        "min_bcd_yield"    : 10.0,
    }

    # Get the histo from the pkl
    histo_dict = pickle.load(gzip.open(args.pkl_file_path))
    histo = histo_dict[abcd_hist_name]
    histo = histo[{"category": cat_for_dnn}]
    histo = plt_tools.group(histo, "process", "process_grp", grp_dict)

    # Build the list of non-ABCD background group names
    other_bkg_names = []
    for grp_name in grp_dict:
        if grp_name not in ["Data", "Signal", "DY", "ttbar", 'VBSWWH_SS', 'VBSWWH_OS', 'VBSWZH', 'VBSZZH']:
            other_bkg_names.append(grp_name)

    # Hists for all the relevant groupings
    histo_sig      = histo[{"process_grp": ["Signal"]}]
    histo_dat      = histo[{"process_grp": ["Data"]}]
    histo_dy       = histo[{"process_grp": ["DY"]}]
    histo_ttbar    = histo[{"process_grp": ["ttbar"]}]
    histo_abcdbkg  = plt_tools.group(histo, "process_grp", "process_grp", {"ABCDBkg": ["DY", "ttbar"]})
    histo_otherbkg = plt_tools.group(histo, "process_grp", "process_grp", {"OtherBkg": other_bkg_names})

    # Print yields
    val_sig,      err_sig      = histo_sig.values(flow=True).sum(),      np.sqrt(histo_sig.variances(flow=True).sum())
    val_data,     err_data     = histo_dat.values(flow=True).sum(),      np.sqrt(histo_dat.variances(flow=True).sum())
    val_dy,       err_dy       = histo_dy.values(flow=True).sum(),       np.sqrt(histo_dy.variances(flow=True).sum())
    val_ttbar,    err_ttbar    = histo_ttbar.values(flow=True).sum(),    np.sqrt(histo_ttbar.variances(flow=True).sum())
    val_abcdbkg,  err_abcdbkg  = histo_abcdbkg.values(flow=True).sum(),  np.sqrt(histo_abcdbkg.variances(flow=True).sum())
    val_otherbkg, err_otherbkg = histo_otherbkg.values(flow=True).sum(), np.sqrt(histo_otherbkg.variances(flow=True).sum())
    print(f"sig: {val_sig} +- {err_sig}")
    print(f"dy: {val_dy} +- {err_dy}")
    print(f"ttbar: {val_ttbar} +- {err_ttbar}")
    print(f"otherbkg: {val_otherbkg} +- {err_otherbkg}")


    # Set the out dirs
    out_dir    = "abcd_scan_outputs_plots"
    out_dir_dc = "abcd_scan_outputs_datacards"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir_dc, exist_ok=True)
    if not os.path.exists(os.path.join(out_dir,    "index.php")): shutil.copyfile(HTML_PC, os.path.join(out_dir,    "index.php"))
    if not os.path.exists(os.path.join(out_dir_dc, "index.php")): shutil.copyfile(HTML_PC, os.path.join(out_dir_dc, "index.php"))



    ####################### Just plot some hists #######################

    # Make the stack plot, borrowing from check_vvh_hists
    years_to_prepend = ["2016postVFP","2016preVFP","2017","2018"]
    cvh.make_plots(histo_dict,grp_dict,years_to_prepend,["2lOSSF_nFJ1_massHi_Zp5Hp5VBSp5"],lepflav_bin="all",save_dir_path=out_dir,make_cat_subdirs=False,vars_to_plot=["vbs_mjj","dnn_score_2lH"])#"njets","njets_counts","vbs_mjj","dnn_score_2lH","dnn_score_2lV"])

    # Just simple make 1d plots of the score
    #plot_1d_stack(histo_sig, histo_dy, histo_ttbar, histo_otherbkg, "dnn_score_2lH", f"{out_dir}/stack_dnn_score.png")
    #plot_1d_stack(histo_sig, histo_dy, histo_ttbar, histo_otherbkg, "dnn_score_2lV", f"{out_dir}/stack_dnn_score.png")


    ####################### Scan over 1d #######################

    if 0:
        # Try to do a scan over just score
        score_only_results, score_only_best = scan_score_only(histo_sig, histo_abcdbkg, histo_otherbkg)
        plot_score_only_scan(score_only_results, score_only_best, f"{out_dir}/dc_score_only_scan.png")
        for rank, result in enumerate(score_only_results[:30]):
            score_str = f"{result['score_cut']:.2f}".replace(".", "p")
            sig_str   = f"{result['significance']:.2f}".replace(".", "p")
            fname     = f"{out_dir}/dc_score_only_rank{rank:02d}.txt"
            write_score_only_datacard(result, fname)

        # Evaluate at a fixed given score
        eval_at_fixed_cut(histo_sig, histo_abcdbkg, histo_otherbkg, score_cut=0.996, label="")



    ####################### Scan over 2d #######################

    # Decorrelation slices plots for each ABCD background sample and combined
    for histo, tag in [(histo_dy, "dy"), (histo_ttbar, "ttbar"), (histo_abcdbkg, "abcdbkg")]:
        plot_mjj_score_slices(histo, tag, constrain_var, output_dir=out_dir)

    # Run the scan
    results = do_abcd_scan(histo_sig, histo_abcdbkg, histo_otherbkg, constrain_var)
    plot_abcd_scan_panels(results, f"{out_dir}/abcd_scan_panels.png")
    plot_abcd_2d_snapshots(histo_sig, histo_dy, histo_ttbar, histo_abcdbkg, histo_otherbkg, results, constrain_var, output_dir=out_dir, make_scan_blocks=False)
    plot_best_working_point(histo_sig, histo_dy, histo_ttbar, histo_abcdbkg, histo_otherbkg, histo_dat, results, constrain_var, output_dir=out_dir, guardrails=guardrails)

    # Optimized slices plots at best working point
    best_idx = get_top_scan_indices(results, n_top=1, **guardrails)[0]
    best_i, best_j = np.unravel_index(best_idx, results["significance"].shape)
    best_score_cut = results["score_cuts"][best_i]
    for histo, tag in [(histo_dy, "dy"), (histo_ttbar, "ttbar"), (histo_abcdbkg, "abcdbkg")]:
        plot_mjj_score_slices_optimized(histo, tag, best_score_cut, constrain_var, output_dir=out_dir)

    # Write datacards
    write_abcd_datacards(histo_sig, histo_abcdbkg, histo_otherbkg, histo_dat, results, constrain_var, output_dir=out_dir_dc, n_top=200, guardrails=guardrails)



    ####################### Plot ABCD regions for a specific working point #######################

    if 0:
        my_score_cut = 0.840
        my_mjj_cut   = 960
        sig_h   = histo_sig[{"process_grp": sum}]
        abcd_h  = histo_abcdbkg[{"process_grp": sum}]
        other_h = histo_otherbkg[{"process_grp": sum}]
        score_edges  = sig_h.axes["dnn_score"].edges
        mjj_edges    = sig_h.axes[constrain_var].edges
        allbkg_vals  = abcd_h.values(flow=False) + other_h.values(flow=False)
        plot_abcd_regions(
            score_edges, mjj_edges, allbkg_vals,
            my_score_cut, my_mjj_cut,
            constrain_var,
            title=f"score>{my_score_cut:.3f}, mjj>{my_mjj_cut:.0f} GeV",
            cbar_label="Total background yield",
            output_path=f"{out_dir}/custom_wp_score{my_score_cut:.3f}_mjj{my_mjj_cut:.0f}.png",
            histo_sig=histo_sig,
            histo_abcdbkg=histo_abcdbkg,
            histo_otherbkg=histo_otherbkg,
        )

main()
