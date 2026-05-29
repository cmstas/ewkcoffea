import argparse
import pickle
import gzip
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ewkcoffea.modules.plotting_tools as plt_tools


def main():

    # Set up the command line parser
    parser = argparse.ArgumentParser()
    parser.add_argument("pkl_file_path", help = "The path to the pkl file")
    args = parser.parse_args()

    sig_procs = ['VBSZZH_c2v1p0_c3_1p0', 'VBSWWH_SS_c2v1p0_c3_1p0', 'VBSWWH_OS_c2v1p0_c3_1p0', 'VBSWZH_c2v1p0_c3_1p0']
    tt_procs = ['TTTo2L2Nu_TuneCP5_13TeV']
    dy_procs = ['DYJetsToLL_M-50_TuneCP5_13TeV']
    #tt_processes = ['TTTo2L2Nu_TuneCP5_13TeV', 'TTToSemiLeptonic_TuneCP5_13TeV', 'TTToHadronic_TuneCP5_13TeV']
    #dy_processes = ['DYJetsToLL_M-50_TuneCP5_13TeV', 'DYJetsToLL_M-10to50_TuneCP5_13TeV']
    proc_lst = sig_procs + dy_procs + tt_procs
    bkg_procs = dy_procs + tt_procs

    # Get the dictionary of histograms from the input pkl file
    histo_dict = pickle.load(gzip.open(args.pkl_file_path))
    name = args.pkl_file_path.split("/")[1][:-7] # Drops leading histos/ and trailing .pkl.gz

    cat_lst = ["2lOSSF_1fjx", "2lOSSF_1fjx_ejj3"]


    ###### Distributions ######
    for cat in cat_lst:
        for var in ["vbs_mjj", "vbs_absdetajj", "vbs_score", "fj0_mparticlenet"]:
            fig, (ax, ax_ratio) = plt.subplots(2, 1, figsize=(6, 5), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

            # Bkg
            histo_bkg = histo_dict[var][{"systematic":"nominal", "category":cat, "lepflav":sum, "process":bkg_procs}]
            histo_bkg = plt_tools.merge_overflow(histo_bkg)
            histo_bkg = plt_tools.rebin(histo_bkg,6)
            bkg_yield = sum(histo_bkg[{"process":p}].values().sum() for p in bkg_procs)

            # Sig
            histo_sig = histo_dict[var][{"systematic":"nominal", "category":cat, "lepflav":sum, "process":sig_procs}]
            histo_sig = plt_tools.merge_overflow(histo_sig)
            histo_sig = plt_tools.rebin(histo_sig,6)
            sig_yield = sum(histo_sig[{"process":p}].values().sum() for p in sig_procs)
            histo_sig_scaled = histo_sig*(bkg_yield/sig_yield)
            histo_sig_scaled_sum = histo_sig_scaled[{"process":sum}]

            # Main plot
            histo_bkg.plot1d(ax=ax, stack=True, histtype="fill")
            #histo_sig.plot1d(ax=ax, stack=True, histtype="fill")
            histo_sig_scaled.plot1d(ax=ax, stack=False, histtype="step")
            histo_sig_scaled_sum.plot1d(ax=ax, stack=False, histtype="step", color="red", linewidth=2, label="VBS VVH SM total")

            # Ratio plot: Cumulative s/sqrt(b) from right to left
            bkg_vals = sum(histo_bkg[{"process":p}].values() for p in bkg_procs)
            sig_vals = sum(histo_sig[{"process":p}].values() for p in sig_procs)
            sig_cumulative = np.cumsum(sig_vals[::-1])[::-1]
            bkg_cumulative = np.cumsum(bkg_vals[::-1])[::-1]
            s_over_sqrtb_cumulative = np.where(bkg_cumulative > 0, sig_cumulative / np.sqrt(bkg_cumulative), 0)
            bins = histo_sig.axes[var].edges
            ax_ratio.stairs(s_over_sqrtb_cumulative, bins, color="black")
            ax_ratio.set_ylabel("S/sqrt(B)")
            ax_ratio.axhline(0, color="gray", linestyle="--", linewidth=0.5)
            ax_ratio.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)
            ymax_ratio = ax_ratio.get_ylim()[1]
            ax_ratio.set_ylim(0, ymax_ratio * 1.2)
            ax_ratio.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))

            # Write text on main plot
            ax.text(0.03, 0.97, f"Tot yield: {bkg_yield+sig_yield:.3f}", transform=ax.transAxes, ha="left", va="top", fontsize=10)
            ax.text(0.03, 0.92, f"Bkg yield: {bkg_yield:.3f}", transform=ax.transAxes, ha="left", va="top", fontsize=10)
            ax.text(0.03, 0.87, f"Sig yield: {sig_yield:.3f}", transform=ax.transAxes, ha="left", va="top", fontsize=10)
            ax.text(0.03, 0.82, f"All sig is x{bkg_yield/sig_yield:.3f}", transform=ax.transAxes, ha="left", va="top", fontsize=10)
            total_s_over_sqrtb = sig_yield / np.sqrt(bkg_yield) if bkg_yield > 0 else 0
            ax.text(0.03, 0.77, f"S/sqrt(B): {total_s_over_sqrtb:.4f}", transform=ax.transAxes, ha="left", va="top", fontsize=10)

            ax.set_ylabel("Events")
            ax.legend(fontsize=6)
            plt.tight_layout()
            ymax = ax.get_ylim()[1]
            ax.set_ylim(0, ymax * 1.5)
            plt.savefig(f"{cat}__{var}.png")
            plt.close()


    ###### Truth plots ######

    thiscat = "2lOSSF_1fjx"
    procs = sig_procs
    print("tot",sum(sum(histo_dict["njets"][{"systematic":"nominal", "category":f"{thiscat}", "process":procs, "lepflav":sum}].values(flow=True))))
    print("h",  sum(sum(histo_dict["njets"][{"systematic":"nominal", "category":f"{thiscat}_fj0matchH", "process":procs, "lepflav":sum}].values(flow=True))))
    print("v1", sum(sum(histo_dict["njets"][{"systematic":"nominal", "category":f"{thiscat}_fj0matchV1", "process":procs, "lepflav":sum}].values(flow=True))))
    print("v2", sum(sum(histo_dict["njets"][{"systematic":"nominal", "category":f"{thiscat}_fj0matchV2", "process":procs, "lepflav":sum}].values(flow=True))))
    print("v",  sum(sum(histo_dict["njets"][{"systematic":"nominal", "category":f"{thiscat}_fj0matchV", "process":procs, "lepflav":sum}].values(flow=True))))
    print("n",  sum(sum(histo_dict["njets"][{"systematic":"nominal", "category":f"{thiscat}_fj0noMatch", "process":procs, "lepflav":sum}].values(flow=True))))

    fig, ax = plt.subplots(figsize=(8, 5))
    cat_lst = ["2lOSSF_1fjx", "2lOSSF_1fjx_ejj3"]
    for cat in cat_lst:
        categories = {
            "H-matched"   : f"{cat}_fj0matchH",
            "V-matched"   : f"{cat}_fj0matchV",
            "Unmatched"   : f"{cat}_fj0noMatch",
        }

        for normalize in [True, False]:
            fig, ax = plt.subplots(figsize=(8, 5))
            for label, cat in categories.items():
                histo = histo_dict["fj0_mparticlenet"][{"systematic":"nominal", "category":cat, "lepflav":sum, "process":sig_procs}]
                vals = sum(histo[{"process":p}].values() for p in sig_procs)
                bins = histo.axes["fj0_mparticlenet"].edges
                raw_yield = vals.sum()
                if normalize and raw_yield > 0:
                    vals = vals / raw_yield
                ax.stairs(vals, bins, label=f"{label} ({raw_yield:.3f})")
            ax.axvline(125, color="gray", linestyle="--", label="m_H = 125 GeV")
            ax.axvline(91,  color="gray", linestyle=":",  label="m_Z = 91 GeV")
            ax.set_xlabel("fj0 particleNet mass [GeV]")
            ax.set_ylabel("Normalized units" if normalize else "Events")
            ax.legend()
            plt.tight_layout()
            suffix = "normalized" if normalize else "raw"
            plt.savefig(f"{cat}__fj0_mparticlenet_truthmatched_{suffix}.png")
            plt.close()

if __name__ == '__main__':
    main()

