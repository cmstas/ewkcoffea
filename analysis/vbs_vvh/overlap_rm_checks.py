import argparse
import pickle
import gzip
import json
import os
import shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import copy

import ewkcoffea.modules.plotting_tools as plt_tools
import topcoffea.modules.MakeLatexTable as mlt


def main():

    # Set up the command line parser
    parser = argparse.ArgumentParser()
    parser.add_argument("pkl_file_path", help = "The path to the pkl file")
    args = parser.parse_args()

    sig_processes = ['VBSZZH_c2v1p0_c3_1p0', 'VBSWWH_SS_c2v1p0_c3_1p0', 'VBSWWH_OS_c2v1p0_c3_1p0', 'VBSWZH_c2v1p0_c3_1p0']
    tt_processes = ['TTTo2L2Nu_TuneCP5_13TeV']
    dy_processes = ['DYJetsToLL_M-50_TuneCP5_13TeV']
    #tt_processes = ['TTTo2L2Nu_TuneCP5_13TeV', 'TTToSemiLeptonic_TuneCP5_13TeV', 'TTToHadronic_TuneCP5_13TeV']
    #dy_processes = ['DYJetsToLL_M-50_TuneCP5_13TeV', 'DYJetsToLL_M-10to50_TuneCP5_13TeV']

    # Get the dictionary of histograms from the input pkl file
    histo_dict = pickle.load(gzip.open(args.pkl_file_path))
    name = args.pkl_file_path.split("/")[1][:-7] # Drops leading histos/ and trailing .pkl.gz

    print("tot",sum(sum(histo_dict["njets"][{"systematic":"nominal", "category":"2lOSSF_1fjx", "process":sig_processes, "lepflav":sum}].values(flow=True))))
    print("h",  sum(sum(histo_dict["njets"][{"systematic":"nominal", "category":"2lOSSF_1fjx_fj0matchH", "process":sig_processes, "lepflav":sum}].values(flow=True))))
    print("v1", sum(sum(histo_dict["njets"][{"systematic":"nominal", "category":"2lOSSF_1fjx_fj0matchV1", "process":sig_processes, "lepflav":sum}].values(flow=True))))
    print("v2", sum(sum(histo_dict["njets"][{"systematic":"nominal", "category":"2lOSSF_1fjx_fj0matchV2", "process":sig_processes, "lepflav":sum}].values(flow=True))))
    print("v",  sum(sum(histo_dict["njets"][{"systematic":"nominal", "category":"2lOSSF_1fjx_fj0matchV", "process":sig_processes, "lepflav":sum}].values(flow=True))))
    print("n",  sum(sum(histo_dict["njets"][{"systematic":"nominal", "category":"2lOSSF_1fjx_fj0noMatch", "process":sig_processes, "lepflav":sum}].values(flow=True))))
    #histo = histo_dict["njets"][{"systematic":"nominal", "category":"2lOSSF_1fjx", "lepflav":sum}]
    #print(histo)

    ### Distributions ###
    proc_lst = sig_processes + dy_processes + tt_processes
    for var in ["vbs_mjj", "vbs_absdetajj", "vbs_score", "fj0_mparticlenet"]:
        fig, ax = plt.subplots(figsize=(6, 4))
        histo = histo_dict[var][{"systematic":"nominal", "category":"2lOSSF_1fjx", "lepflav":sum, "process":proc_lst}]
        histo = plt_tools.merge_overflow(histo)
        histo = plt_tools.rebin(histo,6)
        histo.plot1d(ax=ax, stack=True, histtype="fill")
        total_yield = sum(histo[{"process":p}].values().sum() for p in proc_lst)
        print("total_yield",total_yield)
        ax.text(0.97, 0.97, f"Yield: {total_yield:.3f}", transform=ax.transAxes, ha="right", va="top", fontsize=10)
        ax.set_ylabel("Events")
        ax.legend(fontsize=6)
        plt.tight_layout()
        ymax = ax.get_ylim()[1]
        ax.set_ylim(0, ymax * 1.5)
        plt.savefig(f"{var}.png")
        plt.close()


    ### Truth plots ###
    fig, ax = plt.subplots(figsize=(8, 5))

    categories = {
        "H-matched"   : "2lOSSF_1fjx_fj0matchH",
        "V-matched"   : "2lOSSF_1fjx_fj0matchV",
        "Unmatched"   : "2lOSSF_1fjx_fj0noMatch",
    }

    for normalize in [True, False]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for label, cat in categories.items():
            histo = histo_dict["fj0_mparticlenet"][{"systematic":"nominal", "category":cat, "lepflav":sum, "process":sig_processes}]
            vals = sum(histo[{"process":p}].values() for p in sig_processes)
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
        plt.savefig(f"fj0_mparticlenet_truthmatched_{suffix}.png")
        plt.close()

if __name__ == '__main__':
    main()

