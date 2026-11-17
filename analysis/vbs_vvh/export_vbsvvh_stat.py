# Export the ABCD regions of a channel to the vbsvvh-stat inputs (shapes.root + channel.yaml),
# with the whole run as one summed era ("Run2" or "Run3", see vbsvvh-stat INPUTS.md).
#
#   - Signal: one process per hypothesis, its nominal and Up/Down from the signal-only --do-systs pkl(s),
#     and the PDF (pdf_Higgs_qqbar) from the Hessian members pdf_1 ... pdf_100 of that pkl
#   - bkg: the MC of the ABCD backgrounds (only used by vbsvvh-stat for the MC closure)
#   - data_obs: data in B, C, D; region A is written as 0 (blinded) unless --unblind-a
#   - the data-driven ABCD closure the processor saved in the nominal pkl ("abcd_closure"), at its working point:
#     recorded in channel.yaml (provenance) and printed as the line for the vbsvvh-stat result config,
#     closure: {<tag>: {<run>: <delta>}}
#
# Every systematic is one nuisance for the whole run (named with the energy tag where CMS splits it per era, e.g.
# CMS_scale_j_Absolute_13TeV), except the b-tag uncorrelated ones, which the processor fills per era
#
#   python export_vbsvvh_stat.py --nominal-pkl histos/vvh_3l_r2_v5_nominal.pkl.gz \
#       --syst-pkl histos/vvh_3l_r2_v5_sig_systs.pkl.gz --x-cut 0.54 --y-cut 0.06 -o vbsvvh_stat_inputs/UF3l_Run2_v5
#
# or, for many working points (e.g. the top_working_points.txt that do_abcd_scan writes next to its datacards),
# one input folder per working point under -o:
#   python export_vbsvvh_stat.py --nominal-pkl ... --syst-pkl ... --wp-file <dir>/top_working_points.txt -o vbsvvh_stat_inputs/top_wps

import argparse
import copy
import gzip
import os
import pickle

import hist
import numpy as np
import uproot
import yaml

import ewkcoffea.modules.plotting_tools as plt_tools
import check_vvh_hists as cvh

# Signal hypotheses: vbsvvh-stat name -> coupling tag in the process names
SIGNALS = {
    "Signal_C2V_1p0_C3_1p0"  : "c2v1p0_c3_1p0",
    "Signal_C2V_1p5_C3_1p0"  : "c2v1p5_c3_1p0",
    "Signal_C2V_1p0_C3_10p0" : "c2v1p0_c3_10p0",
}
# Backgrounds estimated with the ABCD method (as in do_abcd_scan)
ABCD_BKGS = ["ttbar", "single-t", "ttX", "rare-top", "DY", "VV", "ewkVV", "VH", "VVV"]
# Processor era token -> vbsvvh-stat era tag
ERA_TAG = {"2024Prompt": "2024"}
RUN_TAG = {"r2": "Run2", "r3": "Run3"}
ENERGY = {"r2": "13TeV", "r3": "13p6TeV"}
# PDF nuisance (NNPDF3.1 Hessian members filled by the processor as layers pdf_1 ... pdf_100)
PDF_NAME = "pdf_Higgs_qqbar"


# vbsvvh-stat (nuisances.yaml) name of a processor systematic layer name (without Up/Down)
def stat_name(src, run):
    base, _, era = src.rpartition("_") if src.rpartition("_")[2] in ("2016preVFP", "2016postVFP", "2017", "2018", "2024Prompt", "2025") else (src, "", "")
    era = ERA_TAG.get(era, era)
    if base.startswith("btag_"):
        name = {"btag_HF_correlated": "CMS_btag_fixedWP_bc_correlated", "btag_LF_correlated": "CMS_btag_fixedWP_light_correlated",
                "btag_HF_uncorrelated": "CMS_btag_fixedWP_bc_uncorrelated", "btag_LF_uncorrelated": "CMS_btag_fixedWP_light_uncorrelated"}[base]
    elif base.startswith("jes"):
        name = "CMS_scale_j_" + base[len("jes"):].removesuffix("Year")
        # A Year source kept as one nuisance for the whole run takes the energy tag
        if base.endswith("Year") and not era: name += f"_{ENERGY[run]}"
    else:
        name = {"eleSF": "CMS_eff_e", "muoSF": "CMS_eff_m", "l1prefiring": "CMS_l1_ecal_prefiring", "pileup": f"CMS_pileup_{ENERGY[run]}",
                "jer": f"CMS_res_j_{ENERGY[run]}", "metuncl": "CMS_scale_met_unclustered_energy" + ("" if era else f"_{ENERGY[run]}"),
                "PSFSR": "ps_fsr", "PSISR": "ps_isr",
                "muF": "QCDscale_fac_vbsvvh", "muR": "QCDscale_ren_vbsvvh"}[base]
    return f"{name}_{era}" if era else name


# Region slices of the 2d plane: A is x >= x_cut and y >= y_cut (as in do_abcd_scan), the cuts must be bin edges
def region_slices(h, x_cut, y_cut):
    ax_x, ax_y = h.axes[-2], h.axes[-1]
    for ax, cut in ((ax_x, x_cut), (ax_y, y_cut)):
        if not np.isclose(ax.edges, cut).any(): raise Exception(f"Cut {cut} is not a bin edge of {ax.name}")
    ix, iy = ax_x.index(x_cut), ax_y.index(y_cut)
    return {"A": (slice(ix, None), slice(iy, None)), "B": (slice(None, ix), slice(iy, None)),
            "C": (slice(ix, None), slice(None, iy)), "D": (slice(None, ix), slice(None, iy))}


# (value, variance) per region of a 2d hist (no flow, as in do_abcd_scan)
def region_yields(h2d, slices):
    return {r: (h2d[s].sum(flow=False).value, h2d[s].sum(flow=False).variance) for r, s in slices.items()}


def th1(value, variance):
    h = hist.Hist(hist.axis.Regular(1, 0, 1), storage=hist.storage.Weight())
    h.view()["value"], h.view()["variance"] = [value], [variance]
    return h


# The data-driven closure of this channel and run that the processor saved in the nominal pkl, if it is at this working point
def closure_at(closures, tag, run_tag, x_cut, y_cut):
    by_tag = closures.get(tag) or (next(iter(closures.values())) if len(closures) == 1 else {})
    c = by_tag.get(run_tag)
    if c is None or not (np.isclose(c["thresholds"]["t1"], x_cut) and np.isclose(c["thresholds"]["t2"], y_cut)):
        return None
    return c


# Working points of a --wp-file: one per line, "rank x_cut y_cut [anything else]", lines starting with # skipped
def read_wp_file(path):
    wps = []
    with open(path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"): continue
            rank, x_cut, y_cut = line.split()[:3]
            wps.append((int(rank), float(x_cut), float(y_cut)))
    return wps


# shapes.root and channel.yaml for one working point, from the signal hists (name -> 2d hist with the systematic
# axis) and the grouped nominal hist (process_grp axis)
def write_inputs(sig, h_nom, x_cut, y_cut, out_dir, args, extra_provenance={}, closure=None):
    run_tag = RUN_TAG[args.run]
    out = {}  # "<run>/<region>/<process>[__<nuisance>Up|Down]" -> (value, variance)
    slices = region_slices(next(iter(sig.values())), x_cut, y_cut)
    for name, h in sig.items():
        nom = region_yields(h[{"systematic": "nominal"}], slices)
        for r in "ABCD": out[f"{run_tag}/{r}/{name}"] = nom[r]
        layers = sorted({s[:-2] for s in h.axes["systematic"] if s.endswith("Up")})
        for src in layers:
            for d in ("Up", "Down"):
                var = region_yields(h[{"systematic": src+d}], slices)
                for r in "ABCD": out[f"{run_tag}/{r}/{name}__{stat_name(src, args.run)}{d}"] = (var[r][0], nom[r][1])
        # PDF, per region: kappa = 1 + sqrt(sum_i (Y_i - Y_0)^2) / Y_0 over the Hessian members, symmetric (Up = Y_0 kappa, Down = Y_0 / kappa)
        pdf_layers = [s for s in h.axes["systematic"] if s.startswith("pdf_") and s[len("pdf_"):].isdigit()]
        if pdf_layers:
            members = [region_yields(h[{"systematic": s}], slices) for s in pdf_layers]
            for r in "ABCD":
                y0 = nom[r][0]
                if y0 <= 0: continue
                kappa = 1 + np.sqrt(sum((m[r][0] - y0)**2 for m in members)) / y0
                out[f"{run_tag}/{r}/{name}__{PDF_NAME}Up"] = (y0 * kappa, nom[r][1])
                out[f"{run_tag}/{r}/{name}__{PDF_NAME}Down"] = (y0 / kappa, nom[r][1])

    # ABCD background MC and data
    bkg = region_yields(h_nom[{"process_grp": ABCD_BKGS}][{"process_grp": sum}], slices)
    dat = region_yields(h_nom[{"process_grp": "Data"}], slices)
    for r in "ABCD":
        out[f"{run_tag}/{r}/bkg"] = bkg[r]
        n_dat = dat[r][0] if (r != "A" or args.unblind_a) else 0.0
        out[f"{run_tag}/{r}/data_obs"] = (n_dat, n_dat)

    os.makedirs(out_dir, exist_ok=True)
    with uproot.recreate(os.path.join(out_dir, "shapes.root")) as f:
        for key, (value, variance) in out.items():
            f[key] = th1(value, variance)
    channel = {
        "channel": args.tag,
        "abcd": {"background": "bkg", "A": "A", "B": "B", "C": "C", "D": "D"},
        "provenance": {"nominal_pkl": os.path.abspath(args.nominal_pkl), "syst_pkl": os.path.abspath(args.syst_pkl),
                       "hist": args.hist, "category": args.cat, "x_cut": x_cut, "y_cut": y_cut, **extra_provenance,
                       "abcd_bkgs": ABCD_BKGS, "region_A_data": "unblinded" if args.unblind_a else "0 (blinded)"},
    }
    if closure is not None:
        channel["provenance"]["abcd_closure"] = {k: closure[k] for k in ("delta", "scheme", "per_scheme", "mc_kappa_A", "min_count", "thresholds") if k in closure}
    with open(os.path.join(out_dir, "channel.yaml"), "w") as f:
        yaml.safe_dump(channel, f, sort_keys=False)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nominal-pkl", required=True, help="Pkl of the nominal run on all samples (bkg MC and data)")
    parser.add_argument("--syst-pkl", required=True, help="Pkl of the signal-only --do-systs run, one layer per source")
    parser.add_argument("-r", "--run", default="r2", choices=["r2", "r3"])
    parser.add_argument("--hist", default="abcd2d_3lChsum3", help="The 2d ABCD hist")
    parser.add_argument("--cat", default="3l_chsum3", help="Its category")
    parser.add_argument("--x-cut", type=float, default=None, help="Cut on the DNN score (x axis)")
    parser.add_argument("--y-cut", type=float, default=None, help="Cut on the constraining variable (y axis)")
    parser.add_argument("--wp-file", default=None, help="Instead of --x-cut/--y-cut: a list of working points (e.g. top_working_points.txt from do_abcd_scan), one input folder each under -o")
    parser.add_argument("-t", "--tag", default="UF3l", help="Channel tag in vbsvvh-stat")
    parser.add_argument("-o", "--out-dir", required=True)
    parser.add_argument("--unblind-a", action="store_true", help="Write the data in region A (default: 0, blinded)")
    args = parser.parse_args()
    if (args.wp_file is None) == (args.x_cut is None or args.y_cut is None):
        parser.error("give either --x-cut and --y-cut, or --wp-file")

    # Signal, from the systematics pkl, and ABCD background MC and data, from the nominal pkl (read once)
    h = pickle.load(gzip.open(args.syst_pkl))[args.hist][{"category": args.cat}]
    sig = {name: h[{"process": [p for p in h.axes["process"] if p.endswith(tag)]}][{"process": sum}] for name, tag in SIGNALS.items()}
    nom_out = pickle.load(gzip.open(args.nominal_pkl))
    closures = nom_out.get("abcd_closure") or {}
    h_nom = nom_out[args.hist][{"category": args.cat}]
    del nom_out
    if "systematic" in h_nom.axes.name: h_nom = h_nom[{"systematic": "nominal"}]
    grp_dict = copy.deepcopy(cvh.GRP_DICT_FULL_R2 if args.run == "r2" else cvh.GRP_DICT_FULL_R3)
    h_nom = plt_tools.group(h_nom, "process", "process_grp", grp_dict)
    run_tag = RUN_TAG[args.run]

    if args.wp_file is None:
        closure = closure_at(closures, args.tag, run_tag, args.x_cut, args.y_cut)
        out = write_inputs(sig, h_nom, args.x_cut, args.y_cut, args.out_dir, args, closure=closure)
        print(f"Wrote {len(out)} histograms to {args.out_dir}/shapes.root")
        for r in "ABCD":
            print(f"  {r}: signal SM {out[f'{run_tag}/{r}/Signal_C2V_1p0_C3_1p0'][0]:.5f}, bkg MC {out[f'{run_tag}/{r}/bkg'][0]:.3f}")
        for name in SIGNALS:
            up, nom = out.get(f"{run_tag}/A/{name}__{PDF_NAME}Up"), out[f"{run_tag}/A/{name}"]
            if up is not None: print(f"  {name}: PDF kappa in A = {up[0] / nom[0]:.4f}")
        if closure is None:
            print("No data-driven closure in the nominal pkl at this working point: vbsvvh-stat would use its MC closure")
        else:
            print(f"Data-driven ABCD closure: delta = {closure['delta']:.4f} (scheme {closure['scheme']}). In the vbsvvh-stat result config:")
            print(f"  closure: {{{args.tag}: {{{run_tag}: {closure['delta']:.4f}}}}}")
        return

    wps = read_wp_file(args.wp_file)
    for rank, x_cut, y_cut in wps:
        name = f"wp{rank:03d}_x{x_cut:g}_y{y_cut:g}".replace(".", "p")
        write_inputs(sig, h_nom, x_cut, y_cut, os.path.join(args.out_dir, name), args,
                     {"wp_file": os.path.abspath(args.wp_file), "wp_rank": rank},
                     closure=closure_at(closures, args.tag, run_tag, x_cut, y_cut))
    print(f"Wrote the inputs of {len(wps)} working points to {args.out_dir}/wp<rank>_x<x_cut>_y<y_cut>/")


if __name__ == "__main__":
    main()
