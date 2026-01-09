#!/usr/bin/env python3
import os
import argparse
import config.plotting_config as CONFIG

from plotter_utils.helpers.general_io import resolve_input_list,resolve_outdir,print_yield_nicely,save_yield_dict
from plotter_utils.histo_reader.loader import load_hist_collection,unpack_hist,cut_order
from plotter_utils.histo_reader.sample_info import ProcessMap
from plotter_utils.plotting.draw import draw

import warnings
warnings.filterwarnings("ignore", message="List indexing selection is experimental.*")

import logging
def setup_logging():
    logging.basicConfig(
        level=getattr(logging, CONFIG.LOG_LEVEL),
        format=CONFIG.LOG_FORMAT,
    )
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser("make_plots")

    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input .pkl.gz files or dirname. should all have the same set of cuts (category axis) ",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output directory. will use the dir of the first pkl.gz file if not provided",
    )
    parser.add_argument(
        "-p", "--plots",
        action="store_true",
        help="Make plots",
    )
    parser.add_argument(
        "-y", "--yields",
        action="store_true",
        help="Print yields",
    )
    parser.add_argument(
        "-c", "--cutname",
        default=None,
        help="choose which cut (selection) to draw, default will draw all",
    )

    return parser.parse_args()

def make_all_plots(all_hist_collection, proc_map, outdir):
    for cut,hist_collection in all_hist_collection.items():
        subdir = os.path.join(outdir, cut) if cut is not None else os.path.join(outdir, 'no_category')

        os.makedirs(subdir, exist_ok=True)
        logger.info(f'start producing plots for cut: {cut}')
        logger.info(f'output will be saved at {subdir}')

        for var in hist_collection.variables_all:
            sig,bkg,data = unpack_hist(hist_collection,var,proc_map)
            fig = draw( 
                sig=sig,
                bkg=bkg,
                data=data,
                proc_map=proc_map,
                config={"SUBPLOTS": CONFIG.SUBPLOTS,"FIG_RATIO": CONFIG.FIG_RATIO},
                title=var,
            )
            fig.savefig(os.path.join(subdir,f"{var}.png"), bbox_inches="tight")
            logger.info(f"    - Saved {var}.png")

def save_yield_json(all_hist_collection, proc_map, output_json_name):
    yield_ref_var = CONFIG.REF_VAR 
    logger.info(f"Using reference variable: {yield_ref_var}")
    yield_dict = {} #initialize yield dict
    uncer_dict = {}
    for cutname,hist_collection in all_hist_collection.items(): #cutname = cat in old script
        sig,bkg,data = unpack_hist(hist_collection,yield_ref_var,proc_map)
        yield_dict[cutname]={
            "Signal":sig.total_yield(),
            "Background":bkg.total_yield(),
            "Data":data.total_yield(),
        }
        bkg_dict = bkg.get_yield_per_type()
        yield_dict[cutname].update(bkg_dict)

        uncer_dict[cutname]={
            "Signal":sig.uncertainty(),
            "Background":bkg.uncertainty(),
            "Data":data.uncertainty()
        }
        bkg_uncert = bkg.get_uncertainty_per_type()
        uncer_dict[cutname].update(bkg_uncert)
    outdict = {}
    for cutname in all_hist_collection.keys():
        outdict[cutname] = {
            proc:(yield_dict[cutname][proc],uncer_dict[cutname][proc]) for proc in yield_dict[cutname].keys()
            }
    ordered_cuts = cut_order(all_hist_collection,yield_ref_var,proc_map)
    outdict_ordered = {k: outdict[k] for k in ordered_cuts}

    for cutname in ordered_cuts:
        logger.info(f'cut: {cutname}')
        print_yield_nicely(outdict_ordered[cutname])

    save_yield_dict(outdict_ordered, output_json_name)

def main(args):
    setup_logging()

    input_list = resolve_input_list(args.inputs)
    outdir = resolve_outdir(args.output,input_list,CONFIG.DEFAULT_OUTPUT_DIRNAME)

    #get proc_map that link process to sample type e.g. QCDpt to QCD
    proc_map = ProcessMap.from_csv(CONFIG.PROC_MAP_CSV)

    #load all hist in given files
    all_hist_collection = load_hist_collection(
        pkl_paths=input_list,
        proc_map=proc_map,
        cutname=args.cutname,
    )

    if args.plots:
        make_all_plots(all_hist_collection, proc_map, outdir)
    if args.yields:
        output_json_name = os.path.join(outdir,'yield.json') 
        save_yield_json(all_hist_collection, proc_map, output_json_name)

if __name__ == "__main__":
    args = parse_args()
    main(args)
