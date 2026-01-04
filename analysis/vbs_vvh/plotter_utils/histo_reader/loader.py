# this gets the histo dict from a list of files (filename.pkl.gz)
# only read the file, get the required cut, and determine the list of var, sig,bkg,data but not doing any transformation on the histos
# check same list of year for sig, bkg, data 
# get list of var for sig_only, mc_only = sig+bkg, all
# verbose show list of var. also print if any variable is not sig_only/mc_only/all

import gzip
import pickle
from collections import defaultdict

from .histcollection import HistCollection

import logging
logger = logging.getLogger(__name__)

def _load_one_file(filename):
    """
    Return loaded histogram dict from a .pkl.gz file
    """
    logging.info(f"Reading file: {filename}")
    histo_dict = pickle.load(gzip.open(filename))
    return histo_dict


def _get_years(hist):
    """
    for checking years list consistency over sig/bkg/data
    """
    if hist is None:
        return None
    return set(hist.axes["year"])


def load_hist_collection(pkl_paths,proc_map,cutname=None):
    """
    Load one or more coffea output pkl.gz files and separate variables into signal, background, and data.
    Checks that sig/bkg/data have the same list of years for each variable.

    Parameters
    ----------
    pkl_paths : list[str]
        List of .pkl.gz files
    proc_map : ProcessMap object as defined in sample_info
    cutname : str
        'category' axis which is the cutname. Use None only when this axis does not exist
    """

    # ---- load & merge ----
    hist_dicts = [_load_one_file(p) for p in pkl_paths]
    #hists = _merge_hist_dicts(hist_dicts)

    views = {}
    vars_sig_only = []
    vars_mc_only = [] # sig&bkg but not data
    vars_all = []

    # ---- inspect each variable ----
    
    for hists in hist_dicts:
        for var, hist_precut in hists.items():
            if cutname is not None:
                hist=hist_precut[{"category":cutname}]
            else:
                hist = hist_precut
            sig = None
            bkg = None
            data = None

            processes = set(hist.axes["process"])
            if var not in views:
                views[var] = {"signal": None, "background": None, "data": None, "years": []}

            processes = set(hist.axes["process"])
            if processes & set(proc_map.signal):
                views[var]["signal"] = hist[{"process": proc_map.signal}]
            if processes & set(proc_map.background):

                available_processes = list(set(proc_map.background) & set(processes))

                views[var]["background"] = hist[{"process": available_processes}]
            if processes & set(proc_map.data):
                views[var]["data"] = hist[{"process": proc_map.data}]

    # ---- check year consistency, classify var ----
    for var, vdict in views.items():
        year_sets = [ys for ys in [_get_years(vdict["signal"]),
                                   _get_years(vdict["background"]),
                                   _get_years(vdict["data"])] if ys is not None]
        if len(year_sets) > 1 and not all(ys == year_sets[0] for ys in year_sets):
            logging.error(f"sig: {_get_years(vdict['signal'])}\n"\
                          f"bkg: {_get_years(vdict['background'])}\n"\
                          f"data: {_get_years(vdict['data'])}\n")
            raise RuntimeError(f"[{var}] Year mismatch across sig/bkg/data")

        vdict["years"] = sorted(year_sets[0]) if year_sets else []

        if vdict["signal"] and not vdict["background"]:
            vars_sig_only.append(var)
        elif vdict["signal"] and vdict["background"] and not vdict["data"]:
            vars_mc_only.append(var)
        elif vdict["signal"] and vdict["background"] and vdict["data"]:
            vars_all.append(var)

    # ---- print summary ----
    logging.info("=== HistCollection summary ===")
    logging.info(f"Total variables: {len(views)}")
    logging.info(f"sig-only vars         :({len(vars_sig_only)}) {vars_sig_only}")
    logging.info(f"sig + bkg vars        :({len(vars_mc_only)}) {vars_mc_only}")
    logging.info(f"sig + bkg + data vars :({len(vars_all)}) {vars_all}")

    others = set(views) - set(vars_all) - set(vars_mc_only) - set(vars_sig_only)
    if others:
        logging.debug(f"variables with unexpected content: {others}")

    return HistCollection(
        views=views,
        variables_sig_only=vars_sig_only,
        variables_mc_only=vars_mc_only,
        variables_all=vars_all,
        proc_map=proc_map
    )