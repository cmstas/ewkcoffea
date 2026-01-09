# this gets the histo dict from a list of files (filename.pkl.gz)
# only read the file, get the required cut, and determine the list of var, sig,bkg,data but not doing any transformation on the histos
# check same list of year for sig, bkg, data 
# get list of var for sig_only, mc_only = sig+bkg, all
# verbose show list of var. also print if any variable is not sig_only/mc_only/all

import gzip
import pickle
from collections import defaultdict

from .histcollection import HistCollection
from plotter_utils.histo_objects import SignalHist, BackgroundHist, DataHist

import logging
logger = logging.getLogger(__name__)


def load_hist_collection(pkl_paths,proc_map,cutname=None):
    """
    Load one or more coffea output pkl.gz files and separate variables into signal, background, and data.
    Should have the given cut, or the same set of cuts 
    Return {cutname:histCollection} for cutname, or over all cuts in category axis if cutname is not given

    Will check that sig/bkg/data have the same list of years for each variable.

    Parameters
    ----------
    pkl_paths : list[str]
        List of .pkl.gz files or dir 
    proc_map : ProcessMap object as defined in sample_info
    cutname : str
        'category' axis which is the cutname. Use None only when this axis does not exist

    """
    # ---- load & merge ----
    hist_dicts = [_load_one_file(p) for p in pkl_paths]
    #hists = _merge_hist_dicts(hist_dicts)
    
    if cutname is not None:
        allcut={cutname}
    else:
        allcut=_get_category_axis(hist_dicts[0])
        for hist in hist_dicts[1:]:
            if _get_category_axis(hist)!=allcut:
                logger.error(f'different set of cuts \n{allcut} \n{_get_category_axis(hist)}\n exit')
                import sys
                sys.exit(1)

    
    all_hist_collection = {
        cut:_get_hist_collection_single_cut(hist_dicts,proc_map,cut) for cut in allcut
    }
        
    return all_hist_collection

def unpack_hist(hist_collection,var,proc_map):
    #need to test when none.
    """distribute into sig,bkg,dataHist for a single variable"""
    view = hist_collection.views[var]
    years = view["years"]
    if view["signal"] is not None:
        sig = SignalHist(
            view["signal"],
            years=years,
            process_grouping={"Signal": proc_map.signal},
            coupling="nominal",
        )
    else:
        sig = None
        
    if view["background"] is not None:
        bkg = BackgroundHist(
            view["background"],
            years=years,
            process_grouping=proc_map.background_groups,
            background_groups=list(proc_map.background_groups),
        )
    else:
        bkg = None

    if view["data"] is not None:
        data = DataHist(
            view["data"],
            years=years,
            process_grouping={"Data": proc_map.data},
        )
    else:
        data = None

    return sig,bkg,data

def cut_order(all_hist_collection,ref_var,proc_map):
    """assume at least one of sig/bkg/data exist for all the cuts"""
    yield_dict = {}

    # Collect yields per cut
    for cutname, hist_collection in all_hist_collection.items():
        sig, bkg, data = unpack_hist(hist_collection, ref_var, proc_map)

        yield_dict[cutname] = {
            "Signal": sig.total_yield() if sig is not None else None,
            "Background": bkg.total_yield() if bkg is not None else None,
            "Data": data.total_yield() if data is not None else None,
        }

    # Determine which reference exists for ALL cuts
    reference = None
    for candidate in ("Signal", "Background", "Data"):
        if all(yield_dict[cut][candidate] is not None for cut in yield_dict):
            reference = candidate
            break

    if reference is None:
        raise RuntimeError(
            "No common reference (Signal / Background / Data) "
            "exists for all cuts."
        )

    # Sort cut names by descending yield
    ordered_cuts = sorted(
        yield_dict.keys(),
        key=lambda cut: yield_dict[cut][reference],
        reverse=True,
    )

    return ordered_cuts

def _load_one_file(filename):
    """
    Return loaded histogram dict from a .pkl.gz file
    """
    logger.info(f"Reading file: {filename}")
    histo_dict = pickle.load(gzip.open(filename))
    return histo_dict


def _get_years(hist):
    """
    for checking years list consistency over sig/bkg/data
    """
    if hist is None:
        return None
    return set(hist.axes["year"])

def _get_category_axis(hist):
    temp_hist = hist[list(hist.keys())[0]]
    if "category" in temp_hist.axes.name:
        cuts = set(temp_hist.axes["category"])
        logger.info(f'discovered cuts {cuts}')
        return cuts
    else:
        logger.info(f'category is not in axis')
        return None


        

def _get_hist_collection_single_cut(hist_dicts,proc_map,cutname=None):

    views = {}
    vars_sig_only = []
    vars_mc_only = [] # sig&bkg but not data
    vars_all = []

    # ---- inspect each variable ----
    
    for hists in hist_dicts:
        for var, hist_precut in hists.items():

            if cutname is not None:
                hist = hist_precut[{"category":cutname}]
            else:
                hist = hist_precut

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
            logger.error(f"sig: {_get_years(vdict['signal'])}\n"\
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
    logger.debug("=== HistCollection summary ===")
    logger.debug(f"Total variables: {len(views)}")
    logger.debug(f"sig-only vars         :({len(vars_sig_only)}) {vars_sig_only}")
    logger.debug(f"sig + bkg vars        :({len(vars_mc_only)}) {vars_mc_only}")
    logger.debug(f"sig + bkg + data vars :({len(vars_all)}) {vars_all}")

    others = set(views) - set(vars_all) - set(vars_mc_only) - set(vars_sig_only)
    if others:
        logger.debug(f"variables with unexpected content: {others}")

    return HistCollection(
        views=views,
        variables_sig_only=vars_sig_only,
        variables_mc_only=vars_mc_only,
        variables_all=vars_all,
        proc_map=proc_map
    )