
#this runs up until histo_objects

import numpy as np

from plotter_utils.histo_reader.loader import load_hist_collection
from plotter_utils.histo_reader.sample_info import ProcessMap,get_all_years
from plotter_utils.histo_objects import (
    SignalHist,
    BackgroundHist,
    DataHist,
)

import config.plotting_config as cfg

import logging
logging.basicConfig(
    level=cfg.LOG_LEVEL,  # Set the minimum level to capture INFO messages
    format=cfg.LOG_FORMAT,
    datefmt='%Y-%m-%d %H:%M:%S'
)

# ---------------------------------------------------------
# 1. Load histograms the SAME way make_plot.py does
# ---------------------------------------------------------

cutname='objsel'   # change if needed
years = get_all_years()
proc_map = ProcessMap.from_csv()


hist_collection = load_hist_collection(
    pkl_paths=['/home/users/pyli/projects/analysis_VVH/coffea/ewkcoffea/analysis/vbs_vvh/histos/test/test2/test2_bkg.pkl.gz','/home/users/pyli/projects/analysis_VVH/coffea/ewkcoffea/analysis/vbs_vvh/histos/test/test2/test2_sig.pkl.gz'],
    proc_map=proc_map,
    cutname=cutname   # change if needed
)

# Pick something simple
#var = hist_collection.variables_mc_only[0]
var = 'Higgs_pt'
view = hist_collection.views[var]

print(f"Testing variable: {var}, category: {cutname}")

# ---------------------------------------------------------
# 2. Process map (years + grouping)
# ---------------------------------------------------------


# ---------------------------------------------------------
# 3. Construct histogram objects
# ---------------------------------------------------------

sig = SignalHist(
    view["signal"],
    years=years,
    process_grouping={"Signal": proc_map.signal},
    coupling="nominal",   # or your real coupling
)
bkg = BackgroundHist(
    view["background"],
    years=years,
    process_grouping=proc_map.background_groups,
    background_groups=list(proc_map.background_groups.keys()),
)


# ---------------------------------------------------------
# 4. Sanity checks (prints, not asserts)
# ---------------------------------------------------------

print("\n--- Signal ---")
print("Total yield:", sig.total_yield())
print("Bin count:", len(sig.values(flow=False)))
print("Has only Signal:", sig.hist.axes["process_grp"])

print("\n--- Background ---")
print("Total yield:", bkg.total_yield())
print("Bin count:", len(bkg.values(flow=False)))
print("Process groups:", bkg.hist.axes["process_grp"])

# ---------------------------------------------------------
# 5. Numerical consistency checks
# ---------------------------------------------------------

print("\n--- Cross checks ---")

print("sig obj info")
print(sig,type(sig))
print("sig methods")
print(dir(sig))
print("bkg obj info")
print(bkg,type(bkg))
print("bkg methods")
print(dir(bkg))

# No NaNs
assert not np.isnan(sig.values()).any(), "NaN in signal values"
assert not np.isnan(bkg.values()).any(), "NaN in background values"

# Background should be >= 0 everywhere
#assert (bkg.values(flow=False) >= 0).all(), "Negative background bin!"

print("All basic checks passed ")
