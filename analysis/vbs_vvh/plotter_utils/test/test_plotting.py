# test_plotting.py

import logging
from plotter_utils.histo_reader.loader import load_hist_collection
from plotter_utils.histo_reader.sample_info import ProcessMap, get_all_years
from plotter_utils.histo_objects import SignalHist, BackgroundHist, DataHist
from plotter_utils.plotting.draw import draw


def setup_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(levelname)s %(name)s: %(message)s"
    )

    for noisy in [
        "matplotlib",
        "matplotlib.font_manager",
        "PIL",
        "numpy",
        "uproot",
        "coffea",
        "numba",
        "boost_histogram",
    ]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

setup_logging()

# ---------------------------------------------------------
# Minimal "config" replacement
# ---------------------------------------------------------

SUBPLOTS = ("dataMC", "significance")
FIG_RATIO = {
    "main": 3,
    "dataMC": 1,
    "significance": 1,
}

CONFIG = {
    "SUBPLOTS": SUBPLOTS,
    "FIG_RATIO": FIG_RATIO,
}

# ---------------------------------------------------------
# Load histograms
# ---------------------------------------------------------

cutname = "objsel"
years = get_all_years()
proc_map = ProcessMap.from_csv()
logging.debug(f"{proc_map.background_colors}")

hist_collection = load_hist_collection(
    pkl_paths=[
        "/home/users/pyli/projects/analysis_VVH/coffea/ewkcoffea/analysis/vbs_vvh/histos/test/test2/test2_bkg.pkl.gz",
        "/home/users/pyli/projects/analysis_VVH/coffea/ewkcoffea/analysis/vbs_vvh/histos/test/test2/test2_sig.pkl.gz",
        "/home/users/pyli/projects/analysis_VVH/coffea/ewkcoffea/analysis/vbs_vvh/histos/test/test2/test2_data.pkl.gz"
    ],
    proc_map=proc_map,
    cutname=cutname,
)

# Pick variable
var = "Higgs_pt"
view = hist_collection.views[var]

# ---------------------------------------------------------
# Build histogram objects
# ---------------------------------------------------------

sig = SignalHist(
    view["signal"],
    years=years,
    process_grouping={"Signal": proc_map.signal},
    coupling="nominal",
)

bkg = BackgroundHist(
    view["background"],
    years=years,
    process_grouping=proc_map.background_groups,
    background_groups=list(proc_map.background_groups.keys()),
)

data = DataHist(
    view["data"],
    years=years,
    process_grouping={"Data": proc_map.data},
) 

logging.debug(f"in main sig {sig.hist}")
logging.debug(f"in main bkg {bkg.hist}")
logging.debug(f"in main data {data.hist}")

# ---------------------------------------------------------
# Draw
# ---------------------------------------------------------
logging.debug(f"in test_main proc_map bkg is {proc_map.background_colors}")
fig = draw(
    sig=sig,
    bkg=bkg,
    data=data,
    proc_map=proc_map,
    config=CONFIG,
    title=f"{cutname} : {var}",
)

fig.savefig(f"test_{var}.png", bbox_inches="tight")
print(f"Saved test_{var}.png")
