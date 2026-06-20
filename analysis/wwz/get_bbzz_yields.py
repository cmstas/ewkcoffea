import pickle
import hist
import gzip

import ewkcoffea.modules.sample_groupings as sg


##################################
# Make sample grouping dict
GROUPING_DICT_R2= {
    #"ZZ"  : ["ZZTo4l", "ggToZZTo2e2mu", "ggToZZTo2e2tau", "ggToZZTo2mu2tau", "ggToZZTo4e", "ggToZZTo4mu", "ggToZZTo4tau"],
    #"ttZ" : [ "TTZToLL_M_1to10", "TTZToLLNuNu_M_10", ],
    "higgs" : [ "ggHToZZ4L", ],
    #"HH" : [ "HHTobbVV", ],
    #"WZ" : [ "WZTo3LNu", ],
}

GROUPING_DICT_R3= {
    "ZZ"  : ["ZZTo4l", "ggToZZTo2e2mu", "ggToZZTo2e2tau", "ggToZZTo2mu2tau", "ggToZZTo4e", "ggToZZTo4mu", "ggToZZTo4tau"],
    "ttZ" : [ "TTZToLL_M_4to50", "TTZToLL_M_50", ],
    "higgs" : [ "ggHToZZ4L", ],
}

def attach_years(year):
    out_dict = {}
    if year == "run2":
        years = ["UL16APV","UL16","UL17","UL18"]
        #years = ["UL18"]
        sample_dict_base = GROUPING_DICT_R2
    elif year == "run3":
        years = ["2022","2022EE","2023","2023BPix"]
        sample_dict_base = GROUPING_DICT_R3
    else:
        raise Exception("Unknown year")
    for proc_group in sample_dict_base.keys():
        out_dict[proc_group] = []
        for proc_base_name in sample_dict_base[proc_group]:
            for year_str in years:
                out_dict[proc_group].append(f"{year_str}_{proc_base_name}")
    return out_dict


##################################


##################################
# Get the list of categories on the sparese axis
def get_axis_cats(histo,axis_name):
    process_list = [x for x in histo.axes[axis_name]]
    return process_list
# Regroup categories (e.g. processes)
def group(h, oldname, newname, grouping):

    # Build up a grouping dict that drops any proc that is not in our h
    grouping_slim = {}
    proc_lst = get_axis_cats(h,oldname)
    for grouping_name in grouping.keys():
        for proc in grouping[grouping_name]:
            if proc in proc_lst:
                if grouping_name not in grouping_slim:
                    grouping_slim[grouping_name] = []
                grouping_slim[grouping_name].append(proc)
            #else:
            #    print(f"WARNING: process {proc} not in this hist")

    # From Nick: https://github.com/CoffeaTeam/coffea/discussions/705#discussioncomment-4604211
    hnew = hist.Hist(
        hist.axis.StrCategory(grouping_slim, name=newname),
        *(ax for ax in h.axes if ax.name != oldname),
        storage=h.storage_type(),
    )
    for i, indices in enumerate(grouping_slim.values()):
        hnew.view(flow=True)[i] = h[{oldname: indices}][{oldname: sum}].view(flow=True)

    return hnew
##################################

def main():

    year = "run2"
    pkl_file_path = "histos/tmp.pkl.gz"
    #pkl_file_path = "histos/r2_zz_h_ttz_noSyst_1.pkl.gz" # Standard selection
    #pkl_file_path = "histos/r2_zz_h_ttz_noSyst_lowpt.pkl.gz" # Lower lep pt
    #year = "run3"
    #pkl_file_path = "histos/r3_zz_h_ttz_noSyst.pkl.gz" # Standard selection

    # Make grouping dict
    GOUPING_DICT = attach_years(year)

    histo_dict = pickle.load(gzip.open(pkl_file_path))

    cat_lst = [
        "all_events",
        "bbzz",
        "bbzz_zzcand",
        "bbzz_zzcand_bb",
        "bbzz_zzcand_bb_m4l",

        "bbzz_zzcand_bb_m4l_4m",
        "bbzz_zzcand_bb_m4l_4e",
        "bbzz_zzcand_bb_m4l_2e2m",
        "sr_4l_bdt_of_trn",
    ]

    histo = histo_dict["njets"][{"systematic":"nominal"}]
    histo_grouped = group(histo,"process","process_grp",GOUPING_DICT)

    for proc in GOUPING_DICT:

        print("\n-------",proc,"-------")

        for cat in cat_lst:
            histo = histo_grouped[{"category":cat,"process_grp":proc}]
            print("\n",cat)
            val = sum(histo.values(flow=True))
            err = (sum(histo.variances(flow=True)))**0.5
            perr = round(100*err/val,1)
            print(f"{val} +- {err}, {perr}%")



main()
