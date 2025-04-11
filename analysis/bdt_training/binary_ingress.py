import argparse
import numpy as np
import pickle
import gzip
import cloudpickle
import ewkcoffea.modules.sample_groupings as sg
from ewkcoffea.modules.paths import ewkcoffea_path as ewkcoffea_path
from topcoffea.modules.get_param_from_jsons import GetParam

_VAR_LST_TMP = [
    "njets",
    "j0pt",
    "nleps",
]
VAR_LST_TMP = [
    "ptl4",
    "scalarptsum_lepmetjet",
    "l0pt",
    "l1pt",
    "l2pt",
    "l3pt",
    "j0pt",
    "j1pt",
    "absdphi_zl0_zl1",
    "absdphi_wl0_wl1",
    "mll_wl0_wl1",
    "mll_zl0_zl1",
    "mll_min_sfos",
    "met",
    "njets",
    "nbtagsl",
    "mjj_j0j1",
    "mjj_nearz",
]

_VAR_LST_TMP = [
    #"absdphi_4l_met",
    #"absdphi_min_afas",
    #"absdphi_min_afos",
    #"absdphi_min_sfos",
    #"absdphi_wl0_met",
    #"absdphi_wl0_wl1",
    #"absdphi_wl1_met",
    #"absdphi_wleps_met",
    #"absdphi_zl0_zl1",
    #"absdphi_zleps_met",
    "absdphi_z_ww",
    #"abs_pdgid_sum",
    #"cos_helicity_x",
    #"dr_wl0_j_min",
    #"dr_wl0_wl1",
    #"dr_wl1_j_min",
    #"dr_wleps_zleps",
    #"dr_zl0_zl1",
    #"j0eta",
    #"j0phi",
    "j0pt",
    #"j1eta",
    #"j1phi",
    "j1pt",
    "l0pt",
    "l1pt",
    "l2pt",
    "l3pt",
    #"metphi",
    #"met",
    "mjj_j0j1",
    "mjj_nearz",
    #"mll_01",
    "mllll",
    #"mll_min_afas",
    #"mll_min_afos",
    #"mll_min_sfos",
    #"mll_wl0_wl1",
    #"mll_zl0_zl1",
    #"mt2",
    "mt_4l_met",
    #"mt_wl0_met",
    #"mt_wl1_met",
    #"mt_wleps_met",
    "nbtagsl",
    "njets",
    #"nleps",
    #"ptl4",
    "pt_wl0_wl1",
    "pt_zl0_zl1",
    "scalarptsum_jet",
    #"scalarptsum_lepmetjet",
    #"scalarptsum_lepmet",
    #"scalarptsum_lep",
    #"w_lep0_eta",
    #"w_lep0_genPartFlav",
    #"w_lep0_phi",
    #"w_lep0_pt",
    #"w_lep1_eta",
    #"w_lep1_genPartFlav",
    #"w_lep1_phi",
    #"w_lep1_pt",
    #"z_lep0_eta",
    #"z_lep0_genPartFlav",
    #"z_lep0_phi",
    #"z_lep0_pt",
    #"z_lep1_eta",
    #"z_lep1_genPartFlav",
    #"z_lep1_phi",
    #"z_lep1_pt",
]

VAR_LST_TMP_FULL = []
for var_base in VAR_LST_TMP:
    VAR_LST_TMP_FULL.append("base_bdt_"+var_base)


def prepare_bdt_training_data(pklfilepath):

    # Open up the pickle file
    d = pickle.load(gzip.open(pklfilepath))

    # Dictionary where we will store our master data
    myd = {}

    list_output_names = [
        "list_bdt_base_proc",
        "list_bdt_base_wgt",
        "list_bdt_base_evt",
    ]

    get_ec_param = GetParam(ewkcoffea_path("params/params.json"))

    # Parse the pickle file and store to myd as numpy arrays
    for list_output_name in list_output_names: 
        myd[list_output_name] = np.array(d[list_output_name])
    for var in VAR_LST_TMP: 
        myd["base_bdt_" + var] = np.array(d["list_base_bdt_" + var])

    # Sample groupings to split the data into signals, and backgrounds
    sample_dict_mc = sg.create_mc_sample_dict("run2")

    # List of categories that will use BDT to label
    #categories = ["ZZZ", "WZZ", "Bkg"]
    categories = ["Sig", "Bkg"]

    # Sample splittings
    sample_splittings = ["train", "test", "all"] # later we can add validate if needed

    # Dictionary where we will store our training and testing data
    bdt_data = {}

    # Creating data structure in bdt_data
    # axis 0: categories
    # axis 1: split
    # axis 2: bdt variables and weight
    for cat in categories:
        bdt_data[cat] = {}
        for split in sample_splittings:
            bdt_data[cat][split] = {}
            # BDT inputs
            for var in VAR_LST_TMP: bdt_data[cat][split]["base_bdt_" + var] = np.array([])
            # also weight
            bdt_data[cat][split]["base_bdt_weight"] = np.array([])
            #bdt_data[cat][split]["sf_bdt_weight"] = np.array([])
            # also event number
            bdt_data[cat][split]["base_bdt_event"] = np.array([])
            #bdt_data[cat][split]["sf_bdt_event"] = np.array([])


    # Loop over sample groupings
    for proc in sample_dict_mc:

        # Grouping handling for background (all backgrounds are grouped as "Bkg")
        if proc in ["ZZ", "ttZ", "VHnobb", "other"]:
            proc_cat = "Bkg"
        if proc in ["ZZZ", "WZZ"]:
            proc_cat = "Sig"

        # Loop over all the groupings and fill the empty numpy array with the actual inputs
        for sample_name in sample_dict_mc[proc]:

            # Processing OF BDT inputs
            for var in VAR_LST_TMP:

                bdt_vname = "base_bdt_" + var

                # Get the original numpy array
                a = bdt_data[proc_cat]["all"][bdt_vname]

                # Get more variables for this sample_name
                # masking is done via "bdt_of/sf_proc_list" which holds sample_name (that is how it was saved in wwz4l.py)
                b = myd[bdt_vname][myd["list_bdt_base_proc"] == sample_name]

                # Concatenate
                c = np.concatenate((a, b))

                # Set it back to bdt_data where we will store
                bdt_data[proc_cat]["all"][bdt_vname] = c


            # We need to save weights as well
            bdt_data[proc_cat]["all"]["base_bdt_weight"] = np.concatenate((bdt_data[proc_cat]["all"]["base_bdt_weight"], myd["list_bdt_base_wgt"][myd["list_bdt_base_proc"] == sample_name]))

            # We need to save event numbers as well
            bdt_data[proc_cat]["all"]["base_bdt_event"] = np.concatenate((bdt_data[proc_cat]["all"]["base_bdt_event"], myd["list_bdt_base_evt"][myd["list_bdt_base_proc"] == sample_name]))



    # Split the testing and training by event number (even event number vs. odd event number)
    for cat in categories:
        for key in bdt_data[cat]["all"].keys():
            print("key",key)
            if "base" in key:
                test = bdt_data[cat]["all"][key][bdt_data[cat]["all"]["base_bdt_event"] % 2 == 1] # Getting events with event number that is even
                train = bdt_data[cat]["all"][key][bdt_data[cat]["all"]["base_bdt_event"] % 2 == 0] # Getting events with event number that is odd
                bdt_data[cat]["test"][key] = test
                bdt_data[cat]["train"][key] = train


    return bdt_data

def main():

    # Set up the command line parser
    parser = argparse.ArgumentParser()
    parser.add_argument("pkl_file_path", help = "The path to the pkl file")
    args = parser.parse_args()

    bdt_data = prepare_bdt_training_data(args.pkl_file_path)

    # Format:
    #{
    #    "Sig" : {
    #        "all" : {
    #            "tag_weight" : [],
    #            "tag_event" : [],
    #            "tag_var1" : [],
    #            "tag_var2" : [],
    #         },
    #        "test" : {
    #            ...
    #        }
    #        "train" : {
    #            ...
    #        },
    #    "Bkg" : {
    #        ...
    #    },
    #}

    #print(bdt_data.keys())
    #for k in bdt_data:
    #    print("\n",k)
    #    print(k,bdt_data[k].keys())
    #    for kk in bdt_data[k]:
    #        print("\n",kk)
    #        for kkk in bdt_data[k][kk]:
    #            print("\nHere:",k,kk,kkk)
    #            #print(type(bdt_data[k][kk][kkk]))
    #            #print(len(bdt_data[k][kk][kkk]),bdt_data[k][kk][kkk])
    #exit()

    # Gathering categories and variable names
    categories = bdt_data.keys()
    base_variables = []
    #sf_variables = []
    for cat in bdt_data.keys():
        for key in bdt_data[cat]["train"].keys():
            if "weight" in key: # skip weight
                continue
            if "event" in key: # skip weight
                continue
            if "base_bdt_" in key:
                base_variables.append(key)

    # Integer labeling (WWZ = 0, ZH = 1, Bkg = 2)
    label_d = {}
    #label_d["ZZZ"] = 0
    #label_d["WZZ"] = 1
    #label_d["Bkg"] = 2

    label_d["Sig"] = 0
    label_d["Bkg"] = 1

    # Remove duplicates and get only unique variable names
    # Since we looped over multiple categories same variable names were put in
    base_variables = list(set(base_variables))
    #sf_variables = list(set(sf_variables))

    # Building XGBoost input (which needs to be in a flat matrix)
    X_train_of = [] # Input variables
    y_train_of = np.array([]) # Output labels
    w_train_of = np.array([]) # Output labels
    X_test_of = [] # Input variables
    y_test_of = np.array([]) # Output labels
    w_test_of = np.array([]) # Output labels

    # Allocate some empty arrays where we will store things
    for var in VAR_LST_TMP : X_train_of.append(np.array([])) # Allocate N variable worth of lists
    for var in VAR_LST_TMP : X_test_of.append(np.array([])) # Allocate N variable worth of lists

    # Loop over the variables and store
    final_var_lst = []
    print(base_variables)
    print(len(base_variables))
    for cat in bdt_data.keys():
        # Store the variables
        for ivar, var in enumerate(VAR_LST_TMP_FULL):
            final_var_lst.append(var)
            X_train_of[ivar] = np.concatenate((X_train_of[ivar], bdt_data[cat]["train"][var]))
            X_test_of[ivar] = np.concatenate((X_test_of[ivar], bdt_data[cat]["test"][var]))
        # Take the last variable and get length and create labels
        y_train_of = np.concatenate((y_train_of, np.full(len(bdt_data[cat]["train"][var]), label_d[cat])))
        y_test_of = np.concatenate((y_test_of, np.full(len(bdt_data[cat]["test"][var]), label_d[cat])))
        w_train_of = np.concatenate((w_train_of, bdt_data[cat]["train"]["base_bdt_weight"]))
        w_test_of = np.concatenate((w_test_of, bdt_data[cat]["test"]["base_bdt_weight"]))

    # Turn them into numpy array
    X_train_of = np.array(X_train_of)
    X_test_of = np.array(X_test_of)

    # Transpose to have the events in rows and variables in columns
    X_train_of = X_train_of.T
    X_test_of = X_test_of.T

    print("Printing OF training sample numpy array shapes")
    print(X_train_of.shape)
    print(y_train_of.shape)
    print(w_train_of.shape)
    print("Printing OF testing sample numpy array shapes")
    print(X_test_of.shape)
    print(y_test_of.shape)
    print(w_test_of.shape)

    dd = {}
    dd["X_train_of"] = X_train_of
    dd["y_train_of"] = y_train_of
    dd["w_train_of"] = w_train_of
    dd["X_test_of"] = X_test_of
    dd["y_test_of"] = y_test_of
    dd["w_test_of"] = w_test_of

    dd["var_name_lst"] = VAR_LST_TMP_FULL

    with gzip.open("bdt.pkl.gz", "wb") as fout:
        cloudpickle.dump(dd, fout)

if __name__ == "__main__":

    main()
