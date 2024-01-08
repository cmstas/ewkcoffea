import numpy as np
import awkward as ak

from coffea.ml_tools.xgboost_wrapper import xgboost_wrapper

from topcoffea.modules.paths import topcoffea_path

from ewkcoffea.modules.paths import ewkcoffea_path
from topcoffea.modules.get_param_from_jsons import GetParam
get_ec_param = GetParam(ewkcoffea_path("params/params.json"))

# Clean collection b (e.g. jets) with collection a (e.g. leps)
def get_cleaned_collection(obj_collection_a,obj_collection_b,drcut=0.4):
    obj_b_nearest_to_any_in_a , dr = obj_collection_b.nearest(obj_collection_a,return_metric=True)
    mask = ak.fill_none(dr>drcut,True)
    return obj_collection_b[mask]


# Wrapper around evaluation of lep ID bdt, note returns flat
# Note this could potentially go somewhere more general like topcoffea modules
def xgb_eval_wrapper(feature_list,in_vals_flat_dict,model_fpath):

    # From https://github.com/CoffeaTeam/coffea/blob/master/tests/test_ml_tools.py#L169-L174
    class xgboost_test(xgboost_wrapper):
        def prepare_awkward(self, events):
            ak = self.get_awkward_lib(events)
            ret = ak.concatenate(
                [events[name][:, np.newaxis] for name in feature_list], axis=1
            )
            return [], dict(data=ret)

    # Reshape the input array
    # Note that if the inputs are object level like pt, then should have already flattned before passing to this function
    # E.g. we want to go from something like this:
    #   {
    #       "a" : ak.Array([1.1,  2.1, 3.2]]), # E.g. could have been [[1.1,2.1] ,[3.2]] prior to flattening
    #       "b" : ak.Array([-1.1,-2.1,-3.2]]), # E.g. could have been [[-1.1,-2.1],[-3.2]] prior to flattening
    #   }
    # To something that looks like this:
    #   input_arr  = ak.Array([
    #       {"a": 1.1, "b": -1.1},
    #       {"a": 2.1, "b": -2.1},
    #       {"a": 3.1, "b": -3.1},
    #   ])
    input_arr = ak.zip(
        {feat_name: in_vals_flat_dict[feat_name] for feat_name in in_vals_flat_dict.keys()}
    )

    # Get the score
    xgb_wrap = xgboost_test(model_fpath)
    score = xgb_wrap(input_arr)

    return score


######### WWZ 4l analysis object selection #########

# WWZ preselection for electrons
def is_presel_wwz_ele(ele,tight):
    mask = (
        (ele.pt               >  get_ec_param("wwz_pres_e_pt")) &
        (abs(ele.eta)         <  get_ec_param("wwz_pres_e_eta")) &
        (abs(ele.dxy)         <  get_ec_param("wwz_pres_e_dxy")) &
        (abs(ele.dz)          <  get_ec_param("wwz_pres_e_dz")) &
        (abs(ele.sip3d)       <  get_ec_param("wwz_pres_e_sip3d")) &
        (ele.miniPFRelIso_all <  get_ec_param("wwz_pres_e_miniPFRelIso_all")) &
        (ele.lostHits         <= get_ec_param("wwz_pres_e_lostHits"))
    )
    if tight: mask = (mask & ele.convVeto & (ele.tightCharge == get_ec_param("wwz_pres_e_tightCharge")))
    return mask


# WWZ preselection for muons
def is_presel_wwz_mu(mu):
    mask = (
        (mu.pt               >  get_ec_param("wwz_pres_m_pt")) &
        (abs(mu.eta)         <  get_ec_param("wwz_pres_m_eta")) &
        (abs(mu.dxy)         <  get_ec_param("wwz_pres_m_dxy")) &
        (abs(mu.dz)          <  get_ec_param("wwz_pres_m_dz")) &
        (abs(mu.sip3d)       <  get_ec_param("wwz_pres_m_sip3d")) &
        (mu.miniPFRelIso_all <  get_ec_param("wwz_pres_m_miniPFRelIso_all")) &
        (mu.mediumId)
    )
    return mask


# Get MVA score from TOP MVA for electrons
def get_topmva_score_ele(events, year):

    ele = events.Electron

    # Get the model path
    if (year == "2016"):      ulbase = "UL16"
    elif (year == "2016APV"): ulbase = "UL16APV"
    elif (year == "2017"):    ulbase = "UL17"
    elif (year == "2018"):    ulbase = "UL18"
    else: raise Exception(f"Error: Unknown year \"{year}\". Exiting...")
    model_fpath = topcoffea_path(f"data/topmva/lepid_weights/el_TOP{ulbase}_XGB.weights.bin")

    # Put some stuff into ele object
    ele["btagDeepFlavB"] = ak.fill_none(ele.matched_jet.btagDeepFlavB, 0)
    ele["jetPtRatio"] = 1./(ele.jetRelIso+1.)
    ele["miniPFRelIso_diff_all_chg"] = ele.miniPFRelIso_all - ele.miniPFRelIso_chg

    # List order comes from https://github.com/cmstas/VVVNanoLooper/blob/8a194165cdbbbee3bcf69f932d837e95a0a265e6/src/ElectronIDHelper.cc#L110-L122
    feature_lst = [
        "pt",
        "eta",
        "jetNDauCharged",
        "miniPFRelIso_chg",
        "miniPFRelIso_diff_all_chg",
        "jetPtRelv2",
        "jetPtRatio",
        "pfRelIso03_all",
        "ak4jet:btagDeepFlavB",
        "sip3d",
        "log_abs_dxy",
        "log_abs_dz",
        "mvaFall17V2noIso",
    ]

    # Flatten, and store in a dict for easy access
    in_vals_flat_dict = {
        "pt"                        : ak.flatten(ele.pt),
        "eta"                       : ak.flatten(ele.eta), # Kirill confirms that signed eta was used in the training
        "jetNDauCharged"            : ak.flatten(ele.jetNDauCharged),
        "miniPFRelIso_chg"          : ak.flatten(ele.miniPFRelIso_chg),
        "miniPFRelIso_diff_all_chg" : ak.flatten(ele.miniPFRelIso_diff_all_chg),
        "jetPtRelv2"                : ak.flatten(ele.jetPtRelv2),
        "jetPtRatio"                : ak.flatten(ele.jetPtRatio),
        "pfRelIso03_all"            : ak.flatten(ele.pfRelIso03_all),
        "ak4jet:btagDeepFlavB"      : ak.flatten(ele.btagDeepFlavB),
        "sip3d"                     : ak.flatten(ele.sip3d),
        "log_abs_dxy"               : ak.flatten(np.log(abs(ele.dxy))),
        "log_abs_dz"                : ak.flatten(np.log(abs(ele.dz))),
        "mvaFall17V2noIso"          : ak.flatten(ele.mvaFall17V2noIso),
    }

    score = xgb_eval_wrapper(feature_lst,in_vals_flat_dict,model_fpath)

    # Restore the shape (i.e. unflatten)
    counts = ak.num(ele.pt)
    score = ak.unflatten(score,counts)

    return score


# Get MVA score from TOP MVA for muons
def get_topmva_score_mu(events, year):

    mu = events.Muon

    # Get the model path
    if (year == "2016"):      ulbase = "UL16"
    elif (year == "2016APV"): ulbase = "UL16APV"
    elif (year == "2017"):    ulbase = "UL17"
    elif (year == "2018"):    ulbase = "UL18"
    else: raise Exception(f"Error: Unknown year \"{year}\". Exiting...")
    model_fpath = topcoffea_path(f"data/topmva/lepid_weights/mu_TOP{ulbase}_XGB.weights.bin")

    # Put some stuff into mu object
    mu["btagDeepFlavB"] = ak.fill_none(mu.matched_jet.btagDeepFlavB, 0)
    mu["jetPtRatio"] = 1./(mu.jetRelIso+1.)
    mu["miniPFRelIso_diff_all_chg"] = mu.miniPFRelIso_all - mu.miniPFRelIso_chg

    # Order comes from https://github.com/cmstas/VVVNanoLooper/blob/8a194165cdbbbee3bcf69f932d837e95a0a265e6/src/MuonIDHelper.cc#L102-L116
    feature_lst = [
        "pt",
        "eta",
        "jetNDauCharged",
        "miniPFRelIso_chg",
        "miniPFRelIso_diff_all_chg",
        "jetPtRelv2",
        "jetPtRatio",
        "pfRelIso03_all",
        "ak4jet:btagDeepFlavB",
        "sip3d",
        "log_abs_dxy",
        "log_abs_dz",
        "segmentComp",
    ]

    # Flatten, and store in a dict for easy access
    in_vals_flat_dict = {
        "pt"                       : ak.flatten(mu.pt),
        "eta"                      : ak.flatten(mu.eta), # Kirill confirms that signed eta was used in the training
        "jetNDauCharged"           : ak.flatten(mu.jetNDauCharged),
        "miniPFRelIso_chg"         : ak.flatten(mu.miniPFRelIso_chg),
        "miniPFRelIso_diff_all_chg": ak.flatten(mu.miniPFRelIso_diff_all_chg),
        "jetPtRelv2"               : ak.flatten(mu.jetPtRelv2),
        "jetPtRatio"               : ak.flatten(mu.jetPtRatio),
        "pfRelIso03_all"           : ak.flatten(mu.pfRelIso03_all),
        "ak4jet:btagDeepFlavB"     : ak.flatten(mu.btagDeepFlavB),
        "sip3d"                    : ak.flatten(mu.sip3d),
        "log_abs_dxy"              : ak.flatten(np.log(abs(mu.dxy))),
        "log_abs_dz"               : ak.flatten(np.log(abs(mu.dz))),
        "segmentComp"              : ak.flatten(mu.segmentComp),
    }

    score = xgb_eval_wrapper(feature_lst,in_vals_flat_dict,model_fpath)

    # Restore the shape (i.e. unflatten)
    counts = ak.num(mu.pt)
    score = ak.unflatten(score,counts)

    return score
