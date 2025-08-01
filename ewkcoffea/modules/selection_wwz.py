import numpy as np
import awkward as ak
import xgboost as xgb
from mt2 import mt2

from coffea.nanoevents.methods import vector

import topcoffea.modules.event_selection as tc_es

from ewkcoffea.modules.paths import ewkcoffea_path
from topcoffea.modules.get_param_from_jsons import GetParam
get_ec_param = GetParam(ewkcoffea_path("params/params.json"))

# Loading of the TMVA classifier
import ROOT as r
import os
dirpath = os.path.dirname(os.path.abspath(__file__))
r.gROOT.ProcessLine(".L {}".format(ewkcoffea_path("data/wwz_zh_ternary_bdt/tmva_multiclassifier.C")))

bdt_of_v7 = r.ewkcoffea.BDT_OF_v7(ewkcoffea_path("data/wwz_zh_ternary_bdt/BDT_OF_v7__050124_Ternary_BDTG_LR0p1.weights.xml"))
bdt_sf_v7 = r.ewkcoffea.BDT_SF_v7(ewkcoffea_path("data/wwz_zh_ternary_bdt/BDT_SF_v7__050124_Ternary_BDTG_LR0p1.weights.xml"))

# The datasets we are using, and the triggers in them
dataset_dict = {

    "2016" : {
        "DoubleMuon" : [
            "Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ",
            "Mu17_TrkIsoVVL_Mu8_TrkIsoVVL",
            "Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL",
            "Mu17_TrkIsoVVL_TkMu8_TrkIsoVVL_DZ",
        ],
        "DoubleEG" : [
            "Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
        ],
        "MuonEG" : [
            "Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL",
            "Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
            "Mu23_TrkIsoVVL_Ele8_CaloIdL_TrackIdL_IsoVL",
            "Mu23_TrkIsoVVL_Ele8_CaloIdL_TrackIdL_IsoVL_DZ",
        ]
    },

    "2017" : {
        "DoubleMuon" : [
            "Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8",
        ],
        "DoubleEG" : [
            "Ele23_Ele12_CaloIdL_TrackIdL_IsoVL",
        ],
        "MuonEG" : [
            "Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
            "Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
        ]
    },

    "2018" : {
        "EGamma" : [
            "Ele23_Ele12_CaloIdL_TrackIdL_IsoVL",
        ],
        "DoubleMuon" : [
            "Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8",
        ],
        "MuonEG" : [
            "Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
            "Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
        ]
    },

    "2022" : {
        "EGamma" : [
            "Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
        ],
        "Muon" : [
            "Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8",
        ],
        "DoubleMuon" : [
            "Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8",
        ],
        "MuonEG" : [
            "Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
            "Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
        ]
    },
    "2023" : {
        "EGamma" : [
            "Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
        ],
        "Muon" : [
            "Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8",
        ],
        "MuonEG" : [
            "Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ",
            "Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ",
        ]
    }
}

trgs_for_matching = {

    "2016" : {
        "m_m" : {
            "trg_lst" : dataset_dict["2016"]["DoubleMuon"],
            "offline_thresholds" : [20.0,10.0],
        },
        "e_e" : {
            "trg_lst" : dataset_dict["2016"]["DoubleEG"],
            "offline_thresholds" : [25.0,15.0],
        },
        "m_e" : {
            "trg_lst" : ["Mu23_TrkIsoVVL_Ele8_CaloIdL_TrackIdL_IsoVL","Mu23_TrkIsoVVL_Ele8_CaloIdL_TrackIdL_IsoVL_DZ"],
            "offline_thresholds" : [25,10],
        },
        "e_m" : {
            "trg_lst" : ["Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL","Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ"],
            "offline_thresholds" : [25.0,10.0],
        },
    },
    "2017" : {
        "m_m" : {
            "trg_lst" : dataset_dict["2017"]["DoubleMuon"],
            "offline_thresholds" : [20.0,10.0],
        },
        "e_e" : {
            "trg_lst" : dataset_dict["2017"]["DoubleEG"],
            "offline_thresholds" : [25.0,15.0],
        },
        "m_e" : {
            "trg_lst" : ["Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ"],
            "offline_thresholds" : [25.0,15.0],
        },
        "e_m" : {
            "trg_lst" : ["Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ"],
            "offline_thresholds" : [25.0,10.0],
        },
    },
    "2018" : {
        "m_m" : {
            "trg_lst" : dataset_dict["2018"]["DoubleMuon"],
            "offline_thresholds" : [20.0,10.0],
        },
        "e_e" : {
            "trg_lst" : dataset_dict["2018"]["EGamma"],
            "offline_thresholds" : [25.0,15.0],
        },
        "m_e" : {
            "trg_lst" : ["Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ"],
            "offline_thresholds" : [25.0,15.0],
        },
        "e_m" : {
            "trg_lst" : ["Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ"],
            "offline_thresholds" : [25.0,10.0],
        },
    },
    "2022" : {
        "m_m" : {
            "trg_lst" : dataset_dict["2022"]["Muon"],
            "offline_thresholds" : [20.0,10.0],
        },
        "e_e" : {
            "trg_lst" : dataset_dict["2022"]["EGamma"],
            "offline_thresholds" : [25.0,15.0],
        },
        "m_e" : {
            "trg_lst" : ["Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ"],
            "offline_thresholds" : [25.0,15.0],
        },
        "e_m" : {
            "trg_lst" : ["Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ"],
            "offline_thresholds" : [25.0,10.0],
        },
    },
    "2023" : {
        "m_m" : {
            "trg_lst" : dataset_dict["2023"]["Muon"],
            "offline_thresholds" : [20.0,10.0],
        },
        "e_e" : {
            "trg_lst" : dataset_dict["2023"]["EGamma"],
            "offline_thresholds" : [25.0,15.0],
        },
        "m_e" : {
            "trg_lst" : ["Mu23_TrkIsoVVL_Ele12_CaloIdL_TrackIdL_IsoVL_DZ"],
            "offline_thresholds" : [25.0,15.0],
        },
        "e_m" : {
            "trg_lst" : ["Mu8_TrkIsoVVL_Ele23_CaloIdL_TrackIdL_IsoVL_DZ"],
            "offline_thresholds" : [25.0,10.0],
        },
    }
}


# Hard coded dictionary for figuring out overlap...
#   - No unique way to do this
#   - Note: In order for this to work properly, you should be processing all of the datastes to be used in the analysis
#   - Otherwise, you may be removing events that show up in other datasets you're not using
# For Era C which has both, the events in (SingleMuon, DoubleMuon) and (Muon) are exclusive so we do not perform duplicate removal bet
# For Era C, SingleMuon and DoubleMuon fall in the run ranges of [355800,357399] while Muon falls in [356400,357400]
exclude_dict = {
    "2016": {
        "DoubleMuon"     : [],
        "DoubleEG"       : dataset_dict["2016"]["DoubleMuon"],
        "MuonEG"         : dataset_dict["2016"]["DoubleMuon"] + dataset_dict["2016"]["DoubleEG"],
    },
    "2017": {
        "DoubleMuon"     : [],
        "DoubleEG"       : dataset_dict["2017"]["DoubleMuon"],
        "MuonEG"         : dataset_dict["2017"]["DoubleMuon"] + dataset_dict["2017"]["DoubleEG"],
    },
    "2018": {
        "DoubleMuon"     : [],
        "EGamma"         : dataset_dict["2018"]["DoubleMuon"],
        "MuonEG"         : dataset_dict["2018"]["DoubleMuon"] + dataset_dict["2018"]["EGamma"],
    },
    "C": {
        "Muon"           : [],
        "DoubleMuon"     : [],
        "EGamma"         : dataset_dict["2022"]["Muon"] + dataset_dict["2022"]["DoubleMuon"],
        "MuonEG"         : dataset_dict["2022"]["Muon"] + dataset_dict["2022"]["DoubleMuon"] + dataset_dict["2022"]["EGamma"],
    },
    "D": {
        "Muon"     : [],
        "EGamma"         : dataset_dict["2022"]["Muon"],
        "MuonEG"         : dataset_dict["2022"]["Muon"] + dataset_dict["2022"]["EGamma"],
    },
    "E": {
        "Muon"     : [],
        "EGamma"         : dataset_dict["2022"]["Muon"],
        "MuonEG"         : dataset_dict["2022"]["Muon"] + dataset_dict["2022"]["EGamma"],
    },
    "F": {
        "Muon"     : [],
        "EGamma"         : dataset_dict["2022"]["Muon"],
        "MuonEG"         : dataset_dict["2022"]["Muon"] + dataset_dict["2022"]["EGamma"],
    },
    "G": {
        "Muon"     : [],
        "EGamma"         : dataset_dict["2022"]["Muon"],
        "MuonEG"         : dataset_dict["2022"]["Muon"] + dataset_dict["2022"]["EGamma"],
    },
    "C1": {
        "Muon"           : [],
        "EGamma"         : dataset_dict["2023"]["Muon"],
        "MuonEG"         : dataset_dict["2023"]["Muon"] + dataset_dict["2023"]["EGamma"],
    },
    "C2": {
        "Muon"           : [],
        "EGamma"         : dataset_dict["2023"]["Muon"],
        "MuonEG"         : dataset_dict["2023"]["Muon"] + dataset_dict["2023"]["EGamma"],
    },
    "C3": {
        "Muon"           : [],
        "EGamma"         : dataset_dict["2023"]["Muon"],
        "MuonEG"         : dataset_dict["2023"]["Muon"] + dataset_dict["2023"]["EGamma"],
    },
    "C4": {
        "Muon"           : [],
        "EGamma"         : dataset_dict["2023"]["Muon"],
        "MuonEG"         : dataset_dict["2023"]["Muon"] + dataset_dict["2023"]["EGamma"],
    },
    "D1": {
        "Muon"           : [],
        "EGamma"         : dataset_dict["2023"]["Muon"],
        "MuonEG"         : dataset_dict["2023"]["Muon"] + dataset_dict["2023"]["EGamma"],
    },
    "D2": {
        "Muon"           : [],
        "EGamma"         : dataset_dict["2023"]["Muon"],
        "MuonEG"         : dataset_dict["2023"]["Muon"] + dataset_dict["2023"]["EGamma"],
    },
}


# Apply trigger matching requirements to make sure pt is above online thresholds
def trg_matching(events,year):

    # The trigger for 2016 and 2016APV are the same along with 2022EE->2022 and 2023BPix->2023
    if year == "2016APV" : year = "2016"
    if year == "2022EE"  : year = "2022"
    if year == "2023BPix": year = "2023"

    # Initialize return array to be True array with same shape as events
    ret_arr = ak.zeros_like(np.array(events.event), dtype=bool)

    # Get the leptons, sort and pad
    el = events.l_wwz_t[abs(events.l_wwz_t.pdgId)==11]
    el = ak.pad_none(el[ak.argsort(el.pt,axis=-1,ascending=False)],2)
    mu = events.l_wwz_t[abs(events.l_wwz_t.pdgId)==13]
    mu = ak.pad_none(mu[ak.argsort(mu.pt,axis=-1,ascending=False)],2)

    # Loop over offline cuts, make sure triggers pass the offline cuts for the associated triggers
    for l_l in trgs_for_matching[year]:

        # Check if lep pt passes the offline cuts
        offline_thresholds = trgs_for_matching[year][l_l]["offline_thresholds"]
        if   l_l == "m_m": offline_cut = ak.fill_none(((mu[:,0].pt > offline_thresholds[0]) & (mu[:,1].pt > offline_thresholds[1])),False)
        elif l_l == "e_e": offline_cut = ak.fill_none(((el[:,0].pt > offline_thresholds[0]) & (el[:,1].pt > offline_thresholds[1])),False)
        elif l_l == "m_e": offline_cut = ak.fill_none(((mu[:,0].pt > offline_thresholds[0]) & (el[:,0].pt > offline_thresholds[1])),False)
        elif l_l == "e_m": offline_cut = ak.fill_none(((el[:,0].pt > offline_thresholds[0]) & (mu[:,0].pt > offline_thresholds[1])),False)
        else: raise Exception("Unknown offline cut.")

        # Check if trigger passes the associated triggers
        trg_lst = trgs_for_matching[year][l_l]["trg_lst"]
        trg_passes = tc_es.passes_trg_inlst(events,trg_lst)

        # Build the return mask
        # The return mask started from an array of False
        # The way an event becomes True is if it passes a trigger AND passes the offline pt cuts associated with that trg
        false_arr = ak.zeros_like(np.array(events.event), dtype=bool) # False array with same shape as events
        ret_arr = ret_arr | ak.where(trg_passes,offline_cut,false_arr)

    return ret_arr


# 4l selection # SYNC
def add4lmask_wwz(events, year, isData, sample_name,is2022,is2023):

    # Leptons and padded leptons
    leps = events.l_wwz_t
    leps_padded = ak.pad_none(leps,4)

    # Filters
    filter_flags = events.Flag
    if (is2022 or is2023):
        filters = filter_flags.goodVertices & filter_flags.globalSuperTightHalo2016Filter & filter_flags.EcalDeadCellTriggerPrimitiveFilter & filter_flags.BadPFMuonFilter & filter_flags.ecalBadCalibFilter & filter_flags.BadPFMuonDzFilter & filter_flags.hfNoisyHitsFilter & filter_flags.eeBadScFilter
    elif year in ["2016","2016APV"]:
        filters = filter_flags.goodVertices & filter_flags.globalSuperTightHalo2016Filter & filter_flags.HBHENoiseFilter & filter_flags.HBHENoiseIsoFilter & filter_flags.EcalDeadCellTriggerPrimitiveFilter & filter_flags.BadPFMuonFilter & filter_flags.BadPFMuonDzFilter & filter_flags.eeBadScFilter
    else:
        filters = filter_flags.goodVertices & filter_flags.globalSuperTightHalo2016Filter & filter_flags.HBHENoiseFilter & filter_flags.HBHENoiseIsoFilter & filter_flags.EcalDeadCellTriggerPrimitiveFilter & filter_flags.BadPFMuonFilter & filter_flags.BadPFMuonDzFilter & filter_flags.eeBadScFilter & filter_flags.ecalBadCalibFilter

    # Lep multiplicity
    nlep_4 = (ak.num(leps) == 4)

    # Check if the leading lep associated with Z has pt>25 TODO Does this method actually do this?
    on_z = ak.fill_none(tc_es.get_Z_peak_mask(leps_padded[:,0:4],pt_window=10.0,zmass=get_ec_param("zmass")),False)

    # Remove low mass resonances
    cleanup = (events.min_mll_afos > 12)

    mask = filters & nlep_4 & on_z & cleanup

    # Do gen cleanups
    if sample_name in get_ec_param("vh_list"):
        genparts = events.GenPart
        is_zh = (abs(genparts[:,2].pdgId) == 23) # 3rd genparticle should be v for these samples
        is_w_from_h = ((abs(genparts.pdgId)==24) & (abs(genparts.distinctParent.pdgId) == 25))
        gen_mask = ~(is_zh & ak.any(is_w_from_h,axis=-1))
        mask = mask & gen_mask

    # TODO: Check if we need this, and add an if statement to not apply to data
    #lep1_match_prompt = ((leps_padded[:,0].genPartFlav==1) | (leps_padded[:,0].genPartFlav == 15))
    #lep2_match_prompt = ((leps_padded[:,1].genPartFlav==1) | (leps_padded[:,1].genPartFlav == 15))
    #lep3_match_prompt = ((leps_padded[:,2].genPartFlav==1) | (leps_padded[:,2].genPartFlav == 15))
    #lep4_match_prompt = ((leps_padded[:,3].genPartFlav==1) | (leps_padded[:,3].genPartFlav == 15))
    #prompt_mask = ( lep1_match_prompt & lep2_match_prompt & lep3_match_prompt & lep4_match_prompt)
    #mask = (mask & prompt_mask)

    # SFs:
    events['sf_4l_muon'] = leps_padded[:,0].sf_nom_muon*leps_padded[:,1].sf_nom_muon*leps_padded[:,2].sf_nom_muon*leps_padded[:,3].sf_nom_muon
    events['sf_4l_elec'] = leps_padded[:,0].sf_nom_elec*leps_padded[:,1].sf_nom_elec*leps_padded[:,2].sf_nom_elec*leps_padded[:,3].sf_nom_elec
    events['sf_4l_hi_muon'] = leps_padded[:,0].sf_hi_muon*leps_padded[:,1].sf_hi_muon*leps_padded[:,2].sf_hi_muon*leps_padded[:,3].sf_hi_muon
    events['sf_4l_hi_elec'] = leps_padded[:,0].sf_hi_elec*leps_padded[:,1].sf_hi_elec*leps_padded[:,2].sf_hi_elec*leps_padded[:,3].sf_hi_elec
    events['sf_4l_lo_muon'] = leps_padded[:,0].sf_lo_muon*leps_padded[:,1].sf_lo_muon*leps_padded[:,2].sf_lo_muon*leps_padded[:,3].sf_lo_muon
    events['sf_4l_lo_elec'] = leps_padded[:,0].sf_lo_elec*leps_padded[:,1].sf_lo_elec*leps_padded[:,2].sf_lo_elec*leps_padded[:,3].sf_lo_elec

    events['is4lWWZ'] = ak.fill_none(mask,False)


# Takes as input the lep collection
# Finds SFOS pair that is closest to the Z peak
# Returns object level mask with "True" for the leptons that are part of the Z candidate and False for others
def get_z_candidate_mask(lep_collection):

    # Attach the local index to the lepton objects
    lep_collection['idx'] = ak.local_index(lep_collection, axis=1)

    # Make all pairs of leptons
    ll_pairs = ak.combinations(lep_collection, 2, fields=["l0","l1"])
    ll_pairs_idx = ak.argcombinations(lep_collection, 2, fields=["l0","l1"])

    # Check each pair to see how far it is from the Z
    dist_from_z_all_pairs = abs((ll_pairs.l0+ll_pairs.l1).mass - get_ec_param("zmass"))

    # Mask out the pairs that are not SFOS (so that we don't include them when finding the one that's closest to Z)
    # And then of the SFOS pairs, get the index of the one that's cosest to the Z
    sfos_mask = (ll_pairs.l0.pdgId == -ll_pairs.l1.pdgId)
    dist_from_z_sfos_pairs = ak.mask(dist_from_z_all_pairs,sfos_mask)
    sfos_pair_closest_to_z_idx = ak.argmin(dist_from_z_sfos_pairs,axis=-1,keepdims=True)

    # Construct a mask (of the shape of the original lep array) corresponding to the leps that are part of the Z candidate
    mask = (lep_collection.idx == ak.flatten(ll_pairs_idx.l0[sfos_pair_closest_to_z_idx]))
    mask = (mask | (lep_collection.idx == ak.flatten(ll_pairs_idx.l1[sfos_pair_closest_to_z_idx])))
    mask = ak.fill_none(mask, False)

    return mask


# Get the pair of leptons that are the Z candidate, and the W candidate leptons
# Basicially this function is convenience wrapper around get_z_candidate_mask()
def get_wwz_candidates(lep_collection):

    z_candidate_mask = get_z_candidate_mask(lep_collection)

    # Now we can grab the Z candidate leptons and the non-Z candidate leptons
    leps_from_z_candidate = lep_collection[z_candidate_mask]
    leps_not_z_candidate = lep_collection[~z_candidate_mask]

    if ak.any(leps_from_z_candidate.pt) & ak.any(leps_not_z_candidate.pt):
        # Temp untill the ak argsort on None issue is resolved
        leps_from_z_candidate_ptordered = leps_from_z_candidate[ak.argsort(leps_from_z_candidate.pt, axis=-1,ascending=False)]
        leps_not_z_candidate_ptordered = leps_not_z_candidate[ak.argsort(leps_not_z_candidate.pt, axis=-1,ascending=False)] # This fails wehn the leps_not_z_candidate.pt is None, if/when we need this to be pt ordered will need to figure out how we want to work around this
        return [leps_from_z_candidate_ptordered,leps_not_z_candidate_ptordered]
    else:
        return [leps_from_z_candidate,leps_not_z_candidate]



# Do WWZ pre selection, construct event level mask
# Convenience function around get_wwz_candidates() and get_z_candidate_mask()
def attach_wwz_preselection_mask(events,lep_collection):

    leps_z_candidate, leps_w_candidate = get_wwz_candidates(lep_collection)

    # Pt requirements (assumes lep_collection is pt sorted and padded)
    pt_mask = ak.fill_none((lep_collection[:,0].pt > 25) & (lep_collection[:,1].pt > 15),False)

    # Build an event level mask for OS requirements for the W candidates
    os_mask = ak.any(((leps_w_candidate[:,0:1].pdgId)*(leps_w_candidate[:,1:2].pdgId)<0),axis=1) # Use ak.any() here so that instead of e.g [[None],None,...] we have [False,None,...]
    os_mask = ak.fill_none(os_mask,False) # Replace the None with False in the mask just to make it easier to think about

    # Build an event level mask for same flavor W lepton candidates
    sf_mask = ak.any((abs(leps_w_candidate[:,0:1].pdgId) == abs(leps_w_candidate[:,1:2].pdgId)),axis=1) # Use ak.any() here so that instead of e.g [[None],None,...] we have [False,None,...]
    sf_mask = ak.fill_none(sf_mask,False) # Replace the None with False in the mask just to make it easier to think about

    # Build an event level mask that checks if the z candidates are close enough to the z
    z_mass = (leps_z_candidate[:,0:1]+leps_z_candidate[:,1:2]).mass
    z_mass_mask = (abs((leps_z_candidate[:,0:1]+leps_z_candidate[:,1:2]).mass - get_ec_param("zmass")) < 10.0)
    z_mass_mask = ak.fill_none(ak.any(z_mass_mask,axis=1),False) # Make sure None entries are false

    # Build an event level mask to check the iso and sip3d for leps from Z and W
    leps_z_e = leps_z_candidate[abs(leps_z_candidate.pdgId)==11] # Just the electrons
    leps_w_e = leps_w_candidate[abs(leps_w_candidate.pdgId)==11] # Just the electrons
    iso_mask_z_e = ak.fill_none(ak.all((leps_z_e.pfRelIso03_all < get_ec_param("wwz_z_iso")),axis=1),False) # This requirement is just on the electrons
    iso_mask_w_e = ak.fill_none(ak.all((leps_w_e.pfRelIso03_all < get_ec_param("wwz_w_iso")),axis=1),False) # This requirement is just on the electrons
    id_mask_z = ak.fill_none(ak.all((leps_z_candidate.sip3d < get_ec_param("wwz_z_sip3d")),axis=1),False)
    id_mask_w = ak.fill_none(ak.all((leps_w_candidate.sip3d < get_ec_param("wwz_w_sip3d")),axis=1),False)
    id_iso_mask = (id_mask_z & id_mask_w & iso_mask_z_e & iso_mask_w_e)

    # The final preselection mask
    wwz_presel_mask = (os_mask & pt_mask & id_iso_mask)

    # Attach to the lepton objects
    events["wwz_presel"] = (wwz_presel_mask)
    events["wwz_presel_sf"] = (wwz_presel_mask & sf_mask)
    events["wwz_presel_of"] = (wwz_presel_mask & ~sf_mask)


# Get the MT variable
# See also https://en.wikipedia.org/wiki/Transverse_mass#Transverse_mass_in_two-particle_systems
def get_mt(p1,p2):
    return np.sqrt(2*p1.pt*p2.pt*(1 - np.cos(p1.delta_phi(p2))))

# Get MT2 for WW
def get_mt2(w_lep0,w_lep1,met):

    # Construct misspart vector, as implimented in c++: https://github.com/sgnoohc/mt2example/blob/main/main.cc#L7 (but pass 0 not pi/2 for met eta)
    nevents = len(np.zeros_like(met))
    misspart = ak.zip(
        {
            "pt": met.pt,
            "eta": 0,
            "phi": met.phi,
            "mass": np.full(nevents, 0),
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior,
    )
    # Do the boosts, as implimented in c++: https://github.com/sgnoohc/mt2example/blob/main/main.cc#L7
    rest_WW = w_lep0 + w_lep1 + misspart
    beta_from_miss_reverse = rest_WW.boostvec
    beta_from_miss = beta_from_miss_reverse.negative()
    w_lep0_boosted = w_lep0.boost(beta_from_miss)
    w_lep1_boosted = w_lep1.boost(beta_from_miss)
    misspart_boosted = misspart.boost(beta_from_miss)

    # Directly plug in e mass since its sometimes negative in naod
    mass_l0 = ak.where(abs(w_lep0.pdgId)==11,0.000511,w_lep0.mass)
    mass_l1 = ak.where(abs(w_lep1.pdgId)==11,0.000511,w_lep1.mass)


    # Get the mt2 variable, use the mt2 package: https://pypi.org/project/mt2/
    mt2_var = mt2(
        mass_l0, w_lep0_boosted.px, w_lep0_boosted.py,
        mass_l1, w_lep1_boosted.px, w_lep1_boosted.py,
        misspart_boosted.px, misspart_boosted.py,
        np.zeros_like(met.pt), np.zeros_like(met.pt),
    )

    return mt2_var


# Helicity function as defined here:
# https://github.com/cmstas/VVVNanoLooper/blob/46ee6437978e8be46a903f8f075e4d50c55f1573/analysis/process.cc#L2326-L2344
def helicity(p1,p2):
    parent = p1+p2
    boost_to_parent = parent.boostvec.negative()
    p1_new = p1.boost(boost_to_parent)
    p1_new_3 = p1_new.pvec # 3 vector
    parent_3 = parent.pvec # 3 vector
    cos_theta_1 = p1_new_3.dot(parent_3) / (p1_new_3.absolute()*parent_3.absolute())
    return abs(cos_theta_1)


# Evaluate the BDTs from Keegan
def eval_sig_bdt(events,in_vals,model_fpath):

    in_vals = np.array(in_vals)
    in_vals = np.transpose(in_vals)
    in_vals = xgb.DMatrix(in_vals) # The format xgb expects

    # Load model and evaluate
    xgb.set_config(verbosity = 0)
    bst = xgb.Booster()
    bst.load_model(model_fpath)
    score = bst.predict(in_vals)
    return score

def eval_of_tern_bdt(in_vals):
    rtn = bdt_of_v7.Eval(
        in_vals[0].to_list(),
        in_vals[1].to_list(),
        in_vals[2].to_list(),
        in_vals[3].to_list(),
        in_vals[4].to_list(),
        in_vals[5].to_list(),
        in_vals[6].to_list(),
        in_vals[7].to_list(),
        in_vals[8].to_list(),
        in_vals[9].to_list(),
        in_vals[10].to_list(),
        in_vals[11].to_list(),
        in_vals[12].to_list(),
        in_vals[13].to_list(),
        in_vals[14].to_list(),
        in_vals[15].to_list(),
        in_vals[16].to_list(),
        in_vals[17].to_list(),
        in_vals[18].to_list(),
        in_vals[19].to_list(),
        in_vals[20].to_list(),
        in_vals[21].to_list(),
        in_vals[22].to_list(),
        in_vals[23].to_list(),
        in_vals[24].to_list(),
        in_vals[25].to_list()
    )
    return rtn

def eval_sf_tern_bdt(in_vals):
    rtn = bdt_sf_v7.Eval(
        in_vals[0].to_list(),
        in_vals[1].to_list(),
        in_vals[2].to_list(),
        in_vals[3].to_list(),
        in_vals[4].to_list(),
        in_vals[5].to_list(),
        in_vals[6].to_list(),
        in_vals[7].to_list(),
        in_vals[8].to_list(),
        in_vals[9].to_list(),
        in_vals[10].to_list(),
        in_vals[11].to_list(),
        in_vals[12].to_list(),
        in_vals[13].to_list(),
        in_vals[14].to_list(),
        in_vals[15].to_list(),
        in_vals[16].to_list(),
        in_vals[17].to_list(),
        in_vals[18].to_list(),
        in_vals[19].to_list(),
        in_vals[20].to_list(),
        in_vals[21].to_list(),
        in_vals[22].to_list(),
        in_vals[23].to_list(),
        in_vals[24].to_list(),
        in_vals[25].to_list(),
        in_vals[26].to_list()
    )
    return rtn


###################### VVH ######################

# Just jilter flags
def get_filter_flag_mask_vvh(events, year, is2022,is2023):

    # Filters
    filter_flags = events.Flag
    if (is2022 or is2023):
        filters = filter_flags.goodVertices & filter_flags.globalSuperTightHalo2016Filter & filter_flags.EcalDeadCellTriggerPrimitiveFilter & filter_flags.BadPFMuonFilter & filter_flags.ecalBadCalibFilter & filter_flags.BadPFMuonDzFilter & filter_flags.hfNoisyHitsFilter & filter_flags.eeBadScFilter
    elif year in ["2016","2016APV"]:
        filters = filter_flags.goodVertices & filter_flags.globalSuperTightHalo2016Filter & filter_flags.HBHENoiseFilter & filter_flags.HBHENoiseIsoFilter & filter_flags.EcalDeadCellTriggerPrimitiveFilter & filter_flags.BadPFMuonFilter & filter_flags.BadPFMuonDzFilter & filter_flags.eeBadScFilter
    else:
        filters = filter_flags.goodVertices & filter_flags.globalSuperTightHalo2016Filter & filter_flags.HBHENoiseFilter & filter_flags.HBHENoiseIsoFilter & filter_flags.EcalDeadCellTriggerPrimitiveFilter & filter_flags.BadPFMuonFilter & filter_flags.BadPFMuonDzFilter & filter_flags.eeBadScFilter & filter_flags.ecalBadCalibFilter

    return filters
