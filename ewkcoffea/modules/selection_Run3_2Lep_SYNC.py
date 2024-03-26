import numpy as np
import awkward as ak



import topcoffea.modules.event_selection as tc_es

from ewkcoffea.modules.paths import ewkcoffea_path
from topcoffea.modules.get_param_from_jsons import GetParam
get_ec_param = GetParam(ewkcoffea_path("params/params.json"))

#------------------------------------------------------------------------------------------------------------------------------
def addMuTriggerMask(events):

    # Filters
    triggers = events.HLT

    trigger_mask = triggers.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass3p8 | triggers.Mu17_TrkIsoVVL_Mu8_TrkIsoVVL_DZ_Mass8 | triggers.IsoMu24 | triggers.IsoMu27 | triggers.Mu50

    mask = trigger_mask

    events['MuTrigMask'] = ak.fill_none(mask,False)

#-----------------------------------------------------------------------------------------------------------------------------
def addEleTriggerMask(events):

    # Filters
    triggers = events.HLT

    trigger_mask = triggers.Ele23_Ele12_CaloIdL_TrackIdL_IsoVL_DZ | triggers.Ele23_Ele12_CaloIdL_TrackIdL_IsoVL | triggers.Ele30_WPTight_Gsf | triggers.Ele32_WPTight_Gsf |triggers.Ele32_WPTight_Gsf_L1DoubleEG |triggers.Ele35_WPTight_Gsf | triggers.Ele115_CaloIdVT_GsfTrkIdT | triggers.DoubleEle25_CaloIdL_MW


    mask = trigger_mask

    events['EleTrigMask'] = ak.fill_none(mask,False)

#-----------------------------------------------------------------------------------------------------------------------------
# 2Lep Selection
def add2lmask_Run3_2Lep(events, year, isData):

    # Leptons and padded leptons
    leps = events.l_Run3_2Lep_tight

    # Lep multiplicity
    nlep_2 = (ak.num(leps) == 2)

    #mask = filters & nlep_2
    mask = nlep_2

    events['is2l'] = ak.fill_none(mask,False)

#------------------------------------------------------------------------------------------------------------------------------
# Do Run3 2Lep pre selection, construct event level mask
# Convenience function around get_Run3_2Lep_candidates() and get_z_candidate_mask()
def attach_Run3_2Lep_preselection_mask(events,lep_collection):

    # Pt requirements (assumes lep_collection is pt sorted and padded)
    pt_mask = ak.fill_none((lep_collection[:,0].pt > 25.0) & (lep_collection[:,1].pt > 25.0),False)
    #pt_mask = ak.fill_none(pt_mask,False) 

    # SFOS and OPOS masks
    os_sf_mask = ak.any((((lep_collection[:,0:1].pdgId) + (lep_collection[:,1:2].pdgId)) == 0),axis=1) 
    ee_mask = ak.any((abs(lep_collection[:,0:1].pdgId) == 11),axis=1) 
    mumu_mask = ak.any((abs(lep_collection[:,0:1].pdgId) == 13),axis=1) 

    os_sf_mask = ak.fill_none(os_sf_mask,False) 
    ee_mask = ak.fill_none(ee_mask,False) 
    mumu_mask = ak.fill_none(mumu_mask,False) 

    # The final preselection mask
    Run3_2Lep_presel_mask = (os_sf_mask & pt_mask)

    # Attach to the lepton objects
    events["Run3_2Lep_presel_sf_ee"] = (Run3_2Lep_presel_mask & ee_mask)
    events["Run3_2Lep_presel_sf_mumu"] = (Run3_2Lep_presel_mask & mumu_mask)


