from hist import axis
from coffea.nanoevents.methods import vector
import awkward as ak
ak.behavior.update(vector.behavior)
import numpy as np

obj = { #reconstruction of H, V1, V2, vbsjs and MET
    "Higgs": lambda events: ak.zip(
        {
            "pt": events.Higgs_pt,
            "mass": events.Higgs_msoftdrop,
            "eta": events.Higgs_eta,
            "phi": events.Higgs_phi,
            "score": events.HiggsScore,
            # "area": events.Higgs_area,
            # "deepTagMD_TvsQCD": events.Higgs_deepTagMD_TvsQCD,
            # "deepTagMD_WvsQCD": events.Higgs_deepTagMD_WvsQCD,
            # "particleNetMD_QCD": events.Higgs_particleNetMD_QCD,
            # "particleNet_TvsQCD": events.Higgs_particleNet_TvsQCD,
            # "particleNet_WvsQCD": events.Higgs_particleNet_WvsQCD,
            # "particleNet_mass": events.Higgs_particleNet_mass,
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior
    ),
    "V1": lambda events: ak.zip(
        {
            "pt": events.V1_pt,
            "mass": events.V1_msoftdrop,
            "eta": events.V1_eta,
            "phi": events.V1_phi,
            "score": events.V1Score,
            # "area": events.V1_area,
            # "deepTagMD_TvsQCD": events.V1_deepTagMD_TvsQCD,
            # "deepTagMD_WvsQCD": events.V1_deepTagMD_WvsQCD,
            # "particleNetMD_QCD": events.V1_particleNetMD_QCD,
            # "particleNet_TvsQCD": events.V1_particleNet_TvsQCD,
            # "particleNet_WvsQCD": events.V1_particleNet_WvsQCD,
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior
    ),
    "vbsj1": lambda events: ak.zip(
        {
            "pt": events.vbsj1_pt,
            "mass": events.vbsj1_m,
            "eta": events.vbsj1_eta,
            "phi": events.vbsj1_phi,
            # "area": ak.firsts(events.goodVBSJets_area[events.vbs_idx_max_Mjj[:,0]]),
            # "nConstituents": ak.firsts(events.goodVBSJets_nConstituents[events.vbs_idx_max_Mjj[:,0]]),
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior
    ),
    "vbsj2": lambda events: ak.zip(
        {
            "pt": events.vbsj2_pt,
            "mass": events.vbsj2_m,
            "eta": events.vbsj2_eta,
            "phi": events.vbsj2_phi,
            # "area": ak.firsts(events.goodVBSJets_area[events.vbs_idx_max_Mjj[:,1]]),
            # "nConstituents": ak.firsts(events.goodVBSJets_nConstituents[events.vbs_idx_max_Mjj[:,1]]),
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior
    ),
    "MET": lambda events: ak.zip(
        {
            "pt": events.Met_pt,
            "phi": events.Met_phi,
            "eta": ak.zeros_like(events.Met_pt),  # Set to 0 to ensure valid LorentzVector
            "mass": ak.zeros_like(events.Met_pt), # Same for mass
            "significance": events.Met_significance,
            'sumEt': events.Met_sumEt,
            'sumPtUnclustered': events.Met_sumPtUnclustered,
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior
    ),
    "leadAK8": lambda events: ak.zip(
        {
            "pt": events.leadAK8_pt,
            "mass": events.leadAK8_msoftdrop,
            "eta": events.leadAK8_eta,
            "phi": events.leadAK8_phi,
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior
    ),
    "goodAK4Jets": lambda events: ak.zip( #for getting lead ak4 dphi met
        {
            "pt": events.goodAK4Jets_pt,
            "mass": events.goodAK4Jets_mass,
            "eta": events.goodAK4Jets_eta,
            "phi": events.goodAK4Jets_phi,
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior
    ),
    "goodAK8Jets": lambda events: ak.zip(
        {
            "pt": events.goodAK8Jets_pt,
            "mass": events.goodAK8Jets_msoftdrop,
            "eta": events.goodAK8Jets_eta,
            "phi": events.goodAK8Jets_phi,
            # "HbbScore":events.goodAK8Jets_HbbScore,
            # "WqqScore":events.goodAK8Jets_WqqScore,
            # "area": events.goodAK8Jets_area,
            # "deepTagMD_TvsQCD": events.goodAK8Jets_deepTagMD_TvsQCD,
            # "deepTagMD_WvsQCD": events.goodAK8Jets_deepTagMD_WvsQCD,
            # "particleNetMD_QCD": events.goodAK8Jets_particleNetMD_QCD,
            # "particleNet_TvsQCD": events.goodAK8Jets_particleNet_TvsQCD,
            # "particleNet_WvsQCD": events.goodAK8Jets_particleNet_WvsQCD,
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior
    ),
    # "electrons": lambda events: ak.zip(
    #     {
    #         "pt": events.Electron_pt,
    #         "mass": events.Electron_mass,
    #         "eta": events.Electron_eta,
    #         "phi": events.Electron_phi,
    #     },
    #     with_name="PtEtaPhiMLorentzVector",
    #     behavior=vector.behavior
    # ),
    # "muons": lambda events: ak.zip(
    #     {
    #         "pt": events.Muon_pt,
    #         "mass": events.Muon_mass,
    #         "eta": events.Muon_eta,
    #         "phi": events.Muon_phi,
    #     },
    #     with_name="PtEtaPhiMLorentzVector",
    #     behavior=vector.behavior
    # ),
    # "taus": lambda events: ak.zip(
    #     {
    #         "pt": events.Tau_pt,
    #         "mass": events.Tau_mass,
    #         "eta": events.Tau_eta,
    #         "phi": events.Tau_phi,
    #     },
    #     with_name="PtEtaPhiMLorentzVector",
    #     behavior=vector.behavior
    # ),
}

other_objs = { #reconstruction of all (good) jets and stuffs if needed
    "goodVBSJets": lambda events,objects: ak.zip(
        {
            "pt": events.goodAK4Jets_pt,
            "mass": events.goodAK4Jets_mass,
            "eta": events.goodAK4Jets_eta,
            "phi": events.goodAK4Jets_phi,
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior
    ),
    "V2": lambda events,objects: ak.zip( #for met2FJ channel, V2 should not exist
        {
            "pt": events.V2_pt,
            "mass": events.V2_msoftdrop,
            "eta": events.V2_eta,
            "phi": events.V2_phi,
            "score": events.V2Score,
        },
        with_name="PtEtaPhiMLorentzVector",
        behavior=vector.behavior
    ),
    "leadAK4": lambda events,objects: get_leading_jet(objects['goodAK4Jets']),
}

def get_leading_jet(jets):
    jets_sorted = jets[ak.argsort(jets.pt, axis=1, ascending=False)]
    return ak.firsts(jets_sorted)


dense_variables_config = { #name of axis must be same as key    
    "nGoodAK4": {
        "axis": axis.Regular(25, 0, 25, name="nGoodAK4", label="nGoodAK4"),
        "expr": lambda events, objects: events.nGoodAK4,
    },
    "nGoodAK8": {
        "axis": axis.Regular(6, 0, 6, name="nGoodAK8", label="nGoodAK8"),
        "expr": lambda events, objects: events.nGoodAK8,
    },
    "nAK4": {
        "axis": axis.Regular(25, 0, 25, name="nAK4", label="nAK4"),
        "expr": lambda events, objects: events.nAK4
    },
    "nAK8": {
        "axis": axis.Regular(6, 0, 6, name="nAK8", label="nAK8"),
        "expr": lambda events, objects: events.nAK8,
    },
    "Higgs_pt": {
        "axis": axis.Regular(50, 0, 2000, name="Higgs_pt", label="Higgs pt (GeV)"),
        "expr":  lambda events, objects: objects["Higgs"].pt,
    },
    "Higgs_phi": {
        "axis": axis.Regular(50, -3.5, 3.5, name="Higgs_phi", label="Higgs phi"),
        "expr":  lambda events, objects: objects["Higgs"].phi,
    },
    "Higgs_eta": {
        "axis": axis.Regular(50, 0,3, name="Higgs_eta", label="Higgs eta"),
        "expr":  lambda events, objects: objects["Higgs"].eta,
    },
    "Higgs_mass": {
        "axis": axis.Regular(50, 0, 400, name="Higgs_mass", label="Higgs mass (GeV)"),
        "expr":  lambda events, objects: objects["Higgs"].mass,
    },
    "Higgs_score": {
        "axis": axis.Regular(50, 0, 1, name="Higgs_score", label="Higgs score"),
        "expr":  lambda events, objects: objects["Higgs"].score,
    },
    "V1_pt": {
        "axis": axis.Regular(50, 0, 2000, name="V1_pt", label="V1 pt (GeV)"),
        "expr":  lambda events, objects: objects["V1"].pt,
    },
    "V1_phi": {
        "axis": axis.Regular(50, -3.5, 3.5, name="V1_phi", label="V1 phi"),
        "expr":  lambda events, objects: objects["V1"].phi,
    },
    "V1_eta": {
        "axis": axis.Regular(50, 0,3, name="V1_eta", label="V1 eta"),
        "expr":  lambda events, objects: objects["V1"].eta,
    },
    "V1_mass": {
        "axis": axis.Regular(50, 0, 400, name="V1_mass", label="V1 mass (GeV)"),
        "expr":  lambda events, objects: objects["V1"].mass,
    },
    "V1_score": {
        "axis": axis.Regular(50, 0, 1, name="V1_score", label="V1 score"),
        "expr":  lambda events, objects: objects["V1"].score,
    },
    "Met_pt":{
        "axis": axis.Regular(50, 0, 2000, name="Met_pt", label="MET pt (GeV)"),
        "expr":  lambda events, objects: objects["MET"].pt,
    },
    "Met_pt_low":{
        "axis": axis.Regular(80, 0, 800, name="Met_pt_low", label="MET pt (GeV)"),
        "expr":  lambda events, objects: objects["MET"].pt,
    },
    "Met_significance":{
        "axis": axis.Regular(50, 0, 2000, name="Met_significance", label="MET significance"),
        "expr":  lambda events, objects: objects["MET"].significance,
    },
    "Met_significance_fine":{
        "axis": axis.Regular(50, 0, 200, name="Met_significance_fine", label="MET significance zoomed in"),
        "expr":  lambda events, objects: objects["MET"].significance,
    },
    "Met_significance_fine10":{
        "axis": axis.Regular(20, 0, 200, name="Met_significance_fine10", label="MET significance zoomed in"),
        "expr":  lambda events, objects: objects["MET"].significance,
    },
    "Met_significance_lowrange":{
        "axis": axis.Regular(20, 0, 50, name="Met_significance_lowrange", label="MET significance in <50 reg."),
        "expr":  lambda events, objects: objects["MET"].significance,
    },
    "Met_phi":{
        "axis": axis.Regular(50, -3.5, 3.5, name="Met_phi", label="MET phi"),
        "expr":  lambda events, objects: objects["MET"].phi,
    },
    "HV1_dR": {
        "axis": axis.Regular(50, 0, 6, name="HV1_dR", label="dR(Higgs, V1)"),
        "expr": lambda events, objects: objects["Higgs"].delta_r(objects["V1"]),
    },
    "HMET_dphi": {
        "axis": axis.Regular(50, 0, 3.5, name="HMET_dphi", label="dphi(Higgs, MET)"),
        "expr": lambda events, objects: deltaPhi(objects["Higgs"],objects["MET"]),
    },
    "V1MET_dphi": {
        "axis": axis.Regular(50, 0, 3.5, name="V1MET_dphi", label="dphi(V1, MET)"),
        "expr": lambda events, objects: deltaPhi(objects["V1"],objects["MET"]),
    },
    "HVMET_sum_pt": {
        "axis": axis.Regular(50, 0, 2000, name="HVMET_sum_pt", label="pt(H + V1) (GeV)"),
        "expr": lambda events, objects: (objects["Higgs"] + objects["V1"] + objects["MET"]).pt,
    },
    "sum_bosonHT":{
        "axis": axis.Regular(50, 0, 5000, name="sum_bosonHT", label="(Higgs_pt+V1_pt+MET_pt)"),
        "expr": lambda events, objects: objects["Higgs"].pt + objects["V1"].pt +objects["MET"].pt,
    },
    "vbsj_deta":{
        "axis": axis.Regular(50, 0, 10, name="vbsj_deta", label="vbsj_deta"),
        "expr": lambda events, objects: np.abs(objects["vbsj1"].eta - objects["vbsj2"].eta),
    },
    "vbsj_Mjj":{
        "axis": axis.Regular(50, 0, 5000, name="vbsj_Mjj", label="vbsj_Mjj (GeV)"),
        "expr": lambda events, objects: (objects["vbsj1"] + objects["vbsj2"]).mass,
    },
    
    "goodAK4Jets_maxAbsDeltaEta":{
        "axis": axis.Regular(50, 0, 10, name="goodAK4Jets_maxAbsDeltaEta", label="goodAK4Jets_maxAbsDeltaEta"),
        "expr": lambda events, objects: events.goodAK4Jets_maxAbsDeltaEta,
    },
    "leadAK8_MET_dphi":{
        "axis": axis.Regular(50, 0, 3.5, name="leadAK8_MET_dphi", label="dphi(leadAK8, MET)"),
        "expr": lambda events, objects: deltaPhi(objects["leadAK8"],objects["MET"]),
    },
    
    "leadAK4_MET_dphi":{ #objects['goodAK4Jets']
        "axis": axis.Regular(50, 0, 3.5, name="leadAK4_MET_dphi", label="dphi(leadAK4, MET)"),
        "expr": lambda events, objects: 
            deltaPhi(objects['leadAK4'],objects["MET"]),
    },
    
    "vbsj1_pt": {
        "axis": axis.Regular(50, 0, 500, name="vbsj1_pt", label="vbsj1 pt (GeV)"),
        "expr":  lambda events, objects: objects["vbsj1"].pt,
    },
    "vbsj1_phi": {
        "axis": axis.Regular(50, -3.5, 3.5, name="vbsj1_phi", label="vbsj1 phi"),
        "expr":  lambda events, objects: objects["vbsj1"].phi,
    },
    "vbsj1_eta": {
        "axis": axis.Regular(50, 0,5, name="vbsj1_eta", label="vbsj1 eta"),
        "expr":  lambda events, objects: objects["vbsj1"].eta,
    },
    "vbsj1_mass": {
        "axis": axis.Regular(50, 0, 100, name="vbsj1_mass", label="vbsj1 mass (GeV)"),
        "expr":  lambda events, objects: objects["vbsj1"].mass,
    },
    "vbsj2_pt": {
        "axis": axis.Regular(50, 0, 500, name="vbsj2_pt", label="vbsj2 pt (GeV)"),
        "expr":  lambda events, objects: objects["vbsj2"].pt,
    },
    "vbsj2_phi": {
        "axis": axis.Regular(50, -3.5, 3.5, name="vbsj2_phi", label="vbsj2 phi"),
        "expr":  lambda events, objects: objects["vbsj2"].phi,
    },
    "vbsj2_eta": {
        "axis": axis.Regular(50, 0,5, name="vbsj2_eta", label="vbsj2 eta"),
        "expr":  lambda events, objects: objects["vbsj2"].eta,
    },
    "vbsj2_mass": {
        "axis": axis.Regular(50, 0, 100, name="vbsj2_mass", label="vbsj2 mass (GeV)"),
        "expr":  lambda events, objects: objects["vbsj2"].mass,
    },
    "met_goodak4_min_dphi": {
        "axis": axis.Regular(50, 0, 3.5, name="met_goodak4_min_dphi", label="met_goodak4_min_dphi"),
        "expr":  lambda events, objects: get_min(deltaPhi(objects['goodAK4Jets'],objects["MET"])),
    },
    "met_goodak8_min_dphi": {
        "axis": axis.Regular(50, 0, 3.5, name="met_goodak8_min_dphi", label="met_goodak8_min_dphi"),
        "expr":  lambda events, objects: get_min(deltaPhi(objects['goodAK8Jets'],objects["MET"])),
    },

    # "nlep": {
    #     "axis": axis.Regular(15, 0, 15, name="nlep", label="n(electron+muon)"),
    #     "expr":  lambda events, objects: events.nlep,
    # },
    # "nTau": {
    #     "axis": axis.Regular(5, 0, 5, name="nTau", label="nTau"),
    #     "expr":  lambda events, objects: events.nTau,
    # },
    # "subjet_ratio": {
    #     "axis": axis.Regular(20, 0, 4, name="subjet_ratio", label="nSubjet/nFatJet"),
    #     "expr":  lambda events, objects: events.subjet_ratio,
    # },
    # "nGoodAK4FromMediumBJet": {
    #     "axis": axis.Regular(5, 0, 5, name="nGoodAK4FromMediumBJet", label="nGoodAK4FromMediumBJet"),
    #     "expr": lambda events, objects: events.ngoodAK4FromMediumBJet, #ak.sum(events.goodAK4FromMediumBJet, axis=1),#
    # },
    # "nGoodAK4FromLooseBJet": {
    #     "axis": axis.Regular(5, 0, 5, name="nGoodAK4FromLooseBJet", label="nGoodAK4FromLooseBJet"),
    #     "expr": lambda events, objects: events.ngoodAK4FromLooseBJet, #ak.sum(events.goodAK4FromLooseBJet, axis=1),#
    # },
    # "Higgs_area": {
    #     "axis": axis.Regular(20, 1.5,2.5, name="Higgs_area", label="Higgs area"),
    #     "expr":  lambda events, objects: objects["Higgs"].area,
    # },
    # "Higgs_deepTagMD_TvsQCD": {
    #     "axis": axis.Regular(50, 0, 1, name="Higgs_deepTagMD_TvsQCD", label="Higgs score deepTagMD_TvsQCD"),
    #     "expr":  lambda events, objects: objects["Higgs"].deepTagMD_TvsQCD,
    # },
    # "Higgs_deepTagMD_WvsQCD": {
    #     "axis": axis.Regular(50, 0, 1, name="Higgs_deepTagMD_WvsQCD", label="Higgs deepTagMD_WvsQCD"),
    #     "expr":  lambda events, objects: objects["Higgs"].deepTagMD_WvsQCD,
    # },
    # "Higgs_particleNetMD_QCD": {
    #     "axis": axis.Regular(50, 0, 1, name="Higgs_particleNetMD_QCD", label="Higgs particleNetMD_QCD"),
    #     "expr":  lambda events, objects: objects["Higgs"].particleNetMD_QCD,
    # },
    # "Higgs_particleNet_TvsQCD": {
    #     "axis": axis.Regular(50, 0, 1, name="Higgs_particleNet_TvsQCD", label="Higgs particleNet_TvsQCD"),
    #     "expr":  lambda events, objects: objects["Higgs"].particleNet_TvsQCD,
    # },
    # "Higgs_particleNet_WvsQCD": {
    #     "axis": axis.Regular(50, 0, 1, name="Higgs_particleNet_WvsQCD", label="Higgs particleNet_WvsQCD"),
    #     "expr":  lambda events, objects: objects["Higgs"].particleNet_WvsQCD,
    # },
    # "V1_area": {
    #     "axis": axis.Regular(50, 1.5,2.5, name="V1_area", label="V1 area"),
    #     "expr":  lambda events, objects: objects["V1"].area,
    # },
    # "V1_deepTagMD_TvsQCD": {
    #     "axis": axis.Regular(50, 0, 1, name="V1_deepTagMD_TvsQCD", label="V1 score deepTagMD_TvsQCD"),
    #     "expr":  lambda events, objects: objects["V1"].deepTagMD_TvsQCD,
    # },
    # "V1_deepTagMD_WvsQCD": {
    #     "axis": axis.Regular(50, 0, 1, name="V1_deepTagMD_WvsQCD", label="V1 deepTagMD_WvsQCD"),
    #     "expr":  lambda events, objects: objects["V1"].deepTagMD_WvsQCD,
    # },
    # "V1_particleNetMD_QCD": {
    #     "axis": axis.Regular(50, 0, 1, name="V1_particleNetMD_QCD", label="V1 particleNetMD_QCD"),
    #     "expr":  lambda events, objects: objects["V1"].particleNetMD_QCD,
    # },
    # "V1_particleNet_TvsQCD": {
    #     "axis": axis.Regular(50, 0, 1, name="V1_particleNet_TvsQCD", label="V1 particleNet_TvsQCD"),
    #     "expr":  lambda events, objects: objects["V1"].particleNet_TvsQCD,
    # },
    # "V1_particleNet_WvsQCD": {
    #     "axis": axis.Regular(50, 0, 1, name="V1_particleNet_WvsQCD", label="V1 particleNet_WvsQCD"),
    #     "expr":  lambda events, objects: objects["V1"].particleNet_WvsQCD,
    # },
    
    # "vbsj1_area": {
    #     "axis": axis.Regular(50, 0.3, 0.7, name="vbsj1_area", label="vbsj2 area"),
    #     "expr":  lambda events, objects: objects["vbsj1"].area,
    # },
    # "vbsj1_nConstituents": {
    #     "axis": axis.Regular(30, 0, 60, name="vbsj1_nConstituents", label="vbsj2 nConstituents"),
    #     "expr":  lambda events, objects: objects["vbsj1"].nConstituents,
    # },
    # "vbsj2_area": {
    #     "axis": axis.Regular(50, 0.3, 0.7, name="vbsj2_area", label="vbsj2 area"),
    #     "expr":  lambda events, objects: objects["vbsj2"].area,
    # },
    # "vbsj2_nConstituents": {
    #     "axis": axis.Regular(30, 0, 60, name="vbsj2_nConstituents", label="vbsj2 nConstituents"),
    #     "expr":  lambda events, objects: objects["vbsj2"].nConstituents,
    # },
    # 'dphi_diff': {
    #     "axis": axis.Regular(50, 0, 3, name="dphi_diff", label="max - min of dphi(MET,H,V1)"),
    #     "expr": lambda events, objects: (
    #         ak.max(
    #             ak.concatenate( 
    #                 [
    #                     ak.singletons(deltaPhi(objects["Higgs"], objects["MET"])),
    #                     ak.singletons(deltaPhi(objects["V1"], objects["MET"])),
    #                     ak.singletons(deltaPhi(objects["Higgs"], objects["V1"])),
    #                 ],
    #                 axis=1,
    #             ),
    #             axis=1,
    #         )
    #         - ak.min(
    #             ak.concatenate(
    #                 [
    #                     ak.singletons(deltaPhi(objects["Higgs"], objects["MET"])),
    #                     ak.singletons(deltaPhi(objects["V1"], objects["MET"])),
    #                     ak.singletons(deltaPhi(objects["Higgs"], objects["V1"])),
    #                 ],
    #                 axis=1,
    #             ),
    #             axis=1,
    #         )
    #     ),
    # },
    # #to remove
    
    # "Met_sumEt": {
    #     "axis": axis.Regular(50, 0, 6000, name="Met_sumEt", label="Met_sumEt"),
    #     "expr":  lambda events, objects: objects["MET"].sumEt,
    # },
    # "Met_sumPtUnclustered": {
    #     "axis": axis.Regular(50, 0, 6000, name="Met_sumPtUnclustered", label="Met_sumPtUnclustered"),
    #     "expr":  lambda events, objects: objects["MET"].sumPtUnclustered,
    # },
    # "bosonMET_HT": {
    #     "axis": axis.Regular(50, 0, 3000, name="bosonMET_HT", label="bosonMET_HT"),
    #     "expr":  lambda events, objects: events.bosonMET_HT,
    # },
    # "ht_ak4": {
    #     "axis": axis.Regular(50, 0, 3000, name="ht_ak4", label="ht_ak4"),
    #     "expr":  lambda events, objects: events.ht_ak4,
    # },
    # "ht_ak8": {
    #     "axis": axis.Regular(50, 0, 3000, name="ht_ak8", label="ht_ak8"),
    #     "expr":  lambda events, objects: events.ht_ak8,
    # },
    # "ht_goodAK4Jets": {
    #     "axis": axis.Regular(50, 0, 3000, name="ht_goodAK4Jets", label="ht_goodAK4Jets"),
    #     "expr":  lambda events, objects: events.ht_goodAK4Jets,
    # },
    # "ht_goodAK8Jets": {
    #     "axis": axis.Regular(50, 0, 3000, name="ht_goodAK8Jets", label="ht_goodAK8Jets"),
    #     "expr":  lambda events, objects: events.ht_goodAK8Jets,
    # },
    # "nCentralAK4": {
    #     "axis": axis.Regular(25, 0, 25, name="nCentralAK4", label="nCentralAK4"),
    #     "expr":  lambda events, objects: events.nCentralAK4,
    # },
    # "nGoodVBS": {
    #     "axis": axis.Regular(25, 0, 25, name="nGoodVBS", label="nGoodVBS"),
    #     "expr":  lambda events, objects: events.nGoodVBS,
    # },
    # "sum_AK4_pt": {
    #     "axis": axis.Regular(50, 0, 3000, name="sum_AK4_pt", label="sum_AK4_pt"),
    #     "expr":  lambda events, objects: events.sum_AK4_pt,
    # },
    # "sum_AK8_pt": {
    #     "axis": axis.Regular(50, 0, 3000, name="sum_AK8_pt", label="sum_AK8_pt"),
    #     "expr":  lambda events, objects: events.sum_AK8_pt,
    # },
    # "vbsj1_dRH": {
    #     "axis": axis.Regular(50, 0, 7, name="vbsj1_dRH", label="vbsj1_dRH"),
    #     "expr":  lambda events, objects: events.vbsj1_dRH,
    # },
    # "vbsj1_dRV1": {
    #     "axis": axis.Regular(50, 0, 7, name="vbsj1_dRV1", label="vbsj1_dRV1"),
    #     "expr":  lambda events, objects: events.vbsj1_dRV1,
    # },
    # "vbsj2_dRH": {
    #     "axis": axis.Regular(50, 0, 7, name="vbsj2_dRH", label="vbsj2_dRH"),
    #     "expr":  lambda events, objects: events.vbsj2_dRH,
    # },
    # "vbsj2_dRV1": {
    #     "axis": axis.Regular(50, 0, 7, name="vbsj2_dRV1", label="vbsj2_dRV1"),
    #     "expr":  lambda events, objects: events.vbsj2_dRV1,
    # },
    # "HVMET_pt": {
    #     "axis": axis.Regular(50, 0, 3000, name="HVMET_pt", label="HVMET_pt"),
    #     "expr":  lambda events, objects: events.HVMET_pt,
    # },
    # "vbsj_dphi": {
    #     "axis": axis.Regular(50, 0, 3.5, name="vbsj_dphi", label="vbsj_dphi"),
    #     "expr":  lambda events, objects: deltaPhi(objects['vbsj1'],objects["vbsj2"])
    # },

    # "electron_goodAK4_min_dR": {
    #     "axis": axis.Regular(50, 0, 5, name="electron_goodAK4_min_dR", label="electron_goodAK4_min_dR"),
    #     "expr":  lambda events, objects: events.electron_goodAK4_min_dR,
    # },
    # "muon_goodAK4_min_dR": {
    #     "axis": axis.Regular(50, 0, 5, name="muon_goodAK4_min_dR", label="muon_AK4_min_dR"),
    #     "expr":  lambda events, objects: events.muon_goodAK4_min_dR,
    # },
    # "tau_goodAK4_min_dR": {
    #     "axis": axis.Regular(50, 0, 5, name="tau_goodAK4_min_dR", label="tau_AK4_min_dR"),
    #     "expr":  lambda events, objects: events.tau_goodAK4_min_dR,
    # },
    # "tau_H_dR": {
    #     "axis": axis.Regular(50, 0, 5, name="tau_H_dR", label="tau_H_dR"),
    #     "expr":  lambda events, objects: events.tau_H_dR,
    # },
    # "tau_V1_dR": {
    #     "axis": axis.Regular(50, 0, 5, name="tau_V1_dR", label="tau_V1_dR"),
    #     "expr":  lambda events, objects: events.tau_V1_dR,
    # },
}

def get_min(array):
    arr_sort = ak.pad_none(ak.sort(array, axis = 1, ascending=True),1, axis=1)
    return arr_sort[:,0]
    
# seems delta_r is the only available function for coffea?  
def deltaR(v1, v2): 
    return v1.delta_r(v2)

def min_dR(v1, v2): 
    dR = []
    for i in range(ak.num(v1, axis=0)):
        if ak.num(v1, axis=1)== 0 or ak.num(v2, axis=1) == 0:
            dR.append(None)
        else:
            k1, k2 = ak.unzip(ak.cartesian([v1, v2]))
            dR.append(get_min(k1.deltaR(k2)))
    return ak.flatten(dR)


def deltaPhi(v1, v2):
    phi1 = v1.phi
    phi2 = v2.phi
    abs_diff = np.abs(phi1 - phi2)
    dphi = ak.where(
        abs_diff < np.pi,
        abs_diff,
        2 * np.pi - abs_diff
    ) #compare element-wise
    return dphi

def deltaPhi_1d(phi1, phi2):
    abs_diff = np.abs(phi1 - phi2)
    dphi = ak.where(
        abs_diff < np.pi,
        abs_diff,
        2 * np.pi - abs_diff
    ) #compare element-wise
    return  dphi

def deltaEta(v1,v2):
    eta1 = v1.eta
    eta2 = v2.eta
    return np.abs(eta1-eta2)