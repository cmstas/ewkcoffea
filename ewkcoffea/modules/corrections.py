import numpy as np
import pickle
import gzip
import awkward as ak

from coffea import lookup_tools

from ewkcoffea.modules.paths import ewkcoffea_path
from topcoffea.modules.paths import topcoffea_path


extLepSF = lookup_tools.extractor()

# TODO: Remove all reco SFs, they are folded into Kirill's tight SFs
# Muon: reco
extLepSF.add_weight_sets(["MuonRecoSF_2018 NUM_TrackerMuons_DEN_genTracks/abseta_pt_value %s" % topcoffea_path('data/leptonSF/muon/Efficiency_muon_generalTracks_Run2018_UL_trackerMuon.json')])
extLepSF.add_weight_sets(["MuonRecoSF_2018_er NUM_TrackerMuons_DEN_genTracks/abseta_pt_error %s" % topcoffea_path('data/leptonSF/muon/Efficiency_muon_generalTracks_Run2018_UL_trackerMuon.json')])
extLepSF.add_weight_sets(["MuonRecoSF_2017 NUM_TrackerMuons_DEN_genTracks/abseta_pt_value %s" % topcoffea_path('data/leptonSF/muon/Efficiency_muon_generalTracks_Run2017_UL_trackerMuon.json')])
extLepSF.add_weight_sets(["MuonRecoSF_2017_er NUM_TrackerMuons_DEN_genTracks/abseta_pt_error %s" % topcoffea_path('data/leptonSF/muon/Efficiency_muon_generalTracks_Run2017_UL_trackerMuon.json')])
extLepSF.add_weight_sets(["MuonRecoSF_2016 NUM_TrackerMuons_DEN_genTracks/abseta_pt_value %s" % topcoffea_path('data/leptonSF/muon/Efficiency_muon_generalTracks_Run2016postVFP_UL_trackerMuon.json')])
extLepSF.add_weight_sets(["MuonRecoSF_2016_er NUM_TrackerMuons_DEN_genTracks/abseta_pt_error %s" % topcoffea_path('data/leptonSF/muon/Efficiency_muon_generalTracks_Run2016postVFP_UL_trackerMuon.json')])
extLepSF.add_weight_sets(["MuonRecoSF_2016APV NUM_TrackerMuons_DEN_genTracks/abseta_pt_value %s" % topcoffea_path('data/leptonSF/muon/Efficiency_muon_generalTracks_Run2016preVFP_UL_trackerMuon.json')])
extLepSF.add_weight_sets(["MuonRecoSF_2016APV_er NUM_TrackerMuons_DEN_genTracks/abseta_pt_error %s" % topcoffea_path('data/leptonSF/muon/Efficiency_muon_generalTracks_Run2016preVFP_UL_trackerMuon.json')])

# TODO: Remove all reco SFs, they are folded into Kirill's tight SFs
# Elec: reco
extLepSF.add_weight_sets(["ElecRecoSFAb_2018 EGamma_SF2D %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2018_ptAbove20_EGM2D.root')])
extLepSF.add_weight_sets(["ElecRecoSFAb_2018_er EGamma_SF2D_error %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2018_ptAbove20_EGM2D.root')])
extLepSF.add_weight_sets(["ElecRecoSFBe_2018 EGamma_SF2D %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2018_ptBelow20_EGM2D.root')])
extLepSF.add_weight_sets(["ElecRecoSFBe_2018_er EGamma_SF2D_error %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2018_ptBelow20_EGM2D.root')])
extLepSF.add_weight_sets(["ElecRecoSFAb_2017 EGamma_SF2D %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2017_ptAbove20_EGM2D.root')])
extLepSF.add_weight_sets(["ElecRecoSFAb_2017_er EGamma_SF2D_error %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2017_ptAbove20_EGM2D.root')])
extLepSF.add_weight_sets(["ElecRecoSFBe_2017 EGamma_SF2D %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2017_ptBelow20_EGM2D.root')])
extLepSF.add_weight_sets(["ElecRecoSFBe_2017_er EGamma_SF2D_error %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2017_ptBelow20_EGM2D.root')])
extLepSF.add_weight_sets(["ElecRecoSFAb_2016 EGamma_SF2D %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2016_ptAbove20_EGM2D.root')])
extLepSF.add_weight_sets(["ElecRecoSFAb_2016_er EGamma_SF2D_error %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2016_ptAbove20_EGM2D.root')])
extLepSF.add_weight_sets(["ElecRecoSFBe_2016 EGamma_SF2D %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2016_ptBelow20_EGM2D.root')])
extLepSF.add_weight_sets(["ElecRecoSFBe_2016_er EGamma_SF2D_error %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2016_ptBelow20_EGM2D.root')])
extLepSF.add_weight_sets(["ElecRecoSFAb_2016APV EGamma_SF2D %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2016APV_ptAbove20_EGM2D.root')])
extLepSF.add_weight_sets(["ElecRecoSFAb_2016APV_er EGamma_SF2D_error %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2016APV_ptAbove20_EGM2D.root')])
extLepSF.add_weight_sets(["ElecRecoSFBe_2016APV EGamma_SF2D %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2016APV_ptBelow20_EGM2D.root')])
extLepSF.add_weight_sets(["ElecRecoSFBe_2016APV_er EGamma_SF2D_error %s" % topcoffea_path('data/leptonSF/elec/egammaEffi2016APV_ptBelow20_EGM2D.root')])

# Muon: tight (topmva)
extLepSF.add_weight_sets(["MuonTightSF_2016 NUM_LeptonMvaTight_DEN_TrackerMuons/abseta_pt_value %s" % ewkcoffea_path('data/topmva_lep_sf/NUM_LeptonMvaTight_DEN_TrackerMuons_abseta_pt_UL16.json')])
extLepSF.add_weight_sets(["MuonTightSF_2016APV NUM_LeptonMvaTight_DEN_TrackerMuons/abseta_pt_value %s" % ewkcoffea_path('data/topmva_lep_sf/NUM_LeptonMvaTight_DEN_TrackerMuons_abseta_pt_UL16APV.json')])
extLepSF.add_weight_sets(["MuonTightSF_2017 NUM_LeptonMvaTight_DEN_TrackerMuons/abseta_pt_value %s" % ewkcoffea_path('data/topmva_lep_sf/NUM_LeptonMvaTight_DEN_TrackerMuons_abseta_pt_UL17.json')])
extLepSF.add_weight_sets(["MuonTightSF_2018 NUM_LeptonMvaTight_DEN_TrackerMuons/abseta_pt_value %s" % ewkcoffea_path('data/topmva_lep_sf/NUM_LeptonMvaTight_DEN_TrackerMuons_abseta_pt_UL18.json')])

# Electron: tight (topmva)
extLepSF.add_weight_sets(["EleTightSF_2016 EGamma_SF2D %s" % ewkcoffea_path('data/topmva_lep_sf/egammaEffi_txt_EGM2D_UL16.root')])
extLepSF.add_weight_sets(["EleTightSF_2016APV EGamma_SF2D %s" % ewkcoffea_path('data/topmva_lep_sf/egammaEffi_txt_EGM2D_UL16APV.root')])
extLepSF.add_weight_sets(["EleTightSF_2017 EGamma_SF2D %s" % ewkcoffea_path('data/topmva_lep_sf/egammaEffi_txt_EGM2D_UL17.root')])
extLepSF.add_weight_sets(["EleTightSF_2018 EGamma_SF2D %s" % ewkcoffea_path('data/topmva_lep_sf/egammaEffi_txt_EGM2D_UL18.root')])

extLepSF.finalize()
SFevaluator = extLepSF.make_evaluator()


def AttachMuonSF(muons, year):
    '''
      Description:
          Inserts 'sf_nom', 'sf_hi', and 'sf_lo' into the muons array passed to this function. These
          values correspond to the nominal, up, and down muon scalefactor values respectively.
    '''
    eta = np.abs(muons.eta)
    pt = muons.pt
    if year not in ['2016','2016APV','2017','2018']: raise Exception(f"Error: Unknown year \"{year}\".")
    reco_sf  = np.where(pt < 20,SFevaluator['MuonRecoSF_{year}'.format(year=year)](eta,pt),1) # sf=1 when pt>20 becuase there is no reco SF available TODO: Remove all reco SFs, they are folded into Kirill's tight SFs
    reco_err = np.where(pt < 20,SFevaluator['MuonRecoSF_{year}_er'.format(year=year)](eta,pt),0) # sf error =0 when pt>20 becuase there is no reco SF available TODO: Remove all reco SFs, they are folded into Kirill's tight SFs

    tight_sf  = SFevaluator[f'MuonTightSF_{year}'](eta,pt)
    #tight_err = SFevaluator[f'MuonTightSF_{year}_er'](eta,pt)

    muons['sf_nom_3l_muon'] = tight_sf
    muons['sf_hi_3l_muon']  = (reco_sf + reco_err) # * (tight_sf + tight_err) # TODO: Remove all reco SFs, they are folded into Kirill's tight SFs
    muons['sf_lo_3l_muon']  = (reco_sf - reco_err) # * (tight_sf - tight_err) # TODO: Remove all reco SFs, they are folded into Kirill's tight SFs
    muons['sf_nom_3l_elec'] = ak.ones_like(reco_sf)
    muons['sf_hi_3l_elec']  = ak.ones_like(reco_sf)
    muons['sf_lo_3l_elec']  = ak.ones_like(reco_sf)

def AttachElectronSF(electrons, year):
    '''
      Description:
          Inserts 'sf_nom', 'sf_hi', and 'sf_lo' into the electrons array passed to this function. These
          values correspond to the nominal, up, and down electron scalefactor values respectively.
    '''
    eta = electrons.eta
    pt = electrons.pt

    if year not in ['2016','2016APV','2017','2018']:
        raise Exception(f"Error: Unknown year \"{year}\".")

    reco_sf  = np.where(
        pt < 20,
        SFevaluator['ElecRecoSFBe_{year}'.format(year=year)](eta,pt),
        SFevaluator['ElecRecoSFAb_{year}'.format(year=year)](eta,pt)
    )
    reco_err = np.where(
        pt < 20,
        SFevaluator['ElecRecoSFBe_{year}_er'.format(year=year)](eta,pt),
        SFevaluator['ElecRecoSFAb_{year}_er'.format(year=year)](eta,pt)
    )

    tight_sf  = SFevaluator[f'EleTightSF_{year}'](eta,pt)
    #tight_err = SFevaluator[f'EleTightSF_{year}_er'](eta,pt)

    electrons['sf_nom_3l_elec'] = tight_sf
    electrons['sf_hi_3l_elec']  = (reco_sf + reco_err) # * (tight_sf + tight_err) # TODO: Remove all reco SFs, they are folded into Kirill's tight SFs
    electrons['sf_lo_3l_elec']  = (reco_sf - reco_err) # * (tight_sf + tight_err) # TODO: Remove all reco SFs, they are folded into Kirill's tight SFs
    electrons['sf_nom_3l_muon'] = ak.ones_like(reco_sf)
    electrons['sf_hi_3l_muon']  = ak.ones_like(reco_sf)
    electrons['sf_lo_3l_muon']  = ak.ones_like(reco_sf)


# Evaluate the btag eff
def btag_eff_eval(jets,wp,year):

    # Get the right process name for the given year and read in the histo
    pname_base = "TTZToLLNuNu_M_10"
    if year == "2016APV":
        pname = f"UL16APV_{pname_base}"
    elif year == "2016":
        pname = f"UL16_{pname_base}"
    elif year == "2017":
        pname = f"UL17_{pname_base}"
    elif year == "2018":
        pname = f"UL18_{pname_base}"
    else:
        raise Exception(f"Not a known year: {year}")

    pkl_file_path = ewkcoffea_path("data/btag_eff/btag_eff_ttZ_srpresel.pkl.gz")
    histo = pickle.load(gzip.open(pkl_file_path))["ptabseta"]
    histo_proc = histo[{"process":pname}]

    # Make sure wp is known
    if (wp != "L") and (wp != "M"):
        raise Exception(f"Not a known WP: {wp}")

    # Create lookup object and evaluate eff
    h_eff = histo_proc[{"tag":wp}] / histo_proc[{"tag":"all"}]
    vals = h_eff.values(flow=True)[1:,1:-1,:-1] # Pt (drop underflow), eta (drop under and over flow), flav (drop overflow, there is not underflow)
    h_eff_lookup = lookup_tools.dense_lookup.dense_lookup(vals, [ax.edges for ax in h_eff.axes])
    eff = h_eff_lookup(jets.pt,abs(jets.eta),jets.hadronFlavour)

    return eff


