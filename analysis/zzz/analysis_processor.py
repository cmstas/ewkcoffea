#!/usr/bin/env python
#import sys
import coffea
import numpy as np
import awkward as ak
np.seterr(divide='ignore', invalid='ignore', over='ignore')
from coffea import processor
import hist
from hist import axis
from coffea.analysis_tools import PackedSelection
from coffea.nanoevents.methods import vector
from mt2 import mt2
#import ewkcoffea.modules.objects_wwz as os_ec
#import ewkcoffea.modules.selection_wwz as es_ec

from ewkcoffea.modules.paths import ewkcoffea_path as ewkcoffea_path

import torch
torch.set_num_threads(1)

import warnings
warnings.filterwarnings(
    "ignore",
    message="Missing cross-reference index",
    category=RuntimeWarning,
    module="coffea.nanoevents.schemas.nanoaod"
)

def to_vec(obj,with_name="PtEtaPhiMCollection"):
    return ak.zip({
        "pt": obj.pt,
        "eta": obj.eta,
        "phi": obj.phi,
        "mass": obj.mass,
    }, with_name=with_name)

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


class AnalysisProcessor(processor.ProcessorABC):

    def __init__(self, samples, wc_names_lst=[], hist_lst=None, do_systematics=False, skip_obj_systematics=False, skip_signal_regions=False, skip_control_regions=False, muonSyst='nominal', dtype=np.float32, siphon_bdt_data=False, rwgt_to_sm=False, ele_cutBased_val=None, mu_pfIsoId_val=None, siphon_out_name="bdt_output"):

        self._samples = samples
        self._wc_names_lst = wc_names_lst
        self._dtype = dtype

        # For ABCDnet evaluations
        self._model = None

        # Create the dense axes for the histograms
        self._dense_axes_dict = {
            "met"   : axis.Regular(180, 0, 750, name="met",  label="met"),
            "metphi": axis.Regular(180, -3.1416, 3.1416, name="metphi", label="met phi"),
            "scalarptsum_jet" : axis.Regular(180, 0, 2000, name="scalarptsum_jet", label="H_T small radius"),
            "scalarptsum_jetFwd" : axis.Regular(180, 0, 1000, name="scalarptsum_jetFwd", label="H_T forward"),
            "scalarptsum_jetCent" : axis.Regular(180, 0, 2000, name="scalarptsum_jetCent", label="H_T central"),
            "scalarptsum_lep" : axis.Regular(180, 0, 800, name="scalarptsum_lep", label="S_T"),
            "scalarptsum_lepmet" : axis.Regular(180, 0, 1500, name="scalarptsum_lepmet", label="S_T + metpt"),
            "scalarptsum_lepmetFJ0" : axis.Regular(180, 0, 3500, name="scalarptsum_lepmetFJ0", label="S_T + metpt + FJ0 pt"),
            "scalarptsum_lepmetFJ01" : axis.Regular(180, 0, 3500, name="scalarptsum_lepmetFJ01", label="S_T + metpt + FJ0 pt + FJ1 pt"),
            "scalarptsum_lepmetalljets" : axis.Regular(180, 0, 2500, name="scalarptsum_lepmetalljets", label="S_T + metpt + H_T all"),
            "scalarptsum_lepmetcentjets" : axis.Regular(180, 0, 2500, name="scalarptsum_lepmetcentjets", label="S_T + metpt + H_T cent"),
            "scalarptsum_lepmetfwdjets" : axis.Regular(180, 0, 1500, name="scalarptsum_lepmetfwdjets", label="S_T + metpt + H_T fwd"),

            "l0_pt"  : axis.Regular(180, 0, 500, name="l0_pt", label="l0 pt"),
            "l0_eta"  : axis.Regular(180, -3,3, name="l0_eta", label="l0 eta"),
            "l0_phi"  : axis.Regular(180, -3.1416, 3.1416, name="l0_phi", label="l0 phi"),
            "l1_pt"  : axis.Regular(180, 0, 400, name="l1_pt", label="l1 pt"),
            "l1_eta"  : axis.Regular(180, -3,3, name="l1_eta", label="l1 eta"),
            "l1_phi"  : axis.Regular(180, -3.1416, 3.1416, name="l1_phi", label="l1 phi"),
            "l2_pt"  : axis.Regular(180, 0, 300, name="l2_pt", label="l2 pt"),
            "l2_eta"  : axis.Regular(180, -3,3, name="l2_eta", label="l2 eta"),
            "l2_phi"  : axis.Regular(180, -3.1416, 3.1416, name="l2_phi", label="l1 phi"),

            "l0_iso"     : axis.Regular(180, 0,0.2, name="l0_iso", label="l0 pfRelIso03_all"),
            "l0_miniiso" : axis.Regular(180, 0,0.2, name="l0_miniiso", label="l0 miniPFRelIso_all"),
            "l1_iso"     : axis.Regular(180, 0,0.2, name="l1_iso", label="l1 pfRelIso03_all"),
            "l1_miniiso" : axis.Regular(180, 0,0.2, name="l1_miniiso", label="l1 miniPFRelIso_all"),
            "l2_iso"     : axis.Regular(180, 0,0.2, name="l2_iso", label="l2 pfRelIso03_all"),
            "l2_miniiso" : axis.Regular(180, 0,0.2, name="l2_miniiso", label="l2 miniPFRelIso_all"),

            "mass_l0l1"      : axis.Regular(180, 0,500, name="mass_l0l1", label="mll of leading two leptons"),
            "dr_l0l1"        : axis.Regular(180, 0, 6, name="dr_l0l1", label="dr between leading two leptons"),
            "pt_l0l1"        : axis.Regular(180, 0, 1000, name="pt_l0l1", label="pt of pair of leading two leptons"),
            "absdphi_l0l1"   : axis.Regular(180, 0, 3.1416, name="absdphi_l0l1", label="abs delta phi between leading two leptons"),
            "absdphi_lepmet" : axis.Regular(180, 0, 3.1416, name="absdphi_lepmet", label="abs delta phi between met and pair of leading leptons"),
            "absdphi_l0met"  : axis.Regular(180, 0, 3.1416, name="absdphi_l0met", label="abs delta phi between met and leading lepton"),
            "dr_lepmet"      : axis.Regular(180, 0, 6, name="dr_lepmet", label="dr between met and pair of leading leptons"),
            "absdphi_FJ0lepmet" : axis.Regular(180, 0, 3.1416, name="absdphi_FJ0lepmet", label="abs delta phi between FJ0 and (met + leptons)"),

            "mlb_min" : axis.Regular(180, 0, 300, name="mlb_min",  label="min mass(b+l)"),
            "mlb_max" : axis.Regular(180, 0, 1000, name="mlb_max",  label="max mass(b+l)"),

            "njets"   : axis.Regular(8, 0, 8, name="njets",   label="Jet multiplicity"),
            "nleps"   : axis.Regular(5, 0, 5, name="nleps",   label="Lep multiplicity"),
            "nbtagsl" : axis.Regular(4, 0, 4, name="nbtagsl", label="Loose btag multiplicity"),
            "nbtagsm" : axis.Regular(4, 0, 4, name="nbtagsm", label="Medium btag multiplicity"),
            "nbtagst" : axis.Regular(4, 0, 4, name="nbtagst", label="Tight btag multiplicity"),

            "njets_counts"   : axis.Regular(30, 0, 30, name="njets_counts",   label="Jet multiplicity counts (total)"),
            "nleps_counts"   : axis.Regular(30, 0, 30, name="nleps_counts",   label="Lep multiplicity counts (total)"),

            "nfatjets"   : axis.Regular(8, 0, 8, name="nfatjets",   label="Fat jet multiplicity"),
            "njets_forward"   : axis.Regular(8, 0, 8, name="njets_forward",   label="Jet multiplicity (forward)"),
            "njets_central"   : axis.Regular(8, 0, 8, name="njets_central",   label="Jet multiplicity (central)"),

            "n_ll_sfos"   : axis.Regular(5, 0, 5, name="n_ll_sfos",   label="Number of SF OS lepton pairs"),
            "abs_ch_sum_3l" : axis.Regular(4, 0, 4, name="abs_ch_sum_3l",   label="Abs sum of charges of the 3l"),

            "fj0_pt"  : axis.Regular(180, 0, 2000, name="fj0_pt", label="fj0 pt"),
            "fj0_mass"  : axis.Regular(180, 0, 250, name="fj0_mass", label="fj0 mass"),
            "fj0_msoftdrop"  : axis.Regular(180, 0, 250, name="fj0_msoftdrop", label="fj0 softdrop mass"),
            "fj0_mparticlenet"  : axis.Regular(180, 0, 250, name="fj0_mparticlenet", label="fj0 particleNet mass"),
            "fj0_eta" : axis.Regular(180, -5, 5, name="fj0_eta", label="fj0 eta"),
            "fj0_phi" : axis.Regular(180, -3.1416, 3.1416, name="fj0_phi", label="j0 phi"),

            "fj0_gptHvsQCD": axis.Regular(180, 0, 1, name="fj0_gptHvsQCD", label="fj0 gloparT H"),
            "fj0_gptWvsQCD": axis.Regular(180, 0, 1, name="fj0_gptWvsQCD", label="fj0 gloparT W"),
            "fj0_gptZvsQCD": axis.Regular(180, 0, 1, name="fj0_gptZvsQCD", label="fj0 gloparT Z"),
            "fj0_gptVvsQCD": axis.Regular(180, 0, 1, name="fj0_gptVvsQCD", label="fj0 gloparT Z"),

            "fj0_pNetH4qvsQCD": axis.Regular(180, 0, 1, name="fj0_pNetH4qvsQCD", label="fj0 pNet H4qvsQCD"),
            "fj0_pNetHbbvsQCD": axis.Regular(180, 0, 1, name="fj0_pNetHbbvsQCD", label="fj0 pNet HbbvsQCD"),
            "fj0_pNetHccvsQCD": axis.Regular(180, 0, 1, name="fj0_pNetHccvsQCD", label="fj0 pNet HccvsQCD"),
            "fj0_pNetQCD"     : axis.Regular(180, 0, 1, name="fj0_pNetQCD",    label="fj0 pNet QCD"),
            "fj0_pNetTvsQCD"  : axis.Regular(180, 0, 1, name="fj0_pNetTvsQCD", label="fj0 pNet TvsQCD"),
            "fj0_pNetWvsQCD"  : axis.Regular(180, 0, 1, name="fj0_pNetWvsQCD", label="fj0 pNet WvsQCD"),
            "fj0_pNetZvsQCD"  : axis.Regular(180, 0, 1, name="fj0_pNetZvsQCD", label="fj0 pNet ZvsQCD"),
            "fj0_gpt_Hfrac" : axis.Regular(180, 0, 1, name="fj0_gpt_Hfrac",   label="H score frac (gptH / (gptH + gptW + gptZ))"),
            "fj0_gpt_Wfrac" : axis.Regular(180, 0, 1, name="fj0_gpt_Wfrac",   label="W score frac (gptW / (gptH + gptW + gptZ))"),
            "fj0_gpt_Zfrac" : axis.Regular(180, 0, 1, name="fj0_gpt_Zfrac",   label="Z score frac (gptZ / (gptH + gptW + gptZ))"),
            "fj0_gpt_Hsf" : axis.Regular(180, 0, 1, name="fj0_gpt_Hsf",   label="H softmax score (exp(gptH) / (exp(gptH) + exp(gptW) + exp(gptZ)))"),
            "fj0_gpt_Wsf" : axis.Regular(180, 0, 1, name="fj0_gpt_Wsf",   label="W softmax score (exp(gptW) / (exp(gptH) + exp(gptW) + exp(gptZ)))"),
            "fj0_gpt_Zsf" : axis.Regular(180, 0, 1, name="fj0_gpt_Zsf",   label="Z softmax score (exp(gptZ) / (exp(gptH) + exp(gptW) + exp(gptZ)))"),
            "fj0_gpt_mass2p" : axis.Regular(180, 0, 250, name="fj0_gpt_mass2p", label="gloParT massCorrX2p"),
            "fj0_gpt_mass"   : axis.Regular(180, 0, 250, name="fj0_gpt_mass",   label="gloParT massCorrGeneric"),

            "j0central_pt"  : axis.Regular(180, 0, 250, name="j0central_pt", label="j0 pt (central jets)"),
            "j0central_eta" : axis.Regular(180, 0, 5, name="j0central_eta", label="j0 abs eta (central jets)"),
            "j0central_phi" : axis.Regular(180, -3.1416, 3.1416, name="j0central_phi", label="j0 phi (central jets)"),

            "j0forward_pt"  : axis.Regular(180, 0, 150, name="j0forward_pt", label="j0 pt (forward jets)"),
            "j0forward_eta" : axis.Regular(180, 0, 5, name="j0forward_eta", label="j0 abs eta (forward jets)"),
            "j0forward_phi" : axis.Regular(180, -3.1416, 3.1416, name="j0forward_phi", label="j0 phi (forward jets)"),

            "j0_pt"  : axis.Regular(180, 0, 250, name="j0_pt", label="j0 pt (all regular jets)"),
            "j0_eta" : axis.Regular(180, 0, 5, name="j0_eta", label="j0 abs eta (all regular jets)"),
            "j0_phi" : axis.Regular(180, -3.1416, 3.1416, name="j0_phi", label="j0 phi (all regular jets)"),

            "dr_fj0l0" : axis.Regular(180, 0, 6, name="dr_fj0l0", label="dr between FJ and lepton"),
            "dr_j0fwdj1fwd" : axis.Regular(180, 0, 6, name="dr_j0fwdj1fwd", label="dr between leading two forward jets"),
            "dr_j0centj1cent" : axis.Regular(180, 0, 6, name="dr_j0centj1cent", label="dr between leading two central jets"),
            "dr_j0j1" : axis.Regular(180, 0, 6, name="dr_j0j1", label="dr between leading two jets"),

            "mass_jFJ_min" : axis.Regular(180, 0, 1500, name="mass_jFJ_min", label="Min mass of jet and FJ pair"),
            "mass_jFJ_max" : axis.Regular(180, 0, 4000, name="mass_jFJ_max", label="Max mass of jet and FJ pair"),
            "mass_lj_min" : axis.Regular(180, 0, 1000, name="mass_lj_min", label="Min mass of jet and lepton pair"),
            "mass_lj_max" : axis.Regular(180, 0, 4000, name="mass_lj_max", label="Max mass of jet and lepton pair"),

            "dr_lj_min" : axis.Regular(180, 0, 6, name="dr_lj_min", label="Min dr between a jet and lepton"),
            "dr_lj_max" : axis.Regular(180, 0, 6, name="dr_lj_max", label="Max dr between a jet and lepton"),

            "absdphi_j0fwdj1fwd"   : axis.Regular(180, 0, 3.1416, name="absdphi_j0fwdj1fwd", label="abs dphi between leading two forward jets"),
            "absdphi_j0centj1cent" : axis.Regular(180, 0, 3.1416, name="absdphi_j0centj1cent", label="abs dphi between leading two central jets"),
            "absdphi_j0j1"   : axis.Regular(180, 0, 3.1416, name="absdphi_j0j1", label="abs dphi between leading two jets"),

            "mass_j0centj1cent" : axis.Regular(180, 0, 250, name="mass_j0centj1cent", label="mjj of two leading (in pt) non-forward jets"),
            "mass_j0fwdj1fwd" : axis.Regular(180, 0, 2500, name="mass_j0fwdj1fwd", label="mjj of two leading (in pt) forward jets"),
            "mass_j0j1" : axis.Regular(180, 0, 1500, name="mass_j0j1", label="mjj of two leading (in pt) jets"),

            "mass_b0b1" : axis.Regular(180, 0, 250, name="mass_b0b1", label="mjj of two leading (pt) b jets"),

            "mass_bbscore0bbscore1" : axis.Regular(180, 0, 250, name="mass_bbscore0bbscore1", label="mjj of two leading (in score) loose b jets"),
            "mass_bmbscore0bmbscore1" : axis.Regular(180, 0, 250, name="mass_bmbscore0bmbscore1", label="mjj of two leading (in score) med b jets"),
            "bbscore0_bscore"  : axis.Regular(180, 0, 1, name="bbscore0_bscore", label="Btag score of b jet with highest btag score"),
            "bbscore1_bscore"  : axis.Regular(180, 0, 1, name="bbscore1_bscore", label="Btag score of b jet with second highest btag score"),

            "mass_jbscore0jbscore1" : axis.Regular(180, 0, 250, name="mass_jbscore0jbscore1", label="mjj of two leading (in score) jets"),
            "jbscore0_bscore"  : axis.Regular(180, 0, 1, name="jbscore0_bscore", label="Btag score of jet with highest btag score"),
            "jbscore1_bscore"  : axis.Regular(180, 0, 1, name="jbscore1_bscore", label="Btag score of jet with second highest btag score"),

            "mjj_max_cent" : axis.Regular(180, 0, 250, name="mjj_max_cent", label="Leading mjj of pair of non-forward jets"),
            "mjj_max_fwd" : axis.Regular(180, 0, 2500, name="mjj_max_fwd", label="Leading mjj of pair of forward jets"),
            "mjj_max_any" : axis.Regular(180, 0, 3000, name="mjj_max_any", label="Leading mjj of pair of any (central or fwd) jets"),
            "absdeta_max_fwd" : axis.Regular(180, 0, 10, name="absdeta_max_fwd", label="Largest abs(delta eta) of pair of forward jets"),
            "absdeta_max_any" : axis.Regular(180, 0, 10, name="absdeta_max_any", label="Largest abs(delta eta) of pair of any (central or fwd) jets"),

            "jj_pairs_atmindr_mjj" : axis.Regular(180, 0, 1000, name="jj_pairs_atmindr_mjj", label="jj_pairs_atmindr_mjj"),

            "mjjjall_nearest_t" : axis.Regular(180, 0, 700, name="mjjjall_nearest_t", label="mjjj closest to top, considering all jets"),
            "mjjjcnt_nearest_t" : axis.Regular(180, 0, 700, name="mjjjcnt_nearest_t", label="mjjj closest to top, considering central jets"),

            "mjjjany" : axis.Regular(180, 0, 3000, name="mjjjany", label="mjjj of leading (in pt) three central or fwd jets"),
            "mjjjcnt" : axis.Regular(180, 0, 3000, name="mjjjcnt", label="mjjj of leading (in pt) three central jets"),

            "mljjjany" : axis.Regular(180, 0, 4000, name="mljjjany", label="mljjj of leading (in pt) lep and three central or fwd jets"),

            "abs_pdgid_sum" : axis.Regular(20, 20, 40, name="abs_pdgid_sum", label="Sum of abs pdgId for the 3 lep"),

            #"ghiggs0_pt" : axis.Regular(180, 0, 1500, name="ghiggs0_pt", label="Gen higgs pt"),
            #"gvectorboson0_pt" : axis.Regular(180, 0, 1500, name="gvectorboson0_pt", label="Gen V pt"),

            "mll_min_afos" : axis.Regular(180, -2, 48, name="mll_min_afos",  label="min mll of all OS pairs"),
            "mll_z" : axis.Regular(180, 0, 150, name="mll_z",  label="mll of the pair of leptons closest to z"),
            "pt_z"  : axis.Regular(180, 0, 150, name="pt_z",   label="pt of the pair of leptons closest to z"),
            #"mt_wlep" : axis.Regular(180,-2,298, name="mt_wlep", label="MT of MET and W lep (ie, lep that is not the SFOS Z pair)"),
            "dr_wlepmet" : axis.Regular(180,0,6, name="dr_wlepmet", label="dr between MET and W lep (ie, lep that is not the SFOS Z pair)"),

            "pt_z1"  : axis.Regular(180, 0, 1000, name="pt_z1",   label="pt of Z1"),
            "pt_z2"  : axis.Regular(180, 0, 1000, name="pt_z2",   label="pt of Z2"),
            "absdphi_z1_met"   : axis.Regular(180, 0, 3.1416, name="absdphi_z1_met", label="abs delta phi between Z1 and met"),
            "absdphi_z2_met"   : axis.Regular(180, 0, 3.1416, name="absdphi_z2_met", label="abs delta phi between Z2 and met"),
            "absdphi_z1z2_met" : axis.Regular(180, 0, 3.1416, name="absdphi_z1z2_met", label="abs delta phi between (Z1+Z2) and met"),
            "absdphi_min_jmet" : axis.Regular(180, -2, 4, name="absdphi_min_jmet", label="min abs delta phi between met and any good jet"),
            "met_sig_proxy"    : axis.Regular(180, 0, 30, name="met_sig_proxy", label="met / sqrt(S_T + H_T)"),
            "mt2_z1"   : axis.Regular(180, 0, 360, name="mt2_z1",   label="MT2 of Z1 leptons and met"),
            "mt2_z2"   : axis.Regular(180, 0, 360, name="mt2_z2",   label="MT2 of Z2 leptons and met"),
            "mt2_zmin" : axis.Regular(180, 0, 360, name="mt2_zmin", label="min MT2 over Z1,Z2 leptons and met"),
            "mt2_zlead" : axis.Regular(180, 0, 360, name="mt2_zlead", label="MT2 of leading-pt Z leptons and met"),
            "mt2_zsub"  : axis.Regular(180, 0, 360, name="mt2_zsub",  label="MT2 of subleading-pt Z leptons and met"),


            "l0_truth"          : axis.Regular(36, -1, 34, name="l0_truth", label="l0 truth flag"),
            "l1_truth"          : axis.Regular(36, -1, 34, name="l1_truth", label="l1 truth flag"),
            "l2_truth"          : axis.Regular(36, -1, 34, name="l2_truth", label="l2 truth flag"),
            "l0_truth_real_pt"  : axis.Regular(180, 0, 500, name="l0_truth_real_pt", label="l0 truth real pt"),
            "l1_truth_real_pt"  : axis.Regular(180, 0, 500, name="l1_truth_real_pt", label="l1 truth real pt"),
            "l2_truth_real_pt"  : axis.Regular(180, 0, 500, name="l2_truth_real_pt", label="l2 truth real pt"),
            "l0_truth_fake_pt"  : axis.Regular(180, 0, 500, name="l0_truth_fake_pt", label="l0 truth fake pt"),
            "l1_truth_fake_pt"  : axis.Regular(180, 0, 500, name="l1_truth_fake_pt", label="l1 truth fake pt"),
            "l2_truth_fake_pt"  : axis.Regular(180, 0, 500, name="l2_truth_fake_pt", label="l2 truth fake pt"),
            "l0_truth_real_iso" : axis.Regular(180, 0, 0.4, name="l0_truth_real_iso", label="l0 truth real pfRelIso03_all"),
            "l1_truth_real_iso" : axis.Regular(180, 0, 0.4, name="l1_truth_real_iso", label="l1 truth real pfRelIso03_all"),
            "l2_truth_real_iso" : axis.Regular(180, 0, 0.4, name="l2_truth_real_iso", label="l2 truth real pfRelIso03_all"),
            "l0_truth_fake_iso" : axis.Regular(180, 0, 0.4, name="l0_truth_fake_iso", label="l0 truth fake pfRelIso03_all"),
            "l1_truth_fake_iso" : axis.Regular(180, 0, 0.4, name="l1_truth_fake_iso", label="l1 truth fake pfRelIso03_all"),
            "l2_truth_fake_iso" : axis.Regular(180, 0, 0.4, name="l2_truth_fake_iso", label="l2 truth fake pfRelIso03_all"),
            "nlep_truth_real"   : axis.Regular(5, 0, 5, name="nlep_truth_real",   label="Lep (truth, real) multiplicity"),
            "nlep_truth_fake"   : axis.Regular(5, 0, 5, name="nlep_truth_fake",   label="Lep (truth, fake) multiplicity"),
            "pt_z1z2met"        : axis.Regular(180, 0, 360, name="pt_z1z2met", label="pt of (Z1 + Z2 + met) system"),
            "mass_z1z2z3"       : axis.Regular(180, 0, 1000, name="mass_z1z2z3", label="mass of (Z1 + Z2 + Z3) system"),
            "mass_z3cand"       : axis.Regular(51, -4, 200, name="mass_z3cand", label="m(third pair) [GeV]"),
            "mass_h_cand"       : axis.Regular(101, -5, 500, name="mass_h_cand", label="m(4l) Higgs candidate [GeV]"),
            "pt_lep_unpaired"   : axis.Regular(51, -4, 200, name="pt_lep_unpaired", label="pt of unpaired lepton after Z1 and Z2 selection (only relevant for 5l)"),


        }

        # Add histograms to dictionary that will be passed on to dict_accumulator
        dout = {}
        for dense_axis_name in self._dense_axes_dict.keys():
            dout[dense_axis_name] = hist.Hist(
                hist.axis.StrCategory([], growth=True, name="process", label="process"),
                hist.axis.StrCategory([], growth=True, name="category", label="category"),
                hist.axis.StrCategory([], growth=True, name="systematic", label="systematic"),
                #hist.axis.StrCategory([], growth=True, name="year", label="year"),
                hist.axis.Integer(0,40, growth=True, name="lepflav", label="lepflav"),
                self._dense_axes_dict[dense_axis_name],
                storage="weight", # Keeps track of sumw2
                name="Counts",
            )

        # Set the accumulator
        self._accumulator = processor.dict_accumulator(dout)

        # Set the list of hists to fill
        if hist_lst is None:
            # If the hist list is none, assume we want to fill all hists
            self._hist_lst = list(self._accumulator.keys())
        else:
            # Otherwise, just fill the specified subset of hists
            for hist_to_include in hist_lst:
                if hist_to_include not in self._accumulator.keys():
                    raise Exception(f"Error: Cannot specify hist \"{hist_to_include}\", it is not defined in the processor.")
            self._hist_lst = hist_lst # Which hists to fill

        if ele_cutBased_val is not None: self._ele_cutBased_val = float(ele_cutBased_val)
        else: self._ele_cutBased_val = ele_cutBased_val

        if mu_pfIsoId_val is not None: self._mu_pfIsoId_val = float(mu_pfIsoId_val)
        else: self._mu_pfIsoId_val = mu_pfIsoId_val

        # Siphon the outputs (these outputs are the inputs for the ML training)
        self._siphon_output_path = f"histos/{siphon_out_name}.root"
        self._siphon_bdt_data = siphon_bdt_data
        #self._siphon_selection = ["2lOSSF_nFJ1_massHi_Zp5Hp5VBSp5"] # NOTE this is hard coded
        self._siphon_selection = ["3l_chsum1_mjj500"] # NOTE this is hard coded
        self._bdt_vars = []
        for varname in list(self._dense_axes_dict.keys()):
            self._bdt_vars.append(varname)
        self._bdt_vars.append("isRun3") # Not in hist dense axis list but we want it
        if self._siphon_bdt_data:
            bdt_out = {var: processor.column_accumulator(np.array([], dtype=np.float32)) for var in self._bdt_vars}
            bdt_out["weight"] = processor.column_accumulator(np.array([], dtype=np.float32))
            self._accumulator["bdt_data"] = processor.dict_accumulator(bdt_out)


    @property
    def accumulator(self):
        return self._accumulator

    @property
    def columns(self):
        return self._columns


    # Main function: run on a given chunk
    def process(self, events):

        histAxisName = events.shortname
        year         = events.year
        xsec         = events.xsec

        # Initialize objects
        ele     = events.electron
        mu      = events.muon
        jets    = events.jet
        met     = events.met
        fatjets = events.fatjet

        # Identify the kind of of chunk that this is (note this check assumes all events in this chunk are of the same kind, should be true)
        isSig  = events.kind[0]=="sig"
        isData = events.kind[0]=="data"

        # Put the relevant tagging scores in fatjets object (this should be in RDF in the future)
        fatjets["gptHvsQCD"] = fatjets.globalParT3_Xbb / (fatjets.globalParT3_Xbb + fatjets.globalParT3_QCD)
        fatjets["gptWvsQCD"] = (fatjets.globalParT3_Xqq/3 + fatjets.globalParT3_Xcs) / (fatjets.globalParT3_Xqq/3 + fatjets.globalParT3_Xcs + fatjets.globalParT3_QCD)
        fatjets["gptZvsQCD"] = (fatjets.globalParT3_Xbb + fatjets.globalParT3_Xcc + fatjets.globalParT3_Xqq) / (fatjets.globalParT3_Xbb + fatjets.globalParT3_Xcc + fatjets.globalParT3_Xqq + fatjets.globalParT3_QCD)
        fatjets["gptVvsQCD"] = ak.where(fatjets.gptZvsQCD>fatjets.gptWvsQCD,fatjets.gptZvsQCD,fatjets.gptWvsQCD) # Max of the W and Z score
        gpt_denom_sf  = np.exp(fatjets.gptHvsQCD) + np.exp(fatjets.gptWvsQCD) + np.exp(fatjets.gptZvsQCD)
        gpt_denom_tot = fatjets.gptHvsQCD + fatjets.gptWvsQCD + fatjets.gptZvsQCD
        fatjets["gpt_Hsf"] = np.exp(fatjets.gptHvsQCD) / gpt_denom_sf
        fatjets["gpt_Wsf"] = np.exp(fatjets.gptWvsQCD) / gpt_denom_sf
        fatjets["gpt_Zsf"] = np.exp(fatjets.gptZvsQCD) / gpt_denom_sf
        fatjets["gpt_Hfrac"] = fatjets.gptHvsQCD / gpt_denom_tot
        fatjets["gpt_Wfrac"] = fatjets.gptWvsQCD / gpt_denom_tot
        fatjets["gpt_Zfrac"] = fatjets.gptZvsQCD / gpt_denom_tot
        fatjets["gpt_mass2p"] = fatjets.globalParT3_massCorrX2p     * fatjets.mass * (1 - fatjets.rawFactor)
        fatjets["gpt_mass"]   = fatjets.globalParT3_massCorrGeneric * fatjets.mass * (1 - fatjets.rawFactor)

        # "4-vector" for met
        met4 = ak.zip(
            {
                "pt": met.pt,
                "eta": ak.zeros_like(met.pt),
                "phi": met.phi,
                "mass": ak.zeros_like(met.pt),
            },
            with_name="PtEtaPhiMLorentzVector",
            behavior=met.behavior,
        )


        # An array of lenght events that is just 1 for each event
        events["nom"] = ak.ones_like(met.pt)

        # A mask that is all True by construction (probably there's a better way to do this...)
        pass_through = ak.full_like(met.pt,True,dtype=bool)


        ################### Lepton selection ####################

        # RDF writes out loosest selection (veto for e, loose for m), which is what we veto on
        n_lep_veto = ak.num(ele) + ak.num(mu)

        # We will use loose e and medium m for analysis, be sure to convert the 0 and 1 in the array to T and F before using as a mask
        #ele = ele[ak.values_astype(ele.isLoose,bool)]
        #mu  = mu[ak.values_astype(mu.isMedium,bool)]

        # Get tight leptons for VVH selection, using mask from RDF
        l_vvh_t = ak.with_name(ak.concatenate([ele,mu],axis=1),'PtEtaPhiMCandidate')
        l_vvh_t = l_vvh_t[ak.argsort(l_vvh_t.pt, axis=-1,ascending=False)] # Sort by pt
        events["l_vvh_t"] = l_vvh_t

        l_vvh_t_padded = ak.pad_none(l_vvh_t, 4)
        l0 = l_vvh_t_padded[:,0]
        l1 = l_vvh_t_padded[:,1]
        l2 = l_vvh_t_padded[:,2]
        nleps = ak.num(l_vvh_t)
        abs_ch_sum_3l = abs(l0.charge + l1.charge + l2.charge)

        # Get leptons into other types, why do we need to do this :/
        l_vvh_t_vecsLZ = to_vec(l_vvh_t,"PtEtaPhiMLorentzVector")  # convert the whole collection first
        l_vvh_t_vecsLZ_padded = ak.pad_none(l_vvh_t_vecsLZ, 4)
        l0vLZ = l_vvh_t_vecsLZ_padded[:, 0]
        l1vLZ = l_vvh_t_vecsLZ_padded[:, 1]
        l_vvh_t_vecs = to_vec(l_vvh_t)  # convert the whole collection first
        l_vvh_t_vecs_padded = ak.pad_none(l_vvh_t_vecs, 4)
        l0v = l_vvh_t_vecs_padded[:, 0]
        l1v = l_vvh_t_vecs_padded[:, 1]



        ######### Normalization and weights ###########

        # Weights object
        # Note: add() will generally modify up/down weights, so if these are needed for any reason after this point, we should instead pass copies to add()
        weights_obj_base = coffea.analysis_tools.Weights(len(events),storeIndividual=True)
        #weights_obj_base.add("norm",events.baseweight)
        weights_obj_base.add("norm",events.weight)


        #################### Jets ####################

        # Jet selection
        #cleanedJets = os_ec.get_cleaned_collection(l_vvh_t,jets) # Clean against leps
        #cleanedJets = os_ec.get_cleaned_collection(fatjets,cleanedJets,drcut=0.8) # Clean against fat jets
        cleanedJets = jets
        goodJets = cleanedJets
        goodJets_central = cleanedJets[(abs(cleanedJets.eta) <= 2.4)]
        goodJets_forward = cleanedJets[(abs(cleanedJets.eta) > 2.4)]

        # Count jets
        njets = ak.num(goodJets)
        njets_forward = ak.num(goodJets_forward)
        njets_central = ak.num(goodJets_central)
        nfatjets = ak.num(fatjets)
        ht = ak.sum(goodJets.pt,axis=-1)

        goodJets_ptordered = goodJets[ak.argsort(goodJets.pt,axis=-1,ascending=False)]
        goodJets_ptordered_padded = ak.pad_none(goodJets_ptordered, 4)
        j0 = goodJets_ptordered_padded[:,0]
        j1 = goodJets_ptordered_padded[:,1]
        j2 = goodJets_ptordered_padded[:,2]
        j3 = goodJets_ptordered_padded[:,3]

        goodJets_forward_ptordered = goodJets_forward[ak.argsort(goodJets_forward.pt,axis=-1,ascending=False)]
        goodJets_forward_ptordered_padded = ak.pad_none(goodJets_forward_ptordered, 2)
        j0forward = goodJets_forward_ptordered_padded[:,0]
        j1forward = goodJets_forward_ptordered_padded[:,1]

        goodJets_central_ptordered = goodJets_central[ak.argsort(goodJets_central.pt,axis=-1,ascending=False)]
        goodJets_central_ptordered_padded = ak.pad_none(goodJets_central_ptordered, 4)
        j0cent = goodJets_central_ptordered_padded[:,0]
        j1cent = goodJets_central_ptordered_padded[:,1]
        j2cent = goodJets_central_ptordered_padded[:,2]
        j3cent = goodJets_central_ptordered_padded[:,3]

        goodfatjets_ptordered = fatjets[ak.argsort(fatjets.pt,axis=-1,ascending=False)]
        goodfatjets_ptordered_padded = ak.pad_none(goodfatjets_ptordered, 2)
        fj0 = goodfatjets_ptordered_padded[:,0]
        fj1 = goodfatjets_ptordered_padded[:,1]

        scalarptsum_jet = ak.sum(goodJets.pt,axis=-1)
        scalarptsum_jetCent = ak.sum(goodJets_central.pt,axis=-1)
        scalarptsum_jetFwd = ak.sum(goodJets_forward.pt,axis=-1)

        mjjjany  = ak.where(njets>=3, (j0+j1+j2).mass, -1)
        mjjjcnt  = ak.where(njets>=3, (j0cent+j1cent+j2cent).mass, -1)
        #mljjjany  = ak.where(njets>=3, (l0+j0+j1+j2).mass, -1)
        mljjjany  = ak.where(njets>=3, (l0v + j0+j1+j2).mass, -1)


        ### Bjets ###

        isBtagJetsLoose  = goodJets.isLooseBTag
        isBtagJetsMedium = goodJets.isMediumBTag
        isBtagJetsTight  = goodJets.isTightBTag
        isNotBtagJetsLoose = np.invert(isBtagJetsLoose)

        bjetsl = goodJets[isBtagJetsLoose]
        bjetsm = goodJets[isBtagJetsMedium]
        bjetst = goodJets[isBtagJetsTight]

        nbtagsl = ak.num(goodJets[isBtagJetsLoose])
        nbtagsm = ak.num(goodJets[isBtagJetsMedium])
        nbtagst = ak.num(goodJets[isBtagJetsTight])


        ######### Get variables we haven't already calculated #########

        # Replace with -1 when there are not a pair of jets
        mjj_tmp = (j0+j1).mass
        mass_j0centj1cent = ak.where(njets>1,mjj_tmp,-1)

        j0forward_eta = ak.where(njets_forward>0,j0forward.eta,-1)

        mass_j0fwdj1fwd = ak.where(njets_forward>1,(j0forward+j1forward).mass,-1)

        # Count lepton pairs
        ll_pairs = ak.combinations(l_vvh_t_padded, 2, fields=["l0", "l1"] )
        sfos_mask = ak.fill_none((ll_pairs.l0.pdgId == -ll_pairs.l1.pdgId),False)
        n_ll_sfos = ak.num(ll_pairs[sfos_mask])

        # Find the mjj of the pair of jets (central + fwd) that have the min delta R
        jj_pairs = ak.combinations(goodJets_ptordered_padded, 2, fields=["j0", "j1"] )
        jj_pairs_dr = jj_pairs.j0.delta_r(jj_pairs.j1)
        jj_pairs_idx_mindr = ak.argmin(jj_pairs_dr,axis=1,keepdims=True)
        jj_pairs_atmindr = jj_pairs[jj_pairs_idx_mindr]
        jj_pairs_atmindr_mjj = (jj_pairs_atmindr.j0 + jj_pairs_atmindr.j1).mass
        jj_pairs_atmindr_mjj = ak.flatten(ak.fill_none(jj_pairs_atmindr_mjj,-999)) # Replace Nones, flatten (so e.g. [[None],[x],[y]] -> [-999,x,y])

        # Find jet triplets clost to top mass
        jetall_triplets = ak.combinations(goodJets_ptordered_padded, 3, fields=["j0", "j1", "j2"] )
        jetcnt_triplets = ak.combinations(goodJets_central_ptordered_padded, 3, fields=["j0", "j1", "j2"] )
        jjjall_4vec = jetall_triplets.j0 + jetall_triplets.j1 + jetall_triplets.j2
        jjjcnt_4vec = jetcnt_triplets.j0 + jetcnt_triplets.j1 + jetcnt_triplets.j2
        tpeak_jall_idx = ak.argmin(abs(jjjall_4vec.mass - 173),keepdims=True,axis=1)
        tpeak_jcnt_idx = ak.argmin(abs(jjjcnt_4vec.mass - 173),keepdims=True,axis=1)
        mjjjall_nearest_t = ak.fill_none(ak.flatten(jjjall_4vec[tpeak_jall_idx].mass),0)
        mjjjcnt_nearest_t = ak.fill_none(ak.flatten(jjjcnt_4vec[tpeak_jcnt_idx].mass),0)

        mass_l0l1 = (l0+l1).mass
        dr_l0l1 = l0.delta_r(l1)
        scalarptsum_lep = ak.sum(l_vvh_t.pt,axis=-1)
        scalarptsum_lepmet = scalarptsum_lep + met.pt
        scalarptsum_lepmetFJ0 = scalarptsum_lep + met.pt + fj0.pt
        scalarptsum_lepmetFJ01 = scalarptsum_lep + met.pt + fj0.pt + fj1.pt
        scalarptsum_lepmetalljets = scalarptsum_lep + met.pt + scalarptsum_jet
        scalarptsum_lepmetcentjets = scalarptsum_lep + met.pt + scalarptsum_jetCent
        scalarptsum_lepmetfwdjets = scalarptsum_lep + met.pt + scalarptsum_jetFwd

        # lb pairs (i.e. always one lep, one bjet)
        lb_pairs = ak.cartesian({"l":to_vec(l_vvh_t),"j": bjetsm})
        mlb_min = ak.min((lb_pairs["l"] + lb_pairs["j"]).mass,axis=-1)
        mlb_max = ak.max((lb_pairs["l"] + lb_pairs["j"]).mass,axis=-1)

        # lj pairs (i.e. always one lep, one jet)
        lj_pairs     = ak.cartesian({"l":to_vec(l_vvh_t),"j": jets})
        dr_lj_min     = ak.min(lj_pairs["l"].delta_r(lj_pairs["j"]),axis=-1)
        dr_lj_max     = ak.max(lj_pairs["l"].delta_r(lj_pairs["j"]),axis=-1)
        mass_lj_min   = ak.min((lj_pairs["l"]+lj_pairs["j"]).mass,axis=-1)
        mass_lj_max   = ak.max((lj_pairs["l"]+lj_pairs["j"]).mass,axis=-1)

        # FJj pairs (i.e. always one FJ, one jet)
        FJj_pairs     = ak.cartesian({"fj":fatjets,"j": jets})
        mass_jFJ_min   = ak.min((FJj_pairs["fj"]+FJj_pairs["j"]).mass,axis=-1)
        mass_jFJ_max   = ak.max((FJj_pairs["fj"]+FJj_pairs["j"]).mass,axis=-1)

        bjets_ptordered = bjetsl[ak.argsort( bjetsl.pt,axis=-1,ascending=False)]
        bjets_ptordered_padded = ak.pad_none(bjets_ptordered, 2)
        b0 = bjets_ptordered_padded[:,0]
        b1 = bjets_ptordered_padded[:,1]
        mass_b0b1_tmp = (b0+b1).mass
        mass_b0b1 = ak.where(nbtagsl>1,mass_b0b1_tmp,-1)

        # Variables related to leading b jet score of b jets
        bjets_bscoreordered = bjetsl[ak.argsort(bjetsl.btagDeepFlavB,axis=-1,ascending=False)]
        bjets_bscoreordered_padded = ak.pad_none(bjets_bscoreordered, 2)
        bbscore0 = bjets_bscoreordered_padded[:,0]
        bbscore1 = bjets_bscoreordered_padded[:,1]
        mass_bbscore0bbscore1 = ak.fill_none((bbscore0+bbscore1).mass,0)
        bbscore0_bscore = ak.fill_none(bbscore0.btagDeepFlavB,0)
        bbscore1_bscore = ak.fill_none(bbscore1.btagDeepFlavB,0)

        # Variables related to leading b jet score of med b jets
        bjetsm_bscoreordered = bjetsm[ak.argsort(bjetsm.btagDeepFlavB,axis=-1,ascending=False)]
        bjetsm_bscoreordered_padded = ak.pad_none(bjetsm_bscoreordered, 2)
        bmbscore0 = bjetsm_bscoreordered_padded[:,0]
        bmbscore1 = bjetsm_bscoreordered_padded[:,1]
        mass_bmbscore0bmbscore1 = ak.fill_none((bmbscore0+bmbscore1).mass,0)

        # Variables related to leading b jet score of jets
        jets_bscoreordered = goodJets_ptordered_padded[ak.argsort(goodJets_ptordered_padded.btagDeepFlavB,axis=-1,ascending=False)]
        jbscore0 = jets_bscoreordered[:,0]
        jbscore1 = jets_bscoreordered[:,1]
        mass_jbscore0jbscore1 = ak.fill_none((jbscore0+jbscore1).mass,0)
        jbscore0_bscore = ak.fill_none(jbscore0.btagDeepFlavB,0)
        jbscore1_bscore = ak.fill_none(jbscore1.btagDeepFlavB,0)

        # Mjj max from any jets
        jjCentFwd_pairs = ak.combinations( goodJets_ptordered_padded, 2, fields=["j0", "j1"] )
        mjj_max_any     = ak.fill_none(ak.max((jjCentFwd_pairs.j0 + jjCentFwd_pairs.j1).mass,axis=-1),0)
        absdeta_max_any = ak.fill_none(ak.max(abs(jjCentFwd_pairs.j0.eta - jjCentFwd_pairs.j1.eta),axis=-1),0)

        # Mjj max from cent jets
        jjCent_pairs = ak.combinations(goodJets_ptordered_padded, 2, fields=["j0", "j1"] )
        mjj_max_cent = ak.fill_none(ak.max((jjCent_pairs.j0 + jjCent_pairs.j1).mass,axis=-1),0)

        # Mjj max from forward jets
        jjFwd_pairs = ak.combinations(goodJets_forward_ptordered_padded, 2, fields=["j0", "j1"] )
        mjj_max_fwd = ak.fill_none(ak.max((jjFwd_pairs.j0 + jjFwd_pairs.j1).mass,axis=-1),0)
        absdeta_max_fwd = ak.fill_none(ak.max(abs(jjFwd_pairs.j0.eta - jjFwd_pairs.j1.eta),axis=-1),0)

        fj0_pNetH4qvsQCD = fj0.particleNetWithMass_H4qvsQCD
        fj0_pNetHbbvsQCD = fj0.particleNetWithMass_HbbvsQCD
        fj0_pNetHccvsQCD = fj0.particleNetWithMass_HccvsQCD
        fj0_pNetQCD      = fj0.particleNetWithMass_QCD
        fj0_pNetTvsQCD   = fj0.particleNetWithMass_TvsQCD
        fj0_pNetWvsQCD   = fj0.particleNetWithMass_WvsQCD
        fj0_pNetZvsQCD   = fj0.particleNetWithMass_ZvsQCD
        fj0_mparticlenet = fj0.particleNetLegacy_mass

        # Compute pair invariant masses for low mass cuts
        ll_pairs_tmp = ak.combinations(l_vvh_t, 2, fields=["i0","i1"])
        ll_idx_pairs = ak.argcombinations(l_vvh_t, 2, fields=["i0", "i1"])
        os_pairs_mask   = ak.fill_none((ll_pairs_tmp.i0.pdgId*ll_pairs_tmp.i1.pdgId < 0),False) # Maks for opposite-sign pairs
        sfos_pairs_mask = ak.fill_none((ll_pairs_tmp.i0.pdgId == -ll_pairs_tmp.i1.pdgId),False) # Mask for same-flavor-opposite-sign pairs
        ll_absdphi_pairs = abs(ll_pairs_tmp.i0.delta_phi(ll_pairs_tmp.i1))
        ll_mass_pairs = (ll_pairs_tmp.i0+ll_pairs_tmp.i1).mass            # The mll for each ll pair
        absdphi_min_afas = ak.min(ll_absdphi_pairs,axis=-1)
        absdphi_min_afos = ak.min(ll_absdphi_pairs[os_pairs_mask],axis=-1)
        absdphi_min_sfos = ak.min(ll_absdphi_pairs[sfos_pairs_mask],axis=-1)
        mll_min_afas = ak.min(ll_mass_pairs,axis=-1)
        mll_min_afos = ak.min(ll_mass_pairs[os_pairs_mask],axis=-1)
        mll_min_sfos = ak.min(ll_mass_pairs[sfos_pairs_mask],axis=-1)

        # Get Z peak pairs
        ll_pairs_sfos = ll_pairs_tmp[sfos_pairs_mask]
        ll_idx_sfos   = ll_idx_pairs[sfos_pairs_mask]
        ll_pairs_4vec = ll_pairs_sfos.i0 + ll_pairs_sfos.i1
        zpeak_idx     = ak.argmin(abs(ll_pairs_4vec.mass - 91.1876), keepdims=True, axis=1)
        mll_z         = ak.fill_none(ak.flatten(ll_pairs_4vec[zpeak_idx].mass), 0)
        pt_z          = ak.fill_none(ak.flatten(ll_pairs_4vec[zpeak_idx].pt), 0)

        # For 3l, find the lepton that's not part of the Z pair
        sfos_mask = ak.any(sfos_pairs_mask, axis=1)
        z_idx0 = ak.flatten(ll_idx_sfos[zpeak_idx].i0, axis=1)
        z_idx1 = ak.flatten(ll_idx_sfos[zpeak_idx].i1, axis=1)
        all_idx = ak.local_index(l_vvh_t, axis=1)
        w_lep_mask = (all_idx != z_idx0) & (all_idx != z_idx1)
        l_w = ak.firsts(l_vvh_t[w_lep_mask])
        #mt_wlep = ak.where(sfos_mask,es_ec.get_mt(l_w, met4),-1)
        dr_wlepmet = ak.where(sfos_mask,l_w.delta_r(met4),-1)

        # NOTE Only defind for exactly 2 and 3 lep
        abs_pdgid_sum = ak.fill_none(ak.where(nleps==3,abs(l0.pdgId) + abs(l1.pdgId) + abs(l2.pdgId),abs(l0.pdgId) + abs(l1.pdgId)),0)

        ########################################################################
        ######### Find the Zs ##########

        MZ = 91.1876
        Z_WINDOW = 20.0

        leps = ak.with_field(l_vvh_t, ak.local_index(l_vvh_t, axis=1), "lep_idx")

        def best_sfos_pair(leps):
            """SFOS pair in `leps` closest to MZ, as a single 4-vector object.
            Z (and the indices) are None for events with no SFOS pair within Z_WINDOW."""
            pairs = ak.combinations(leps, 2, fields=["l0", "l1"])
            sfos_mask = ak.fill_none(pairs.l0.pdgId == -pairs.l1.pdgId, False)
            pairs = pairs[sfos_mask]

            dist = abs((pairs.l0 + pairs.l1).mass - MZ)
            best = ak.argmin(dist, axis=1, keepdims=True)
            in_window = ak.fill_none(ak.firsts(dist[best] < Z_WINDOW), False)

            l0, l1 = ak.firsts(pairs.l0[best]), ak.firsts(pairs.l1[best])
            Z    = ak.mask(l0 + l1, in_window)
            lep0 = ak.mask(l0, in_window)
            lep1 = ak.mask(l1, in_window)
            idx0 = ak.mask(l0.lep_idx, in_window)
            idx1 = ak.mask(l1.lep_idx, in_window)
            return Z, lep0, lep1, idx0, idx1

        # Z1: best SFOS pair among all leptons
        Z1, z1_l0, z1_l1, z1_i0, z1_i1 = best_sfos_pair(leps)

        # Z2: best SFOS pair among leptons not used by Z1
        leps_minus_z1 = leps[(leps.lep_idx != ak.fill_none(z1_i0, -1)) & (leps.lep_idx != ak.fill_none(z1_i1, -1))]
        Z2, z2_l0, z2_l1, z2_i0, z2_i1 = best_sfos_pair(leps_minus_z1)

        # Z3: best SFOS pair among leptons not used by Z1 or Z2
        leps_minus_z1z2 = leps_minus_z1[(leps_minus_z1.lep_idx != ak.fill_none(z2_i0, -1)) & (leps_minus_z1.lep_idx != ak.fill_none(z2_i1, -1))]
        Z3, z3_l0, z3_l1, z3_i0, z3_i1 = best_sfos_pair(leps_minus_z1z2)

        # The two leptons beyond Z1 and Z2 (the third pair). On Z in the 6l 3Z channel,
        # outside the window in the 6l 2Z channel. Exists only for 6l with both Z1 and Z2 found.
        has_pair3 = (ak.num(leps_minus_z1z2, axis=1) == 2)
        mass_z3cand = ak.fill_none(ak.mask(leps_minus_z1z2.sum(axis=1).mass, has_pair3), -1)

        # The single lepton left unpaired once Z1 and Z2 have claimed theirs (5l channel).
        # Prompt and hard when signal has lost a 6th lepton to acceptance; soft when it is
        # the nonprompt lepton that promotes ZZ->4l into the 5l category.
        has_lep_unpaired = (ak.num(leps_minus_z1z2, axis=1) == 1)
        pt_lep_unpaired = ak.fill_none(ak.mask(ak.firsts(leps_minus_z1z2).pt, has_lep_unpaired), -1)

        # Higgs candidate for the 6l regions: the third pair plus whichever of Z1/Z2 is
        # not the associated Z. Both of those are on Z, so the choice is genuinely
        # ambiguous -- take whichever combination lands closer to mH. Summing lepton
        # collections rather than adding Z four-vectors avoids the Candidate/LorentzVector
        # dispatch problem that Z1+Z2+Z3 hits.
        leps_minus_z2 = leps[(leps.lep_idx != ak.fill_none(z2_i0, -1)) & (leps.lep_idx != ak.fill_none(z2_i1, -1))]
        m_h_with_z1 = leps_minus_z2.sum(axis=1).mass    # Z1 + third pair
        m_h_with_z2 = leps_minus_z1.sum(axis=1).mass    # Z2 + third pair
        mass_h_cand = ak.fill_none(ak.mask( ak.where(abs(m_h_with_z1 - 125.0) < abs(m_h_with_z2 - 125.0), m_h_with_z1, m_h_with_z2), has_pair3), -1)

        # Number of valid Z candidates found (0-3)
        n_sfosz = (
            ak.values_astype(~ak.is_none(Z1.mass), "int32")
            + ak.values_astype(~ak.is_none(Z2.mass), "int32")
            + ak.values_astype(~ak.is_none(Z3.mass), "int32")
        )

        absdphi_min_jmet = ak.fill_none(ak.min(abs(goodJets.delta_phi(met4)), axis=-1), -1)
        met_sig_proxy = met.pt / np.sqrt(scalarptsum_lep + scalarptsum_jet)

        # MT2 of each Z's lepton pair against MET. Backgrounds where the pair is really
        # two W legs (WWZ, ttZ, ttbar) have an endpoint at mW; a genuine Z does not.
        def _safe_lep(lep):
            """get_mt2 can't take option-type input. Fill Z-undefined events with
            dummy values; only the 2Z categories look at these variables."""
            return ak.zip(
                {
                    "pt":    ak.fill_none(lep.pt,   10.0),
                    "eta":   ak.fill_none(lep.eta,   0.0),
                    "phi":   ak.fill_none(lep.phi,   0.0),
                    "mass":  ak.fill_none(lep.mass,  0.0),
                    "pdgId": ak.fill_none(lep.pdgId,  13),
                },
                with_name="PtEtaPhiMLorentzVector",
                behavior=vector.behavior,
            )

        mt2_z1   = get_mt2(_safe_lep(z1_l0), _safe_lep(z1_l1), met)
        mt2_z2   = get_mt2(_safe_lep(z2_l0), _safe_lep(z2_l1), met)
        mt2_zmin = np.minimum(mt2_z1, mt2_z2)

        z2_is_lead = ak.fill_none(Z2.pt > Z1.pt, False)
        mt2_zlead  = ak.where(z2_is_lead, mt2_z2, mt2_z1)
        mt2_zsub   = ak.where(z2_is_lead, mt2_z1, mt2_z2)




        ########################################################################

        # Put the variables we'll plot into a dictionary for easy access later
        dense_variables_dict = {

            "met" : met.pt,
            "metphi" : met.phi,
            "scalarptsum_lep" : scalarptsum_lep,
            "scalarptsum_jet" : scalarptsum_jet,
            "scalarptsum_jetCent" : scalarptsum_jetCent,
            "scalarptsum_jetFwd" : scalarptsum_jetFwd,
            "scalarptsum_lepmet" : scalarptsum_lepmet,
            "scalarptsum_lepmetFJ0" : scalarptsum_lepmetFJ0,
            "scalarptsum_lepmetFJ01" : scalarptsum_lepmetFJ01,
            "scalarptsum_lepmetalljets" : scalarptsum_lepmetalljets,
            "scalarptsum_lepmetcentjets" : scalarptsum_lepmetcentjets,
            "scalarptsum_lepmetfwdjets" : scalarptsum_lepmetfwdjets,
            "l0_pt"  : l0.pt,
            "l0_eta" : l0.eta,
            "l0_phi" : l0.phi,
            "l1_pt"  : l1.pt,
            "l1_eta" : l1.eta,
            "l1_phi" : l1.phi,
            "l2_pt"  : l2.pt,
            "l2_eta" : l2.eta,
            "l2_phi" : l2.phi,
            "mass_l0l1" : mass_l0l1,
            "dr_l0l1" : dr_l0l1,
            "pt_l0l1" : (l0+l1).pt,
            "absdphi_l0l1" : abs(l0.delta_phi(l1)),
            "absdphi_lepmet" : abs(met4.delta_phi(l0+l1)),
            "absdphi_l0met" : abs(met4.delta_phi(l0)),
            "absdphi_FJ0lepmet" : abs(fj0.delta_phi(met4+l0vLZ+l1vLZ)),
            "dr_lepmet" : met4.delta_r(l0+l1),
            "l0_iso"     : l0.pfRelIso03_all,
            "l0_miniiso" : l0.miniPFRelIso_all,
            "l1_iso"     : l1.pfRelIso03_all,
            "l1_miniiso" : l1.miniPFRelIso_all,
            "l2_iso"     : l2.pfRelIso03_all,
            "l2_miniiso" : l2.miniPFRelIso_all,

            "j0central_pt"  : j0cent.pt,
            "j0central_eta" : j0cent.eta,
            "j0central_phi" : j0cent.phi,

            "j0forward_pt"  : j0forward.pt,
            "j0forward_eta" : j0forward_eta,
            "j0forward_phi" : j0forward.phi,

            "j0_pt"  : j0.pt,
            "j0_eta" : j0.eta,
            "j0_phi" : j0.phi,

            "nleps" : nleps,
            "njets" : njets,

            "nleps_counts" : nleps,
            "njets_counts" : njets,

            "nbtagst" : nbtagst,
            "nbtagsm" : nbtagsm,
            "nbtagsl" : nbtagsl,

            "nfatjets" : nfatjets,
            "njets_forward" : njets_forward,
            "njets_central" : njets_central,
            "fj0_pt" : fj0.pt,
            "fj0_mass" : fj0.mass,
            "fj0_msoftdrop" : fj0.msoftdrop,
            "fj0_eta" : fj0.eta,
            "fj0_phi" : fj0.phi,

            "dr_fj0l0" : fj0.delta_r(l0),
            "dr_j0fwdj1fwd" : j0forward.delta_r(j1forward),
            "dr_j0centj1cent" : j0cent.delta_r(j1cent),
            "dr_j0j1" : j0.delta_r(j1),
            "absdphi_j0fwdj1fwd"   : abs(j0forward.delta_phi(j1forward)),
            "absdphi_j0centj1cent" : abs(j0cent.delta_phi(j1cent)),
            "absdphi_j0j1"   : abs(j0.delta_phi(j1)),

            "mass_j0centj1cent" : mass_j0centj1cent,
            "mass_j0fwdj1fwd" : mass_j0fwdj1fwd,
            "mass_j0j1" : (j0+j1).mass,

            "mass_b0b1" : mass_b0b1,

            "fj0_pNetH4qvsQCD" : fj0_pNetH4qvsQCD,
            "fj0_pNetHbbvsQCD" : fj0_pNetHbbvsQCD,
            "fj0_pNetHccvsQCD" : fj0_pNetHccvsQCD,
            "fj0_pNetQCD"      : fj0_pNetQCD,
            "fj0_pNetTvsQCD"   : fj0_pNetTvsQCD,
            "fj0_pNetWvsQCD"   : fj0_pNetWvsQCD,
            "fj0_pNetZvsQCD"   : fj0_pNetZvsQCD,
            "fj0_mparticlenet" : fj0_mparticlenet,
            "fj0_gptHvsQCD"    : fj0.gptHvsQCD,
            "fj0_gptWvsQCD"    : fj0.gptWvsQCD,
            "fj0_gptZvsQCD"    : fj0.gptZvsQCD,
            "fj0_gptVvsQCD"    : fj0.gptVvsQCD,
            "fj0_gpt_Hsf"      : fj0.gpt_Hsf,
            "fj0_gpt_Wsf"      : fj0.gpt_Wsf,
            "fj0_gpt_Zsf"      : fj0.gpt_Zsf,
            "fj0_gpt_mass2p"   : fj0.gpt_mass2p,
            "fj0_gpt_mass"     : fj0.gpt_mass,
            "fj0_gpt_Hfrac"    : fj0.gpt_Hfrac,
            "fj0_gpt_Wfrac"    : fj0.gpt_Wfrac,
            "fj0_gpt_Zfrac"    : fj0.gpt_Zfrac,

            "jj_pairs_atmindr_mjj" : jj_pairs_atmindr_mjj,

            "bbscore0_bscore" : bbscore0_bscore,
            "bbscore1_bscore" : bbscore1_bscore,
            "mass_bbscore0bbscore1" : mass_bbscore0bbscore1,
            "mass_bmbscore0bmbscore1" : mass_bmbscore0bmbscore1,

            "jbscore0_bscore" : jbscore0_bscore,
            "jbscore1_bscore" : jbscore1_bscore,
            "mass_jbscore0jbscore1" : mass_jbscore0jbscore1,

            "mjj_max_any" : mjj_max_any,
            "mjj_max_cent" : mjj_max_cent,
            "mjj_max_fwd" : mjj_max_fwd,

            "absdeta_max_fwd" : absdeta_max_fwd,
            "absdeta_max_any" : absdeta_max_any,

            "mjjjall_nearest_t": mjjjall_nearest_t,
            "mjjjcnt_nearest_t": mjjjcnt_nearest_t,

            "mjjjany" : mjjjany,
            "mjjjcnt" : mjjjcnt,

            "mljjjany" : mljjjany,

            "mlb_min" : mlb_min,
            "mlb_max" : mlb_max,

            "mass_jFJ_min" :  mass_jFJ_min,
            "mass_jFJ_max" :  mass_jFJ_max,
            "mass_lj_min" :  mass_lj_min,
            "mass_lj_max" :  mass_lj_max,

            "mass_jFJ_min" :  mass_jFJ_min,
            "mass_jFJ_max" :  mass_jFJ_max,

            "dr_lj_min" : dr_lj_min,
            "dr_lj_max" : dr_lj_max,

            #"ghiggs0_pt" : ghiggs0.pt,
            #"gvectorboson0_pt" : gvectorboson0.pt,

            "n_ll_sfos": n_ll_sfos,
            "abs_ch_sum_3l": abs_ch_sum_3l,
            "abs_pdgid_sum": abs_pdgid_sum,

            "mll_min_afos" : mll_min_afos,
            "mll_z" : mll_z,
            "pt_z"  : pt_z,
            #"mt_wlep":mt_wlep,
            "dr_wlepmet":dr_wlepmet,

            # We want to include this in the siponed output, but probably not make hists for it
            "isRun3" : events.isRun3,

            "pt_z1" : Z1.pt,
            "pt_z2" : Z2.pt,
            "absdphi_z1_met" : abs(met4.delta_phi(Z1)),
            "absdphi_z2_met" : abs(met4.delta_phi(Z2)),
            "absdphi_z1z2_met" : abs(met4.delta_phi(Z1+Z2)),
            "absdphi_min_jmet" : absdphi_min_jmet,
            "met_sig_proxy" : met_sig_proxy,
            "mt2_z1"   : mt2_z1,
            "mt2_z2"   : mt2_z2,
            "mt2_zmin" : mt2_zmin,
            "mt2_zlead" : mt2_zlead,
            "mt2_zsub"  : mt2_zsub,
            "pt_z1z2met" : (Z1 + Z2 + met4).pt,
            "mass_z1z2z3" : l_vvh_t.sum(axis=1).mass,
            "mass_z3cand" : mass_z3cand,
            "mass_h_cand" : mass_h_cand,
            "pt_lep_unpaired" : pt_lep_unpaired,

        }


        ### Lepton truth variables ###
        if not isData:

            lep_truth_real_mask = ((l_vvh_t.provenance == 23) | (l_vvh_t.provenance == 24) | (l_vvh_t.provenance == 33) | (l_vvh_t.provenance == 34))
            lep_truth_real = l_vvh_t_padded[lep_truth_real_mask]
            lep_truth_fake = l_vvh_t_padded[~lep_truth_real_mask]

            lep_truth_real = lep_truth_real[ak.argsort(lep_truth_real.pt,axis=-1,ascending=False)]
            lep_truth_fake = lep_truth_fake[ak.argsort(lep_truth_fake.pt,axis=-1,ascending=False)]

            lep_truth_real_padded = ak.pad_none(lep_truth_real, 3)
            lep_truth_fake_padded = ak.pad_none(lep_truth_fake, 3)

            l0_truth_real = lep_truth_real_padded[:,0]
            l1_truth_real = lep_truth_real_padded[:,1]
            l2_truth_real = lep_truth_real_padded[:,2]
            l0_truth_fake = lep_truth_fake_padded[:,0]
            l1_truth_fake = lep_truth_fake_padded[:,1]
            l2_truth_fake = lep_truth_fake_padded[:,2]

            nlep_truth_real = ak.num(lep_truth_real)
            nlep_truth_fake = ak.num(lep_truth_fake)

            dense_variables_dict["l0_truth"] = l0.provenance
            dense_variables_dict["l1_truth"] = l1.provenance
            dense_variables_dict["l2_truth"] = l2.provenance

            dense_variables_dict["l0_truth_real_pt"] = l0_truth_real.pt
            dense_variables_dict["l1_truth_real_pt"] = l1_truth_real.pt
            dense_variables_dict["l2_truth_real_pt"] = l2_truth_real.pt
            dense_variables_dict["l0_truth_fake_pt"] = l0_truth_fake.pt
            dense_variables_dict["l1_truth_fake_pt"] = l1_truth_fake.pt
            dense_variables_dict["l2_truth_fake_pt"] = l2_truth_fake.pt

            dense_variables_dict["l0_truth_real_iso"] = l0_truth_real.pfRelIso03_all
            dense_variables_dict["l1_truth_real_iso"] = l1_truth_real.pfRelIso03_all
            dense_variables_dict["l2_truth_real_iso"] = l2_truth_real.pfRelIso03_all
            dense_variables_dict["l0_truth_fake_iso"] = l0_truth_fake.pfRelIso03_all
            dense_variables_dict["l1_truth_fake_iso"] = l1_truth_fake.pfRelIso03_all
            dense_variables_dict["l2_truth_fake_iso"] = l2_truth_fake.pfRelIso03_all

            dense_variables_dict["nlep_truth_real"] = nlep_truth_real
            dense_variables_dict["nlep_truth_fake"] = nlep_truth_fake



        ######### Store boolean masks with PackedSelection ##########

        selections = PackedSelection(dtype='uint64')

        # Form some useful masks for SRs

        is_os = l0.pdgId*l1.pdgId<0
        is_sf = abs(l0.pdgId) == abs(l1.pdgId)

        low_mll_cut_3l = ak.where(abs_ch_sum_3l==1,mll_min_afos>12,pass_through)
        is_onZ = abs(mll_z-91.1876) < 10

        selections.add("all_events", pass_through)


        ### 6 lepton selections ###

        is_4l           = (nleps>=4)
        is_4l_minmll    = (nleps>=4) & (mll_min_sfos>12)

        selections.add("6l",       (nleps==6))
        selections.add("g6l",      (nleps>6))

        selections.add("4l",                    is_4l)
        selections.add("4l_minmll",             is_4l_minmll)
        selections.add("4l_minmll_2z",          is_4l_minmll & (n_sfosz>=2))
        selections.add("4l_minmll_2z_0b",       is_4l_minmll & (n_sfosz>=2) & (nbtagst==0))

        selections.add("4l_minmll_2z_0b_4lx_0fj_met90l",          is_4l_minmll & (n_sfosz>=2) & (nbtagst==0) & (nleps==4) & (nfatjets==0) & (met.pt<90))
        selections.add("4l_minmll_2z_2b_4lx",                     is_4l_minmll & (n_sfosz>=2) & (nbtagst>=2) & (nleps==4))

        selections.add("4l_minmll_2z_0b_4lx_0fj",                is_4l_minmll & (n_sfosz>=2) & (nbtagst==0) & (nleps==4) & (nfatjets==0))
        selections.add("4l_minmll_2z_0b_4lx_0fj_met90",          is_4l_minmll & (n_sfosz>=2) & (nbtagst==0) & (nleps==4) & (nfatjets==0) & (met.pt>90))
        selections.add("4l_minmll_2z_0b_4lx_0fj_met90_phimetzz", is_4l_minmll & (n_sfosz>=2) & (nbtagst==0) & (nleps==4) & (nfatjets==0) & (met.pt>90) & (abs(met4.delta_phi(Z1+Z2))>2))
        selections.add("4l_minmll_2z_0b_4lx_1fj",                is_4l_minmll & (n_sfosz>=2) & (nbtagst==0) & (nleps==4) & (nfatjets==1))
        selections.add("4l_minmll_2z_0b_4lx_1fj_gpt0p5",         is_4l_minmll & (n_sfosz>=2) & (nbtagst==0) & (nleps==4) & (nfatjets==1) & (fj0.gptZvsQCD>0.5))
        selections.add("4l_minmll_2z_0b_4lx_1fj_pn0p5",          is_4l_minmll & (n_sfosz>=2) & (nbtagst==0) & (nleps==4) & (nfatjets==1) & (fj0_pNetZvsQCD>0.5))

        selections.add("4l_minmll_2z_0b_5lx",                    is_4l_minmll & (n_sfosz>=2) & (nbtagst==0) & (nleps==5))

        selections.add("4l_minmll_2z_0b_6l",           is_4l_minmll & (n_sfosz>=2) & (nbtagst==0) & (nleps==6))
        selections.add("4l_minmll_2z_0b_6l_2z",        is_4l_minmll & (n_sfosz>=2) & (nbtagst==0) & (nleps==6) & (n_sfosz==2))
        selections.add("4l_minmll_2z_0b_6l_2z_mh150l", is_4l_minmll & (n_sfosz>=2) & (nbtagst==0) & (nleps==6) & (n_sfosz==2) & (mass_h_cand<150))
        selections.add("4l_minmll_2z_0b_6l_3z",        is_4l_minmll & (n_sfosz>=2) & (nbtagst==0) & (nleps==6) & (n_sfosz==3))


        # Keep track of the cats we want to actually fill
        cat_dict = {
            "lep_chan_lst" : [

                "all_events",
                "6l",
                "g6l",

                "4l",
                "4l_minmll",
                "4l_minmll_2z",
                "4l_minmll_2z_0b",

                "4l_minmll_2z_0b_4lx_0fj_met90l",
                "4l_minmll_2z_2b_4lx",

                "4l_minmll_2z_0b_4lx_0fj",
                "4l_minmll_2z_0b_4lx_0fj_met90",
                "4l_minmll_2z_0b_4lx_0fj_met90_phimetzz",
                "4l_minmll_2z_0b_4lx_1fj",
                "4l_minmll_2z_0b_4lx_1fj_gpt0p5",
                "4l_minmll_2z_0b_4lx_1fj_pn0p5",

                "4l_minmll_2z_0b_5lx",

                "4l_minmll_2z_0b_6l",
                "4l_minmll_2z_0b_6l_2z",
                "4l_minmll_2z_0b_6l_2z_mh150l",
                "4l_minmll_2z_0b_6l_3z",

            ]
        }


        ### Gen truth matched categories for signal ###
        if isSig:
            gen_h  = ak.zip({"pt": ak.ones_like(events.gen.h_eta), "eta": events.gen.h_eta,  "phi": events.gen.h_phi,  "mass": ak.ones_like(events.gen.h_eta)}, with_name="PtEtaPhiMCollection")
            gen_v1 = ak.zip({"pt": ak.ones_like(events.gen.v1_eta), "eta": events.gen.v1_eta, "phi": events.gen.v1_phi, "mass": ak.ones_like(events.gen.v1_eta)}, with_name="PtEtaPhiMCollection")
            gen_v2 = ak.zip({"pt": ak.ones_like(events.gen.v2_eta), "eta": events.gen.v2_eta, "phi": events.gen.v2_phi, "mass": ak.ones_like(events.gen.v2_eta)}, with_name="PtEtaPhiMCollection")
            dR_fj0_h  = fj0.delta_r(gen_h)
            dR_fj0_v1 = fj0.delta_r(gen_v1)
            dR_fj0_v2 = fj0.delta_r(gen_v2)
            dR_threshold = 0.8
            fj0_matchedH  = (dR_fj0_h < dR_threshold)  & (dR_fj0_h  < dR_fj0_v1) & (dR_fj0_h  < dR_fj0_v2)
            fj0_matchedV1 = (dR_fj0_v1 < dR_threshold) & (dR_fj0_v1 < dR_fj0_h)  & (dR_fj0_v1 < dR_fj0_v2)
            fj0_matchedV2 = (dR_fj0_v2 < dR_threshold) & (dR_fj0_v2 < dR_fj0_h)  & (dR_fj0_v2 < dR_fj0_v1)
            fj0_matchedV  = fj0_matchedV1 | fj0_matchedV2
            fj0_noMatch   = ~(dR_fj0_h < dR_threshold) & ~(dR_fj0_v1 < dR_threshold) & ~(dR_fj0_v2 < dR_threshold)
            selections.add("2lOSSF_1fjx_fj0matchH",  is_2l & is_os & is_sf & (nfatjets==1) & ak.fill_none(fj0_matchedH,  False))
            selections.add("2lOSSF_1fjx_fj0matchV1", is_2l & is_os & is_sf & (nfatjets==1) & ak.fill_none(fj0_matchedV1, False))
            selections.add("2lOSSF_1fjx_fj0matchV2", is_2l & is_os & is_sf & (nfatjets==1) & ak.fill_none(fj0_matchedV2, False))
            selections.add("2lOSSF_1fjx_fj0matchV",  is_2l & is_os & is_sf & (nfatjets==1) & ak.fill_none(fj0_matchedV, False))
            selections.add("2lOSSF_1fjx_fj0noMatch", is_2l & is_os & is_sf & (nfatjets==1) & ak.fill_none(fj0_noMatch, False))

            #cat_dict["lep_chan_lst"].append("2lOSSF_1fjx_fj0matchH")
            #cat_dict["lep_chan_lst"].append("2lOSSF_1fjx_fj0matchV1")
            #cat_dict["lep_chan_lst"].append("2lOSSF_1fjx_fj0matchV2")
            #cat_dict["lep_chan_lst"].append("2lOSSF_1fjx_fj0matchV")
            #cat_dict["lep_chan_lst"].append("2lOSSF_1fjx_fj0noMatch")
            #cat_dict["lep_chan_lst"].append("2lOSSF_1fjx_ejj3_fj0matchH")
            #cat_dict["lep_chan_lst"].append("2lOSSF_1fjx_ejj3_fj0matchV1")
            #cat_dict["lep_chan_lst"].append("2lOSSF_1fjx_ejj3_fj0matchV2")
            #cat_dict["lep_chan_lst"].append("2lOSSF_1fjx_ejj3_fj0matchV")
            #cat_dict["lep_chan_lst"].append("2lOSSF_1fjx_ejj3_fj0noMatch")




        ######### Siphon outputs for ABCDnet training #########

        if self._siphon_bdt_data:
            siphon_mask = selections.all(*self._siphon_selection)
            for var in self._bdt_vars:
                if var not in dense_variables_dict:
                    raise Exception(f"BDT var '{var}' not found in dense_variables_dict")
                self._accumulator["bdt_data"][var] += processor.column_accumulator(
                    ak.to_numpy(ak.fill_none(dense_variables_dict[var][siphon_mask], -999)).astype(np.float32)
                )
            self._accumulator["bdt_data"]["weight"] += processor.column_accumulator(
                ak.to_numpy(ak.fill_none(weights_obj_base.weight(None)[siphon_mask], 0)).astype(np.float32)
            )



        ######### Fill 1d histos #########

        # Checks of our input dicts
        vlst = dense_variables_dict.keys()
        hlst = self._dense_axes_dict.keys()
        if len(vlst) != len(set(vlst)): raise Exception("Variable list has a repeat")
        if len(hlst) != len(set(hlst)): raise Exception("Hist list has a repeat")
        for x in vlst:
            if (x == "isRun2") or (x == "isRun3"): continue # Don't expect these in hist list
            if x not in hlst: raise Exception(f"Hist list is missing: {x}")
        #for x in hlst:
        #    if x not in vlst: raise Exception(f"Var list is missing: {x}")

        wgt_correction_syst_lst = []

        # Set up the list of weight fluctuations to loop over
        # For now the syst do not depend on the category, so we can figure this out outside of the filling loop
        wgt_var_lst = ["nominal"]

        # Loop over the hists we want to fill
        for dense_axis_name, dense_axis_vals in dense_variables_dict.items():
            if dense_axis_name not in self._hist_lst:
                #print(f"Skipping \"{dense_axis_name}\", it is not in the list of hists to include.")
                continue

            # Loop over weight fluctuations
            for wgt_fluct in wgt_var_lst:

                # Get the appropriate weight fluctuation
                if (wgt_fluct == "nominal"):
                    # In the case of "nominal", no weight systematic variation is used
                    weight = weights_obj_base.weight(None)
                else:
                    # Otherwise get the weight from the Weights object
                    weight = weights_obj_base.weight(wgt_fluct)


                # Loop over categories
                for sr_cat in cat_dict["lep_chan_lst"]:

                    # If this is a counts hist, forget the weights and just fill with unit weights
                    if dense_axis_name.endswith("_counts"): weight = events.nom

                    # Make the cuts mask
                    cuts_lst = [sr_cat]
                    all_cuts_mask = selections.all(*cuts_lst)

                    # Print info about the events
                    #import sys
                    #run = events.run[all_cuts_mask]
                    #luminosityBlock = events.luminosityBlock[all_cuts_mask]
                    #event = events.event[all_cuts_mask]
                    #w = weight[all_cuts_mask]
                    #if dense_axis_name == "njets":
                    #    print("\nSTARTPRINT")
                    #    for i,j in enumerate(w):
                    #        out_str = f"PRINTTAG {i} {dense_axis_name} {year} {sr_cat} {event[i]} {run[i]} {luminosityBlock[i]} {w[i]}"
                    #        print(out_str,file=sys.stderr,flush=True)
                    #    print("ENDPRINT\n")
                    #print("\ndense_axis_name",dense_axis_name)
                    #print("sr_cat",sr_cat)
                    #print("dense_axis_vals[all_cuts_mask]",dense_axis_vals[all_cuts_mask])
                    #print("end")

                    # Fill the histos
                    axes_fill_info_dict = {
                        dense_axis_name : ak.fill_none(dense_axis_vals[all_cuts_mask],-1), # Don't like this fill_none
                        "weight"        : ak.fill_none(weight[all_cuts_mask],-1),          # Don't like this fill_none
                        "process"       : histAxisName[all_cuts_mask],
                        "category"      : sr_cat,
                        "systematic"    : wgt_fluct,
                        #"year"          : events.year[all_cuts_mask],
                        "lepflav"       : abs_pdgid_sum[all_cuts_mask],
                    }

                    self.accumulator[dense_axis_name].fill(**axes_fill_info_dict)

        return self.accumulator

    def postprocess(self, accumulator):
        if self._siphon_bdt_data:
            import uproot
            import os
            out_dict = {k: v.value for k, v in accumulator["bdt_data"].items()}
            os.makedirs(os.path.dirname(self._siphon_output_path), exist_ok=True)
            with uproot.recreate(self._siphon_output_path) as f:
                f["Events"] = out_dict
        return accumulator
