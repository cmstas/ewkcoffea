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
#import ewkcoffea.modules.objects_wwz as os_ec
import ewkcoffea.modules.selection_wwz as es_ec

from ewkcoffea.modules.paths import ewkcoffea_path as ewkcoffea_path

import torch
torch.set_num_threads(1)
from ewkcoffea.modules.abcd_model import ABCDLightningModule

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


class AnalysisProcessor(processor.ProcessorABC):

    def __init__(self, samples, wc_names_lst=[], hist_lst=None, do_systematics=False, skip_obj_systematics=False, skip_signal_regions=False, skip_control_regions=False, muonSyst='nominal', dtype=np.float32, siphon_bdt_data=False, rwgt_to_sm=False, ele_cutBased_val=None, mu_pfIsoId_val=None, siphon_out_name="bdt_output"):

        self._samples = samples
        self._wc_names_lst = wc_names_lst
        self._dtype = dtype

        # For ABCDnet evaluations
        self._model = None

        # Create the hist for the 2d abcd
        self.mjj_cap = 5000
        self._abcd_histo_dict = {
            "abcd2d_2lH": hist.Hist(
                hist.axis.StrCategory([], growth=True, name="process", label="process"),
                hist.axis.StrCategory([], growth=True, name="category", label="category"),
                axis.Regular(50, 0, 1, name="dnn_score",   label="DNN score from ABCDnet"),
                axis.Regular(50, 0, self.mjj_cap, name="vbs_mjj", label="Mjj of vbs"),
                #axis.Regular(50, 0, 1, name="vbs_score", label="Score of vbs"),
                storage="weight", name="Counts",
            ),
            "abcd2d_2lV": hist.Hist(
                hist.axis.StrCategory([], growth=True, name="process", label="process"),
                hist.axis.StrCategory([], growth=True, name="category", label="category"),
                axis.Regular(100, 0, 1, name="dnn_score",   label="DNN score from ABCDnet"),
                axis.Regular(100, 0, self.mjj_cap, name="vbs_mjj", label="Mjj of vbs"),
                #axis.Regular(50, 0, 1, name="vbs_score", label="Score of vbs"),
                storage="weight", name="Counts",
            ),
        }

        # Create the dense axes for the histograms
        self._dense_axes_dict = {
            "met"   : axis.Regular(180, 0, 750, name="met",  label="met"),
            "metphi": axis.Regular(180, -3.1416, 3.1416, name="metphi", label="met phi"),
            "scalarptsum_jet" : axis.Regular(180, 0, 2000, name="scalarptsum_jet", label="H_T small radius"),
            "scalarptsum_jetFwd" : axis.Regular(180, 0, 1000, name="scalarptsum_jetFwd", label="H_T forward"),
            "scalarptsum_jetCent" : axis.Regular(180, 0, 2000, name="scalarptsum_jetCent", label="H_T central"),
            "scalarptsum_lep" : axis.Regular(180, 0, 2000, name="scalarptsum_lep", label="S_T"),
            "scalarptsum_lepmet" : axis.Regular(180, 0, 1500, name="scalarptsum_lepmet", label="S_T + metpt"),
            "scalarptsum_lepmetFJ0" : axis.Regular(180, 0, 3500, name="scalarptsum_lepmetFJ0", label="S_T + metpt + FJ0 pt"),
            "scalarptsum_lepmetFJ01" : axis.Regular(180, 0, 3500, name="scalarptsum_lepmetFJ01", label="S_T + metpt + FJ0 pt + FJ1 pt"),
            "scalarptsum_lepmetvbsFJ0" : axis.Regular(180, 0, 3500, name="scalarptsum_lepmetvbsFJ0", label="S_T + metpt + vbs1pt + vbs2pt + FJ0pt"),
            "scalarptsum_lepmetalljets" : axis.Regular(180, 0, 2500, name="scalarptsum_lepmetalljets", label="S_T + metpt + H_T all"),
            "scalarptsum_lepmetcentjets" : axis.Regular(180, 0, 2500, name="scalarptsum_lepmetcentjets", label="S_T + metpt + H_T cent"),
            "scalarptsum_lepmetfwdjets" : axis.Regular(180, 0, 1500, name="scalarptsum_lepmetfwdjets", label="S_T + metpt + H_T fwd"),
            "vectorsum_lepmetvbsFJ0_pt" : axis.Regular(180, 0, 2000, name="vectorsum_lepmetvbsFJ0_pt", label="Pt of vector sum of lep0+lep1+MET+vbs1+vbs2+FJ0"),

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
            "dr_ljnvbs_min" : axis.Regular(180, 0, 6, name="dr_ljnvbs_min", label="Min dr between a non-vbs jet and lepton"),
            "dr_ljnvbs_max" : axis.Regular(180, 0, 6, name="dr_ljnvbs_max", label="Max dr between a non-vbs jet and lepton"),

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
            "mt_wlep" : axis.Regular(180,-2,298, name="mt_wlep", label="MT of MET and W lep (ie, lep that is not the SFOS Z pair)"),
            "dr_wlepmet" : axis.Regular(180,0,6, name="dr_wlepmet", label="dr between MET and W lep (ie, lep that is not the SFOS Z pair)"),

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

            "dnn_score_2lH"   : axis.Regular(180, 0, 1, name="dnn_score_2lH",   label="DNN ABCDnet score for 2l1FJ H region"),
            "dnn_score_2lV"   : axis.Regular(180, 0, 1, name="dnn_score_2lV",   label="DNN ABCDnet score for 1l1FJ V region"),

            "vbs_mjj"       : axis.Regular(180, 0, 4000, name="vbs_mjj",       label="VBS candidate mjj [GeV]"),
            "vbs_absdetajj" : axis.Regular(180, 0, 10,   name="vbs_absdetajj", label="VBS candidate abs delta eta jj"),
            "vbs_score"     : axis.Regular(180, 0, 1,    name="vbs_score",     label="VBS BDT score"),

            "vbs1_pt"  : axis.Regular(180, 0, 400,          name="vbs1_pt",  label="VBS jet 1 pt"),
            "vbs2_pt"  : axis.Regular(180, 0, 400,          name="vbs2_pt",  label="VBS jet 2 pt"),
            "vbs1_eta" : axis.Regular(180, -5, 5,           name="vbs1_eta", label="VBS jet 1 eta"),
            "vbs2_eta" : axis.Regular(180, -5, 5,           name="vbs2_eta", label="VBS jet 2 eta"),
            "vbs1_phi" : axis.Regular(180, -3.1416, 3.1416, name="vbs1_phi", label="VBS jet 1 phi"),
            "vbs2_phi" : axis.Regular(180, -3.1416, 3.1416, name="vbs2_phi", label="VBS jet 2 phi"),


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
        for abcd_hist_name in self._abcd_histo_dict:
            dout[abcd_hist_name] = self._abcd_histo_dict[abcd_hist_name]

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
        self._siphon_selection = ["2lOSSF_nFJ1_massHi_Zp5Hp5VBSp5"] # NOTE this is hard coded
        self._bdt_vars = []
        for varname in list(self._dense_axes_dict.keys()):
            self._bdt_vars.append(varname)
        self._bdt_vars.append("isRun2") # Not in hist dense axis list but we want it
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


    #################################################################################
    ### For ABCDnet evaluations ###
    def _load_model(self, checkpoint_path, model_key):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device
        if not hasattr(self, '_models'):
            self._models = {}
        self._models[model_key] = ABCDLightningModule.load_from_checkpoint(checkpoint_path, map_location=device)
        self._models[model_key].to(device)
        self._models[model_key].eval()

    def _run_abcd_inference(self, events, dense_variables_dict, model):
        if model == "2lH":
            scaler_path = ewkcoffea_path("data/vvh_abcd_models/single_abcdisco_2l1fj_forH_scaler_params.json")
            checkpoint_path = ewkcoffea_path("data/vvh_abcd_models/single_abcdisco_2l1fj_forH.ckpt")
        elif model == "2lV":
            scaler_path = ewkcoffea_path("data/vvh_abcd_models/single_abcdisco_2l1fj_forV_scaler_params.json")
            checkpoint_path = ewkcoffea_path("data/vvh_abcd_models/single_abcdisco_2l1fj_forV.ckpt")
        else:
            raise Exception(f"Unknown model {model}")

        if not hasattr(self, '_models'):
            self._models = {}
        if model not in self._models:
            self._load_model(checkpoint_path, model)

        if not hasattr(self, '_scaler_params_dict'):
            self._scaler_params_dict = {}
        if model not in self._scaler_params_dict:
            import json
            with open(scaler_path) as f:
                self._scaler_params_dict[model] = json.load(f)

        scaler_params = self._scaler_params_dict[model]

        def scale(name, values):
            params = scaler_params[name]
            arr = np.array(values, dtype=np.float64)
            if params["transform"] == "log":
                arr = np.log(np.clip(arr, 1e-9, None))
            lo, hi = params["min"], params["max"]
            denom = hi - lo
            if denom > 0:
                arr = (arr - lo) / denom
            return np.clip(arr, 0.0, 1.0).astype(np.float32)

        feature_matrix = np.column_stack([
            scale(feat, ak.to_numpy(ak.fill_none(dense_variables_dict[feat], -1.0)))
            for feat in scaler_params["_training_features"]
        ])

        features_tensor = torch.from_numpy(feature_matrix).to(self._device)
        with torch.no_grad():
            logits = self._models[model](features_tensor)
            if logits.ndim == 1:
                logits = logits.unsqueeze(-1)
            scores = torch.sigmoid(logits).cpu().numpy()[:, 0]
        return scores
    #################################################################################


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
        vbsjets = events.vbs

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

        # Form the collection of non-vbs jets by masking out the vbs ones
        mask_nvbsjet = (ak.local_index(jets)!=vbsjets.jet1_idx) & (ak.local_index(jets)!=vbsjets.jet2_idx)
        nvbsjets = jets[mask_nvbsjet]

        # Grab the vbs jet objects
        vbs1 = ak.flatten(jets[ak.local_index(jets)==vbsjets.jet1_idx])
        vbs2 = ak.flatten(jets[ak.local_index(jets)==vbsjets.jet2_idx])

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
        ele = ele[ak.values_astype(ele.isLoose,bool)]
        mu  = mu[ak.values_astype(mu.isMedium,bool)]

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
        weights_obj_base.add("norm",events.baseweight)


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
        scalarptsum_lepmetvbsFJ0 = scalarptsum_lep + met.pt + vbs1.pt + vbs2.pt + fj0.pt
        vectorsum_lepmetvbsFJ0_pt = (l0v + l1v + vbs1 + vbs2 + met4 + fj0).pt

        # lb pairs (i.e. always one lep, one bjet)
        lb_pairs = ak.cartesian({"l":to_vec(l_vvh_t),"j": bjetsm})
        mlb_min = ak.min((lb_pairs["l"] + lb_pairs["j"]).mass,axis=-1)
        mlb_max = ak.max((lb_pairs["l"] + lb_pairs["j"]).mass,axis=-1)

        # lj pairs (i.e. always one lep, one jet)
        lj_pairs     = ak.cartesian({"l":to_vec(l_vvh_t),"j": jets})
        ljnvbs_pairs = ak.cartesian({"l":to_vec(l_vvh_t),"j": nvbsjets})
        dr_lj_min     = ak.min(lj_pairs["l"].delta_r(lj_pairs["j"]),axis=-1)
        dr_lj_max     = ak.max(lj_pairs["l"].delta_r(lj_pairs["j"]),axis=-1)
        mass_lj_min   = ak.min((lj_pairs["l"]+lj_pairs["j"]).mass,axis=-1)
        mass_lj_max   = ak.max((lj_pairs["l"]+lj_pairs["j"]).mass,axis=-1)
        dr_ljnvbs_min = ak.min(ljnvbs_pairs["l"].delta_r(ljnvbs_pairs["j"]),axis=-1)
        dr_ljnvbs_max = ak.max(ljnvbs_pairs["l"].delta_r(ljnvbs_pairs["j"]),axis=-1)

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

        # For 3l, find the lepton that's not part of the Z pair
        sfos_mask = ak.any(sfos_pairs_mask, axis=1)
        z_idx0 = ak.flatten(ll_idx_sfos[zpeak_idx].i0, axis=1)
        z_idx1 = ak.flatten(ll_idx_sfos[zpeak_idx].i1, axis=1)
        all_idx = ak.local_index(l_vvh_t, axis=1)
        w_lep_mask = (all_idx != z_idx0) & (all_idx != z_idx1)
        l_w = ak.firsts(l_vvh_t[w_lep_mask])
        mt_wlep = ak.where(sfos_mask,es_ec.get_mt(l_w, met4),-1)
        dr_wlepmet = ak.where(sfos_mask,l_w.delta_r(met4),-1)

        # NOTE Only defind for exactly 2 and 3 lep
        abs_pdgid_sum = ak.fill_none(ak.where(nleps==3,abs(l0.pdgId) + abs(l1.pdgId) + abs(l2.pdgId),abs(l0.pdgId) + abs(l1.pdgId)),0)


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
            "scalarptsum_lepmetvbsFJ0" : scalarptsum_lepmetvbsFJ0,
            "vectorsum_lepmetvbsFJ0_pt" : vectorsum_lepmetvbsFJ0_pt,
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
            "dr_ljnvbs_min" : dr_ljnvbs_min,
            "dr_ljnvbs_max" : dr_ljnvbs_max,

            #"ghiggs0_pt" : ghiggs0.pt,
            #"gvectorboson0_pt" : gvectorboson0.pt,

            "n_ll_sfos": n_ll_sfos,
            "abs_ch_sum_3l": abs_ch_sum_3l,
            "abs_pdgid_sum": abs_pdgid_sum,

            "mll_min_afos" : mll_min_afos,
            "mll_z" : mll_z,
            "mt_wlep":mt_wlep,
            "dr_wlepmet":dr_wlepmet,

            "vbs_mjj"    : vbsjets.mjj,
            "vbs_absdetajj" : vbsjets.detajj,
            "vbs_score"  : vbsjets.score,

            "vbs1_pt": vbs1.pt,
            "vbs2_pt": vbs2.pt,
            "vbs1_eta": vbs1.eta,
            "vbs2_eta": vbs2.eta,
            "vbs1_phi": vbs1.phi,
            "vbs2_phi": vbs2.phi,

            # We want to include this in the siponed output, but probably not make hists for it
            "isRun2" : events.isRun2,
            "isRun3" : events.isRun3,

        }

        # For ABCDnet evaluations
        # This must come after dense_variables_dict since pass all vars from dense_variables_dict to evaluation since any/all might be needed (depending on which model we're using)
        # Once we finish evaluating, add the score to the dense_variables_dict too
        dnn_score_2lH = self._run_abcd_inference(events, dense_variables_dict,"2lH")
        dnn_score_2lV = self._run_abcd_inference(events, dense_variables_dict,"2lV")
        dense_variables_dict["dnn_score_2lH"] = dnn_score_2lH
        dense_variables_dict["dnn_score_2lV"] = dnn_score_2lV


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

        is_2l              = (n_lep_veto==2) & (nleps==2) & (l0.pt>25) & (l1.pt>15)
        is_3l_prelowmllcut = (n_lep_veto==3) & (nleps==3) & (l0.pt>25) & (l1.pt>15) & (l2.pt>10)
        is_3l = is_3l_prelowmllcut & low_mll_cut_3l

        is_VFJ       = (fj0_mparticlenet <= 100.) & (fj0_mparticlenet > 65)
        is_HFJ       = (fj0_mparticlenet >  110.) & (fj0_mparticlenet <= 150.)
        is_HFJTagHbb = (fj0_pNetHbbvsQCD > 0.95)

        selections.add("all_events", pass_through)


        ### 2lOS + 1FJ ###

        selections.add("2l",                                     is_2l)
        selections.add("2lOS",                                   is_2l & is_os)
        selections.add("2lOSSF",                                 is_2l & is_os & is_sf)
        selections.add("2lOSSF_nFJ1",                            is_2l & is_os & is_sf & (nfatjets==1))
        selections.add("2lOSSF_nFJ1_massLo",                     is_2l & is_os & is_sf & (nfatjets==1) & (fj0.gpt_mass2p <  110))
        selections.add("2lOSSF_nFJ1_massHi",                     is_2l & is_os & is_sf & (nfatjets==1) & (fj0.gpt_mass2p >= 110))
        selections.add("2lOSSF_nFJ1_massLo_Zp2",                 is_2l & is_os & is_sf & (nfatjets==1) & (fj0.gpt_mass2p <  110) & (fj0.gptZvsQCD>0.2))
        selections.add("2lOSSF_nFJ1_massHi_Zp2",                 is_2l & is_os & is_sf & (nfatjets==1) & (fj0.gpt_mass2p >= 110) & (fj0.gptZvsQCD>0.2))
        selections.add("2lOSSF_nFJ1_massHi_Zp2_A",               is_2l & is_os & is_sf & (nfatjets==1) & (fj0.gpt_mass2p >= 110) & (fj0.gptZvsQCD>0.2) & (vbsjets.mjj>1560) & (dnn_score_2lH>0.88))
        selections.add("2lOSSF_nFJ1_massLo_Zp2_A",               is_2l & is_os & is_sf & (nfatjets==1) & (fj0.gpt_mass2p <  110) & (fj0.gptZvsQCD>0.2) & (vbsjets.mjj>1080) & (dnn_score_2lH>0.73))
        selections.add("2lOSSF_nFJ1_massHi_Zp5Hp5VBSp5",         is_2l & is_os & is_sf & (nfatjets==1) & (fj0.gpt_mass2p >= 110) & (fj0.gptZvsQCD>0.5) & (fj0.gptHvsQCD>0.5) & (vbsjets.score>0.5))
        selections.add("2lOSSF_nFJ1_massLo_Zp4VBSp6",            is_2l & is_os & is_sf & (nfatjets==1) & (fj0.gpt_mass2p <  110) & (fj0.gptZvsQCD>0.4) & (vbsjets.score>0.6))

        selections.add("2lOSSF_nFJ1_mjj1k",                      is_2l & is_os & is_sf & (nfatjets==1) & (vbsjets.mjj>1000))
        selections.add("2lOSSF_nFJ1_mjj1k_HFJ",                  is_2l & is_os & is_sf & (nfatjets==1) & (vbsjets.mjj>1000) & is_HFJ)
        selections.add("2lOSSF_nFJ1_mjj1k_HFJtag",               is_2l & is_os & is_sf & (nfatjets==1) & (vbsjets.mjj>1000) & is_HFJ & is_HFJTagHbb)
        selections.add("2lOSSF_nFJ1_mjj1k_HFJtag_nb0",           is_2l & is_os & is_sf & (nfatjets==1) & (vbsjets.mjj>1000) & is_HFJ & is_HFJTagHbb & (nbtagst==0))

        selections.add("2lOSSF_nFJ1_onZ_0b",                     is_2l & is_os & is_sf & (nfatjets==1) & is_onZ  & (nbtagsl==0))
        selections.add("2lOSSF_nFJ1_offZ_2b",                    is_2l & is_os & is_sf & (nfatjets==1) & ~is_onZ & (nbtagst==2))


        ### 3l ###

        selections.add("3l_prelowmllcut",                 is_3l_prelowmllcut)
        selections.add("3l",                              is_3l)

        selections.add("3l_chsum3",                       is_3l & (abs_ch_sum_3l==3))
        selections.add("3l_chsum3_mjj500",                is_3l & (abs_ch_sum_3l==3) & (vbsjets.mjj>500))
        selections.add("3l_chsum3_mjj500_nb0",            is_3l & (abs_ch_sum_3l==3) & (vbsjets.mjj>500) & (nbtagst==0))

        selections.add("3l_chsum1",                       is_3l & (abs_ch_sum_3l==1))
        selections.add("3l_chsum1_nFJg0",                 is_3l & (abs_ch_sum_3l==1) & (nfatjets>=1))
        selections.add("3l_chsum1_nFJg0_mjj500",          is_3l & (abs_ch_sum_3l==1) & (nfatjets>=1) & (vbsjets.mjj>500))
        selections.add("3l_chsum1_nFJ0",                  is_3l & (abs_ch_sum_3l==1) & (nfatjets==0))
        selections.add("3l_chsum1_nFJ0_nSFOSg0",          is_3l & (abs_ch_sum_3l==1) & (nfatjets==0) & (n_ll_sfos>=1))
        selections.add("3l_chsum1_nFJ0_nSFOSg0_mjj2k",    is_3l & (abs_ch_sum_3l==1) & (nfatjets==0) & (n_ll_sfos>=1) & (vbsjets.mjj>2000))
        selections.add("3l_chsum1_nFJ0_nSFOS0",           is_3l & (abs_ch_sum_3l==1) & (nfatjets==0) & (n_ll_sfos==0))
        selections.add("3l_chsum1_nFJ0_nSFOS0_mjj1k",     is_3l & (abs_ch_sum_3l==1) & (nfatjets==0) & (n_ll_sfos==0) & (vbsjets.mjj>1000))
        selections.add("3l_chsum1_nFJ0_nSFOS0_mjj1k_nb0", is_3l & (abs_ch_sum_3l==1) & (nfatjets==0) & (n_ll_sfos==0) & (vbsjets.mjj>1000) & (nbtagst==0))

        selections.add("3l_onZ_0b",                       is_3l & is_onZ & (nbtagsl==0))
        selections.add("3l_onZ_0b_mtlmet60",              is_3l & is_onZ & (nbtagsl==0) & (mt_wlep>60))
        selections.add("3l_onZ_0b_mtlmet60_met75",        is_3l & is_onZ & (nbtagsl==0) & (mt_wlep>60) & (met.pt>75))


        # Keep track of the cats we want to actually fill
        cat_dict = {
            "lep_chan_lst" : [

                "all_events",

                ### 2l OS SF 1FJ ###

                "2lOSSF_nFJ1",
                #"2lOSSF_nFJ1_massLo",
                #"2lOSSF_nFJ1_massHi",
                "2lOSSF_nFJ1_massHi_Zp5Hp5VBSp5",
                #"2lOSSF_nFJ1_massLo_Zp4VBSp6",

                # From cut based optimization
                #"2lOSSF_nFJ1_mjj1k",
                #"2lOSSF_nFJ1_mjj1k_HFJ",
                #"2lOSSF_nFJ1_mjj1k_HFJtag",
                #"2lOSSF_nFJ1_mjj1k_HFJtag_nb0",
                #"2lOSSF_nFJ1_HFJ",

                #"2lOSSF_nFJ1_onZ_0b",
                #"2lOSSF_nFJ1_offZ_2b",

                #### 3l ###

                #"3l_prelowmllcut",
                #"3l",
                #"3l_onZ_0b",
                #"3l_onZ_0b_mtlmet60",
                #"3l_onZ_0b_mtlmet60_met75",

                # From cut based optimization
                #"3l_chsum3",
                #"3l_chsum3_mjj500",
                #"3l_chsum3_mjj500_nb0",
                #"3l_chsum1",
                #"3l_chsum1_nFJg0",
                #"3l_chsum1_nFJg0_mjj500",
                #"3l_chsum1_nFJ0",
                #"3l_chsum1_nFJ0_nSFOSg0",
                #"3l_chsum1_nFJ0_nSFOSg0_mjj2k",
                #"3l_chsum1_nFJ0_nSFOS0",
                #"3l_chsum1_nFJ0_nSFOS0_mjj1k",
                #"3l_chsum1_nFJ0_nSFOS0_mjj1k_nb0",

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
            selections.add("2lOSSF_1fjx_ejj3_fj0matchH",  is_2l & is_os & is_sf & (nfatjets==1) & (vbsjets.detajj > 3) & ak.fill_none(fj0_matchedH,  False))
            selections.add("2lOSSF_1fjx_ejj3_fj0matchV1", is_2l & is_os & is_sf & (nfatjets==1) & (vbsjets.detajj > 3) & ak.fill_none(fj0_matchedV1, False))
            selections.add("2lOSSF_1fjx_ejj3_fj0matchV2", is_2l & is_os & is_sf & (nfatjets==1) & (vbsjets.detajj > 3) & ak.fill_none(fj0_matchedV2, False))
            selections.add("2lOSSF_1fjx_ejj3_fj0matchV",  is_2l & is_os & is_sf & (nfatjets==1) & (vbsjets.detajj > 3) & ak.fill_none(fj0_matchedV, False))
            selections.add("2lOSSF_1fjx_ejj3_fj0noMatch", is_2l & is_os & is_sf & (nfatjets==1) & (vbsjets.detajj > 3) & ak.fill_none(fj0_noMatch, False))

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


        ######### Fill the 2d ABCDnet histo #########

        #fill_abcd_2d = False  # At some point should make this an option
        fill_abcd_2d = True # At some point should make this an option
        vbs_mjj_flow = ak.where(vbsjets.mjj<self.mjj_cap,vbsjets.mjj,self.mjj_cap-0.01)
        if fill_abcd_2d:
            # Specify the regions to use for 2d hists (NOTE these are hard coded)
            cat2lH = "2lOSSF_nFJ1_massHi_Zp5Hp5VBSp5"
            cat2lV = "2lOSSF_nFJ1_massLo_Zp2"
            all_cuts_mask_H = selections.all(cat2lH)
            all_cuts_mask_V = selections.all(cat2lV)
            self.accumulator["abcd2d_2lH"].fill(
                #vbs_score = vbsjets.score[all_cuts_mask_H],
                vbs_mjj   = vbs_mjj_flow[all_cuts_mask_H],
                dnn_score = dnn_score_2lH[all_cuts_mask_H],
                weight    = weights_obj_base.weight(None)[all_cuts_mask_H],
                process   = histAxisName[all_cuts_mask_H],
                category  = cat2lH,
            )
            self.accumulator["abcd2d_2lV"].fill(
                #vbs_score = vbsjets.score[all_cuts_mask_V],
                vbs_mjj   = vbs_mjj_flow[all_cuts_mask_V],
                dnn_score = dnn_score_2lV[all_cuts_mask_V],
                weight    = weights_obj_base.weight(None)[all_cuts_mask_V],
                process   = histAxisName[all_cuts_mask_V],
                category  = cat2lV,
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
