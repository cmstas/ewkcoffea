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
import ewkcoffea.modules.objects_wwz as os_ec
#import ewkcoffea.modules.selection_wwz as es_ec


class AnalysisProcessor(processor.ProcessorABC):

    def __init__(self, samples, wc_names_lst=[], hist_lst=None, do_systematics=False, skip_obj_systematics=False, skip_signal_regions=False, skip_control_regions=False, muonSyst='nominal', dtype=np.float32, siphon_bdt_data=False, rwgt_to_sm=False, ele_cutBased_val=None, mu_pfIsoId_val=None):

        self._samples = samples
        self._wc_names_lst = wc_names_lst
        self._dtype = dtype

        # Create the dense axes for the histograms
        self._dense_axes_dict = {
            "met"   : axis.Regular(180, 0, 750, name="met",  label="met"),
            "metphi": axis.Regular(180, -3.1416, 3.1416, name="metphi", label="met phi"),
            "scalarptsum_jetCentFwd" : axis.Regular(180, 0, 2000, name="scalarptsum_jetCentFwd", label="H_T small radius"),
            "scalarptsum_jetFwd" : axis.Regular(180, 0, 1000, name="scalarptsum_jetFwd", label="H_T forward"),
            "scalarptsum_jetCent" : axis.Regular(180, 0, 2000, name="scalarptsum_jetCent", label="H_T central"),
            "scalarptsum_lep" : axis.Regular(180, 0, 2000, name="scalarptsum_lep", label="S_T"),
            "scalarptsum_lepmet" : axis.Regular(180, 0, 1500, name="scalarptsum_lepmet", label="S_T + metpt"),
            "scalarptsum_lepmetFJ" : axis.Regular(180, 0, 3500, name="scalarptsum_lepmetFJ", label="S_T + metpt + FJ pt"),
            "scalarptsum_lepmetFJ10" : axis.Regular(180, 0, 3500, name="scalarptsum_lepmetFJ10", label="S_T + metpt + FJ0 + FJ1 pt"),
            "scalarptsum_lepmetalljets" : axis.Regular(180, 0, 2500, name="scalarptsum_lepmetalljets", label="S_T + metpt + H_T all"),
            "scalarptsum_lepmetcentjets" : axis.Regular(180, 0, 2500, name="scalarptsum_lepmetcentjets", label="S_T + metpt + H_T cent"),
            "scalarptsum_lepmetfwdjets" : axis.Regular(180, 0, 1500, name="scalarptsum_lepmetfwdjets", label="S_T + metpt + H_T fwd"),
            "l0_pt"  : axis.Regular(180, 0, 500, name="l0_pt", label="l0 pt"),
            "l0_eta"  : axis.Regular(180, -3,3, name="l0_eta", label="l0 eta"),
            "l1_pt"  : axis.Regular(180, 0, 400, name="l1_pt", label="l1 pt"),
            "l1_eta"  : axis.Regular(180, -3,3, name="l1_eta", label="l1 eta"),
            "l2_pt"  : axis.Regular(180, 0, 300, name="l2_pt", label="l2 pt"),
            "l2_eta"  : axis.Regular(180, -3,3, name="l2_eta", label="l2 eta"),

            "l0_iso"     : axis.Regular(180, 0,0.2, name="l0_iso", label="l0 pfRelIso03_all"),
            "l0_miniiso" : axis.Regular(180, 0,0.2, name="l0_miniiso", label="l0 miniPFRelIso_all"),
            "l1_iso"     : axis.Regular(180, 0,0.2, name="l1_iso", label="l1 pfRelIso03_all"),
            "l1_miniiso" : axis.Regular(180, 0,0.2, name="l1_miniiso", label="l1 miniPFRelIso_all"),
            "l2_iso"     : axis.Regular(180, 0,0.2, name="l2_iso", label="l2 pfRelIso03_all"),
            "l2_miniiso" : axis.Regular(180, 0,0.2, name="l2_miniiso", label="l2 miniPFRelIso_all"),

            "mass_l0l1"  : axis.Regular(180, 0,500, name="mass_l0l1", label="mll of leading two leptons"),
            "dr_l0l1" : axis.Regular(180, 0, 6, name="dr_l0l1", label="dr between leading two leptons"),

            #"mlb_min" : axis.Regular(180, 0, 300, name="mlb_min",  label="min mass(b+l)"),
            #"mlb_max" : axis.Regular(180, 0, 1000, name="mlb_max",  label="max mass(b+l)"),

            "njets"   : axis.Regular(8, 0, 8, name="njets",   label="Jet multiplicity"),
            "nleps"   : axis.Regular(5, 0, 5, name="nleps",   label="Lep multiplicity"),
            "nbtagsl" : axis.Regular(4, 0, 4, name="nbtagsl", label="Loose btag multiplicity"),
            "nbtagsm" : axis.Regular(4, 0, 4, name="nbtagsm", label="Medium btag multiplicity"),

            "njets_counts"   : axis.Regular(30, 0, 30, name="njets_counts",   label="Jet multiplicity counts (central)"),
            "nleps_counts"   : axis.Regular(30, 0, 30, name="nleps_counts",   label="Lep multiplicity counts (central)"),

            "nfatjets"   : axis.Regular(8, 0, 8, name="nfatjets",   label="Fat jet multiplicity"),
            "njets_forward"   : axis.Regular(8, 0, 8, name="njets_forward",   label="Jet multiplicity (forward)"),
            "njets_tot"   : axis.Regular(8, 0, 8, name="njets_tot",   label="Jet multiplicity (central and forward)"),

            "n_ll_sfos"   : axis.Regular(5, 0, 5, name="n_ll_sfos",   label="Number of SF OS lepton pairs"),
            "abs_ch_sum_3l" : axis.Regular(4, 0, 4, name="abs_ch_sum_3l",   label="Abs sum of charges of the 3l"),

            "fj0_pt"  : axis.Regular(180, 0, 2000, name="fj0_pt", label="fj0 pt"),
            "fj0_mass"  : axis.Regular(180, 0, 250, name="fj0_mass", label="fj0 mass"),
            "fj0_msoftdrop"  : axis.Regular(180, 0, 250, name="fj0_msoftdrop", label="fj0 softdrop mass"),
            "fj0_mparticlenet"  : axis.Regular(180, 0, 250, name="fj0_mparticlenet", label="fj0 particleNet mass"),
            "fj0_eta" : axis.Regular(180, -5, 5, name="fj0_eta", label="fj0 eta"),
            "fj0_phi" : axis.Regular(180, -3.1416, 3.1416, name="fj0_phi", label="j0 phi"),

            "fj0_pNetH4qvsQCD": axis.Regular(180, 0, 1, name="fj0_pNetH4qvsQCD", label="fj0 pNet H4qvsQCD"),
            "fj0_pNetHbbvsQCD": axis.Regular(180, 0, 1, name="fj0_pNetHbbvsQCD", label="fj0 pNet HbbvsQCD"),
            "fj0_pNetHccvsQCD": axis.Regular(180, 0, 1, name="fj0_pNetHccvsQCD", label="fj0 pNet HccvsQCD"),
            "fj0_pNetQCD"     : axis.Regular(180, 0, 1, name="fj0_pNetQCD",    label="fj0 pNet QCD"),
            "fj0_pNetTvsQCD"  : axis.Regular(180, 0, 1, name="fj0_pNetTvsQCD", label="fj0 pNet TvsQCD"),
            "fj0_pNetWvsQCD"  : axis.Regular(180, 0, 1, name="fj0_pNetWvsQCD", label="fj0 pNet WvsQCD"),
            "fj0_pNetZvsQCD"  : axis.Regular(180, 0, 1, name="fj0_pNetZvsQCD", label="fj0 pNet ZvsQCD"),

            "j0central_pt"  : axis.Regular(180, 0, 250, name="j0central_pt", label="j0 pt (central jets)"), # Naming
            "j0central_eta" : axis.Regular(180, -5, 5, name="j0central_eta", label="j0 eta (central jets)"), # Naming
            "j0central_phi" : axis.Regular(180, -3.1416, 3.1416, name="j0central_phi", label="j0 phi (central jets)"), # Naming


            "j0forward_pt"  : axis.Regular(180, 0, 150, name="j0forward_pt", label="j0 pt (forward jets)"),
            "j0forward_eta" : axis.Regular(180, -5, 5, name="j0forward_eta", label="j0 eta (forward jets)"),
            "j0forward_phi" : axis.Regular(180, -3.1416, 3.1416, name="j0forward_phi", label="j0 phi (forward jets)"),

            "j0any_pt"  : axis.Regular(180, 0, 250, name="j0any_pt", label="j0 pt (all regular jets)"),
            "j0any_eta" : axis.Regular(180, -5, 5, name="j0any_eta", label="j0 eta (all regular jets)"),
            "j0any_phi" : axis.Regular(180, -3.1416, 3.1416, name="j0any_phi", label="j0 phi (all regular jets)"),

            "dr_fj0l0" : axis.Regular(180, 0, 6, name="dr_fj0l0", label="dr between FJ and lepton"),
            "dr_j0fwdj1fwd" : axis.Regular(180, 0, 6, name="dr_j0fwdj1fwd", label="dr between leading two forward jets"),
            "dr_j0centj1cent" : axis.Regular(180, 0, 6, name="dr_j0centj1cent", label="dr between leading two central jets"),
            "dr_j0anyj1any" : axis.Regular(180, 0, 6, name="dr_j0anyj1any", label="dr between leading two jets"),

            "absdphi_j0fwdj1fwd"   : axis.Regular(180, 0, 3.1416, name="absdphi_j0fwdj1fwd", label="abs dphi between leading two forward jets"),
            "absdphi_j0centj1cent" : axis.Regular(180, 0, 3.1416, name="absdphi_j0centj1cent", label="abs dphi between leading two central jets"),
            "absdphi_j0anyj1any"   : axis.Regular(180, 0, 3.1416, name="absdphi_j0anyj1any", label="abs dphi between leading two jets"),

            "mass_j0centj1cent" : axis.Regular(180, 0, 250, name="mass_j0centj1cent", label="mjj of two leading (in pt) non-forward jets"),
            "mass_j0fwdj1fwd" : axis.Regular(180, 0, 2500, name="mass_j0fwdj1fwd", label="mjj of two leading (in pt) forward jets"),
            "mass_j0anyj1any" : axis.Regular(180, 0, 1500, name="mass_j0anyj1any", label="mjj of two leading (in pt) jets"),

            "mass_b0b1" : axis.Regular(180, 0, 250, name="mass_b0b1", label="mjj of two leading (pt) b jets"),

            "mass_bbscore0bbscore1" : axis.Regular(180, 0, 250, name="mass_bbscore0bbscore1", label="mjj of two leading (in score) loose b jets"),
            "mass_bmbscore0bmbscore1" : axis.Regular(180, 0, 250, name="mass_bmbscore0bmbscore1", label="mjj of two leading (in score) med b jets"),
            "bbscore0_bscore"  : axis.Regular(180, 0, 1, name="bbscore0_bscore", label="Btag score of b jet with highest btag score"),
            "bbscore1_bscore"  : axis.Regular(180, 0, 1, name="bbscore1_bscore", label="Btag score of b jet with second highest btag score"),

            "mass_jbscore0jbscore1" : axis.Regular(180, 0, 250, name="mass_jbscore0jbscore1", label="mjj of two leading (in score) central jets"),
            "jbscore0_bscore"  : axis.Regular(180, 0, 1, name="jbscore0_bscore", label="Btag score of central jet with highest btag score"),
            "jbscore1_bscore"  : axis.Regular(180, 0, 1, name="jbscore1_bscore", label="Btag score of central jet with second highest btag score"),

            "mjj_max_cent" : axis.Regular(180, 0, 250, name="mjj_max_cent", label="Leading mjj of pair of non-forward jets"),
            "mjj_max_fwd" : axis.Regular(180, 0, 2500, name="mjj_max_fwd", label="Leading mjj of pair of forward jets"),
            "mjj_max_any" : axis.Regular(180, 0, 1500, name="mjj_max_any", label="Leading mjj of pair of any (central or fwd) jets"),
            "absdeta_max_fwd" : axis.Regular(180, 0, 10, name="absdeta_max_fwd", label="Largest abs(delta eta) of pair of forward jets"),
            "absdeta_max_any" : axis.Regular(180, 0, 10, name="absdeta_max_any", label="Largest abs(delta eta) of pair of any (central or fwd) jets"),

            "jj_pairs_atmindr_mjj" : axis.Regular(180, 0, 1000, name="jj_pairs_atmindr_mjj", label="jj_pairs_atmindr_mjj"),

            "mjjjall_nearest_t" : axis.Regular(180, 0, 700, name="mjjjall_nearest_t", label="mjjj closest to top, considering all jets"),
            "mjjjcnt_nearest_t" : axis.Regular(180, 0, 700, name="mjjjcnt_nearest_t", label="mjjj closest to top, considering central jets"),

            "mjjjany" : axis.Regular(180, 0, 3000, name="mjjjany", label="mjjj of leading (in pt) three central or fwd jets"),
            "mjjjcnt" : axis.Regular(180, 0, 3000, name="mjjjcnt", label="mjjj of leading (in pt) three central jets"),
            "mjjjjany" : axis.Regular(180, 0, 4000, name="mjjjjany", label="mjjjj of leading (in pt) four central or fwd jets"),
            "mjjjjcnt" : axis.Regular(180, 0, 4000, name="mjjjjcnt", label="mjjjj of leading (in pt) four central jets"),

            "mljjjany" : axis.Regular(180, 0, 4000, name="mljjjany", label="mljjj of leading (in pt) lep and three central or fwd jets"),
            "mljjjcnt" : axis.Regular(180, 0, 4000, name="mljjjcnt", label="mljjj of leading (in pt) lep and three central jets"),
            "mljjjjany" : axis.Regular(180, 0, 4000, name="mljjjjany", label="mljjjj of leading (in pt) lep and four central or fwd jets"),
            "mljjjjcnt" : axis.Regular(180, 0, 4000, name="mljjjjcnt", label="mljjjj of leading (in pt) lep and four central jets"),

            "abs_pdgid_sum" : axis.Regular(20, 20, 40, name="abs_pdgid_sum", label="Sum of abs pdgId for the 3 lep"),

            #"ghiggs0_pt" : axis.Regular(180, 0, 1500, name="ghiggs0_pt", label="Gen higgs pt"),
            #"gvectorboson0_pt" : axis.Regular(180, 0, 1500, name="gvectorboson0_pt", label="Gen V pt"),

            "mll_min_afos" : axis.Regular(180, 0, 50, name="mll_min_afos",  label="min mll of all OS pairs"),
            "mll_z" : axis.Regular(180, 0, 150, name="mll_z",  label="mll of the pair of leptons closest to z"),

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

        self._ele_cutBased_val = float(ele_cutBased_val)
        self._mu_pfIsoId_val = float(mu_pfIsoId_val)

    @property
    def accumulator(self):
        return self._accumulator

    @property
    def columns(self):
        return self._columns

    # Main function: run on a given dataset
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

        # An array of lenght events that is just 1 for each event
        events.nom = ak.ones_like(met.pt)

        # A mask that is all True by construction (probably there's a better way to do this...)
        pass_through = ak.full_like(met.pt,True,dtype=bool)


        ################### Lepton selection ####################

        # Apply the ID on top of RDF object
        if self._ele_cutBased_val == None: raise Exception("No val for self._ele_cutBased_val")
        if self._mu_pfIsoId_val   == None: raise Exception("No val for pfIsoId_val")
        ele = ele[ele.cutBased >= self._ele_cutBased_val]
        mu = mu[mu.pfIsoId >= self._mu_pfIsoId_val]

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


        ######### Normalization and weights ###########

        # Weights object
        # Note: add() will generally modify up/down weights, so if these are needed for any reason after this point, we should instead pass copies to add()
        weights_obj_base = coffea.analysis_tools.Weights(len(events),storeIndividual=True)
        weights_obj_base.add("norm",events.baseweight)


        #################### Jets ####################

        # Fat jets
        goodfatjets = fatjets

        # Clean with dr (though another option is to use jetIdx)
        cleanedJets = os_ec.get_cleaned_collection(l_vvh_t,jets) # Clean against leps
        cleanedJets = os_ec.get_cleaned_collection(goodfatjets,cleanedJets,drcut=0.8) # Clean against fat jets

        # Selecting jets and cleaning them (already in RDF)
        goodJets = cleanedJets[(abs(cleanedJets.eta) <= 2.4)]
        goodJets_forward = cleanedJets[(abs(cleanedJets.eta) > 2.4)]

        # Count jets
        njets = ak.num(goodJets)
        njets_forward = ak.num(goodJets_forward)
        njets_tot = njets + njets_forward
        nfatjets = ak.num(goodfatjets)
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

        goodJetsCentFwd = ak.with_name(ak.concatenate([goodJets,goodJets_forward],axis=1),'PtEtaPhiMLorentzVector')
        goodJetsCentFwd_ptordered = goodJetsCentFwd[ak.argsort(goodJetsCentFwd.pt,axis=-1,ascending=False)]
        goodJetsCentFwd_ptordered_padded = ak.pad_none(goodJetsCentFwd_ptordered, 4)
        j0any = goodJetsCentFwd_ptordered_padded[:,0]
        j1any = goodJetsCentFwd_ptordered_padded[:,1]
        j2any = goodJetsCentFwd_ptordered_padded[:,2]
        j3any = goodJetsCentFwd_ptordered_padded[:,3]

        goodfatjets_ptordered = goodfatjets[ak.argsort(goodfatjets.pt,axis=-1,ascending=False)]
        goodfatjets_ptordered_padded = ak.pad_none(goodfatjets_ptordered, 2)
        fj0 = goodfatjets_ptordered_padded[:,0]
        fj1 = goodfatjets_ptordered_padded[:,1]

        scalarptsum_jetCentFwd = ak.sum(goodJetsCentFwd.pt,axis=-1)
        scalarptsum_jetCent = ak.sum(goodJets.pt,axis=-1)
        scalarptsum_jetFwd = ak.sum(goodJets_forward.pt,axis=-1)

        mjjjany  = ak.where(njets_tot>=3, (j0any+j1any+j2any).mass, 0)
        mjjjcnt  = ak.where(njets>=3, (j0+j1+j2).mass, 0)
        mjjjjany = ak.where(njets_tot>=4, (j0any+j1any+j2any+j3any).mass, 0)
        mjjjjcnt = ak.where(njets>=4, (j0+j1+j2+j3).mass, 0)

        mljjjany  = ak.where(njets_tot>=3, (l0 + j0any+j1any+j2any).mass, 0)
        mljjjcnt  = ak.where(njets>=3, (l0 + j0+j1+j2).mass, 0)
        mljjjjany = ak.where(njets_tot>=4, (l0 + j0any+j1any+j2any+j3any).mass, 0)
        mljjjjcnt = ak.where(njets>=4, (l0 + j0+j1+j2+j3).mass, 0)


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

        # Replace with 0 when there are not a pair of jets
        mjj_tmp = (j0+j1).mass
        mass_j0centj1cent = ak.where(njets>1,mjj_tmp,0)

        j0forward_eta = ak.where(njets_forward>0,j0forward.eta,0)

        j0any_pt = ak.where(njets_tot>0,j0any.pt,0)

        mass_j0anyj1any = ak.where(njets_tot>1,(j0any+j1any).mass,0)

        mass_j0fwdj1fwd = ak.where(njets_forward>1,(j0forward+j1forward).mass,0)

        # Count lepton pairs
        ll_pairs = ak.combinations(l_vvh_t_padded, 2, fields=["l0", "l1"] )
        sfos_mask = ak.fill_none((ll_pairs.l0.pdgId == -ll_pairs.l1.pdgId),False)
        n_ll_sfos = ak.num(ll_pairs[sfos_mask])

        # Find the mjj of the pair of jets (central + fwd) that have the min delta R
        jj_pairs = ak.combinations(goodJetsCentFwd_ptordered_padded, 2, fields=["j0", "j1"] )
        jj_pairs_dr = jj_pairs.j0.delta_r(jj_pairs.j1)
        jj_pairs_idx_mindr = ak.argmin(jj_pairs_dr,axis=1,keepdims=True)
        jj_pairs_atmindr = jj_pairs[jj_pairs_idx_mindr]
        jj_pairs_atmindr_mjj = (jj_pairs_atmindr.j0 + jj_pairs_atmindr.j1).mass
        jj_pairs_atmindr_mjj = ak.flatten(ak.fill_none(jj_pairs_atmindr_mjj,-999)) # Replace Nones, flatten (so e.g. [[None],[x],[y]] -> [-999,x,y])

        # Find jet triplets clost to top mass
        jetall_triplets = ak.combinations(goodJetsCentFwd_ptordered_padded, 3, fields=["j0", "j1", "j2"] )
        jetcnt_triplets = ak.combinations(goodJets_ptordered_padded,        3, fields=["j0", "j1", "j2"] )
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
        scalarptsum_lepmetFJ = scalarptsum_lep + met.pt + fj0.pt
        scalarptsum_lepmetFJ10 = scalarptsum_lep + met.pt + fj0.pt + fj1.pt
        scalarptsum_lepmetalljets = scalarptsum_lep + met.pt + scalarptsum_jetCentFwd
        scalarptsum_lepmetcentjets = scalarptsum_lep + met.pt + scalarptsum_jetCent
        scalarptsum_lepmetfwdjets = scalarptsum_lep + met.pt + scalarptsum_jetFwd

        # lb pairs (i.e. always one lep, one bjet)
        lb_pairs = ak.cartesian({"l":l_vvh_t,"j": bjetsl})
        mlb_min = ak.min((lb_pairs["l"] + lb_pairs["j"]).mass,axis=-1)
        mlb_max = ak.max((lb_pairs["l"] + lb_pairs["j"]).mass,axis=-1)

        bjets_ptordered = bjetsl[ak.argsort( bjetsl.pt,axis=-1,ascending=False)]
        bjets_ptordered_padded = ak.pad_none(bjets_ptordered, 2)
        b0 = bjets_ptordered_padded[:,0]
        b1 = bjets_ptordered_padded[:,1]
        mass_b0b1_tmp = (b0+b1).mass
        mass_b0b1 = ak.where(nbtagsl>1,mass_b0b1_tmp,0)

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

        # Variables related to leading b jet score of central jets
        #centraljets_bscoreordered = goodJets_ptordered_padded[ak.argsort(goodJets_ptordered_padded.btagDeepFlavB,axis=-1,ascending=False)]
        #jbscore0 = centraljets_bscoreordered[:,0]
        #jbscore1 = centraljets_bscoreordered[:,1]
        #mass_jbscore0jbscore1 = ak.fill_none((jbscore0+jbscore1).mass,0)
        #jbscore0_bscore = ak.fill_none(jbscore0.btagDeepFlavB,0)
        #jbscore1_bscore = ak.fill_none(jbscore1.btagDeepFlavB,0)

        # Mjj max from any jets
        jjCentFwd_pairs = ak.combinations( goodJetsCentFwd_ptordered_padded, 2, fields=["j0", "j1"] )
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

        # For WWZ: Compute pair invariant masses
        llpairs_wwz = ak.combinations(l_vvh_t, 2, fields=["l0","l1"])
        os_pairs_mask = (llpairs_wwz.l0.pdgId*llpairs_wwz.l1.pdgId < 0)   # Maks for opposite-sign pairs
        sfos_pairs_mask = (llpairs_wwz.l0.pdgId == -llpairs_wwz.l1.pdgId) # Mask for same-flavor-opposite-sign pairs
        ll_absdphi_pairs = abs(llpairs_wwz.l0.delta_phi(llpairs_wwz.l1))
        ll_mass_pairs = (llpairs_wwz.l0+llpairs_wwz.l1).mass            # The mll for each ll pair
        absdphi_min_afas = ak.min(ll_absdphi_pairs,axis=-1)
        absdphi_min_afos = ak.min(ll_absdphi_pairs[os_pairs_mask],axis=-1)
        absdphi_min_sfos = ak.min(ll_absdphi_pairs[sfos_pairs_mask],axis=-1)
        mll_min_afas = ak.min(ll_mass_pairs,axis=-1)
        mll_min_afos = ak.min(ll_mass_pairs[os_pairs_mask],axis=-1)
        mll_min_sfos = ak.min(ll_mass_pairs[sfos_pairs_mask],axis=-1)

        ll_pairs_tmp = ak.combinations(l_vvh_t_padded, 2, fields=["l0", "l1"] )
        ll_pairs_4vec = ll_pairs_tmp.l0 + ll_pairs_tmp.l1
        zpeak_idx = ak.argmin(abs(ll_pairs_4vec.mass-91.1876),keepdims=True,axis=1)
        mll_z = ak.fill_none(ak.flatten(ll_pairs_4vec[zpeak_idx].mass),0)

        # NOTE Only defind for exactly 2 and 3 lep
        abs_pdgid_sum = ak.fill_none(ak.where(nleps==3,abs(l0.pdgId) + abs(l1.pdgId) + abs(l2.pdgId),abs(l0.pdgId) + abs(l1.pdgId)),0)

        # Put the variables we'll plot into a dictionary for easy access later
        dense_variables_dict = {
            "met" : met.pt,
            "metphi" : met.phi,
            "scalarptsum_lep" : scalarptsum_lep,
            "scalarptsum_jetCentFwd" : scalarptsum_jetCentFwd,
            "scalarptsum_jetCent" : scalarptsum_jetCent,
            "scalarptsum_jetFwd" : scalarptsum_jetFwd,
            "scalarptsum_lepmet" : scalarptsum_lepmet,
            "scalarptsum_lepmetFJ" : scalarptsum_lepmetFJ,
            "scalarptsum_lepmetFJ10" : scalarptsum_lepmetFJ10,
            "scalarptsum_lepmetalljets" : scalarptsum_lepmetalljets,
            "scalarptsum_lepmetcentjets" : scalarptsum_lepmetcentjets,
            "scalarptsum_lepmetfwdjets" : scalarptsum_lepmetfwdjets,
            "l0_pt"  : l0.pt,
            "l0_eta" : l0.eta,
            "l1_pt"  : l1.pt,
            "l1_eta" : l1.eta,
            "l2_pt"  : l2.pt,
            "l2_eta" : l2.eta,
            "mass_l0l1" : mass_l0l1,
            "dr_l0l1" : dr_l0l1,
            "l0_iso"     : l0.pfRelIso03_all,
            "l0_miniiso" : l0.miniPFRelIso_all,
            "l1_iso"     : l1.pfRelIso03_all,
            "l1_miniiso" : l1.miniPFRelIso_all,
            "l2_iso"     : l2.pfRelIso03_all,
            "l2_miniiso" : l2.miniPFRelIso_all,

            "j0central_pt"  : j0.pt,
            "j0central_eta" : j0.eta,
            "j0central_phi" : j0.phi,

            "j0forward_pt"  : j0forward.pt,
            "j0forward_eta" : j0forward_eta,
            "j0forward_phi" : j0forward.phi,

            "j0any_pt"  : j0any_pt,
            "j0any_eta" : j0any.eta,
            "j0any_phi" : j0any.phi,

            "nleps" : nleps,
            "njets" : njets,
            "nbtagsl" : nbtagsl,

            "nleps_counts" : nleps,
            "njets_counts" : njets,
            "nbtagsl_counts" : nbtagsl,

            "nbtagsm" : nbtagsm,
            "nbtagsl" : nbtagsl,

            "nfatjets" : nfatjets,
            "njets_forward" : njets_forward,
            "njets_tot" : njets_tot,
            "fj0_pt" : fj0.pt,
            "fj0_mass" : fj0.mass,
            "fj0_msoftdrop" : fj0.msoftdrop,
            "fj0_eta" : fj0.eta,
            "fj0_phi" : fj0.phi,

            "j0_pt" : j0.pt,
            "j0_eta" : j0.eta,
            "j0_phi" : j0.phi,

            "dr_fj0l0" : fj0.delta_r(l0),
            "dr_j0fwdj1fwd" : j0forward.delta_r(j1forward),
            "dr_j0centj1cent" : j0.delta_r(j1),
            "dr_j0anyj1any" : j0any.delta_r(j1any),
            "absdphi_j0fwdj1fwd"   : abs(j0forward.delta_phi(j1forward)),
            "absdphi_j0centj1cent" : abs(j0.delta_phi(j1)),
            "absdphi_j0anyj1any"   : abs(j0any.delta_phi(j1any)),

            "mass_j0centj1cent" : mass_j0centj1cent,
            "mass_j0fwdj1fwd" : mass_j0fwdj1fwd,
            "mass_j0anyj1any" : mass_j0anyj1any,

            "mass_b0b1" : mass_b0b1,

            "fj0_pNetH4qvsQCD" : fj0_pNetH4qvsQCD,
            "fj0_pNetHbbvsQCD" : fj0_pNetHbbvsQCD,
            "fj0_pNetHccvsQCD" : fj0_pNetHccvsQCD,
            "fj0_pNetQCD"      : fj0_pNetQCD,
            "fj0_pNetTvsQCD"   : fj0_pNetTvsQCD,
            "fj0_pNetWvsQCD"   : fj0_pNetWvsQCD,
            "fj0_pNetZvsQCD"   : fj0_pNetZvsQCD,
            "fj0_mparticlenet" : fj0_mparticlenet,

            "jj_pairs_atmindr_mjj" : jj_pairs_atmindr_mjj,

            "bbscore0_bscore" : bbscore0_bscore,
            "bbscore1_bscore" : bbscore1_bscore,
            "mass_bbscore0bbscore1" : mass_bbscore0bbscore1,
            "mass_bmbscore0bmbscore1" : mass_bmbscore0bmbscore1,

            #"jbscore0_bscore" : jbscore0_bscore,
            #"jbscore1_bscore" : jbscore1_bscore,
            #"mass_jbscore0jbscore1" : mass_jbscore0jbscore1,

            "mjj_max_any" : mjj_max_any,
            "mjj_max_cent" : mjj_max_cent,
            "mjj_max_fwd" : mjj_max_fwd,

            "absdeta_max_fwd" : absdeta_max_fwd,
            "absdeta_max_any" : absdeta_max_any,

            "mjjjall_nearest_t": mjjjall_nearest_t,
            "mjjjcnt_nearest_t": mjjjcnt_nearest_t,

            "mjjjany" : mjjjany,
            "mjjjcnt" : mjjjcnt,
            "mjjjjany" : mjjjjany,
            "mjjjjcnt" : mjjjjcnt,

            "mljjjany" : mljjjany,
            "mljjjcnt" : mljjjcnt,
            "mljjjjany" : mljjjjany,
            "mljjjjcnt" : mljjjjcnt,

            #"ghiggs0_pt" : ghiggs0.pt,
            #"gvectorboson0_pt" : gvectorboson0.pt,

            "n_ll_sfos": n_ll_sfos,
            "abs_ch_sum_3l": abs_ch_sum_3l,
            "abs_pdgid_sum": abs_pdgid_sum,

            "mll_min_afos" : mll_min_afos,
            "mll_z" : mll_z,

        }

        # Lepton truth info (note this check assumes all events in this chunk are of the same kind, should be true)
        isData = False
        if events.kind[0]=="data": isData = True
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

        is_2l = (nleps==2) & (l0.pt>25) & (l1.pt>15)
        is_3l = (nleps==3) & (l0.pt>25) & (l1.pt>15) & (l2.pt>10)
        #is_2l_mll12 = is_2l & (mll_min_afos>12)
        is_3l_mll12 = is_3l & (mll_min_afos>12)

        is_VFJ       = (fj0_mparticlenet <= 100.) & (fj0_mparticlenet > 65)
        is_HFJ       = (fj0_mparticlenet >  110.) & (fj0_mparticlenet <= 150.)
        is_HFJTagHbb = (fj0_pNetHbbvsQCD > 0.98)

        is_onZ = abs(mass_l0l1 - 91.1876) < 20

        selections.add("all_events", pass_through)


        ### 2lOS + 1FJ ###

        selections.add("2l",                                      is_2l)
        selections.add("2lOS",                                    is_2l & is_os)
        selections.add("2lOSSF",                                  is_2l & is_os & is_sf)
        selections.add("2lOSSF_1fj",                              is_2l & is_os & is_sf & (nfatjets>=1))
        selections.add("2lOSSF_1fjx",                             is_2l & is_os & is_sf & (nfatjets==1))
        selections.add("2lOSSF_1fjx_2j",                          is_2l & is_os & is_sf & (nfatjets==1) & (njets_tot>=2))
        selections.add("2lOSSF_1fjx_HFJ",                         is_2l & is_os & is_sf & (nfatjets==1) & is_HFJ)
        selections.add("2lOSSF_1fjx_HFJtag",                      is_2l & is_os & is_sf & (nfatjets==1) & is_HFJ & is_HFJTagHbb)
        selections.add("2lOSSF_1fjx_HFJtag_nj2",                  is_2l & is_os & is_sf & (nfatjets==1) & is_HFJ & is_HFJTagHbb & (njets_tot>=2))
        selections.add("2lOSSF_1fjx_HFJtag_nj2_mjj600",           is_2l & is_os & is_sf & (nfatjets==1) & is_HFJ & is_HFJTagHbb & (njets_tot>=2) & (mjj_max_any>600))
        selections.add("2lOSSF_1fjx_HFJtag_nj2_mjj600_nbm0",      is_2l & is_os & is_sf & (nfatjets==1) & is_HFJ & is_HFJTagHbb & (njets_tot>=2) & (mjj_max_any>600) & (nbtagsm==0))
        selections.add("2lOSSF_1fjx_HFJtag_nj2_mjj600_nbm0_onZ",  is_2l & is_os & is_sf & (nfatjets==1) & is_HFJ & is_HFJTagHbb & (njets_tot>=2) & (mjj_max_any>600) & (nbtagsm==0) & is_onZ)
        selections.add("2lOSSF_1fjx_HFJtag_nj2_mjj600_nbm0_offZ", is_2l & is_os & is_sf & (nfatjets==1) & is_HFJ & is_HFJTagHbb & (njets_tot>=2) & (mjj_max_any>600) & (nbtagsm==0) & ~is_onZ)


        ### 3l ###

        selections.add("3l",                                   is_3l)

        selections.add("3l_chsum3",                            is_3l       & (abs_ch_sum_3l==3))

        selections.add("3l_chsum3_mjj400",                     is_3l       & (abs_ch_sum_3l==3) & (mjj_max_any>400))
        selections.add("3l_chsum3_mjj400_b0p4",                is_3l       & (abs_ch_sum_3l==3) & (mjj_max_any>400) & (bbscore0_bscore<0.4))

        selections.add("3l_chsum1",                            is_3l       & (abs_ch_sum_3l==1))
        selections.add("3l_chsum1_mll12",                      is_3l_mll12 & (abs_ch_sum_3l==1))

        selections.add("3l_chsum1_mll12_sfos0",                is_3l_mll12 & (abs_ch_sum_3l==1) & (n_ll_sfos==0))
        selections.add("3l_chsum1_mll12_sfos0_mjj400",         is_3l_mll12 & (abs_ch_sum_3l==1) & (n_ll_sfos==0) & (mjj_max_any>400))
        selections.add("3l_chsum1_mll12_sfos0_mjj400_b0p4",    is_3l_mll12 & (abs_ch_sum_3l==1) & (n_ll_sfos==0) & (mjj_max_any>400) & (bbscore0_bscore<0.4))

        selections.add("3l_chsum1_mll12_sfos1",                is_3l_mll12 & (abs_ch_sum_3l==1) & (n_ll_sfos==1))
        selections.add("3l_chsum1_mll12_sfos1_mjj400",         is_3l_mll12 & (abs_ch_sum_3l==1) & (n_ll_sfos==1) & (mjj_max_any>400))
        selections.add("3l_chsum1_mll12_sfos1_mjj400_jf0pt50", is_3l_mll12 & (abs_ch_sum_3l==1) & (n_ll_sfos==1) & (mjj_max_any>400) & (j0forward.pt>50))

        selections.add("3l_chsum1_mll12_sfos2",                is_3l_mll12 & (abs_ch_sum_3l==1) & (n_ll_sfos==2))
        selections.add("3l_chsum1_mll12_sfos2_mjj400",         is_3l_mll12 & (abs_ch_sum_3l==1) & (n_ll_sfos==2) & (mjj_max_any>400))
        selections.add("3l_chsum1_mll12_sfos2_mjj400_jf0pt50", is_3l_mll12 & (abs_ch_sum_3l==1) & (n_ll_sfos==2) & (mjj_max_any>400) & (j0forward.pt>50))



        # Keep track of the ones we want to actually fill
        cat_dict = {
            "lep_chan_lst" : [

                "all_events",

                ### 2l OS SF 1FJ ###

                "2l",
                "2lOS",
                "2lOSSF",
                "2lOSSF_1fj",
                "2lOSSF_1fjx",
                "2lOSSF_1fjx_2j",
                "2lOSSF_1fjx_HFJ",
                "2lOSSF_1fjx_HFJtag",
                "2lOSSF_1fjx_HFJtag_nj2",
                "2lOSSF_1fjx_HFJtag_nj2_mjj600",
                "2lOSSF_1fjx_HFJtag_nj2_mjj600_nbm0",
                "2lOSSF_1fjx_HFJtag_nj2_mjj600_nbm0_onZ",
                "2lOSSF_1fjx_HFJtag_nj2_mjj600_nbm0_offZ",

                ### 3l ###

                "3l",
                "3l_chsum3",
                "3l_chsum3_mjj400",
                "3l_chsum3_mjj400_b0p4",
                "3l_chsum1",
                "3l_chsum1_mll12",
                "3l_chsum1_mll12_sfos0",
                "3l_chsum1_mll12_sfos0_mjj400",
                "3l_chsum1_mll12_sfos0_mjj400_b0p4",
                "3l_chsum1_mll12_sfos1",
                "3l_chsum1_mll12_sfos1_mjj400",
                "3l_chsum1_mll12_sfos1_mjj400_jf0pt50",
                "3l_chsum1_mll12_sfos2",
                "3l_chsum1_mll12_sfos2_mjj400",
                "3l_chsum1_mll12_sfos2_mjj400_jf0pt50",
            ]
        }


        ######### Fill histos #########

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
                        dense_axis_name : ak.fill_none(dense_axis_vals[all_cuts_mask],0), # Don't like this fill_none
                        "weight"        : ak.fill_none(weight[all_cuts_mask],0),          # Don't like this fill_none
                        "process"       : histAxisName[all_cuts_mask],
                        "category"      : sr_cat,
                        "systematic"    : wgt_fluct,
                        #"year"          : events.year[all_cuts_mask],
                        "lepflav"       : abs_pdgid_sum[all_cuts_mask],
                    }

                    self.accumulator[dense_axis_name].fill(**axes_fill_info_dict)

        return self.accumulator

    def postprocess(self, accumulator):
        return accumulator
