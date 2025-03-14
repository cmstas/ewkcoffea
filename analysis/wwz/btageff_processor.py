#!/usr/bin/env python
#import sys
import numpy as np
import awkward as ak
np.seterr(divide='ignore', invalid='ignore', over='ignore')
from coffea import processor
import hist
from hist import axis
from coffea.analysis_tools import PackedSelection

from topcoffea.modules.paths import topcoffea_path
import topcoffea.modules.event_selection as es_tc
import topcoffea.modules.object_selection as os_tc

from ewkcoffea.modules.paths import ewkcoffea_path as ewkcoffea_path
import ewkcoffea.modules.selection_wwz as es_ec
import ewkcoffea.modules.objects_wwz as os_ec
import ewkcoffea.modules.corrections as cor_ec

from topcoffea.modules.get_param_from_jsons import GetParam
get_tc_param = GetParam(topcoffea_path("params/params.json"))
get_ec_param = GetParam(ewkcoffea_path("params/params.json"))


class AnalysisProcessor(processor.ProcessorABC):

    def __init__(self, samples, wc_names_lst=[], hist_lst=None, ecut_threshold=None, do_errors=False, do_systematics=False, split_by_lepton_flavor=False, skip_signal_regions=False, skip_control_regions=False, muonSyst='nominal', dtype=np.float32,siphon_bdt_data=False):

        self._samples = samples
        self._wc_names_lst = wc_names_lst
        self._dtype = dtype

        # Create the hist for this dense axis variable
        self._histo_dict = {
            "ptabseta": hist.Hist(
                hist.axis.StrCategory([], growth=True, name="process", label="process"),
                hist.axis.StrCategory([], growth=True, name="tag", label="tag"),
                axis.Variable([20, 30, 60, 120], name="pt",  label="pt"),
                axis.Variable([0, 1, 1.8, 2.4], name="abseta",  label="abseta"),
                axis.IntCategory([0, 1, 2, 3, 4, 5], name="flavor"),
                storage="double", # Keeps track of sumw2
                name="Counts",
            )
        }


    @property
    def columns(self):
        return self._columns

    # Main function: run on a given dataset
    def process(self, events):

        # Dataset parameters
        json_name = events.metadata["dataset"]
        #dataset = events.metadata["dataset"]

        isData             = self._samples[json_name]["isData"]
        histAxisName       = self._samples[json_name]["histAxisName"]
        year               = self._samples[json_name]["year"]
        xsec               = self._samples[json_name]["xsec"]
        sow                = self._samples[json_name]["nSumOfWeights"]

        # Set a flag if this is a 2022 year
        is2022 = year in ["2022","2022EE"]
        is2023 = year in ["2023","2023BPix"]

        # If this is a 2022 sample, get the era info
        if isData and (is2022 or is2023):
            era = self._samples[json_name]["era"]
        else:
            era = None

        # Get the dataset name (used for duplicate removal) and check to make sure it is an expected name
        # Get name for MC cases too, since "dataset" is passed to overlap removal function in all cases (though it's not actually used in the MC case)
        dataset = json_name.split('_')[0]
        if isData:
            datasets = ["SingleMuon", "SingleElectron", "EGamma", "MuonEG", "DoubleMuon", "DoubleElectron", "DoubleEG","Muon","Muon0","Muon1","EGamma0","EGamma1"]
            if dataset not in datasets:
                raise Exception("ERROR: Unexpected dataset name for data file.")

        # Initialize objects
        met  = events.PuppiMET
        ele  = events.Electron
        mu   = events.Muon
        tau  = events.Tau
        jets = events.Jet


        ################### Lepton selection ####################

        # Do the object selection for the WWZ eleectrons
        ele_presl_mask = os_ec.is_presel_wwz_ele(ele,is2022,is2023)
        if not (is2022 or is2023):
            ele["topmva"] = os_ec.get_topmva_score_ele(events, year)
            ele["is_tight_lep_for_wwz"] = ((ele.topmva > get_tc_param("topmva_wp_t_e")) & ele_presl_mask)
        else:
            ele["is_tight_lep_for_wwz"] = (ele_presl_mask)

        # Do the object selection for the WWZ muons
        mu_presl_mask = os_ec.is_presel_wwz_mu(mu,is2022,is2023)
        if not (is2022 or is2023):
            mu["topmva"] = os_ec.get_topmva_score_mu(events, year)
            mu["is_tight_lep_for_wwz"] = ((mu.topmva > get_tc_param("topmva_wp_t_m")) & mu_presl_mask)
        else:
            mu["is_tight_lep_for_wwz"] = (mu_presl_mask)

        # Get tight leptons for WWZ selection
        ele_wwz_t = ele[ele.is_tight_lep_for_wwz]
        mu_wwz_t = mu[mu.is_tight_lep_for_wwz]

        # Attach the lepton SFs to the electron and muons collections
        if (is2022 or is2023):
            cor_ec.run3_muons_sf_attach(mu_wwz_t,year,"NUM_MediumID_DEN_TrackerMuons","NUM_LoosePFIso_DEN_MediumID") #TODO: Is there a way to not have these parameters hard-coded? I could work on merging the SF methods for Run2 and Run3 as the year is an input. But, I am not sure if we want this or not
            cor_ec.run3_electrons_sf_attach(ele_wwz_t,year,"wp90iso")
        else:
            cor_ec.AttachElectronSF(ele_wwz_t,year=year)
            cor_ec.AttachMuonSF(mu_wwz_t,year=year)

        l_wwz_t = ak.with_name(ak.concatenate([ele_wwz_t,mu_wwz_t],axis=1),'PtEtaPhiMCandidate')
        l_wwz_t = l_wwz_t[ak.argsort(l_wwz_t.pt, axis=-1,ascending=False)] # Sort by pt

        # For WWZ: Compute pair invariant masses
        llpairs_wwz = ak.combinations(l_wwz_t, 2, fields=["l0","l1"])
        os_pairs_mask = (llpairs_wwz.l0.pdgId*llpairs_wwz.l1.pdgId < 0)   # Maks for opposite-sign pairs
        ll_mass_pairs = (llpairs_wwz.l0+llpairs_wwz.l1).mass            # The mll for each ll pair
        mll_min_afos = ak.min(ll_mass_pairs[os_pairs_mask],axis=-1)
        events["min_mll_afos"] = mll_min_afos # Attach this one to the event info since we need it for selection

        # For WWZ
        l_wwz_t_padded = ak.pad_none(l_wwz_t, 4)
        l0 = l_wwz_t_padded[:,0]
        l1 = l_wwz_t_padded[:,1]
        l2 = l_wwz_t_padded[:,2]
        l3 = l_wwz_t_padded[:,3]

        events["l_wwz_t"] = l_wwz_t
        es_ec.add4lmask_wwz(events, year, isData, histAxisName,is2022,is2023)


        #################### Jets ####################

        # Clean with dr for now
        cleanedJets = os_ec.get_cleaned_collection(l_wwz_t,jets)

        # Selecting jets and cleaning them
        # NOTE: The jet id cut is commented for now in objects.py for the sync
        jetptname = "pt_nom" if hasattr(cleanedJets, "pt_nom") else "pt"
        cleanedJets["is_good"] = os_tc.is_tight_jet(getattr(cleanedJets, jetptname), cleanedJets.eta, cleanedJets.jetId, pt_cut=20., eta_cut=get_ec_param("wwz_eta_j_cut"), id_cut=get_ec_param("wwz_jet_id_cut"))
        goodJets = cleanedJets[cleanedJets.is_good]

        # B tagging
        btagger = "btag" # For deep flavor WPs
        if year=="2016":
            btagwpl = get_tc_param(f"{btagger}_wp_loose_UL16")
            btagwpm = get_tc_param(f"{btagger}_wp_medium_UL16")
        elif year=="2016APV":
            btagwpl = get_tc_param(f"{btagger}_wp_loose_UL16APV")
            btagwpm = get_tc_param(f"{btagger}_wp_medium_UL16APV")
        elif year == "2017":
            btagwpl = get_tc_param(f"{btagger}_wp_loose_UL17")
            btagwpm = get_tc_param(f"{btagger}_wp_medium_UL17")
        elif year == "2018":
            btagwpl = get_tc_param(f"{btagger}_wp_loose_UL18")
            btagwpm = get_tc_param(f"{btagger}_wp_medium_UL18")
        elif year == "2022":
            btagwpl = get_tc_param(f"{btagger}_wp_loose_2022")
            btagwpm = get_tc_param(f"{btagger}_wp_medium_2022")
        elif year == "2022EE":
            btagwpl = get_tc_param(f"{btagger}_wp_loose_2022EE")
            btagwpm = get_tc_param(f"{btagger}_wp_medium_2022EE")
        elif year == "2023":
            btagwpl = get_tc_param(f"{btagger}_wp_loose_2023")
            btagwpm = get_tc_param(f"{btagger}_wp_medium_2023")
        elif year == "2023BPix":
            btagwpl = get_tc_param(f"{btagger}_wp_loose_2023BPix")
            btagwpm = get_tc_param(f"{btagger}_wp_medium_2023BPix")
        else:
            raise ValueError(f"Error: Unknown year \"{year}\".")

        if btagger == "btag":
            isBtagJetsLoose = (goodJets.btagDeepFlavB > btagwpl)
            isBtagJetsMedium = (goodJets.btagDeepFlavB > btagwpm)
        else: Exception("Unknown b tagger")
        nbtagsl = ak.num(goodJets[isBtagJetsLoose])


        ######### Masks we need for the event selection ##########

        # Pass trigger mask
        pass_trg = es_tc.trg_pass_no_overlap(events,isData,dataset,str(year),dataset_dict=es_ec.dataset_dict,exclude_dict=es_ec.exclude_dict,era=era)
        if not (is2022 or is2023):
            pass_trg = (pass_trg & es_ec.trg_matching(events,year))

        # Get some preliminary things we'll need
        es_ec.attach_wwz_preselection_mask(events,l_wwz_t_padded[:,0:4]) # Attach preselection sf and of flags to the events
        leps_from_z_candidate_ptordered, leps_not_z_candidate_ptordered = es_ec.get_wwz_candidates(l_wwz_t_padded[:,0:4]) # Get ahold of the leptons from the Z and from the W
        w_lep0 = leps_not_z_candidate_ptordered[:,0]
        w_lep1 = leps_not_z_candidate_ptordered[:,1]
        mll_wl0_wl1 = (w_lep0 + w_lep1).mass
        w_candidates_mll_far_from_z = ak.fill_none(abs(mll_wl0_wl1 - get_ec_param("zmass")) > 10.0,False) # Will enforce this for SF in the PackedSelection
        bmask_exactly0loose = (nbtagsl==0)

        selections = PackedSelection(dtype='uint64')
        selections.add("all_events", (events.is4lWWZ | (~events.is4lWWZ))) # All events.. this logic is a bit roundabout to just get an array of True
        selections.add("4l_presel", (events.is4lWWZ)) # This matches the VVV looper selection (object selection and event selection)
        #selections.add("sr_4l_sf_presel", (pass_trg & events.is4lWWZ & bmask_exactly0loose & events.wwz_presel_sf & w_candidates_mll_far_from_z & (met.pt > 65.0)))
        #selections.add("sr_4l_of_presel", (pass_trg & events.is4lWWZ & bmask_exactly0loose & events.wwz_presel_of))
        selections.add("sr_4l_sf_presel", (pass_trg & events.is4lWWZ & events.wwz_presel_sf & w_candidates_mll_far_from_z & (met.pt > 65.0)))
        selections.add("sr_4l_of_presel", (pass_trg & events.is4lWWZ & events.wwz_presel_of))


        ######### Fill histos #########

        hout = self._histo_dict
        dense_axis_name = "ptabseta"

        # Get flat jets for all selected events (lose event level info, we no longer care about event level info)
        #all_cuts_mask = selections.all("all_events") # Just for cross checks
        all_cuts_mask = selections.all("sr_4l_sf_presel") | selections.all("sr_4l_of_presel")
        jets_sel = ak.flatten(goodJets[all_cuts_mask])
        pt = jets_sel.pt
        abseta = abs(jets_sel.eta)
        flav = jets_sel.hadronFlavour
        weights = ak.ones_like(jets_sel.pt)

        tag_mask_dict = {
            "all" : jets_sel.btagDeepFlavB > -9999,
            "L"   : jets_sel.btagDeepFlavB > btagwpl,
            "M"   : jets_sel.btagDeepFlavB > btagwpm,
        }

        # Fill the histos
        for tag_mask_name in tag_mask_dict:

            mask = tag_mask_dict[tag_mask_name]

            axes_fill_info_dict = {
                "process"  : histAxisName,
                "tag"      : tag_mask_name,
                "weight"   : weights[mask],
                "pt"       : pt[mask],
                "abseta"   : abseta[mask],
                "flavor"   : flav[mask],
            }
            hout[dense_axis_name].fill(**axes_fill_info_dict)

        return hout

    def postprocess(self, accumulator):
        return accumulator

