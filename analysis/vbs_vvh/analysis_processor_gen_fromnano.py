#!/usr/bin/env python
import numpy as np
import awkward as ak
np.seterr(divide='ignore', invalid='ignore', over='ignore')
from coffea import processor
import hist
from hist import axis

from topcoffea.modules.paths import topcoffea_path
from topcoffea.modules.get_param_from_jsons import GetParam
get_tc_param = GetParam(topcoffea_path("params/params.json"))


class AnalysisProcessor(processor.ProcessorABC):

    def __init__(self, samples, wc_names_lst=[], hist_lst=None, do_systematics=False, skip_obj_systematics=False, skip_signal_regions=False, skip_control_regions=False, muonSyst='nominal', dtype=np.float32, siphon_bdt_data=False):

        self._samples = samples
        self._wc_names_lst = wc_names_lst
        self._dtype = dtype

        # Create the dense axes for the histograms
        self._dense_axes_dict = {
            "l0_pt"   : axis.Regular(60, 0, 500, name="l0_pt", label="l0 pt"),
            "j0_pt"   : axis.Regular(60, 0, 1000, name="j0_pt", label="j0 pt"),
            "fj0_pt"  : axis.Regular(30, 0, 700, name="fj0_pt", label="fj0 pt"),
            "nleps"   : axis.Regular(10, 0, 20, name="nleps",   label="Lep multiplicity"),
            "njets"   : axis.Regular(10, 0, 20, name="njets",   label="Jet multiplicity"),
            "nfj"     : axis.Regular(10, 0, 10, name="nfj",   label="Fat jet multiplicity"),

        }

        # Add histograms to dictionary that will be passed on to dict_accumulator
        dout = {}
        for dense_axis_name in self._dense_axes_dict.keys():
            dout[dense_axis_name] = hist.Hist(
                hist.axis.StrCategory([], growth=True, name="process", label="process"),
                hist.axis.StrCategory([], growth=True, name="mask", label="mask"),
                self._dense_axes_dict[dense_axis_name],
                storage="weight", # Keeps track of sumw2
                name="Counts",
            )

        # Set the accumulator
        self._accumulator = processor.dict_accumulator(dout)

    @property
    def accumulator(self):
        return self._accumulator

    @property
    def columns(self):
        return self._columns

    # Main function: run on a given dataset
    def process(self, events):

        # Dataset parameters
        json_name = events.metadata["dataset"]

        histAxisName       = self._samples[json_name]["histAxisName"]
        xsec               = self._samples[json_name]["xsec"]
        sow                = self._samples[json_name]["nSumOfWeights"]
        year               = self._samples[json_name]["year"]

        # Initialize objects
        met  = events.PuppiMET
        #ele  = events.Electron
        #mu   = events.Muon
        #jets = events.Jet
        fatjets = events.FatJet

        ele = events.GenPart[abs(events.GenPart.pdgId)==11]
        mu = events.GenPart[abs(events.GenPart.pdgId)==13]
        jets = events.GenJet

        events.nom = ak.ones_like(met.pt)


        ################### Obj selection ####################

        l_vvh_t = ak.with_name(ak.concatenate([ele,mu],axis=1),'PtEtaPhiMCandidate')
        l_vvh_t = l_vvh_t[ak.argsort(l_vvh_t.pt, axis=-1,ascending=False)] # Sort by pt
        events["l_vvh_t"] = l_vvh_t

        l_vvh_t_padded = ak.pad_none(l_vvh_t, 4)
        l0 = l_vvh_t_padded[:,0]
        nleps = ak.num(l_vvh_t)

        # Count jets
        goodJets = jets
        njets = ak.num(goodJets)
        ht = ak.sum(goodJets.pt,axis=-1)

        goodJets_ptordered = goodJets[ak.argsort(goodJets.pt,axis=-1,ascending=False)]
        goodJets_ptordered_padded = ak.pad_none(goodJets_ptordered, 4)
        j0 = goodJets_ptordered_padded[:,0]

        nfj = ak.num(fatjets)
        fj_ptordered = fatjets[ak.argsort(fatjets.pt,axis=-1,ascending=False)]
        fj_ptordered_padded = ak.pad_none(fj_ptordered, 4)
        fj0 = fj_ptordered_padded[:,0]


        ######### Masks we need for the selection ##########
        # Put the variables we'll plot into a dictionary for easy access later
        dense_variables_dict = {
            "l0_pt" : ak.where(nleps>0,l0.pt,0),
            "j0_pt" : ak.where(njets>0,j0.pt,0),
            "fj0_pt" : ak.where(nfj>0,fj0.pt,0),
            "nleps" : nleps,
            "njets" : njets,
            "nfj" : nfj,
        }


        ######### Fill histos #########

        #mask = njets > -1
        mask_dict = {
            "all_events" : njets > -1,
            "nlep1p" : nleps >= 1,
            "nfj2p" : nfj >= 2,
            "nlep1p_nfj2p" : (nleps >= 1) & (nfj>=2),
        }
        #weight = events.nom
        #if ak.any(events["LHEReweightingWeight"]):
        #    genw = events["LHEReweightingWeight"][:,60]
        #else:
        #    genw = events["genWeight"]
        genw = events["genWeight"]
        #genw = events["LHEReweightingWeight"][:,60]
        #genw = events["LHEReweightingWeight"][:,15]

        #lumi = 1000.0*get_tc_param(f"lumi_{year}")
        lumi = 1000.0*138
        weight = (xsec/sow)*genw*lumi

        for dense_axis_name, dense_axis_vals in dense_variables_dict.items():

            for mask_name,mask in mask_dict.items():

                # Fill the histos
                axes_fill_info_dict = {
                    dense_axis_name : dense_axis_vals[mask],
                    "weight"        : weight[mask],
                    "process"       : histAxisName,
                    "mask"          : mask_name,
                }
                self.accumulator[dense_axis_name].fill(**axes_fill_info_dict)

        return self.accumulator

    def postprocess(self, accumulator):
        return accumulator
