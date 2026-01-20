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

    def __init__(self, samples, wc_names_lst=[], hist_lst=None, do_systematics=False, skip_obj_systematics=False, skip_signal_regions=False, skip_control_regions=False, muonSyst='nominal', dtype=np.float32, siphon_bdt_data=False, rwgt_to_sm=None):

        self._samples = samples
        self._wc_names_lst = wc_names_lst
        self._dtype = dtype

        # Create the dense axes for the histograms
        self._dense_axes_dict = {
            "l0_pt" : axis.Regular(60, 0, 500, name="l0_pt", label="l0 pt"),
            "nlep"  : axis.Regular(10, 0, 20, name="nlep",   label="Lep multiplicity"),
            "nnu"   : axis.Regular(10, 0, 20, name="nnu",   label="Nu multiplicity"),

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

        #ele   = events.GenPart[(abs(events.GenPart.pdgId)==11) & events.GenPart.hasFlags("isLastCopy")]
        #mu    = events.GenPart[(abs(events.GenPart.pdgId)==13) & events.GenPart.hasFlags("isLastCopy")]
        #nuele = events.GenPart[(abs(events.GenPart.pdgId)==12) & events.GenPart.hasFlags("isLastCopy")]
        #numu  = events.GenPart[(abs(events.GenPart.pdgId)==14) & events.GenPart.hasFlags("isLastCopy")]
        #ele   = events.GenPart[(abs(events.GenPart.pdgId)==11)]
        #muo   = events.GenPart[(abs(events.GenPart.pdgId)==13)]
        #nuele = events.GenPart[(abs(events.GenPart.pdgId)==12)]
        #numuo = events.GenPart[(abs(events.GenPart.pdgId)==14)]
        #ele   = ele[(ele.parent.pdgId==23)     | (ele.parent.pdgId==24)   | (ele.parent.pdgId==-24)]
        #muo   = muo[(muo.parent.pdgId==23)     | (muo.parent.pdgId==24)   | (muo.parent.pdgId==-24)]
        #nuele = nuele[(nuele.parent.pdgId==23) | (nuele.parent.pdgId==24) | (nuele.parent.pdgId==-24)]
        #numuo = numuo[(numuo.parent.pdgId==23) | (numuo.parent.pdgId==24) | (numuo.parent.pdgId==-24)]
        #leps = ak.with_name(ak.concatenate([ele,muo],axis=1),'PtEtaPhiMCandidate')
        #leps = leps[ak.argsort(leps.pt, axis=-1,ascending=False)] # Sort by pt
        #neutrinos = ak.with_name(ak.concatenate([nuele,numuo],axis=1),'PtEtaPhiMCandidate')

        lep = events.GenPart[(abs(events.GenPart.pdgId)==11) | (abs(events.GenPart.pdgId)==13) | (abs(events.GenPart.pdgId)==15)]
        nu  = events.GenPart[(abs(events.GenPart.pdgId)==12) | (abs(events.GenPart.pdgId)==14) | (abs(events.GenPart.pdgId)==16)]

        lep_parent_is_v = ak.fill_none((lep.parent.pdgId==23) | (lep.parent.pdgId==24) | (lep.parent.pdgId==-24),False)
        nu_parent_is_z  = ak.fill_none((nu.parent.pdgId==23),False)
        lep = lep[lep_parent_is_v]
        nu = nu[nu_parent_is_z]


        ################### Obj selection ####################

        lep_padded = ak.pad_none(lep, 4)
        l0 = lep_padded[:,0]
        nlep = ak.num(lep)

        nnu = ak.num(nu)
        hasMet = nnu > 0
        noMet = nnu == 0

        # Put the variables we'll plot into a dictionary for easy access later
        dense_variables_dict = {
            "l0_pt" : ak.where(nlep>0,l0.pt,0),
            "nlep" : nlep,
            "nnu" : nnu,
        }


        ######### Fill histos #########

        mask_dict = {
            "all_events" : nlep > -1,
            "0l_yMet" : ((nlep==0) & hasMet),
            "0l_nMet" : ((nlep==0) & noMet) ,
            "1l_yMet" : ((nlep==1) & hasMet),
            "1l_nMet" : ((nlep==1) & noMet) ,#
            "2l_yMet" : ((nlep==2) & hasMet),
            "2l_nMet" : ((nlep==2) & noMet) ,
            "3l_yMet" : ((nlep==3) & hasMet),
            "3l_nMet" : ((nlep==3) & noMet) ,#
            "4l_yMet" : ((nlep==4) & hasMet),
            "4l_nMet" : ((nlep==4) & noMet) ,
            "5l_yMet" : ((nlep==5) & hasMet),
            "5l_nMet" : ((nlep==5) & noMet) ,#
            "6l_yMet" : ((nlep==6) & hasMet),
            "6l_nMet" : ((nlep==6) & noMet) ,#
            #"7lp" : (nlep>=7),
        }
        genw = events["genWeight"]
        #lumi = 1000.0*get_tc_param(f"lumi_{year}")
        lumi = 1000.0*138
        weight = (xsec/sow)*genw*lumi

        # Loop over variables and fill hists
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
