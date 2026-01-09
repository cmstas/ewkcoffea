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
from config.variables.var_config import obj,other_objs, dense_variables_config

from config.paths import objsel_cf,get_cutflow,cutflow_yamls_dir,default_cutflow_yaml
from config.variables.extra_definitions import VV_ndaughters, Higgs_final_state

class AnalysisProcessor(processor.ProcessorABC):

    def __init__(self, samples, wc_names_lst=[], n_minus_1=False,project=None, cutflow_name=None):
        print('debug',project,cutflow_name)
        
        if cutflow_name is None:
            print(f"default cutflow is used.")
            self.cutflow = objsel_cf
        else: 
            print(f'{cutflow_yamls_dir}/{project}.yaml',cutflow_name)
            try: #try project, then try all.yaml, finally just use basic
                self.cutflow = get_cutflow(f'{cutflow_yamls_dir}/{project}.yaml',cutflow_name)
            except:
                try:
                    self.cutflow = get_cutflow(default_cutflow_yaml,cutflow_name)
                    print(f"Cannot get {cutflow_name} in {str(project)}.yaml. Found cutflow in defualt yaml and use it instead")
                except:
                    try:
                        print(get_cutflow(f'{cutflow_yamls_dir}/{project}.yaml'))
                        print("can get project yaml but not the cutflow. Use default cutflow instead.")
                        self.cutflow = objsel_cf
                    except:
                        self.cutflow = objsel_cf
                        print(f"Warning: Cutflow {cutflow_name} not found. Use default cutflow instead.")
            else:
                print(f"Cutflow {cutflow_name} is used.")
        self._n_minus_1 = n_minus_1 #sel mask will be n-1 cut

        self._samples = samples
        self._wc_names_lst = wc_names_lst #wilson coefficients; not used
        self._dtype = np.float32 #note that this has been changed from function input to this fixed defintion
        

        # Create the dense axes for the histograms
        self._dense_axes_dict = {
            var_name: cfg["axis"]
            for var_name, cfg in dense_variables_config.items()
        }

        # Add histograms to dictionary that will be passed on to dict_accumulator
        dout = {}
        for dense_axis_name in self._dense_axes_dict.keys():
            dout[dense_axis_name] = hist.Hist(
                hist.axis.StrCategory([], growth=True, name="process", label="process"),
                hist.axis.StrCategory([], growth=True, name="category", label="category"),
                hist.axis.StrCategory([], growth=True, name="systematic", label="systematic"),
                hist.axis.StrCategory([], growth=True, name="year", label="year"),
                self._dense_axes_dict[dense_axis_name],
                storage="weight", # Keeps track of sumw2
                name="Counts",
            )

        # Set the accumulator
        self._accumulator = processor.dict_accumulator(dout)
        '''
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
        '''

    @property
    def accumulator(self):
        return self._accumulator

    @property
    def columns(self):
        return self._columns

    # Main function: run on a given dataset
    def process(self, events):

        ak.behavior.update(vector.behavior)

        #move triggers to sparse axes? later?
        Flag_NoAK4BJet = events.Pass_NoAK4BJet
        
        objects = {
            name: builder(events)
            for name, builder in obj.items()
        }
        objects.update({
            name: builder(events,objects)
            for name, builder in other_objs.items()
        })
        #jets = objects['goodAK4Jets']
        #jets_sorted = jets[ak.argsort(jets.pt, axis=1, ascending=False)]
        #objects['leading_jet'] = jets_sorted[:, 0]

        var = {
            var_name: cfg["expr"](events, objects)
            for var_name, cfg in dense_variables_config.items()
        }
        # Dataset parameters
        json_name = events.metadata["dataset"]
        isSig             = self._samples[json_name]["isSig"]
        isData            = self._samples[json_name]["isData"]
        '''
        isData             = self._samples[json_name]["isData"]
        histAxisName       = self._samples[json_name]["histAxisName"]
        year               = self._samples[json_name]["year"]
        xsec               = self._samples[json_name]["xsec"]
        sow                = self._samples[json_name]["nSumOfWeights"]
        '''

        # An array of lenght events that is just 1 for each event
        # Probably there's a better way to do this, but we use this method elsewhere so I guess why not..
        events['nom'] = ak.ones_like(var['nGoodAK4'])

        ################### Lepton selection ####################

        ######### Normalization and weights ###########


        # These weights can go outside of the outside sys loop since they do not depend on pt of mu or jets
        # We only calculate these values if not isData
        # Note: add() will generally modify up/down weights, so if these are needed for any reason after this point, we should instead pass copies to add()
        # Note: Here we will to the weights object the SFs that do not depend on any of the forthcoming loops
        # If we're doing systematics and this isn't data, we will loop over the obj correction syst lst list
        obj_corr_syst_var_list = ['nominal']

        # Loop over the list of systematic variations (that impact object kinematics) that we've constructed
        # Make a copy of the base weights object, so that each time through the loop we do not double count systs
        # In this loop over systs that impact kinematics, we will add to the weights objects the SFs that depend on the object kinematics

        #################### Jets ####################
        
        # Put the variables we'll plot into a dictionary for easy access later
        dense_variables_dict  = var

        ######### Store boolean masks with PackedSelection ##########

        selections = PackedSelection(dtype='uint64')
        '''
        for sel, crit_fn_tuple in self.cutflow.items():
            crit_fn = crit_fn_tuple[0]   # get the function
            selections.add(sel, crit_fn(events, var, obj))

        for sel, crit_fn in objsel_cf.items():
            crit_fn = crit_fn_tuple[0]   # get the function
            selections.add(sel, crit_fn(events, var, obj))'''

        for sel, crit_fn in self.cutflow.items():
            mask = crit_fn(events, var, obj)
            selections.add(sel, mask)
        '''
        for sel, crit_fn_tuple in objsel_cf.items():
            crit_fn = crit_fn_tuple[0]                     # extract the function
            mask = crit_fn(events, var, obj)               # evaluate → bool array
            selections.add(sel, mask)
        '''
        ######### Fill histos #########

        exclude_var_dict = {} # Any particular ones to skip

        # Set up the list of weight fluctuations to loop over
        # For now the syst do not depend on the category, so we can figure this out outside of the filling loop
        # Define all weight variations
        wgt_var_dict = {
            "nominal": events.weight,
            #"count"  : events.nom,
        }

        if isSig:                 # or use the ak.firsts trick
            print("Add other coupling points")
            wgt_var_dict["SM"] = events.weight * ak.fill_none(events.LHEReweightingWeight[:,60],0)
            wgt_var_dict["c2v_1p5"] = events.weight * ak.fill_none(events.LHEReweightingWeight[:,72],0)
            daughters = VV_ndaughters(events)
            n_had = ak.fill_none(daughters['n_had'], 0)
            n_MET = ak.fill_none(daughters['n_MET'], 0)
            mask1 = (n_had == 2) & (n_MET == 2)
            mask2 = (n_had == 2) & (n_MET == 1)
            wgt_var_dict["c2v_1p5_qqNuNu"] = events.weight * ak.fill_none(events.LHEReweightingWeight[:,72],0) * mask1
            wgt_var_dict["c2v_1p5_qqlNu"] = events.weight * ak.fill_none(events.LHEReweightingWeight[:,72],0) * mask2

            # add other cp
        # -----------------------------------------------------------

        # Build cut mask dictionary for each selection
        cut_mask_dict = {}
        if self._n_minus_1:
            # n-1 cutflow: exclude the current cut from the mask
            for sel in self.cutflow:
                if 'objsel' in self.cutflow:
                    exclusive_sels = [k for k in self.cutflow if k != sel] +['objsel'] #+ list(objsel_cf)
                else:
                    exclusive_sels = [k for k in self.cutflow if k != sel] + list(objsel_cf)
                cut_mask_dict[sel] = selections.all(*exclusive_sels)
        else:
            # normal cutflow: cumulative cuts
            cumulative_cuts = []
            for sel in self.cutflow:
                cumulative_cuts.append(sel)
                cut_mask_dict[sel] = selections.all(*cumulative_cuts)

        # Produce histograms for each weight variation and each selection
        for wgt_key, wgt in wgt_var_dict.items():
            for sel, sel_mask in cut_mask_dict.items():
                for dense_axis_name, dense_axis_vals in dense_variables_dict.items():
                    axes_fill_info_dict = {
                        dense_axis_name : ak.fill_none(dense_axis_vals[sel_mask], 0),
                        "weight"        : ak.fill_none(wgt[sel_mask], 0),
                        "process"       : ak.fill_none(events.name[sel_mask], "unknown"), #changed from sample_name to name
                        "category"      : sel,
                        "systematic"    : wgt_key,
                    }
                    # Include year if needed (can be safely added here)
                    if "year" in self.accumulator[dense_axis_name].axes.name:
                        axes_fill_info_dict["year"] = ak.fill_none(events.year[sel_mask], "unknown")#changed from sample_year to year
                        

                    self.accumulator[dense_axis_name].fill(**axes_fill_info_dict)
        return self.accumulator

    def postprocess(self, accumulator):
        return accumulator
