#!/usr/bin/env python

import argparse
import json
import time
import cloudpickle
import gzip
import os
import dask
import dask_awkward as dak
from distributed import Client

from coffea.nanoevents import NanoAODSchema
from coffea.nanoevents import NanoEventsFactory

from coffea.dataset_tools import preprocess
from coffea.dataset_tools import apply_to_fileset

import topcoffea.modules.remote_environment as remote_environment

from ndcctools.taskvine import DaskVine

import wwz4l

LST_OF_KNOWN_EXECUTORS = ["futures","work_queue","iterative"]

WGT_VAR_LST = [
    "nSumOfWeights_ISRUp",
    "nSumOfWeights_ISRDown",
    "nSumOfWeights_FSRUp",
    "nSumOfWeights_FSRDown",
    "nSumOfWeights_renormUp",
    "nSumOfWeights_renormDown",
    "nSumOfWeights_factUp",
    "nSumOfWeights_factDown",
    "nSumOfWeights_renormfactUp",
    "nSumOfWeights_renormfactDown",
]

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='You can customize your run')
    parser.add_argument('jsonFiles'        , nargs='?', default='', help = 'Json file(s) containing files and metadata')
    parser.add_argument('--executor','-x'  , default='work_queue', help = 'Which executor to use', choices=LST_OF_KNOWN_EXECUTORS)
    parser.add_argument('--prefix', '-r'   , nargs='?', default='', help = 'Prefix or redirector to look for the files')
    parser.add_argument('--test','-t'       , action='store_true'  , help = 'To perform a test, run over a few events in a couple of chunks')
    parser.add_argument('--pretend'        , action='store_true', help = 'Read json files but, not execute the analysis')
    parser.add_argument('--nworkers','-n'   , default=8  , help = 'Number of workers')
    parser.add_argument('--chunksize','-s' , default=100000, help = 'Number of events per chunk')
    parser.add_argument('--nchunks','-c'   , default=None, help = 'You can choose to run only a number of chunks')
    parser.add_argument('--outname','-o'   , default='plotsTopEFT', help = 'Name of the output file with histograms')
    parser.add_argument('--outpath','-p'   , default='histos', help = 'Name of the output directory')
    parser.add_argument('--treename'       , default='Events', help = 'Name of the tree inside the files')
    parser.add_argument('--do-errors'      , action='store_true', help = 'Save the w**2 coefficients')
    parser.add_argument('--do-systs', action='store_true', help = 'Compute systematic variations')
    parser.add_argument('--split-lep-flavor', action='store_true', help = 'Split up categories by lepton flavor')
    parser.add_argument('--skip-sr', action='store_true', help = 'Skip all signal region categories')
    parser.add_argument('--skip-cr', action='store_true', help = 'Skip all control region categories')
    parser.add_argument('--do-np'  , action='store_true', help = 'Perform nonprompt estimation on the output hist, and save a new hist with the np contribution included. Note that signal, background and data samples should all be processed together in order for this option to make sense.')
    parser.add_argument('--wc-list', action='extend', nargs='+', help = 'Specify a list of Wilson coefficients to use in filling histograms.')
    parser.add_argument('--hist-list', action='extend', nargs='+', help = 'Specify a list of histograms to fill.')
    parser.add_argument('--ecut', default=None  , help = 'Energy cut threshold i.e. throw out events above this (GeV)')
    parser.add_argument('--port', default='9123-9130', help = 'Specify the Work Queue port. An integer PORT or an integer range PORT_MIN-PORT_MAX.')


    args = parser.parse_args()
    jsonFiles  = args.jsonFiles
    prefix     = args.prefix
    executor   = args.executor
    dotest     = args.test
    nworkers   = int(args.nworkers)
    chunksize  = int(args.chunksize)
    nchunks    = int(args.nchunks) if not args.nchunks is None else args.nchunks
    outname    = args.outname
    outpath    = args.outpath
    pretend    = args.pretend
    treename   = args.treename
    do_errors  = args.do_errors
    do_systs   = args.do_systs
    split_lep_flavor = args.split_lep_flavor
    skip_sr    = args.skip_sr
    skip_cr    = args.skip_cr
    wc_lst = args.wc_list if args.wc_list is not None else []

    # Check if we have valid options
    if dotest:
        if executor == "futures":
            nchunks = 2
            chunksize = 10000
            nworkers = 1
            print('Running a fast test with %i workers, %i chunks of %i events'%(nworkers, nchunks, chunksize))
        else:
            raise Exception(f"The \"test\" option is not set up to work with the {executor} executor. Exiting.")


    # Set the threshold for the ecut (if not applying a cut, should be None)
    ecut_threshold = args.ecut
    if ecut_threshold is not None: ecut_threshold = float(args.ecut)

    if executor == "work_queue":
        # construct wq port range
        port = list(map(int, args.port.split('-')))
        if len(port) < 1:
            raise ValueError("At least one port value should be specified.")
        if len(port) > 2:
            raise ValueError("More than one port range was specified.")
        if len(port) == 1:
            # convert single values into a range of one element
            port.append(port[0])

    # Figure out which hists to include
    if args.hist_list == ["ana"]:
        # Here we hardcode a list of hists used for the analysis
        hist_lst = ["njets","lj0pt","ptz"]
    elif args.hist_list == ["cr"]:
        # Here we hardcode a list of hists used for the CRs
        hist_lst = ["lj0pt", "ptz", "met", "ljptsum", "l0pt", "l0eta", "l1pt", "l1eta", "j0pt", "j0eta", "njets", "nbtagsl", "invmass"]
    else:
        # We want to specify a custom list
        # If we don't specify this argument, it will be None, and the processor will fill all hists
        hist_lst = args.hist_list


    ### Load samples from json
    samplesdict = {}
    allInputFiles = []

    def LoadJsonToSampleName(jsonFile, prefix):
        sampleName = jsonFile if not '/' in jsonFile else jsonFile[jsonFile.rfind('/')+1:]
        if sampleName.endswith('.json'): sampleName = sampleName[:-5]
        with open(jsonFile) as jf:
            samplesdict[sampleName] = json.load(jf)
            samplesdict[sampleName]['redirector'] = prefix

    if isinstance(jsonFiles, str) and ',' in jsonFiles:
        jsonFiles = jsonFiles.replace(' ', '').split(',')
    elif isinstance(jsonFiles, str):
        jsonFiles = [jsonFiles]
    for jsonFile in jsonFiles:
        if os.path.isdir(jsonFile):
            if not jsonFile.endswith('/'): jsonFile+='/'
            for f in os.path.listdir(jsonFile):
                if f.endswith('.json'): allInputFiles.append(jsonFile+f)
        else:
            allInputFiles.append(jsonFile)

    # Read from cfg files
    for f in allInputFiles:
        if not os.path.isfile(f):
            raise Exception(f'[ERROR] Input file {f} not found!')
        # This input file is a json file, not a cfg
        if f.endswith('.json'):
            LoadJsonToSampleName(f, prefix)
        # Open cfg files
        else:
            with open(f) as fin:
                print(' >> Reading json from cfg file...')
                lines = fin.readlines()
                for l in lines:
                    if '#' in l:
                        l=l[:l.find('#')]
                    l = l.replace(' ', '').replace('\n', '')
                    if l == '': continue
                    if ',' in l:
                        l = l.split(',')
                        for nl in l:
                            if not os.path.isfile(l):
                                prefix = nl
                            else:
                                LoadJsonToSampleName(nl, prefix)
                    else:
                        if not os.path.isfile(l):
                            prefix = l
                        else:
                            LoadJsonToSampleName(l, prefix)

    fdict = {}
    nevts_total = 0
    for sname in samplesdict.keys():
        redirector = samplesdict[sname]['redirector']
        fdict[sname] = [(redirector+f) for f in samplesdict[sname]['files']]
        samplesdict[sname]['year'] = samplesdict[sname]['year']
        samplesdict[sname]['xsec'] = float(samplesdict[sname]['xsec'])
        samplesdict[sname]['nEvents'] = int(samplesdict[sname]['nEvents'])
        nevts_total += samplesdict[sname]['nEvents']
        samplesdict[sname]['nGenEvents'] = int(samplesdict[sname]['nGenEvents'])
        samplesdict[sname]['nSumOfWeights'] = float(samplesdict[sname]['nSumOfWeights'])
        if not samplesdict[sname]["isData"]:
            for wgt_var in WGT_VAR_LST:
                # Check that MC samples have all needed weight sums (only needed if doing systs)
                if do_systs:
                    if (wgt_var not in samplesdict[sname]):
                        raise Exception(f"Missing weight variation \"{wgt_var}\".")
                    else:
                        samplesdict[sname][wgt_var] = float(samplesdict[sname][wgt_var])
        # Print file info
        print('>> '+sname)
        print('   - isData?      : %s'   %('YES' if samplesdict[sname]['isData'] else 'NO'))
        print('   - year         : %s'   %samplesdict[sname]['year'])
        print('   - xsec         : %f'   %samplesdict[sname]['xsec'])
        print('   - histAxisName : %s'   %samplesdict[sname]['histAxisName'])
        print('   - options      : %s'   %samplesdict[sname]['options'])
        print('   - tree         : %s'   %samplesdict[sname]['treeName'])
        print('   - nEvents      : %i'   %samplesdict[sname]['nEvents'])
        print('   - nGenEvents   : %i'   %samplesdict[sname]['nGenEvents'])
        print('   - SumWeights   : %i'   %samplesdict[sname]['nSumOfWeights'])
        if not samplesdict[sname]["isData"]:
            for wgt_var in WGT_VAR_LST:
                if wgt_var in samplesdict[sname]:
                    print(f'   - {wgt_var}: {samplesdict[sname][wgt_var]}')
        print('   - Prefix       : %s'   %samplesdict[sname]['redirector'])
        print('   - nFiles       : %i'   %len(samplesdict[sname]['files']))
        for fname in samplesdict[sname]['files']: print('     %s'%fname)

    if pretend:
        print('pretending...')
        exit()

    # Extract the list of all WCs, as long as we haven't already specified one.
    if len(wc_lst) == 0:
        for k in samplesdict.keys():
            for wc in samplesdict[k]['WCnames']:
                if wc not in wc_lst:
                    wc_lst.append(wc)

    if len(wc_lst) > 0:
        # Yes, why not have the output be in correct English?
        if len(wc_lst) == 1:
            wc_print = wc_lst[0]
        elif len(wc_lst) == 2:
            wc_print = wc_lst[0] + ' and ' + wc_lst[1]
        else:
            wc_print = ', '.join(wc_lst[:-1]) + ', and ' + wc_lst[-1]
            print('Wilson Coefficients: {}.'.format(wc_print))
    else:
        print('No Wilson coefficients specified')

    processor_instance = wwz4l.AnalysisProcessor(samplesdict,wc_lst,hist_lst,ecut_threshold,do_errors,do_systs,split_lep_flavor,skip_sr,skip_cr)

    if executor == "work_queue":
        executor_args = {
            'master_name': '{}-workqueue-coffea'.format(os.environ['USER']),

            # find a port to run work queue in this range:
            'port': port,

            'debug_log': 'debug.log',
            'transactions_log': 'tr.log',
            'stats_log': 'stats.log',
            'tasks_accum_log': 'tasks.log',

            'environment_file': remote_environment.get_environment(
                extra_pip=["mt2","xgboost"],
                extra_pip_local = {"ewkcoffea": ["ewkcoffea", "setup.py"]},
            ),
            'extra_input_files': ["wwz4l.py"],

            'retries': 5,

            # use mid-range compression for chunks results. 9 is the default for work
            # queue in coffea. Valid values are 0 (minimum compression, less memory
            # usage) to 16 (maximum compression, more memory usage).
            'compression': 9,

            # automatically find an adequate resource allocation for tasks.
            # tasks are first tried using the maximum resources seen of previously ran
            # tasks. on resource exhaustion, they are retried with the maximum resource
            # values, if specified below. if a maximum is not specified, the task waits
            # forever until a larger worker connects.
            'resource_monitor': True,
            'resources_mode': 'auto',

            # this resource values may be omitted when using
            # resources_mode: 'auto', but they do make the initial portion
            # of a workflow run a little bit faster.
            # Rather than using whole workers in the exploratory mode of
            # resources_mode: auto, tasks are forever limited to a maximum
            # of 8GB of mem and disk.
            #
            # NOTE: The very first tasks in the exploratory
            # mode will use the values specified here, so workers need to be at least
            # this large. If left unspecified, tasks will use whole workers in the
            # exploratory mode.
            # 'cores': 1,
            # 'disk': 8000,   #MB
            # 'memory': 10000, #MB

            # control the size of accumulation tasks. Results are
            # accumulated in groups of size chunks_per_accum, keeping at
            # most chunks_per_accum at the same time in memory per task.
            'chunks_per_accum': 25,
            'chunks_accum_in_mem': 2,

            # terminate workers on which tasks have been running longer than average.
            # This is useful for temporary conditions on worker nodes where a task will
            # be finish faster is ran in another worker.
            # the time limit is computed by multipliying the average runtime of tasks
            # by the value of 'fast_terminate_workers'.  Since some tasks can be
            # legitimately slow, no task can trigger the termination of workers twice.
            #
            # warning: small values (e.g. close to 1) may cause the workflow to misbehave,
            # as most tasks will be terminated.
            #
            # Less than 1 disables it.
            'fast_terminate_workers': 0,

            # print messages when tasks are submitted, finished, etc.,
            # together with their resource allocation and usage. If a task
            # fails, its standard output is also printed, so we can turn
            # off print_stdout for all tasks.
            'verbose': True,
            'print_stdout': False,
        }

    # Run the processor and get the output
    tstart = time.time()

    ####################################3
    ### coffea2023 ###

    # Get fileset
    fileset = {}
    for name, fpaths in fdict.items():
        fileset[name] = {}
        fileset[name]["files"] = {}
        for fpath in fpaths:
            fileset[name]["files"][fpath] = {"object_path": "Events"}
            fileset[name]["metadata"] = {"dataset": name}
    print(fileset)
    print("Number of datasets:",len(fdict))


    #### Try with distributed Client ####

    #with dask.config.set({"scheduler": "sync"}): # Single thread
    #with Client() as _: # distributed Client scheduler
    with Client() as client:

        # Run preprocess
        print("\nRunning preprocess...")
        dataset_runnable, dataset_updated = preprocess(
            fileset,
            maybe_step_size=50_000,
            align_clusters=False,
            files_per_batch=1,
            #skip_bad_files=True,
            #calculate_form=True,
        )

        # Run apply_to_fileset
        print("\nRunning apply_to_fileset...")
        histos_to_compute, reports = apply_to_fileset(
            processor_instance,
            dataset_runnable,
            uproot_options={"allow_read_errors_with_report": True}
        )

        # Check columns to be read
        print("\nRunning necessary_columns...")
        columns_read = dak.necessary_columns(histos_to_compute[list(histos_to_compute.keys())[0]])
        print(columns_read)

        # Compute
        print("\nRunning compute...")
        output_futures, report_futures = {}, {}
        for key in histos_to_compute:
            output_futures[key], report_futures[key] = client.compute((histos_to_compute[key], reports[key],)) # , scheduler=taskvine

        coutputs, creports = client.gather((output_futures, report_futures,))



    ### Task vine testing ###
    do_tv = 0
    if do_tv:

        fdict = {"UL17_WWZJetsTo4L2Nu_forCI": ["/home/k.mohrman/coffea_dir/migrate_to_coffea2023_repo/ewkcoffea/analysis/wwz/output_1.root"]}

        # Create dict of events objects
        print("Number of datasets:",len(fdict))
        events_dict = {}
        for name, fpaths in fdict.items():
            events_dict[name] = NanoEventsFactory.from_root(
                {fpath: "/Events" for fpath in fpaths},
                schemaclass=NanoAODSchema,
                metadata={"dataset": name},
            ).events()

        # Get and compute the histograms
        histos_to_compute = {}
        for json_name in fdict.keys():
            print(f"Getting histos for {json_name}")
            histos_to_compute[json_name] = processor_instance.process(events_dict[json_name])

        m = DaskVine([9123,9128], name=f"coffea-vine-{os.environ['USER']}")

        print("Compute histos")
        #output = dask.compute(histos_to_compute)[0] # Output of dask.compute is a tuple
        coutputs = dask.compute(histos_to_compute, scheduler=m.get, resources={"cores": 1}, resources_mode=None, lazy_transfers=True)


    dt = time.time() - tstart

    # Save the output
    if not os.path.isdir(outpath): os.system("mkdir -p %s"%outpath)
    out_pkl_file = os.path.join(outpath,outname+".pkl.gz")
    print(f"\nSaving output in {out_pkl_file}...")
    with gzip.open(out_pkl_file, "wb") as fout:
        cloudpickle.dump(coutputs, fout)
    print("Done!")
