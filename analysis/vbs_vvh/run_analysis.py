#!/usr/bin/env python

import argparse
import json
import time
import cloudpickle
import gzip
import os
import socket
from coffea import processor
from coffea.nanoevents import NanoAODSchema
NanoAODSchema.warn_missing_crossrefs = False
import topcoffea.modules.remote_environment as remote_environment

LST_OF_KNOWN_EXECUTORS = ["futures","work_queue","iterative"]
LST_OF_KNOWN_PROCESSORS = ["1lep1jf","1lep1jf_nano"]


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='You can customize your run')
    parser.add_argument('jsonFiles'        , nargs='?', default='', help = 'Json file(s) containing files and metadata')
    parser.add_argument('--executor','-x'  , default='work_queue', help = 'Which executor to use', choices=LST_OF_KNOWN_EXECUTORS)
    parser.add_argument('--prefix', '-r'   , nargs='?', default='', help = 'Prefix or redirector to look for the files')
    parser.add_argument('--nworkers','-n'   , default=8  , help = 'Number of workers')
    parser.add_argument('--chunksize','-s' , default=100000, help = 'Number of events per chunk')
    parser.add_argument('--nchunks','-c'   , default=None, help = 'You can choose to run only a number of chunks')
    parser.add_argument('--outname','-o'   , default='plotsTopEFT', help = 'Name of the output file with histograms')
    parser.add_argument('--outpath',         default='histos', help = 'Name of the output directory')
    parser.add_argument('--treename'       , default='Events', help = 'Name of the tree inside the files')
    parser.add_argument('--do-systs', action='store_true', help = 'Compute systematic variations')
    parser.add_argument('--skip-obj-systs', action='store_true', help = 'Skip systematic variations that impact obj kinematics')
    parser.add_argument('--skip-sr', action='store_true', help = 'Skip all signal region categories')
    parser.add_argument('--skip-cr', action='store_true', help = 'Skip all control region categories')
    parser.add_argument('--siphon' , action='store_true', help = 'Siphon BDT data')
    parser.add_argument('--wc-list', action='extend', nargs='+', help = 'Specify a list of Wilson coefficients to use in filling histograms.')
    parser.add_argument('--hist-list', action='extend', nargs='+', help = 'Specify a list of histograms to fill.')
    parser.add_argument('--port', default='9123-9130', help = 'Specify the Work Queue port. An integer PORT or an integer range PORT_MIN-PORT_MAX.')
    parser.add_argument('--processor', '-p', default='1lep1jf', help = 'Which processor to execute', choices=LST_OF_KNOWN_PROCESSORS)


    args = parser.parse_args()
    jsonFiles  = args.jsonFiles
    prefix     = args.prefix
    executor   = args.executor
    nworkers   = int(args.nworkers)
    chunksize  = int(args.chunksize)
    nchunks    = int(args.nchunks) if not args.nchunks is None else args.nchunks
    outname    = args.outname
    outpath    = args.outpath
    treename   = args.treename
    do_systs   = args.do_systs
    skip_obj_systs = args.skip_obj_systs
    siphon     = args.siphon
    skip_sr    = args.skip_sr
    skip_cr    = args.skip_cr
    wc_lst = args.wc_list if args.wc_list is not None else []

    # Import the proper processor, based on option specified
    if args.processor == "1lep1jf":
        import analysis_processor_1l1fj  as analysis_processor
    elif args.processor == "1lep1jf_nano":
        import analysis_processor_1l1fj_fromnano as analysis_processor

    # Check that if on UF login node, we're using WQ
    hostname = socket.gethostname()
    if "login" in hostname:
        # We are on a UF login node, better be using WQ
        # Note if this ends up catching more than UF, can also check for "login"&"ufhpc" in name
        if (executor != "work_queue"):
            raise Exception(f"\nError: We seem to be on a UF login node ({hostname}). If running from here, need to run with WQ.")


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
    if args.hist_list == ["few"]:
        # Here we hardcode a reduced list of a few hists
        hist_lst = ["j0pt", "njets", "njets_counts", "nbtagsl", "nleps", "met", "l0pt", "abs_pdgid_sum"]
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

    flist = {}
    nevts_total = 0
    for sname in samplesdict.keys():
        redirector = samplesdict[sname]['redirector']
        flist[sname] = [(redirector+f) for f in samplesdict[sname]['files']]
        samplesdict[sname]['year'] = samplesdict[sname]['year']
        samplesdict[sname]['xsec'] = float(samplesdict[sname]['xsec'])
        samplesdict[sname]['nEvents'] = int(samplesdict[sname]['nEvents'])
        nevts_total += samplesdict[sname]['nEvents']
        samplesdict[sname]['nGenEvents'] = int(samplesdict[sname]['nGenEvents'])
        samplesdict[sname]['nSumOfWeights'] = float(samplesdict[sname]['nSumOfWeights'])
        if not samplesdict[sname]["isData"]:
            # Check that MC samples have all needed weight sums (only needed if doing systs)
            if do_systs:
                if ("nSumOfLheWeights" not in samplesdict[sname]):
                    raise Exception(f"Sample is missing scale variations: {sname}")
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
            if "nSumOfLheWeights" in samplesdict[sname]:
                print(f'   - nSumOfLheWeights : {samplesdict[sname]["nSumOfLheWeights"]}')
        print('   - Prefix       : %s'   %samplesdict[sname]['redirector'])
        print('   - nFiles       : %i'   %len(samplesdict[sname]['files']))
        for fname in samplesdict[sname]['files']: print('     %s'%fname)

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

    processor_instance = analysis_processor.AnalysisProcessor(samplesdict,wc_lst,hist_lst,do_systs,skip_obj_systs,skip_sr,skip_cr,siphon_bdt_data=siphon)

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
                extra_conda=["root"],
                extra_pip=["mt2","xgboost"],
                extra_pip_local = {"ewkcoffea": ["ewkcoffea", "setup.py"]},
            ),
            'extra_input_files': ["analysis_processor.py"],

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

    if executor == "futures":
        exec_instance = processor.FuturesExecutor(workers=nworkers, merging=(1, 30, 10000))
        runner = processor.Runner(exec_instance, schema=NanoAODSchema, chunksize=chunksize, maxchunks=nchunks)
    elif executor == "iterative":
        exec_instance = processor.IterativeExecutor()
        runner = processor.Runner(exec_instance, schema=NanoAODSchema, chunksize=chunksize, maxchunks=nchunks)
    elif executor ==  "work_queue":
        executor = processor.WorkQueueExecutor(**executor_args)
        runner = processor.Runner(executor, schema=NanoAODSchema, chunksize=chunksize, maxchunks=nchunks, skipbadfiles=False, xrootdtimeout=180)

    output = runner(flist, treename, processor_instance)

    dt = time.time() - tstart

    if executor == "work_queue":
        print('Processed {} events in {} seconds ({:.2f} evts/sec).'.format(nevts_total,dt,nevts_total/dt))

    #nbins = sum(sum(arr.size for arr in h._sumw.values()) for h in output.values() if isinstance(h, hist.Hist))
    #nfilled = sum(sum(np.sum(arr > 0) for arr in h._sumw.values()) for h in output.values() if isinstance(h, hist.Hist))
    #print("Filled %.0f bins, nonzero bins: %1.1f %%" % (nbins, 100*nfilled/nbins,))

    if executor == "futures":
        print("Processing time: %1.2f s with %i workers (%.2f s cpu overall)" % (dt, nworkers, dt*nworkers, ))

    # Save the output
    if not os.path.isdir(outpath): os.system("mkdir -p %s"%outpath)
    out_pkl_file = os.path.join(outpath,outname+".pkl.gz")
    print(f"\nSaving output in {out_pkl_file}...")
    with gzip.open(out_pkl_file, "wb") as fout:
        cloudpickle.dump(output, fout)
    print("Done!")
