# Get the file the CI uses, and move it to the directory the JSON expects
printf "\nDownloading root file...\n"
wget -nc http://uaf-10.t2.ucsd.edu/~kmohrman/for_ci/for_wwz/WWZJetsTo4L2Nu_4F_TuneCP5_13TeV-amcatnlo-pythia8_RunIISummer20UL17NanoAODv9-106X_mc2017_realistic_v9-v2_NANOAODSIM_3LepTau_4Lep/output_1.root

# Run the processor via the run_wwz4l.py script
# Note the -x executor argument does not do anything anymore, though should keep it in for now since without it the code tries to default to wq so tries to start packaging up env

time python run_wwz4l.py ../../input_samples/sample_jsons/test_samples/UL17_WWZJetsTo4L2Nu_forCI.json,../../input_samples/sample_jsons/test_samples/UL17_WWZJetsTo4L2Nu_forCI_extra.json -x iterative
