# This script reproduces the reference yields file that the CI compares against
# Run this script when you want to update the reference file

# Get the file the CI uses, and move it to the directory the JSON expects
printf "\nDownloading root file...\n"
wget -nc http://uaf-10.t2.ucsd.edu/~kmohrman/large_files_no_backup/for_ci/vvh/feb10_2026/merged.root

# Run the processor
printf "\nRunning the processor...\n"
time python run_analysis.py ../../input_samples/sample_jsons/test_samples/UL16APV_WWHSS_forCI.json -x iterative -o new_ref_histos -p semilep_nano

# Make the JSON file of the yields
printf "\nMaking the yields JSON file...\n"
python check_vvh_hists.py histos/new_ref_histos.pkl.gz -j -o vvh_yld_for_ci_new

# Compare the JSON file of the yields
printf "\nCompare the new yields JSON file to old ref...\n"
python ../../ewkcoffea/scripts/comp_json_yields.py vvh_yld_for_ci_new.json ref_for_ci/vvh_yld_for_ci.json -t1 "New yields" -t2 "Old ref yields"

# Replace the reference yields with the new reference yields
printf "\nReplacing ref yields JSON with new file...\n"
mv vvh_yld_for_ci_new.json ref_for_ci/vvh_yld_for_ci.json
printf "\n\nDone.\n\n"
