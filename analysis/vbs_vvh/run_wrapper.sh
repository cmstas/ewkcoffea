# Example run commands

### Example for the from-Nano processor ###
##time python run_analysis.py input_cfg_r2.cfg -x futures -n 64 -o vvh_hists_from_nano -p semilep_nano -s 30000


### Example for the from-RDF processor ###

# Examples of prefixes
#PREFIX="/ceph/cms/store/user/kmohrman/vbsvvh/preselection/merged_2lep_1FJ_r2_20260422112528_2lep_1FJ/"
#PREFIX="/cmsuf/data/store/user/phchang/vvh/3l/"
#time python run_analysis.py input_forcoffea_fromrdf/input_2l1fj.json -x futures -n 32 -o vvh_2l -r $PREFIX -p semilep -s 20000 --hist-list njets njets_counts

# Examples for 2l, 3l, run 2, run 3 (the paths assume RDF)
time python run_analysis.py input_jsons/input_2l1fj_r2.json -x futures -n 32 -o vvh_2l_r2 -p semilep -s 20000 --hist-list njets njets_counts
time python run_analysis.py input_jsons/input_2l1fj_r3.json -x futures -n 32 -o vvh_2l_r3 -p semilep -s 20000 --hist-list njets njets_counts
time python run_analysis.py input_jsons/input_3l_r2.json    -x futures -n 32 -o vvh_3l_r2 -p semilep -s 20000 --hist-list njets njets_counts
time python run_analysis.py input_jsons/input_3l_r3.json    -x futures -n 32 -o vvh_3l_r3 -p semilep -s 20000 --hist-list njets njets_counts


