import pickle, gzip
import json
import os, sys

test_file_path = '/home/users/pyli/projects/analysis_VVH/coffea/ewkcoffea/analysis/vbs_vvh/histos/test/histos/test1_hists.pkl.gz'
def test_file():
    print(f'importing test file from {test_file_path}')
    return pickle.load(gzip.open(test_file_path))