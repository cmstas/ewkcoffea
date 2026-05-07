import os
import json
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
#import awkward as ak
import warnings

warnings.filterwarnings( "ignore", message=".*Missing cross-reference index.*", category=RuntimeWarning)

def make_json(path,name):

    files_lst = []
    tot_events = 0

    # Loop over files in the path
    for dataset_name in os.listdir(path):
        print(dataset_name)
        fullpath_to_dir = os.path.join(path,dataset_name)
        for fname in os.listdir(fullpath_to_dir):
            fullpath = os.path.join(fullpath_to_dir,fname)
            events = NanoEventsFactory.from_root( {fullpath: "Events"}, schemaclass=NanoAODSchema,).events()
            if len(events) == 0:
                print("\nNo events!",fullpath,"\n")
            else:
                files_lst.append(fullpath)
            tot_events = tot_events + len(events)

    print("tot_events:",tot_events)

    files_lst = sorted(files_lst)

    out_dict = {
        "xsec": -999,
        "year": "",
        "treeName": "Events",
        "histAxisName": "",
        "options": "",
        "WCnames": [],
        "nEvents": -999,
        "nGenEvents": -999,
        "nSumOfWeights": -999,
        "isData": False,
        "path": "",
        "nSumOfLheWeights": [],
        "files": files_lst,
    }

    with open(f"input_jsons/{name}.json", "w") as fp:
        json.dump(out_dict, fp, indent=4)


def main():

    files = {
        "input_2l1fj" : "/ceph/cms/store/user/kmohrman/vbsvvh/preselection/merged_2lep_1FJ_r2_20260505160823_2lep_1FJ",
        "input_3l"    : "/ceph/cms/store/user/kmohrman/vbsvvh/preselection/merged_3lep_r2_20260505160723_3lep",
        "input_2l1fj" : "/ceph/cms/store/user/kmohrman/vbsvvh/preselection/merged_2lep_1FJ_r3_20260505161059_2lep_1FJ",
        "input_3l"    : "/ceph/cms/store/user/kmohrman/vbsvvh/preselection/merged_3lep_r3_20260505160942_3lep",
    }

    for name, path in files.items():
        make_json(path,name)


main()
