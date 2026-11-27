import os
import json
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
#import awkward as ak
import warnings

warnings.filterwarnings( "ignore", message=".*Missing cross-reference index.*", category=RuntimeWarning)

def make_json(path,name,era=None):

    files_lst = []
    tot_events = 0

    # Loop over files in the path
    for dataset_name in os.listdir(path):
        print(dataset_name)

        # Only include the relevant era
        if "_Run2022" in dataset_name: continue
        if "_Run2023" in dataset_name: continue
        if era is not None:
            is_2025 = ("Summer24for2025" in dataset_name) or ("_Run2025" in dataset_name)
            is_2024 = (not is_2025) and (("RunIII2024Summer24" in dataset_name) or ("_Run2024" in dataset_name))
            if era == "2024" and not is_2024: continue
            if era == "2025" and not is_2025: continue

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
        "input_zzz_r2"  : ("/cmsuf/data/store/user/kmohrman/vbs_vvh_rdf/zzz_nov27_ref/zzz_r2/", None),
        "input_zzz_r24" : ("/cmsuf/data/store/user/kmohrman/vbs_vvh_rdf/zzz_nov27_ref/zzz_r3/", "2024"),
        "input_zzz_r25" : ("/cmsuf/data/store/user/kmohrman/vbs_vvh_rdf/zzz_nov27_ref/zzz_r3/", "2025"),
    }

    for name, (path,era) in files.items():
        make_json(path,name,era)


main()
