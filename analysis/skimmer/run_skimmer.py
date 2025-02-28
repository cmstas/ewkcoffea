import yaml
import json
import os

import dask
from ndcctools.taskvine import DaskVine

import uproot
from coffea.nanoevents import NanoAODSchema
from coffea.dataset_tools import preprocess, apply_to_fileset


import skimmer_processor as skimming_processor

def read_file(filename):
    with open(filename) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    return content


if __name__ == '__main__':

    ###### Get info from the input jsons ######

    # Get the prefix and json names from the cfg file
    prefix = ""
    json_lst = []
    lines = read_file("samples.cfg")
    for line in lines:
        if line.startswith("#"): continue
        if line == "": continue
        elif line.startswith("prefix:"):
            prefix = line.split()[1]
        else:
            json_lst.append(line)

    # Build a sample dict with all info in the jsons
    samples_dict = {}
    for json_name in json_lst:
        with open(json_name) as jf:
            samples_dict[json_name] = json.load(jf)

    # Make the dataset object the processor wants
    #print(samples_dict)
    dataset_dict = {}
    for json_path in samples_dict.keys():
        tag = json_path.split("/")[-1][:-5]
        dataset_dict[tag] = {}
        dataset_dict[tag]["files"] = {}
        for filename in samples_dict[json_path]["files"]:
            fullpath = prefix+filename
            dataset_dict[tag]["files"][fullpath] = "Events"
    print(dataset_dict)


    ###### Run ######

    # Run preprocess
    print("Running preprocessing")  # To obtain file splitting
    dataset_runnable, _ = preprocess(
        dataset_dict,
        align_clusters=False,
        step_size=100_000,  # You may want to set this to something slightly smaller to avoid loading too much in memory
        files_per_batch=1,
        skip_bad_files=True,
        save_form=False,
    )


    # Run apply_to_fileset
    print("Computing dask task graph")
    skimmed_dict = apply_to_fileset(
        skimming_processor.make_skimmed_events, dataset_runnable, schemaclass=NanoAODSchema
    )

    ### Set up DaskVine stuff ###
    m = DaskVine(
        [9123,9128],
        name=f"coffea-vine-{os.environ['USER']}",
        run_info_path="/blue/p.chang/k.mohrman/vine-run-info",
    )
    #############################



    # Executing task graph and saving
    print("Executing task graph and saving")
    for dataset, skimmed in skimmed_dict.items():
        print("dataset name:",dataset)

        # What does this do
        skimmed = skimming_processor.uproot_writeable(
            skimmed,
        )

        # Reparititioning so that output file contains ~100_000 eventspartition
        skimmed = skimmed.repartition(
            n_to_one=1_000
        )

        # What does this do
        dask_write_out = uproot.dask_write(
            skimmed,
            destination="skimtest/",
            prefix=f"{dataset}/skimmed",
            compute=False,
        )

        # Call compute on the skimmed output
        print(type(dask_write_out))
        dask.compute(
            dask_write_out,
            #scheduler=m.get,
        )

    print("Done!")

