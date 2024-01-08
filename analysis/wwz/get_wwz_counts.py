import argparse
import json
import pickle
import gzip

# This script opens a pkl file of histograms produced by wwz processor
# Reads the histograms and dumps out the event counts
# Example usage: python get_yld_check.py -f histos/tmp_histo.pkl.gz


# Get number of evens in SRs
def get_counts(histos_dict):

    wwz_sync_sample = 'UL17_WWZJetsTo4L2Nu'
    wwz_sync_sample_lst = ['UL17_WWZJetsTo4L2Nu']
    #wwz_sync_sample_lst = ['UL16_WWZJetsTo4L2Nu','UL16APV_WWZJetsTo4L2Nu','UL17_WWZJetsTo4L2Nu','UL18_WWZJetsTo4L2Nu']
    #wwz_sync_sample_lst = ['UL16_VHnobb','UL16APV_VHnobb','UL17_VHnobb','UL18_VHnobb']

    out_dict = {}
    out_dict[wwz_sync_sample] = {}

    # Get object multiplicity counts (nleps, njets, nbtags)
    ojb_lst = ["nleps_counts","njets_counts","nbtagsl_counts"]
    for obj in ojb_lst:
        nobjs_hist = sum(histos_dict[obj][{"category":"all_events","process":wwz_sync_sample_lst}].values(flow=True))
        tot_objs = 0
        for i,v in enumerate(nobjs_hist):
            # Have to adjust for the fact that first bin is underflow, second bin is 0 objects
            # Note this breaks if there are objects with higher multiplicity than n+1 (since we don't know how to weight them in the sum, as they all just fall into the (n+1)th bin)
            tot_objs = tot_objs + (i-1.)*v
        out_dict[wwz_sync_sample][obj] = (tot_objs,None) # Save err as None
        #print("\ntotobj",obj,tot_objs)

    # Look at the event counts in one histo (e.g. njets)
    dense_axis = "njets_counts"
    for cat_name in histos_dict[dense_axis].axes["category"]:
        val = sum(sum(histos_dict[dense_axis][{"category":cat_name,"process":wwz_sync_sample_lst}].values(flow=True)))
        out_dict[wwz_sync_sample][cat_name+"_counts"] = (val,None) # Save err as None

    # Look at the yields in one histo (e.g. njets)
    dense_axis = "njets"
    for cat_name in histos_dict[dense_axis].axes["category"]:
        val = sum(sum(histos_dict[dense_axis][{"category":cat_name,"process":wwz_sync_sample_lst}].values(flow=True)))
        out_dict[wwz_sync_sample][cat_name] = (val,None) # Save err as None


    return out_dict


# Restructure hists so samples are in category axis
def restructure_hist(in_dict):

    dense_axes = list(list(in_dict.values())[0].keys())
    restructured_hist = {}
    for dense_axis in dense_axes:
        for dataset_name in in_dict.keys():
            if dense_axis not in restructured_hist:
                restructured_hist[dense_axis] = in_dict[dataset_name][dense_axis]
            else:
                restructured_hist[dense_axis] += in_dict[dataset_name][dense_axis]
    return restructured_hist


def main():

    # Set up the command line parser
    parser = argparse.ArgumentParser()
    parser.add_argument("pkl_file_path", help = "The path to the pkl file")
    parser.add_argument("-o", "--output-path", default=".", help = "The path the output files should be saved to")
    parser.add_argument("-n", "--output-name", default="counts_wwz_sync", help = "A name for the output directory")
    args = parser.parse_args()

    # Get the counts from the input hiso
    histo_dict = pickle.load(gzip.open(args.pkl_file_path))
    histo_dict = restructure_hist(histo_dict)

    counts_dict = get_counts(histo_dict)

    # Print the counts
    print("\nCounts:")
    for proc in counts_dict.keys():
        for cat,val in counts_dict[proc].items(): print(f"  {cat}: {val[0]}")

    # Dump counts dict to json
    if "json" not in args.output_name: output_name = args.output_name + ".json"
    else: output_name = args.output_name
    with open(output_name,"w") as out_file: json.dump(counts_dict, out_file, indent=4)
    print(f"\nSaved json file: {output_name}\n")


if __name__ == "__main__":
    main()

