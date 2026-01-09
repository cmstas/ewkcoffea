# for producing process map (sample_name.csv) that links process to bkg type 
# use for plotting

import os
import json
import csv
import argparse
import subprocess
from collections import defaultdict

# Constants
years = ['2016preVFP', '2016postVFP', '2017', '2018']
categories = ['sig', 'bkg', 'data']
input_json_template = "input_{}_{}.json"
output_csv = "sample_names.csv"

sample_colour_mpl = {
    "data": "blue",
    "sig": "red",
    "WWH_OS": "crimson",
    "WWH_SS": "lightcoral",
    "WZH": "orangered",
    "ZZH": "tomato",
    "bkg_total": "dimgray",
    "DY": "palegreen",
    "ttbar": "mediumturquoise",
    "ttbar_had": "powderblue",
    "ttbar_SL": "deepskyblue",
    "ttx": "cadetblue",
    "ttX": "cadetblue",
    "ST": "lightsteelblue",
    "WJets": "darkorange",
    "WJet": "darkorange",
    "ZJets": "lightyellow",
    "ZJet": "lightyellow",
    "EWKV": "skyblue",
    "EWK": "skyblue",
    "QCD": "lightcyan",
    "bosons": "sandybrown",
    "Other": "orchid",
}


def run_makeConfig():
    for cat in categories:
        for year in years:
            print(f"Running: python3 etc/makeConfig.py --categories {cat} --sample_year {year}")
            subprocess.run(["python3", "etc/makeConfig.py", "--categories", cat, "--sample_year", year], check=True)


def extract_samples():
    signal_tags = ['WWH_OS', 'WWH_SS', 'WZH', 'ZZH']
    sample_list = []

    for cat in categories:
        for year in years:
            json_file = input_json_template.format(year, cat)
            if not os.path.isfile(json_file):
                print(f"Missing {json_file}, skipping.")
                continue

            with open(json_file) as f:
                json_data = json.load(f)

            samples = json_data.get("samples", {})
            for sample_key, sample_info in samples.items():
                metadata = sample_info.get("metadata", {})

                sample_year = metadata.get("sample_year", year)
                sample_category = metadata.get("sample_category", cat)
                sample_name = metadata.get("sample_name", sample_key)

                # Determine sample_type
                if sample_category == 'sig':
                    matched_type = next((tag for tag in signal_tags if tag in sample_key), "unknown_sig")
                    sample_type = matched_type
                else:
                    sample_type = metadata.get("sample_type", "unknown")

                sample_list.append({
                    "sample_year": sample_year,
                    "sample_category": sample_category,
                    "sample_type": sample_type,
                    "sample_name": sample_name,
                    "xsec": metadata.get("xsec", -1),
                    "lumi": metadata.get("lumi", -1),
                    "sumws": metadata.get("sumws", -1),
                })
    return sample_list

def assign_codes_and_colours(sample_list):
    # Assign codes
    code_prefix = {'sig': 1, 'bkg': 2, 'data': 3}
    sample_code_counter = defaultdict(dict)
    final_sample_list = []

    current_counter = defaultdict(int)
    # sort by sample_category, then sample_type
    sample_list.sort(key=lambda x: (code_prefix[x['sample_category']], x['sample_type'], x['sample_name']))

    for sample in sample_list:
        cat = sample['sample_category']
        name = sample['sample_name']
        if name not in sample_code_counter[cat]:
            sample_code_counter[cat][name] = current_counter[cat]
            current_counter[cat] += 1
        sample_code = f"{code_prefix[cat]}{sample_code_counter[cat][name]}"
        sample['sample_code'] = sample_code

        # Determine plotting color
        sample_type = sample["sample_type"]
        sample["plotting_colour"] = sample_colour_mpl.get(sample_type, "black")
        final_sample_list.append(sample)

    return final_sample_list


def write_csv(sample_list):
    header = ["sample_year", "sample_category", "sample_type", "sample_name",
              "xsec", "lumi", "sumws", "sample_code", "plotting_colour"]
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()
        for sample in sample_list:
            writer.writerow(sample)


def write_sample_code_map(sample_list, output_path="sample_code.csv"):
    # Use a set to avoid duplicates
    seen = set()
    rows = []

    for sample in sample_list:
        key = (sample["sample_category"], sample["sample_type"], sample["sample_name"], sample["sample_code"], )
        if key not in seen:
            seen.add(key)
            rows.append({"sample_category": sample["sample_category"], "sample_type": sample["sample_type"], "sample_name": sample["sample_name"], "sample_code": sample["sample_code"]})

    # Sort by code
    rows.sort(key=lambda x: int(x["sample_code"]))

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["sample_code","sample_category","sample_type","sample_name"])
        writer.writeheader()
        writer.writerows(rows)

def main():

    #run_makeConfig()

    print("Extracting samples from JSON...")
    raw_sample_list = extract_samples()

    print("Assigning codes and plotting colors...")
    processed_sample_list = assign_codes_and_colours(raw_sample_list)

    print(f"Writing to {output_csv}...")
    write_csv(processed_sample_list)


    print("Writing sample_code.csv...")
    write_sample_code_map(processed_sample_list)

    print("Done.")


if __name__ == "__main__":
    main()