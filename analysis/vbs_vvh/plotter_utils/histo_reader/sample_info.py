# from sample_name.csv group sample_name into sig/bkg/data and get plotting colors

import csv
from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Set, Optional


DEFAULT_SAMPLE_CSV = "/home/users/pyli/projects/analysis_VVH/coffea/ewkcoffea/analysis/vbs_vvh/config/sample_names.csv"

#get list of years
def get_all_years(csv_path=None):
    """
    get set of year in the sample_name.csv
    (should be 2016preVFP,... for run2)
    since years are usually summed, order is not considered
    """
    if csv_path is None:
        csv_path = DEFAULT_SAMPLE_CSV
    
    all_years = set()
    with open(csv_path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            sample_year = row["sample_year"].strip()
            all_years.add(sample_year)
    return all_years

@dataclass
class ProcessMap:
    """
    From sample_name.csv, get 
    - the list of processes (sample_name) for signal, bkg, data
    - the list of bkg groups and process name under each group
    - color for plotting
    
    :var csv_path: Description
    :vartype csv_path: str
    """
    signal: List[str]
    background: List[str]
    data: List[str]

    # extra, useful metadata
    background_groups: Dict[str, List[str]]
    background_colors: Dict[str, str]

    @classmethod
    def from_csv(
        cls,
        csv_path: Optional[str] = None
    ):
        """
        Build ProcessMap from sample CSV.

        Parameters
        ----------
        csv_path : str, optional
            Path to sample CSV (uses default if None)
        """
        if csv_path is None:
            csv_path = DEFAULT_SAMPLE_CSV

        grp_dict = defaultdict(set) 
        bkg_color_map = {}

        with open(csv_path, newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                cat = row["sample_category"].strip().lower()
                sample_type = row["sample_type"].strip()
                sample_name = row["sample_name"].strip()
                color = row["plotting_colour"].strip()

                if cat == "sig":
                    grp_dict["Signal"].add(sample_name)

                elif cat == "bkg":
                    grp_dict[sample_type].add(sample_name)
                    bkg_color_map.setdefault(sample_type, color)

                elif cat == "data":
                    grp_dict["Data"].add(sample_name)

        signal = list(grp_dict.get("Signal"))
        data = list(grp_dict.get("Data"))
        background_groups = {k: list(v) for k, v in grp_dict.items() if k not in ("Signal", "Data")}
        background = list(p for procs in background_groups.values() for p in procs)

        return cls(
            signal=signal,
            background=background,
            data=data,
            background_groups=background_groups,
            background_colors=bkg_color_map,
        )