import csv,json
import os

from plotter_utils.helpers.get_input_filename import get_all_pkl_in_folder

import logging
logger = logging.getLogger(__name__)

def save_array_to_csv(arr, file_name):
    import csv
    """
    Save a 2D array to a CSV file.
    Assumes arr[0] is the header row, arr[1:] are data rows.
    """
    with open(file_name, mode='w', newline='') as f:
        writer = csv.writer(f)
        for row in arr:
            writer.writerow(row)

def save_yield_dict(outdict, file_name):
    """
    Save yield dictionary to JSON.

    Expected input structure:
      outdict[cutname][proc] = (yield, variance)

    Output JSON structure:
      outdict[cutname][proc] = [yield, error]

    where error = sqrt(variance) if take_sqrt_variance=True
    """

    json_out = {}

    for cutname, proc_dict in outdict.items():
        json_out[cutname] = {}
        for proc, yv in proc_dict.items():
            if yv is None:
                json_out[cutname][proc] = None
                continue

            yld, err = yv
            json_out[cutname][proc] = [yld, err]

    with open(file_name, "w") as f:
        json.dump(json_out, f, indent=2)

    logger.info("Saved yield JSON to %s", file_name)
            
def resolve_input_list(inputs):
    """
    unpack any dir and get the list of pkl.gz files
    """
    input_list = []
    for name in inputs:
        if name.endswith(".pkl.gz"):
            input_list.append(name)
        elif os.path.isdir(name):
            files = get_all_pkl_in_folder(name)
            input_list+=files
            if len(files)==0:
                logger.warning(f'input dir does not contain pkl.gz files {name}')
        else:
            logger.warning(f'input file is not pkl.gz file nor dir: {name}')
    return input_list

def resolve_outdir(output,input_list,default_output_dirname):
    """
    figure out output folder name
    """
    if output is None:
        outname = os.path.dirname(input_list[0])
        outdir = os.path.join(outname,default_output_dirname)
    else:
        outdir = output
    return outdir

def print_yield_nicely(yield_dict,indent = 10):
    """
    yield_dict can be:
      - {proc_grp: (yield, variance)}
      - {proc_grp: {proc: (yield, variance)}}
    """

    def format_yield(y):
        if y is None:
            return "None"
        if isinstance(y, (list, tuple)) and len(y) == 2:
            return f"{y[0]:.3f} +/- {y[1]:.3f}"
        if isinstance(y, float):
            return f"{y:.3f} +/- ?"
        return f"{y}"

    for proc_grp, d in yield_dict.items():
        if isinstance(d, dict):
            logger.info("%s", proc_grp)
            for proc, y in d.items():
                # align process name to max_proc_len
                logger.info("    - %-*s : %s", indent, proc, format_yield(y))
        else:
            logger.info("    %-*s : %s", indent, proc_grp, format_yield(d))
    