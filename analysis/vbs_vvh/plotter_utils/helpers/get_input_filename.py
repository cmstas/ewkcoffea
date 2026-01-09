"""
get input pkl name by the following methods
- if a file name or multiple file names in .pkl.gz is provided, directly use it
- elif a directory is provided, search for .pkl.gz in that directory (not including subdir)
- elif none is provided, but project and cutflow is provided, guess the file name
- else the code should complain
"""
import os

def pkl_name_from_project(project_name,cutname,n_minus_1=False):
    """
    get individual file name from a given project name and cut name
    """
    from config.paths import default_output_dir
    if not n_minus_1:
        return f'{default_output_dir}/histos/{project_name}/histos/{cutname}_histos.pkl.gz'
    else:
        return f'{default_output_dir}/histos/{project_name}/histos/{cutname}_histos_m.pkl.gz'
    
def get_all_pkl_in_folder(dirpath):
    import os
    """
    Return an array of filenames for all .pkl.gz files directly inside `dirpath`
    Does NOT search subdirectories.
    """
    if not os.path.isdir(dirpath):
        raise ValueError(f"Not a directory: {dirpath}")

    return [
        os.path.join(dirpath,fname)
        for fname in os.listdir(dirpath)
        if fname.endswith(".pkl.gz")
        and os.path.isfile(os.path.join(dirpath, fname))
    ]

