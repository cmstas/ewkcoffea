
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)

#scripts
cutflow_yamls_dir =  os.path.join(base_dir,'config','cutflows')

#var stuffs
basic_cutflow = f'{cutflow_yamls_dir}/test.yaml'
from config.config_handling import get_cutflow
objsel_cf = get_cutflow(basic_cutflow,'test1')
default_cutflow_yaml = f'{cutflow_yamls_dir}/all.yaml'