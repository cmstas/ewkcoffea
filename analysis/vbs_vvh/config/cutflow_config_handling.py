import yaml
import awkward as ak

def get_cutflow(yaml_file, cutflow=None):
    """
    Load cutflow from a YAML file.
    If cutflow is None, return the full dictionary with lambdas.
    """
    
    def make_lambda(expr: str):
        """
        Turn a YAML expression into a lambda for Coffea cutflow.
        """
        return eval(f"lambda events, var, obj: {expr}", {"ak": ak})
    
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)

    cutflow_dict = {}

    for cf_name, cf_steps in config.items():
        step_dict = {}
        for step, expr in cf_steps.items():
            # Always store as a 1‑tuple of lambdas
            step_dict[step] = make_lambda(expr)
        cutflow_dict[cf_name] = step_dict

    if cutflow is None:
        print(f"read cutflow from project: {yaml_file}")
        return cutflow_dict
    else:
        print(f"read cutflow from project: {yaml_file}/{cutflow}")
        return cutflow_dict[cutflow]
    

    
def export_cutflow_to_yaml(cutflow_dict, yaml_file):
    import inspect
    """
    Convert a cutflow_dict with lambda functions into YAML file with string expressions.
    """
    yaml_dict = {}
    for cf_name, cf_steps in cutflow_dict.items():
        yaml_dict[cf_name] = {}
        for step_name, func in cf_steps.items():
            # Extract the source of the lambda function
            try:
                src = inspect.getsource(func).strip()
                # Example: 'lambda events, var, obj: events.Pass_MetTriggers == 1'
                expr = src.split(":", 1)[1].strip()  # take only the expression part
            except (OSError, TypeError):
                expr = "<unknown>"
            yaml_dict[cf_name][step_name] = expr

    # Save to YAML
    with open(yaml_file, 'w') as f:
        yaml.dump(yaml_dict, f, sort_keys=False)

    print(f"Cutflow exported to {yaml_file}")
