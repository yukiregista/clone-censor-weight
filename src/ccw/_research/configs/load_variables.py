from .scenario1 import load_scenario1_params_and_variables
from .scenario2 import load_scenario2_params_and_variables
from .scenario3 import load_scenario3_params_and_variables

from .experiments.all_experiments import ALL_EXPERIMENT_CONFIGS

def load_params_and_variables(scenario_name: str):
    """
    Load variables from a scenario file.

    Args:
        scenario_name (str): The name of the scenario file to load.

    Returns:
        list: A list of Variable objects defined in the scenario.
    """
    if scenario_name == "scenario1":
        return load_scenario1_params_and_variables()
    elif scenario_name == "scenario2":
        return load_scenario2_params_and_variables()
    elif scenario_name == "scenario3":
        return load_scenario3_params_and_variables()
    else:
        raise ValueError(f"Scenario '{scenario_name}' not found.")

def load_experiment_settings(experiment_name: str):
    stored_configs = ALL_EXPERIMENT_CONFIGS.get(experiment_name)
    if stored_configs is None:
        raise ValueError(f"Experiment '{experiment_name}' not found.")
    configs = dict(stored_configs)
    scenario = configs.get("scenario")
    if scenario is None:
        raise ValueError(f"Scenario not defined in experiment '{experiment_name}'.")
    parameters, variables, dataset_configs = load_params_and_variables(scenario)
    configs.update(dataset_configs)
    return parameters, variables, configs
