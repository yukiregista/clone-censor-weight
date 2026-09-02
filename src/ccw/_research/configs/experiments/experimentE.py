from ccw import CensoringModel
from ccw._research.data_generation.scenarios.scenario import ScenarioVariables
from ccw._research.configs.experiments.experiment_common import default_preprocess_pipeline


def load_experiment_configs():
    configs = {
        "cut_data_after_outcome": True,
        "ipw_explanatory_formula": lambda column_maps: {
            "control": {
                'all': f'{ScenarioVariables.AGE.name} + {column_maps[ScenarioVariables.SEX.name]} + {ScenarioVariables.CCI.name} + {ScenarioVariables.SPO2.name}',
            },
            "intervention": {
                'all': f'{ScenarioVariables.AGE.name} + {column_maps[ScenarioVariables.SEX.name]} + {ScenarioVariables.CCI.name} + {ScenarioVariables.SPO2.name}'
            }
        },
        "iptw_explanatory_formula": lambda column_maps: 
            f"{ScenarioVariables.AGE.name} + {column_maps[ScenarioVariables.SEX.name]} + {ScenarioVariables.CCI.name}",
        "preprocess_pipeline": default_preprocess_pipeline,
        "censor_vars": [ScenarioVariables.CENS],
        "censor_day0": [False], # whether we have censoring at day 0; will be used for masking
            # [ScenarioVariables.CENS, ScenarioVariables.COMP],
        "scenario": "scenario3",
        "censoring_model": CensoringModel.JOINT,
    }
    return configs
