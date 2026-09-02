from ccw import CensoringModel
from ccw._research.configs.experiments.experiment_common import default_preprocess_pipeline
from ccw._research.data_generation.scenarios.scenario import ScenarioVariables


def load_experiment_configs():
    configs = {
        "cut_data_after_outcome": True,
        "ipw_explanatory_formula": lambda column_maps: {
            "control": {
                "artificial_censor": f"C(tstart) + {ScenarioVariables.AGE.name} + {column_maps[ScenarioVariables.SEX.name]} + {ScenarioVariables.CCI.name} + {ScenarioVariables.SPO2.name}",
            },
            "intervention": {
                "artificial_censor": f"{ScenarioVariables.AGE.name} + {column_maps[ScenarioVariables.SEX.name]} + {ScenarioVariables.CCI.name} + {ScenarioVariables.SPO2.name}",
            },
        },
        "iptw_explanatory_formula": lambda column_maps: f"{ScenarioVariables.AGE.name} + {column_maps[ScenarioVariables.SEX.name]} + {ScenarioVariables.CCI.name}",
        "preprocess_pipeline": default_preprocess_pipeline,
        "censor_vars": [ScenarioVariables.CENS],
        "censor_day0": [False],
        "scenario": "scenario3",
        "censoring_model": CensoringModel.PROTOCOL_ONLY,
    }
    return configs
