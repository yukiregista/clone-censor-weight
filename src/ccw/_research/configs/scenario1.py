import numpy as np
import yaml
from ccw import InitiateBy, NoInitiationThrough
from ccw._research.data_generation.scenarios.scenario import (
    AGE_CPD,
    A_CPD,
    CCI_CPD,
    D_CPD,
    ScenarioVariables,
    ScenarioVariableTypes,
    SEX_CPD,
    SPO2_COVID_EFFECT_CPD,
    SPO2_CPD,
    SPO2_NEW_MEAN_CPD,
    SPO2_OLD_MEAN_CPD,
)
from ccw._research.data_generation.scenarios.lag_configs import LagConfigs
from ccw._research.data_generation.core import Variable
from ccw._research.configs.utils import define_spo2_normalizer, define_age_normalizer
from ccw._research.configs.config_override import read_yaml_text



def theta_SPO2(age, cci):  # treatment effect の出方
    return np.exp(-5 + (age-50)/100 - cci)  # 適当...


def psi_A(time):
    return np.exp(-0.1 * time)



def load_scenario1_parameters():
    text = read_yaml_text("scenario1.yaml", __package__)
    raw = yaml.safe_load(text)
    # post-process: turn lists into numpy arrays where you need them
    raw["CCI"]["probs"] = {k: np.array(v)
                           for k, v in raw["CCI"]["probs"].items()}
    raw["SPO2_OLD_MEAN"]["SPO2_old_mean_std"] = {
        k: np.array(v) for k, v in raw["SPO2_OLD_MEAN"]["SPO2_old_mean_std"].items()}
    raw["SPO2_COVID_EFFECT"]["SPO2_covid_effect"] = {
        k: np.array(v) for k, v in raw["SPO2_COVID_EFFECT"]["SPO2_covid_effect"].items()}
    raw["SPO2_NEW_MEAN"]["SPO2_new_mean_beta_params"] = {
        k: np.array(v) for k, v in raw["SPO2_NEW_MEAN"]["SPO2_new_mean_beta_params"].items()}

    # …similarly for all mixture lists…
    # Attach your functions by name:
    raw["SPO2"]["theta"] = theta_SPO2
    raw["A"]["psi"] = psi_A

    variable_normalizers = {
        ScenarioVariables.AGE: define_age_normalizer(raw),
        ScenarioVariables.SPO2: define_spo2_normalizer(raw),
    }
    raw[ScenarioVariables.A.name]["variable_normalizers"] = variable_normalizers
    raw[ScenarioVariables.D.name]["variable_normalizers"] = variable_normalizers
    return raw


def load_scenario1_params_and_variables():
    PARAMETERS = load_scenario1_parameters()
    LAG_CONFIGS = LagConfigs.get_scenario1_lags()
    AllVariables = [Variable(
        id=ScenarioVariables.AGE,
        value_type=ScenarioVariableTypes[ScenarioVariables.AGE],
        is_time_varying=False,
        parent_variables=LAG_CONFIGS[ScenarioVariables.AGE],
        CPD=AGE_CPD(**PARAMETERS[ScenarioVariables.AGE.name]),
    ),
        Variable(
        id=ScenarioVariables.SEX,
        value_type=ScenarioVariableTypes[ScenarioVariables.SEX],
        is_time_varying=False,
        parent_variables=LAG_CONFIGS[ScenarioVariables.SEX],
        CPD=SEX_CPD(**PARAMETERS[ScenarioVariables.SEX.name]),
    ),
        Variable(
        id=ScenarioVariables.CCI,
        value_type=ScenarioVariableTypes[ScenarioVariables.CCI],
        is_time_varying=False,
        parent_variables=LAG_CONFIGS[ScenarioVariables.CCI],
        CPD=CCI_CPD(**PARAMETERS[ScenarioVariables.CCI.name]),
    ),
        Variable(
        id=ScenarioVariables.SPO2_OLD_MEAN,
        value_type=ScenarioVariableTypes[ScenarioVariables.SPO2_OLD_MEAN],
        is_time_varying=False,
        parent_variables=LAG_CONFIGS[ScenarioVariables.SPO2_OLD_MEAN],
        CPD=SPO2_OLD_MEAN_CPD(
            **PARAMETERS[ScenarioVariables.SPO2_OLD_MEAN.name])
    ),
        Variable(
        id=ScenarioVariables.SPO2_COVID_EFFECT,
        value_type=ScenarioVariableTypes[ScenarioVariables.SPO2_COVID_EFFECT],
        is_time_varying=False,
        parent_variables=LAG_CONFIGS[ScenarioVariables.SPO2_COVID_EFFECT],
        CPD=SPO2_COVID_EFFECT_CPD(
            **PARAMETERS[ScenarioVariables.SPO2_COVID_EFFECT.name])
    ),
        Variable(
        id=ScenarioVariables.SPO2_NEW_MEAN,
        value_type=ScenarioVariableTypes[ScenarioVariables.SPO2_NEW_MEAN],
        is_time_varying=False,
        parent_variables=LAG_CONFIGS[ScenarioVariables.SPO2_NEW_MEAN],
        CPD=SPO2_NEW_MEAN_CPD(
            **PARAMETERS[ScenarioVariables.SPO2_NEW_MEAN.name])
    ),
        Variable(
        id=ScenarioVariables.SPO2,
        value_type=ScenarioVariableTypes[ScenarioVariables.SPO2],
        is_time_varying=True,
        parent_variables=LAG_CONFIGS[ScenarioVariables.SPO2],
        CPD=SPO2_CPD(**PARAMETERS[ScenarioVariables.SPO2.name]),
    ),
        Variable(
        id=ScenarioVariables.A,
        value_type=ScenarioVariableTypes[ScenarioVariables.A],
        is_time_varying=True,
        parent_variables=LAG_CONFIGS[ScenarioVariables.A],
        CPD=A_CPD(**PARAMETERS[ScenarioVariables.A.name]),
    ),
        Variable(
        id=ScenarioVariables.D,
        value_type=ScenarioVariableTypes[ScenarioVariables.D],
        is_time_varying=True,
        parent_variables=LAG_CONFIGS[ScenarioVariables.D],
        CPD=D_CPD(**PARAMETERS[ScenarioVariables.D.name]),
    )
    ]
    configs = {
        "treatment_var": ScenarioVariables.A,
        "outcome_var": ScenarioVariables.D,
        "strategy_creator": {
            "intervention": lambda args: InitiateBy(args.cutoff_time_of_intervention),
            "control": lambda args: NoInitiationThrough(args.cutoff_time_of_intervention),
        },
        "time_varying_vars": [ScenarioVariables.SPO2],
    }
    return PARAMETERS, AllVariables, configs
