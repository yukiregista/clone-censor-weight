from .. import LagSpec
from .scenario import ScenarioVariables

class LagConfigs:
    """Configuration class for parent variable lag specifications"""
    
    @staticmethod
    def get_scenario1_lags():
        return {
            ScenarioVariables.AGE: [],
            
            ScenarioVariables.SEX: [],
            
            ScenarioVariables.CCI: [
                (ScenarioVariables.AGE, LagSpec(begin_lag=None, end_lag=None)),
                (ScenarioVariables.SEX, LagSpec(begin_lag=None, end_lag=None))
            ],
            
            ScenarioVariables.SPO2_OLD_MEAN: [
                (ScenarioVariables.AGE, LagSpec(begin_lag=None, end_lag=None)),
                (ScenarioVariables.SEX, LagSpec(begin_lag=None, end_lag=None))
            ],
            
            ScenarioVariables.SPO2_COVID_EFFECT: [
                (ScenarioVariables.AGE, LagSpec(begin_lag=None, end_lag=None)),
                (ScenarioVariables.SPO2_OLD_MEAN, LagSpec(begin_lag=None, end_lag=None))
            ],
            
            ScenarioVariables.SPO2_NEW_MEAN: [
                (ScenarioVariables.AGE, LagSpec(begin_lag=None, end_lag=None)),
                (ScenarioVariables.SPO2_OLD_MEAN, LagSpec(begin_lag=None, end_lag=None)),
                (ScenarioVariables.SPO2_COVID_EFFECT, LagSpec(begin_lag=None, end_lag=None))
            ],
            
            ScenarioVariables.SPO2: [
                (ScenarioVariables.AGE, LagSpec(begin_lag=None, end_lag=None)),
                (ScenarioVariables.SPO2_OLD_MEAN, LagSpec(begin_lag=None, end_lag=None)),
                (ScenarioVariables.SPO2_COVID_EFFECT, LagSpec(begin_lag=None, end_lag=None)),
                (ScenarioVariables.SPO2_NEW_MEAN, LagSpec(begin_lag=None, end_lag=None)),
                (ScenarioVariables.CCI, LagSpec(begin_lag=None, end_lag=None)),
                (ScenarioVariables.SPO2, LagSpec(begin_lag=1, end_lag=None)),
                (ScenarioVariables.A, LagSpec(begin_lag=None, end_lag=1)),
                (ScenarioVariables.D, LagSpec(begin_lag=0, end_lag=None))
            ],
            
            ScenarioVariables.A: [
                (ScenarioVariables.AGE, LagSpec(begin_lag=None, end_lag=None)),
                (ScenarioVariables.SEX, LagSpec(begin_lag=None, end_lag=None)),
                (ScenarioVariables.CCI, LagSpec(begin_lag=None, end_lag=None)),
                (ScenarioVariables.SPO2, LagSpec(begin_lag=0, end_lag=None)),
                (ScenarioVariables.A, LagSpec(begin_lag=None, end_lag=1)),
                (ScenarioVariables.D, LagSpec(begin_lag=0, end_lag=None))
            ],
            
            ScenarioVariables.D: [
                (ScenarioVariables.AGE, LagSpec(begin_lag=None, end_lag=None)),
                (ScenarioVariables.SEX, LagSpec(begin_lag=None, end_lag=None)),
                (ScenarioVariables.CCI, LagSpec(begin_lag=None, end_lag=None)),
                (ScenarioVariables.SPO2, LagSpec(begin_lag=1, end_lag=None)),
                (ScenarioVariables.A, LagSpec(begin_lag=None, end_lag=1)),
                (ScenarioVariables.D, LagSpec(begin_lag=1, end_lag=None))
            ]
        }
    @staticmethod
    def get_scenario2_lags():
        return {
            ScenarioVariables.AGE: [],
            
            ScenarioVariables.SEX: [],
            
            ScenarioVariables.CCI: [
                (ScenarioVariables.AGE, LagSpec(begin_lag=None, end_lag=None)),
                (ScenarioVariables.SEX, LagSpec(begin_lag=None, end_lag=None))
            ],
            
            ScenarioVariables.SPO2_OLD_MEAN: [
                (ScenarioVariables.AGE, LagSpec(begin_lag=None, end_lag=None)),
                (ScenarioVariables.SEX, LagSpec(begin_lag=None, end_lag=None))
            ],
            
            ScenarioVariables.SPO2_COVID_EFFECT: [
                (ScenarioVariables.AGE, LagSpec(begin_lag=None, end_lag=None)),
                (ScenarioVariables.SPO2_OLD_MEAN, LagSpec(begin_lag=None, end_lag=None))
            ],
            
            ScenarioVariables.SPO2_NEW_MEAN: [
                (ScenarioVariables.AGE, LagSpec(begin_lag=None, end_lag=None)),
                (ScenarioVariables.SPO2_OLD_MEAN, LagSpec(begin_lag=None, end_lag=None)),
                (ScenarioVariables.SPO2_COVID_EFFECT, LagSpec(begin_lag=None, end_lag=None))
            ],
            
            ScenarioVariables.SPO2: [
                (ScenarioVariables.AGE, LagSpec(begin_lag=None, end_lag=None)),
                (ScenarioVariables.SPO2_OLD_MEAN, LagSpec(begin_lag=None, end_lag=None)),
                (ScenarioVariables.SPO2_COVID_EFFECT, LagSpec(begin_lag=None, end_lag=None)),
                (ScenarioVariables.SPO2_NEW_MEAN, LagSpec(begin_lag=None, end_lag=None)),
                (ScenarioVariables.CCI, LagSpec(begin_lag=None, end_lag=None)),
                (ScenarioVariables.SPO2, LagSpec(begin_lag=1, end_lag=None)),
                (ScenarioVariables.A, LagSpec(begin_lag=None, end_lag=1)),
                (ScenarioVariables.D, LagSpec(begin_lag=0, end_lag=None))
            ],
            
            ScenarioVariables.A: [
                (ScenarioVariables.AGE, LagSpec(begin_lag=None, end_lag=None)),
                (ScenarioVariables.SEX, LagSpec(begin_lag=None, end_lag=None)),
                (ScenarioVariables.CCI, LagSpec(begin_lag=None, end_lag=None)),
                # (ScenarioVariables.SPO2, LagSpec(begin_lag=0, end_lag=None)), # remove dependency on SPO2
                (ScenarioVariables.A, LagSpec(begin_lag=None, end_lag=1)),
                (ScenarioVariables.D, LagSpec(begin_lag=0, end_lag=None))
            ],
            
            ScenarioVariables.D: [
                (ScenarioVariables.AGE, LagSpec(begin_lag=None, end_lag=None)),
                (ScenarioVariables.SEX, LagSpec(begin_lag=None, end_lag=None)),
                (ScenarioVariables.CCI, LagSpec(begin_lag=None, end_lag=None)),
                (ScenarioVariables.SPO2, LagSpec(begin_lag=1, end_lag=None)),
                (ScenarioVariables.A, LagSpec(begin_lag=None, end_lag=1)),
                (ScenarioVariables.D, LagSpec(begin_lag=1, end_lag=None))
            ]
        }
    @staticmethod
    def get_scenario3_lags():
        return {
            ScenarioVariables.AGE: [],
            
            ScenarioVariables.SEX: [],
            
            ScenarioVariables.CCI: [
                (ScenarioVariables.AGE, LagSpec(begin_lag=None, end_lag=None)),
                (ScenarioVariables.SEX, LagSpec(begin_lag=None, end_lag=None))
            ],
            
            ScenarioVariables.SPO2_OLD_MEAN: [
                (ScenarioVariables.AGE, LagSpec(begin_lag=None, end_lag=None)),
                (ScenarioVariables.SEX, LagSpec(begin_lag=None, end_lag=None))
            ],
            
            ScenarioVariables.SPO2_COVID_EFFECT: [
                (ScenarioVariables.AGE, LagSpec(begin_lag=None, end_lag=None)),
                (ScenarioVariables.SPO2_OLD_MEAN, LagSpec(begin_lag=None, end_lag=None))
            ],
            
            ScenarioVariables.SPO2_NEW_MEAN: [
                (ScenarioVariables.AGE, LagSpec(begin_lag=None, end_lag=None)),
                (ScenarioVariables.SPO2_OLD_MEAN, LagSpec(begin_lag=None, end_lag=None)),
                (ScenarioVariables.SPO2_COVID_EFFECT, LagSpec(begin_lag=None, end_lag=None))
            ],
            
            ScenarioVariables.SPO2: [
                (ScenarioVariables.AGE, LagSpec(begin_lag=None, end_lag=None)),
                (ScenarioVariables.SPO2_OLD_MEAN, LagSpec(begin_lag=None, end_lag=None)),
                (ScenarioVariables.SPO2_COVID_EFFECT, LagSpec(begin_lag=None, end_lag=None)),
                (ScenarioVariables.SPO2_NEW_MEAN, LagSpec(begin_lag=None, end_lag=None)),
                (ScenarioVariables.CCI, LagSpec(begin_lag=None, end_lag=None)),
                (ScenarioVariables.SPO2, LagSpec(begin_lag=1, end_lag=None)),
                (ScenarioVariables.A, LagSpec(begin_lag=None, end_lag=1)),
                (ScenarioVariables.D, LagSpec(begin_lag=0, end_lag=None)),
                (ScenarioVariables.CENS, LagSpec(begin_lag=None, end_lag=0)),
                # (ScenarioVariables.COMP, LagSpec(begin_lag=None, end_lag=0))
            ],
            
            ScenarioVariables.A: [
                (ScenarioVariables.AGE, LagSpec(begin_lag=None, end_lag=None)),
                (ScenarioVariables.SEX, LagSpec(begin_lag=None, end_lag=None)),
                (ScenarioVariables.CCI, LagSpec(begin_lag=None, end_lag=None)),
                (ScenarioVariables.SPO2, LagSpec(begin_lag=0, end_lag=None)),
                (ScenarioVariables.A, LagSpec(begin_lag=None, end_lag=1)),
                (ScenarioVariables.D, LagSpec(begin_lag=0, end_lag=None)),
                (ScenarioVariables.CENS, LagSpec(begin_lag=None, end_lag=0)),
                # (ScenarioVariables.COMP, LagSpec(begin_lag=None, end_lag=0))
            ],         
            ScenarioVariables.D: [
                (ScenarioVariables.AGE, LagSpec(begin_lag=None, end_lag=None)),
                (ScenarioVariables.SEX, LagSpec(begin_lag=None, end_lag=None)),
                (ScenarioVariables.CCI, LagSpec(begin_lag=None, end_lag=None)),
                (ScenarioVariables.SPO2, LagSpec(begin_lag=1, end_lag=None)),
                (ScenarioVariables.A, LagSpec(begin_lag=None, end_lag=1)),
                (ScenarioVariables.D, LagSpec(begin_lag=1, end_lag=None)),
                (ScenarioVariables.CENS, LagSpec(begin_lag=None, end_lag=0)),
                # (ScenarioVariables.COMP, LagSpec(begin_lag=None, end_lag=0))
            ],
            ScenarioVariables.CENS: [
                (ScenarioVariables.AGE, LagSpec(begin_lag=None, end_lag=None)),
                (ScenarioVariables.SEX, LagSpec(begin_lag=None, end_lag=None)),
                (ScenarioVariables.CCI, LagSpec(begin_lag=None, end_lag=None)),
                (ScenarioVariables.CENS, LagSpec(begin_lag=None, end_lag=1)),
            ],
            ScenarioVariables.COMP: [
                (ScenarioVariables.AGE, LagSpec(begin_lag=None, end_lag=None)),
                (ScenarioVariables.SEX, LagSpec(begin_lag=None, end_lag=None)),
                (ScenarioVariables.CCI, LagSpec(begin_lag=None, end_lag=None)),
                # (ScenarioVariables.COMP, LagSpec(begin_lag=None, end_lag=1)),
            ],
        }
    
