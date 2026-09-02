from ccw._research.configs.experiments import (
    experimentA,
    experimentB,
    experimentC,
    experimentD,
    experimentE,
    experimentF,
)

ALL_EXPERIMENT_CONFIGS = {
    "experimentA": experimentA.load_experiment_configs(),
    "experimentB": experimentB.load_experiment_configs(),
    "experimentC": experimentC.load_experiment_configs(),
    "experimentD": experimentD.load_experiment_configs(),
    "experimentE": experimentE.load_experiment_configs(),
    "experimentF": experimentF.load_experiment_configs(),
}
