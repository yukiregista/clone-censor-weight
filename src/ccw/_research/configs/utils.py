from ccw._research.utils import calculate_mixture_mean_std
from ccw._research.data_generation.scenarios.scenario import ScenarioVariables
import numpy as np
import scipy.stats as stats


def define_spo2_normalizer(PARAMETERS):
    beta_dist = stats.beta(3.22, 0.28)

    def spo2_normalizer(spo2):
        return 8 * (np.minimum(beta_dist.cdf(spo2), 0.5) - 0.25)
    return spo2_normalizer


def define_age_normalizer(PARAMETERS):
    def age_normalizer(age):
        mean, std = calculate_mixture_mean_std(
            PARAMETERS[ScenarioVariables.AGE.name]["mixture_weights"],
            PARAMETERS[ScenarioVariables.AGE.name]["means"],
            PARAMETERS[ScenarioVariables.AGE.name]["stds"]
        )
        return (age - mean) / std
    return age_normalizer
