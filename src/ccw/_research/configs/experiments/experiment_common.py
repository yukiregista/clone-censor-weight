
from ccw._research.utils import convert_binary_data
from ccw._research.data_generation.scenarios.scenario import ScenarioVariables
from scipy import stats
import numpy as np


def default_preprocess_pipeline(df, bootstrap=False):
    if not bootstrap:
        df, column_maps = convert_binary_data(
            df, [ScenarioVariables.SEX.name]
        )
        # spo2 normalization
        df['SPO2_CDF'] = np.minimum(
            stats.beta(3.22, 0.28).cdf(df['SPO2']),
            0.5,
        )
        df['SPO2_NORM'] = df['SPO2_CDF']
        # now make SPO2_NORM be SPO2
        df['SPO2_original'] = df['SPO2']
        df['SPO2'] = df['SPO2_NORM']
    else:
        column_maps = {}
        df['SPO2_CDF'] = np.minimum(
            stats.beta(3.22, 0.28).cdf(df['SPO2_original']),
            0.5,
        )
        df['SPO2_NORM'] = df['SPO2_CDF']
        df['SPO2'] = df['SPO2_NORM']
        # df['SPO2_original'] remains the same
    return df, column_maps
