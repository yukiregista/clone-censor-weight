from .. import VariableIDs, valueClass, VariableTypes
from enum import auto
import numpy as np
from typing import Callable
import itertools
import scipy.stats as stats


class ScenarioVariables(VariableIDs):
    AGE = auto()
    SEX = auto()
    CCI = auto()
    SPO2_OLD_MEAN = auto()  # COVID前の個人のSPO2の平均値
    SPO2_COVID_EFFECT = auto()  # COVIDによるSPO2の平均値の変化
    SPO2_NEW_MEAN = auto()  # COVID後の個人のSPO2の平均値
    SPO2 = auto()
    A = auto()
    D = auto()
    CENS = auto()  # informative censoring
    COMP = auto()  # competing risk
    A_sustained = auto()  # sustained treatment


ScenarioVariableTypes = {
    ScenarioVariables.AGE: VariableTypes.CONTINUOUS,
    ScenarioVariables.SEX: VariableTypes.BINARY,
    ScenarioVariables.CCI: VariableTypes.ORDERED_CONTINUOUS,
    ScenarioVariables.SPO2_OLD_MEAN: VariableTypes.CONTINUOUS,
    ScenarioVariables.SPO2_COVID_EFFECT: VariableTypes.CONTINUOUS,
    ScenarioVariables.SPO2_NEW_MEAN: VariableTypes.CONTINUOUS,
    ScenarioVariables.SPO2: VariableTypes.CONTINUOUS,
    ScenarioVariables.A: VariableTypes.EVENT_BINARY,
    ScenarioVariables.D: VariableTypes.EVENT_BINARY,
    ScenarioVariables.CENS: VariableTypes.EVENT_BINARY,
    ScenarioVariables.COMP: VariableTypes.EVENT_BINARY,
    ScenarioVariables.A_sustained: VariableTypes.BINARY
}


# def logit(p):
#     return np.log(p / (1 - p))
def logistic(x):
    return 1 / (1 + np.exp(-x))


def _has_comp(parent_vars: dict) -> bool:
    """
    COMP が履歴のどこかで 1 を取っていれば True。
    NaN や None は無視する。
    """
    comp_values = parent_vars.get(ScenarioVariables.COMP, {}).values(
    ) if ScenarioVariables.COMP in parent_vars else []
    for v in comp_values:
        try:
            if v is None:
                continue
            if isinstance(v, float) and np.isnan(v):
                continue
            if float(v) >= 1:
                return True
        except (TypeError, ValueError):
            continue
    return False


def _has_comp_vectorized(parent_arrays: dict, time: int, sample_size: int) -> np.ndarray:
    """
    Vectorized version: returns boolean array indicating which samples have competing events.

    Args:
        parent_arrays: Dictionary of parent variable arrays
        time: Current time point
        sample_size: Number of samples

    Returns:
        Boolean array of shape (sample_size,) where True indicates competing event occurred
    """
    has_comp = np.zeros(sample_size, dtype=bool)

    if ScenarioVariables.COMP not in parent_arrays:
        return has_comp

    # Check all time points up to current time
    for t, comp_array in parent_arrays[ScenarioVariables.COMP].items():
        if t <= time:
            has_comp |= (comp_array == 1)

    return has_comp


# AGE
# AGE_Parents = []  # 親なし


class AGE_CPD:
    def __init__(
        self,
        n_mixtures: int = 2,
        mixture_weights: tuple[float, ...] = (1 / 3, 2 / 3),
        means: tuple[float, ...] = (50, 80),
        stds: tuple[float, ...] = (15, 10),
    ):
        self.n_mixtures = n_mixtures
        self.mixture_weights = mixture_weights
        self.means = means
        self.stds = stds

    def __call__(self, parent_vars: dict[VariableIDs, dict[int, valueClass]], time: int, seed: int):
        # 混合分布からサンプル
        np.random.seed(seed)
        choice = np.random.choice(
            [i for i in range(self.n_mixtures)], p=self.mixture_weights)
        return np.random.normal(self.means[choice], self.stds[choice])

    def sample_batch(self, parent_arrays: dict[VariableIDs, dict[int, np.ndarray]],
                     time: int, seeds: np.ndarray, sample_size: int) -> np.ndarray:
        """Vectorized batch sampling for AGE"""
        rng = np.random.default_rng(seeds[0])

        # Sample mixture components for all samples at once
        choices = rng.choice(
            self.n_mixtures, size=sample_size, p=self.mixture_weights)

        # Vectorized normal sampling
        means_array = np.array(self.means)[choices]
        stds_array = np.array(self.stds)[choices]

        return rng.normal(means_array, stds_array)

# SEX
# SEX_Parents = []  # 親なし


class SEX_CPD:
    def __init__(self, probs: tuple[float, float] = (0.5, 0.5)):
        self.probs = probs

    def __call__(self, parent_vars: dict[VariableIDs, dict[int, valueClass]], time: int, seed: int):
        # 混合分布からサンプル
        np.random.seed(seed)
        return np.random.choice(["M", "F"], p=self.probs)

    def sample_batch(self, parent_arrays: dict[VariableIDs, dict[int, np.ndarray]],
                     time: int, seeds: np.ndarray, sample_size: int) -> np.ndarray:
        """Vectorized batch sampling for SEX"""
        rng = np.random.default_rng(seeds[0])
        return rng.choice(["M", "F"], size=sample_size, p=self.probs)


# CCI
# CCI_Parents = [(ScenarioVariables.AGE, LagSpec(begin_lag=None, end_lag=None)),
#                (ScenarioVariables.SEX, LagSpec(begin_lag=None, end_lag=None))]

CCI_Distribution_Default = {
    "YM": np.array([0.4, 0.3, 0.3]),  # cci prob for young male
    "OM": np.array([0.1, 0.3, 0.6]),  # cci prob for old male
    "YF": np.array([0.5, 0.4, 0.1]),  # cci prob for young female
    "OF": np.array([0.2, 0.4, 0.4])  # cci prob for old female
}


class CCI_CPD:
    def __init__(self, probs: dict[str, np.ndarray] = CCI_Distribution_Default, young_threshold: int = 50, old_threshold: int = 80):
        self.probs = probs
        self.young_threshold = young_threshold
        self.old_threshold = old_threshold

    def __call__(self, parent_vars: dict[VariableIDs, dict[int, valueClass]], time: int, seed: int):
        # 年齢と性別に応じてサンプル
        np.random.seed(seed)
        age = parent_vars[ScenarioVariables.AGE][0]
        sex = parent_vars[ScenarioVariables.SEX][0]

        category = None
        if age <= self.young_threshold:
            category = "YM" if sex == "M" else "YF"
        elif age >= self.old_threshold:
            category = "OM" if sex == "M" else "OF"

        if category is None:
            # between 50 and 80, so interpolate
            prob = self.probs["YM"] * (self.old_threshold - age) / (self.old_threshold - self.young_threshold) \
                + self.probs["OM"] * (age - self.young_threshold) / (self.old_threshold - self.young_threshold) if sex == "M" \
                else self.probs["YF"] * (self.old_threshold - age) / (self.old_threshold - self.young_threshold) \
                + self.probs["OF"] * (age - self.young_threshold) / \
                (self.old_threshold - self.young_threshold)
        else:
            prob = self.probs[category]
        # サンプリング
        return np.random.choice([0, 1, 2], p=prob)

    def sample_batch(self, parent_arrays: dict[VariableIDs, dict[int, np.ndarray]],
                     time: int, seeds: np.ndarray, sample_size: int) -> np.ndarray:
        """Vectorized batch sampling for CCI - fully vectorized version"""
        rng = np.random.default_rng(seeds[0])
        ages = parent_arrays[ScenarioVariables.AGE][0]
        sexes = parent_arrays[ScenarioVariables.SEX][0]

        # Compute weights (unified formula)
        weight_young = np.clip(
            (self.old_threshold - ages) /
            (self.old_threshold - self.young_threshold),
            0, 1
        )
        weight_old = 1 - weight_young
        is_male = sexes == "M"

        # Compute interpolated probabilities for all samples
        probs_male = (
            weight_young[:, np.newaxis] * self.probs["YM"] +
            weight_old[:, np.newaxis] * self.probs["OM"]
        )
        probs_female = (
            weight_young[:, np.newaxis] * self.probs["YF"] +
            weight_old[:, np.newaxis] * self.probs["OF"]
        )

        probs = np.where(is_male[:, np.newaxis], probs_male, probs_female)

        # Fully vectorized sampling using inverse CDF method
        # Convert categorical probabilities to cumulative distribution
        cumsum = np.cumsum(probs, axis=1)

        # Generate uniform random numbers
        u = rng.random(sample_size)

        # Find which category each sample falls into
        results = np.zeros(sample_size, dtype=int)
        results[u > cumsum[:, 0]] = 1
        results[u > cumsum[:, 1]] = 2

        return results


# SPO2_OLD_MEAN
# SPO2_OLD_MEAN_Parents = [
#     (ScenarioVariables.AGE, LagSpec(begin_lag=None, end_lag=None)),
#     (ScenarioVariables.SEX, LagSpec(begin_lag=None, end_lag=None))
# ]

SPO2_OLD_MEAN_STD_Default = {
    "YM": np.array([.95, .02]),  # spo2 mean and std for young male
    "OM": np.array([.90, .03]),  # spo2 mean and std for old male
    "YF": np.array([.94, .02]),  # spo2 mean and std for young female
    "OF": np.array([.89, .03])  # spo2 mean and std for old female
}

# beta分布のパラメータを計算する関数


def beta_params(mean, std):
    alpha = ((1 - mean) / std**2 - 1 / mean) * mean**2
    beta = alpha * (1 / mean - 1)
    return np.array([alpha, beta])


class SPO2_OLD_MEAN_CPD:
    def __init__(self, young_threshold: int = 50, old_threshold: int = 80,
                 SPO2_old_mean_std: dict[str, np.ndarray] = SPO2_OLD_MEAN_STD_Default):
        self.young_threshold = young_threshold
        self.old_threshold = old_threshold
        self.SPO2_old_mean_std = SPO2_old_mean_std
        self.SPO2_ALPHA_BETA = {
            k: np.asarray(beta_params(v[0], v[1]), dtype=np.float64)
            for k, v in self.SPO2_old_mean_std.items()
        }

    def __call__(self, parent_vars: dict[VariableIDs, dict[int, valueClass]], time: int, seed: int):
        np.random.seed(seed)
        # reparaneterization into shape parameters
        SPO2_ALPHA_BETA = {
            k: beta_params(v[0], v[1]) for k, v in self.SPO2_old_mean_std.items()
        }

        age = parent_vars[ScenarioVariables.AGE][0]
        sex = parent_vars[ScenarioVariables.SEX][0]

        category = None
        if age <= self.young_threshold:
            category = "YM" if sex == "M" else "YF"
        elif age >= self.old_threshold:
            category = "OM" if sex == "M" else "OF"

        if category is None:
            # between 50 and 80, so interpolate
            params = SPO2_ALPHA_BETA["YM"] * (self.old_threshold - age) / (self.old_threshold - self.young_threshold) \
                + SPO2_ALPHA_BETA["OM"] * (age - self.young_threshold) / (self.old_threshold - self.young_threshold) if sex == "M" \
                else SPO2_ALPHA_BETA["YF"] * (self.old_threshold - age) / (self.old_threshold - self.young_threshold) \
                + SPO2_ALPHA_BETA["OF"] * (age - self.young_threshold) / \
                (self.old_threshold - self.young_threshold)
        else:
            params = SPO2_ALPHA_BETA[category]

        # ベータ分布からサンプリング
        return np.random.beta(params[0], params[1])

    def sample_batch(self, parent_arrays: dict[VariableIDs, dict[int, np.ndarray]],
                     time: int, seeds: np.ndarray, sample_size: int) -> np.ndarray:
        """Vectorized batch sampling for SPO2_OLD_MEAN"""
        rng = np.random.default_rng(seeds[0])
        ages = parent_arrays[ScenarioVariables.AGE][0]
        sexes = parent_arrays[ScenarioVariables.SEX][0]

        # Compute weights (unified formula)
        weight_young = np.clip(
            (self.old_threshold - ages) /
            (self.old_threshold - self.young_threshold),
            0, 1
        )
        weight_old = 1 - weight_young
        is_male = sexes == "M"

        # Compute interpolated beta parameters for all samples
        # params_male[i] = [alpha_i, beta_i] for male sample i
        params_male = (
            weight_young[:, np.newaxis] * self.SPO2_ALPHA_BETA["YM"] +
            weight_old[:, np.newaxis] * self.SPO2_ALPHA_BETA["OM"]
        )
        params_female = (
            weight_young[:, np.newaxis] * self.SPO2_ALPHA_BETA["YF"] +
            weight_old[:, np.newaxis] * self.SPO2_ALPHA_BETA["OF"]
        )

        # Select appropriate parameters based on sex
        params = np.where(is_male[:, np.newaxis], params_male, params_female)

        # Vectorized beta sampling
        # Extract alpha and beta parameters
        alphas = params[:, 0]
        betas = params[:, 1]

        # Sample from beta distribution for all samples at once
        # パラメータでベクトライズ可能
        results = rng.beta(alphas, betas)
        return results


# SPO2_COVID_EFFECT
# SPO2_COVID_EFFECT_Parents = [
#     (ScenarioVariables.AGE, LagSpec(begin_lag=None, end_lag=None)),
#     (ScenarioVariables.SPO2_OLD_MEAN, LagSpec(begin_lag=None, end_lag=None))
# ]

SPO2_COVID_EFFECT_Default = {
    # SPO2 decreases this amount at time zero for young
    "Y": np.array([.05, .01]),
    # SPO2 decreases this amount at time zero for old
    "O": np.array([.10, .02])
}


class SPO2_COVID_EFFECT_CPD:
    def __init__(self, young_threshold: int = 50, old_threshold: int = 80,
                 SPO2_covid_effect: dict[str, np.ndarray] = SPO2_COVID_EFFECT_Default):
        self.young_threshold = young_threshold
        self.old_threshold = old_threshold
        self.SPO2_covid_effect = SPO2_covid_effect
        self.ALPHA_BETA = {
            k: np.asarray(beta_params(v[0], v[1]), dtype=np.float64)
            for k, v in self.SPO2_covid_effect.items()
        }

    def __call__(self, parent_vars: dict[VariableIDs, dict[int, valueClass]], time: int, seed: int):
        np.random.seed(seed)
        age = parent_vars[ScenarioVariables.AGE][0]
        spo2_old_mean = parent_vars[ScenarioVariables.SPO2_OLD_MEAN][0]

        ALPHA_BETA = {
            k: beta_params(v[0], v[1]) for k, v in self.SPO2_covid_effect.items()
        }

        category = None
        if age <= self.young_threshold:
            category = "Y"
        elif age >= self.old_threshold:
            category = "O"
        if category is None:
            # between 50 and 80, so interpolate
            params = ALPHA_BETA["Y"] * (self.old_threshold - age) / (self.old_threshold - self.young_threshold) \
                + ALPHA_BETA["O"] * (age - self.young_threshold) / \
                (self.old_threshold - self.young_threshold)
        else:
            params = ALPHA_BETA[category]

        # ベータ分布からサンプリング
        # 一応SPO2が負にならないように気を付ける
        effect = np.random.beta(params[0], params[1])
        if effect > spo2_old_mean:
            effect = spo2_old_mean - 0.01  # SPO2>=1%にしておく（ほとんど起こらない想定）
        return effect

    def sample_batch(self, parent_arrays: dict[VariableIDs, dict[int, np.ndarray]],
                     time: int, seeds: np.ndarray, sample_size: int) -> np.ndarray:
        """Vectorized batch sampling for SPO2_COVID_EFFECT"""
        rng = np.random.default_rng(seeds[0])

        ages = parent_arrays[ScenarioVariables.AGE][0]
        spo2_old_means = parent_arrays[ScenarioVariables.SPO2_OLD_MEAN][0]

        # Compute weights (unified formula)
        # Note: Only age matters for COVID effect (no sex dependence)
        weight_young = np.clip(
            (self.old_threshold - ages) /
            (self.old_threshold - self.young_threshold),
            0, 1
        )
        weight_old = 1 - weight_young

        # Compute interpolated beta parameters for all samples
        # params[i] = [alpha_i, beta_i] for sample i
        params = (
            weight_young[:, np.newaxis] * self.ALPHA_BETA["Y"] +
            weight_old[:, np.newaxis] * self.ALPHA_BETA["O"]
        )

        # Extract alpha and beta parameters
        alphas = params[:, 0]
        betas = params[:, 1]

        # Sample from beta distribution for all samples at once
        effects = rng.beta(alphas, betas)

        # Clip effects to ensure SPO2 doesn't go below 1%
        # effect must be <= spo2_old_mean - 0.01
        effects = np.minimum(effects, spo2_old_means - 0.01)

        return effects


# SPO2_NEW_MEAN
# SPO2_NEW_MEAN_Parents = [
#     (ScenarioVariables.AGE, LagSpec(begin_lag=None, end_lag=None)),
#     (ScenarioVariables.SPO2_OLD_MEAN, LagSpec(begin_lag=None, end_lag=None)),
#     (ScenarioVariables.SPO2_COVID_EFFECT, LagSpec(begin_lag=None, end_lag=None))
# ]

SPO2_NEW_MEAN_BETA_PARAMS_Default = {
    "Y": np.array([.09, .01]),
    "O": np.array([.09, .01])
}  # NEW_MEANの生成に用いるベータ分布のパラメータ：ベータ分布からのサンプルの値が1-> OLD_MEANと同じ、0 -> OLD_MEAN - COVID_EFFECTと同じ


class SPO2_NEW_MEAN_CPD:
    def __init__(self, young_threshold: int = 50, old_threshold: int = 80,
                 SPO2_new_mean_beta_params: dict[str, np.ndarray] = SPO2_NEW_MEAN_BETA_PARAMS_Default):
        self.young_threshold = young_threshold
        self.old_threshold = old_threshold
        self.SPO2_new_mean_beta_params = SPO2_new_mean_beta_params

    def __call__(self, parent_vars: dict[VariableIDs, dict[int, valueClass]], time: int, seed: int):
        np.random.seed(seed)
        age = parent_vars[ScenarioVariables.AGE][0]
        spo2_old_mean = parent_vars[ScenarioVariables.SPO2_OLD_MEAN][0]
        spo2_covid_effect = parent_vars[ScenarioVariables.SPO2_COVID_EFFECT][0]

        category = None
        if age <= self.young_threshold:
            category = "Y"
        elif age >= self.old_threshold:
            category = "O"

        if category is None:
            # between 50 and 80, so interpolate
            params = self.SPO2_new_mean_beta_params["Y"] * (self.old_threshold - age) / (self.old_threshold - self.young_threshold) \
                + self.SPO2_new_mean_beta_params["O"] * (age - self.young_threshold) / (
                    self.old_threshold - self.young_threshold)
        else:
            params = self.SPO2_new_mean_beta_params[category]

        # ベータ分布からサンプリング
        beta_sample = np.random.beta(params[0], params[1])

        # SPO2の新しい平均値を計算
        new_mean = beta_sample * spo2_old_mean + \
            (1 - beta_sample) * (spo2_old_mean - 2 * spo2_covid_effect)
        # truncate new_mean to be >= 0.01
        if new_mean < 0.01:
            new_mean = 0.01
        return new_mean

    def sample_batch(self, parent_arrays: dict[VariableIDs, dict[int, np.ndarray]],
                     time: int, seeds: np.ndarray, sample_size: int) -> np.ndarray:
        """Vectorized batch sampling for SPO2_NEW_MEAN"""
        rng = np.random.default_rng(seeds[0])

        ages = parent_arrays[ScenarioVariables.AGE][0]
        spo2_old_means = parent_arrays[ScenarioVariables.SPO2_OLD_MEAN][0]
        spo2_covid_effects = parent_arrays[ScenarioVariables.SPO2_COVID_EFFECT][0]

        # Compute weights (unified formula)
        # Only age matters for NEW_MEAN (no sex dependence)
        weight_young = np.clip(
            (self.old_threshold - ages) /
            (self.old_threshold - self.young_threshold),
            0, 1
        )
        weight_old = 1 - weight_young

        # Compute interpolated beta parameters for all samples
        # params[i] = [alpha_i, beta_i] for sample i
        params = (
            weight_young[:, np.newaxis] * self.SPO2_new_mean_beta_params["Y"] +
            weight_old[:, np.newaxis] * self.SPO2_new_mean_beta_params["O"]
        )

        # Extract alpha and beta parameters
        alphas = params[:, 0]
        betas = params[:, 1]

        # Sample from beta distribution for all samples at once
        beta_samples = rng.beta(alphas, betas)

        # Vectorized computation of new mean
        # new_mean = beta_sample * old_mean + (1 - beta_sample) * (old_mean - 2 * covid_effect)
        new_means = (
            beta_samples * spo2_old_means +
            (1 - beta_samples) * (spo2_old_means - 2 * spo2_covid_effects)
        )

        # Clip to ensure new_mean >= 0.01
        new_means = np.maximum(new_means, 0.01)

        return new_means


# SPO2
# SPO2_Parents = [(ScenarioVariables.AGE, LagSpec(begin_lag=None, end_lag=None)),
#                 (ScenarioVariables.SPO2_OLD_MEAN,
#                  LagSpec(begin_lag=None, end_lag=None)),
#                 (ScenarioVariables.SPO2_COVID_EFFECT,
#                  LagSpec(begin_lag=None, end_lag=None)),
#                 (ScenarioVariables.SPO2_NEW_MEAN,
#                  LagSpec(begin_lag=None, end_lag=None)),
#                 (ScenarioVariables.CCI, LagSpec(
#                     begin_lag=None, end_lag=None)),  # for time >= 1
#                 (ScenarioVariables.SPO2, LagSpec(
#                     begin_lag=1, end_lag=None)),  # for time >= 1
#                 (ScenarioVariables.A, LagSpec(
#                     begin_lag=1, end_lag=1)),  # for time >= 1
#                 (ScenarioVariables.D, LagSpec(
#                     begin_lag=0, end_lag=None))  # for time >= 1
#                 ]

# def psi_SPO2_default(t, age, cci):
#     # コロナ治療開始からの戻りやすさ具合をコントロール
#     return -np.exp(-t * ((50/age) + (3-cci))) # 適当...


def theta(age, cci):  # treatment effect の出方
    return np.exp((age-50)/50 - cci)


class SPO2_CPD:
    def __init__(self,
                 phi: float = 0.9,
                 #  psi: Callable[[int, int, int],float]  = psi_SPO2_default, # psi は廃止
                 theta: Callable[[int, int], float] = theta,
                 #  std_system: float = 1.0,
                 beta_precision: float = 100,  # 精度
                 coef_treatment: float = -0.1  # 治療効果の強さ
                 ):
        self.phi = phi
        # self.psi = psi
        self.theta = theta
        # self.std_system = std_system
        self.beta_precision = beta_precision
        self.coef_treatment = coef_treatment
        self.sqrt_1_minus_phi2 = np.sqrt(1 - self.phi**2)  # Add this line!

    def CPD_time_zero(self, parent_vars: dict[VariableIDs, dict[int, valueClass]], time: int, seed: int):
        old_mean = parent_vars[ScenarioVariables.SPO2_OLD_MEAN][0]
        covid_effect = parent_vars[ScenarioVariables.SPO2_COVID_EFFECT][0]
        return old_mean - covid_effect  # randomnessはここではなしの設定

    def CPD_time_t(self, parent_vars: dict[VariableIDs, dict[int, valueClass]], time: int, seed: int):
        np.random.seed(seed)
        if _has_comp(parent_vars):
            return np.nan
        # すでに死亡している場合は、SPO2はNaN
        if parent_vars[ScenarioVariables.D][time] == 1:
            return np.nan

        # probit モデルを用いる
#        new_mean = parent_vars[ScenarioVariables.SPO2_NEW_MEAN][0]

        # 基本のnew_meanを取得
        base_new_mean = parent_vars[ScenarioVariables.SPO2_NEW_MEAN][0]

        # 治療履歴に基づいて調整
        treatment_var = ScenarioVariables.A if ScenarioVariables.A in parent_vars else ScenarioVariables.A_sustained
        treatment_history = np.sum(
            np.array(list(parent_vars[treatment_var].values())))

        if treatment_history > 0:
            # 治療効果でbase_new_meanとold_meanの間を補間
            old_mean = parent_vars[ScenarioVariables.SPO2_OLD_MEAN][0]
            # memo: defaultではbase_new_meanはtime=0で決定されたSPO2_NEW_MEANを引き継いでいる。それはSPO2_OLD_MEANよりも小さい。そこで介入があればSPO2_NEW_MEANとSPO2_OLD_MEANの間をとる様に変更する。
            # coef_treatmentを補間係数として使用（0-1の範囲を想定）
            interpolation_factor = abs(self.coef_treatment)  # 負値の場合は絶対値を取る
            # 線形補間: new_mean = base_new_mean + factor * (old_mean - base_new_mean)
            new_mean = base_new_mean + interpolation_factor * \
                (old_mean - base_new_mean)
        else:
            new_mean = base_new_mean

        prev_mean = base_new_mean
        if treatment_history > 0 and parent_vars[treatment_var][time-1] == 0:
            # treatment occured before but not at previous time point
            assert time-1 == max(parent_vars[treatment_var].keys(
            )), f"treatment_varの履歴が不連続{parent_vars[treatment_var].keys()}:{time}"
            prev_mean = new_mean

        # N(0,1)のAR(1)
        # first, convert the previous SPO2 value using CDF of Beta(mean*precision, (1-mean)*precision)

        # latentの復元の際にはprev_meanを用いる
        U = stats.beta.cdf(parent_vars[ScenarioVariables.SPO2][time-1],
                           prev_mean * self.beta_precision, (1 - prev_mean) * self.beta_precision)
        # Use inverse CDF of standard normal distribution to get the previous value in standard normal space
        latent_prev = stats.norm.ppf(U)  # 0-1の値に変換しておく

        # Simulate AR(1) process Z_t=\rho Z_{t-1} + \sqrt{1-\rho^2} \epsilon_t, \epsilon_t \sim N(0,1)
        latent_now = self.phi * latent_prev + \
            np.sqrt(1 - self.phi**2) * np.random.normal(0, 1)

#        treatment_var = ScenarioVariables.A if ScenarioVariables.A in parent_vars else ScenarioVariables.A_sustained
#
#        # add one-time effect of treatment if treatment was given before
#        treatment_history = np.sum(
#            np.array(list(parent_vars[treatment_var].values())))
#        if treatment_history > 0:
#            latent_now += self.coef_treatment * treatment_history
#

        # Convert back to the original scale using standard normal CDF and the inverse CDF of Beta distribution
        X = stats.norm.cdf(latent_now)  # 0-1の値に変換しておく
        # Xを新しいSPO2の平均値に変換する
        X = stats.beta.ppf(X, new_mean * self.beta_precision,
                           (1 - new_mean) * self.beta_precision)

        return X

    def __call__(self, parent_vars: dict[VariableIDs, dict[int, valueClass]], time: int, seed: int):
        if time == 0:
            return self.CPD_time_zero(parent_vars, time, seed)
        else:
            return self.CPD_time_t(parent_vars, time, seed)

    def sample_batch(self, parent_arrays: dict[VariableIDs, dict[int, np.ndarray]],
                     time: int, seeds: np.ndarray, sample_size: int) -> np.ndarray:
        """Vectorized batch sampling for SPO2"""
        rng = np.random.default_rng(seeds[0])

        # Time zero case: deterministic
        if time == 0:
            old_means = parent_arrays[ScenarioVariables.SPO2_OLD_MEAN][0]
            covid_effects = parent_arrays[ScenarioVariables.SPO2_COVID_EFFECT][0]
            return old_means - covid_effects

        # Check for competing events
        has_comp = _has_comp_vectorized(parent_arrays, time, sample_size)

        # Check for death
        deaths = parent_arrays[ScenarioVariables.D][time]
        mask_invalid = has_comp | (deaths == 1)

        results = np.full(sample_size, np.nan, dtype=float)
        valid_mask = ~mask_invalid
        n_valid = np.sum(valid_mask)

        if n_valid == 0:
            return results

        # Extract arrays for valid samples
        base_new_means = parent_arrays[ScenarioVariables.SPO2_NEW_MEAN][0][valid_mask]
        old_means = parent_arrays[ScenarioVariables.SPO2_OLD_MEAN][0][valid_mask]
        prev_spo2 = parent_arrays[ScenarioVariables.SPO2][time - 1][valid_mask]

        # Determine which treatment variable to use
        treatment_var = ScenarioVariables.A if ScenarioVariables.A in parent_arrays else ScenarioVariables.A_sustained

        # Compute treatment history for all valid samples
        treatment_history = np.zeros(n_valid)
        for t in parent_arrays[treatment_var].keys():
            if t <= time:
                treatment_history += parent_arrays[treatment_var][t][valid_mask]

        # Compute new_mean with treatment effect
        has_treatment = treatment_history > 0
        interpolation_factor = abs(self.coef_treatment)
        new_means = np.where(
            has_treatment,
            base_new_means + interpolation_factor *
            (old_means - base_new_means),
            base_new_means
        )

        # Compute prev_mean (for latent space conversion)
        prev_means = base_new_means.copy()
        if time > 1 and time - 1 in parent_arrays[treatment_var]:
            prev_treatment = parent_arrays[treatment_var][time - 1][valid_mask]
            # If had treatment before but not at t-1, use new_mean as prev_mean
            mask_prev_adjust = has_treatment & (prev_treatment == 0)
            prev_means[mask_prev_adjust] = new_means[mask_prev_adjust]

        # Convert previous SPO2 to latent space (vectorized)
        U = stats.beta.cdf(
            prev_spo2,
            prev_means * self.beta_precision,
            (1 - prev_means) * self.beta_precision
        )
        latent_prev = stats.norm.ppf(U)

        # AR(1) dynamics in latent space (vectorized)
        epsilon = rng.normal(0, 1, size=n_valid)
        latent_now = self.phi * latent_prev + self.sqrt_1_minus_phi2 * epsilon

        # Convert back to original scale (vectorized)
        X = stats.norm.cdf(latent_now)
        results[valid_mask] = stats.beta.ppf(
            X,
            new_means * self.beta_precision,
            (1 - new_means) * self.beta_precision
        )

        return results


# A
# A_Parents = [
#     (ScenarioVariables.AGE, LagSpec(begin_lag=None, end_lag=None)),
#     (ScenarioVariables.SEX, LagSpec(begin_lag=None, end_lag=None)),
#     (ScenarioVariables.CCI, LagSpec(begin_lag=None, end_lag=None)),
#     (ScenarioVariables.SPO2, LagSpec(begin_lag=0, end_lag=None)),
#     (ScenarioVariables.A, LagSpec(begin_lag=None, end_lag=1)),
#     (ScenarioVariables.D, LagSpec(begin_lag=0, end_lag=None))
# ]


def psi_A_default(time):
    return np.exp(-time)  # 適当...


class A_CPD:
    def __init__(self, psi: Callable[[int], float] = psi_A_default, beta0: float = -1, beta_male: float = 0.1,
                 beta_age: float = -0.1 / 50, beta_cci: float = -0.1, beta_spo2_probit: float = -0.1,
                 variable_normalizers: dict[VariableIDs, Callable[[valueClass], valueClass]] | None = None):
        self.psi = psi
        self.beta0 = beta0
        self.beta_male = beta_male
        self.beta_age = beta_age
        self.beta_cci = beta_cci
        self.beta_spo2_probit = beta_spo2_probit
        self.variable_normalizers = variable_normalizers if variable_normalizers is not None else {}

    def _treatment_prob(self, parent_vars: dict[VariableIDs, dict[int, valueClass]], time: int):
        # それ以外の場合はロジット回帰
        is_male = int(parent_vars[ScenarioVariables.SEX][0] == "M")

        # 回帰に入れる変数の正規化
        X_variables = {
            ScenarioVariables.AGE: parent_vars[ScenarioVariables.AGE][0],
            ScenarioVariables.CCI: parent_vars[ScenarioVariables.CCI][0],
            ScenarioVariables.SPO2: parent_vars[ScenarioVariables.SPO2][time]
        }
        for key, normalizer in self.variable_normalizers.items():
            X_variables[key] = normalizer(X_variables[key])

        prob = logistic(
            self.beta0 + self.beta_male * is_male
            + self.beta_age * X_variables[ScenarioVariables.AGE]
            + self.beta_cci * X_variables[ScenarioVariables.CCI]
            + self.beta_spo2_probit * X_variables[ScenarioVariables.SPO2]
            + self.psi(time)
        )
        return prob

    def _treatment_prob_vectorized(self, sexes: np.ndarray, ages: np.ndarray,
                                   ccis: np.ndarray, spo2s: np.ndarray, time: int) -> np.ndarray:
        """
        Vectorized treatment probability computation.

        Args:
            sexes: Array of sex values ("M" or "F")
            ages: Array of age values
            ccis: Array of CCI values
            spo2s: Array of SPO2 values
            time: Current time point

        Returns:
            Array of treatment probabilities
        """
        # Compute is_male indicator (vectorized)
        is_male = (sexes == "M").astype(int)

        # Apply normalizers (vectorized)
        ages_norm = ages.copy()
        ccis_norm = ccis.copy()
        spo2s_norm = spo2s.copy()

        if ScenarioVariables.AGE in self.variable_normalizers:
            normalizer = self.variable_normalizers[ScenarioVariables.AGE]
            ages_norm = np.vectorize(normalizer)(ages)

        if ScenarioVariables.CCI in self.variable_normalizers:
            normalizer = self.variable_normalizers[ScenarioVariables.CCI]
            ccis_norm = np.vectorize(normalizer)(ccis)

        if ScenarioVariables.SPO2 in self.variable_normalizers:
            normalizer = self.variable_normalizers[ScenarioVariables.SPO2]
            spo2s_norm = np.vectorize(normalizer)(spo2s)

        # Vectorized logistic regression
        logits = (
            self.beta0 +
            self.beta_male * is_male +
            self.beta_age * ages_norm +
            self.beta_cci * ccis_norm +
            self.beta_spo2_probit * spo2s_norm +
            self.psi(time)
        )

        probs = logistic(logits)
        return probs

    def __call__(self, parent_vars: dict[VariableIDs, dict[int, valueClass]], time: int, seed: int):
        np.random.seed(seed)
        if _has_comp(parent_vars):
            return np.nan
        # すでに死亡している場合 | 治療している場合 は治療しない
        if time > 0 and (parent_vars[ScenarioVariables.D][time] == 1 or np.sum(np.array(list(parent_vars[ScenarioVariables.A].values()))) >= 1):
            return 0

        prob = self._treatment_prob(parent_vars, time)

        # サンプリング
        return int(np.random.choice([0, 1], p=[1-prob, prob]))

    def sample_batch(self, parent_arrays: dict[VariableIDs, dict[int, np.ndarray]],
                     time: int, seeds: np.ndarray, sample_size: int) -> np.ndarray:
        """Vectorized batch sampling for A (treatment)"""
        rng = np.random.default_rng(seeds[0])

        has_comp = _has_comp_vectorized(parent_arrays, time, sample_size)

        # Initialize results with NaN for competing events
        results = np.full(sample_size, np.nan, dtype=float)

        # Mask for samples that cannot receive treatment
        mask_cannot_treat = has_comp.copy()

        # For time > 0, check death and previous treatment
        if time > 0:
            deaths = parent_arrays[ScenarioVariables.D][time]

            # Check if already treated (any previous A == 1)
            already_treated = np.zeros(sample_size, dtype=bool)
            for t in parent_arrays[ScenarioVariables.A].keys():
                if t < time:
                    already_treated |= (
                        parent_arrays[ScenarioVariables.A][t] == 1)

            # Cannot treat if dead or already treated
            mask_cannot_treat |= (deaths == 1) | already_treated

        # Set dead/already treated to 0 (not NaN)
        results[mask_cannot_treat & ~has_comp] = 0

        # Valid mask: samples that can potentially receive treatment
        valid_mask = ~mask_cannot_treat
        n_valid = np.sum(valid_mask)

        if n_valid == 0:
            return results.astype(int)

        # Extract arrays for valid samples
        sexes = parent_arrays[ScenarioVariables.SEX][0][valid_mask]
        ages = parent_arrays[ScenarioVariables.AGE][0][valid_mask]
        ccis = parent_arrays[ScenarioVariables.CCI][0][valid_mask]
        spo2s = parent_arrays[ScenarioVariables.SPO2][time][valid_mask]

        # Compute treatment probabilities using vectorized method
        probs = self._treatment_prob_vectorized(sexes, ages, ccis, spo2s, time)

        # Vectorized sampling using inverse CDF method
        u = rng.random(n_valid)
        treatments = (u < probs).astype(int)

        # Assign to results
        results[valid_mask] = treatments

        return results.astype(int)


class A_CPD_wo_SPO2:
    def __init__(
        self,
        psi: Callable[[int], float] = psi_A_default,
        beta0: float = -1,
        beta_male: float = 0.1,
        beta_age: float = -0.1 / 50,
        beta_cci: float = -0.1,
        variable_normalizers: dict[VariableIDs, Callable[[
            valueClass], valueClass]] | None = None,
    ):
        self.psi = psi
        self.beta0 = beta0
        self.beta_male = beta_male
        self.beta_age = beta_age
        self.beta_cci = beta_cci
        self.variable_normalizers = variable_normalizers if variable_normalizers is not None else {}

    def _treatment_prob(self, parent_vars: dict[VariableIDs, dict[int, valueClass]], time: int):
        # それ以外の場合はロジット回帰
        is_male = int(parent_vars[ScenarioVariables.SEX][0] == "M")

        # 回帰に入れる変数の正規化
        X_variables = {
            ScenarioVariables.AGE: parent_vars[ScenarioVariables.AGE][0],
            ScenarioVariables.CCI: parent_vars[ScenarioVariables.CCI][0],
        }
        for key, normalizer in self.variable_normalizers.items():
            if key in X_variables:
                X_variables[key] = normalizer(X_variables[key])

        prob = logistic(
            self.beta0
            + self.beta_male * is_male
            + self.beta_age * X_variables[ScenarioVariables.AGE]
            + self.beta_cci * X_variables[ScenarioVariables.CCI]
            + self.psi(time)
        )
        return prob

    def _treatment_prob_vectorized(self, sexes: np.ndarray, ages: np.ndarray,
                                   ccis: np.ndarray, time: int) -> np.ndarray:
        """
        Vectorized treatment probability computation (without SPO2).

        Args:
            sexes: Array of sex values ("M" or "F")
            ages: Array of age values
            ccis: Array of CCI values
            time: Current time point

        Returns:
            Array of treatment probabilities
        """
        # Compute is_male indicator (vectorized)
        is_male = (sexes == "M").astype(int)

        # Apply normalizers (vectorized)
        ages_norm = ages.copy()
        ccis_norm = ccis.copy()

        if ScenarioVariables.AGE in self.variable_normalizers:
            normalizer = self.variable_normalizers[ScenarioVariables.AGE]
            ages_norm = np.vectorize(normalizer)(ages)

        if ScenarioVariables.CCI in self.variable_normalizers:
            normalizer = self.variable_normalizers[ScenarioVariables.CCI]
            ccis_norm = np.vectorize(normalizer)(ccis)

        # Vectorized logistic regression
        logits = (
            self.beta0 +
            self.beta_male * is_male +
            self.beta_age * ages_norm +
            self.beta_cci * ccis_norm +
            self.psi(time)
        )

        probs = logistic(logits)
        return probs

    def __call__(self, parent_vars: dict[VariableIDs, dict[int, valueClass]], time: int, seed: int):
        np.random.seed(seed)
        if _has_comp(parent_vars):
            return np.nan
        # すでに死亡している場合 | 治療している場合 は治療しない
        if time > 0 and (
            parent_vars[ScenarioVariables.D][time] == 1
            or np.sum(np.array(list(parent_vars[ScenarioVariables.A].values()))) >= 1
        ):
            return 0

        prob = self._treatment_prob(parent_vars, time)

        # サンプリング
        return int(np.random.choice([0, 1], p=[1 - prob, prob]))

    def sample_batch(self, parent_arrays: dict[VariableIDs, dict[int, np.ndarray]],
                     time: int, seeds: np.ndarray, sample_size: int) -> np.ndarray:
        """Vectorized batch sampling for A_wo_SPO2 (treatment without SPO2 covariate)"""
        rng = np.random.default_rng(seeds[0])

        has_comp = _has_comp_vectorized(parent_arrays, time, sample_size)

        # Initialize results with NaN for competing events
        results = np.full(sample_size, np.nan, dtype=float)

        # Mask for samples that cannot receive treatment
        mask_cannot_treat = has_comp.copy()

        # For time > 0, check death and previous treatment
        if time > 0:
            deaths = parent_arrays[ScenarioVariables.D][time]

            # Check if already treated (any previous A == 1)
            already_treated = np.zeros(sample_size, dtype=bool)
            for t in parent_arrays[ScenarioVariables.A].keys():
                if t < time:
                    already_treated |= (
                        parent_arrays[ScenarioVariables.A][t] == 1)

            # Cannot treat if dead or already treated
            mask_cannot_treat |= (deaths == 1) | already_treated

        # Set dead/already treated to 0 (not NaN)
        results[mask_cannot_treat & ~has_comp] = 0

        # Valid mask: samples that can potentially receive treatment
        valid_mask = ~mask_cannot_treat
        n_valid = np.sum(valid_mask)

        if n_valid == 0:
            return results.astype(int)

        # Extract arrays for valid samples (no SPO2 needed)
        sexes = parent_arrays[ScenarioVariables.SEX][0][valid_mask]
        ages = parent_arrays[ScenarioVariables.AGE][0][valid_mask]
        ccis = parent_arrays[ScenarioVariables.CCI][0][valid_mask]

        # Compute treatment probabilities using vectorized method
        probs = self._treatment_prob_vectorized(sexes, ages, ccis, time)

        # Vectorized sampling using inverse CDF method
        u = rng.random(n_valid)
        treatments = (u < probs).astype(int)

        # Assign to results
        results[valid_mask] = treatments

        return results.astype(int)


class A_sustained_CPD:
    def __init__(self,
                 psi: Callable[[int], float] = psi_A_default,
                 beta0: float = -1,
                 beta_male: float = 0.1,
                 beta_age: float = -0.1 / 50,
                 beta_cci: float = -0.1,
                 beta_spo2_probit: float = -0.1,
                 gamma0_stop: float = 0.0,
                 gamma_spo2_probit_stop: float = 0.5,
                 gamma0_restart: float = 0.0,
                 gamma_spo2_probit_restart: float = -0.5,
                 variable_normalizers: dict[VariableIDs, Callable[[valueClass], valueClass]] | None = None):
        self.psi = psi
        self.beta0 = beta0
        self.beta_male = beta_male
        self.beta_age = beta_age
        self.beta_cci = beta_cci
        self.beta_spo2_probit = beta_spo2_probit
        self.gamma0_stop = gamma0_stop
        self.gamma_spo2_probit_stop = gamma_spo2_probit_stop
        self.gamma0_restart = gamma0_restart
        self.gamma_spo2_probit_restart = gamma_spo2_probit_restart
        self.variable_normalizers = variable_normalizers if variable_normalizers is not None else {}

    def __call__(self, parent_vars: dict[VariableIDs, dict[int, valueClass]], time: int, seed: int):
        np.random.seed(seed)
        if _has_comp(parent_vars):
            return np.nan
        # 死亡時点では治療なし
        if time > 0 and parent_vars[ScenarioVariables.D][time] == 1:
            return 0

        # 直前時点の投与状態
        prev_on = 0 if time == 0 else int(
            parent_vars[ScenarioVariables.A_sustained][time-1])
        ever_started = 0 if time == 0 else int(np.sum(
            np.array(list(parent_vars[ScenarioVariables.A_sustained].values()))) >= 1)

        spo2_normalized_now = parent_vars[ScenarioVariables.SPO2][time]
        if ScenarioVariables.SPO2 in self.variable_normalizers:
            spo2_normalized_now = self.variable_normalizers[ScenarioVariables.SPO2](
                spo2_normalized_now)

        # ケース1) まだ開始していない（これまで一度も1がない）→ 従来ロジックで開始確率を計算
        if ever_started == 0:
            is_male = int(parent_vars[ScenarioVariables.SEX][0] == "M")
            X_variables = {
                ScenarioVariables.AGE: parent_vars[ScenarioVariables.AGE][0],
                ScenarioVariables.CCI: parent_vars[ScenarioVariables.CCI][0],
                ScenarioVariables.SPO2: spo2_normalized_now,
            }
            # Apply normalizers to all variables (same as A_CPD)
            for key, normalizer in self.variable_normalizers.items():
                if key != ScenarioVariables.SPO2:
                    X_variables[key] = normalizer(X_variables[key])

            prob_start = logistic(
                self.beta0 + self.beta_male * is_male
                + self.beta_age * X_variables[ScenarioVariables.AGE]
                + self.beta_cci * X_variables[ScenarioVariables.CCI]
                + self.beta_spo2_probit * X_variables[ScenarioVariables.SPO2]
                + self.psi(time)
            )
            return int(np.random.choice([0, 1], p=[1 - prob_start, prob_start]))

        # ケース2) 直前が投与中 → 停止確率はSPO2のみ（高SPO2ほど停止しやすい）
        if prev_on == 1:
            p_stop = logistic(self.gamma0_stop +
                              self.gamma_spo2_probit_stop * spo2_normalized_now)
            # 0=停止, 1=継続
            return int(np.random.choice([0, 1], p=[p_stop, 1 - p_stop]))

        # ケース3) 以前開始したが直前は非投与 → 再開確率はSPO2のみ（高SPO2ほど再開しにくい）
        p_restart = logistic(self.gamma0_restart +
                             self.gamma_spo2_probit_restart * spo2_normalized_now)
        return int(np.random.choice([0, 1], p=[1 - p_restart, p_restart]))

    def sample_batch(self, parent_arrays: dict[VariableIDs, dict[int, np.ndarray]],
                     time: int, seeds: np.ndarray, sample_size: int) -> np.ndarray:
        """Vectorized batch sampling for A_sustained (sustained treatment)"""
        rng = np.random.default_rng(seeds[0])

        # Check for competing events
        has_comp = _has_comp_vectorized(parent_arrays, time, sample_size)

        # Initialize results with NaN for competing events
        results = np.full(sample_size, np.nan, dtype=float)

        # Mask for samples that cannot receive treatment
        mask_cannot_treat = has_comp.copy()

        # For time > 0, check death
        if time > 0:
            deaths = parent_arrays[ScenarioVariables.D][time]
            mask_cannot_treat |= (deaths == 1)

        # Set dead samples to 0 (not NaN)
        results[mask_cannot_treat & ~has_comp] = 0

        # Valid mask: samples that can potentially receive treatment
        valid_mask = ~mask_cannot_treat
        n_valid = np.sum(valid_mask)

        if n_valid == 0:
            return results.astype(int)

        # Extract arrays for valid samples
        sexes = parent_arrays[ScenarioVariables.SEX][0][valid_mask]
        ages = parent_arrays[ScenarioVariables.AGE][0][valid_mask]
        ccis = parent_arrays[ScenarioVariables.CCI][0][valid_mask]
        spo2s = parent_arrays[ScenarioVariables.SPO2][time][valid_mask]

        spo2s_norm = spo2s.copy()
        if ScenarioVariables.SPO2 in self.variable_normalizers:
            normalizer = self.variable_normalizers[ScenarioVariables.SPO2]
            spo2s_norm = np.vectorize(normalizer)(spo2s)

        # Determine previous treatment status and history
        if time == 0:
            prev_on = np.zeros(n_valid, dtype=int)
            ever_started = np.zeros(n_valid, dtype=bool)
        else:
            prev_on = parent_arrays[ScenarioVariables.A_sustained][time -
                                                                   1][valid_mask].astype(int)

            # Check if ever started (any previous A_sustained == 1)
            ever_started = np.zeros(n_valid, dtype=bool)
            for t in parent_arrays[ScenarioVariables.A_sustained].keys():
                if t < time:
                    ever_started |= (
                        parent_arrays[ScenarioVariables.A_sustained][t][valid_mask] == 1)

        # Case 1: Never started treatment (ever_started == 0)
        mask_never_started = ~ever_started
        n_never_started = np.sum(mask_never_started)

        if n_never_started > 0:
            # Apply normalizers only for start case (same as A_CPD)
            sexes_ns = sexes[mask_never_started]
            ages_ns = ages[mask_never_started]
            ccis_ns = ccis[mask_never_started]
            spo2s_norm_ns = spo2s_norm[mask_never_started]

            is_male = (sexes_ns == "M").astype(int)

            # Apply normalizers (same as __call__ and A_CPD)
            ages_norm = ages_ns.copy()
            ccis_norm = ccis_ns.copy()
            spo2s_norm_for_start = spo2s_norm_ns.copy()

            if ScenarioVariables.AGE in self.variable_normalizers:
                normalizer = self.variable_normalizers[ScenarioVariables.AGE]
                ages_norm = np.vectorize(normalizer)(ages_ns)

            if ScenarioVariables.CCI in self.variable_normalizers:
                normalizer = self.variable_normalizers[ScenarioVariables.CCI]
                ccis_norm = np.vectorize(normalizer)(ccis_ns)

            # Compute start probability
            logits = (
                self.beta0 +
                self.beta_male * is_male +
                self.beta_age * ages_norm +
                self.beta_cci * ccis_norm +
                self.beta_spo2_probit * spo2s_norm_for_start +
                self.psi(time)
            )
            probs_start = logistic(logits)

            # Sample
            u = rng.random(n_never_started)
            treatments = (u < probs_start).astype(int)

            # Map back to valid indices
            valid_indices = np.where(valid_mask)[0]
            never_started_indices = valid_indices[mask_never_started]
            results[never_started_indices] = treatments

        # Case 2: Currently on treatment (prev_on == 1)
        mask_currently_on = ever_started & (prev_on == 1)
        n_currently_on = np.sum(mask_currently_on)

        if n_currently_on > 0:
            spo2s_norm_on = spo2s_norm[mask_currently_on]
            logits_stop = self.gamma0_stop + self.gamma_spo2_probit_stop * spo2s_norm_on
            probs_stop = logistic(logits_stop)

            # Sample: 0=stop, 1=continue
            u = rng.random(n_currently_on)
            treatments = (u >= probs_stop).astype(
                int)  # continue if u >= p_stop

            valid_indices = np.where(valid_mask)[0]
            currently_on_indices = valid_indices[mask_currently_on]
            results[currently_on_indices] = treatments

        # Case 3: Previously started but currently off (prev_on == 0)
        mask_restart = ever_started & (prev_on == 0)
        n_restart = np.sum(mask_restart)

        if n_restart > 0:
            spo2s_norm_restart = spo2s_norm[mask_restart]
            logits_restart = self.gamma0_restart + \
                self.gamma_spo2_probit_restart * spo2s_norm_restart
            probs_restart = logistic(logits_restart)

            # Sample
            u = rng.random(n_restart)
            treatments = (u < probs_restart).astype(int)

            valid_indices = np.where(valid_mask)[0]
            restart_indices = valid_indices[mask_restart]
            results[restart_indices] = treatments

        return results.astype(int)

# D_Parents = [
#     (ScenarioVariables.AGE, LagSpec(begin_lag=None, end_lag=None)),
#     (ScenarioVariables.SEX, LagSpec(begin_lag=None, end_lag=None)),
#     (ScenarioVariables.CCI, LagSpec(begin_lag=None, end_lag=None)),
#     (ScenarioVariables.SPO2, LagSpec(begin_lag=1, end_lag=None)),
#     (ScenarioVariables.A, LagSpec(begin_lag=None, end_lag=1)),
#     (ScenarioVariables.D, LagSpec(begin_lag=1, end_lag=None))
# ]


class D_CPD:
    def __init__(self, beta0: float = -5, beta_male: float = 0.1,
                 beta_age: float = 0.1 / 50, beta_cci: float = 0.1,
                 beta_spo2_probit: float = -0.1, treatment_effect: float = 0.1,
                 variable_normalizers: dict[VariableIDs, Callable[[valueClass], valueClass]] | None = None):
        self.beta0 = beta0
        self.beta_male = beta_male
        self.beta_age = beta_age
        self.beta_cci = beta_cci
        self.beta_spo2_probit = beta_spo2_probit
        self.treatment_effect = treatment_effect
        self.variable_normalizers = variable_normalizers if variable_normalizers is not None else {}

    # 0: alive, 1: dead
    def __call__(self, parent_vars: dict[VariableIDs, dict[int, valueClass]], time: int, seed: int):
        np.random.seed(seed)
        if _has_comp(parent_vars):
            return np.nan
        if time == 0:
            return 0
        # すでに死亡している場合は死亡のまま
        if parent_vars[ScenarioVariables.D][time-1] == 1:
            return 1

        treatment_var = ScenarioVariables.A if ScenarioVariables.A in parent_vars else ScenarioVariables.A_sustained

        # それ以外の場合はロジット回帰
        is_male = int(parent_vars[ScenarioVariables.SEX][0] == "M")
        treatment_history = np.sum(
            np.array(list(parent_vars[treatment_var].values())))

        X_variables = {
            ScenarioVariables.AGE: parent_vars[ScenarioVariables.AGE][0],
            ScenarioVariables.CCI: parent_vars[ScenarioVariables.CCI][0],
            ScenarioVariables.SPO2: parent_vars[ScenarioVariables.SPO2][time-1],
            treatment_var: treatment_history
        }

        for key, normalizer in self.variable_normalizers.items():
            X_variables[key] = normalizer(X_variables[key])

        prob = logistic(
            self.beta0 + self.beta_male * is_male
            + self.beta_age * X_variables[ScenarioVariables.AGE]
            + self.beta_cci * X_variables[ScenarioVariables.CCI]
            + self.beta_spo2_probit * X_variables[ScenarioVariables.SPO2]
            + self.treatment_effect * X_variables[treatment_var]
        )
        # サンプリング
        return int(np.random.choice([0, 1], p=[1-prob, prob]))

    def sample_batch(self, parent_arrays: dict[VariableIDs, dict[int, np.ndarray]],
                     time: int, seeds: np.ndarray, sample_size: int) -> np.ndarray:
        """Vectorized batch sampling for D (death)"""
        rng = np.random.default_rng(seeds[0])

        # Time 0: all alive
        if time == 0:
            return np.zeros(sample_size, dtype=int)

        # Check for competing events
        has_comp = _has_comp_vectorized(parent_arrays, time, sample_size)

        # Initialize with NaN for competing events
        results = np.full(sample_size, np.nan, dtype=float)

        # Already dead at t-1 remain dead
        prev_dead = parent_arrays[ScenarioVariables.D][time - 1] == 1
        results[prev_dead & ~has_comp] = 1

        # Valid mask: alive at t-1 and no competing event
        valid_mask = ~prev_dead & ~has_comp
        n_valid = np.sum(valid_mask)

        if n_valid == 0:
            return results.astype(int)

        # Extract arrays for valid samples
        sexes = parent_arrays[ScenarioVariables.SEX][0][valid_mask]
        ages = parent_arrays[ScenarioVariables.AGE][0][valid_mask]
        ccis = parent_arrays[ScenarioVariables.CCI][0][valid_mask]
        prev_spo2 = parent_arrays[ScenarioVariables.SPO2][time - 1][valid_mask]

        # Determine treatment variable
        treatment_var = ScenarioVariables.A if ScenarioVariables.A in parent_arrays else ScenarioVariables.A_sustained

        # Compute treatment history
        treatment_history = np.zeros(n_valid)
        for t in parent_arrays[treatment_var].keys():
            if t <= time:
                treatment_history += parent_arrays[treatment_var][t][valid_mask]

        # Apply normalizers
        is_male = (sexes == "M").astype(int)
        ages_norm = ages.copy()
        ccis_norm = ccis.copy()
        spo2_norm = prev_spo2.copy()
        treatment_norm = treatment_history.copy()

        if ScenarioVariables.AGE in self.variable_normalizers:
            ages_norm = np.vectorize(
                self.variable_normalizers[ScenarioVariables.AGE])(ages)
        if ScenarioVariables.CCI in self.variable_normalizers:
            ccis_norm = np.vectorize(
                self.variable_normalizers[ScenarioVariables.CCI])(ccis)
        if ScenarioVariables.SPO2 in self.variable_normalizers:
            spo2_norm = np.vectorize(
                self.variable_normalizers[ScenarioVariables.SPO2])(prev_spo2)
        if treatment_var in self.variable_normalizers:
            treatment_norm = np.vectorize(
                self.variable_normalizers[treatment_var])(treatment_history)

        # Compute death probabilities
        logits = (
            self.beta0 +
            self.beta_male * is_male +
            self.beta_age * ages_norm +
            self.beta_cci * ccis_norm +
            self.beta_spo2_probit * spo2_norm +
            self.treatment_effect * treatment_norm
        )
        probs = logistic(logits)

        # Sample
        u = rng.random(n_valid)
        results[valid_mask] = (u < probs).astype(int)

        return results.astype(int)


class CENS_CPD:
    def __init__(self, beta0: float = -2.5, beta_male: float = 0.05,
                 beta_age: float = 0.05 / 50, beta_cci: float = 0.05,
                 variable_normalizers: dict[VariableIDs, Callable[[valueClass], valueClass]] | None = None):
        self.beta0 = beta0
        self.beta_male = beta_male
        self.beta_age = beta_age
        self.beta_cci = beta_cci
        self.variable_normalizers = variable_normalizers if variable_normalizers is not None else {}

    def _compute_cens_prob(self, parent_vars: dict[VariableIDs, dict[int, valueClass]], time: int):
        is_male = int(parent_vars[ScenarioVariables.SEX][0] == "M")

        X_variables = {
            ScenarioVariables.AGE: parent_vars[ScenarioVariables.AGE][0],
            ScenarioVariables.CCI: parent_vars[ScenarioVariables.CCI][0],
        }

        for key, normalizer in self.variable_normalizers.items():
            X_variables[key] = normalizer(X_variables[key])

        prob = logistic(
            self.beta0 + self.beta_male * is_male
            + self.beta_age * X_variables[ScenarioVariables.AGE]
            + self.beta_cci * X_variables[ScenarioVariables.CCI]
        )
        return prob

    def _compute_cens_prob_vectorized(self, sexes: np.ndarray, ages: np.ndarray,
                                      ccis: np.ndarray) -> np.ndarray:
        """
        Vectorized censoring probability computation.

        Args:
            sexes: Array of sex values ("M" or "F")
            ages: Array of age values
            ccis: Array of CCI values

        Returns:
            Array of censoring probabilities
        """
        # Compute is_male indicator (vectorized)
        is_male = (sexes == "M").astype(int)

        # Apply normalizers (vectorized)
        ages_norm = ages.copy()
        ccis_norm = ccis.copy()

        if ScenarioVariables.AGE in self.variable_normalizers:
            normalizer = self.variable_normalizers[ScenarioVariables.AGE]
            ages_norm = np.vectorize(normalizer)(ages)

        if ScenarioVariables.CCI in self.variable_normalizers:
            normalizer = self.variable_normalizers[ScenarioVariables.CCI]
            ccis_norm = np.vectorize(normalizer)(ccis)

        # Vectorized logistic regression
        logits = (
            self.beta0 +
            self.beta_male * is_male +
            self.beta_age * ages_norm +
            self.beta_cci * ccis_norm
        )

        probs = logistic(logits)
        return probs

    # 0: not censored, 1: censored

    def __call__(self, parent_vars: dict[VariableIDs, dict[int, valueClass]], time: int, seed: int):
        np.random.seed(seed)
        if _has_comp(parent_vars):
            return np.nan
        if time == 0:
            return 0
        # すでに検閲済みの場合はそのまま
        if time > 0 and parent_vars[ScenarioVariables.CENS][time-1] == 1:
            return 1

        prob = self._compute_cens_prob(parent_vars, time)

        return int(np.random.choice([0, 1], p=[1 - prob, prob]))

    def sample_batch(self, parent_arrays: dict[VariableIDs, dict[int, np.ndarray]],
                     time: int, seeds: np.ndarray, sample_size: int) -> np.ndarray:
        """Vectorized batch sampling for CENS (censoring)"""
        rng = np.random.default_rng(seeds[0])

        # Time 0: no censoring
        if time == 0:
            return np.zeros(sample_size, dtype=int)

        # Check for competing events
        has_comp = _has_comp_vectorized(parent_arrays, time, sample_size)

        # Initialize with NaN for competing events
        results = np.full(sample_size, np.nan, dtype=float)

        # Already censored at t-1 remain censored
        prev_cens = parent_arrays[ScenarioVariables.CENS][time - 1] == 1
        results[prev_cens & ~has_comp] = 1

        # Valid mask: not censored at t-1 and no competing event
        valid_mask = ~prev_cens & ~has_comp
        n_valid = np.sum(valid_mask)

        if n_valid == 0:
            return results.astype(int)

        # Extract arrays for valid samples
        sexes = parent_arrays[ScenarioVariables.SEX][0][valid_mask]
        ages = parent_arrays[ScenarioVariables.AGE][0][valid_mask]
        ccis = parent_arrays[ScenarioVariables.CCI][0][valid_mask]

        # Compute censoring probabilities using vectorized method
        probs = self._compute_cens_prob_vectorized(sexes, ages, ccis)

        # Sample
        u = rng.random(n_valid)
        results[valid_mask] = (u < probs).astype(int)

        return results.astype(int)


class COMP_CPD:
    def __init__(self, beta0: float = -2.5,
                 beta_age: float = 0.05 / 50, beta_cci: float = 0.05,
                 variable_normalizers: dict[VariableIDs, Callable[[valueClass], valueClass]] | None = None):
        self.beta0 = beta0
        self.beta_age = beta_age
        self.beta_cci = beta_cci
        self.variable_normalizers = variable_normalizers if variable_normalizers is not None else {}

    # 0: no competing event, 1: competing event
    def __call__(self, parent_vars: dict[VariableIDs, dict[int, valueClass]], time: int, seed: int):
        np.random.seed(seed)
        if _has_comp(parent_vars):
            return np.nan
        if time == 0:
            return 0
        # すでに発生済みの場合はそのまま
        if parent_vars[ScenarioVariables.COMP][time-1] == 1:
            return 1

        X_variables = {
            ScenarioVariables.AGE: parent_vars[ScenarioVariables.AGE][0],
            ScenarioVariables.CCI: parent_vars[ScenarioVariables.CCI][0],
        }

        for key, normalizer in self.variable_normalizers.items():
            X_variables[key] = normalizer(X_variables[key])

        prob = logistic(
            self.beta0
            + self.beta_age * X_variables[ScenarioVariables.AGE]
            + self.beta_cci * X_variables[ScenarioVariables.CCI]
        )

        return int(np.random.choice([0, 1], p=[1 - prob, prob]))

    def sample_batch(self, parent_arrays: dict[VariableIDs, dict[int, np.ndarray]],
                     time: int, seeds: np.ndarray, sample_size: int) -> np.ndarray:
        """Vectorized batch sampling for COMP (competing event)"""
        rng = np.random.default_rng(seeds[0])

        # Time 0: no competing event
        if time == 0:
            return np.zeros(sample_size, dtype=int)

        # Initialize with NaN for already occurred competing events
        results = np.full(sample_size, np.nan, dtype=float)

        # Already has competing event at t-1
        prev_comp = parent_arrays[ScenarioVariables.COMP][time - 1] == 1
        results[prev_comp] = 1

        # Valid mask: no competing event at t-1
        valid_mask = ~prev_comp
        n_valid = np.sum(valid_mask)

        if n_valid == 0:
            return results.astype(int)

        # Extract arrays for valid samples (no sex needed for COMP)
        ages = parent_arrays[ScenarioVariables.AGE][0][valid_mask]
        ccis = parent_arrays[ScenarioVariables.CCI][0][valid_mask]

        # Apply normalizers
        ages_norm = ages.copy()
        ccis_norm = ccis.copy()

        if ScenarioVariables.AGE in self.variable_normalizers:
            ages_norm = np.vectorize(
                self.variable_normalizers[ScenarioVariables.AGE])(ages)
        if ScenarioVariables.CCI in self.variable_normalizers:
            ccis_norm = np.vectorize(
                self.variable_normalizers[ScenarioVariables.CCI])(ccis)

        # Compute competing event probabilities (no sex coefficient)
        logits = (
            self.beta0 +
            self.beta_age * ages_norm +
            self.beta_cci * ccis_norm
        )
        probs = logistic(logits)

        # Sample
        u = rng.random(n_valid)
        results[valid_mask] = (u < probs).astype(int)

        return results.astype(int)

# カテゴリわけ
# 年齢、性別、CCIからカテゴリを定義する


class ScenarioCategories:
    def __init__(self, young_threshold: int = 50, old_threshold: int = 80):
        self.young_threshold = young_threshold
        self.old_threshold = old_threshold

    def get_category(self, sample: list[dict[VariableIDs, dict[int, valueClass]]],
                     is_young: bool = True, is_old: bool = False, is_male: bool = True, cci: int = 0):

        # assert that is_young and is_old cannot be True at the same time
        assert not (
            is_young and is_old), "is_young and is_old cannot be True at the same time"

        age_filter = []
        if is_young:
            age_filter = [i for i in range(
                len(sample)) if sample[i][ScenarioVariables.AGE][0] <= self.young_threshold]
        elif is_old:
            age_filter = [i for i in range(
                len(sample)) if sample[i][ScenarioVariables.AGE][0] >= self.old_threshold]
        else:
            age_filter = [i for i in range(len(sample)) if sample[i][ScenarioVariables.AGE][0] >
                          self.young_threshold and sample[i][ScenarioVariables.AGE][0] < self.old_threshold]

        age_sex_filter = [i for i in age_filter if sample[i]
                          [ScenarioVariables.SEX][0] == ("M" if is_male else "F")]

        age_sex_cci_filter = [
            i for i in age_sex_filter if sample[i][ScenarioVariables.CCI][0] == cci]

        # only return sample where indices are included in the filter
        return [sample[i] for i in age_sex_cci_filter]

    def get_all_categories(self, sample: list[dict[VariableIDs, dict[int, valueClass]]]):
        age_categories = ["young", "old", "middle"]
        sex_categories = ["M", "F"]
        cci_categories = [0, 1, 2]

        for age, sex, cci in itertools.product(age_categories, sex_categories, cci_categories):
            category_label = ""
            if age == "young":
                is_young = True
                is_old = False
                category_label += "Y"
            elif age == "old":
                is_young = False
                is_old = True
                category_label += "O"
            else:
                is_young = False
                is_old = False
                category_label += "M"

            is_male = True if sex == "M" else False
            category_label += f"{sex}{cci}"

            yield category_label, self.get_category(sample, is_young=is_young, is_old=is_old, is_male=is_male, cci=cci)
