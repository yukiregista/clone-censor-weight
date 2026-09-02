from typing import Any, Callable, Dict, List, Protocol
import copy

from ccw._research.data_generation.core import BayesianNetwork, Variable, VariableIDs


class CounterfactualStrategy(Protocol):
    """Structural interface required by the counterfactual simulator."""

    def transform_cpd(
        self,
        original_cpd: Callable[..., Any],
        treatment_variable: VariableIDs,
    ) -> Callable[..., Any]:
        """Return the treatment CPD induced by the strategy."""

        ...


class CounterfactualSimulator:
    def __init__(self,
                 base_variables: List[Variable],
                 n_time: int):
        """
        base_variables: 元のVariableオブジェクトのリスト
        n_time: シミュレーション期間（時点数）
        """
        self.base_variables = base_variables
        self.n_time = n_time
        self.bn = dict()

    def _create_network(self,
                        intervention_dict: Dict[VariableIDs, CounterfactualStrategy]) -> BayesianNetwork:
        """
        intervention_dict: 介入対象変数ID -> CounterfactualStrategy
        """
        # Variableオブジェクトを複製し、必要に応じてCPDを置き換え
        vars_for_bn = []
        for var in self.base_variables:
            new_var = copy.copy(var)
            if var.id in intervention_dict:
                new_var.CPD = intervention_dict[var.id].transform_cpd(
                    var.CPD, var.id)
            vars_for_bn.append(new_var)
        return BayesianNetwork(vars_for_bn, self.n_time)

    def simulate(self,
                 sample_size: int,
                 seed: int,
                 strategies: Dict[str, Dict[VariableIDs, CounterfactualStrategy]]
                 ) -> Dict[str, List[Dict[VariableIDs, Dict[int, any]]]]:
        """
        sample_size: サンプル数
        seed: 乱数シード
        strategies: 戦略ラベル -> (介入変数ID -> CounterfactualStrategy)
        戻り値: 戦略ラベル -> サンプルリスト
        """
        results: Dict[str, List[Dict[VariableIDs, Dict[int, any]]]] = {}
        for label, strategy_map in strategies.items():
            if self.bn.get(label) is None:
                bn = self._create_network(strategy_map)
                self.bn[label] = bn
            results[label] = self.bn[label].sample(sample_size, seed)
        return results
