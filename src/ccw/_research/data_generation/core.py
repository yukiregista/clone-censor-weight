from enum import Enum
from typing import Callable, Collection
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

valueClass = int | float | str


class VariableIDs(Enum):
    """
    変数ID の Enum クラス。 
    """
    pass


class VariableTypes(Enum):
    CONTINUOUS = "continuous"
    ORDERED_CONTINUOUS = "ordered_continuous"
    CATEGORICAL = "categorical"
    ORDERED_CATEGORICAL = "ordered_categorical"
    BINARY = "binary"
    EVENT_BINARY = "event_binary"


class LagSpec:
    def __init__(self, begin_lag: int | None = None, end_lag: int | None = None):
        """
        DAGの矢印の始点の変数の時刻のリストを指定するクラス。

        Parameters
        ----------
        begin_lag : int | None, optional
            Noneの場合はTime Zeroに対応, by default None
        end_lag : int | None, optional
            Noneの場合は`begin_lag == end_lag`とする, by default None
        """
        self.begin_lag = begin_lag
        self.end_lag = end_lag

    def get_lags(self, current_time: int) -> list[int]:
        """
        現在時刻 current_time に対して、矢印が伸びるべき時刻のリストを返す。
        """
        begin_time = current_time - self.begin_lag if self.begin_lag is not None else 0
        end_time = current_time - self.end_lag if self.end_lag is not None else begin_time
        return list(range(max(0, begin_time), end_time + 1))


class Variable:
    def __init__(self, id: VariableIDs, value_type: VariableTypes, is_time_varying: bool = False, parent_variables: list[tuple[VariableIDs, LagSpec | None]] | None = None, CPD: Callable[[dict[VariableIDs, dict[int, valueClass]], int, int], valueClass] = None):
        self.id: VariableIDs = id
        self.value_type: VariableTypes = value_type  # 変数の型
        self.is_time_varying: bool = is_time_varying  # 時系列変数かどうか
        self.parent_variables: dict[VariableIDs, LagSpec | None] = dict(
            parent_variables or ())  # 親変数のリスト
        self.CPD: Callable[[dict[VariableIDs, dict[int, valueClass]],
                            int, int], valueClass] = CPD  # 条件付き確率分布

    def sample(self, parent_values: dict[VariableIDs, dict[int, valueClass]], t: int, seed: int) -> valueClass:
        try:
            observation = self.CPD(parent_values, t, seed)
        except Exception as exc:
            raise ValueError(f"CPD function for {self.id} failed: {exc}") from exc
        return observation

    def __repr__(self):
        return f"Variable(id={self.id}, is_time_varying={self.is_time_varying})"


class BayesianNetwork:
    def _create_dag(self, variables: list[Variable], n_time: int) -> nx.DiGraph:
        dag = nx.DiGraph()
        # first, add all nodes to the graph
        var_to_nodes = {var: {} for var in variables}
        for var in variables:
            if var.is_time_varying:
                for t in range(n_time):
                    node = f"{var.id.name}_{t}"
                    dag.add_node(node, id=var.id, time=t)
                    var_to_nodes[var][t] = node
            else:
                node = f"{var.id.name}_0"
                # time of the node is set to 0 for non-time-varying variables
                dag.add_node(node, id=var.id, time=0)
                var_to_nodes[var][0] = node

        # then, add edges based on the parent variables
        for var in variables:
            for parent_var_id, lag in var.parent_variables.items():
                parent_var = self.variable_ids_to_variables[parent_var_id]
                for t, node in var_to_nodes[var].items():
                    # if the parent is not time-varying, add an edge from the parent node
                    if not parent_var.is_time_varying:
                        parent_node = var_to_nodes[parent_var][0]
                        dag.add_edge(parent_node, node)
                    # if the parent is time-varying, add edges from the parent nodes at times obtained from the lag
                    if parent_var.is_time_varying:
                        times = lag.get_lags(t)
                        for time in times:
                            parent_node = var_to_nodes[parent_var][time]
                            dag.add_edge(parent_node, node)

        return dag

    def draw_dag(self) -> None:
        pos = nx.spring_layout(self.DAG)
        nx.draw(self.DAG, pos, with_labels=True,
                node_color="lightblue", edge_color="gray", node_size=1500)
        plt.title("Bayesian Network DAG")
        plt.show()

    def __init__(self, variables: list[Variable], n_time: int):
        self.variables: list[Variable] = variables
        self.variable_ids_to_variables: dict[VariableIDs, Variable] = {
            var.id: var for var in variables}
        self.n_time: int = n_time
        self.DAG: nx.DiGraph = self._create_dag(variables, n_time)
        if not nx.is_directed_acyclic_graph(self.DAG):
            cycles = list(nx.simple_cycles(self.DAG))
            raise ValueError(f"Cycle detected in the DAG: {cycles}")
        # self.draw_dag()
        self.topological_order: list[str] = list(nx.topological_sort(self.DAG))

    def sample(
        self,
        sample_size: int,
        seed: int,
        skip_if_event_at_t0_for: Collection[Variable] | None = None,
    ) -> list[dict[VariableIDs, dict[int, valueClass]]]:
        if skip_if_event_at_t0_for is None:
            return self.sample_vectorized(sample_size, seed)

        rng = np.random.default_rng(seed)
        samples = [{var.id: {} for var in self.variables}
                   for _ in range(sample_size)]
        sample_index = 0
        skip_set = set(skip_if_event_at_t0_for or [])

        while True:
            iter_success = True
            if sample_index == sample_size:
                break
            for node_name in self.topological_order:
                # generate the node value
                node_attrs = self.DAG.nodes[node_name]
                node_variable = self.variable_ids_to_variables[node_attrs['id']]

                # get the parent values
                parent_values = {}
                for parent in self.DAG.predecessors(node_name):
                    parent_attrs = self.DAG.nodes[parent]
                    parent_variable = self.variable_ids_to_variables[parent_attrs['id']]
                    parent_time = parent_attrs['time']
                    if parent_variable.id not in parent_values:
                        parent_values[parent_variable.id] = {}
                    parent_values[parent_variable.id][parent_time] = samples[sample_index][parent_variable.id][parent_time]

                # generate from the CPD
                _seed = rng.integers(0, 2**32 - 1)
                value = node_variable.sample(
                    parent_values, node_attrs['time'], _seed)

                # 変更: time=0 のスキップ判定
                if (
                    node_attrs['time'] == 0
                    and node_variable.id in skip_set
                    and value == 1
                ):
                    # 該当イベントが time=0 で発生 → サンプルやり直し
                    samples[sample_index] = {var.id: {}
                                             for var in self.variables}
                    iter_success = False
                    break

                samples[sample_index][node_variable.id][node_attrs['time']] = value

            if iter_success:
                sample_index += 1

        return samples

    def sample_vectorized(
        self,
        sample_size: int,
        seed: int,
    ) -> list[dict[VariableIDs, dict[int, valueClass]]]:
        """
        Fully vectorized sampling from the Bayesian Network.
        Uses separate dtype-specific arrays for each variable.
        """
        rng = np.random.default_rng(seed)

        # Helper to determine storage dtype based on variable type
        def _get_dtype(var_type, var_id=None):
            """
            Determine numpy dtype based on variable type.

            Args:
                var_type: VariableTypes enum value
                var_id: Optional variable ID for special handling

            Returns:
                numpy dtype
            """
            if var_type in [VariableTypes.CONTINUOUS, VariableTypes.ORDERED_CONTINUOUS]:
                return np.float64
            elif var_type == VariableTypes.BINARY:
                # Special case: SEX is binary but uses strings, not integers
                if var_id.name == "SEX":  # temporary: 本当は各Variableがdtype属性を持つべき
                    return object
                return np.int64
            elif var_type in [VariableTypes.EVENT_BINARY, VariableTypes.ORDERED_CATEGORICAL]:
                return np.int64
            else:
                # CATEGORICAL or other string-based types
                return object

        # Separate time-varying and non-time-varying variables
        time_varying_vars = [
            var for var in self.variables if var.is_time_varying]
        static_vars = [
            var for var in self.variables if not var.is_time_varying]

        # Pre-allocate arrays with appropriate dtypes for each variable
        # tv_samples: dict[var_id -> (sample_size, n_time) array with appropriate dtype]
        tv_samples = {
            var.id: np.empty((sample_size, self.n_time),
                             dtype=_get_dtype(var.value_type, var.id))
            for var in time_varying_vars
        }

        # static_samples: dict[var_id -> (sample_size,) array with appropriate dtype]
        static_samples = {
            var.id: np.empty(sample_size, dtype=_get_dtype(
                var.value_type, var.id))
            for var in static_vars
        }

        # Pre-compute node information with GROUPED parent time points
        node_info_list = []
        for node_name in self.topological_order:
            node_attrs = self.DAG.nodes[node_name]
            node_var_id = node_attrs['id']
            node_time = node_attrs['time']
            node_variable = self.variable_ids_to_variables[node_var_id]

            is_time_varying = node_variable.is_time_varying

            # Group parents by variable ID to handle multiple time lags efficiently
            parent_time_groups = {}

            for parent in self.DAG.predecessors(node_name):
                parent_attrs = self.DAG.nodes[parent]
                parent_var_id = parent_attrs['id']
                parent_time = parent_attrs['time']
                parent_variable = self.variable_ids_to_variables[parent_var_id]

                parent_is_tv = parent_variable.is_time_varying

                if parent_var_id not in parent_time_groups:
                    parent_time_groups[parent_var_id] = {
                        'is_time_varying': parent_is_tv,
                        'times': []
                    }
                parent_time_groups[parent_var_id]['times'].append(parent_time)

            node_info_list.append({
                'variable': node_variable,
                'time': node_time,
                'is_time_varying': is_time_varying,
                'parent_time_groups': parent_time_groups
            })

        # Generate all seeds upfront
        all_seeds = rng.integers(
            0, 2**32 - 1, size=(len(self.topological_order), sample_size))

        # Sample in topological order - VECTORIZED with optimized parent extraction
        for node_idx, node_info in enumerate(node_info_list):
            node_variable = node_info['variable']
            node_var_id = node_variable.id
            node_time = node_info['time']
            node_is_tv = node_info['is_time_varying']

            # Extract parent values - directly from dtype-specific arrays
            parent_arrays = {}
            for parent_var_id, parent_info in node_info['parent_time_groups'].items():
                parent_is_tv = parent_info['is_time_varying']
                times = parent_info['times']

                parent_arrays[parent_var_id] = {}

                if parent_is_tv:
                    # Extract from time-varying samples dict
                    # Shape: (sample_size, n_time)
                    parent_array = tv_samples[parent_var_id]

                    if len(times) == 1:
                        # Single time point - simple slice
                        parent_arrays[parent_var_id][times[0]
                                                     ] = parent_array[:, times[0]]
                    else:
                        # Multiple time points - extract all at once
                        for t in times:
                            parent_arrays[parent_var_id][t] = parent_array[:, t]
                else:
                    # Static variable - same value for all requested times
                    # Shape: (sample_size,)
                    static_vals = static_samples[parent_var_id]
                    for t in times:
                        parent_arrays[parent_var_id][t] = static_vals

            # Check if CPD supports batch processing
            if hasattr(node_variable.CPD, 'sample_batch'):
                # VECTORIZED PATH: Call batch sampling
                values = node_variable.CPD.sample_batch(
                    parent_arrays=parent_arrays,
                    time=node_time,
                    seeds=all_seeds[node_idx, :],
                    sample_size=sample_size
                )
            else:
                # FALLBACK PATH: Call individual sampling (slower)
                values = np.empty(sample_size, dtype=object)
                for sample_idx in range(sample_size):
                    parent_values = {}
                    for parent_var_id, time_dict in parent_arrays.items():
                        parent_values[parent_var_id] = {
                            t: time_dict[t][sample_idx] for t in time_dict.keys()
                        }
                    values[sample_idx] = node_variable.sample(
                        parent_values, node_time, all_seeds[node_idx, sample_idx]
                    )

            # Store in appropriate array (already has correct dtype)
            if node_is_tv:
                tv_samples[node_var_id][:, node_time] = values
            else:
                static_samples[node_var_id][:] = values

        # Convert back to expected output format
        samples = []
        for sample_idx in range(sample_size):
            sample = {}

            # Add time-varying variables
            for var in time_varying_vars:
                sample[var.id] = {}
                for t in range(self.n_time):
                    sample[var.id][t] = tv_samples[var.id][sample_idx, t]

            # Add static variables
            for var in static_vars:
                sample[var.id] = {0: static_samples[var.id][sample_idx]}

            samples.append(sample)

        return samples

    def __repr__(self):
        return f"BayesianNetwork(variables={self.variables}, n_time={self.n_time})"
