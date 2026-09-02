from ccw._research.data_generation.core import VariableIDs
import numpy as np


class IncidentRate():
    def __init__(self, outcome_var: VariableIDs, eval_time: int):
        self.outcome_var = outcome_var
        self.eval_time = eval_time

    def convert_to_incident(self, sample: list[dict[VariableIDs, dict[int, any]]]) -> np.ndarray:
        assert set(np.unique([s[self.outcome_var][self.eval_time]
                   for s in sample])).issubset({0, 1}), f"Outcome variable {self.outcome_var} at time {self.eval_time} must be binary (0 or 1). Current values: {np.unique([s[self.outcome_var][self.eval_time] for s in sample])}"
        return np.array([s[self.outcome_var][self.eval_time] for s in sample], dtype=np.int8)

    def __call__(self, sample: list[dict[VariableIDs, dict[int, any]]]) -> float:
        """
        サンプルから指定された時点での事象率を計算する。 
        """
        incidents = self.convert_to_incident(sample)
        count = np.sum(incidents)
        total = len(incidents)
        return count / total if total > 0 else 0.0
