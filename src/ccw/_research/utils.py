from .data_generation.core import BayesianNetwork, valueClass, VariableIDs
import pandas as pd
import numpy as np


def calculate_mixture_mean_std(mixture_weights: list[float], mixture_means: list[float], mixture_variances: list[float]):
    mean = sum(weight * mean for weight,
               mean in zip(mixture_weights, mixture_means, strict=True))
    variance = sum(weight * (var + (mean - mu) ** 2) for weight, mu,
                   var in zip(
                       mixture_weights,
                       mixture_means,
                       mixture_variances,
                       strict=True,
                   ))
    std = np.sqrt(variance)
    return mean, std


def create_datasets_in_df(bn: BayesianNetwork, sample: list[dict[VariableIDs, dict[int, valueClass]]],
                          treatment_var: VariableIDs, outcome_var: VariableIDs,
                          cut_data_after_outcome: bool,
                          cutoff_time_of_observation: int,
                          censor_vars: list[VariableIDs] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Get the list of time-varying and static variables
    time_varying_vars = [
        var for var in bn.variable_ids_to_variables.values() if var.is_time_varying]
    baseline_vars = [
        var for var in bn.variable_ids_to_variables.values() if not var.is_time_varying]
    # Note: we are assuming implicitly throughout the project that the time-invariant variables are all baseline variables.

    assert treatment_var in bn.variable_ids_to_variables, "Treatment variable not in Bayesian Network."
    assert outcome_var in bn.variable_ids_to_variables, "Outcome variable not in Bayesian Network."
    if cut_data_after_outcome:
        assert outcome_var in {
            item.id for item in time_varying_vars}, "Outcome variable must be time-varying if cut_data_after_outcome is True."

    # ① df_baseline の作成
    baseline_dict = {'id': [idx for idx in range(len(sample))]}
    for var in baseline_vars:
        baseline_dict[var.id.name] = []
    for person in sample:
        for var in baseline_vars:
            # time==0 の値を取得
            baseline_dict[var.id.name].append(person[var.id][0])
    df_baseline = pd.DataFrame.from_dict(baseline_dict, orient='columns')

    # ② df_time_varying, ③ df_intervention_outcome の作成
    tv_dict = {'id': [], 'time': []}
    for var in time_varying_vars:
        tv_dict[var.id.name] = []
    io_dict = {'id': [idx for idx in range(len(sample))],
               'time_to_intervention': [],
               'time_to_outcome': []}
    for idx, person in enumerate(sample):
        # time の一覧（辞書のキー）を取得
        times = sorted(person[time_varying_vars[0].id].keys())
        times = [t for t in times if t <= cutoff_time_of_observation]
        times_I = [t for t in times if person[treatment_var][t] == 1]
        time_to_intervention = min(times_I) if times_I else np.nan
        io_dict['time_to_intervention'].append(time_to_intervention)
        # Outcome が 1 となる最初の time
        times_O = [t for t in times if person[outcome_var][t] == 1]
        time_to_outcome = min(times_O) if times_O else np.nan
        io_dict['time_to_outcome'].append(time_to_outcome)
        
        # censor_vars のどれかが 1 となる最初の time
        time_to_censor = np.nan
        if censor_vars is not None and len(censor_vars) > 0:
            times_C_list = []
            for censor_var in censor_vars:
                times_C = [t for t in times if person[censor_var][t] == 1]
                if times_C:
                    times_C_list.append(min(times_C))
            time_to_censor = min(times_C_list) if times_C_list else np.nan
        
        # censor の次以降の time を除外
        if not np.isnan(time_to_censor):
            times = [t for t in times if t <= time_to_censor]
            # modify time_to_intervention & time_to_outcome
            if not np.isnan(time_to_intervention) and time_to_intervention > time_to_censor:
                time_to_intervention = np.nan
                io_dict['time_to_intervention'][-1] = np.nan
            if not np.isnan(time_to_outcome) and time_to_outcome > time_to_censor:
                time_to_outcome = np.nan
                io_dict['time_to_outcome'][-1] = np.nan

        if cut_data_after_outcome:
            # Outcome が 1 となる最初の time より前のデータのみを使用
            times = [
                t for t in times if pd.isna(time_to_outcome) or t <= time_to_outcome]
        tv_dict['id'].extend([idx] * len(times))
        tv_dict['time'].extend(times)
        for var in time_varying_vars:
            tv_dict[var.id.name].extend([person[var.id][t] for t in times])
    df_time_varying = pd.DataFrame.from_dict(tv_dict, orient='columns')
    df_intervention_outcome = pd.DataFrame.from_dict(io_dict, orient='columns')
    df_intervention_outcome = df_intervention_outcome.astype({
        'time_to_intervention': 'Int64',
        'time_to_outcome':     'Int64'
    })

    # Joinしたdfも作っておく
    df_joined = pd.merge(df_baseline, df_time_varying, on='id', how='right')
    df_joined = pd.merge(
        df_joined, df_intervention_outcome, on='id', how='left')

    return df_baseline, df_time_varying, df_intervention_outcome, df_joined


def convert_binary_data(df: pd.DataFrame, binary_cols: list[str]):
    column_maps = {col: col for col in binary_cols}
    for col in binary_cols:
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].astype(
                int)
        else:
            # convert it to [0, 1] and change the column name
            binary_vals = sorted(df[col].unique())
            # remove nan
            binary_vals = [
                val for val in binary_vals if not pd.isna(val)]
            assert len(
                binary_vals) == 2, f"Binary variable {col} must have exactly two unique values."
            df[col] = df[col].map(
                {binary_vals[0]: 0, binary_vals[1]: 1})
            # change the column name
            df.rename(
                columns={col: f"{col}_is_{binary_vals[1]}"}, inplace=True)
            column_maps[col] = f"{col}_is_{binary_vals[1]}"
    return df, column_maps

# def convert_categorical_data(joined_df: pd.DataFrame, bn: BayesianNetwork) -> pd.DataFrame:
#     """
#     Convert categorical data in the joined DataFrame to numerical values based on the variable definitions in the Bayesian Network.
#     """
#     categorical_types = {VariableTypes.CATEGORICAL,
#                          VariableTypes.BINARY, VariableTypes.ORDERED_CATEGORICAL}
#     for var in bn.variable_ids_to_variables.values():
#         if var.value_type in categorical_types and var.id.name in joined_df.columns:
#             if var.value_type == VariableTypes.BINARY:
#                 # If the column dtype is numeric or boolean, convert it to integer
#                 if pd.api.types.is_numeric_dtype(joined_df[var.id.name]) or pd.api.types.is_bool_dtype(joined_df[var.id.name]):
#                     joined_df[var.id.name] = joined_df[var.id.name].astype(int)
#                 else:
#                     # convert it to [0, 1] and change the column name
#                     binary_vals = joined_df[var.id.name].unique()
#                     # remove nan
#                     binary_vals = [
#                         val for val in binary_vals if not pd.isna(val)]
#                     assert len(
#                         binary_vals) == 2, f"Binary variable {var.id.name} must have exactly two unique values."
#                     joined_df[var.id.name] = joined_df[var.id.name].map(
#                         {binary_vals[0]: 1, binary_vals[1]: 0})
#                     # change the column name
#                     joined_df.rename(
#                         columns={var.id.name: f"{var.id.name}_is_{binary_vals[1]}"}, inplace=True)
#             elif var.value_type == VariableTypes.CATEGORICAL:
#                 # If the column dtype is numeric, convert it to categorical
#                 if pd.api.types.is_numeric_dtype(joined_df[var.id.name]):
#                     joined_df[var.id.name] = joined_df[var.id.name].astype(
#                         'category')
#                 else:
#                     # convert it to categorical and change the column name
#                     categories = joined_df[var.id.name].unique()
#                     categories = [
#                         cat for cat in categories if not pd.isna(cat)]
#                     joined_df[var.id.name] = pd.Categorical(
#                         joined_df[var.id.name], categories=categories)
#                     # change the column name
#                     joined_df.rename(
#                         columns={var.id.name: f"{var.id.name}_cat"}, inplace=True)
#     return joined_df


def flatten_dict(d, parent_key="", sep="."):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items
