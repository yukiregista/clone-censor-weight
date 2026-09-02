from ..data_generation import VariableIDs, valueClass, VariableTypes
from ..data_generation.core import BayesianNetwork
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np



# サンプルの形は list[dict[VariableIDs, dict[int, valueClass]]] で、各時刻の値が入っている。


def _create_histogram_continuous(sample: list[dict[VariableIDs, dict[int, valueClass]]], var_id: VariableIDs):
    df = pd.DataFrame(
        {f"{var_id.name}(ID: {j})": [sample[j][var_id][i] for i in range(
            len(sample[0][var_id]))] for j in range(len(sample))}
    )
    df['time'] = df.index
    hist_data = {
        time: df[df['time'] == time].drop(
            columns='time').dropna(axis=1).values.flatten()
        for time in df['time']
    }
    all_data = np.concatenate(list(hist_data.values()))
    all_bins = np.histogram_bin_edges(all_data, bins='fd')

    # start with the first time point
    initial_time = df['time'].iloc[0]
    bin_size = all_bins[1] - all_bins[0]

    fig_hist = go.Figure()
    # add initial histogram trace
    fig_hist.add_trace(
        go.Histogram(
            x=hist_data[initial_time],
            xbins=dict(start=all_bins[0], end=all_bins[-1], size=bin_size),
            histnorm='probability',
            marker_color='rgba(31,119,180,0.7)'
        )
    )

    # create a button for each time point
    buttons = []

    for time, data in hist_data.items():
        # do not recompute bins per time if desired
        bins = all_bins  # np.histogram_bin_edges(data, bins='fd')
        size = bins[1] - bins[0]
        buttons.append(
            dict(
                method='update',
                label=f"time {str(time)}",
                args=[
                    {'x': [data],    # new data for the trace
                        'xbins': dict(start=bins[0], end=bins[-1], size=size)},
                    {'title': f"Distribution of {var_id.name} at time {time}"}
                ]
            )
        )

    updatemenus = [dict(
        active=0,
        buttons=buttons,
        x=0.1,
        y=1.15,
        xanchor='left',
        yanchor='top'
    )]
    if len(df['time']) == 1:
        updatemenus = []

    fig_hist.update_layout(
        title=f"Distribution of {var_id.name} at time {initial_time}",
        xaxis_title=var_id.name,
        yaxis_title="Probability",
        xaxis=dict(range=[all_bins[0], all_bins[-1]]),
        updatemenus=updatemenus
    )

    return fig_hist


def _create_histogram_categorical(sample: list[dict[VariableIDs, dict[int, valueClass]]], var_id: VariableIDs):
    df = pd.DataFrame(
        {f"{var_id.name}(ID: {j})": [sample[j][var_id][i] for i in range(
            len(sample[0][var_id]))] for j in range(len(sample))}
    )
    df['time'] = df.index
    # 各時刻のデータを文字列化してカテゴリとして扱う
    hist_data = {
        time: pd.Series(
            df[df['time'] == time].drop(columns='time').values.flatten()
        ).dropna().astype(str).values
        for time in df['time']
    }
    all_data = np.concatenate(list(hist_data.values())) if len(hist_data) > 0 else np.array([], dtype=str)
    # すべてのカテゴリ（文字列）を一意化して安定順序を確定
    all_categories = np.unique(all_data)

    # start with the first time point
    initial_time = df['time'].iloc[0]

    fig_hist = go.Figure()
    # x 軸をカテゴリとして扱い、順序を固定
    fig_hist.update_xaxes(
        type='category',
        categoryorder='array',
        categoryarray=all_categories.tolist()
    )
    # add initial histogram trace
    fig_hist.add_trace(
        go.Histogram(
            x=hist_data[initial_time],
            histnorm='probability',
            marker_color='rgba(31,119,180,0.7)'
        )
    )

    # create a button for each time point
    buttons = []
    for time, data in hist_data.items():
        buttons.append(
            dict(
                method='update',
                label=f"time {str(time)}",
                args=[
                    {'x': [data]},
                    {'title': f"Distribution of {var_id.name} at time {time}"}
                ]
            )
        )

    updatemenus = [dict(
        active=0,
        buttons=buttons,
        x=0.1,
        y=1.15,
        xanchor='left',
        yanchor='top'
    )]
    if len(df['time']) == 1:
        updatemenus = []

    fig_hist.update_layout(
        title=f"Distribution of {var_id.name} at time {initial_time}",
        xaxis_title=var_id.name,
        yaxis_title="Probability",
        updatemenus=updatemenus
    )

    return fig_hist


def _continuous_distribution_plot(sample: list[dict[VariableIDs, dict[int, valueClass]]],
                                  var_id: VariableIDs,
                                  time_varying: bool):
    all_figures = []
    if time_varying:
        df = pd.DataFrame(
            {f"{var_id.name}(ID: {j})": [sample[j][var_id][i] for i in range(
                len(sample[0][var_id]))] for j in range(len(sample))}
        )
        mean_df = df.mean(axis=1)
        std_df = df.std(axis=1)
        upper_df = mean_df + std_df
        lower_df = mean_df - std_df

        # plot for individuals
        df['time'] = df.index
        df_long = df.melt(id_vars=['time'],
                          var_name="Series", value_name="Value")
        fig = px.line(df_long,
                      x="time",
                      y="Value",
                      color="Series",
                      title=f"Individual time series of {var_id.name}"
                      )
        for trace in fig.data:
            trace.visible = "legendonly"

        all_figures.append(fig)

        # plot for group mean and std
        mean_traces = []
        mean_traces.append(
            go.Scatter(
                # Concatenate time points forward and backward
                x=np.concatenate([df['time'].values, df['time'].values[::-1]]),
                # Concatenate upper bound and reversed lower bound
                y=np.concatenate([upper_df.values, lower_df.values[::-1]]),
                fill='toself',  # Fill the area defined by the x and y points
                fillcolor='rgba(0,100,80,0.2)',  # Semi-transparent fill color
                # No line for the band's boundary
                line=dict(color='rgba(255,255,255,0)'),
                name='SD',  # Name for the legend
                hoverinfo='skip',  # Don't show hover info for the band itself
                showlegend=True,
                mode='none',  # No markers or lines for the band boundaries
            )
        )

        mean_traces.append(go.Scatter(
            x=df['time'].values,
            y=mean_df.values,
            mode='lines',
            # Darker line for the mean
            line=dict(color='rgb(0,100,80)', width=2),
            name='Mean',  # Name for the legend
            showlegend=True
        ))

        fig_mean = go.Figure(data=mean_traces)
        fig_mean.update_layout(
            title=f"Mean and SD of {var_id.name} over time",
            xaxis_title="Time",
            yaxis_title=var_id.name,
        )
        all_figures.append(fig_mean)

        # hist plot at each time point
        # prepare histogram data for each time point
        fig_hist = _create_histogram_continuous(sample, var_id)

        all_figures.append(fig_hist)
    else:
        fig_hist = _create_histogram_continuous(sample, var_id)
        all_figures.append(fig_hist)

    return all_figures

def _categorical_distribution_plot(sample: list[dict[VariableIDs, dict[int, valueClass]]],
                                   var_id: VariableIDs,
                                   time_varying: bool):
    all_figures = []
    if time_varying:
        # 個体×時刻の表を作成
        df = pd.DataFrame(
            {f"{var_id.name}(ID: {j})": [sample[j][var_id][i] for i in range(
                len(sample[0][var_id]))] for j in range(len(sample))}
        )
        df['time'] = df.index

        # 時刻ごとのデータ配列（NaN除去）を文字列化
        hist_data = {
            time: pd.Series(
                df[df['time'] == time].drop(columns='time').values.flatten()
            ).dropna().astype(str).values
            for time in df['time']
        }

        # カテゴリ一覧（文字列）
        all_data = np.concatenate(list(hist_data.values())) if len(hist_data) > 0 else np.array([], dtype=str)
        all_categories = np.unique(all_data)

        # 個体ごとのカテゴリ推移（y=カテゴリ, x=時間）
        df_long = df.melt(id_vars=['time'], var_name="Series", value_name="Value").dropna(subset=['Value'])
        df_long["Value"] = df_long["Value"].astype(str)
        fig_ind = px.line(
            df_long,
            x="time",
            y="Value",
            color="Series",
            line_shape="hv",
            markers=True,
            title=f"Individual category transitions of {var_id.name}"
        )
        # 混雑回避のため各系列は初期は非表示
        for tr in fig_ind.data:
            tr.visible = "legendonly"
        fig_ind.update_layout(
            yaxis=dict(
                type='category',
                categoryorder='array',
                categoryarray=all_categories.tolist()
            )
        )
        all_figures.append(fig_ind)

        # 各カテゴリの時刻ごとの比率を算出
        prop_df = pd.DataFrame(0.0, index=df['time'].values, columns=all_categories)
        for t, data in hist_data.items():
            if data.size == 0:
                continue
            counts = pd.Series(data).value_counts()
            total = float(data.size)
            for cat, cnt in counts.items():
                # cat は文字列
                if cat in prop_df.columns:
                    prop_df.loc[t, cat] = cnt / total

        # 比率の折れ線図（カテゴリは系列名）
        traces = []
        for cat in all_categories:
            traces.append(
                go.Scatter(
                    x=prop_df.index,
                    y=prop_df[cat],
                    mode='lines+markers',
                    name=str(cat),
                )
            )
        fig_prop = go.Figure(traces)
        fig_prop.update_layout(
            title=f"Category proportions of {var_id.name} over time",
            xaxis_title="Time",
            yaxis_title="Proportion",
            yaxis=dict(range=[0, 1]),
        )
        all_figures.append(fig_prop)

        # 時刻ごとのヒストグラム（切替ボタン付き）
        fig_hist = _create_histogram_categorical(sample, var_id)
        all_figures.append(fig_hist)
    else:
        # 非時間変動は単純なヒストグラム
        all_figures.append(_create_histogram_categorical(sample, var_id))

    return all_figures

def _binary_distribution_plot(sample: list[dict[VariableIDs, dict[int, valueClass]]],
                              var_id: VariableIDs,
                              time_varying: bool):
    all_figures = []
    if time_varying:
        # we first need to convert binary values to event time data
        # sample[i][var_id] is an array of 0s and 1s
        # we set the event time to the index of the first 1 in the array
        # if there is no 1, we set the event time to the last time point + 1

        event_times = []
        for i in range(len(sample)):
            event_time = len(sample[0][var_id])
            for j in range(len(sample[0][var_id])):
                if sample[i][var_id][j] == 1:
                    event_time = j
                    break
            event_times.append(event_time)

        # create a histogram of event times
        # bins are the unique event times
        all_bins = np.arange(len(sample[0][var_id]) + 2)
        fig_hist = go.Figure()
        fig_hist.add_trace(
            go.Histogram(
                x=event_times,
                xbins=dict(start=all_bins[0], end=all_bins[-1], size=1),
                histnorm='probability',
                marker_color='rgba(31,119,180,0.7)'
            )
        )

        fig_hist.update_layout(
            title=f"Distribution of {var_id.name} at time {0}",
            xaxis_title=var_id.name,
            yaxis_title="Probability",
            # set the bin labels to be the unique event times or no event
            xaxis=dict(
                tickvals=all_bins,
                ticktext=[str(i) for i in all_bins[:-1]] + ["No event"],
                categoryorder="array",
            )
        )
        all_figures.append(fig_hist)
    else:
        # just create a histogram of the binary values
        fig_hist = _create_histogram_categorical(sample, var_id)
        all_figures.append(fig_hist)

    return all_figures


def distribution_plot(
    sample: list[dict[VariableIDs, dict[int, valueClass]]],
    bn: BayesianNetwork
):
    all_figures = {}
    for var_id, var in bn.variable_ids_to_variables.items():
        all_figures[var_id] = []
        time_varying = len(sample[0][var_id]) > 1
        var_type = var.value_type
        if var_type == VariableTypes.CONTINUOUS:
            all_figures[var_id] = _continuous_distribution_plot(
                sample, var_id, time_varying)
        elif var_type in {VariableTypes.CATEGORICAL, VariableTypes.ORDERED_CATEGORICAL, VariableTypes.ORDERED_CONTINUOUS, VariableTypes.BINARY}:
            all_figures[var_id] = _categorical_distribution_plot(
                sample, var_id, time_varying)
        elif var_type == VariableTypes.EVENT_BINARY:
            all_figures[var_id] = _binary_distribution_plot(
                sample, var_id, time_varying)

    return all_figures
