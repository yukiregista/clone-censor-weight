import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    import yaml
    from matplotlib.lines import Line2D

    return Line2D, Path, mo, pd, plt, sns, yaml


@app.cell
def _(mo):
    metrics_path_text = mo.ui.text(
        label="Paper metrics CSV",
        value="output_diagnostics/summary/tables/paper_metrics_one_row_per_run.csv",
        full_width=True,
    )
    config_path_text = mo.ui.text(
        label="Figure config YAML",
        value="scripts/paper/figure_config.yaml",
        full_width=True,
    )
    selected_setting = mo.ui.text(label="Setting", value="a1d1")
    heatmap_sample_size = mo.ui.number(
        label="Heatmap sample size",
        value=10000,
        start=1,
    )
    figure_output_dir_text = mo.ui.text(
        label="Figure output directory",
        value="output_diagnostics/summary/figures/performance",
        full_width=True,
    )
    save_button = mo.ui.run_button(label="Build and save figures")
    controls_panel = mo.vstack(
        [
            metrics_path_text,
            config_path_text,
            selected_setting,
            heatmap_sample_size,
            figure_output_dir_text,
            save_button,
        ]
    )
    return (
        config_path_text,
        controls_panel,
        figure_output_dir_text,
        heatmap_sample_size,
        metrics_path_text,
        save_button,
        selected_setting,
    )


@app.cell(hide_code=True)
def _(controls_panel, mo):
    mo.vstack(
        [
            mo.md(
                """
                # Paper performance figures

                This notebook reads the table produced by
                `scripts/paper/4_build_paper_tables.py`. It creates the absolute-bias
                heatmap and the RMSE, absolute-bias, and coverage curves used to
                review the paper experiments.
                """
            ),
            controls_panel,
        ]
    )
    return


@app.cell
def _(Path, config_path_text, mo, plt, yaml):
    default_figure_config = {
        "global_style": {
            "font_family": "DejaVu Sans",
            "default_font_size": 9,
            "dpi": 300,
        },
        "bias_heatmap": {
            "width_mm": 203.2,
            "height_mm": 127.0,
            "cmap": "Reds",
            "annotation_font_size": 8,
            "label_font_size": 10,
            "tick_font_size": 9,
            "x_tick_rotation": 35,
        },
        "sample_size_panels": {
            "width_mm": 406.4,
            "height_mm": 203.2,
            "title_font_size": 10,
            "label_font_size": 10,
            "tick_font_size": 9,
        },
        "experiment_legend": {
            "width_mm": 203.2,
            "height_mm": 15.24,
            "font_size": 8,
        },
    }

    def merge_config(base, override):
        merged = dict(base)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = merge_config(merged[key], value)
            else:
                merged[key] = value
        return merged

    figure_config_path = Path(str(config_path_text.value)).expanduser()
    if figure_config_path.is_file():
        loaded_figure_config = yaml.safe_load(figure_config_path.read_text()) or {}
        figure_config = merge_config(default_figure_config, loaded_figure_config)
        config_status = mo.md(f"Loaded figure config: `{figure_config_path}`")
    else:
        figure_config = default_figure_config
        config_status = mo.md(
            f"Figure config not found; using notebook defaults: `{figure_config_path}`"
        )

    global_style = figure_config["global_style"]
    plt.rcParams.update(
        {
            "font.family": str(global_style["font_family"]),
            "font.size": float(global_style["default_font_size"]),
        }
    )
    config_status
    return (figure_config,)


@app.cell
def _(Path, metrics_path_text, mo, pd):
    paper_metrics_path = Path(str(metrics_path_text.value)).expanduser()
    if paper_metrics_path.is_file():
        paper_metrics = pd.read_csv(paper_metrics_path)
        metrics_status = mo.md(
            f"Loaded `{paper_metrics_path}` with `{len(paper_metrics)}` rows."
        )
    else:
        paper_metrics = pd.DataFrame()
        metrics_status = mo.md(f"Missing metrics table: `{paper_metrics_path}`")
    metrics_status
    return (paper_metrics,)


@app.cell
def _(paper_metrics, metrics_path_text, mo):
    mo.vstack(
        [
            mo.md(f"## Table preview\n\nSource: `{metrics_path_text.value}`"),
            paper_metrics.head(200),
        ]
    )
    return


@app.cell
def _(
    Line2D,
    Path,
    figure_config,
    figure_output_dir_text,
    heatmap_sample_size,
    mo,
    paper_metrics,
    pd,
    plt,
    save_button,
    selected_setting,
    sns,
):
    def figure_size(config):
        return float(config["width_mm"]) / 25.4, float(config["height_mm"]) / 25.4

    def experiment_label(value):
        text = str(value)
        return f"exp. {text.removeprefix('experiment')}" if text.startswith("experiment") else text

    figures = {}
    figure_messages = []
    saved_figure_paths = []

    selected_metrics = paper_metrics.copy()
    if not selected_metrics.empty:
        if "experiment_display" not in selected_metrics.columns:
            selected_metrics["experiment_display"] = selected_metrics["experiment"]
        selected_metrics = selected_metrics[
            selected_metrics["setting"].astype(str) == str(selected_setting.value)
        ].copy()

    if selected_metrics.empty:
        figure_messages.append(f"No rows found for setting {selected_setting.value!r}.")
    else:
        experiment_values = sorted(
            str(value) for value in selected_metrics["experiment_display"].dropna().unique()
        )
        palette_values = sns.color_palette(n_colors=max(len(experiment_values), 1))
        experiment_palette = {
            experiment: palette_values[index]
            for index, experiment in enumerate(experiment_values)
        }

        legend_config = figure_config["experiment_legend"]
        legend_figure, legend_axis = plt.subplots(figsize=figure_size(legend_config))
        legend_axis.axis("off")
        legend_handles = [
            Line2D(
                [0],
                [0],
                color=experiment_palette[experiment],
                marker="o",
                linewidth=1.4,
                label=experiment,
            )
            for experiment in experiment_values
        ]
        if legend_handles:
            legend_axis.legend(
                handles=legend_handles,
                loc="center",
                ncol=len(legend_handles),
                frameon=False,
                fontsize=float(legend_config["font_size"]),
            )
        legend_figure.tight_layout(pad=0.05)
        figures["experiment_legend"] = legend_figure

        heatmap_rows = selected_metrics[
            selected_metrics["sample_size"] == int(heatmap_sample_size.value)
        ].copy()
        heat_matrix = pd.DataFrame(
            index=[experiment_label(value) for value in experiment_values]
        )
        for strategy, strategy_label in (("control", "Ctrl"), ("intervention", "Interv.")):
            bias_column = f"mortality_{strategy}_bias"
            if bias_column not in heatmap_rows.columns:
                continue
            for cutoff in sorted(int(value) for value in heatmap_rows["cutoff"].dropna().unique()):
                values = (
                    heatmap_rows.loc[
                        heatmap_rows["cutoff"] == cutoff,
                        ["experiment_display", bias_column],
                    ]
                    .dropna(subset=["experiment_display"])
                    .set_index("experiment_display")[bias_column]
                    .abs()
                )
                values.index = [experiment_label(value) for value in values.index]
                heat_matrix[f"{strategy_label}, cut {cutoff}"] = values

        if not heat_matrix.empty and heat_matrix.shape[1] > 0:
            heatmap_config = figure_config["bias_heatmap"]
            heatmap_figure, heatmap_axis = plt.subplots(figsize=figure_size(heatmap_config))
            sns.heatmap(
                heat_matrix,
                annot=True,
                fmt=".4f",
                cmap=str(heatmap_config["cmap"]),
                cbar=True,
                ax=heatmap_axis,
                annot_kws={"fontsize": float(heatmap_config["annotation_font_size"])},
            )
            heatmap_axis.set_xlabel(
                "Absolute mortality bias",
                fontsize=float(heatmap_config["label_font_size"]),
            )
            heatmap_axis.set_ylabel("")
            heatmap_axis.tick_params(
                axis="both",
                labelsize=float(heatmap_config["tick_font_size"]),
                length=0,
            )
            heatmap_axis.set_xticklabels(
                heatmap_axis.get_xticklabels(),
                rotation=float(heatmap_config["x_tick_rotation"]),
                ha="right",
                rotation_mode="anchor",
            )
            heatmap_figure.tight_layout()
            figures["absolute_bias_heatmap"] = heatmap_figure
        else:
            figure_messages.append(
                f"No heatmap rows found for sample size {int(heatmap_sample_size.value)}."
            )

        def line_grid(metric_suffix, y_label, *, absolute=False, y_limits=None):
            panel_config = figure_config["sample_size_panels"]
            panel_figure, panel_axes = plt.subplots(
                nrows=2,
                ncols=3,
                figsize=figure_size(panel_config),
                sharex=True,
            )
            for row_index, strategy in enumerate(("control", "intervention")):
                metric_column = f"mortality_{strategy}_{metric_suffix}"
                for column_index, cutoff in enumerate((0, 2, 4)):
                    axis = panel_axes[row_index, column_index]
                    plot_data = selected_metrics[selected_metrics["cutoff"] == cutoff].copy()
                    if metric_column not in plot_data.columns or plot_data.empty:
                        axis.text(0.5, 0.5, "No rows", ha="center", va="center")
                    else:
                        plotted_column = metric_column
                        if absolute:
                            plotted_column = f"{metric_column}_absolute"
                            plot_data[plotted_column] = plot_data[metric_column].abs()
                        sns.lineplot(
                            data=plot_data,
                            x="sample_size",
                            y=plotted_column,
                            hue="experiment_display",
                            hue_order=experiment_values,
                            palette=experiment_palette,
                            marker="o",
                            legend=False,
                            ax=axis,
                        )
                    axis.set_xscale("log")
                    axis.set_title(
                        f"{strategy.capitalize()}, cutoff {cutoff}",
                        fontsize=float(panel_config["title_font_size"]),
                    )
                    axis.set_xlabel("Sample size", fontsize=float(panel_config["label_font_size"]))
                    axis.set_ylabel(y_label, fontsize=float(panel_config["label_font_size"]))
                    axis.tick_params(
                        axis="both",
                        labelsize=float(panel_config["tick_font_size"]),
                    )
                    if y_limits is not None:
                        axis.set_ylim(*y_limits)
            panel_figure.tight_layout()
            return panel_figure

        figures["rmse_by_sample_size"] = line_grid("rmse", "Mortality RMSE")
        figures["absolute_bias_by_sample_size"] = line_grid(
            "bias",
            "Absolute mortality bias",
            absolute=True,
        )
        figures["coverage_by_sample_size"] = line_grid(
            "coverage",
            "Mortality coverage",
            y_limits=(0, 1),
        )

    if save_button.value and figures:
        output_directory = Path(str(figure_output_dir_text.value)).expanduser()
        output_directory.mkdir(parents=True, exist_ok=True)
        dpi = int(figure_config["global_style"]["dpi"])
        setting_slug = "".join(
            character if character.isalnum() or character in ".-_" else "_"
            for character in str(selected_setting.value)
        ).strip("_") or "setting"
        for _figure_name, _figure in figures.items():
            suffix = (
                f"_n{int(heatmap_sample_size.value)}"
                if _figure_name == "absolute_bias_heatmap"
                else ""
            )
            output_path = output_directory / f"{_figure_name}_{setting_slug}{suffix}.png"
            _figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
            saved_figure_paths.append(output_path.resolve())
        figure_messages.append(f"Saved {len(saved_figure_paths)} figures to {output_directory.resolve()}.")

    figure_status = mo.md(
        "\n".join(f"- {message}" for message in figure_messages)
        if figure_messages
        else "Figures are ready. Click **Build and save figures** to write PNG files."
    )
    return figure_status, figures, saved_figure_paths


@app.cell
def _(figure_status, figures, mo, saved_figure_paths):
    figure_items = []
    for _display_name, _display_figure in figures.items():
        figure_items.extend(
            [mo.md(f"## {_display_name.replace('_', ' ').title()}"), _display_figure]
        )
    saved_text = "\n".join(f"- `{path}`" for path in saved_figure_paths)
    mo.vstack(
        [
            figure_status,
            mo.md(f"### Saved files\n{saved_text}" if saved_text else "### Saved files\nNone yet."),
            *figure_items,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
