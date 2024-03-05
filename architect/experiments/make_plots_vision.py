import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import wandb

if __name__ == "__main__":
    # Define your list of wandb projects
    project_list = [
        [
            "highway_walls-predict-repair-noise_1.0",
            "iclr_intersection_sharp0.5-predict-repair-5x10",
        ],
        [
            "drone-predict-repair-10x10",
        ],
        [
            "grasping_0.003-predict-repair-5x10",
        ],
    ]
    projects_unwrapped = [project for sublist in project_list for project in sublist]

    # Define your filters for each project
    filters = {
        "highway_walls-predict-repair-noise_1.0": {
            "$and": [
                {"config.num_rounds": 10},
                {"config.dp_mcmc_step_size": 2e-5},
                {"config.ep_mcmc_step_size": 1e-2},
                {"config.L": {"$ne": 5}},
            ]
        },
        "iclr_intersection_sharp0.5-predict-repair-5x10": {},
        "drone-predict-repair-10x10": {
            "$and": [
                {"config.L": {"$ne": 5}},
                {"config.L": {"$ne": 20}},
                {
                    "$or": [
                        {"group": "reinforce_l2c_0.05_step_lr_1.0e-02/2.0e-04"},
                        {"group": "rmh_lr_1.0e-02/2.0e-04"},
                        {"group": "mala_lr_1.0e-02/2.0e-04_clip_1.0e+00"},
                        {"group": "gd_lr_1.0e-02/1.0e-04_clip_1.0e+00"},
                    ]
                },
            ]
        },
        "grasping_0.003-predict-repair-5x10": {},
    }

    # Define a display name for each project
    project_display_names = {
        "highway_walls-predict-repair-noise_1.0": "AV (highway)",
        "iclr_intersection_sharp0.5-predict-repair-5x10": "AV (intersection)",
        "drone-predict-repair-10x10": "Drone (static)",
        "grasping_0.003-predict-repair-5x10": "Grasping",
    }

    # Define cost y limits for each project
    max_cost_y_limits = {
        "AV": ("linear", (0.35, 1.0)),
        "Drone": ("linear", (0.4, 1.0)),
        "Grasping": ("linear", (-0.4, 1.1)),
    }
    mean_cost_y_limits = {
        "AV": ("linear", (0.35, 1.0)),
        "Drone": ("linear", (-0.4, 0.2)),
        "Grasping": ("linear", (-1.0, -0.5)),
    }

    # Define groups and diplay names
    group_display_names = {
        "reinforce-predict-repair": "L2C",
        "rmh-predict-repair": "R0",
        # "original": "$\theta_0$ (DR)",
        "gd": "ADV",
        "mala-predict-repair": "R1",
    }
    groups = list(group_display_names.keys())
    n_groups = len(groups)

    # Define metrics of interest
    metrics = {
        "Failure rate (test)": "Failure rate",
        "Mean Cost": "Mean Cost",
        "Max Cost": "Max Cost",
    }

    # Initialize a dictionary to store the summary statistics for each run
    summary_stats = []

    # Initialize the wandb API
    api = wandb.Api()

    # Loop through each project
    for project in projects_unwrapped:
        print(project)
        # Get all runs for the current project
        runs = api.runs(project, filters=filters[project], per_page=100)

        if len(runs) != n_groups * 4:  # Standard number of seeds
            print(
                f"WARN: {project} has {len(runs)} runs, do you have the right filters?"
            )

        # Extract summary statistics for each run
        for run in runs:
            # If there is no exact match for group, match on the first word
            if run.group not in groups:
                run.group = re.split("-|_", run.group)[0]

                for group in groups:
                    if run.group in group:
                        run.group = group
                        break

            project_name = project_display_names[project]
            if "grasping" in project:
                project_name += f" ({run.config['object_type']})"

            summary_stats.append(
                {
                    "project": project_name,
                    "group": group_display_names[run.group],
                }
                | {name: run.summary.get(metric) for metric, name in metrics.items()}
            )

            # # Get statistic for DR/original policy by getting the history
            # # at the first step
            # metric_history = run.history(samples=100)
            # summary_stats.append(
            #     {
            #         "project": project_name,
            #         "group": "original",
            #     }
            #     | {
            #         name: metric_history.iloc[0][metric]
            #         for metric, name in metrics.items()
            #     }
            # )

    stats_df = pd.DataFrame(summary_stats)

    # Normalize all costs by the maximum test cost
    max_test_cost = stats_df.groupby("project")[metrics["Max Cost"]].max()
    min_test_cost = 0.5 * max_test_cost  # Use 50% of max cost as min cost
    stats_df[metrics["Max Cost"]] = stats_df.apply(
        lambda row: (row[metrics["Max Cost"]] - min_test_cost[row["project"]])
        / (max_test_cost[row["project"]] - min_test_cost[row["project"]]),
        axis=1,
    )
    stats_df[metrics["Mean Cost"]] = stats_df.apply(
        lambda row: (row[metrics["Mean Cost"]] - min_test_cost[row["project"]])
        / (max_test_cost[row["project"]] - min_test_cost[row["project"]]),
        axis=1,
    )

    # Replace zero test failure rates with 10^-3
    stats_df[metrics["Failure rate (test)"]] = stats_df[
        metrics["Failure rate (test)"]
    ].replace(0, 1e-3)

    # Melt the dataframe for use with seaborn
    stats_df = stats_df.melt(
        id_vars=["project", "group"],
        value_vars=[
            col for col in stats_df.columns if col != "project" and col != "group"
        ],
        var_name="metric",
        value_name="value",
    )

    g = sns.FacetGrid(
        stats_df, row="metric", col="project", height=3, aspect=1, sharey=False
    )
    g.set_titles(template="{col_name}")
    g.map_dataframe(
        sns.pointplot,
        data=stats_df,
        x="group",
        y="value",
        hue="group",
        linestyles="none",
        order=group_display_names.values(),
        hue_order=group_display_names.values(),
        palette="colorblind",
    )
    g.set_axis_labels("", "")
    # g.add_legend()
    # sns.move_legend(
    #     g.axes[0][0],
    #     "upper center",
    #     ncols=len(groups),
    #     bbox_to_anchor=(0.5, 1.15),
    #     title=None,
    # )

    # Make all plots log scale
    for row in g.axes:
        for ax in row:
            ax.set_yscale("log", nonpositive="clip")

    # # Set failure rate plots to a consistent scale
    # fail_indices = [
    #     i
    #     for i, metric in enumerate(stats_df["metric"].unique())
    #     if "Failure rate" in metric
    # ]
    # for i in fail_indices:
    #     for ax in g.axes[i, :]:
    #         ax.set_ylim(1e-2, 1.5)

    # Set consistent cost scale for projects in the same domain
    cost_row_indices = [
        i
        for i, metric in enumerate(stats_df["metric"].unique())
        if "Max Cost" in metric
    ]
    for i in cost_row_indices:
        for j, ax in enumerate(g.axes[i]):
            title = g.axes[i, j].get_title()
            for domain, (scale, limits) in max_cost_y_limits.items():
                if domain in title:
                    ax.set_yscale(scale)
                    ax.set_ylim(*limits)

    # Set consistent cost scale for projects in the same domain
    cost_row_indices = [
        i
        for i, metric in enumerate(stats_df["metric"].unique())
        if "Mean Cost" in metric
    ]
    for i in cost_row_indices:
        for j, ax in enumerate(g.axes[i]):
            title = g.axes[i, j].get_title()
            for domain, (scale, limits) in mean_cost_y_limits.items():
                if domain in title:
                    ax.set_yscale(scale)
                    ax.set_ylim(*limits)

    # # Add a black line at 0 for the normalized predicted cost
    # pred_cost_indices = [
    #     i
    #     for i, metric in enumerate(stats_df["metric"].unique())
    #     if "Predicted" in metric
    # ]
    # for i in pred_cost_indices:
    #     for ax in g.axes[i, :]:
    #         ax.axhline(1, color="black", linestyle="--")
    #         ax.set_yscale("linear")

    for row in g.axes:
        for ax in row:
            ax.axvline(1.5, color="grey", linestyle="--", alpha=0.5)

    # Label y axes
    for ax, metric_name in zip(g.axes[:, 0], stats_df["metric"].unique()):
        ax.set_ylabel(metric_name)

    # Remove titles from axes not in the top row
    for row in g.axes[1:, :]:
        for ax in row:
            ax.set_title("")

    # Increase title font size
    for ax in g.axes[0, :]:
        ax.title.set_fontsize(16)

    # Increase y label font size
    for ax in g.axes[:, 0]:
        ax.yaxis.label.set_fontsize(16)

    # Increase x tick label font size
    for ax in g.axes[-1, :]:
        ax.tick_params(axis="x", labelsize=12)

    fig = g.figure
    fig.tight_layout()
    plt.savefig("paper_plots/vision.png", dpi=300)
