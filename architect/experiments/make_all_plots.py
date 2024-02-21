import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import wandb

if __name__ == "__main__":
    # Define your list of wandb projects
    project_list = [
        [
            "tro2-scopf-case14",
            "tro2-scopf-case57",
        ],
        [
            "tro2-formation-collision-halfkernel-5-agents",
            "tro2-formation-collision-halfkernel-10-agents",
        ],
        [
            "tro2-hideseek2-5-hiders-3-seekers-clipped",
            "tro2-hideseek2-20-hiders-12-seekers",
        ],
        [
            "tro4-highway",
            "tro4-intersection",
        ],
    ]
    projects_unwrapped = [project for sublist in project_list for project in sublist]

    # Define your filters for each project
    filters = {
        "tro2-scopf-case14": {
            "$and": [
                {"config.L": 500.0},
                {
                    "$or": [
                        {"group": {"$ne": "mala-predict-repair"}},
                        {"config.temper": False},
                    ]
                },
            ]
        },
        "tro2-scopf-case57": {
            "$and": [
                {"config.failure_level": 6.0},
                {"config.num_rounds": 50},
                {"config.ep_mcmc_step_size": 1e-4},
                {"config.L": 500.0},
                {"config.grad_clip": 100.0},
                {
                    "$or": [
                        {"group": {"$ne": "mala-predict-repair"}},
                        {"config.temper": False},
                    ]
                },
            ]
        },
        "tro2-formation-collision-halfkernel-5-agents": {
            "$and": [
                {"config.grad_clip": 100.0},
                {"config.max_wind_thrust": 1.0},
                {"config.ep_mcmc_step_size": 1e-5},
            ]
        },
        "tro2-formation-collision-halfkernel-10-agents": {
            "$and": [
                {"config.grad_clip": 100.0},
                {"config.max_wind_thrust": 1.0},
                {
                    "$or": [
                        {"group": {"$ne": "mala-predict-repair"}},
                        {"config.temper": True},
                    ]
                },
            ]
        },
        "tro2-hideseek2-5-hiders-3-seekers-clipped": {},
        "tro2-hideseek2-20-hiders-12-seekers": {},
        "tro4-highway": {},
        "tro4-intersection": {},
    }

    # Define a display name for each project
    project_display_names = {
        "tro2-scopf-case14": "Grid 14",
        "tro2-scopf-case57": "Grid 57",
        "tro2-formation-collision-halfkernel-5-agents": "Formation 5",
        "tro2-formation-collision-halfkernel-10-agents": "Formation 10",
        "tro2-hideseek2-5-hiders-3-seekers-clipped": "Search 3v5",
        "tro2-hideseek2-20-hiders-12-seekers": "Search 12v20",
        "tro4-highway": "AV (highway)",
        "tro4-intersection": "AV (intersection)",
    }

    # Define cost y limits for each project
    max_cost_y_limits = {
        "Grid": ("log", (5e-4, 1.5)),
        "Formation": ("linear", (0.0, 1.1)),
        "Search": ("linear", (0.0, 1.1)),
        "AV": ("linear", (0.0, 1.1)),
    }
    mean_cost_y_limits = {
        "Grid": ("log", (5e-4, 1.5)),
        "Formation": ("linear", (0.0, 0.2)),
        "Search": ("linear", (0.0, 0.4)),
        "AV": ("linear", (0.0, 1.1)),
    }

    # Define groups and diplay names
    group_display_names = {
        "reinforce-predict-repair": "L2C",
        "rmh-predict-repair": "R0",
        "gd-repair": "GD-DR",
        "gd-predict-repair": "GD-ADV",
        "mala-predict-repair": "R1",
    }
    groups = list(group_display_names.keys())
    n_groups = len(groups)

    # Define metrics of interest
    metrics = [
        "Failure rate/test",
        "Test Cost Percentiles/99.00",
        "Mean Cost/test",
        "Predicted Cost Percentiles/99.00",
    ]

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

            summary_stats.append(
                {
                    "project": project_display_names[project],
                    "group": group_display_names[run.group],
                }
                | {metric: run.summary.get(metric) for metric in metrics}
            )

    stats_df = pd.DataFrame(summary_stats)

    # Normalize all costs by the maximum test cost
    max_test_cost = stats_df.groupby("project")["Test Cost Percentiles/99.00"].max()
    min_test_cost = stats_df.groupby("project")["Mean Cost/test"].min()
    min_test_cost = min_test_cost - 0.1 * min_test_cost.abs()  # Add a small buffer
    stats_df["Test Cost Percentiles/99.00"] = stats_df.apply(
        lambda row: (row["Test Cost Percentiles/99.00"] - min_test_cost[row["project"]])
        / (max_test_cost[row["project"]] - min_test_cost[row["project"]]),
        axis=1,
    )
    stats_df["Mean Cost/test"] = stats_df.apply(
        lambda row: (row["Mean Cost/test"] - min_test_cost[row["project"]])
        / (max_test_cost[row["project"]] - min_test_cost[row["project"]]),
        axis=1,
    )
    stats_df["Predicted Cost Percentiles/99.00"] = stats_df.apply(
        lambda row: (
            row["Predicted Cost Percentiles/99.00"] - min_test_cost[row["project"]]
        )
        / (max_test_cost[row["project"]] - min_test_cost[row["project"]]),
        axis=1,
    )

    # # Compute some summary statistics
    # stats_df["Predicted Cost Percentiles/99.00 (normalized)"] = (
    #     stats_df["Predicted Cost Percentiles/99.00"]
    #     / stats_df["Test Cost Percentiles/99.00"]
    # )

    # Remove the un-normalized predicted cost
    stats_df = stats_df.drop(columns=["Predicted Cost Percentiles/99.00"])

    # Replace zero test failure rates with 10^-3
    stats_df["Failure rate/test"] = stats_df["Failure rate/test"].replace(0, 1e-3)

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
        for ax in row[:-1]:  # TODO log scale all plots
            ax.set_yscale("log", nonpositive="clip")

    # Set failure rate plots to a consistent scale
    fail_indices = [
        i
        for i, metric in enumerate(stats_df["metric"].unique())
        if "Failure rate" in metric
    ]
    for i in fail_indices:
        for ax in g.axes[i, :-1]:  # TODO log scale all plots
            ax.set_ylim(5e-4, 1.5)

    # Set consistent cost scale for projects in the same domain
    cost_row_indices = [
        i
        for i, metric in enumerate(stats_df["metric"].unique())
        if "Percentile" in metric
    ]
    for i in cost_row_indices:
        for ax in g.axes[i, :-1]:
            title = ax.get_title()
            for domain, (scale, limits) in max_cost_y_limits.items():
                if domain in title:
                    ax.set_yscale(scale)
                    ax.set_ylim(*limits)

    cost_row_indices = [
        i
        for i, metric in enumerate(stats_df["metric"].unique())
        if "Mean Cost" in metric
    ]
    for i in cost_row_indices:
        for ax in g.axes[i, :-1]:
            title = ax.get_title()
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

    # Label y axes
    for ax, metric_name in zip(g.axes[:, 0], stats_df["metric"].unique()):
        ax.set_ylabel(metric_name)

    # Remove titles from axes not in the top row
    for ax in g.axes[1, :]:
        ax.set_title("")

    fig = g.figure
    fig.tight_layout()
    plt.savefig("paper_plots/debug.png")
