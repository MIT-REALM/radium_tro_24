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
    }

    # Define a display name for each project
    project_display_names = {
        "tro2-scopf-case14": "Grid 14",
        "tro2-scopf-case57": "Grid 57",
        "tro2-formation-collision-halfkernel-5-agents": "Formation 5",
        "tro2-formation-collision-halfkernel-10-agents": "Formation 10",
        "tro2-hideseek2-5-hiders-3-seekers-clipped": "Search 3v5",
        "tro2-hideseek2-20-hiders-12-seekers": "Search 12v20",
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

    # Define metrics of interest
    metrics = [
        "Failure rate/test",
        "Mean Cost/test",
        "Test Cost Percentiles/99.00",
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

        if len(runs) != len(groups) * 4:  # Standard number of seeds
            print(
                f"WARN: {project} has {len(runs)} runs, do you have the right filters?"
            )

        # Extract summary statistics for each run
        for run in runs:
            summary_stats.append(
                {
                    "project": project_display_names[project],
                    "group": group_display_names[run.group],
                }
                | {metric: run.summary.get(metric) for metric in metrics}
            )

    stats_df = pd.DataFrame(summary_stats)

    # Compute some summary statistics
    stats_df["Predicted Cost Percentiles/99.00 (normalized)"] = (
        stats_df["Predicted Cost Percentiles/99.00"]
        / stats_df["Test Cost Percentiles/99.00"]
    )

    # Remove the un-normalized predicted cost
    stats_df = stats_df.drop(columns=["Predicted Cost Percentiles/99.00"])

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

    # Make costs for grid projects log scale
    grid_indices = [
        i for i, project in enumerate(stats_df["project"].unique()) if "Grid" in project
    ]
    for i in grid_indices:
        for ax in g.axes[:, i]:
            ax.set_yscale("log")

    # Add a black line at 0 for the normalized predicted cost
    pred_cost_indices = [
        i
        for i, metric in enumerate(stats_df["metric"].unique())
        if "Predicted" in metric
    ]
    for i in pred_cost_indices:
        for ax in g.axes[i, :]:
            ax.axhline(1, color="black", linestyle="--")

    # Label y axes
    for ax, metric_name in zip(g.axes[:, 0], stats_df["metric"].unique()):
        ax.set_ylabel(metric_name)

    # Remove titles from axes not in the top row
    for ax in g.axes[1, :]:
        ax.set_title("")

    fig = g.figure
    fig.tight_layout()
    plt.savefig("paper_plots/debug.png")
