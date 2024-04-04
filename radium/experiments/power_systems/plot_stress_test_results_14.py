import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data = [
    {
        "Algorithm": "DR",
        "Worst case (predicted)": 0.0,
        "Worst case (observed)": 2.689948,
        "Likelihood of underestimate": 0.33210,
        "generation_cost": 0.54969215,
        "predicted_failures": "results/acopf/case14/L_1.0e+02_1000_samples_10_quench_10_chains_step_dp_1.0e-06_ep_1.0e-02_gd_predicted_failures.npz",  # noqa"
        "stress_test_results": "results/acopf/case14/L_1.0e+02_1000_samples_10_quench_10_chains_step_dp_1.0e-06_ep_1.0e-02_gd_stress_test.npz",  # noqa
    },
    {
        "Algorithm": "GD",
        "Worst case (predicted)": 0.0,
        "Worst case (observed)": 2.6572380,
        "Likelihood of underestimate": 0.17741300,
        "generation_cost": 0.549297,
        "predicted_failures": "results/scacopf/case14/L_1.0e+02_1000_samples_10_quench_10_chains_step_dp_1.0e-06_ep_1.0e-02_gd_predicted_failures.npz",  # noqa"
        "stress_test_results": "results/scacopf/case14/L_1.0e+02_1000_samples_10_quench_10_chains_step_dp_1.0e-06_ep_1.0e-02_gd_stress_test.npz",  # noqa
    },
    {
        "Algorithm": "Ours (RMH)",
        "Worst case (predicted)": 3.318828105,
        "Worst case (observed)": 5.32660293,
        "Likelihood of underestimate": 0.0002,
        "generation_cost": 1.487012,
        "predicted_failures": "results/scacopf/case14/L_1.0e+02_1000_samples_10_quench_10_chains_step_dp_1.0e-06_ep_1.0e-02_rmh_predicted_failures.npz",  # noqa"
        "stress_test_results": "results/scacopf/case14/L_1.0e+02_1000_samples_10_quench_10_chains_step_dp_1.0e-06_ep_1.0e-02_rmh_stress_test.npz",  # noqa
    },
    {
        "Algorithm": "Ours (MALA)",
        "Worst case (predicted)": 0.35271,
        "Worst case (observed)": 0.2735613,
        "Likelihood of underestimate": 0.00,
        "generation_cost": 1.4132679,
        "predicted_failures": "results/scacopf/case14/L_1.0e+02_1000_samples_10_quench_10_chains_step_dp_1.0e-06_ep_1.0e-02_mala_predicted_failures.npz",  # noqa"
        "stress_test_results": "results/scacopf/case14/L_1.0e+02_1000_samples_10_quench_10_chains_step_dp_1.0e-06_ep_1.0e-02_mala_stress_test.npz",  # noqa
    },
]

L = 100

tidy_data = []
for packet in data:
    tidy_data.append(
        {
            "Algorithm": packet["Algorithm"],
            "Constraint violation": packet["Worst case (predicted)"],
            "Type": "Predicted",
        }
    )
    tidy_data.append(
        {
            "Algorithm": packet["Algorithm"],
            "Constraint violation": packet["Worst case (observed)"],
            "Type": "Observed",
        }
    )

if __name__ == "__main__":
    results = pd.DataFrame(data)

    sns.set_theme(style="whitegrid", context="paper", font_scale=2.0)
    fig = plt.figure(figsize=(9, 4), constrained_layout=True)

    # Load distributions from files
    stress_test_potentials = []
    predicted_potentials = []
    potentials = pd.Series([], dtype=float)
    algs = pd.Series([], dtype=str)
    types = pd.Series([], dtype=str)
    for entry in data:
        # Load predicted failures
        new_potentials = jnp.load(entry["predicted_failures"])
        N = new_potentials.shape[0]
        potentials = pd.concat(
            [potentials, pd.Series(new_potentials)],
            ignore_index=True,
        )
        algs = pd.concat(
            [algs, pd.Series([entry["Algorithm"] for _ in range(N)])], ignore_index=True
        )
        types = pd.concat(
            [types, pd.Series(["Predicted" for _ in range(N)])], ignore_index=True
        )

        # Load stress test results
        new_potentials = jnp.load(entry["stress_test_results"])
        N = new_potentials.shape[0]
        potentials = pd.concat(
            [potentials, pd.Series(new_potentials)],
            ignore_index=True,
        )
        algs = pd.concat(
            [algs, pd.Series([entry["Algorithm"] for _ in range(N)])], ignore_index=True
        )
        types = pd.concat(
            [types, pd.Series(["Observed" for _ in range(N)])], ignore_index=True
        )

    plotting_df = pd.DataFrame()
    plotting_df["Algorithm"] = algs
    plotting_df["Type"] = types
    plotting_df["Cost"] = potentials

    # Plot the distributions of potentials
    observed_df = plotting_df[plotting_df["Type"] == "Observed"]
    sns.boxenplot(
        x="Algorithm",
        y="Cost",
        hue="Type",
        showfliers=False,
        outlier_prop=1e-7,
        # flier_kws={"s": 20},
        data=observed_df,
    )

    predicted_df = plotting_df[plotting_df["Type"] == "Predicted"]
    min_predicted = predicted_df.groupby("Algorithm", sort=False)["Cost"].min()
    max_predicted = predicted_df.groupby("Algorithm", sort=False)["Cost"].max()
    mean_predicted = predicted_df.groupby("Algorithm", sort=False)["Cost"].mean()
    plt.scatter(
        min_predicted.index,
        min_predicted,
        marker="^",
        s=50,
        linewidth=2,
        color="orangered",
    )
    plt.scatter(
        mean_predicted.index,
        mean_predicted,
        marker="_",
        s=100,
        linewidth=2,
        color="orangered",
    )
    plt.scatter(
        max_predicted.index,
        max_predicted,
        marker="v",
        s=50,
        linewidth=2,
        color="orangered",
    )
    plt.scatter(
        [],
        [],
        # marker="$⧗$",
        marker=r"$\frac{\blacktriangledown}{\blacktriangle}$",
        label="Predicted min/mean/max",
        color="orangered",
        s=50,
    )

    plt.ylabel("Cost")
    plt.yscale("log")
    plt.legend(markerscale=1.5)
    plt.xlabel("")
    plt.gca().legend().remove()

    plt.show()
