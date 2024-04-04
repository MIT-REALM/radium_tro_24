import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data = [
    {
        "Algorithm": "DR",
        "predicted_failures": "results/formation2d_grad_norm_netconn/5/L_1.0e+00/3_T/250_samples/5_quench/no_temper5_chains/dp_1.0e-03/ep_1.0e-03/repair/no_predict/gd_predicted_failures.npz",  # noqa"
        "stress_test_results": "results/formation2d_grad_norm_netconn/5/L_1.0e+00/3_T/250_samples/5_quench/no_temper5_chains/dp_1.0e-03/ep_1.0e-03/repair/no_predict/gd_stress_test.npz",  # noqa
    },
    {
        "Algorithm": "GD",
        "predicted_failures": "results/formation2d_grad_norm_netconn/5/L_1.0e+00/3_T/250_samples/5_quench/no_temper5_chains/dp_1.0e-03/ep_1.0e-03/repair/predict/gd_predicted_failures.npz",  # noqa"
        "stress_test_results": "results/formation2d_grad_norm_netconn/5/L_1.0e+00/3_T/250_samples/5_quench/no_temper5_chains/dp_1.0e-03/ep_1.0e-03/repair/predict/gd_stress_test.npz",  # noqa
    },
    {
        "Algorithm": "Ours (RMH)",
        "predicted_failures": "results/formation2d_grad_norm_netconn/5/L_1.0e+00/3_T/250_samples/0_quench/no_temper5_chains/dp_1.0e-03/ep_1.0e-03/repair/predict/rmh_predicted_failures.npz",  # noqa"
        "stress_test_results": "results/formation2d_grad_norm_netconn/5/L_1.0e+00/3_T/250_samples/0_quench/no_temper5_chains/dp_1.0e-03/ep_1.0e-03/repair/predict/rmh_stress_test.npz",  # noqa
    },
    {
        "Algorithm": "Ours (MALA)",
        "predicted_failures": "results/formation2d_grad_norm_netconn/5/L_1.0e+00/3_T/250_samples/5_quench/no_temper5_chains/dp_1.0e-03/ep_1.0e-03/repair/predict/mala_predicted_failures.npz",  # noqa"
        "stress_test_results": "results/formation2d_grad_norm_netconn/5/L_1.0e+00/3_T/250_samples/5_quench/no_temper5_chains/dp_1.0e-03/ep_1.0e-03/repair/predict/mala_stress_test.npz",  # noqa
    },
]

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
    # plt.yscale("log")
    plt.legend(markerscale=1.5)
    plt.xlabel("")
    plt.gca().legend().remove()

    plt.show()
