import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data = [
    {
        "Algorithm": "GD",
        "Worst case (predicted)": 0.0,
        "Worst case (observed)": 2.6572380,
        "Likelihood of underestimate": 0.17741300,
        "generation_cost": 0.549297,
        "predicted_failures": "results/scacopf/case14/L_1.0e+02_1000_samples_10_quench_10_chains_step_dp_1.0e-06_ep_1.0e-02_gd_predicted_failures.npz",  # noqa"
        "stress_test_results": "results/scacopf/case14/L_1.0e+02_1000_samples_10_quench_10_chains_step_dp_1.0e-06_ep_1.0e-02_gd_stress_test.npz",  # noqa
        "line_states_results": "results/scacopf/case14/L_1.0e+02_1000_samples_10_quench_10_chains_step_dp_1.0e-06_ep_1.0e-02_gd_stress_test_line_states.npz",  # noqa
        "voltage_violation_results": "results/scacopf/case14/L_1.0e+02_1000_samples_10_quench_10_chains_step_dp_1.0e-06_ep_1.0e-02_gd_stress_test_voltage_violations.npz",  # noqa
        "case": "14-bus",
    },
    {
        "Algorithm": "GD",
        "Worst case (predicted)": 0.0,
        "Worst case (observed)": 2.6572380,
        "Likelihood of underestimate": 0.17741300,
        "generation_cost": 0.549297,
        "predicted_failures": "results/scacopf/case57/L_1.0e+02_1000_samples_0_quench_10_chains_step_dp_1.0e-06_ep_1.0e-03_gd_predicted_failures.npz",  # noqa"
        "stress_test_results": "results/scacopf/case57/L_1.0e+02_1000_samples_0_quench_10_chains_step_dp_1.0e-06_ep_1.0e-03_gd_stress_test.npz",  # noqa
        "case": "57-bus",
    },
    {
        "Algorithm": "Ours",
        "Worst case (predicted)": 0.35271,
        "Worst case (observed)": 0.2735613,
        "Likelihood of underestimate": 0.00,
        "generation_cost": 1.4132679,
        "predicted_failures": "results/scacopf/case14/L_1.0e+02_1000_samples_10_quench_10_chains_step_dp_1.0e-06_ep_1.0e-02_mala_predicted_failures.npz",  # noqa"
        "stress_test_results": "results/scacopf/case14/L_1.0e+02_1000_samples_10_quench_10_chains_step_dp_1.0e-06_ep_1.0e-02_mala_stress_test.npz",  # noqa
        "line_states_results": "results/scacopf/case14/L_1.0e+02_1000_samples_10_quench_10_chains_step_dp_1.0e-06_ep_1.0e-02_mala_stress_test_line_states.npz",  # noqa
        "voltage_violation_results": "results/scacopf/case14/L_1.0e+02_1000_samples_10_quench_10_chains_step_dp_1.0e-06_ep_1.0e-02_mala_stress_test_voltage_violations.npz",  # noqa
        "case": "14-bus",
    },
    {
        "Algorithm": "Ours",
        "Worst case (predicted)": 0.35271,
        "Worst case (observed)": 0.2735613,
        "Likelihood of underestimate": 0.00,
        "generation_cost": 1.4132679,
        "predicted_failures": "results/scacopf/case57/L_1.0e+02_1000_samples_10_quench_10_chains_step_dp_1.0e-06_ep_1.0e-03_mala_predicted_failures.npz",  # noqa"
        "stress_test_results": "results/scacopf/case57/L_1.0e+02_1000_samples_10_quench_10_chains_step_dp_1.0e-06_ep_1.0e-03_mala_stress_test.npz",  # noqa
        "case": "57-bus",
    },
]


if __name__ == "__main__":
    results = pd.DataFrame(data)

    sns.set_theme(style="whitegrid", context="paper", font_scale=2.0)
    fig = plt.figure(figsize=(9, 6), constrained_layout=True)

    # Load distributions from files
    potentials = pd.Series([], dtype=float)
    algs = pd.Series([], dtype=str)
    types = pd.Series([], dtype=str)
    test_case = pd.Series([], dtype=str)
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
            [types, pd.Series(["Predicted contingencies" for _ in range(N)])],
            ignore_index=True,
        )
        test_case = pd.concat(
            [test_case, pd.Series([entry["case"] for _ in range(N)])], ignore_index=True
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
            [types, pd.Series(["Random contingencies" for _ in range(N)])],
            ignore_index=True,
        )
        test_case = pd.concat(
            [test_case, pd.Series([entry["case"] for _ in range(N)])], ignore_index=True
        )

    plotting_df = pd.DataFrame()
    plotting_df["Algorithm"] = algs
    plotting_df["Type"] = types
    plotting_df["Cost"] = potentials
    plotting_df["Network"] = test_case

    # Plot the distributions of potentials
    method = "GD"
    method_df = plotting_df[plotting_df["Algorithm"] == method]
    observed_df = method_df[method_df["Type"] == "Random contingencies"]
    predicted_df = method_df[method_df["Type"] == "Predicted contingencies"]

    min_predicted = predicted_df.groupby("Network", sort=False)["Cost"].min()
    max_predicted = predicted_df.groupby("Network", sort=False)["Cost"].max()
    mean_predicted = predicted_df.groupby("Network", sort=False)["Cost"].mean()
    plt.scatter(
        min_predicted.index,
        min_predicted,
        marker="^",
        s=75,
        linewidth=2,
        color="orangered",
        zorder=10,
    )
    plt.scatter(
        mean_predicted.index,
        mean_predicted,
        marker="_",
        s=100,
        linewidth=2,
        color="orangered",
        zorder=10,
    )
    plt.scatter(
        max_predicted.index,
        max_predicted,
        marker="v",
        s=75,
        linewidth=2,
        color="orangered",
        zorder=10,
    )
    plt.scatter(
        [],
        [],
        marker=r"$\frac{\blacktriangledown}{\blacktriangle}$",
        label="Predicted min/mean/max",
        color="orangered",
        zorder=10,
        s=500,
    )

    sns.boxenplot(
        y="Cost",
        x="Network",
        hue="Type",
        showfliers=False,
        outlier_prop=1e-7,
        data=observed_df,
    )

    plt.ylabel("Cost")
    plt.yscale("log")
    plt.legend(loc="upper left")

    plt.show()
