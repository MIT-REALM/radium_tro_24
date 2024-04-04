import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data = [
    {
        "Algorithm": "Adversarial optimization",
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
    # {
    #     "Algorithm": "Adversarial optimization",
    #     "Worst case (predicted)": 0.0,
    #     "Worst case (observed)": 2.6572380,
    #     "Likelihood of underestimate": 0.17741300,
    #     "generation_cost": 0.549297,
    #     "predicted_failures": "results/scacopf/case57/L_1.0e+02_1000_samples_0_quench_10_chains_step_dp_1.0e-06_ep_1.0e-03_gd_predicted_failures.npz",  # noqa"
    #     "stress_test_results": "results/scacopf/case57/L_1.0e+02_1000_samples_0_quench_10_chains_step_dp_1.0e-06_ep_1.0e-03_gd_stress_test.npz",  # noqa
    #     "case": "57-bus",
    # },
    {
        "Algorithm": "Adversarial sampling (ours)",
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
    # {
    #     "Algorithm": "Adversarial sampling (ours)",
    #     "Worst case (predicted)": 0.35271,
    #     "Worst case (observed)": 0.2735613,
    #     "Likelihood of underestimate": 0.00,
    #     "generation_cost": 1.4132679,
    #     "predicted_failures": "results/scacopf/case57/L_1.0e+02_1000_samples_10_quench_10_chains_step_dp_1.0e-06_ep_1.0e-03_mala_predicted_failures.npz",  # noqa"
    #     "stress_test_results": "results/scacopf/case57/L_1.0e+02_1000_samples_10_quench_10_chains_step_dp_1.0e-06_ep_1.0e-03_mala_stress_test.npz",  # noqa
    #     "case": "57-bus",
    # },
]


if __name__ == "__main__":
    results = pd.DataFrame(data)

    sns.set_theme(style="whitegrid", context="paper", font_scale=2.0)
    fig = plt.figure(figsize=(9, 6), constrained_layout=True)

    # Load distributions from files
    potentials = pd.Series([], dtype=float)
    voltage_violations = pd.Series([], dtype=float)
    outages = pd.Series([], dtype=int)
    algs = pd.Series([], dtype=str)
    test_case = pd.Series([], dtype=str)
    for entry in data:
        # Load stress test results
        new_potentials = jnp.load(entry["stress_test_results"])
        N = new_potentials.shape[0]
        potentials = pd.concat(
            [potentials, pd.Series(new_potentials)],
            ignore_index=True,
        )
        voltage_violations = pd.concat(
            [
                voltage_violations,
                pd.Series(jnp.load(entry["voltage_violation_results"])),
            ],
            ignore_index=True,
        )
        line_states = jnp.load(entry["line_states_results"])
        line_states = line_states.reshape((-1, line_states.shape[-1]))
        line_strength = jax.nn.sigmoid(2 * line_states)
        line_failures = line_strength < 0.9
        outages = pd.concat(
            [outages, pd.Series(line_failures.sum(axis=-1), dtype=int)],
            ignore_index=True,
        )
        algs = pd.concat(
            [algs, pd.Series([entry["Algorithm"] for _ in range(N)])], ignore_index=True
        )
        test_case = pd.concat(
            [test_case, pd.Series([entry["case"] for _ in range(N)])], ignore_index=True
        )

    plotting_df = pd.DataFrame()
    plotting_df["Algorithm"] = algs
    plotting_df["Cost"] = potentials
    plotting_df["Network"] = test_case
    plotting_df["# of affected lines"] = outages
    plotting_df["Voltage collapse"] = voltage_violations

    # Condition on failure cases
    gd_df = plotting_df[plotting_df["Algorithm"] == "Adversarial optimization"]
    gd_df = gd_df[gd_df["Cost"] > 10.0]
    ours_df = plotting_df[plotting_df["Algorithm"] == "Adversarial sampling (ours)"]
    ours_df = ours_df[ours_df["Cost"] > 10.0]
    failure_df = pd.concat([gd_df, ours_df], ignore_index=True)

    sns.histplot(
        data=failure_df,
        x="# of affected lines",
        hue="Algorithm",
        # common_norm=True,
        stat="count",
        multiple="dodge",
        discrete=True,
        shrink=0.8,
    )

    plt.ylabel("Number of failures")
    plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9])

    plt.show()
