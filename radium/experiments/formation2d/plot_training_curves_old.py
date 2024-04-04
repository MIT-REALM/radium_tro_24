import jax.numpy as jnp
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data = [
    {
        "Algorithm": "RMH (gradient-free)",
        "training_curve_file": "results/formation2d_grad_norm_netconn/10/L_1.0e+00/3_T/250_samples/0_quench/no_temper5_chains/dp_1.0e-03/ep_1.0e-03/repair/predict/rmh_training_progress.npz",  # noqa
    },
    {
        "Algorithm": "MALA (gradient-based)",
        "training_curve_file": "results/formation2d_grad_norm_netconn/10/L_1.0e+00/3_T/250_samples/5_quench/no_temper5_chains/dp_1.0e-03/ep_1.0e-03/repair/predict/mala_training_progress.npz",  # noqa
    },
]

if __name__ == "__main__":
    results = pd.DataFrame(data)

    sns.set_theme(style="whitegrid", context="paper", font_scale=2.0)
    fig = plt.figure(figsize=(9, 4), constrained_layout=True)

    # Load distributions from files
    for entry in data:
        # Load potential on test set
        entry["test_potentials"] = jnp.load(entry["training_curve_file"])

        T = 50
        x = jnp.arange(T)
        h = plt.plot(
            x,
            entry["test_potentials"].mean(axis=-1)[:T],
            label=entry["Algorithm"],
            linewidth=3,
        )
        # plt.fill_between(
        #     x,
        #     entry["test_potentials"].min(axis=-1)[:T],
        #     entry["test_potentials"].max(axis=-1)[:T],
        #     color=h[0].get_color(),
        #     alpha=0.2,
        # )

    # plt.yscale("log")
    plt.ylabel("Avg. Cost (test set)")
    plt.xlabel("# Sampling Rounds")
    plt.legend()
    # plt.show()
    plt.savefig("test.png")
