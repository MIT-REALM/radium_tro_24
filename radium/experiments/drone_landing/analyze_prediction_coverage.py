"""A script for analyzing the results of the prediction experiments."""
import json
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyemd
import scipy
import seaborn as sns
from jaxtyping import Array, Shaped
from tqdm import tqdm

from radium.experiments.drone_landing.predict_and_mitigate import simulate
from radium.experiments.drone_landing.train_drone_agent import make_drone_landing_env
from radium.systems.drone_landing.env import DroneState
from radium.systems.drone_landing.policy import DroneLandingPolicy

# How many monte carlo trials to use to compute true failure rate
N = 1000
BATCHES = 200
# should we re-run the analysis (True) or just load the previously-saved summary (False)
REANALYZE = False
# path to save summary data to
SUMMARY_PATH = (
    "results/drone_landing_smooth/predict/coverage_summary_gradnorm_mcmc_1e-2.json"
)
# path to load convergence data from
CONVERGENCE_SUMMARY_PATH = (
    "results/drone_landing_smooth/predict/convergence_summary_gradnorm_mcmc_1e-2.json"
)
# Define data sources from individual experiments
SEEDS = [0, 1, 2, 3]
DATA_SOURCES = {
    "mala_tempered": {
        "path_prefix": "results/drone_landing_smooth/predict/L_1.0e+00/30_samples_30x1/10_chains/0_quench/dp_1.0e-02/ep_1.0e-02/grad_norm/grad_clip_inf/mala_tempered_40",
        "display_name": "Ours (tempered)",
    },
    "rmh": {
        "path_prefix": "results/drone_landing_smooth/predict/L_1.0e+00/30_samples_30x1/10_chains/0_quench/dp_1.0e-02/ep_1.0e-02/grad_norm/grad_clip_inf/rmh",
        "display_name": "ROCUS",
    },
    "gd": {
        "path_prefix": "results/drone_landing_smooth/predict/L_1.0e+00/30_samples_30x1/10_chains/0_quench/dp_3.0e-03/ep_3.0e-03/grad_norm/grad_clip_inf/gd",
        "display_name": "ML",
    },
    "reinforce": {
        "path_prefix": "results/drone_landing_smooth/predict/L_1.0e+00/30_samples_30x1/10_chains/0_quench/dp_1.0e-03/ep_1.0e-03/grad_norm/grad_clip_inf/reinforce_l2c_0.05_step",
        "display_name": "L2C",
    },
    # "random": {
    #     "path_prefix": "results/drone_landing_smooth/predict/L_1.0e+00/1_samples_1x1/10_chains/0_quench/dp_1.0e-05/ep_1.0e-05/grad_norm/grad_clip_inf/static_tempered_40",
    #     "display_name": "Random sampling",
    # },
}


def load_data_sources_from_json():
    """Load data sources from a JSON file."""
    loaded_data = {}

    # Load the results from each data source
    for alg in DATA_SOURCES:
        loaded_data[alg] = []
        for seed in SEEDS:
            with open(DATA_SOURCES[alg]["path_prefix"] + f"_{seed}" + ".json") as f:
                data = json.load(f)

                new_data = {
                    "display_name": DATA_SOURCES[alg]["display_name"],
                    "eps_trace": jax.tree_util.tree_map(
                        lambda x: jnp.array(x),
                        data["eps_trace"],
                        is_leaf=lambda x: isinstance(x, list),
                    ),
                    "T": data["T"],
                    "num_trees": data["num_trees"],
                    "failure_level": data["failure_level"],
                }

            # Also load in the design parameters
            image_shape = (32, 32)
            dummy_policy = DroneLandingPolicy(jax.random.PRNGKey(0), image_shape)
            full_policy = eqx.tree_deserialise_leaves(
                DATA_SOURCES[alg]["path_prefix"] + f"_{seed}" + ".eqx", dummy_policy
            )
            dp, static_policy = eqx.partition(full_policy, eqx.is_array)
            new_data["dp"] = dp
            new_data["static_policy"] = static_policy

            loaded_data[alg].append(new_data)

    return loaded_data


def monte_carlo_test(N, batches, loaded_data):
    """Stress test the given policy using N samples in batches"""
    alg = "mala_tempered"
    image_shape = (32, 32)
    env = make_drone_landing_env(image_shape, loaded_data[alg][0]["num_trees"])
    cost_fn = lambda ep: simulate(
        env,
        loaded_data[alg][0]["dp"],
        ep,
        loaded_data[alg][0]["static_policy"],
        loaded_data[alg][0]["T"],
    ).potential
    cost_fn = jax.jit(cost_fn)

    # Sample N environmental parameters at random
    key = jax.random.PRNGKey(0)
    costs = []
    print("Running stress test...")
    for _ in tqdm(range(batches)):
        key, subkey = jax.random.split(key)
        initial_state_keys = jax.random.split(subkey, N)
        eps = jax.vmap(env.reset)(initial_state_keys)
        costs.append(jax.vmap(cost_fn)(eps))

    return jnp.concatenate(costs)


if __name__ == "__main__":
    # Activate seaborn styling
    sns.set_theme(context="paper", style="whitegrid")

    # Load convergence data (has costs for each algorithm)
    with open(CONVERGENCE_SUMMARY_PATH, "rb") as f:
        convergence_summary_data = json.load(f)

        for alg in convergence_summary_data:
            for result in convergence_summary_data[alg]:
                result["ep_costs"] = jnp.array(result["ep_costs"])
                result["ep_logpriors"] = jnp.array(result["ep_logpriors"])
                result["ep_logprobs"] = jnp.array(result["ep_logprobs"])

    if REANALYZE or not os.path.exists(SUMMARY_PATH):
        # Load the data
        data = load_data_sources_from_json()

        # Compute costs
        monte_carlo_costs = monte_carlo_test(N, BATCHES, data)

        # Extract the summary data we want to save
        summary_data = {
            "random_sample_costs": monte_carlo_costs,
            "N": N,
            "batches": BATCHES,
        }
        for alg in convergence_summary_data:
            summary_data[alg] = []
            for seed in SEEDS:
                summary_data[alg].append(
                    {
                        "display_name": convergence_summary_data[alg][seed][
                            "display_name"
                        ],
                        "ep_costs": convergence_summary_data[alg][seed]["ep_costs"],
                        "ep_logpriors": convergence_summary_data[alg][seed][
                            "ep_logpriors"
                        ],
                        "ep_logprobs": convergence_summary_data[alg][seed][
                            "ep_logprobs"
                        ],
                        "failure_level": convergence_summary_data[alg][seed][
                            "failure_level"
                        ],
                    }
                )

        # Save the data
        with open(SUMMARY_PATH, "w") as f:
            json.dump(
                summary_data,
                f,
                default=lambda x: x.tolist()
                if isinstance(x, Shaped[Array, "..."])
                else x,
            )
    else:
        # Load the data
        with open(SUMMARY_PATH, "rb") as f:
            summary_data = json.load(f)

            summary_data["random_sample_costs"] = jnp.array(
                summary_data["random_sample_costs"]
            )
            for alg in DATA_SOURCES:
                for result in summary_data[alg]:
                    result["ep_costs"] = jnp.array(result["ep_costs"])
                    result["ep_logpriors"] = jnp.array(result["ep_logpriors"])
                    result["ep_logprobs"] = jnp.array(result["ep_logprobs"])

    # Post-process into a dataframe
    df = pd.DataFrame(
        {
            "Algorithm": "Monte Carlo from prior",
            "Cost": summary_data["random_sample_costs"],
        }
    )
    for alg in DATA_SOURCES:
        for result in summary_data[alg]:
            df = df.append(
                pd.DataFrame(
                    {
                        "Algorithm": result["display_name"],
                        "Cost": result["ep_costs"].flatten(),
                    }
                )
            )

    # Compute wassertein distance of each algorithm from the prior
    bins = 10
    gt, gt_bins = jnp.histogram(
        summary_data["random_sample_costs"],
        bins,
        density=True,
        weights=jnp.exp(-jax.nn.elu(25.0 - summary_data["random_sample_costs"])),
    )
    gt_bin_centers = (gt_bins[1:] + gt_bins[:-1]) / 2.0
    print(
        "Wasserstein distance from ground truth (importance sampling w. "
        + f"N={summary_data['N'] * summary_data['batches']}):"
    )
    for alg in DATA_SOURCES:
        costs = jnp.concatenate(
            [x["ep_costs"][5:].reshape(-1) for x in summary_data[alg]]
        )
        hist, hist_bins = jnp.histogram(costs, bins, density=True)
        hist_bin_centers = (hist_bins[1:] + hist_bins[:-1]) / 2.0

        distance = lambda x, y: jnp.linalg.norm(x - y)
        distance_to_gt = lambda x: jax.vmap(distance, in_axes=(0, None))(
            gt_bin_centers, x
        )
        distance_matrix = jax.vmap(distance_to_gt)(hist_bin_centers)

        wasserstein = pyemd.emd(
            np.array(gt, dtype=np.float64),
            np.array(hist, dtype=np.float64),
            np.array(distance_matrix, dtype=np.float64),
        )
        print(f"{DATA_SOURCES[alg]['display_name']}: {wasserstein}")

    # Also compute and print the JS divergence
    print(f"JS divergence from ground truth (importance sampling w. {N * BATCHES}):")
    for alg in DATA_SOURCES:
        costs = jnp.concatenate(
            [x["ep_costs"][0:].reshape(-1) for x in summary_data[alg]]
        )
        # re-compute the histogram to use the same bins as the ground truth
        hist, hist_bins = jnp.histogram(costs, bins, density=True)

        # Convert to numpy
        gt = np.array(gt, dtype=np.float64)
        hist = np.array(hist, dtype=np.float64)
        js = scipy.spatial.distance.jensenshannon(gt, hist)
        print(f"{DATA_SOURCES[alg]['display_name']} JS: {js}")

    # Also give the failure rate on prior data
    print(
        f"Failure rate from prior: {(summary_data['random_sample_costs'] >= 25.0).mean()}"
    )

    # Plot!
    plt.figure(figsize=(12, 8))
    sns.boxenplot(
        x="Algorithm",
        y="Cost",
        showfliers=False,
        outlier_prop=1e-7,
        # flier_kws={"s": 20},
        data=df,
    )
    plt.gca().set_xlabel("")

    plt.show()
