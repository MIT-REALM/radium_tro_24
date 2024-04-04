"""A script for analyzing the results of the prediction experiments."""
import json
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from jaxtyping import Array, Shaped
from tqdm import tqdm

from radium.experiments.simple_grasping.predict_and_mitigate import (
    GraspExogenousParams,
    ep_logprior,
    sample_ep,
    simulate,
)
from radium.systems.simple_grasping.policy import AffordancePredictor

# How many monte carlo trials to use to compute true failure rate
N = 100
BATCHES = 10
# should we re-run the analysis (True) or just load the previously-saved summary (False)
REANALYZE = False
# object_type = "mug"
# failure_level = 0.25
# lr = 1e-3

object_type = "box"
failure_level = 0.1
lr = 1e-3

lr = f"{lr:.1e}"  # format as string
# path to save summary data to
SUMMARY_PATH = (
    f"results/grasping_{object_type}/predict_repair_1.0/stress_test_{lr}.json"
)
# Define data sources from individual experiments
SEEDS = [0, 1, 2, 3]
DATA_SOURCES = {
    "mala_tempered": {
        "path_prefix": f"results/grasping_{object_type}/predict_repair_1.0/L_1.0e+01/25_samples_5x5/5_chains/0_quench/dp_{lr}/ep_{lr}/grad_norm/grad_clip_inf/mala_tempered_40+0.1",
        "display_name": "RADIUM (ours)",
    },
    "rmh": {
        "path_prefix": f"results/grasping_{object_type}/predict_repair_1.0/L_1.0e+01/25_samples_5x5/5_chains/0_quench/dp_{lr}/ep_{lr}/grad_norm/grad_clip_inf/rmh",
        "display_name": "ROCUS",
    },
    "gd": {
        "path_prefix": f"results/grasping_{object_type}/predict_repair_1.0/L_1.0e+00/25_samples_5x5/5_chains/0_quench/dp_{lr}/ep_{lr}/grad_norm/grad_clip_inf/gd",
        "display_name": "ML",
    },
    "reinforce": {
        "path_prefix": f"results/grasping_{object_type}/predict_repair_1.0/L_1.0e+00/25_samples_5x5/5_chains/0_quench/dp_{lr}/ep_{lr}/grad_norm/grad_clip_inf/reinforce_l2c_0.05_step",
        "display_name": "L2C",
    },
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
                    "object_type": data["object_type"],
                    "failure_level": data["failure_level"],
                }

            # Also load in the design parameters
            dummy_policy = AffordancePredictor(jax.random.PRNGKey(0))
            full_policy = eqx.tree_deserialise_leaves(
                DATA_SOURCES[alg]["path_prefix"] + f"_{seed}" + ".eqx", dummy_policy
            )
            dp, static_policy = eqx.partition(full_policy, eqx.is_array)
            new_data["dp"] = dp
            new_data["static_policy"] = static_policy

            loaded_data[alg].append(new_data)

    return loaded_data


def monte_carlo_test(N, batches, loaded_data, alg, seed):
    """Stress test the given policy using N samples in batches"""
    cost_fn = lambda ep: simulate(
        loaded_data[alg][seed]["object_type"],
        ep,
        loaded_data[alg][seed]["dp"],
        loaded_data[alg][seed]["static_policy"],
    ).potential
    cost_fn = jax.jit(cost_fn)

    sample_fn = sample_ep

    # Sample N environmental parameters at random
    key = jax.random.PRNGKey(0)
    costs = []
    print("Running stress test...")
    for _ in tqdm(range(batches)):
        key, subkey = jax.random.split(key)
        ep_keys = jax.random.split(subkey, N)
        eps = jax.vmap(sample_fn)(ep_keys)
        costs.append(jax.vmap(cost_fn)(eps))

    return jnp.concatenate(costs)


if __name__ == "__main__":
    # Activate seaborn styling
    sns.set_theme(context="paper", style="whitegrid")

    if REANALYZE or not os.path.exists(SUMMARY_PATH):
        # Load the data
        data = load_data_sources_from_json()

        # Compute costs for each seed for each algorithm
        for alg in data:
            for seed, result in enumerate(data[alg]):
                print(f"Computing costs for {alg} seed {seed}")
                result["costs"] = monte_carlo_test(N, BATCHES, data, alg, seed)

        # Extract the summary data we want to save
        summary_data = {}
        for alg in data:
            summary_data[alg] = []
            for seed in SEEDS:
                summary_data[alg].append(
                    {
                        "display_name": data[alg][seed]["display_name"],
                        "costs": data[alg][seed]["costs"],
                        "failure_level": data[alg][seed]["failure_level"],
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

            for alg in DATA_SOURCES:
                for result in summary_data[alg]:
                    result["costs"] = jnp.array(result["costs"])

    # Post-process into a dataframe
    algs = []
    costs = []
    seeds = []
    for alg in DATA_SOURCES:
        for seed, result in enumerate(summary_data[alg]):
            algs += [result["display_name"]] * len(result["costs"])
            costs += result["costs"].flatten().tolist()
            seeds += [seed] * len(result["costs"])

    df = pd.DataFrame({"Algorithm": algs, "Cost": costs, "Seed": seeds})

    # Assign maximum cost to any nans
    df["Cost"] = df["Cost"].fillna(df["Cost"].max())

    # Count failures
    df["Failure"] = df["Cost"] >= failure_level

    # Print failure rates
    print("Mean cost")
    print(df.groupby(["Algorithm"])["Cost"].mean())
    print("cost std err")
    print(
        df.groupby(["Algorithm", "Seed"])["Cost"].mean().groupby(["Algorithm"]).std()
        / 2
    )
    print(f"Failure rate {failure_level}")
    print(df.groupby(["Algorithm"])["Failure"].mean())
    print(f"Failure rate {failure_level} stderr")
    print(
        df.groupby(["Algorithm", "Seed"])["Failure"].mean().groupby(["Algorithm"]).std()
        / 2
    )

    # Plot!
    plt.figure(figsize=(12, 8))
    sns.violinplot(
        x="Algorithm",
        y="Cost",
        data=df,
    )
    plt.gca().set_xlabel("")
    # plt.savefig('results/highway/predict_repair_1.0/seed_0.png') #saving images to file for each seed
    plt.show()
