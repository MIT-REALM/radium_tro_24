"""A script for analyzing the results of the repair experiments."""

import json
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import pandas as pd
from jaxtyping import Array, Shaped
from tqdm import tqdm

from radium.experiments.highway.predict_and_mitigate import (
    LinearTrajectory2D,
    MultiAgentTrajectoryLinear,
    sample_non_ego_trajectory,
    simulate,
)
from radium.experiments.highway.train_highway_agent import make_highway_env
from radium.systems.highway.driving_policy import DrivingPolicy
from radium.systems.highway.highway_env import HighwayState

# How many monte carlo trials to use to compute true failure rate
N = 100
BATCHES = 10
# should we re-run the analysis (True) or just load the previously-saved summary (False)
REANALYZE = False
lr = 1e-2
lr = f"{lr:.1e}"
# path to save summary data to in predict_repair folder
SUMMARY_PATH = (
    f"results/neurips_submission/highway/predict_repair_1.0/stress_test_{lr}.json"
)
# Define data sources from individual experiments
SEEDS = [0, 1, 2, 3]
DATA_SOURCES = {
    "mala_tempered": {
        "path_prefix": f"results/neurips_submission/highway/predict_repair_1.0/noise_5.0e-01/L_1.0e+01/25_samples/5_chains/0_quench/dp_{lr}/ep_{lr}/mala_20tempered+0.1",
        "display_name": "Ours (tempered)",
    },
    "rmh": {
        "path_prefix": f"results/neurips_submission/highway/predict_repair_1.0/noise_5.0e-01/L_1.0e+01/25_samples/5_chains/0_quench/dp_{lr}/ep_{lr}/rmh",
        "display_name": "ROCUS",
    },
    "gd": {
        "path_prefix": f"results/neurips_submission/highway/predict_repair_1.0/noise_5.0e-01/L_1.0e+00/25_samples/5_chains/0_quench/dp_{lr}/ep_{lr}/gd",
        "display_name": "ML",
    },
    "reinforce": {
        "path_prefix": f"results/neurips_submission/highway/predict_repair_1.0/noise_5.0e-01/L_1.0e+00/25_samples/5_chains/0_quench/dp_{lr}/ep_{lr}/reinforce_l2c",
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
                    "final_eps": jnp.array(data["action_trajectory"]),
                    "eps_trace": jax.tree_util.tree_map(
                        lambda x: jnp.array(x),
                        data["action_trajectory_trace"],
                        is_leaf=lambda x: isinstance(x, list),
                    ),
                    "failure_level": 3.9,  # data["failure_level"],
                    "noise_scale": data["noise_scale"],
                    "initial_state": jax.tree_util.tree_map(
                        lambda x: jnp.array(x),
                        data["initial_state"],
                        is_leaf=lambda x: isinstance(x, list),
                    ),
                }
                new_data["T"] = new_data["eps_trace"].shape[2]

            # Also load in the design parameters
            image_shape = (32, 32)
            dummy_policy = DrivingPolicy(jax.random.PRNGKey(0), image_shape)
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
    image_shape = (32, 32)
    env = make_highway_env(image_shape)
    env._collision_penalty = 10.0
    initial_state = HighwayState(
        ego_state=loaded_data[alg][seed]["initial_state"]["ego_state"],
        non_ego_states=loaded_data[alg][seed]["initial_state"]["non_ego_states"],
        shading_light_direction=loaded_data[alg][seed]["initial_state"][
            "shading_light_direction"
        ],
        non_ego_colors=loaded_data[alg][seed]["initial_state"]["non_ego_colors"],
    )

    cost_fn = lambda ep: simulate(
        env,
        loaded_data[alg][seed]["dp"],
        initial_state,
        ep,
        loaded_data[alg][seed]["static_policy"],
        loaded_data[alg][seed]["T"],
    ).potential
    cost_fn = jax.jit(cost_fn)

    # The nominal non-ego behavior is to drive straight
    drive_straight = LinearTrajectory2D(
        p=jnp.array(
            [
                [10.0, 0.0],
                [20.0, 0.0],
                [30.0, 0.0],
                [40.0, 0.0],
                [50.0, 0.0],
            ]
        )
    )
    nominal_trajectory = MultiAgentTrajectoryLinear(
        trajectories=[drive_straight, drive_straight]
    )

    sample_fn = lambda key: sample_non_ego_trajectory(
        key, nominal_trajectory, noise_scale
    )

    # Sample N environmental parameters at random
    key = jax.random.PRNGKey(0)
    costs = []
    print("Running stress test...")
    for _ in tqdm(range(batches)):
        key, subkey = jax.random.split(key)
        initial_state_keys = jax.random.split(subkey, N)
        eps = jax.vmap(sample_fn)(initial_state_keys)
        costs.append(jax.vmap(cost_fn)(eps))

    return jnp.concatenate(costs)


if __name__ == "__main__":
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
                        "failure_level": 3.9,  # data[alg][seed]["failure_level"],
                    }
                )

        # Save the data
        with open(SUMMARY_PATH, "w") as f:
            json.dump(
                summary_data,
                f,
                default=lambda x: (
                    x.tolist() if isinstance(x, Shaped[Array, "..."]) else x
                ),
            )
    else:
        # Load the data
        with open(SUMMARY_PATH, "rb") as f:
            summary_data = json.load(f)

            for alg in DATA_SOURCES:
                for result in summary_data[alg]:
                    result["costs"] = jnp.array(result["costs"])
                    result["failure_level"] = 3.9

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
    failure_level = summary_data["mala_tempered"][0]["failure_level"]
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
