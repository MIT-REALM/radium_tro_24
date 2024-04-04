"""A script for analyzing the results of the prediction experiments."""
import json
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import pandas as pd
from jaxtyping import Array, Shaped

from radium.experiments.intersection.predict_and_mitigate import (
    non_ego_actions_prior_logprob,
    simulate,
)
from radium.experiments.intersection.train_intersection_agent_bc import (
    make_intersection_env,
)
from radium.systems.highway.highway_env import HighwayState
from radium.systems.intersection.policy import DrivingPolicy

# should we re-run the analysis (True) or just load the previously-saved summary (False)
REANALYZE = False
# path to save summary data to
lr = 1e-3
lr = f"{lr:.1e}"
SUMMARY_PATH = f"results/intersection/predict/convergence_summary_{lr}.json"
# Define data sources from individual experiments
SEEDS = [0, 1, 2, 3]
DATA_SOURCES = {
    "mala_tempered": {
        "path_prefix": f"results/intersection/predict/noise_5.0e-01/L_1.0e+01/25_samples/5_chains/0_quench/dp_{lr}/ep_{lr}/mala_20tempered+0.1",
        "display_name": "RADIUM (ours)",
    },
    "rmh": {
        "path_prefix": f"results/intersection/predict/noise_5.0e-01/L_1.0e+01/25_samples/5_chains/0_quench/dp_{lr}/ep_{lr}/rmh",
        "display_name": "ROCUS",
    },
    "gd": {
        "path_prefix": f"results/intersection/predict/noise_5.0e-01/L_1.0e+00/25_samples/5_chains/0_quench/dp_{lr}/ep_{lr}/gd",
        "display_name": "ML",
    },
    "reinforce": {
        "path_prefix": f"results/intersection/predict/noise_5.0e-01/L_1.0e+00/25_samples/5_chains/0_quench/dp_{lr}/ep_{lr}/reinforce_l2c",
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
                    "failure_level": 3.5,  # data["failure_level"], # measured from plots
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
                DATA_SOURCES[alg]["path_prefix"] + f"_{seed}" + ".eqx",
                dummy_policy,
            )
            dp, static_policy = eqx.partition(full_policy, eqx.is_array)
            new_data["dp"] = dp
            new_data["static_policy"] = static_policy

            loaded_data[alg].append(new_data)

    return loaded_data


def get_costs(loaded_data):
    """Get the cost at each step of each algorithm in the loaded data."""
    # Pre-compile the same cost function for all
    alg = "mala_tempered"
    image_shape = (32, 32)
    env = make_intersection_env(image_shape)
    env._collision_penalty = 10.0
    initial_state = HighwayState(
        ego_state=loaded_data[alg][0]["initial_state"]["ego_state"],
        non_ego_states=loaded_data[alg][0]["initial_state"]["non_ego_states"],
        shading_light_direction=loaded_data[alg][0]["initial_state"][
            "shading_light_direction"
        ],
        non_ego_colors=loaded_data[alg][0]["initial_state"]["non_ego_colors"],
    )

    cost_fn = lambda ep: simulate(
        env,
        loaded_data[alg][0]["dp"],
        initial_state,
        ep,
        loaded_data[alg][0]["static_policy"],
        loaded_data[alg][0]["T"],
    ).potential
    cost_fn = jax.jit(cost_fn)
    ep_logprior_fn = jax.jit(
        lambda ep: non_ego_actions_prior_logprob(
            ep, env, loaded_data[alg][0]["noise_scale"]
        )
    )

    for alg in loaded_data:
        for i, result in enumerate(loaded_data[alg]):
            print(f"Computing costs for {alg}, seed {i}...")
            eps = result["eps_trace"]

            result["ep_costs"] = jax.vmap(jax.vmap(cost_fn))(eps)
            result["ep_logpriors"] = jax.vmap(jax.vmap(ep_logprior_fn))(eps)
            result["ep_logprobs"] = result["ep_logpriors"] - jax.nn.elu(
                result["failure_level"] - result["ep_costs"]
            )

    return loaded_data


if __name__ == "__main__":
    if REANALYZE or not os.path.exists(SUMMARY_PATH):
        # Load the data
        data = load_data_sources_from_json()

        # Compute costs
        data = get_costs(data)

        # Extract the summary data we want to save
        summary_data = {}
        for alg in data:
            summary_data[alg] = []
            for result in data[alg]:
                summary_data[alg].append(
                    {
                        "display_name": result["display_name"],
                        "ep_costs": result["ep_costs"],
                        "ep_logpriors": result["ep_logpriors"],
                        "ep_logprobs": result["ep_logprobs"],
                        "failure_level": result["failure_level"],
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

            for alg in summary_data:
                for result in summary_data[alg]:
                    result["ep_costs"] = jnp.array(result["ep_costs"])
                    result["ep_logpriors"] = jnp.array(result["ep_logpriors"])
                    result["ep_logprobs"] = jnp.array(result["ep_logprobs"])
                    result["failure_level"] = 3.5  # measured from plots

    # Post-process
    for alg in DATA_SOURCES:
        for result in summary_data[alg]:
            failure_level = result["failure_level"]
            costs = result["ep_costs"]
            num_failures = (costs >= failure_level).sum(axis=-1)
            # # Cumulative max = 'how many failures have we seen so far?'
            # num_failures = jax.lax.cummax(num_failures)
            # Add a 0 at the start (randomly sampling 10 failures gives 0 failures at step 0)
            num_failures = jnp.concatenate([jnp.zeros(1), num_failures])
            result["num_failures"] = num_failures

    # Make into pandas dataframe
    iters = pd.Series([], dtype=int)
    logprobs = pd.Series([], dtype=float)
    costs = pd.Series([], dtype=float)
    num_failures = pd.Series([], dtype=float)
    algs = pd.Series([], dtype=str)
    seeds = pd.Series([], dtype=int)
    for alg in DATA_SOURCES:
        for seed_i, result in enumerate(summary_data[alg]):
            num_iters = result["ep_logprobs"].shape[0]
            num_chains = result["ep_logprobs"].shape[1]

            # # Add the number of failures discovered initially
            # iters = pd.concat(
            #     [iters, pd.Series(jnp.zeros(num_chains, dtype=int))], ignore_index=True
            # )
            # seeds = pd.concat(
            #     [seeds, pd.Series(jnp.zeros(num_chains, dtype=int) + seed_i)],
            #     ignore_index=True,
            # )
            # num_failures = pd.concat(
            #     [
            #         num_failures,
            #         pd.Series([float(result["num_failures"][0])] * num_chains),
            #     ],
            #     ignore_index=True,
            # )
            # logprobs = pd.concat(
            #     [logprobs, pd.Series(jnp.zeros(num_chains))], ignore_index=True
            # )
            # costs = pd.concat(
            #     [costs, pd.Series(jnp.zeros(num_chains))], ignore_index=True
            # )
            # algs = pd.concat(
            #     [algs, pd.Series([result["display_name"]] * num_chains)],
            #     ignore_index=True,
            # )

            # Add the data for the rest of the iterations
            for i in range(num_iters):
                iters = pd.concat(
                    [iters, pd.Series(jnp.zeros(num_chains, dtype=int) + i + 1)],
                    ignore_index=True,
                )
                seeds = pd.concat(
                    [seeds, pd.Series(jnp.zeros(num_chains, dtype=int) + seed_i)],
                    ignore_index=True,
                )
                logprobs = pd.concat(
                    [logprobs, pd.Series(result["ep_logprobs"][i, :])],
                    ignore_index=True,
                )
                costs = pd.concat(
                    [
                        costs,
                        pd.Series(
                            -jax.nn.elu(
                                result["failure_level"] - result["ep_costs"][i, :]
                            )
                        ),
                    ],
                    ignore_index=True,
                )
                num_failures = pd.concat(
                    [
                        num_failures,
                        pd.Series([float(result["num_failures"][i + 1])] * num_chains),
                    ],
                    ignore_index=True,
                )
                algs = pd.concat(
                    [algs, pd.Series([result["display_name"]] * num_chains)],
                    ignore_index=True,
                )

    df = pd.DataFrame()
    df["Diffusion steps"] = iters
    df["Overall log likelihood"] = logprobs
    df["$[J^* - J]_+$"] = costs
    df["# failures discovered"] = num_failures
    df["Algorithm"] = algs
    df["Seed"] = seeds

    print("Collision rate (mean)")
    print(
        df[df["Diffusion steps"] >= 3]
        .groupby(["Algorithm"])["# failures discovered"]
        .mean()
        / 5
    )
    # print("Collision rate (std)")
    # print(
    #     df[df["Diffusion steps"] >= 3]
    #     .groupby(["Algorithm"])["# failures discovered"]
    #     .std()
    #     / 5
    # )
    print("Collision rate (75th)")
    print(
        df[df["Diffusion steps"] >= 3]
        .groupby(["Algorithm"])["# failures discovered"]
        .quantile(0.75)
        / 5
    )
    print("Collision rate (25)")
    print(
        df[df["Diffusion steps"] >= 3]
        .groupby(["Algorithm"])["# failures discovered"]
        .quantile(0.25)
        / 5
    )
