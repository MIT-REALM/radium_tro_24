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

from architect.experiments.drone_landing.predict_and_mitigate import simulate
from architect.experiments.drone_landing.train_drone_agent import make_drone_landing_env
from architect.systems.drone_landing.env import DroneState
from architect.systems.drone_landing.policy import DroneLandingPolicy

# should we re-run the analysis (True) or just load the previously-saved summary (False)
REANALYZE = True
# path to save summary data to
SUMMARY_PATH = "results/drone_landing_smooth/predict/convergence_summary.json"
# Define data sources from individual experiments
SEEDS = [0, 1, 2, 3]
DATA_SOURCES = {
    "mala_tempered": {
        "path_prefix": "results/drone_landing_smooth/predict/L_1.0e+00/30_samples_30x1/10_chains/0_quench/dp_1.0e-02/ep_1.0e-02/grad_norm/grad_clip_inf/mala_tempered_40",
        "display_name": "RADIUM (ours)",
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
                DATA_SOURCES[alg]["path_prefix"] + ".eqx", dummy_policy
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
    env = make_drone_landing_env(image_shape, loaded_data[alg][0]["num_trees"])
    cost_fn = lambda ep: simulate(
        env,
        loaded_data[alg][0]["dp"],
        ep,
        loaded_data[alg][0]["static_policy"],
        loaded_data[alg][0]["T"],
    ).potential
    cost_fn = jax.jit(cost_fn)
    ep_logprior_fn = jax.jit(env.initial_state_logprior)

    for alg in loaded_data:
        for i, result in enumerate(loaded_data[alg]):
            print(f"Computing costs for {alg} seed {i}...")
            eps = DroneState(
                drone_state=result["eps_trace"]["drone_state"],
                tree_locations=result["eps_trace"]["tree_locations"],
                wind_speed=result["eps_trace"]["wind_speed"],
            )
            result["ep_costs"] = jax.vmap(jax.vmap(cost_fn))(eps)
            result["ep_logpriors"] = jax.vmap(jax.vmap(ep_logprior_fn))(eps)
            result["ep_logprobs"] = result["ep_logpriors"] - jax.nn.elu(
                result["failure_level"] - result["ep_costs"]
            )

    return loaded_data


if __name__ == "__main__":
    # Activate seaborn styling
    sns.set_theme(context="paper", style="whitegrid", font_scale=1.5)

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
                        "display_name": data[alg]["display_name"],
                        "ep_costs": data[alg]["ep_costs"],
                        "ep_logpriors": data[alg]["ep_logpriors"],
                        "ep_logprobs": data[alg]["ep_logprobs"],
                        "failure_level": data[alg]["failure_level"],
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

    # Post-process
    for alg in DATA_SOURCES:
        for result in summary_data[alg]:
            failure_level = result["failure_level"]
            costs = result["ep_costs"]
            num_failures = (costs >= failure_level).sum(axis=-1)
            # # Cumulative max = 'how many failures have we seen so far?'
            # num_failures = jax.lax.cummax(num_failures)
            # Add a 1 at the start (randomly sampling 10 failures gives 1 failure at step 0)
            num_failures = jnp.concatenate([jnp.ones(1), num_failures])
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

            # Add the number of failures discovered initially
            iters = pd.concat(
                [iters, pd.Series(jnp.zeros(num_chains, dtype=int))], ignore_index=True
            )
            seeds = pd.concat(
                [seeds, pd.Series(jnp.zeros(num_chains, dtype=int) + seed_i)],
                ignore_index=True,
            )
            num_failures = pd.concat(
                [
                    num_failures,
                    pd.Series([float(result["num_failures"][0])] * num_chains),
                ],
                ignore_index=True,
            )
            logprobs = pd.concat(
                [logprobs, pd.Series(jnp.zeros(num_chains))], ignore_index=True
            )
            costs = pd.concat(
                [costs, pd.Series(jnp.zeros(num_chains))], ignore_index=True
            )
            algs = pd.concat(
                [algs, pd.Series([result["display_name"]] * num_chains)],
                ignore_index=True,
            )

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
    print(df.groupby(["Algorithm"])["# failures discovered"].mean() / 10)
    print("Collision rate (std)")
    print(df.groupby(["Algorithm"])["# failures discovered"].std() / 10)

    # Plot!
    plt.figure(figsize=(12, 8))
    # sns.barplot(
    #     data=df[(df["Diffusion steps"] % 10 == 0)],
    #     x="Diffusion steps",
    #     y="# failures discovered",
    #     hue="Algorithm",
    # )
    sns.lineplot(
        data=df,
        x="Diffusion steps",
        y="# failures discovered",
        hue="Algorithm",
    )

    plt.show()
