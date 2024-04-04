import argparse
import json

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from tqdm import tqdm

from radium.systems.push_tilt.simulator import (
    Planner,
    sample_object_parameters,
    sample_observations,
    simulate,
)

"""
CUDA_VISIBLE_DEVICES=2, XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 python radium/experiments/push_tilt/stress_test.py --file_prefix results/push_tilt/L_1.0e+01/100_samples/0_quench/tempered10_chains/dp_1.0e-03/ep_1.0e-03/repair/predict/mala &
CUDA_VISIBLE_DEVICES=2, XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 python radium/experiments/push_tilt/stress_test.py --file_prefix results/push_tilt/L_1.0e+01/100_samples/0_quench/tempered10_chains/dp_1.0e-03/ep_1.0e-03/repair/predict/gd &
CUDA_VISIBLE_DEVICES=3, XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 python radium/experiments/push_tilt/stress_test.py --file_prefix results/push_tilt/L_1.0e+01/100_samples/0_quench/tempered10_chains/dp_1.0e-03/ep_1.0e-03/repair/predict/reinforce &
CUDA_VISIBLE_DEVICES=3, XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 python radium/experiments/push_tilt/stress_test.py --file_prefix results/push_tilt/L_1.0e+01/100_samples/0_quench/tempered10_chains/dp_1.0e-03/ep_1.0e-03/repair/predict/rmh &
CUDA_VISIBLE_DEVICES=3, XLA_PYTHON_CLIENT_MEM_FRACTION=0.2 python radium/experiments/push_tilt/stress_test.py --file_prefix results/push_tilt/L_1.0e+01/100_samples/0_quench/tempered10_chains/dp_1.0e-03/ep_1.0e-03/repair/no_predict/gd &
"""

if __name__ == "__main__":
    # Set up arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_prefix", type=str)
    parser.add_argument("--N", type=int, nargs="?", default=10_000)
    parser.add_argument("--batches", type=int, nargs="?", default=10)
    args = parser.parse_args()

    # Hyperparameters
    file_prefix = args.file_prefix
    N = args.N
    batches = args.batches

    prng_key = jrandom.PRNGKey(0)

    # Test the quality of the solution (and the predicted failure modes)
    # by sampling a whole heckin' bunch of exogenous parameters

    # First load the design and identified failure modes
    ep_filename = file_prefix + ".json"
    with open(ep_filename, "r") as f:
        saved_data = json.load(f)

    # Load the exogenous parameters
    eps = (jnp.array(saved_data["final_eps"][0]), jnp.array(saved_data["final_eps"][1]))

    dp_filename = file_prefix + ".eqx"
    dummy_controller = Planner(prng_key)
    final_dps = eqx.tree_deserialise_leaves(dp_filename, dummy_controller)

    # Wrap the simulator function
    simulate_fn = lambda dp, ep: simulate(dp, ep[1], ep[0])

    # See how the design performs on the identified failure modes
    result = jax.vmap(simulate_fn, in_axes=(None, 0))(final_dps, eps)
    predicted_worst_case = result.potential.max()

    # Save the results to a file
    save_filename = file_prefix + "_predicted_failures.npz"
    with open(save_filename, "wb") as f:
        jnp.save(f, jnp.array(result.potential).reshape(-1))

    # See how the chosen dispatch performs against a BUNCH of test cases
    stress_test_potentials = []
    stress_test_worst_case = []
    n_gt_predicted = []
    sim_fn = jax.jit(jax.vmap(simulate_fn, in_axes=(None, 0)))
    print("Running stress test")
    for i in tqdm(range(batches)):
        prng_key, param_keys = jrandom.split(prng_key)
        param_keys = jrandom.split(param_keys, N)
        stress_test_params = jax.vmap(sample_object_parameters)(param_keys)
        prng_key, obs_keys = jrandom.split(prng_key)
        obs_keys = jrandom.split(obs_keys, N)
        stress_test_observations = jax.vmap(sample_observations)(
            obs_keys, stress_test_params
        )
        stress_test_eps = (stress_test_params, stress_test_observations)

        stress_test_result = sim_fn(final_dps, stress_test_eps)
        stress_test_worst_case.append(stress_test_result.potential.max())
        n_gt_predicted.append(
            (stress_test_result.potential > predicted_worst_case).sum()
        )
        stress_test_potentials.append(stress_test_result.potential)

    print("")
    print(file_prefix)
    print(f"Tested on {N * batches} random eps")
    print(f"Predicted worst case potential: {predicted_worst_case}")
    print(f"Worst case identified by stress test: {max(stress_test_worst_case)}")
    print(f"{100 * sum(n_gt_predicted) / (N * batches)}% are worse than predicted")

    # Save the results to a file
    save_filename = file_prefix + "_stress_test.npz"
    with open(save_filename, "wb") as f:
        jnp.save(f, jnp.array(stress_test_potentials).reshape(-1))
