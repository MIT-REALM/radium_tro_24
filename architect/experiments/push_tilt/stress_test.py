import argparse
import json

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from tqdm import tqdm

from architect.systems.f16.simulator import (
    ResidualControl,
    initial_state_logprior,
    sample_initial_states,
    simulate,
)

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
    eps = jnp.array(saved_data["final_eps"])

    dp_filename = file_prefix + ".eqx"
    n_chains = 10
    dummy_controllers = jax.vmap(lambda _: ResidualControl(prng_key))(
        jnp.arange(n_chains)
    )
    final_dps = eqx.tree_deserialise_leaves(dp_filename, dummy_controllers)

    # Wrap the simulator function
    simulate_fn = lambda dp, ep: simulate(ep, dp)  # weird argument order

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
        prng_key, stress_test_key = jrandom.split(prng_key)
        stress_test_keys = jrandom.split(stress_test_key, N)
        initial_states = jax.vmap(sample_initial_states)(stress_test_keys)

        stress_test_result = sim_fn(final_dps, initial_states)
        stress_test_worst_case.append(stress_test_result.potential.max())
        n_gt_predicted.append(
            (stress_test_result.potential > predicted_worst_case).sum()
        )
        stress_test_potentials.append(stress_test_result.potential)

    print("")
    print(file_prefix)
    print(f"Testes on {N * batches} random eps")
    print(f"Predicted worst case potential: {predicted_worst_case}")
    print(f"Worst case identified by stress test: {max(stress_test_worst_case)}")
    print(f"{100 * sum(n_gt_predicted) / (N * batches)}% are worse than predicted")

    # Save the results to a file
    save_filename = file_prefix + "_stress_test.npz"
    with open(save_filename, "wb") as f:
        jnp.save(f, jnp.array(stress_test_potentials).reshape(-1))
