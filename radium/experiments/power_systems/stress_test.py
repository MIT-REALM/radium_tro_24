import argparse
import json

import jax
import jax.numpy as jnp
import jax.random as jrandom
from tqdm import tqdm

from radium.systems.power_systems.acopf_types import (
    Dispatch,
    GenerationDispatch,
    LoadDispatch,
    NetworkState,
)
from radium.systems.power_systems.load_test_network import load_test_network

if __name__ == "__main__":
    # Set up arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str)
    parser.add_argument("--N", type=int, nargs="?", default=100_000)
    parser.add_argument("--batches", type=int, nargs="?", default=10)
    args = parser.parse_args()

    # Hyperparameters
    filename = args.filename
    N = args.N
    batches = args.batches

    prng_key = jrandom.PRNGKey(0)

    # Test the quality of the solution (and the predicted failure modes)
    # by sampling a whole heckin' bunch of exogenous parameters

    # First load the design and identified failure modes
    with open(filename, "r") as f:
        saved_data = json.load(f)

    sys = load_test_network(saved_data["case"], penalty=saved_data["L"])

    gen_json = saved_data["dispatch"]["gen"]
    load_json = saved_data["dispatch"]["load"]
    final_dps = Dispatch(
        GenerationDispatch(
            P=jnp.array(gen_json[0]),
            voltage_amplitudes=jnp.array(gen_json[1]),
        ),
        LoadDispatch(
            P=jnp.array(load_json[0]),
            Q=jnp.array(load_json[1]),
        ),
    )

    predicted_eps = NetworkState(
        line_states=jnp.array(saved_data["network_state"]["line_states"])
    )

    # See how the design performs on the identified failure modes
    result = jax.vmap(sys, in_axes=(None, 0))(final_dps, predicted_eps)
    predicted_violation = (
        result.P_gen_violation.sum(axis=-1)
        + result.Q_gen_violation.sum(axis=-1)
        + result.P_load_violation.sum(axis=-1)
        + result.Q_load_violation.sum(axis=-1)
        + result.V_violation.sum(axis=-1)
    )
    predicted_worst_case = predicted_violation.max()
    num_opf_failues = (result.acopf_residual > 1e-3).sum()
    pct_opf_failues = 100 * num_opf_failues / predicted_eps.line_states.shape[0]
    print(f"Predicted worst case constraint violation: {predicted_worst_case}")
    print(f"\tGauss-Newton failed on {pct_opf_failues:.2f}%")

    # Save the results to a file
    save_filename = filename[:-5] + "_predicted_failures.npz"
    with open(save_filename, "wb") as f:
        jnp.save(f, jnp.array(result.potential).reshape(-1))

    # See how the chosen dispatch performs against a BUNCH of test cases
    stress_test_potentials = []
    stress_test_worst_case = []
    n_gt_predicted = []
    num_opf_failues = []
    line_states = []
    voltage_violations = []
    sim_fn = jax.jit(jax.vmap(sys, in_axes=(None, 0)))
    print("Running stress test")
    for i in tqdm(range(batches)):
        prng_key, stress_test_key = jrandom.split(prng_key)
        stress_test_keys = jrandom.split(stress_test_key, N)
        stress_test_eps = jax.vmap(sys.sample_random_network_state)(stress_test_keys)
        stress_test_result = sim_fn(final_dps, stress_test_eps)
        stress_test_violation = (
            stress_test_result.P_gen_violation.sum(axis=-1)
            + stress_test_result.Q_gen_violation.sum(axis=-1)
            + stress_test_result.P_load_violation.sum(axis=-1)
            + stress_test_result.Q_load_violation.sum(axis=-1)
            + stress_test_result.V_violation.sum(axis=-1)
        )
        stress_test_worst_case.append(stress_test_violation.max())
        n_gt_predicted.append((stress_test_violation > predicted_worst_case).sum())
        num_opf_failues.append((stress_test_result.acopf_residual > 1e-3).sum())
        stress_test_potentials.append(stress_test_result.potential)
        line_states.append(stress_test_eps.line_states)
        voltage_violations.append(stress_test_result.V_violation.sum(axis=-1))

    print("")
    print(f"Worst case identified by stress test: {max(stress_test_worst_case)}")
    print(f"\t{100 * sum(n_gt_predicted) / (N * batches)}% are worse than predicted")
    pct_opf_failues = 100 * sum(num_opf_failues) / (N * batches)
    print(f"\tGauss-Newton failed on {pct_opf_failues:.2f}%")

    # Save the results to a file
    save_filename = filename[:-5] + "_stress_test.npz"
    with open(save_filename, "wb") as f:
        jnp.save(f, jnp.array(stress_test_potentials).reshape(-1))

    save_filename = filename[:-5] + "_stress_test_line_states.npz"
    with open(save_filename, "wb") as f:
        jnp.save(f, jnp.array(line_states))

    save_filename = filename[:-5] + "_stress_test_voltage_violations.npz"
    with open(save_filename, "wb") as f:
        jnp.save(f, jnp.array(voltage_violations).reshape(-1))
