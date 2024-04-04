import argparse
import json

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu

from radium.systems.power_systems.acopf_types import (
    Dispatch,
    GenerationDispatch,
    LoadDispatch,
)
from radium.systems.power_systems.load_test_network import load_test_network

if __name__ == "__main__":
    # Set up arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str)
    parser.add_argument("--N", type=int, nargs="?", default=100)
    args = parser.parse_args()

    # Hyperparameters
    filename = args.filename
    N = args.N

    prng_key = jrandom.PRNGKey(0)

    # First load the design and identified failure modes
    with open(filename, "r") as f:
        saved_data = json.load(f)

    sys = load_test_network(saved_data["case"], penalty=100.0)

    gen_json = saved_data["dispatch"]["gen"]
    load_json = saved_data["dispatch"]["load"]
    dp_trace = Dispatch(
        GenerationDispatch(
            P=jnp.array(gen_json[0]),
            voltage_amplitudes=jnp.array(gen_json[1]),
        ),
        LoadDispatch(
            P=jnp.array(load_json[0]),
            Q=jnp.array(load_json[1]),
        ),
    )

    prng_key, dispatch_key = jrandom.split(prng_key)
    dispatch_keys = jrandom.split(dispatch_key, dp_trace.gen.P.shape[1])
    init_dps = jax.vmap(sys.sample_random_dispatch)(dispatch_keys)

    # Create a test set
    prng_key, test_set_key = jrandom.split(prng_key)
    test_set_keys = jrandom.split(test_set_key, N)
    test_set_eps = jax.vmap(sys.sample_random_network_state)(test_set_keys)

    # Define the test set performance as the mean cost over all test examples
    performance_fn = lambda dp, ep: sys(dp, ep).potential
    test_set_performance_fn = lambda dp: jax.vmap(performance_fn, in_axes=(None, 0))(
        dp, test_set_eps
    ).mean()
    test_set_performance_fn_batched = jax.jit(jax.vmap(test_set_performance_fn))

    # Run the examples on the test set
    T = dp_trace.gen.P.shape[0]
    test_set_performances = []
    test_set_performances.append(test_set_performance_fn_batched(init_dps))
    for t in range(T):
        dp_t = jtu.tree_map(lambda leaf: leaf[t], dp_trace)
        test_set_performances.append(test_set_performance_fn_batched(dp_t))
        print(f"Step {t}")

    # Save the results to a file
    save_filename = filename[:-5] + "_training_progress.npz"
    with open(save_filename, "wb") as f:
        jnp.save(f, jnp.array(test_set_performances))
