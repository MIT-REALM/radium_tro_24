import argparse
import json

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
from tqdm import tqdm

from radium.systems.formation2d.simulator import (
    WindField,
    sample_random_connection_strengths,
    simulate,
)
from radium.systems.hide_and_seek.hide_and_seek_types import (
    Arena,
    MultiAgentTrajectory,
    Trajectory2D,
)

if __name__ == "__main__":
    # Set up arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_prefix", type=str)
    parser.add_argument("--N", type=int, nargs="?", default=100)
    args = parser.parse_args()

    # Hyperparameters
    file_prefix = args.file_prefix
    N = args.N

    prng_key = jrandom.PRNGKey(0)

    # First load the design and identified failure modes
    dp_filename = file_prefix + ".json"
    with open(dp_filename, "r") as f:
        saved_data = json.load(f)

    # Create the game
    width = saved_data["width"]
    height = saved_data["height"]
    R = saved_data["R"]
    n = saved_data["n"]
    duration = saved_data["duration"]
    T = saved_data["T"]
    max_wind_thrust = saved_data["max_wind_thrust"]
    arena = Arena(width, height, R)
    initial_states = jnp.stack(
        (
            jnp.zeros(n) - width / 2.0 + R,
            jnp.linspace(-height / 2.0 + R, height / 2.0 - R, n),
            jnp.zeros(n),
            jnp.zeros(n),
        )
    ).T
    goal_com_position = jnp.array([width / 2.0 - R, 0.0])

    # Wrap the simulator function
    simulate_fn = lambda dp, ep: simulate(
        dp,
        initial_states,
        ep[0],
        ep[1],
        goal_com_position,
        max_wind_thrust=max_wind_thrust,
        duration=duration,
        dt=0.05,
        communication_range=R,
    )

    dp_filename = file_prefix + "_dp_trace.json"
    with open(dp_filename, "r") as f:
        saved_data = json.load(f)

    traj_data = jnp.array(saved_data["seeker_trajs"])
    dp_trace = MultiAgentTrajectory([Trajectory2D(traj) for traj in traj_data])

    prng_key, dp_key = jrandom.split(prng_key)
    dp_keys = jrandom.split(dp_key, dp_trace.trajectories[0].p.shape[2])
    init_dps = jax.vmap(
        lambda key: arena.sample_random_multi_trajectory(
            key, initial_states[:, :2], T=T, fixed=False
        )
    )(dp_keys)

    # Create a test set
    prng_key, test_set_key = jrandom.split(prng_key)
    test_set_keys = jrandom.split(test_set_key, N)
    test_set_wind = jax.vmap(WindField)(test_set_keys)

    prng_key, conn_key = jrandom.split(prng_key)
    conn_keys = jrandom.split(conn_key, N)
    conn = jax.vmap(sample_random_connection_strengths, in_axes=(0, None))(conn_keys, n)

    test_set_eps = (test_set_wind, conn)

    # Define the test set performance as the mean cost over all test examples
    performance_fn = lambda dp, ep: simulate_fn(dp, ep).potential
    test_set_performance_fn = lambda dp: jax.vmap(performance_fn, in_axes=(None, 0))(
        dp, test_set_eps
    ).mean()
    test_set_performance_fn_batched = jax.jit(jax.vmap(test_set_performance_fn))

    # Run the examples on the test set
    T = dp_trace.trajectories[0].p.shape[1]
    test_set_performances = []
    test_set_performances.append(test_set_performance_fn_batched(init_dps))
    for t in tqdm(range(T)):
        dp_t = jtu.tree_map(lambda leaf: leaf[0, t], dp_trace)
        test_set_performances.append(test_set_performance_fn_batched(dp_t))

    # Save the results to a file
    save_filename = file_prefix + "_training_progress.npz"
    with open(save_filename, "wb") as f:
        jnp.save(f, jnp.array(test_set_performances))
