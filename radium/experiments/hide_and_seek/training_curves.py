import argparse
import json

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu

from radium.systems.hide_and_seek.hide_and_seek import Game
from radium.systems.hide_and_seek.hide_and_seek_types import (
    Arena,
    MultiAgentTrajectory,
    Trajectory2D,
)

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

    # Create the game
    width = saved_data["width"]
    height = saved_data["height"]
    buffer = saved_data["buffer"]
    n_seekers = saved_data["n_seekers"]
    n_hiders = saved_data["n_hiders"]
    duration = saved_data["duration"]
    arena = Arena(width, height, buffer)
    initial_seeker_positions = jnp.stack(
        (
            jnp.zeros(n_seekers) - width / 2.0 + buffer,
            jnp.linspace(-height / 2.0 + buffer, height / 2.0 - buffer, n_seekers),
        )
    ).T
    initial_hider_positions = jnp.stack(
        (
            jnp.zeros(n_hiders) + width / 2.0 - buffer,
            jnp.linspace(-height / 2.0 + buffer, height / 2.0 - buffer, n_hiders),
        )
    ).T
    game = Game(
        initial_seeker_positions,
        initial_hider_positions,
        duration=duration,
        dt=0.1,
        sensing_range=0.5,
        seeker_max_speed=0.2,
        hider_max_speed=0.2,
        b=100.0,
    )

    seeker_traj = jnp.array(saved_data["seeker_trajs"])
    dp_trace = MultiAgentTrajectory([Trajectory2D(traj) for traj in seeker_traj])

    prng_key, dp_key = jrandom.split(prng_key)
    dp_keys = jrandom.split(dp_key, dp_trace.trajectories[0].p.shape[2])
    init_dps = jax.vmap(
        lambda key: arena.sample_random_multi_trajectory(
            key, initial_seeker_positions, T=5
        )
    )(dp_keys)

    # Create a test set
    prng_key, hider_key = jrandom.split(prng_key)
    hider_keys = jrandom.split(hider_key, N)
    hider_traj = jax.vmap(
        lambda key: arena.sample_random_multi_trajectory(
            key, initial_hider_positions, T=5
        )
    )(hider_keys)
    prng_key, dist_key = jrandom.split(prng_key)
    dist_keys = jrandom.split(dist_key, N)
    dist_traj = jax.vmap(
        lambda key: arena.sample_random_multi_trajectory(
            key, jnp.zeros_like(initial_seeker_positions), T=5
        )
    )(dist_keys)
    test_set_eps = (hider_traj, dist_traj)

    # Define the test set performance as the mean cost over all test examples
    performance_fn = lambda dp, ep: game(dp, ep[0], ep[1]).potential
    test_set_performance_fn = lambda dp: jax.vmap(performance_fn, in_axes=(None, 0))(
        dp, test_set_eps
    ).mean()
    test_set_performance_fn_batched = jax.jit(jax.vmap(test_set_performance_fn))

    # Run the examples on the test set
    T = dp_trace.trajectories[0].p.shape[1]
    test_set_performances = []
    test_set_performances.append(test_set_performance_fn_batched(init_dps))
    for t in range(T):
        dp_t = jtu.tree_map(lambda leaf: leaf[0, t], dp_trace)
        test_set_performances.append(test_set_performance_fn_batched(dp_t))
        print(f"Step {t}")

    # Save the results to a file
    save_filename = filename[:-5] + "_training_progress.npz"
    with open(save_filename, "wb") as f:
        jnp.save(f, jnp.array(test_set_performances))
