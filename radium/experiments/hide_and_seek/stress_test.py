import argparse
import json

import jax
import jax.numpy as jnp
import jax.random as jrandom
from tqdm import tqdm

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
    parser.add_argument("--N", type=int, nargs="?", default=10_000)
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

    # Create the game
    width = saved_data["width"]
    height = saved_data["height"]
    buffer = saved_data["buffer"]
    n_seekers = saved_data["n_seekers"]
    n_hiders = saved_data["n_hiders"]
    duration = saved_data["duration"]
    T = saved_data["T"]
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

    seeker_traj = jnp.array(saved_data["seeker_traj"]["trajectories"])
    final_dps = MultiAgentTrajectory([Trajectory2D(traj) for traj in seeker_traj])
    hider_trajs = jnp.array(saved_data["hider_trajs"]["trajectories"])
    hider_trajs = MultiAgentTrajectory([Trajectory2D(traj) for traj in hider_trajs])
    disturbance_trajs = jnp.array(saved_data["disturbance_trajs"]["trajectories"])
    disturbance_trajs = MultiAgentTrajectory(
        [Trajectory2D(traj) for traj in disturbance_trajs]
    )
    predicted_eps = (hider_trajs, disturbance_trajs)

    # See how the design performs on the identified failure modes
    result = jax.vmap(game, in_axes=(None, 0, 0))(
        final_dps, hider_trajs, disturbance_trajs
    )
    predicted_worst_case = result.potential.max()

    # Save the results to a file
    save_filename = filename[:-5] + "_predicted_failures.npz"
    with open(save_filename, "wb") as f:
        jnp.save(f, jnp.array(result.potential).reshape(-1))

    # See how the chosen dispatch performs against a BUNCH of test cases
    stress_test_potentials = []
    stress_test_worst_case = []
    n_gt_predicted = []
    sim_fn = jax.jit(jax.vmap(game, in_axes=(None, 0, 0)))
    print("Running stress test")
    for i in tqdm(range(batches)):
        prng_key, hider_key = jrandom.split(prng_key)
        hider_keys = jrandom.split(hider_key, N)
        hider_trajs = jax.vmap(
            lambda key: arena.sample_random_multi_trajectory(
                key, initial_hider_positions, T=T
            )
        )(hider_keys)
        prng_key, dist_key = jrandom.split(prng_key)
        dist_keys = jrandom.split(dist_key, N)
        disturbance_trajs = jax.vmap(
            lambda key: arena.sample_random_multi_trajectory(
                key, jnp.zeros_like(initial_seeker_positions), T=T
            )
        )(dist_keys)

        stress_test_result = sim_fn(final_dps, hider_trajs, disturbance_trajs)
        stress_test_worst_case.append(stress_test_result.potential.max())
        n_gt_predicted.append(
            (stress_test_result.potential > predicted_worst_case).sum()
        )
        stress_test_potentials.append(stress_test_result.potential)

    print("")
    print(filename)
    print(f"Testes on {N * batches} random eps")
    print(f"Predicted worst case potential: {predicted_worst_case}")
    print(f"Worst case identified by stress test: {max(stress_test_worst_case)}")
    print(f"{100 * sum(n_gt_predicted) / (N * batches)}% are worse than predicted")

    # Save the results to a file
    save_filename = filename[:-5] + "_stress_test.npz"
    with open(save_filename, "wb") as f:
        jnp.save(f, jnp.array(stress_test_potentials).reshape(-1))
