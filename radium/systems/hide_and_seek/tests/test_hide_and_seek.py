import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from radium.systems.hide_and_seek.hide_and_seek import Game
from radium.systems.hide_and_seek.hide_and_seek_types import (
    MultiAgentTrajectory,
    Trajectory2D,
)


def test_Game(plot=False):
    # Create the game
    game = Game(
        jnp.array([[0.0, 0.0], [0.0, 1.0]]),
        jnp.array([[1.0, 0.5], [1.0, 1.5]]),
        duration=100.0,
        dt=0.1,
        sensing_range=0.5,
        seeker_max_speed=0.5,
        hider_max_speed=0.5,
        b=5.0,
    )

    # Define test trajectories
    p1 = jnp.array([[0.0, 0.0], [0.9, 0.5], [1.0, 0.0]])
    traj1 = Trajectory2D(p1)
    p2 = jnp.array([[0.0, 1.0], [0.9, 0.5], [1.0, 1.0]])
    traj2 = Trajectory2D(p2)
    seeker_traj = MultiAgentTrajectory([traj1, traj2])

    p1 = jnp.array([[1.0, 0.5], [0.0, 0.5]])
    traj1 = Trajectory2D(p1)
    p2 = jnp.array([[1.0, 1.5], [0.0, 1.5]])
    traj2 = Trajectory2D(p2)
    hider_traj = MultiAgentTrajectory([traj1, traj2])

    seeker_disturbance_traj = MultiAgentTrajectory(
        [
            Trajectory2D(jnp.zeros((2, 2))),
            Trajectory2D(jnp.zeros((2, 2))),
        ]
    )

    # Evaluate the game
    result = game(seeker_traj, hider_traj, seeker_disturbance_traj)

    # Plot if requested
    if plot:
        # Plot initial setup
        plt.scatter(
            game.initial_seeker_positions[:, 0],
            game.initial_seeker_positions[:, 1],
            color="b",
            marker="x",
            label="Initial seeker positions",
        )
        plt.scatter(
            game.initial_hider_positions[:, 0],
            game.initial_hider_positions[:, 1],
            color="r",
            marker="x",
            label="Initial hider positions",
        )

        # Plot planned trajectories
        t = jnp.linspace(0, 1, 100)
        for traj in seeker_traj.trajectories:
            pts = jax.vmap(traj)(t)
            plt.plot(pts[:, 0], pts[:, 1], "b:")

        for traj in hider_traj.trajectories:
            pts = jax.vmap(traj)(t)
            plt.plot(pts[:, 0], pts[:, 1], "r:")

        # Plot resulting trajectories
        for i in range(game.initial_seeker_positions.shape[0]):
            plt.plot(
                result.seeker_positions[:, i, 0], result.seeker_positions[:, i, 1], "b-"
            )

        for i in range(game.initial_hider_positions.shape[0]):
            plt.plot(
                result.hider_positions[:, i, 0], result.hider_positions[:, i, 1], "r-"
            )

        plt.legend()
        plt.show()


def benchmark_Game():
    # Create the game
    game = Game(
        jnp.array([[0.0, 0.0], [0.0, 1.0]]),
        jnp.array([[1.0, 0.5], [1.0, 1.5]]),
        duration=100.0,
        dt=0.1,
        sensing_range=0.5,
        seeker_max_speed=0.5,
        hider_max_speed=0.5,
        b=5.0,
    )

    # Define test trajectories
    p1 = jnp.array([[0.0, 0.0], [0.9, 0.5], [1.0, 0.0]])
    traj1 = Trajectory2D(p1)
    p2 = jnp.array([[0.0, 1.0], [0.9, 0.5], [1.0, 1.0]])
    traj2 = Trajectory2D(p2)
    seeker_traj = MultiAgentTrajectory([traj1, traj2])

    p1 = jnp.array([[1.0, 0.5], [0.0, 0.5]])
    traj1 = Trajectory2D(p1)
    p2 = jnp.array([[1.0, 1.5], [0.0, 1.5]])
    traj2 = Trajectory2D(p2)
    hider_traj = MultiAgentTrajectory([traj1, traj2])

    seeker_disturbance_traj = MultiAgentTrajectory(
        [
            Trajectory2D(jnp.zeros((2, 2))),
            Trajectory2D(jnp.zeros((2, 2))),
        ]
    )

    # See how fast it takes to run
    N_trials = 10
    potential_fn = lambda seeker_traj, hider_traj, seeker_disturbance_traj: game(
        seeker_traj, hider_traj, seeker_disturbance_traj
    ).potential
    potential_fn = jax.jit(potential_fn)

    jit_start_time = time.perf_counter()
    potential_fn(seeker_traj, hider_traj, seeker_disturbance_traj)
    jit_end_time = time.perf_counter()
    print(f"Time to jit (no grad): {jit_end_time - jit_start_time} s")
    runtime = 0.0
    for _ in range(N_trials):
        start_time = time.perf_counter()
        potential_fn(seeker_traj, hider_traj, seeker_disturbance_traj)
        runtime += time.perf_counter() - start_time

    print(f"Average runtime (w/ jit): {runtime / N_trials} s")

    # Also test the gradient quality
    @jax.jit
    def f(z):
        p1 = jnp.array([[1.0, z + 0.5], [0.0, z + 0.5]])
        traj1 = Trajectory2D(p1)
        p2 = jnp.array([[1.0, z + 1.5], [0.0, z + 1.5]])
        traj2 = Trajectory2D(p2)
        hider_traj = MultiAgentTrajectory([traj1, traj2])
        return potential_fn(seeker_traj, hider_traj, seeker_disturbance_traj)

    grad_f = jax.jit(jax.grad(f))
    jit_start_time = time.perf_counter()
    grad_f(0.0)
    jit_end_time = time.perf_counter()
    print(f"Time to jit (yes grad): {jit_end_time - jit_start_time} s")
    runtime = 0.0
    for _ in range(N_trials):
        start_time = time.perf_counter()
        grad_f(0.0)
        runtime += time.perf_counter() - start_time

    print(f"Average runtime (w/ grad and jit): {runtime / N_trials} s")

    z_vals = jnp.linspace(-0.5, 0.5, 100)
    f_vals = jax.vmap(f)(z_vals)
    f_grad_vals = jax.vmap(jax.grad(f))(z_vals)
    plt.plot(z_vals, f_vals, "k--")
    plt.plot(z_vals, f_grad_vals, "r-", label="Gradient")
    f_grad_vals_fd = np.gradient(f_vals, z_vals)
    plt.plot(z_vals, f_grad_vals_fd, "b--", label="Finite diff")
    plt.show()


if __name__ == "__main__":
    # test_Game(plot=True)
    benchmark_Game()
