"""Test gradient descent on various potential functions."""
import functools as ft

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt
import numpy as np
import scipy

from radium.experiments.highway.train_highway_agent import make_highway_env
from radium.systems.components.sensing.vision.shapes import Scene, Sphere
from radium.systems.highway.driving_policy import DrivingPolicy
from radium.systems.highway.highway_scene import HighwayScene
from radium.utils import softmin


# A utility function for getting the LQR feedback gain matrix
def dlqr(A, B, Q, R):
    """
    Solve the discrete time lqr controller.

    x[k+1] = A x[k] + B u[k]

    cost = sum x[k].T*Q*x[k] + u[k].T*R*u[k]
    """
    # ref Bertsekas, p.151

    # first, try to solve the ricatti equation
    X = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))

    # compute the LQR gain
    K = np.matrix(scipy.linalg.inv(B.T * X * B + R) * (B.T * X * A))

    eigVals, _ = scipy.linalg.eig(A - B * K)

    return K, X, eigVals


# get LQR gains
A = np.array(
    [
        [1.0, 0.0, 0.0, 0.1],
        [0.0, 1.0, 8.5 * 0.1, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
B = np.array(
    [
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 8.5 * 0.1 / 1.0],
        [0.1, 0.0],
    ]
)
Q = np.eye(4)
R = np.eye(2)
K, _, _ = dlqr(A, B, Q, R)
K = jnp.array(K)


if __name__ == "__main__":
    prng_key = jax.random.PRNGKey(0)

    # Can we optimize for a collision?
    scene = HighwayScene(num_lanes=3, lane_width=5.0, segment_length=200.0)
    env = make_highway_env((32, 32))
    env._initial_state_covariance = 1e-3 * env._initial_state_covariance
    env._shading_light_direction_covariance = (
        1e-3 * env._shading_light_direction_covariance
    )
    # env._initial_non_ego_states = jnp.array([[-100.0, 3.0, 0.0, 10.0]])
    # env._initial_ego_state = jnp.array([-100.0, -3.0, 0.0, 10.0])

    dummy_policy = DrivingPolicy(jrandom.PRNGKey(0), (32, 32))
    policy = eqx.tree_deserialise_leaves(
        "results/highway/initial_policy.eqx", dummy_policy
    )

    prng_key, initial_state_key = jrandom.split(prng_key)
    initial_state = env.reset(initial_state_key)

    def get_distance_to_collision(ego_state, non_ego_state) -> float:
        return scene.check_for_collision(
            ego_state[:3],  # trim out speed; not needed for collision checking
            non_ego_state[:, :3],
            include_ground=False,
        )

    def get_min_distance_to_collision(ego_trajectory, non_ego_trajectory):
        collision_clearances = jax.vmap(get_distance_to_collision)(
            ego_trajectory, non_ego_trajectory
        )
        return softmin(collision_clearances, sharpness=0.5), collision_clearances.min()

    def make_trajectory(initial_state, actions):
        def step(carry, input):
            state, obs = carry
            non_ego_actions = input

            # Add LQR to the non-ego actions
            target = initial_state.non_ego_states
            target = target.at[:, 0].set(state.non_ego_states[:, 0])
            compute_lqr = lambda non_ego_state, initial_state: -K @ (
                non_ego_state - initial_state
            )
            non_ego_stable_action = jax.vmap(compute_lqr)(
                state.non_ego_states,
                target,
            )

            ego_action, _, _ = policy(obs, jrandom.PRNGKey(0), deterministic=True)

            next_state, next_obs, _, _ = env.step(
                state,
                ego_action,
                non_ego_actions + non_ego_stable_action,
                jrandom.PRNGKey(0),
                reset=False,
            )

            return (next_state, next_obs), next_state

        initial_obs = env.get_obs(initial_state)
        _, trajectory = jax.lax.scan(step, (initial_state, initial_obs), actions)

        return trajectory

    def cost(actions):
        trajectory = make_trajectory(initial_state, actions)
        return get_min_distance_to_collision(
            trajectory.ego_state, trajectory.non_ego_states
        )[0]

    # Can we optimize for a collision?
    lr = 1e-2
    steps = 100
    value_and_grad_fn = jax.jit(jax.value_and_grad(cost))
    T = 60
    ep = jnp.zeros((T, initial_state.non_ego_states.shape[0], 2))
    values = []
    for i in range(steps):
        v, g = value_and_grad_fn(ep)
        values.append(v)
        if i % 10 == 0:
            print(f"Step {i}: potential: {v}, grad norm: {jnp.linalg.norm(g)}")
        ep = ep - lr * g  # gradient DESCENT to minimize distance

    trajectory = make_trajectory(initial_state, ep)
    _, min_dist = get_min_distance_to_collision(
        trajectory.ego_state, trajectory.non_ego_states
    )
    print(f"Minimum distance to collision: {min_dist}")

    # Plot the results
    fig = plt.figure(figsize=(32, 16), constrained_layout=True)
    axs = fig.subplot_mosaic([["trajectory"]])

    # Plot the trajectories for each case, color-coded by potential
    axs["trajectory"].axhline(7.5, linestyle="--", color="k")
    axs["trajectory"].axhline(-7.5, linestyle="--", color="k")
    axs["trajectory"].plot(
        trajectory.ego_state[:, 0].T,
        trajectory.ego_state[:, 1].T,
        linestyle="-",
        marker="o",
        label="Ego",
        color="r",
    )
    axs["trajectory"].plot(
        trajectory.non_ego_states[:, 0, 0],
        trajectory.non_ego_states[:, 0, 1],
        linestyle="-.",
        marker="o",
        label="Non-ego 1",
        color="b",
    )
    axs["trajectory"].plot(
        trajectory.non_ego_states[:, 1, 0],
        trajectory.non_ego_states[:, 1, 1],
        linestyle="--",
        marker="o",
        label="Non-ego 1",
        color="b",
    )
    plt.show()
