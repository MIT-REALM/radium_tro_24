"""Implement an oracle for the intersection problem."""
import functools

import jax
import jax.nn
import jax.numpy as jnp
import jax.random as jrandom
import optax
from beartype import beartype
from beartype.typing import NamedTuple, Tuple
from jaxtyping import Array, Bool, Float, jaxtyped

from radium.experiments.highway.predict_and_mitigate import K
from radium.systems.components.sensing.vision.render import (
    CameraExtrinsics,
    CameraIntrinsics,
)
from radium.systems.highway.highway_env import HighwayState
from radium.systems.intersection.env import IntersectionEnv
from radium.types import PRNGKeyArray


@functools.partial(jax.jit, static_argnums=(3,))
def rollout(
    initial_state: HighwayState,
    ego_action_sequence: Float[Array, "T 2"],
    non_ego_action_sequence: Float[Array, "T n_non_ego 2"],
    env: IntersectionEnv,
) -> Float[Array, ""]:
    """
    Rollout the environment and get the cost of the rollout.

    Args:
        initial_state: the initial state of the environment.
        ego_action_sequence: the sequence of actions to take.
        non_ego_action_sequence: the sequence of actions to take for the non-ego cars.
        env: the environment.

    Returns:
        The cost of the rollout.
    """

    # Create a function to take one step with the policy
    def step(state, action):
        ego_action, non_ego_action = action

        # Update the state using the dynamics
        next_ego_state = env.car_dynamics(state.ego_state, ego_action)
        next_non_ego_states = jax.vmap(env.car_dynamics)(
            state.non_ego_states, non_ego_action
        )
        next_state = HighwayState(
            next_ego_state,
            next_non_ego_states,
            state.shading_light_direction,
            state.non_ego_colors,
        )

        # Get the running cost (squared distance to goal plus lane keeping)
        target = jnp.array([20.0, -7.5 / 2, 0.0, 3.0])
        Q = jnp.diag(jnp.array([0.1, 2.0, 2.0, 1.0]))
        state_err = next_ego_state - target
        cost = state_err.T @ Q @ state_err * env._dt

        # Add the collision cost
        min_distance_to_obstacle = env._scene.check_for_collision(
            next_ego_state[:3],  # trim out speed; not needed for collision checking
            next_non_ego_states[:, :3],
            50.0,
        )
        cost += 1000.0 * jax.nn.sigmoid(-10 * (min_distance_to_obstacle))

        return next_state, cost

    # Run the rollout
    _, costs = jax.lax.scan(
        step, initial_state, (ego_action_sequence, non_ego_action_sequence)
    )

    # Return the total cost
    return jnp.mean(costs)


@functools.partial(
    jax.jit,
    static_argnums=(
        1,
        2,
        4,
    ),
)
def oracle_policy(
    initial_state: HighwayState,
    horizon: int,
    env: IntersectionEnv,
    key: PRNGKeyArray,
    population_size: int = 10_000,
) -> Float[Array, "horizon 2"]:
    """
    A policy that uses total information about the environment to compute the optimal
    action sequence.

    Assumes the non-ego agents drive straight.

    Args:
        initial_state: the initial state of the environment.
        horizon: the number of steps to take in the rollout.
        env: the environment.
        key: a random number generator key.
        population_size: the number of action sequences to sample.
    """
    # Sample a population of action sequences
    a_sequence = jax.random.multivariate_normal(
        key, jnp.zeros(horizon), 2.0 * jnp.eye(horizon), (population_size,)
    )
    delta_sequence = jax.random.multivariate_normal(
        key, jnp.zeros(horizon), 2.0 * jnp.eye(horizon), (population_size,)
    )
    action_sequence = jnp.stack((a_sequence, delta_sequence), axis=-1)

    # Assume the non-ego agents drive straight
    non_ego_action_sequence = jnp.zeros(
        (horizon, initial_state.non_ego_states.shape[0], 2)
    )

    # Rollout the action sequences
    costs = jax.vmap(rollout, in_axes=(None, 0, None, None))(
        initial_state, action_sequence, non_ego_action_sequence, env
    )

    # Select the best actions using softmax
    # action_logits = jax.nn.softmax(-costs)
    # action_sequence = jnp.sum(action_logits[:, None, None] * action_sequence, axis=0)
    best_idx = jnp.argmin(costs)
    action_sequence = action_sequence[best_idx]

    # Return the first action from the action sequence
    return action_sequence[0]
