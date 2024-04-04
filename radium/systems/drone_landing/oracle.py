"""Implement an oracle for the drone landing problem."""
import functools

import jax
import jax.nn
import jax.numpy as jnp
import jax.random as jrandom
import optax
from beartype import beartype
from beartype.typing import NamedTuple, Tuple
from jaxtyping import Array, Bool, Float, jaxtyped

from radium.systems.components.sensing.vision.render import (
    CameraExtrinsics,
    CameraIntrinsics,
)
from radium.systems.drone_landing.env import DroneLandingEnv, DroneState
from radium.types import PRNGKeyArray


@functools.partial(jax.jit, static_argnums=(2,))
def rollout(
    initial_state: DroneState,
    action_sequence: Float[Array, "T 2"],
    env: DroneLandingEnv,
) -> Float[Array, ""]:
    """
    Rollout the environment and get the cost of the rollout.

    Args:
        x_init: the initial state of the drone.
        action_sequence: the sequence of actions to take.
        env: the environment.

    Returns:
        The cost of the rollout.
    """

    # Create a function to take one step with the policy
    def step(state, action):
        # Update the state using the dynamics
        new_drone_state = env.drone_dynamics(
            state.drone_state, action, state.wind_speed
        )
        # Update the tree positions
        new_tree_locations = state.tree_locations + state.tree_velocities * env._dt
        new_state = DroneState(
            new_drone_state,
            new_tree_locations,
            state.tree_velocities,
            state.wind_speed,
        )

        # Get the running cost (squared distance to goal)
        target = jnp.array([0.0, 0.0, 0.5, 0.0])
        cost = jnp.sum((new_drone_state - target)[:3] ** 2) * env._dt

        # Add a cost for bearing error relative to the goal
        bearing_to_goal = jnp.arctan2(
            target[1] - new_drone_state[1], target[0] - new_drone_state[0]
        )
        bearing_error = new_drone_state[3] - bearing_to_goal
        cost += 2 * jnp.sum(bearing_error**2) * env._dt

        # Add the collision cost
        min_distance_to_obstacle = env._scene.check_for_collision(
            new_drone_state[:3],  # trim out yaw; not needed for collision checking
            new_state.wind_speed,
            new_state.tree_locations,
            50.0,
        )
        cost += 1e2 * jax.nn.sigmoid(-10 * (min_distance_to_obstacle - 0.1))

        return new_state, cost

    # Run the rollout
    _, costs = jax.lax.scan(step, initial_state, action_sequence)

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
    initial_state: DroneState,
    horizon: int,
    env: DroneLandingEnv,
    key: PRNGKeyArray,
    optim_iters: int = 50,
) -> Float[Array, "horizon 2"]:
    """
    A policy that uses total information about the environment to compute the optimal
    action sequence.

    Args:
        initial_state: the initial state of the environment.
        horizon: the number of steps to take in the rollout.
        env: the environment.
        key: a random number generator key.
        optim_iters: the number of iterations to run the optimizer for.
    """
    # Define the loss and grad function
    grad_fn = jax.grad(rollout, argnums=1)

    # Initialize a fixed action sequence
    action_sequence = jnp.zeros((horizon, 2))
    action_sequence = action_sequence.at[:, 0].set(1.0)

    # Define an optimizer
    optimizer = optax.adam(1e-2)
    opt_state = optimizer.init(action_sequence)

    # Define a function to step the optimizer
    def step_optim(_, carry):
        # Unpack carry
        opt_state, action_sequence = carry

        # Compute the loss and grad, then step the optimizer
        grad = grad_fn(initial_state, action_sequence, env)
        updates, new_opt_state = optimizer.update(grad, opt_state)
        new_action_sequence = optax.apply_updates(action_sequence, updates)

        # Return the new carry
        return new_opt_state, new_action_sequence

    # Run the optimizer in a jax for loop
    opt_state, action_sequence = jax.lax.fori_loop(
        0, optim_iters, step_optim, (opt_state, action_sequence)
    )

    # Return the first action from the action sequence
    return action_sequence[0]
