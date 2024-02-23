"""Test a simple policy for the highway environment."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt
from beartype.typing import NamedTuple, Tuple
from jaxtyping import Array, Float

from architect.experiments.highway.predict_and_mitigate import (
    K,
    LinearTrajectory2D,
    MultiAgentTrajectoryLinear,
)
from architect.systems.components.sensing.vision.render import CameraIntrinsics
from architect.systems.highway.driving_policy import DrivingPolicy
from architect.systems.highway.highway_env import HighwayEnv, HighwayObs, HighwayState
from architect.systems.highway.highway_scene import HighwayScene
from architect.utils import softmin


def make_highway_env(image_shape: Tuple[int, int], focal_length: float = 0.1):
    """Make the highway environment."""
    scene = HighwayScene(num_lanes=3, lane_width=5.0, segment_length=200.0)
    intrinsics = CameraIntrinsics(
        focal_length=focal_length,
        sensor_size=(0.1, 0.1),
        resolution=image_shape,
    )
    initial_ego_state = jnp.array([-100.0, -3.0, 0.0, 10.0])
    initial_non_ego_states = jnp.array(
        [
            [-90.0, -3.0, 0.0, 7.0],
            [-70, 3.0, 0.0, 8.0],
        ]
    )
    initial_state_covariance = jnp.diag(jnp.array([0.5, 0.5, 0.001, 0.5]) ** 2)

    # Set the direction of light shading
    shading_light_direction = jnp.array([-0.2, -1.0, 1.5])
    shading_light_direction /= jnp.linalg.norm(shading_light_direction)
    shading_direction_covariance = (0.25) ** 2 * jnp.eye(3)

    env = HighwayEnv(
        scene,
        intrinsics,
        dt=0.1,
        initial_ego_state=initial_ego_state,
        initial_non_ego_states=initial_non_ego_states,
        initial_state_covariance=initial_state_covariance,
        collision_penalty=5.0,
        mean_shading_light_direction=shading_light_direction,
        shading_light_direction_covariance=shading_direction_covariance,
    )
    return env


# Utilities for blurring the image to get a simple CBF-style control action
def gaussian_kernel(size, sigma=1.0):
    """Generates a 2D Gaussian kernel."""
    ax = jnp.arange(-size // 2, size // 2)
    xx, yy = jnp.meshgrid(ax, ax)
    kernel = jnp.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return kernel / jnp.sum(kernel)


class SimpleDepthPolicy(eqx.Module):
    """Define a policy that decides how much to brake/steer based on a depth image."""

    actor_fcn: eqx.nn.MLP
    trajectory: LinearTrajectory2D

    def __init__(self, trajectory, key, image_shape):
        """Initialize the policy."""
        super().__init__()
        self.actor_fcn = eqx.nn.MLP(
            in_size=image_shape[0] * image_shape[1],
            out_size=2,
            width_size=32,
            depth=3,
            key=key,
            activation=jax.nn.tanh,
        )
        self.trajectory = LinearTrajectory2D(p=trajectory)

    def __call__(
        self,
        obs: HighwayObs,
        t: Float[Array, ""],
        initial_ego_state: Float[Array, " 4"],
    ) -> Float[Array, " 2"]:
        """Compute the action.

        Args:
            obs: The observation of the current state.
            t: The current time.
            initial_ego_state: The initial state of the ego vehicle.

        Returns:
            The action
        """
        # There are three components to the action:
        # 1. Steer towards the current point on the trajectory
        # 2. Brake to avoid collisions
        # 3. add a residual term from the neural network

        # First, steer towards the current point on the trajectory
        waypoint = self.trajectory(t)
        target = initial_ego_state
        target = target.at[2].set(0.0)
        target = target.at[:2].add(waypoint)
        lqr_action = -K @ (obs.ego_state - target)

        # Second, brake to avoid collisions based on the average distance to the
        # obstacle in the center of the image
        depth_image = obs.depth_image
        depth_image = jnp.where(depth_image < 1e-3, 100.0, depth_image)
        kernel = gaussian_kernel(depth_image.shape[0], 1.0)
        mean_distance = jnp.sum(depth_image * kernel) / jnp.sum(kernel)
        min_distance = 10.0
        vision_accel = jnp.clip(mean_distance - min_distance, 0.0, 2.0)

        # Third, get the steering and braking action from the neural network
        image = jnp.reshape(depth_image, (-1,))
        residual_action = 0.1 * self.actor_fcn(image)

        # Add them all together
        action = lqr_action
        action = action.at[0].add(vision_accel)
        action = action + residual_action

        return action


class SimulationResults(NamedTuple):
    """A class for storing the results of a simulation."""

    potential: Float[Array, ""]
    initial_obs: HighwayObs
    final_obs: HighwayObs
    ego_trajectory: Float[Array, "T 4"]
    non_ego_trajectory: Float[Array, "T n_non_ego 4"]


def simulate(
    env: HighwayEnv,
    policy: DrivingPolicy,
    initial_state: HighwayState,
    non_ego_reference_trajectory: MultiAgentTrajectoryLinear,
    static_policy: DrivingPolicy,
    max_steps: int = 60,
) -> Float[Array, ""]:
    """Simulate the highway environment.

    Disables randomness in the policy and environment (all randomness should be
    factored out into the initial_state argument).

    If the environment terminates before `max_steps` steps, it will not be reset and
    all further reward will be zero.

    Args:
        env: The environment to simulate.
        policy: The parts of the policy that are design parameters.
        initial_state: The initial state of the environment.
        non_ego_reference_trajectory: The reference trajectory for the non-ego agents.
        static_policy: the parts of the policy that are not design parameters.
        max_steps: The maximum number of steps to simulate.

    Returns:
        SimulationResults object
    """
    # Merge the policy back together
    policy = eqx.combine(policy, static_policy)

    @jax.checkpoint
    def step(carry, scan_inputs):
        # Unpack the input
        key, t = scan_inputs
        reference_waypoint = non_ego_reference_trajectory(t)

        # Unpack the carry
        action, state, already_done = carry

        # Track the reference waypoint using an LQR controller
        compute_lqr = lambda non_ego_state, waypoint_state: -K @ (
            non_ego_state - waypoint_state
        )
        target = initial_state.non_ego_states  # copy initial heading, velocity, etc.
        # add the waypoint relative to the initial position
        target = target.at[:, :2].add(reference_waypoint)
        non_ego_stable_action = jax.vmap(compute_lqr)(state.non_ego_states, target)

        # Take a step in the environment using the action carried over from the previous
        # step.
        next_state, next_observation, reward, done = env.step(
            state, action, non_ego_stable_action, key
        )

        # Compute the action for the next step
        next_action = policy(next_observation, t, initial_state.ego_state)

        # If the environment has already terminated, set the reward to zero.
        reward = jax.lax.cond(already_done, lambda: 0.0, lambda: reward)
        already_done = jnp.logical_or(already_done, done)

        # Don't step if the environment has terminated
        next_action = jax.lax.cond(already_done, lambda: action, lambda: next_action)
        next_state = jax.lax.cond(already_done, lambda: state, lambda: next_state)

        next_carry = (next_action, next_state, already_done)
        output = (reward, state)
        return next_carry, output

    # Get the initial observation and action
    initial_obs = env.get_obs(initial_state)
    initial_action = policy(initial_obs, 0.0, initial_state.ego_state)

    # Transform and rollout!
    keys = jrandom.split(jrandom.PRNGKey(0), max_steps)
    t = jnp.linspace(0, 1, max_steps)  # index into the reference trajectory
    (_, final_state, _), (reward, state_traj) = jax.lax.scan(
        step, (initial_action, initial_state, False), (keys, t)
    )

    # Get the final observation
    final_obs = env.get_obs(final_state)

    # The potential is the negative of the (soft) minimum reward observed
    potential = -softmin(reward, sharpness=0.5)

    return SimulationResults(
        potential,
        initial_obs,
        final_obs,
        state_traj.ego_state,
        state_traj.non_ego_states,
    )


if __name__ == "__main__":
    seed = 0
    image_width = 16
    focal_length = 0.25
    T = 60

    # Make a PRNG key (#sorandom)
    prng_key = jrandom.PRNGKey(seed)

    # Make the environment to use
    image_shape = (image_width, image_width)
    env = make_highway_env(image_shape, focal_length)

    # Make the policy
    policy = SimpleDepthPolicy(
        trajectory=jnp.array(
            [
                [10.0, 0.0],
                [20.0, 0.0],
                [30.0, 0.0],
                [40.0, 0.0],
                [50.0, 0.0],
            ]
        ),
        key=prng_key,
        image_shape=image_shape,
    )
    dynamic_policy, static_policy = eqx.partition(policy, eqx.is_array)

    # Set up environment

    # The nominal non-ego behavior is to drive straight
    drive_straight = LinearTrajectory2D(
        p=jnp.array(
            [
                [10.0, 0.0],
                [20.0, 0.0],
                [30.0, 0.0],
                [40.0, 0.0],
                [50.0, 0.0],
            ]
        )
    )
    nominal_trajectory = MultiAgentTrajectoryLinear(
        trajectories=[drive_straight, drive_straight]
    )

    # Run the policy
    initial_state = env.reset(prng_key)  # TODO fix bad key management
    result = simulate(
        env, dynamic_policy, initial_state, nominal_trajectory, static_policy, T
    )

    # Plot the results
    fig = plt.figure(figsize=(32, 8), constrained_layout=True)
    axs = fig.subplot_mosaic(
        [
            ["trajectory", "trajectory", "trajectory", "trajectory", "trajectory"],
        ],
    )
    axs["trajectory"].axhline(7.5, linestyle="--", color="k")
    axs["trajectory"].axhline(-7.5, linestyle="--", color="k")
    axs["trajectory"].plot(
        result.ego_trajectory[:, 0].T,
        result.ego_trajectory[:, 1].T,
        linestyle="-",
        color="blue",
        label="Ego",
    )
    axs["trajectory"].plot(
        result.non_ego_trajectory[:, 0, 0],
        result.non_ego_trajectory[:, 0, 1],
        linestyle="-.",
        color="blue",
        label="Non-ego 1",
    )
    axs["trajectory"].plot(
        result.non_ego_trajectory[:, 1, 0],
        result.non_ego_trajectory[:, 1, 1],
        linestyle="--",
        color="blue",
        label="Non-ego 2",
    )

    axs["trajectory"].scatter(
        nominal_trajectory.trajectories[0].p[:, 0] + result.non_ego_trajectory[0, 0, 0],
        nominal_trajectory.trajectories[0].p[:, 1] + result.non_ego_trajectory[0, 0, 1],
        color="black",
        marker="o",
    )
    axs["trajectory"].scatter(
        nominal_trajectory.trajectories[1].p[:, 0] + result.non_ego_trajectory[0, 1, 0],
        nominal_trajectory.trajectories[1].p[:, 1] + result.non_ego_trajectory[0, 1, 1],
        color="black",
        marker="s",
    )
    axs["trajectory"].scatter(
        policy.trajectory.p[:, 0] + result.ego_trajectory[0, 0],
        policy.trajectory.p[:, 1] + result.ego_trajectory[0, 1],
        color="black",
        marker="s",
    )

    plt.show()
