"""Test a simple policy for the highway environment."""

import os

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import optax
import tqdm
from beartype.typing import NamedTuple, Tuple
from jaxtyping import Array, Float
from matplotlib.transforms import Affine2D

from radium.experiments.highway.predict_and_mitigate import (
    LinearTrajectory2D,
    MultiAgentTrajectoryLinear,
    dlqr,
)
from radium.systems.components.sensing.vision.render import CameraIntrinsics
from radium.systems.components.sensing.vision.shapes import Box
from radium.systems.highway.driving_policy import DrivingPolicy
from radium.systems.highway.highway_env import HighwayEnv, HighwayObs, HighwayState
from radium.systems.highway.highway_scene import Car, HighwayScene
from radium.utils import softmin

# get LQR gains
f1tenth_axle_length = 0.28
v_target = 0.5
A = np.array(
    [
        [1.0, 0.0, 0.0, 0.1],
        [0.0, 1.0, v_target * 0.1, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
B = np.array(
    [
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, v_target * 0.1 / f1tenth_axle_length],
        [0.1, 0.0],
    ]
)
Q = np.eye(4)
R = np.eye(2)
K, _, _ = dlqr(A, B, Q, R)
K = jnp.array(K)


def make_highway_env(image_shape: Tuple[int, int]):
    """Make the highway environment."""
    scene = HighwayScene(
        num_lanes=2, lane_width=0.6, segment_length=10.0
    )  # TODO match to f1tenth
    scene.car = Car(
        w_base=jnp.array(0.28),  # width at base of car
        w_top=jnp.array(0.23),  # width at top of car
        h_base=jnp.array(0.04),  # height to undecarriage
        h_chassis=jnp.array(0.1),  # height of chassis
        h_top=jnp.array(0.075),  # height of top of car
        l_hood=jnp.array(0.09),  # length of hood
        l_trunk=jnp.array(0.04),  # length of trunk
        l_cabin=jnp.array(0.3),  # length of cabin
        r_wheel=jnp.array(0.04),  # radius of wheel
        w_wheel=jnp.array(0.03),  # width of wheel
        rounding=jnp.array(0.01),
    )
    scene.walls = [
        Box(
            center=w.center,
            extent=w.extent,
            rotation=w.rotation,
            c=w.c,
            rounding=jnp.array(0.0),
        )
        for w in scene.walls
    ]

    intrinsics = CameraIntrinsics(
        focal_length=1.93e-3,
        sensor_size=(3.855e-3, 3.855e-3),
        resolution=image_shape,
    )
    initial_ego_state = jnp.array(
        [
            -5.5,
            -0.5,
            0.0,
            3.0 * v_target,
        ]
    )
    initial_non_ego_states = jnp.array(
        [
            [-3.5, -0.5, 0.0, v_target],
            [-1.5, 0.5, 0.0, v_target],
        ]
    )
    initial_state_covariance = jnp.diag(jnp.array([0.1, 0.1, 0.001, 0.1]) ** 2)

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

    # Update axle length to match f1tenth
    env._axle_length = f1tenth_axle_length

    return env


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

    def gaussian_kernel(self, shape, sigma=1.0):
        """Generates a 2D Gaussian kernel."""
        ay = jnp.arange(-shape[0] // 2, shape[0] // 2)
        ax = jnp.arange(-shape[1] // 2, shape[1] // 2)
        xx, yy = jnp.meshgrid(ax, ay)
        kernel = jnp.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        return kernel / jnp.sum(kernel)

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
        min_distance = 1.0
        depth_image = jnp.where(depth_image < 1e-3, min_distance, depth_image)
        kernel = self.gaussian_kernel(depth_image.shape, 1.0)
        mean_distance = jnp.sum(depth_image * kernel) / jnp.sum(kernel)
        vision_accel = 5 * jnp.clip(mean_distance - min_distance, -2.0, 0.0)

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
            state, action, non_ego_stable_action, key, collision_sharpness=20.0
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
    (_, final_state, done), (reward, state_traj) = jax.lax.scan(
        step, (initial_action, initial_state, False), (keys, t)
    )
    # Get the final observation
    final_obs = env.get_obs(final_state)

    # The potential is the negative of the (soft) minimum reward observed
    potential = -softmin(reward, sharpness=10.0)

    return SimulationResults(
        potential,
        initial_obs,
        final_obs,
        state_traj.ego_state,
        state_traj.non_ego_states,
    )


if __name__ == "__main__":
    np.set_printoptions(precision=3)

    seed = 0
    image_width = 16
    aspect = 4.0 / 3.0
    T = 100

    # Make a PRNG key (#sorandom)
    prng_key = jrandom.PRNGKey(seed)

    # Make the environment to use
    image_shape = (image_width, int(image_width / aspect))
    env = make_highway_env(image_shape)

    # Make the policy

    policy = SimpleDepthPolicy(
        trajectory=jnp.array(
            [
                [0.0, 0.0],
                [2.0, 0.3],
                [4.0, 0.6],
                [6.0, 0.6],
                [8.0, 0.0],
                [12.0, 0.0],
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
                [0.0, 0.0],
                [5.0, 0.0],
            ]
        )
    )
    nominal_trajectory = MultiAgentTrajectoryLinear(
        trajectories=[drive_straight, drive_straight]
    )

    # Define a cost function
    def cost_fn(policy):
        initial_state = env.reset(prng_key)  # TODO fix bad key management
        result = simulate(
            env, policy, initial_state, nominal_trajectory, static_policy, T
        )
        return result.potential

    # Optimize the policy
    steps = 100
    lr = 1e-3
    optimizer = optax.adam(learning_rate=lr)
    value_and_grad_fn = jax.jit(jax.value_and_grad(cost_fn))
    opt_state = optimizer.init(dynamic_policy)

    pbar = tqdm.tqdm(range(steps))
    for _ in pbar:
        # Compute the gradients
        cost, grads = value_and_grad_fn(dynamic_policy)
        pbar.set_description(f"Cost: {cost:.3f}")

        # Update the parameters
        updates, opt_state = optimizer.update(grads, opt_state)
        dynamic_policy = optax.apply_updates(dynamic_policy, updates)

    # Add the initial states to the trajectories so we can re-create them
    non_ego_traj = MultiAgentTrajectoryLinear(
        trajectories=[
            LinearTrajectory2D(
                p=nominal_trajectory.trajectories[0].p
                + env._initial_non_ego_states[0, :2]
            ),
            LinearTrajectory2D(
                p=nominal_trajectory.trajectories[1].p
                + env._initial_non_ego_states[1, :2]
            ),
        ]
    )
    policy = eqx.combine(dynamic_policy, static_policy)
    ego_traj = LinearTrajectory2D(p=policy.trajectory.p + env._initial_ego_state[:2])

    # Save the policy/trajectory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(current_dir, "base")
    os.makedirs(save_dir, exist_ok=True)
    mlp_policy_file = os.path.join(save_dir, "mlp.eqx")
    ego_traj_file = os.path.join(save_dir, "ego_traj.eqx")
    eqx.tree_serialise_leaves(mlp_policy_file, policy.actor_fcn)
    eqx.tree_serialise_leaves(ego_traj_file, ego_traj)
    for i, traj in enumerate(nominal_trajectory.trajectories):
        non_ego_traj_file = os.path.join(save_dir, f"non_ego_traj_{i}.eqx")
        eqx.tree_serialise_leaves(non_ego_traj_file, non_ego_traj.trajectories[i])

    # Run the policy
    initial_state = env.reset(prng_key)  # TODO fix bad key management
    result = simulate(
        env, dynamic_policy, initial_state, nominal_trajectory, static_policy, T
    )
    print(result.potential)

    # Plot the results
    fig = plt.figure(figsize=(32, 8), constrained_layout=True)
    axs = fig.subplot_mosaic(
        [
            ["trajectory", "trajectory", "trajectory", "trajectory", "trajectory"],
        ],
    )
    axs["trajectory"].axhline(
        env._highway_scene.walls[0].center[1], linestyle="--", color="k"
    )
    axs["trajectory"].axhline(
        env._highway_scene.walls[1].center[1], linestyle="--", color="k"
    )
    axs["trajectory"].plot(
        result.ego_trajectory[:, 0].T,
        result.ego_trajectory[:, 1].T,
        linestyle="-",
        color="red",
        label="Actual trajectory (Ego)",
    )
    axs["trajectory"].plot(
        result.non_ego_trajectory[:, 0, 0],
        result.non_ego_trajectory[:, 0, 1],
        linestyle="-",
        color="blue",
        label="Actual trajectory (Non-ego 1)",
    )
    axs["trajectory"].plot(
        result.non_ego_trajectory[:, 1, 0],
        result.non_ego_trajectory[:, 1, 1],
        linestyle="-",
        color="blue",
        label="Actual trajectory (Non-ego 2)",
    )

    # Plot the trajectories
    t = jnp.linspace(0, 1, 100)
    policy = eqx.combine(dynamic_policy, static_policy)
    ego_planned_trajectory = jax.vmap(policy.trajectory)(t)
    non_ego_planned_trajectory = jax.vmap(nominal_trajectory)(t)
    axs["trajectory"].plot(
        ego_planned_trajectory[:, 0] + result.ego_trajectory[0, 0],
        ego_planned_trajectory[:, 1] + result.ego_trajectory[0, 1],
        linestyle="--",
        color="red",
        label="Plan (Ego)",
    )
    axs["trajectory"].plot(
        non_ego_planned_trajectory[:, :, 0] + result.non_ego_trajectory[0, :, 0],
        non_ego_planned_trajectory[:, :, 1] + result.non_ego_trajectory[0, :, 1],
        linestyle="--",
        color="blue",
        label="Plan (Non-ego)",
    )

    # Draw a rectangular patch at the final car positions
    ego_car_pos = result.ego_trajectory[-1, :2]
    ego_car_heading = result.ego_trajectory[-1, 2]
    ego_car_width = env._highway_scene.car.width
    ego_car_length = env._highway_scene.car.length
    ego_car_patch = patches.Rectangle(
        (ego_car_pos[0] - ego_car_length / 2, ego_car_pos[1] - ego_car_width / 2),
        ego_car_length,
        ego_car_width,
        linewidth=1,
        edgecolor="r",
        facecolor="none",
    )
    t = (
        Affine2D().rotate_deg_around(
            ego_car_pos[0], ego_car_pos[1], ego_car_heading * 180 / np.pi
        )
        + axs["trajectory"].transData
    )
    ego_car_patch.set_transform(t)
    axs["trajectory"].add_patch(ego_car_patch)

    for i in [0, 1]:
        non_ego_car_pos = result.non_ego_trajectory[-1, i, :2]
        non_ego_car_heading = result.non_ego_trajectory[-1, i, 2]
        non_ego_car_width = env._highway_scene.car.width
        non_ego_car_length = env._highway_scene.car.length
        non_ego_car_patch = patches.Rectangle(
            (
                non_ego_car_pos[0] - non_ego_car_length / 2,
                non_ego_car_pos[1] - non_ego_car_width / 2,
            ),
            non_ego_car_length,
            non_ego_car_width,
            linewidth=1,
            edgecolor="b",
            facecolor="none",
        )
        t = (
            Affine2D().rotate_deg_around(
                non_ego_car_pos[0],
                non_ego_car_pos[1],
                non_ego_car_heading * 180 / np.pi,
            )
            + axs["trajectory"].transData
        )
        non_ego_car_patch.set_transform(t)
        axs["trajectory"].add_patch(non_ego_car_patch)

    axs["trajectory"].legend()
    axs["trajectory"].set_aspect("equal")

    plt.savefig("debug.png")
