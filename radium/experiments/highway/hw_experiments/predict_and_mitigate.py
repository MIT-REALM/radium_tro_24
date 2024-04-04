"""Code to predict and mitigate failure modes in the highway scenario."""

import argparse
import os
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from beartype.typing import NamedTuple
from jaxtyping import Array, Float

from beartype.typing import Tuple

import wandb
from radium.engines import predict_and_mitigate_failure_modes
from radium.engines.blackjax import make_hmc_step_and_initial_state
from radium.engines.reinforce import init_sampler as init_reinforce_sampler
from radium.engines.reinforce import make_kernel as make_reinforce_kernel
from radium.engines.samplers import init_sampler as init_mcmc_sampler
from radium.engines.samplers import make_kernel as make_mcmc_kernel
from radium.systems.highway.driving_policy import DrivingPolicy
from radium.systems.highway.highway_env import HighwayEnv, HighwayObs, HighwayState
from radium.utils import softmin

from radium.experiments.highway.predict_and_mitigate import (
    LinearTrajectory2D,
    MultiAgentTrajectoryLinear,
    sample_non_ego_trajectory,
    dlqr,
    non_ego_trajectory_prior_logprob,
)
from radium.systems.components.sensing.vision.render import CameraIntrinsics
from radium.systems.components.sensing.vision.shapes import Box
from radium.systems.highway.highway_scene import Car, HighwayScene

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


def plotting_cb(dp, eps, T=60):
    """Plot the results of the simulation with the given DP and all given EPs.

    Args:
        dp: The DP to plot.
        eps: The EPs to plot.
    """
    # initial_state = env.reset(prng_key)  # TODO fix bad key management
    result = eqx.filter_vmap(
        lambda dp, ep: simulate(env, dp, initial_state, ep, static_policy, T),
        in_axes=(None, 0),
    )(dp, eps)

    max_potential = jnp.max(result.potential)
    min_potential = jnp.min(result.potential) - 1e-3
    normalized_potential = (result.potential - min_potential) / (
        max_potential - min_potential
    )

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
    for chain_idx in range(num_chains):
        axs["trajectory"].plot(
            result.ego_trajectory[chain_idx, :, 0].T,
            result.ego_trajectory[chain_idx, :, 1].T,
            linestyle="-",
            color=plt.cm.plasma(normalized_potential[chain_idx]),
            label=f"potential: {result.potential[chain_idx]}",
        )
        axs["trajectory"].plot(
            result.non_ego_trajectory[chain_idx, :, 0, 0],
            result.non_ego_trajectory[chain_idx, :, 0, 1],
            linestyle="-",
            color=plt.cm.plasma(normalized_potential[chain_idx]),
        )
        axs["trajectory"].plot(
            result.non_ego_trajectory[chain_idx, :, 1, 0],
            result.non_ego_trajectory[chain_idx, :, 1, 1],
            linestyle="-",
            color=plt.cm.plasma(normalized_potential[chain_idx]),
        )

        # Plot the trajectories
        axs["trajectory"].scatter(
            dp.trajectory.p[:, 0] + result.ego_trajectory[chain_idx, 0, 0],
            dp.trajectory.p[:, 1] + result.ego_trajectory[chain_idx, 0, 1],
            color=plt.cm.plasma(normalized_potential[chain_idx]),
            marker="o",
        )
        axs["trajectory"].scatter(
            eps.trajectories[0].p[chain_idx, :, 0]
            + result.non_ego_trajectory[chain_idx, 0, 0, 0],
            eps.trajectories[0].p[chain_idx, :, 1]
            + result.non_ego_trajectory[chain_idx, 0, 0, 1],
            color=plt.cm.plasma(normalized_potential[chain_idx]),
            marker="s",
        )
        axs["trajectory"].scatter(
            eps.trajectories[1].p[chain_idx, :, 0]
            + result.non_ego_trajectory[chain_idx, 0, 1, 0],
            eps.trajectories[1].p[chain_idx, :, 1]
            + result.non_ego_trajectory[chain_idx, 0, 1, 1],
            color=plt.cm.plasma(normalized_potential[chain_idx]),
            marker="s",
        )

    axs["trajectory"].legend()
    axs["trajectory"].set_aspect("equal")

    # log the figure to wandb
    wandb.log({"plot": wandb.Image(fig)}, commit=False)

    # Close the figure
    plt.close()


if __name__ == "__main__":
    matplotlib.use("Agg")

    # Set up arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlp_path", type=str, required=True)
    parser.add_argument("--ego_traj_path", type=str, required=True)
    parser.add_argument("--savename", type=str, default="highway_hw")
    parser.add_argument("--failure_level", type=float, nargs="?", default=1.8)
    parser.add_argument("--noise_scale", type=float, nargs="?", default=0.05)
    parser.add_argument("--T", type=int, nargs="?", default=60)
    parser.add_argument("--seed", type=int, nargs="?", default=0)
    parser.add_argument("--L", type=float, nargs="?", default=1.0)
    parser.add_argument("--dp_logprior_scale", type=float, nargs="?", default=1.0)
    parser.add_argument("--dp_mcmc_step_size", type=float, nargs="?", default=1e-3)
    parser.add_argument("--ep_mcmc_step_size", type=float, nargs="?", default=1e-3)
    parser.add_argument("--num_rounds", type=int, nargs="?", default=10)
    parser.add_argument("--num_steps_per_round", type=int, nargs="?", default=10)
    parser.add_argument("--num_chains", type=int, nargs="?", default=10)
    parser.add_argument("--quench_rounds", type=int, nargs="?", default=0)
    parser.add_argument("--disable_gradients", action="store_true")
    parser.add_argument("--disable_stochasticity", action="store_true")
    parser.add_argument("--disable_mh", action="store_true")
    parser.add_argument("--reinforce", action="store_true")
    parser.add_argument("--zero_order_gradients", action="store_true")
    parser.add_argument("--num_stress_test_cases", type=int, nargs="?", default=1_000)
    boolean_action = argparse.BooleanOptionalAction
    parser.add_argument("--repair", action=boolean_action, default=False)
    parser.add_argument("--hmc", action=boolean_action, default=False)
    parser.add_argument("--num_hmc_integration_steps", type=int, nargs="?", default=10)
    parser.add_argument("--predict", action=boolean_action, default=True)
    parser.add_argument("--temper", action=boolean_action, default=True)
    parser.add_argument("--grad_clip", type=float, nargs="?", default=1.0)
    parser.add_argument("--dont_normalize_gradients", action="store_true")
    args = parser.parse_args()

    # Hyperparameters
    L = args.L
    T = args.T
    failure_level = args.failure_level
    noise_scale = args.noise_scale
    seed = args.seed
    dp_logprior_scale = args.dp_logprior_scale
    dp_mcmc_step_size = args.dp_mcmc_step_size
    ep_mcmc_step_size = args.ep_mcmc_step_size
    num_rounds = args.num_rounds
    num_steps_per_round = args.num_steps_per_round
    num_chains = args.num_chains
    use_gradients = not args.disable_gradients
    use_stochasticity = not args.disable_stochasticity
    use_mh = not args.disable_mh
    reinforce = args.reinforce
    hmc = args.hmc
    num_hmc_integration_steps = args.num_hmc_integration_steps
    zero_order_gradients = args.zero_order_gradients
    num_stress_test_cases = args.num_stress_test_cases
    repair = args.repair
    predict = args.predict
    temper = args.temper
    quench_rounds = args.quench_rounds
    grad_clip = args.grad_clip
    normalize_gradients = not args.dont_normalize_gradients

    quench_dps_only = False
    if reinforce:
        alg_type = "reinforce_l2c_0.05_step"
    elif hmc:
        alg_type = f"hmc_steps_{num_hmc_integration_steps}"
    elif use_gradients and use_stochasticity and use_mh and not zero_order_gradients:
        alg_type = "mala"
        quench_dps_only = True
    elif use_gradients and use_stochasticity and use_mh and zero_order_gradients:
        alg_type = "mala_zo"
        quench_dps_only = True
    elif use_gradients and use_stochasticity and not use_mh:
        alg_type = "ula"
    elif use_gradients and not use_stochasticity:
        alg_type = "gd"
    elif not use_gradients and use_stochasticity and use_mh:
        alg_type = "rmh"
    elif not use_gradients and use_stochasticity and not use_mh:
        alg_type = "random_walk"
    else:
        alg_type = "static"

    # Initialize logger
    wandb.init(
        project=args.savename,
        group=alg_type
        + ("-predict" if predict else "")
        + ("-repair" if repair else ""),
        config={
            "L": L,
            "failure_level": failure_level,
            "T": T,
            "noise_scale": noise_scale,
            "seed": seed,
            "dp_logprior_scale": dp_logprior_scale,
            "dp_mcmc_step_size": dp_mcmc_step_size,
            "ep_mcmc_step_size": ep_mcmc_step_size,
            "num_rounds": num_rounds,
            "num_steps_per_round": num_steps_per_round,
            "num_chains": num_chains,
            "use_gradients": use_gradients,
            "use_stochasticity": use_stochasticity,
            "use_mh": use_mh,
            "reinforce": reinforce,
            "hmc": hmc,
            "num_hmc_integration_steps": num_hmc_integration_steps,
            "zero_order_gradients": zero_order_gradients,
            "repair": repair,
            "predict": predict,
            "temper": temper,
            "quench_rounds": quench_rounds,
            "grad_clip": grad_clip,
            "normalize_gradients": normalize_gradients,
            "num_stress_test_cases": num_stress_test_cases,
            "quench_dps_only": quench_dps_only,
        },
    )

    # Add exponential tempering if using
    t = jnp.linspace(0, 1, num_rounds) + 0.1
    tempering_schedule = 1 - jnp.exp(-20 * t) if temper else None

    # Make a PRNG key (#sorandom)
    prng_key = jrandom.PRNGKey(seed)

    # Make the environment to use
    image_width = 16
    aspect = 4.0 / 3.0
    image_shape = (image_width, int(image_width / aspect))
    env = make_highway_env(image_shape)

    # Load the model (key doesn't matter; we'll replace all leaves with the saved
    # parameters), duplicating the model for each chain. We'll also split partition
    # out just the continuous parameters, which will be our design parameters
    dummy_policy = SimpleDepthPolicy(
        trajectory=jnp.zeros((6, 2)),
        key=prng_key,
        image_shape=image_shape,
    )

    def load_policy(_):
        trajectory = eqx.tree_deserialise_leaves(
            args.ego_traj_path, dummy_policy.trajectory
        )
        mlp = eqx.tree_deserialise_leaves(args.mlp_path, dummy_policy.actor_fcn)
        policy = SimpleDepthPolicy(
            trajectory.p - env._initial_ego_state[:2], prng_key, image_shape
        )
        policy = eqx.tree_at(lambda pi: pi.actor_fcn, policy, mlp)
        return policy

    get_dps = lambda _: eqx.partition(load_policy(_), eqx.is_array)[0]
    initial_dps = eqx.filter_vmap(get_dps)(jnp.arange(num_chains))
    # Also save out the static part of the policy
    initial_dp, static_policy = eqx.partition(load_policy(None), eqx.is_array)

    # Make a prior logprob for the policy that penalizes large updates to the policy
    # parameters
    def dp_prior_logprob(dp):
        block_logprobs = jtu.tree_map(
            lambda x_updated, x: jax.scipy.stats.norm.logpdf(
                x_updated - x, scale=dp_logprior_scale
            ).mean(),
            dp,
            initial_dp,
        )
        # Take a block mean rather than the sum of all blocks to avoid a crazy large
        # logprob
        overall_logprob = jax.flatten_util.ravel_pytree(block_logprobs)[0].mean()
        return overall_logprob

    # Initialize some fixed initial states
    # prng_key, initial_state_key = jrandom.split(prng_key)  # bad key management
    initial_state = env.reset(prng_key)

    # The nominal non-ego behavior is to drive straight
    drive_straight = LinearTrajectory2D(
        p=jnp.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [3.0, 0.0],
                [4.0, 0.0],
                [5.0, 0.0],
            ]
        )
    )
    nominal_trajectory = MultiAgentTrajectoryLinear(
        trajectories=[drive_straight, drive_straight]
    )

    # Initialize some random non-ego action trajectories as exogenous parameters
    prng_key, ep_key = jrandom.split(prng_key)
    ep_keys = jrandom.split(ep_key, num_chains)
    initial_eps = jax.vmap(
        lambda key: sample_non_ego_trajectory(key, nominal_trajectory, noise_scale)
    )(ep_keys)

    # Also initialize a bunch of exogenous parameters to serve as stress test cases
    # for the policy
    prng_key, ep_key = jrandom.split(prng_key)
    ep_keys = jrandom.split(ep_key, num_stress_test_cases)
    stress_test_eps = jax.vmap(
        lambda key: sample_non_ego_trajectory(key, nominal_trajectory, noise_scale)
    )(ep_keys)

    # Choose which sampler to use
    if reinforce:
        init_sampler_fn = init_reinforce_sampler
        make_kernel_fn = lambda _1, logprob_fn, step_size, _2: make_reinforce_kernel(
            logprob_fn,
            step_size,
            perturbation_stddev=noise_scale,
            baseline_update_rate=0.5,
        )
    elif hmc:
        init_sampler_fn = lambda params, logprob_fn: make_hmc_step_and_initial_state(
            logprob_fn,
            params,
            step_size=ep_mcmc_step_size,  # dummy
            num_integration_steps=num_hmc_integration_steps,
        )[1]

        make_kernel_fn = (
            lambda params, logprob_fn, step_size, _: make_hmc_step_and_initial_state(
                logprob_fn,
                params,
                step_size=step_size,
                num_integration_steps=num_hmc_integration_steps,
            )[0]
        )
    else:
        # This sampler yields either MALA, GD, or RMH depending on whether gradients
        # and/or stochasticity are enabled
        init_sampler_fn = lambda params, logprob_fn: init_mcmc_sampler(
            params,
            logprob_fn,
            normalize_gradients,
            gradient_clip=grad_clip,
            estimate_gradients=zero_order_gradients,
        )
        make_kernel_fn = (
            lambda _, logprob_fn, step_size, stochasticity: make_mcmc_kernel(
                logprob_fn,
                step_size,
                use_gradients,
                stochasticity,
                grad_clip,
                normalize_gradients,
                use_mh,
                zero_order_gradients,
            )
        )

    # Adjust scaling based on the dimension of the dps and eps
    dp_dimensions = jax.flatten_util.ravel_pytree(initial_dps)[0].shape[0] / num_chains
    ep_dimensions = jax.flatten_util.ravel_pytree(initial_eps)[0].shape[0] / num_chains
    print(f"dp_dimensions: {dp_dimensions}")
    print(f"ep_dimensions: {ep_dimensions}")
    # L_dp = L * dp_dimensions
    # L_ep = L * ep_dimensions
    L_dp = 10 * L
    L_ep = L

    # Run the prediction+mitigation process
    t_start = time.perf_counter()
    dps, eps, dp_logprobs, ep_logprobs = predict_and_mitigate_failure_modes(
        prng_key,
        initial_dps,
        initial_eps,
        dp_logprior_fn=dp_prior_logprob,
        ep_logprior_fn=lambda ep: non_ego_trajectory_prior_logprob(
            ep, nominal_trajectory, noise_scale
        ),
        ep_potential_fn=lambda dp, ep: -L_ep
        * jax.nn.elu(
            failure_level
            - simulate(env, dp, initial_state, ep, static_policy, T).potential
        ),
        dp_potential_fn=lambda dp, ep: -L_dp
        * jax.nn.elu(
            simulate(env, dp, initial_state, ep, static_policy, T).potential
            - failure_level
        ),
        init_sampler=init_sampler_fn,
        make_kernel=make_kernel_fn,
        num_rounds=num_rounds,
        num_mcmc_steps_per_round=num_steps_per_round,
        dp_mcmc_step_size=dp_mcmc_step_size,
        ep_mcmc_step_size=ep_mcmc_step_size,
        use_stochasticity=use_stochasticity,
        repair=repair,
        predict=predict,
        quench_rounds=quench_rounds,
        quench_dps_only=quench_dps_only,
        tempering_schedule=tempering_schedule,
        logging_prefix=f"{args.savename}/{alg_type}[{os.getpid()}]",
        stress_test_cases=stress_test_eps,
        potential_fn=lambda dp, ep: simulate(
            env, dp, initial_state, ep, static_policy, T
        ).potential,
        failure_level=failure_level,
        plotting_cb=plotting_cb,
        test_every=5,
    )
    t_end = time.perf_counter()
    print(
        f"Ran {num_rounds:,} rounds with {num_chains} chains in {t_end - t_start:.2f} s"
    )

    # Select the policy that performs best against all predicted failures before
    # the final round (choose from all chains)
    if repair:
        most_likely_dps_idx = jnp.argmax(dp_logprobs[-1], axis=-1)
        final_dps = jtu.tree_map(lambda leaf: leaf[-1, most_likely_dps_idx], dps)
    else:
        # Just pick one policy arbitrarily if we didn't optimize the policies.
        final_dps = jtu.tree_map(lambda leaf: leaf[-1, 0], dps)

    # Pick the best adversarial trajectory from the final round
    if predict:
        most_likely_eps_idx = jnp.argmax(ep_logprobs[-1], axis=-1)
        final_eps = jtu.tree_map(lambda leaf: leaf[-1, most_likely_eps_idx], eps)
    else:
        # Just pick one trajectory arbitrarily if we didn't optimize the trajectories.
        final_eps = jtu.tree_map(lambda leaf: leaf[-1, 0], eps)

    # Add the initial states to the trajectories so we can re-create them
    non_ego_traj = MultiAgentTrajectoryLinear(
        trajectories=[
            LinearTrajectory2D(
                p=final_eps.trajectories[0].p + env._initial_non_ego_states[0, :2]
            ),
            LinearTrajectory2D(
                p=final_eps.trajectories[1].p + env._initial_non_ego_states[1, :2]
            ),
        ]
    )
    policy = eqx.combine(final_dps, static_policy)
    ego_traj = LinearTrajectory2D(p=policy.trajectory.p + env._initial_ego_state[:2])

    # Figure out where to save
    current_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(
        current_dir,
        ("predict" if predict else "") + ("repair" if repair else ""),
    )
    os.makedirs(save_dir, exist_ok=True)

    mlp_policy_file = os.path.join(save_dir, "mlp.eqx")
    ego_traj_file = os.path.join(save_dir, "ego_traj.eqx")
    eqx.tree_serialise_leaves(mlp_policy_file, policy.actor_fcn)
    eqx.tree_serialise_leaves(ego_traj_file, ego_traj)
    for i, traj in enumerate(final_eps.trajectories):
        non_ego_traj_file = os.path.join(save_dir, f"non_ego_traj_{i}.eqx")
        eqx.tree_serialise_leaves(non_ego_traj_file, non_ego_traj.trajectories[i])
