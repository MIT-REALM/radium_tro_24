"""Code to predict and mitigate failure modes in the intersection scenario."""

import argparse
import json
import os
import shutil
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
from beartype.typing import NamedTuple
from jaxtyping import Array, Float, Shaped

import wandb
from architect.engines import predict_and_mitigate_failure_modes
from architect.engines.reinforce import init_sampler as init_reinforce_sampler
from architect.engines.reinforce import make_kernel as make_reinforce_kernel
from architect.engines.samplers import init_sampler as init_mcmc_sampler
from architect.engines.samplers import make_kernel as make_mcmc_kernel
from architect.experiments.highway.predict_and_mitigate import (
    LinearTrajectory2D,
    MultiAgentTrajectoryLinear,
    non_ego_trajectory_prior_logprob,
    sample_non_ego_trajectory,
)
from architect.experiments.intersection.train_intersection_agent_bc import (
    make_intersection_env,
)
from architect.systems.highway.highway_env import HighwayObs, HighwayState
from architect.systems.intersection.env import IntersectionEnv
from architect.systems.intersection.policy import DrivingPolicy
from architect.utils import softmin

# Type for non ego action trajectory
NonEgoActions = Float[Array, "T n_non_ego_agents num_actions"]


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

    eigVals, eigVecs = scipy.linalg.eig(A - B * K)

    return K, X, eigVals


# get LQR gains
v = 4.5
dt = 0.1
A = lambda theta: np.array(
    [
        [1.0, 0.0, -v * np.sin(theta) * dt, np.cos(theta) * dt],
        [0.0, 1.0, v * np.cos(theta) * dt, np.sin(theta) * dt],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
B = np.array(
    [
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 2 * 0.1 / 1.0],
        [0.1, 0.0],
    ]
)
Q = np.eye(4)
R = np.eye(2)
K_left, _, _ = dlqr(A(-np.pi / 2), B, Q, R)
K_left = jnp.array(K_left)
K_right, _, _ = dlqr(A(np.pi / 2), B, Q, R)
K_right = jnp.array(K_right)


class SimulationResults(NamedTuple):
    """A class for storing the results of a simulation."""

    potential: Float[Array, ""]
    initial_obs: HighwayObs
    final_obs: HighwayObs
    ego_trajectory: Float[Array, "T 4"]
    non_ego_trajectory: Float[Array, "T n_non_ego 4"]


def simulate(
    env: IntersectionEnv,
    policy: DrivingPolicy,
    initial_state: HighwayState,
    non_ego_reference_trajectory: MultiAgentTrajectoryLinear,
    static_policy: DrivingPolicy,
    max_steps: int = 60,
) -> Float[Array, ""]:
    """Simulate the intersection environment.

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

        # PRNG key management. These don't have any effect, but they need to be passed
        # to the environment and policy.
        step_subkey, action_subkey = jrandom.split(key)

        # Track the non-ego agent's reference trajectory
        compute_lqr_left = lambda non_ego_state, target_state: -K_left @ (
            non_ego_state - target_state
        )
        compute_lqr_right = lambda non_ego_state, target_state: -K_right @ (
            non_ego_state - target_state
        )
        compute_lqr = lambda non_ego_state, target_state: jax.lax.cond(
            non_ego_state[2] < 0.0,
            compute_lqr_left,
            compute_lqr_right,
            non_ego_state,
            target_state,
        )
        target = initial_state.non_ego_states  # copy initial heading, velocity, etc.
        # add the waypoint relative to the initial position
        target = target.at[:, :2].add(reference_waypoint)
        non_ego_stable_action = jax.vmap(compute_lqr)(state.non_ego_states, target)

        # Take a step in the environment using the action carried over from the previous
        # step.
        next_state, next_observation, reward, done = env.step(
            state,
            action,
            non_ego_stable_action,
            step_subkey,
            eval=True,
        )

        # Compute the action for the next step
        next_action, _, _ = policy(next_observation, action_subkey, deterministic=True)

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
    initial_action, _, _ = policy(initial_obs, jrandom.PRNGKey(0), deterministic=True)

    # Transform and rollout!
    keys = jrandom.split(jrandom.PRNGKey(0), max_steps)
    t = jnp.linspace(0, 1, max_steps)  # index into the reference trajectory
    (_, final_state, _), (reward, state_traj) = jax.lax.scan(
        step, (initial_action, initial_state, False), (keys, t)
    )

    # Get the final observation
    final_obs = env.get_obs(final_state)

    # The potential is the negative of the (soft) minimum reward observed
    # and penalize the agent for not crossing the road
    target_x = 20.0
    distance_to_target_x = jax.nn.elu(target_x - final_state.ego_state[0])
    potential = -softmin(reward, sharpness=0.5) + 0.1 * distance_to_target_x

    return SimulationResults(
        potential,
        initial_obs,
        final_obs,
        state_traj.ego_state,
        state_traj.non_ego_states,
    )


def plotting_cb(dp, eps):
    """Plot the results of the simulation with the given DP and all given EPs.

    Args:
        dp: The DP to plot.
        eps: The EPs to plot.
    """
    # Evaluate the solutions proposed by the prediction+mitigation algorithm
    result = eqx.filter_vmap(
        lambda dp, ep: simulate(env, dp, initial_state, ep, static_policy, T),
        in_axes=(None, 0),
    )(dp, eps)

    # Plot the results
    fig = plt.figure(figsize=(32, 16), constrained_layout=True)
    axs = fig.subplot_mosaic(
        [
            ["trajectory", "trajectory", "trajectory", "trajectory", "trajectory"],
            ["trajectory", "trajectory", "trajectory", "trajectory", "trajectory"],
            ["trajectory", "trajectory", "trajectory", "trajectory", "trajectory"],
            ["final_obs", "final_obs", "final_obs", "final_obs", "final_obs"],
            ["reward", "reward", "reward", "reward", "reward"],
        ],
    )

    # Plot the trajectories for each case, color-coded by potential
    max_potential = jnp.max(result.potential)
    min_potential = jnp.min(result.potential)
    normalized_potential = (result.potential - min_potential) / (
        max_potential - min_potential
    )
    axs["trajectory"].plot([-7.5 - 15, -7.5, -7.5], [7.5, 7.5, 7.5 + 15], "k--")
    axs["trajectory"].plot([-7.5 - 15, -7.5, -7.5], [-7.5, -7.5, -7.5 - 15], "k--")
    axs["trajectory"].plot([7.5 + 15, 7.5, 7.5], [7.5, 7.5, 7.5 + 15], "k--")
    axs["trajectory"].plot([7.5 + 15, 7.5, 7.5], [-7.5, -7.5, -7.5 - 15], "k--")
    axs["trajectory"].set_xlim([-7.5 - 15, 7.5 + 15])
    axs["trajectory"].set_ylim([-7.5 - 15, 7.5 + 15])
    for chain_idx in range(num_chains):
        axs["trajectory"].plot(
            result.ego_trajectory[chain_idx, :, 0].T,
            result.ego_trajectory[chain_idx, :, 1].T,
            linestyle="-",
            color=plt.cm.plasma(normalized_potential[chain_idx]),
            label="Ego" if chain_idx == 0 else None,
        )
        axs["trajectory"].plot(
            result.non_ego_trajectory[chain_idx, :, 0, 0],
            result.non_ego_trajectory[chain_idx, :, 0, 1],
            linestyle="-.",
            color=plt.cm.plasma(normalized_potential[chain_idx]),
            label="Non-ego 1" if chain_idx == 0 else None,
        )
        axs["trajectory"].scatter(
            eps.trajectories[0].p[chain_idx, :, 0]
            + result.non_ego_trajectory[chain_idx, 0, 0, 0],
            eps.trajectories[0].p[chain_idx, :, 1]
            + result.non_ego_trajectory[chain_idx, 0, 0, 1],
            color=plt.cm.plasma(normalized_potential[chain_idx]),
            marker="o",
        )
        axs["trajectory"].plot(
            result.non_ego_trajectory[chain_idx, :, 1, 0],
            result.non_ego_trajectory[chain_idx, :, 1, 1],
            linestyle="--",
            color=plt.cm.plasma(normalized_potential[chain_idx]),
            label="Non-ego 2" if chain_idx == 0 else None,
        )
        axs["trajectory"].scatter(
            eps.trajectories[1].p[chain_idx, :, 0]
            + result.non_ego_trajectory[chain_idx, 0, 1, 0],
            eps.trajectories[1].p[chain_idx, :, 1]
            + result.non_ego_trajectory[chain_idx, 0, 1, 1],
            color=plt.cm.plasma(normalized_potential[chain_idx]),
            marker="s",
        )
        axs["trajectory"].plot(
            result.non_ego_trajectory[chain_idx, :, 2, 0],
            result.non_ego_trajectory[chain_idx, :, 2, 1],
            linestyle="--",
            color=plt.cm.plasma(normalized_potential[chain_idx]),
            label="Non-ego 3" if chain_idx == 0 else None,
        )

        axs["trajectory"].scatter(
            eps.trajectories[2].p[chain_idx, :, 0]
            + result.non_ego_trajectory[chain_idx, 0, 2, 0],
            eps.trajectories[2].p[chain_idx, :, 1]
            + result.non_ego_trajectory[chain_idx, 0, 2, 1],
            color=plt.cm.plasma(normalized_potential[chain_idx]),
            marker="^",
        )
    axs["trajectory"].set_aspect("equal")

    # Plot the reward across all failure cases
    axs["reward"].plot(result.potential, "ko")
    axs["reward"].set_ylabel("Potential (negative min reward)")

    # Plot the final RGB observations tiled on the same subplot
    axs["final_obs"].imshow(
        jnp.concatenate(result.final_obs.color_image.transpose(0, 2, 1, 3), axis=1)
    )

    # log the figure to wandb
    wandb.log({"plot": fig}, commit=False)

    # Close the figure
    plt.close()


if __name__ == "__main__":
    matplotlib.use("Agg")
    # Set up arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--savename", type=str, default="intersection_lqr")
    parser.add_argument("--image_w", type=int, nargs="?", default=32)
    parser.add_argument("--image_h", type=int, nargs="?", default=32)
    parser.add_argument("--noise_scale", type=float, nargs="?", default=1.0)
    parser.add_argument("--failure_level", type=float, nargs="?", default=11.0)
    parser.add_argument("--T", type=int, nargs="?", default=100)
    parser.add_argument("--seed", type=int, nargs="?", default=0)
    parser.add_argument("--L", type=float, nargs="?", default=1.0)
    parser.add_argument("--dp_logprior_scale", type=float, nargs="?", default=1.0)
    parser.add_argument("--dp_mcmc_step_size", type=float, nargs="?", default=1e-3)
    parser.add_argument("--ep_mcmc_step_size", type=float, nargs="?", default=1e-3)
    parser.add_argument("--num_rounds", type=int, nargs="?", default=5)
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
    parser.add_argument("--predict", action=boolean_action, default=True)
    parser.add_argument("--temper", action=boolean_action, default=True)
    parser.add_argument("--grad_clip", type=float, nargs="?", default=float("inf"))
    parser.add_argument("--dont_normalize_gradients", action="store_true")
    args = parser.parse_args()

    # Hyperparameters
    L = args.L
    noise_scale = args.noise_scale
    failure_level = args.failure_level
    T = args.T
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
            "noise_scale": noise_scale,
            "failure_level": failure_level,
            "T": T,
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
    image_shape = (args.image_h, args.image_w)
    env = make_intersection_env(image_shape)
    env._collision_penalty = 10.0
    # if repair:
    #     T = T // 2
    #     env._substeps = 2  # speed up sim for repair

    # Load the model (key doesn't matter; we'll replace all leaves with the saved
    # parameters), duplicating the model for each chain. We'll also split partition
    # out just the continuous parameters, which will be our design parameters
    dummy_policy = DrivingPolicy(jrandom.PRNGKey(0), image_shape)
    load_policy = lambda _: eqx.tree_deserialise_leaves(args.model_path, dummy_policy)
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
    # Hack: reset covariances to remove variation in initial state
    env._initial_state_covariance = 1e-3 * env._initial_state_covariance
    env._shading_light_direction_covariance = (
        1e-3 * env._shading_light_direction_covariance
    )
    prng_key, initial_state_key = jrandom.split(prng_key)
    initial_state = env.reset(initial_state_key)

    # The nominal non-ego behavior is to drive straight
    drive_straight_left2right = LinearTrajectory2D(
        p=jnp.array(
            [
                [0.0, -10.0],
                [0.0, -20.0],
                [0.0, -30.0],
                [0.0, -40.0],
                [0.0, -50.0],
            ]
        )
    )
    drive_straight_right2left = LinearTrajectory2D(
        p=jnp.array(
            [
                [0.0, 10.0],
                [0.0, 20.0],
                [0.0, 30.0],
                [0.0, 40.0],
                [0.0, 50.0],
            ]
        )
    )
    nominal_trajectory = MultiAgentTrajectoryLinear(
        trajectories=[
            drive_straight_left2right,
            drive_straight_left2right,
            drive_straight_right2left,
        ]
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
        ep_potential_fn=lambda dp, ep: -L
        * jax.nn.elu(
            failure_level
            - simulate(env, dp, initial_state, ep, static_policy, T).potential
        ),
        dp_potential_fn=lambda dp, ep: -L
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

    # Evaluate this single policy against all failures
    final_eps = jtu.tree_map(lambda leaf: leaf[-1], eps)

    # # TODO for debugging plots, just use the initial policy and eps
    # t_end = 0.0
    # t_start = 0.0
    # final_dps = jtu.tree_map(lambda leaf: leaf[-1], initial_dps)
    # final_eps = initial_eps
    # dp_logprobs = jnp.zeros((num_rounds, num_chains))
    # ep_logprobs = jnp.zeros((num_rounds, num_chains))
    # # TODO debugging bit ends here

    # Figure out where to save
    save_dir = (
        f"results/{args.savename}/{'predict' if predict else ''}"
        f"{'_' if repair else ''}{'repair_' + str(dp_logprior_scale) if repair else ''}/"
        f"noise_{noise_scale:0.1e}/"
        f"L_{L:0.1e}/"
        f"{num_rounds * num_steps_per_round}_samples/"
        f"{num_chains}_chains/"
        f"{quench_rounds}_quench/"
        f"dp_{dp_mcmc_step_size:0.1e}/"
        f"ep_{ep_mcmc_step_size:0.1e}/"
    )
    filename = save_dir + f"{alg_type}{'_20tempered+0.1' if temper else ''}"
    filename += f"{'_nogradnorm' if not normalize_gradients else ''}"
    filename += f"_{seed}"
    print(f"Saving results to: {filename}")
    os.makedirs(save_dir, exist_ok=True)

    # Save the final design parameters (joined back into the full policy)
    final_policy = eqx.combine(final_dps, static_policy)
    eqx.tree_serialise_leaves(filename + ".eqx", final_policy)

    # # Save the trace of policies
    # eqx.tree_serialise_leaves(filename + "_trace.eqx", dps)

    # Save the initial policy
    try:
        shutil.copyfile(
            args.model_path, f"results/{args.savename}/" + "initial_policy.eqx"
        )
    except shutil.SameFileError:
        pass

    # Save the hyperparameters
    with open(filename + ".json", "w") as f:
        json.dump(
            {
                "initial_state": initial_state._asdict(),
                "action_trajectory": final_eps,
                "action_trajectory_trace": eps,
                "time": t_end - t_start,
                "L": L,
                "noise_scale": noise_scale,
                "failure_level": failure_level,
                "image_w": args.image_w,
                "image_h": args.image_h,
                "dp_logprior_scale": dp_logprior_scale,
                "dp_mcmc_step_size": dp_mcmc_step_size,
                "ep_mcmc_step_size": ep_mcmc_step_size,
                "num_rounds": num_rounds,
                "num_steps_per_round": num_steps_per_round,
                "num_chains": num_chains,
                "use_gradients": use_gradients,
                "use_stochasticity": use_stochasticity,
                "repair": repair,
                "predict": predict,
                "quench_rounds": quench_rounds,
                "tempering_schedule": tempering_schedule,
                "ep_logprobs": ep_logprobs,
                "dp_logprobs": dp_logprobs,
            },
            f,
            default=lambda x: x.tolist() if isinstance(x, Shaped[Array, "..."]) else x,
        )
