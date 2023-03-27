"""Code to predict and mitigate failure modes in the highway scenario."""
import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
from jax.nn import sigmoid
import equinox as eqx
import matplotlib.pyplot as plt
import seaborn as sns
from jaxtyping import Float, Array, Shaped
from beartype.typing import NamedTuple

from architect.engines import predict_and_mitigate_failure_modes

from architect.systems.highway.driving_policy import DrivingPolicy
from architect.systems.highway.highway_env import HighwayState, HighwayEnv, HighwayObs
from architect.experiments.highway.train_highway_agent import make_highway_env


class SimulationResults(NamedTuple):
    """A class for storing the results of a simulation."""

    reward: Float[Array, ""]
    initial_obs: HighwayObs
    final_obs: HighwayObs


def simulate(
    env: HighwayEnv,
    policy: DrivingPolicy,
    initial_state: HighwayState,
    max_steps: int = 8,  # 0,
) -> Float[Array, ""]:
    """Simulate the highway environment.

    Disables randomness in the policy and environment (all randomness should be
    factored out into the initial_state argument).

    If the environment terminates before `max_steps` steps, it will not be reset and
    all further reward will be zero.

    Args:
        env: The environment to simulate.
        policy: The policy to use to drive the car.
        initial_state: The initial state of the environment.
        max_steps: The maximum number of steps to simulate.

    Returns:
        SimulationResults object
    """
    # Make sure the policy is deterministic
    policy = eqx.tree_at(
        lambda policy: policy.log_action_std, policy, jnp.array(-jnp.inf)
    )

    def step(carry, key):
        # Unpack the carry
        obs, state, already_done = carry

        # PRNG key management. These don't have any effect, but they need to be passed
        # to the environment and policy.
        step_key, action_subkey = jrandom.split(key)

        # Sample an action from the policy
        action, action_logprob, value = policy(obs, action_subkey)

        # Fix non-ego actions
        non_ego_actions = jnp.zeros((state.non_ego_states.shape[0], 2))

        # Take a step in the environment using that action
        next_state, next_observation, reward, done = env.step(
            state, action, non_ego_actions, step_key
        )

        # If the environment has already terminated, set the reward to zero.
        reward = jax.lax.cond(already_done, lambda: 0.0, lambda: reward)
        already_done = jnp.logical_or(already_done, done)

        # Don't step if the environment has terminated
        next_observation = jax.lax.cond(
            already_done, lambda: obs, lambda: next_observation
        )
        next_state = jax.lax.cond(already_done, lambda: state, lambda: next_state)

        next_carry = (next_observation, next_state, already_done)
        output = (reward, already_done)
        return next_carry, output

    # Get the initial observation
    initial_obs = env.get_obs(initial_state)

    # Transform and rollout!
    keys = jrandom.split(jrandom.PRNGKey(0), max_steps)
    (final_obs, _, _), (reward, already_done) = jax.lax.scan(
        step, (initial_obs, initial_state, False), keys
    )

    return SimulationResults(jnp.mean(reward), initial_obs, final_obs)


if __name__ == "__main__":
    # Set up arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--savename", type=str, default="overtake")
    parser.add_argument("--image_w", type=int, nargs="?", default=64)
    parser.add_argument("--image_h", type=int, nargs="?", default=64)
    parser.add_argument("--L", type=float, nargs="?", default=100.0)
    parser.add_argument("--dp_mcmc_step_size", type=float, nargs="?", default=1e-6)
    parser.add_argument("--ep_mcmc_step_size", type=float, nargs="?", default=1e-2)
    parser.add_argument("--num_rounds", type=int, nargs="?", default=100)
    parser.add_argument("--num_mcmc_steps_per_round", type=int, nargs="?", default=10)
    parser.add_argument("--num_chains", type=int, nargs="?", default=10)
    parser.add_argument("--quench_rounds", type=int, nargs="?", default=10)
    parser.add_argument("--disable_gradients", action="store_true")
    parser.add_argument("--disable_stochasticity", action="store_true")
    boolean_action = argparse.BooleanOptionalAction
    parser.add_argument("--repair", action=boolean_action, default=True)
    parser.add_argument("--predict", action=boolean_action, default=True)
    parser.add_argument("--temper", action=boolean_action, default=False)
    parser.add_argument("--dp_grad_clip", type=float, nargs="?", default=float("inf"))
    parser.add_argument("--ep_grad_clip", type=float, nargs="?", default=float("inf"))
    args = parser.parse_args()

    # Hyperparameters
    L = args.L
    dp_mcmc_step_size = args.dp_mcmc_step_size
    ep_mcmc_step_size = args.ep_mcmc_step_size
    num_rounds = args.num_rounds
    num_mcmc_steps_per_round = args.num_mcmc_steps_per_round
    num_chains = args.num_chains
    use_gradients = not args.disable_gradients
    use_stochasticity = not args.disable_stochasticity
    repair = args.repair
    predict = args.predict
    temper = args.temper
    quench_rounds = args.quench_rounds
    dp_grad_clip = args.dp_grad_clip
    ep_grad_clip = args.ep_grad_clip

    print(f"Running prediction/mitigation on overtake with hyperparameters:")
    print(f"\tmodel_path = {args.model_path}")
    print(f"\timage dimensions (w x h) = {args.image_w} x {args.image_h}")
    print(f"\tL = {L}")
    print(f"\tdp_mcmc_step_size = {dp_mcmc_step_size}")
    print(f"\tep_mcmc_step_size = {ep_mcmc_step_size}")
    print(f"\tnum_rounds = {num_rounds}")
    print(f"\tnum_mcmc_steps_per_round = {num_mcmc_steps_per_round}")
    print(f"\tnum_chains = {num_chains}")
    print(f"\tuse_gradients = {use_gradients}")
    print(f"\tuse_stochasticity = {use_stochasticity}")
    print(f"\trepair = {repair}")
    print(f"\tpredict = {predict}")
    print(f"\ttemper = {temper}")
    print(f"\tquench_rounds = {quench_rounds}")
    print(f"\tdp_grad_clip = {dp_grad_clip}")
    print(f"\tep_grad_clip = {ep_grad_clip}")

    # Add exponential tempering if using
    t = jnp.linspace(0, 1, num_rounds)
    tempering_schedule = 1 - jnp.exp(-5 * t) if temper else None

    # Make a PRNG key (#sorandom)
    prng_key = jrandom.PRNGKey(0)

    # Make the environment to use
    image_shape = (args.image_h, args.image_w)
    env = make_highway_env(image_shape)

    # Load the model (key doesn't matter; we'll replace all leaves with the saved
    # parameters), duplicating the model for each chain
    dummy_policy = DrivingPolicy(jrandom.PRNGKey(0), image_shape)
    load_policy = lambda _: eqx.tree_deserialise_leaves(args.model_path, dummy_policy)
    policy = eqx.filter_vmap(load_policy)(jnp.arange(num_chains))

    # Initialize some random initial states
    prng_key, initial_state_key = jrandom.split(prng_key)
    initial_state_keys = jrandom.split(initial_state_key, num_chains)
    initial_states = eqx.filter_vmap(env.reset)(initial_state_keys)

    # # Run the prediction+mitigation process
    # t_start = time.perf_counter()
    # dps, eps, dp_logprobs, ep_logprobs = predict_and_mitigate_failure_modes(
    #     prng_key,
    #     policy,
    #     initial_states,
    #     dp_logprior_fn=lambda dp: jnp.array(0.0),  # uniform prior over policies
    #     ep_logprior_fn=env.overall_prior_logprob,
    #     potential_fn=lambda dp, ep: L * simulate(env, dp, ep).reward,
    #     num_rounds=num_rounds,
    #     num_mcmc_steps_per_round=num_mcmc_steps_per_round,
    #     dp_mcmc_step_size=dp_mcmc_step_size,
    #     ep_mcmc_step_size=ep_mcmc_step_size,
    #     use_gradients=use_gradients,
    #     use_stochasticity=use_stochasticity,
    #     repair=repair,
    #     predict=predict,
    #     quench_rounds=quench_rounds,
    #     tempering_schedule=tempering_schedule,
    #     dp_grad_clip=dp_grad_clip,
    #     ep_grad_clip=ep_grad_clip,
    # )
    # t_end = time.perf_counter()
    # print(
    #     f"Ran {num_rounds:,} rounds with {num_chains} chains in {t_end - t_start:.2f} s"
    # )

    # # Select the policy that performs best against all predicted failures before
    # # the final round (choose from all chains)
    # if repair:
    #     most_likely_dps_idx = jnp.argmax(dp_logprobs[-1], axis=-1)
    #     final_dps = jtu.tree_map(lambda leaf: leaf[-1, most_likely_dps_idx], dps)
    # else:
    #     # Just pick one policy arbitrarily if we didn't optimize the policies.
    #     final_dps = jtu.tree_map(lambda leaf: leaf[-1, 0], dps)

    # # Evaluate this single policy against all failures
    # final_eps = jtu.tree_map(lambda leaf: leaf[-1], eps)
    # TODO for debugging plots, just use the initial policy and eps
    t_end = 0.0
    t_start = 0.0
    final_dps = jtu.tree_map(
        lambda leaf: leaf[-1] if eqx.is_array(leaf) else leaf, policy
    )
    final_eps = initial_states
    dp_logprobs = jnp.zeros((num_rounds, num_chains))
    ep_logprobs = jnp.zeros((num_rounds, num_chains))
    # TODO debugging bit ends here

    result = eqx.filter_vmap(lambda dp, ep: simulate(env, dp, ep), in_axes=(None, 0))(
        final_dps, final_eps
    )
    # TODO debug reward!!!

    # Plot the results
    fig = plt.figure(figsize=(32, 16), constrained_layout=True)
    axs = fig.subplot_mosaic(
        [
            ["initial_state", "trace", "reward"],
            ["initial_obs", "initial_obs", "initial_obs"],
            ["final_obs", "final_obs", "final_obs"],
        ]
    )

    # Plot the chain convergence
    if predict:
        axs["trace"].plot(ep_logprobs)
        axs["trace"].set_ylabel("Log probability after contingency update")
    else:
        axs["trace"].plot(dp_logprobs)
        axs["trace"].set_ylabel("Log probability after repair")

    axs["trace"].set_xlabel("# Samples")

    # Plot the initial states
    sns.swarmplot(
        data=[
            final_eps.non_ego_states[:, 0, 3],  # first car velocity
            final_eps.non_ego_states[:, 1, 3],  # second car velocity
            final_eps.shading_light_direction[:, 0],  # light direction x
            final_eps.shading_light_direction[:, 1],  # light direction y
            final_eps.shading_light_direction[:, 2],  # light direction z
        ],
        ax=axs["initial_state"],
    )
    axs["initial_state"].set_xticklabels(
        ["Car 1 v", "Car 2 v", "Light x", "Light y", "Light z"]
    )

    # Plot the reward across all failure cases
    sns.histplot(result.reward, ax=axs["reward"])
    axs["reward"].set_xlabel("Reward")

    # Plot the initial RGB observations tiled on the same subplot
    axs["initial_obs"].imshow(
        jnp.concatenate(result.initial_obs.color_image.transpose(0, 2, 1, 3), axis=1)
    )
    # And do the same for the final observations
    axs["final_obs"].imshow(
        jnp.concatenate(result.final_obs.color_image.transpose(0, 2, 1, 3), axis=1)
    )

    if use_gradients and use_stochasticity:
        alg_type = "mala"
    elif use_gradients and not use_stochasticity:
        alg_type = "gd"
    elif not use_gradients and use_stochasticity:
        alg_type = "rmh"
    else:
        alg_type = "static"
    filename = (
        f"results/{args.savename}/L_{L:0.1e}_"
        f"{num_rounds * num_mcmc_steps_per_round}_samples_"
        f"{quench_rounds}_quench_{'tempered_' if temper else ''}"
        f"{num_chains}_chains_step_dp_{dp_mcmc_step_size:0.1e}_"
        f"ep_{ep_mcmc_step_size:0.1e}_{alg_type}"
    )
    print(f"Saving results to: {filename}")
    os.makedirs(f"results/{args.savename}", exist_ok=True)
    plt.savefig(filename + ".png")

    # Save the final policy
    eqx.tree_serialise_leaves(filename + ".eqx", final_dps)

    # # Save the trace of policies
    # eqx.tree_serialise_leaves(filename + "_trace.eqx", dps)

    # Save the initial policy
    eqx.tree_serialise_leaves(
        f"results/{args.savename}/" + "initial_policy.eqx", policy
    )

    # Save the hyperparameters
    with open(filename + ".json", "w") as f:
        json.dump(
            {
                "initial_states": final_eps._asdict(),
                "time": t_end - t_start,
                "L": L,
                "dp_mcmc_step_size": dp_mcmc_step_size,
                "ep_mcmc_step_size": ep_mcmc_step_size,
                "num_rounds": num_rounds,
                "num_mcmc_steps_per_round": num_mcmc_steps_per_round,
                "num_chains": num_chains,
                "use_gradients": use_gradients,
                "use_stochasticity": use_stochasticity,
                "repair": repair,
                "predict": predict,
                "quench_rounds": quench_rounds,
                "tempering_schedule": tempering_schedule,
            },
            f,
            default=lambda x: x.tolist() if isinstance(x, Shaped[Array, "..."]) else x,
        )
