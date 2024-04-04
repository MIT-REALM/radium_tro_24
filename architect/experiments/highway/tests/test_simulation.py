"""Test the highway simulation"""
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt

from architect.engines.samplers import normalize_gradients_with_threshold
from architect.experiments.highway.predict_and_mitigate import (
    non_ego_actions_prior_logprob,
    sample_non_ego_actions,
    simulate,
)
from architect.experiments.highway.train_highway_agent import make_highway_env
from architect.systems.highway.driving_policy import DrivingPolicy
from architect.utils import softmin

if __name__ == "__main__":
    # Make a PRNG key (#sorandom)
    prng_key = jrandom.PRNGKey(2)

    # Make the environment to use
    image_shape = (32, 32)
    env = make_highway_env(image_shape)
    env._collision_penalty = 10.0

    # Load the model (key doesn't matter; we'll replace all leaves with the saved
    # parameters), duplicating the model for each chain. We'll also split partition
    # out just the continuous parameters, which will be our design parameters
    num_dp_chains = 2
    dummy_policy = DrivingPolicy(jrandom.PRNGKey(0), image_shape)
    load_policy = lambda _: eqx.tree_deserialise_leaves(
        "results/highway/initial_policy.eqx", dummy_policy
    )
    get_dps = lambda _: eqx.partition(load_policy(_), eqx.is_array)[0]
    initial_dps = eqx.filter_vmap(get_dps)(jnp.arange(num_dp_chains))
    # Also save out the static part of the policy
    initial_dp, static_policy = eqx.partition(load_policy(None), eqx.is_array)

    # Initialize some fixed initial states
    # Hack: reset covariances to remove variation in initial state
    env._initial_state_covariance = 1e-3 * env._initial_state_covariance
    env._shading_light_direction_covariance = (
        1e-3 * env._shading_light_direction_covariance
    )
    # Hack: change initial conditions
    # env._initial_ego_state = jnp.array([-100.0, -3.0, 0.0, 10.0])
    # env._initial_non_ego_states = jnp.array(
    #     [
    #         [-95.0, 3.0, 0.0, 5.0],
    #         [-70, 3.0, 0.0, 8.0],
    #     ]
    # )
    prng_key, initial_state_key = jrandom.split(prng_key)
    initial_state = env.reset(initial_state_key)

    # Initialize some random actions for the non-ego vehicles
    num_ep_chains = 2
    prng_key, ep_key = jrandom.split(prng_key)
    ep_keys = jrandom.split(ep_key, num_ep_chains)
    noise_scale = 0.5
    T = 60
    sample_eps_fn = lambda key: sample_non_ego_actions(key, env, T, 2, noise_scale)
    initial_eps = jax.vmap(sample_eps_fn)(ep_keys)

    # Define the logprob function
    L = 10.0
    failure_level = 7.4
    potential_fn = lambda actions, policy, t: -L * jax.nn.elu(
        failure_level
        - simulate(env, policy, initial_state, actions, static_policy, t).potential
    )
    logprior_fn = lambda actions: non_ego_actions_prior_logprob(
        actions, env, noise_scale
    )
    # logprob_fn = lambda actions, policies, t: softmin(
    #     jax.vmap(potential_fn, in_axes=(None, 0, None))(actions, policies, t),
    #     sharpness=0.05,
    # ) + logprior_fn(actions)
    logprob_fn = lambda actions, policies, t: jax.vmap(
        potential_fn, in_axes=(None, 0, None)
    )(actions, policies, t).min() + logprior_fn(actions)

    # # Get the norm of the gradient of actions wrt actions for different simulation
    # # lengths
    # ts = [1, 2, 4, 8, 16, 32, 60]
    # grad_norms = []
    # for t in ts:
    #     actions = initial_eps[:, :t]
    #     logprob, grad = jax.vmap(
    #         jax.value_and_grad(logprob_fn), in_axes=(0, None, None)
    #     )(actions, initial_dps, t)
    #     grad_norms.append(jax.vmap(jnp.linalg.norm)(grad))
    #     print(f"{t}: grad norms: {grad_norms[-1]} (values: {logprob})")

    # Test whether gradient ascent works
    t = 60
    lr = 1e-1
    steps = 10
    value_and_grad_fn = jax.jit(
        jax.value_and_grad(
            # lambda ep: simulate(
            #     env, initial_dp, initial_state, ep, static_policy, t
            # ).potential
            lambda ep: logprob_fn(ep, initial_dps, t)
        )
    )
    ep = initial_eps[1, :t]
    for i in range(steps):
        v, g = value_and_grad_fn(ep)
        print(f"Step {i}: potential: {v}, grad norm: {jnp.linalg.norm(g)}", end="")
        g = normalize_gradients_with_threshold(g, 10.0)
        print(f" (normalized: {jnp.linalg.norm(g)})")
        ep = ep + lr * g  # gradient ASCENT to increase potential

    # Simulate
    result = simulate(env, initial_dp, initial_state, ep, static_policy, t)

    # Plot the results
    fig = plt.figure(figsize=(32, 16), constrained_layout=True)
    axs = fig.subplot_mosaic(
        [
            ["trajectory", "trajectory", "trajectory", "trajectory", "trajectory"],
            ["initial_obs", "initial_obs", "initial_obs", "initial_obs", "initial_obs"],
            ["final_obs", "final_obs", "final_obs", "final_obs", "final_obs"],
        ],
    )

    # Plot the trajectories for each case, color-coded by potential
    max_potential = jnp.max(result.potential)
    min_potential = jnp.min(result.potential)
    normalized_potential = (result.potential - min_potential) / (
        max_potential - min_potential
    )
    axs["trajectory"].axhline(7.5, linestyle="--", color="k")
    axs["trajectory"].axhline(-7.5, linestyle="--", color="k")
    axs["trajectory"].plot(
        result.ego_trajectory[:, 0].T,
        result.ego_trajectory[:, 1].T,
        linestyle="-",
        label="Ego",
    )
    axs["trajectory"].plot(
        result.non_ego_trajectory[:, 0, 0],
        result.non_ego_trajectory[:, 0, 1],
        linestyle="-.",
        label="Non-ego 1",
    )
    axs["trajectory"].plot(
        result.non_ego_trajectory[:, 1, 0],
        result.non_ego_trajectory[:, 1, 1],
        linestyle="--",
        label="Non-ego 2",
    )

    # Plot the initial RGB observations tiled on the same subplot
    axs["initial_obs"].imshow(result.initial_obs.color_image.transpose(1, 0, 2))
    # And do the same for the final observations
    axs["final_obs"].imshow(result.final_obs.color_image.transpose(1, 0, 2))
    plt.show()
