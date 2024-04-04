import argparse
import json
import os
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from jax.config import config
from jaxtyping import Array, Shaped

from architect.engines import predict_and_mitigate_failure_modes
from architect.engines.reinforce import init_sampler as init_reinforce_sampler
from architect.engines.reinforce import make_kernel as make_reinforce_kernel
from architect.engines.samplers import init_sampler as init_mcmc_sampler
from architect.engines.samplers import make_kernel as make_mcmc_kernel
from architect.systems.push_tilt.simulator import (
    Planner,
    PushTiltResult,
    object_parameters_logprior,
    observation_logprior,
    sample_object_parameters,
    sample_observations,
    simulate,
)


def plotting_cb(dp, ep):
    # Unpack
    planner = dp
    object_params, observations = ep

    # Simulate
    result = jax.vmap(simulate, in_axes=(None, 0, 0))(
        planner, observations, object_params
    )

    # Plot a bar chart with the net force and net moment
    fig, ax = plt.subplots()
    sns.swarmplot(
        data=[
            result.net_force,
            result.net_moment,
        ],
        ax=ax,
    )
    ax.set_xticklabels(["Net Force", "Net Moment"])
    # Also add a black line at 0 for reference
    ax.axhline(0, color="black", linestyle="-")

    # log the figure to wandb
    wandb.log({"plot": wandb.Image(fig)}, commit=False)

    # Close the figure
    plt.close()


if __name__ == "__main__":
    # Set up arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--savename", type=str, default="push_tilt")
    parser.add_argument("--seed", type=int, nargs="?", default=0)
    parser.add_argument("--L", type=float, nargs="?", default=10.0)
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
    parser.add_argument("--num_stress_test_cases", type=int, nargs="?", default=100)
    boolean_action = argparse.BooleanOptionalAction
    parser.add_argument("--repair", action=boolean_action, default=True)
    parser.add_argument("--predict", action=boolean_action, default=True)
    parser.add_argument("--temper", action=boolean_action, default=True)
    parser.add_argument("--grad_clip", type=float, nargs="?", default=10.0)
    parser.add_argument("--dont_normalize_gradients", action="store_true")
    args = parser.parse_args()

    # Hyperparameters
    L = args.L
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
        alg_type = "reinforce"
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
        project=args.savename + f"-{num_rounds}x{num_steps_per_round}",
        group=alg_type
        + ("-predict" if predict else "")
        + ("-repair" if repair else ""),
        config={
            "L": L,
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
    t = jnp.linspace(0, 1, num_rounds)
    tempering_schedule = 1 - jnp.exp(-10 * t) if temper else None

    # Make a PRNG key (#sorandom)
    prng_key = jrandom.PRNGKey(seed)

    # Initialize the planner policies (these will be DPs)
    prng_key, planner_keys = jrandom.split(prng_key)
    planner_keys = jrandom.split(planner_keys, num_chains)
    initial_dps = jax.vmap(Planner)(planner_keys)

    # Initialize the object parameters and observations (these will be the EPs)
    prng_key, param_keys = jrandom.split(prng_key)
    param_keys = jrandom.split(param_keys, num_chains)
    initial_params = jax.vmap(sample_object_parameters)(param_keys)
    prng_key, obs_keys = jrandom.split(prng_key)
    obs_keys = jrandom.split(obs_keys, num_chains)
    initial_observations = jax.vmap(sample_observations)(obs_keys, initial_params)
    initial_eps = (initial_params, initial_observations)

    # Also initialize a bunch of exogenous parameters to serve as stress test cases
    # for the policy
    prng_key, param_keys = jrandom.split(prng_key)
    param_keys = jrandom.split(param_keys, num_stress_test_cases)
    stress_test_params = jax.vmap(sample_object_parameters)(param_keys)
    prng_key, obs_keys = jrandom.split(prng_key)
    obs_keys = jrandom.split(obs_keys, num_stress_test_cases)
    stress_test_observations = jax.vmap(sample_observations)(
        obs_keys, stress_test_params
    )
    stress_test_eps = (stress_test_params, stress_test_observations)

    # Define a uniform prior over policies
    dp_logprior = lambda dps: jnp.array(0.0)

    # Define a composite logprior for eps
    ep_logprior = lambda eps: object_parameters_logprior(eps[0]) + observation_logprior(
        eps[0], eps[1]
    )

    # Choose which sampler to use
    if reinforce:
        init_sampler_fn = init_reinforce_sampler
        make_kernel_fn = lambda _, logprob_fn, step_size, _: make_reinforce_kernel(
            logprob_fn,
            step_size,
            perturbation_stddev=0.05,
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
        dp_logprior_fn=dp_logprior,
        ep_logprior_fn=ep_logprior,
        ep_potential_fn=lambda dp, ep: L * simulate(dp, ep[1], ep[0]).potential,
        dp_potential_fn=lambda dp, ep: -L * simulate(dp, ep[1], ep[0]).potential,
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
        potential_fn=lambda dp, ep: simulate(dp, ep[1], ep[0]).potential,
        failure_level=0.0,
        plotting_cb=plotting_cb,
    )
    t_end = time.perf_counter()
    print(
        f"Ran {num_rounds:,} rounds with {num_chains} chains in {t_end - t_start:.2f} s"
    )

    # Select the seeker trajectory that performs best against all hider strategies
    # predicted before the final round (choose from all chains)
    if repair:
        most_likely_dps_idx = jnp.argmax(dp_logprobs[-1], axis=-1)
        final_dps = jtu.tree_map(lambda leaf: leaf[-1, most_likely_dps_idx], dps)
    else:
        # Just pick one dispatch arbitrarily
        final_dps = jtu.tree_map(lambda leaf: leaf[-1, 0], dps)

    # Evaluate this against all contingencies
    final_eps = jtu.tree_map(lambda leaf: leaf[-1], eps)

    path = (
        f"results/{args.savename}/L_{L:0.1e}/"
        f"{num_rounds * num_steps_per_round}_samples/"
        f"{quench_rounds}_quench/{'tempered' if temper else 'no_temper'}"
        f"{num_chains}_chains/dp_{dp_mcmc_step_size:0.1e}/"
        f"ep_{ep_mcmc_step_size:0.1e}/"
        f"{'repair' if repair else 'no_repair'}/"
        f"{'predict' if predict else 'no_predict'}"
    )
    filename = os.path.join(path, alg_type)
    print(f"Saving results to: {filename}")
    os.makedirs(path, exist_ok=True)

    # Save the final design parameters (joined back into the full policy)
    eqx.tree_serialise_leaves(filename + ".eqx", final_dps)
    print(f"Saved final design parameters to: {filename}.eqx")

    # Save the trace of design parameters
    eqx.tree_serialise_leaves(filename + "_dp_trace.eqx", dps)

    # Save the results
    with open(filename + ".json", "w") as f:
        json.dump(
            {
                "final_eps": final_eps,
                "ep_logprobs": ep_logprobs,
                "time": t_end - t_start,
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
            },
            f,
            default=lambda x: x.tolist() if isinstance(x, Shaped[Array, "..."]) else x,
        )
