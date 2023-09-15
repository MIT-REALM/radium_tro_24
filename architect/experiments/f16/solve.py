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
import wandb
from jax.config import config
from jax_f16.f16 import F16
from jaxtyping import Array, Shaped

from architect.engines import predict_and_mitigate_failure_modes
from architect.engines.reinforce import init_sampler as init_reinforce_sampler
from architect.engines.reinforce import make_kernel as make_reinforce_kernel
from architect.engines.samplers import init_sampler as init_mcmc_sampler
from architect.engines.samplers import make_kernel as make_mcmc_kernel
from architect.systems.f16.simulator import (
    ResidualControl,
    initial_state_logprior,
    sample_initial_states,
    simulate,
)


def plotting_cb(dp, ep):
    result = jax.vmap(simulate, in_axes=(0, None))(ep, dp)

    # Plot the results
    fig = plt.figure(figsize=(16, 8), constrained_layout=True)
    axs = fig.subplot_mosaic(
        [
            ["altitude", "roll"],
        ]
    )

    # Plot the trajectories for each case, color-coded by potential
    max_potential = jnp.max(result.potential)
    min_potential = jnp.min(result.potential) - 1e-3
    normalized_potential = (result.potential - min_potential) / (
        max_potential - min_potential
    )
    for chain_idx in range(result.states.shape[0]):
        # Plot the altitude over time
        axs["altitude"].plot(
            result.states[chain_idx, :, F16.H],
            color=plt.cm.plasma(normalized_potential[chain_idx]),
        )
        axs["altitude"].set_ylabel("Altitude (ft)")
        axs["altitude"].set_xlabel("Time (s)")

        # Also plot the roll angle and rate
        axs["roll"].plot(
            result.states[chain_idx, :, F16.PHI],
            label="Roll Angle",
            color=plt.cm.plasma(normalized_potential[chain_idx]),
        )
        axs["roll"].plot(
            result.states[chain_idx, :, F16.P],
            label="Roll Rate",
            linestyle="--",
            color=plt.cm.plasma(normalized_potential[chain_idx]),
        )
        axs["roll"].set_xlabel("Time (s)")

        if chain_idx == 0:
            axs["roll"].legend()

    axs["altitude"].set_ylim([-100.0, 5e3])

    # log the figure to wandb
    wandb.log({"plot": wandb.Image(fig)}, commit=False)

    # Close the figure
    plt.close()


if __name__ == "__main__":
    # Set up arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--savename", type=str, default="f16_ep_1e-4")
    parser.add_argument(
        "--model_path", type=str, default="results/f16/initial_policy.eqx"
    )
    parser.add_argument("--seed", type=int, nargs="?", default=0)
    parser.add_argument("--L", type=float, nargs="?", default=1.0)
    parser.add_argument("--dp_logprior_scale", type=float, nargs="?", default=1.0)
    parser.add_argument("--dp_mcmc_step_size", type=float, nargs="?", default=1e-2)
    parser.add_argument("--ep_mcmc_step_size", type=float, nargs="?", default=1e-4)
    parser.add_argument("--num_rounds", type=int, nargs="?", default=100)
    parser.add_argument("--num_steps_per_round", type=int, nargs="?", default=5)
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
    parser.add_argument("--grad_clip", type=float, nargs="?", default=float("inf"))
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
            "model_path": args.model_path,
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

    # Initialize the residual policies (these will be DPs)
    # dummy_policy = ResidualControl(jrandom.PRNGKey(0))
    # load_policy = lambda _: eqx.tree_deserialise_leaves(args.model_path, dummy_policy)
    # get_dps = lambda _: eqx.partition(load_policy(_), eqx.is_array)[0]
    # initial_dps = eqx.filter_vmap(get_dps)(jnp.arange(num_chains))
    prng_key, dp_key = jrandom.split(prng_key)
    dp_keys = jrandom.split(dp_key, num_chains)
    initial_dps = jax.vmap(ResidualControl)(dp_keys)

    # Initialize the initial states (these will be EPs)
    prng_key, x0_key = jrandom.split(prng_key)
    x0_keys = jrandom.split(x0_key, num_chains)
    initial_eps = jax.vmap(sample_initial_states)(x0_keys)

    # Also initialize a bunch of exogenous parameters to serve as stress test cases
    # for the policy
    prng_key, ep_key = jrandom.split(prng_key)
    ep_keys = jrandom.split(ep_key, num_stress_test_cases)
    stress_test_eps = jax.vmap(lambda key: sample_initial_states(key))(ep_keys)

    # Define a uniform prior over residual policies
    residual_control_logprior = lambda dps: jnp.array(0.0)

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
        dp_logprior_fn=residual_control_logprior,
        ep_logprior_fn=initial_state_logprior,
        ep_potential_fn=lambda dp, ep: L * simulate(ep, dp).potential,
        dp_potential_fn=lambda dp, ep: -L * simulate(ep, dp).potential,
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
        potential_fn=lambda dp, ep: simulate(ep, dp).potential,
        failure_level=15.0,
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
    filename = os.path.join(path, alg_type + f"_{seed}")
    print(f"Saving results to: {filename}")
    os.makedirs(path, exist_ok=True)

    # Save the final design parameters (joined back into the full policy)
    eqx.tree_serialise_leaves(filename + ".eqx", final_dps)

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
