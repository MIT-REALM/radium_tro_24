import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from jax.nn import sigmoid
from jaxtyping import Array, Shaped

import wandb
from architect.engines import predict_and_mitigate_failure_modes
from architect.engines.reinforce import init_sampler as init_reinforce_sampler
from architect.engines.reinforce import make_kernel as make_reinforce_kernel
from architect.engines.samplers import init_sampler as init_mcmc_sampler
from architect.engines.samplers import make_kernel as make_mcmc_kernel
from architect.systems.power_systems.load_test_network import load_test_network


def plotting_cb(dp, eps):
    result = jax.vmap(sys, in_axes=(None, 0))(dp, eps)

    # Plot the results
    fig = plt.figure(figsize=(32, 16), constrained_layout=True)
    axs = fig.subplot_mosaic(
        [
            ["constraints", "cost"],
            ["generation", "voltage"],
            ["network", "network"],
        ]
    )

    # Plot the violations at the best dispatch from each chain
    sns.swarmplot(
        data=[
            result.P_gen_violation.sum(axis=-1),
            result.Q_gen_violation.sum(axis=-1),
            result.P_load_violation.sum(axis=-1),
            result.Q_load_violation.sum(axis=-1),
            result.V_violation.sum(axis=-1),
            result.acopf_residual,
        ],
        ax=axs["constraints"],
    )
    axs["constraints"].set_xticklabels(["Pg", "Qg", "Pd", "Qd", "V", "ACOPF error"])
    axs["constraints"].set_ylabel("Constraint Violation")

    # Plot generation cost vs constraint violation
    total_constraint_violation = (
        result.P_gen_violation.sum(axis=-1)
        + result.Q_gen_violation.sum(axis=-1)
        + result.P_load_violation.sum(axis=-1)
        + result.Q_load_violation.sum(axis=-1)
        + result.V_violation.sum(axis=-1)
    )
    axs["cost"].scatter(result.potential, total_constraint_violation)
    axs["cost"].set_xlabel("potential")
    axs["cost"].set_ylabel("Total constraint violation")

    # # Plot the generations along with their limits
    # bus = sys.gen_spec.buses
    # P_min, P_max = sys.gen_spec.P_limits.T
    # for i in range(num_chains):
    #     P = result.dispatch.gen.P[i, :]
    #     lower_error = P - P_min
    #     upper_error = P_max - P
    #     errs = jnp.vstack((lower_error, upper_error))
    #     axs["generation"].errorbar(
    #         bus,
    #         P,
    #         yerr=errs,
    #         linestyle="None",
    #         marker="o",
    #         markersize=10,
    #         linewidth=3.0,
    #         capsize=10.0,
    #         capthick=3.0,
    #     )
    # axs["generation"].set_ylabel("$P_g$ (p.u.)")

    # # Plot the voltages along with their limits
    # bus = jnp.arange(sys.n_bus)
    # V_min, V_max = sys.bus_voltage_limits.T
    # for i in range(num_chains):
    #     V = result.voltage_amplitudes[i, :]
    #     lower_error = V - V_min
    #     upper_error = V_max - V
    #     errs = jnp.vstack((lower_error, upper_error))
    #     axs["voltage"].errorbar(
    #         bus,
    #         V,
    #         yerr=errs,
    #         linestyle="None",
    #         marker="o",
    #         markersize=10,
    #         linewidth=3.0,
    #         capsize=10.0,
    #         capthick=3.0,
    #     )
    # axs["voltage"].set_ylabel("$|V|$ (p.u.)")

    # Plot the network states
    line = jnp.arange(sys.n_line)
    for i in range(num_chains):
        line_states = result.network_state.line_states[i, :]
        axs["network"].scatter(
            line,
            sigmoid(2 * line_states),
            marker="o",
            s=100 * total_constraint_violation[i] + 5,
        )
    axs["network"].set_ylabel("Line strength")
    axs["network"].set_xticks(line)

    # log the figure to wandb
    wandb.log({"plot": wandb.Image(fig)}, commit=False)

    # Close the figure
    plt.close()


if __name__ == "__main__":
    matplotlib.use("Agg")

    # Set up arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_name", type=str, nargs="?", default="case14")
    parser.add_argument("--L", type=float, nargs="?", default=100.0)
    parser.add_argument("--failure_level", type=float, nargs="?", default=4.0)
    parser.add_argument("--seed", type=int, nargs="?", default=0)
    parser.add_argument("--dp_mcmc_step_size", type=float, nargs="?", default=1e-6)
    parser.add_argument("--ep_mcmc_step_size", type=float, nargs="?", default=1e-2)
    parser.add_argument("--num_rounds", type=int, nargs="?", default=100)
    parser.add_argument("--num_mcmc_steps_per_round", type=int, nargs="?", default=10)
    parser.add_argument("--num_chains", type=int, nargs="?", default=10)
    parser.add_argument("--quench_rounds", type=int, nargs="?", default=20)
    parser.add_argument("--disable_gradients", action="store_true")
    parser.add_argument("--disable_stochasticity", action="store_true")
    parser.add_argument("--num_stress_test_cases", type=int, nargs="?", default=1000)
    boolean_action = argparse.BooleanOptionalAction
    parser.add_argument("--reinforce", action="store_true")
    parser.add_argument("--repair", action=boolean_action, default=True)
    parser.add_argument("--predict", action=boolean_action, default=True)
    parser.add_argument("--temper", action=boolean_action, default=False)
    parser.add_argument("--grad_clip", type=float, nargs="?", default=float("inf"))
    args = parser.parse_args()

    # Hyperparameters
    case_name = args.case_name
    L = args.L
    failure_level = args.failure_level
    dp_mcmc_step_size = args.dp_mcmc_step_size
    ep_mcmc_step_size = args.ep_mcmc_step_size
    num_rounds = args.num_rounds
    num_mcmc_steps_per_round = args.num_mcmc_steps_per_round
    num_chains = args.num_chains
    use_gradients = not args.disable_gradients
    use_stochasticity = not args.disable_stochasticity
    reinforce = args.reinforce
    repair = args.repair
    predict = args.predict
    temper = args.temper
    quench_rounds = args.quench_rounds
    grad_clip = args.grad_clip

    print(f"Running SC-ACOPF on {case_name} with hyperparameters:")
    print(f"\tcase_name = {case_name}")
    print(f"\tL = {L}")
    print(f"\tfailure_level = {failure_level}")
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
    print(f"\tgrad_clip = {grad_clip}")

    if reinforce:
        alg_type = "reinforce"
    elif use_gradients and use_stochasticity:
        alg_type = "mala"
    elif use_gradients and not use_stochasticity:
        alg_type = "gd"
    elif not use_gradients and use_stochasticity:
        alg_type = "rmh"
    else:
        alg_type = "static"

    # Initialize logger
    wandb.init(
        project=f"tro2-scopf-{case_name}",
        group=alg_type
        + ("-predict" if predict else "")
        + ("-repair" if repair else ""),
        config={
            "L": L,
            "case_name": case_name,
            "seed": args.seed,
            "failure_level": failure_level,
            "reinforce": reinforce,
            "dp_mcmc_step_size": dp_mcmc_step_size,
            "ep_mcmc_step_size": ep_mcmc_step_size,
            "num_rounds": num_rounds,
            "num_steps_per_round": num_mcmc_steps_per_round,
            "num_chains": num_chains,
            "use_gradients": use_gradients,
            "use_stochasticity": use_stochasticity,
            "repair": repair,
            "predict": predict,
            "temper": temper,
            "quench_rounds": quench_rounds,
            "grad_clip": grad_clip,
        },
    )

    # Add exponential tempering if using
    t = jnp.linspace(0, 1, num_rounds)
    tempering_schedule = 1 - jnp.exp(-10 * t) if temper else None

    # Load the test case
    sys = load_test_network(case_name, penalty=L)

    # Make a PRNG key (#sorandom)
    prng_key = jrandom.PRNGKey(args.seed)

    # Initialize the dispatch randomly
    prng_key, dispatch_key = jrandom.split(prng_key)
    dispatch_keys = jrandom.split(dispatch_key, num_chains)
    init_design_params = jax.vmap(sys.sample_random_dispatch)(dispatch_keys)

    # Initialize the network randomly
    prng_key, network_key = jrandom.split(prng_key)
    network_keys = jrandom.split(network_key, num_chains)
    init_exogenous_params = jax.vmap(sys.sample_random_network_state)(network_keys)

    # Initialize stress test parameters
    prng_key, network_key = jrandom.split(prng_key)
    network_keys = jrandom.split(network_key, args.num_stress_test_cases)
    stress_test_eps = jax.vmap(sys.sample_random_network_state)(network_keys)

    # This sampler yields either MALA, GD, or RMH depending on whether gradients
    # and/or stochasticity are enabled
    if reinforce:
        init_sampler_fn = init_reinforce_sampler
        noise_scale = 0.1
        make_kernel_fn = lambda _1, logprob_fn, step_size, _2: make_reinforce_kernel(
            logprob_fn,
            step_size,
            perturbation_stddev=noise_scale,
            baseline_update_rate=0.5,
        )
    else:
        # This sampler yields either MALA, GD, or RMH depending on whether gradients and/or
        # stochasticity are enabled
        init_sampler_fn = lambda params, logprob_fn: init_mcmc_sampler(
            params,
            logprob_fn,
            True,  # TODO don't normalize gradients
        )
        make_kernel_fn = (
            lambda _, logprob_fn, step_size, stochasticity: make_mcmc_kernel(
                logprob_fn,
                step_size,
                use_gradients,
                stochasticity,
                grad_clip,
                True,  # TODO don't normalize gradients
                True,  # use metroplis-hastings
            )
        )

    # Run the prediction+mitigation process
    t_start = time.perf_counter()
    dps, eps, dp_logprobs, ep_logprobs = predict_and_mitigate_failure_modes(
        prng_key,
        init_design_params,
        init_exogenous_params,
        dp_logprior_fn=sys.dispatch_prior_logprob,
        ep_logprior_fn=sys.network_state_prior_logprob,
        ep_potential_fn=lambda dp, ep: -jax.nn.elu(
            failure_level - sys(dp, ep).potential
        ),
        dp_potential_fn=lambda dp, ep: -jax.nn.elu(
            sys(dp, ep).potential - failure_level
        ),
        # ep_potential_fn=lambda dp, ep: -(failure_level - sys(dp, ep).potential),
        # dp_potential_fn=lambda dp, ep: -(sys(dp, ep).potential - failure_level),
        init_sampler=init_sampler_fn,
        make_kernel=make_kernel_fn,
        num_rounds=num_rounds,
        num_mcmc_steps_per_round=num_mcmc_steps_per_round,
        dp_mcmc_step_size=dp_mcmc_step_size,
        ep_mcmc_step_size=ep_mcmc_step_size,
        use_stochasticity=use_stochasticity,
        repair=repair,
        predict=predict,
        quench_rounds=quench_rounds,
        tempering_schedule=tempering_schedule,
        stress_test_cases=stress_test_eps,
        potential_fn=lambda dp, ep: sys(dp, ep).potential,
        plotting_cb=plotting_cb,
        quench_dps_only=True,
        failure_level=failure_level,
    )
    t_end = time.perf_counter()
    print(
        f"Ran {num_rounds:,} rounds with {num_chains} chains in {t_end - t_start:.2f} s"
    )

    # Select the dispatch that performs best against all contingencies predicted before
    # the final round (choose from all chains)
    if repair:
        most_likely_dps_idx = jnp.argmax(dp_logprobs[-1], axis=-1)
        final_dps = jtu.tree_map(lambda leaf: leaf[-1, most_likely_dps_idx], dps)
    else:
        # Just pick one dispatch arbitrarily
        final_dps = jtu.tree_map(lambda leaf: leaf[-1, 0], dps)
    # Evaluate this against all contingencies
    final_eps = jtu.tree_map(lambda leaf: leaf[-1], eps)
    result = jax.vmap(sys, in_axes=(None, 0))(final_dps, final_eps)

    # Plot the results
    fig = plt.figure(figsize=(32, 16), constrained_layout=True)
    axs = fig.subplot_mosaic(
        [
            ["constraints", "cost"],
            ["trace", "trace"],
            ["generation", "voltage"],
            ["network", "network"],
        ]
    )

    # Plot the violations at the best dispatch from each chain
    sns.swarmplot(
        data=[
            result.P_gen_violation.sum(axis=-1),
            result.Q_gen_violation.sum(axis=-1),
            result.P_load_violation.sum(axis=-1),
            result.Q_load_violation.sum(axis=-1),
            result.V_violation.sum(axis=-1),
            result.acopf_residual,
        ],
        ax=axs["constraints"],
    )
    axs["constraints"].set_xticklabels(["Pg", "Qg", "Pd", "Qd", "V", "ACOPF error"])
    axs["constraints"].set_ylabel("Constraint Violation")

    # Plot generation cost vs constraint violation
    total_constraint_violation = (
        result.P_gen_violation.sum(axis=-1)
        + result.Q_gen_violation.sum(axis=-1)
        + result.P_load_violation.sum(axis=-1)
        + result.Q_load_violation.sum(axis=-1)
        + result.V_violation.sum(axis=-1)
    )
    axs["cost"].scatter(result.generation_cost, total_constraint_violation)
    axs["cost"].set_xlabel("Generation cost")
    axs["cost"].set_ylabel("Total constraint violation")

    # Plot the chain convergence
    if predict:
        axs["trace"].plot(ep_logprobs)
        axs["trace"].set_ylabel("Log probability after contingency update")
    else:
        axs["trace"].plot(dp_logprobs)
        axs["trace"].set_ylabel("Log probability after repair")

    axs["trace"].set_xlabel("# Samples")

    # Plot the generations along with their limits
    bus = sys.gen_spec.buses
    P_min, P_max = sys.gen_spec.P_limits.T
    for i in range(num_chains):
        P = result.dispatch.gen.P[i, :]
        lower_error = P - P_min
        upper_error = P_max - P
        errs = jnp.vstack((lower_error, upper_error))
        axs["generation"].errorbar(
            bus,
            P,
            yerr=errs,
            linestyle="None",
            marker="o",
            markersize=10,
            linewidth=3.0,
            capsize=10.0,
            capthick=3.0,
        )
    axs["generation"].set_ylabel("$P_g$ (p.u.)")

    # Plot the voltages along with their limits
    bus = jnp.arange(sys.n_bus)
    V_min, V_max = sys.bus_voltage_limits.T
    for i in range(num_chains):
        V = result.voltage_amplitudes[i, :]
        lower_error = V - V_min
        upper_error = V_max - V
        errs = jnp.vstack((lower_error, upper_error))
        axs["voltage"].errorbar(
            bus,
            V,
            yerr=errs,
            linestyle="None",
            marker="o",
            markersize=10,
            linewidth=3.0,
            capsize=10.0,
            capthick=3.0,
        )
    axs["voltage"].set_ylabel("$|V|$ (p.u.)")

    # Plot the network states
    line = jnp.arange(sys.n_line)
    for i in range(num_chains):
        line_states = result.network_state.line_states[i, :]
        axs["network"].scatter(
            line,
            sigmoid(2 * line_states),
            marker="o",
            s=100 * total_constraint_violation[i] + 5,
        )
    axs["network"].set_ylabel("Line strength")
    axs["network"].set_xticks(line)

    experiment_type = "scacopf" if predict else "acopf"
    if use_gradients and use_stochasticity:
        alg_type = "mala"
    elif use_gradients and not use_stochasticity:
        alg_type = "gd"
    elif not use_gradients and use_stochasticity:
        alg_type = "rmh"
    else:
        alg_type = "static"
    filename = (
        f"results/{experiment_type}/{case_name}/L_{L:0.1e}_"
        f"{num_rounds * num_mcmc_steps_per_round}_samples_"
        f"{quench_rounds}_quench_{'tempered_' if temper else ''}"
        f"{num_chains}_chains_step_dp_{dp_mcmc_step_size:0.1e}_"
        f"ep_{ep_mcmc_step_size:0.1e}_{alg_type}"
    )
    print(f"Saving results to: {filename}")
    os.makedirs(f"results/{experiment_type}/{case_name}", exist_ok=True)
    plt.savefig(filename + ".png")

    # Save the dispatch
    with open(filename + ".json", "w") as f:
        json.dump(
            {
                "case": case_name,
                "dispatch": final_dps._asdict(),
                "network_state": final_eps._asdict(),
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

    # Save the trace of design parameters
    with open(filename + "_dp_trace.json", "w") as f:
        json.dump(
            {
                "case": case_name,
                "dispatch": dps._asdict(),
                "num_rounds": num_rounds,
                "num_chains": num_chains,
            },
            f,
            default=lambda x: x.tolist() if isinstance(x, Shaped[Array, "..."]) else x,
        )
