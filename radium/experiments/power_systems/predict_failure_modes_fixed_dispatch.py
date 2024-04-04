import argparse
import time

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import seaborn as sns
from jax.nn import sigmoid

from radium.engines import predict_and_mitigate_failure_modes
from radium.engines.samplers import init_sampler as init_mcmc_sampler
from radium.engines.samplers import make_kernel as make_mcmc_kernel
from radium.systems.power_systems.load_test_network import load_test_network

if __name__ == "__main__":
    # Set up arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--case_name", type=str, nargs="?", default="case14")
    parser.add_argument("--L", type=float, nargs="?", default=100.0)
    parser.add_argument("--dp_mcmc_step_size", type=float, nargs="?", default=1e-6)
    parser.add_argument("--ep_mcmc_step_size", type=float, nargs="?", default=1e-2)
    parser.add_argument("--num_gd_steps", type=int, nargs="?", default=200)
    parser.add_argument("--num_rounds", type=int, nargs="?", default=10)
    parser.add_argument("--num_mcmc_steps_per_round", type=int, nargs="?", default=20)
    parser.add_argument("--num_chains", type=int, nargs="?", default=10)
    args = parser.parse_args()

    # Hyperparameters
    case_name = args.case_name
    L = args.L
    dp_mcmc_step_size = args.dp_mcmc_step_size
    ep_mcmc_step_size = args.ep_mcmc_step_size
    num_gd_steps = args.num_gd_steps
    num_rounds = args.num_rounds
    num_mcmc_steps_per_round = args.num_mcmc_steps_per_round
    num_chains = args.num_chains

    # Load the test case
    sys = load_test_network(case_name, penalty=L)

    # Make a PRNG key (#sorandom)
    prng_key = jrandom.PRNGKey(0)

    # Initialize the dispatch randomly
    prng_key, dispatch_key = jrandom.split(prng_key)
    dispatch_keys = jrandom.split(dispatch_key, num_chains)
    init_design_params = jax.vmap(sys.sample_random_dispatch)(dispatch_keys)

    # Initialize the network randomly
    prng_key, network_key = jrandom.split(prng_key)
    network_keys = jrandom.split(network_key, num_chains)
    init_exogenous_params = jax.vmap(sys.sample_random_network_state)(network_keys)

    # This sampler yields either MALA, GD, or RMH depending on whether gradients and/or
    # stochasticity are enabled
    init_sampler_fn = lambda params, logprob_fn: init_mcmc_sampler(
        params,
        logprob_fn,
        False,  # don't normalize gradients
    )
    make_kernel_fn = lambda _, logprob_fn, step_size, stochasticity: make_mcmc_kernel(
        logprob_fn,
        step_size,
        True,  # use gradients
        stochasticity,
        float("inf"),  # don't clip gradients
        False,  # don't normalize gradients
        True,  # use metroplis-hastings
    )

    # Run gradient descent with no prediction to get a (non-robust) dispatch.
    t_start = time.perf_counter()
    dps, eps, dp_logprobs, ep_logprobs = predict_and_mitigate_failure_modes(
        prng_key,
        init_design_params,
        init_exogenous_params,
        dp_logprior_fn=sys.dispatch_prior_logprob,
        ep_logprior_fn=sys.network_state_prior_logprob,
        ep_potential_fn=lambda dp, ep: sys(dp, ep).potential,
        dp_potential_fn=lambda dp, ep: -sys(dp, ep).potential,
        init_sampler=init_sampler_fn,
        make_kernel=make_kernel_fn,
        num_rounds=num_gd_steps,
        num_mcmc_steps_per_round=1,
        dp_mcmc_step_size=dp_mcmc_step_size,
        ep_mcmc_step_size=ep_mcmc_step_size,
        use_stochasticity=False,
        repair=True,
        predict=False,
        quench_rounds=0,
        tempering_schedule=None,
    )
    t_end = time.perf_counter()
    print(
        f"Ran {num_rounds:,} rounds with {num_chains} chains in {t_end - t_start:.2f} s"
    )

    # Select the dispatch that performs best against all contingencies
    most_likely_dps_idx = jnp.argmax(dp_logprobs[-1], axis=-1)
    final_dps = jtu.tree_map(
        lambda leaf: leaf[-1, most_likely_dps_idx].reshape(1, -1), dps
    )

    # Now re-solve for contingencies that specifically attack this dispatch
    t_start = time.perf_counter()
    _, eps, _, ep_logprobs = predict_and_mitigate_failure_modes(
        prng_key,
        final_dps,
        init_exogenous_params,
        dp_logprior_fn=sys.dispatch_prior_logprob,
        ep_logprior_fn=sys.network_state_prior_logprob,
        ep_potential_fn=lambda dp, ep: sys(dp, ep).potential,
        dp_potential_fn=lambda dp, ep: -sys(dp, ep).potential,
        num_rounds=num_rounds,
        num_mcmc_steps_per_round=num_mcmc_steps_per_round,
        dp_mcmc_step_size=dp_mcmc_step_size,
        ep_mcmc_step_size=ep_mcmc_step_size,
        use_gradients=True,
        use_stochasticity=True,
        repair=False,
        predict=True,
        quench_rounds=0,
        tempering_schedule=None,
    )
    t_end = time.perf_counter()
    print(
        f"Ran {num_rounds:,} rounds with {num_chains} chains in {t_end - t_start:.2f} s"
    )

    # Evaluate this against the top 4 contingencies contingencies
    final_dps = jtu.tree_map(lambda leaf: leaf.reshape(-1), final_dps)

    most_likely_eps_idx = jnp.argsort(ep_logprobs[-1], axis=-1)[-4:]
    final_eps = jtu.tree_map(lambda leaf: leaf[-1, most_likely_eps_idx], eps)
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
    axs["trace"].plot(ep_logprobs)
    axs["trace"].set_ylabel("Log probability after prediction")

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

    plt.show()
