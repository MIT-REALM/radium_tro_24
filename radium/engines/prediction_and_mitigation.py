"""
Failure mode prediction and mitigation code.

Provides system-agnostic code for failure mode prediction and mitigation.
"""

import operator
import time

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
from beartype import beartype
from beartype.typing import Any, Callable, Optional, Tuple
from jaxtyping import Array, Float, Integer, PyTree, jaxtyped
from tqdm import tqdm

import wandb
from radium.types import LogLikelihood, Params, Sampler


@jaxtyped(typechecker=beartype)
def run_chain(
    prng_key: Integer[Array, "..."],
    kernel: Sampler,
    initial_state: PyTree,
    num_samples: int,
) -> Tuple[Params, Float[Array, ""], Float[Array, ""], Any]:
    """
    Run the given MCMC kernel for the specified number of steps, returning last sample
    and associated log likelihood.

    args:
        prng_key: JAX rng key
        kernel: the MCMC kernel to run
        initial_state: the starting state for the sampler
        num_samples: how many samples to take
    returns:
        the last state of the chain, the associated log probability, and acceptance
        rate, along with arbitrary debug information.
    """

    def one_step(_, carry):
        # unpack carry
        state, prng_key = carry

        # Take a step
        prng_key, subkey = jrandom.split(prng_key)
        new_state = kernel(subkey, state)

        # Re-pack the carry and return
        carry = (new_state, prng_key)
        return carry

    initial_carry = (initial_state, prng_key)
    final_state, _ = jax.lax.fori_loop(0, num_samples, one_step, initial_carry)

    # Compute acceptance rate if the sampler supports it
    if hasattr(final_state, "num_accepts"):
        # convert to float
        accept_rate = final_state.num_accepts.astype(jnp.float32) / num_samples
    else:
        accept_rate = jnp.array(1.0)

    # Add debug information
    debug = {}
    if hasattr(final_state, "logdensity_grad"):
        # Compute the norm of the gradient
        debug["final_grad_norm"] = jnp.sqrt(
            jtu.tree_reduce(
                operator.add,
                jtu.tree_map(lambda x: jnp.sum(x * x), final_state.logdensity_grad),
            )
        )
        debug["final_grad"] = final_state.logdensity_grad

    return (
        final_state.position,
        final_state.logdensity,
        accept_rate,
        debug,
    )


# @jaxtyped(typechecker=beartype)
def predict_and_mitigate_failure_modes(
    prng_key: Integer[Array, "..."],
    design_params: Params,
    exogenous_params: Params,
    dp_logprior_fn: LogLikelihood,
    ep_logprior_fn: LogLikelihood,
    ep_potential_fn: Callable[[Params, Params], Float[Array, ""]],
    dp_potential_fn: Callable[[Params, Params], Float[Array, ""]],
    init_sampler: Callable[[Params, LogLikelihood], PyTree],
    make_kernel: Callable[[Params, LogLikelihood, float, bool], Sampler],
    num_rounds: int,
    num_mcmc_steps_per_round: int,
    dp_mcmc_step_size: float,
    ep_mcmc_step_size: float,
    use_stochasticity: bool = True,
    repair: bool = True,
    predict: bool = True,
    quench_rounds: int = 0,
    quench_dps_only: bool = True,
    tempering_schedule: Optional[Float[Array, " num_rounds"]] = None,
    logging_prefix: str = "",
    stress_test_cases: Optional[Params] = None,
    potential_fn: Optional[Callable[[Params, Params], Float[Array, ""]]] = None,
    failure_level: float = 0.0,
    plotting_cb: Optional[Callable[[Params, Params], None]] = None,
    test_every: int = 1,
) -> Tuple[
    Params,
    Params,
    Optional[Float[Array, "num_rounds n"]],
    Optional[Float[Array, "num_rounds m"]],
]:
    """
    Run the sequential monte carlo algorithm for SC-ACOPF

    args:
        prng_key: JAX rng key
        design_params: initial controllable parameter population
        exogenous_params: initial uncontrollable parameter population
        dp_logprior_fn: function taking a single set of design parameters, returning
            the non-normalized log probability of those parameters according to the
            prior distribution
        ep_logprior_fn: function taking a single set of exogenous parameters,returning
            the non-normalized log probability of those parameters according to the
            prior distribution
        ep_potential_fn: function taking a single set of design and exogenous parameters,
            returning the cost metric for the system's performance. Used for EPs.
            Higher potential -> higher likelihood.
        dp_potential_fn: function taking a single set of design and exogenous parameters,
            returning the cost metric for the system's performance. Used for DPs
            Higher potential -> higher likelihood.
        init_sampler: function taking a single set of design parameters and a log
            likelihood function, returning the initial state for a sampler/optimization
            algorithm.
        make_kernel: function taking parameters, a log likelihood function, a step size,
             and a boolean flag indicating whether or not to disable stochasticity,
            returning a sampler (or optimizer) to use.
        num_rounds: how many steps to take for the top-level predict-repair algorithm
        num_mcmc_steps_per_round: the number of steps to run for each set of MCMC chains
            within each round
        dp_mcmc_step_size: the size of MCMC steps for design parameters
        ep_mcmc_step_size: the size of MCMC steps for exogenous parameters
        use_gradients: if True, use gradients in the proposal and acceptance steps
        use_stochasticity: if True, add a Gaussian to the proposal and acceptance steps
        repair: if True, run the repair steps
        predict: if True, run the predict steps
        quench_rounds: if True, turn off stochasticity for the last rounds
        quench_dps_only: if True, turn off stochasticity for the design parameters only
            (i.e. don't quench the exogenous parameters).
        tempering_schedule: a monotonically increasing array of positive tempering
            values. If not provided, defaults to all 1s (no tempering).
        logging_prefix: a string to prepend to all logging messages
        stress_test_cases: test the DPs at the end of each round with these EPs,
            recording the failure rate and average cost, if not None.
        potential_fn: the function used for stress testing
        failure_level: the level of cost at which a failure is deemed to occur.
        plotting_cb: a callback function to call at the end of each round, if not None.
            The callback function should take the current best dp and all predicted
            eps as arguments, and it should log any desired images or plots to wandb,
            but it should NOT commit those logs.
        test_every: the number of steps between stress tests and plotting

    returns:
        - A trace of updated populations of design parameters
        - A trace of updated populations of exogenous parameters
        - A trace of overall log probabilities for the design parameters (before the
            following update to the exogenous parameters)
        - A trace of overall log probabilities for the exogenous parameters
    """
    # Provide defaults
    if tempering_schedule is None:
        tempering_schedule = jnp.ones(num_rounds)

    # Make the function to run one round of repair/predict
    @jax.jit
    def one_smc_step(carry, scan_input):
        # Unpack the input
        prng_key, tempering = scan_input
        # Unpack the carry
        i, current_dps, current_eps = carry

        # Set the outputs to default values
        dp_logprobs, ep_logprobs = jnp.array(0.0), jnp.array(0.0)
        dp_accept_rate, ep_accept_rate = jnp.array(0.0), jnp.array(0.0)
        dp_debug, ep_debug = jnp.array(0.0), jnp.array(0.0)

        # Update the iteration index
        i += 1

        # If we're in the last round, turn off stochasticity if we're quenching
        ep_stochasticity = jax.lax.cond(
            jnp.logical_and(not quench_dps_only, i > num_rounds - quench_rounds),
            lambda: False,
            lambda: use_stochasticity,
        )
        dp_stochasticity = jax.lax.cond(
            i > num_rounds - quench_rounds,
            lambda: False,
            lambda: use_stochasticity,
        )

        ###############################################
        # Repair the currently predicted failure modes
        ###############################################
        if repair:
            # Create a loglikelihood function to minimize the average potential across
            # all currently predicted failure modes.
            dp_mean_potential_fn = lambda dp: jax.vmap(
                dp_potential_fn, in_axes=(None, 0)
            )(dp, current_eps).mean()
            dp_logprob_fn = lambda dp: dp_logprior_fn(
                dp
            ) + tempering * dp_mean_potential_fn(dp)

            # Initialize the chains for this kernel
            initial_dp_sampler_states = jax.vmap(init_sampler, in_axes=(0, None))(
                current_dps, dp_logprob_fn
            )

            # Make the sampling kernel
            example_dp = jtu.tree_map(lambda leaf: leaf[0], current_dps)
            dp_kernel = make_kernel(
                example_dp, dp_logprob_fn, dp_mcmc_step_size, dp_stochasticity
            )

            # Run the chains and update the design parameters
            n_chains = initial_dp_sampler_states.logdensity.shape[0]
            prng_key, sample_key = jrandom.split(prng_key)
            sample_keys = jrandom.split(sample_key, n_chains)
            current_dps, dp_logprobs, dp_accept_rate, dp_debug = jax.vmap(
                run_chain, in_axes=(0, None, 0, None)
            )(
                sample_keys,
                dp_kernel,
                initial_dp_sampler_states,
                num_mcmc_steps_per_round,
            )

        ################################################
        # Predict failures for current design parameters
        ################################################

        if predict:
            # Create a loglikelihood function that maximizes the minimum potential
            # across all current design parameters. Need to make this positive so that
            # large potentials/costs -> higher likelihoods
            # ep_mean_potential_fn = lambda ep: softmin(
            #     jax.vmap(ep_potential_fn, in_axes=(0, None))(current_dps, ep),
            #     sharpness=0.05,
            # )
            ep_mean_potential_fn = lambda ep: jax.vmap(
                ep_potential_fn, in_axes=(0, None)
            )(
                current_dps, ep
            ).mean()  # TODO min or mean?
            # pick_one_dps = jtu.tree_map(lambda leaf: leaf[0], current_dps)
            # ep_mean_potential_fn = lambda ep: ep_potential_fn(pick_one_dps, ep)  # TODO

            ep_logprob_fn = lambda ep: ep_logprior_fn(
                ep
            ) + tempering * ep_mean_potential_fn(ep)

            # Initialize the chains for this kernel
            initial_ep_sampler_states = jax.vmap(init_sampler, in_axes=(0, None))(
                current_eps, ep_logprob_fn
            )

            # Make the sampling kernel
            example_ep = jtu.tree_map(lambda leaf: leaf[0], current_eps)
            ep_kernel = make_kernel(
                example_ep, ep_logprob_fn, ep_mcmc_step_size, ep_stochasticity
            )

            # Run the chains and update the design parameters
            n_chains = initial_ep_sampler_states.logdensity.shape[0]
            prng_key, sample_key = jrandom.split(prng_key)
            sample_keys = jrandom.split(sample_key, n_chains)
            current_eps, ep_logprobs, ep_accept_rate, ep_debug = jax.vmap(
                run_chain, in_axes=(0, None, 0, None)
            )(
                sample_keys,
                ep_kernel,
                initial_ep_sampler_states,
                num_mcmc_steps_per_round,
            )

        # Return the updated params and logprobs (but only carry the index and params)
        output = (
            i,
            current_dps,
            current_eps,
            dp_logprobs,
            ep_logprobs,
            dp_accept_rate,
            ep_accept_rate,
            {"dp_debug": dp_debug, "ep_debug": ep_debug},
        )
        return output[:3], output

    # Make a function to test the current best DP on the stress test set
    @jax.jit
    def stress_test_dps(dp, eps):
        costs = jax.vmap(potential_fn, in_axes=(None, 0))(dp, eps)
        return costs

    # Log some metrics for the initial (random) solutions
    log_dict = {}
    # Choose a set of DPs arbitrarily to log for the first iteration
    current_best_dps = jtu.tree_map(lambda leaf: leaf[0], design_params)

    predicted_costs = stress_test_dps(current_best_dps, exogenous_params)
    log_dict["Mean Cost/predicted"] = predicted_costs.mean()
    log_dict["Min Cost/predicted"] = predicted_costs.min()
    log_dict["Max Cost/predicted"] = predicted_costs.max()
    log_dict["Std Cost/predicted"] = predicted_costs.std()
    log_dict["Failure rate/predicted"] = (predicted_costs > failure_level).mean()
    qs = jnp.array([25, 50, 75, 90, 95, 99, 99.9, 100])
    percentiles = jnp.nanpercentile(predicted_costs, qs)
    log_dict = log_dict | {
        f"Predicted Cost Percentiles/{q:.2f}": cost for q, cost in zip(qs, percentiles)
    }

    if stress_test_cases is not None:
        # Stress test current best DP
        stress_test_costs = stress_test_dps(current_best_dps, stress_test_cases)

        log_dict = log_dict | {
            "Mean Cost/test": stress_test_costs.mean(),
            "Min Cost/test": stress_test_costs.min(),
            "Max Cost/test": stress_test_costs.max(),
            "Std Cost/test": stress_test_costs.std(),
            "Failure rate/test": (stress_test_costs > failure_level).mean(),
        }

        percentiles = jnp.nanpercentile(stress_test_costs, qs)
        log_dict = log_dict | {
            f"Test Cost Percentiles/{q:.2f}": cost for q, cost in zip(qs, percentiles)
        }

    if plotting_cb is not None:
        plotting_cb(current_best_dps, exogenous_params)

    # Log to wandb
    wandb.log(log_dict, commit=True)

    # Run the appropriate number of steps
    carry = (0, design_params, exogenous_params)
    keys = jrandom.split(prng_key, num_rounds)
    results = []
    for i, (key, tempering) in enumerate(zip(tqdm(keys), tempering_schedule)):
        # Run the SMC round to update both the design and exogenous parameters
        # print(f"{logging_prefix} - Iteration {i}")
        start = time.perf_counter()
        carry, y = one_smc_step(carry, (key, tempering))
        end = time.perf_counter()
        (
            _,
            current_dps,
            current_eps,
            dp_logprobs,
            ep_logprobs,
            dp_accept_rate,
            ep_accept_rate,
            debug_info,
        ) = y

        # Log the results to wandb
        log_dict = {
            "Step time (s)": end - start,
            "DP Accept Rate": dp_accept_rate.mean(),
            "EP Accept Rate": ep_accept_rate.mean(),
            "Mean DP Logprob": dp_logprobs.mean(),
            "Mean EP Logprob": ep_logprobs.mean(),
        }

        # Stress test and plot the current best design parameters
        most_likely_dps_idx = jnp.argmax(dp_logprobs)
        current_best_dps = jtu.tree_map(
            lambda leaf: leaf[most_likely_dps_idx], current_dps
        )
        if stress_test_cases is not None and (
            i == num_rounds - 1 or i % test_every == 0
        ):
            # Stress test current best DP
            stress_test_costs = stress_test_dps(current_best_dps, stress_test_cases)

            log_dict = log_dict | {
                "Mean Cost/test": stress_test_costs.mean(),
                "Min Cost/test": stress_test_costs.min(),
                "Max Cost/test": stress_test_costs.max(),
                "Std Cost/test": stress_test_costs.std(),
                "Failure rate/test": (stress_test_costs > failure_level).mean(),
            }
            percentiles = jnp.nanpercentile(stress_test_costs, qs)
            log_dict = log_dict | {
                f"Test Cost Percentiles/{q:.2f}": cost
                for q, cost in zip(qs, percentiles)
            }

        # Log the failure rate on the predicted test cases
        predicted_costs = stress_test_dps(current_best_dps, current_eps)
        log_dict["Mean Cost/predicted"] = predicted_costs.mean()
        log_dict["Min Cost/predicted"] = predicted_costs.min()
        log_dict["Max Cost/predicted"] = predicted_costs.max()
        log_dict["Std Cost/predicted"] = predicted_costs.std()
        log_dict["Failure rate/predicted"] = (predicted_costs > failure_level).mean()
        percentiles = jnp.nanpercentile(predicted_costs, qs)
        log_dict = log_dict | {
            f"Predicted Cost Percentiles/{q:.2f}": cost
            for q, cost in zip(qs, percentiles)
        }

        if plotting_cb is not None and (i == num_rounds - 1 or i % test_every == 0):
            plotting_cb(current_best_dps, current_eps)

        # Print some results to the console
        try:
            log_dict["EP Grad Norm"] = debug_info["ep_debug"]["final_grad_norm"].mean()
        except:
            pass  # No grad norm info, so don't log it

        try:
            log_dict["DP Grad Norm"] = debug_info["dp_debug"]["final_grad_norm"].mean()
        except:
            pass  # No grad norm info, so don't log it

        # Log to wandb
        wandb.log(log_dict, commit=True)

        results.append(y)

    # The python for loop above is faster than this scan, since we skip a long
    # compilation
    # _, results = jax.lax.scan(one_smc_step, carry, (keys, tempering_schedule))

    # Stack the results as a pytree
    leaves_list = []
    treedef_list = []
    for tree in results:
        leaves, treedef = jtu.tree_flatten(tree)
        leaves_list.append(leaves)
        treedef_list.append(treedef)

    grouped_leaves = zip(*leaves_list)
    result_leaves = [jnp.stack(leaf) for leaf in grouped_leaves]
    results = treedef_list[0].unflatten(result_leaves)

    # Unpack the results and return
    (
        _,
        design_params,
        exogenous_params,
        dp_logprobs,
        ep_logprobs,
        dp_accept,
        ep_accept,
        _,
    ) = results
    return design_params, exogenous_params, dp_logprobs, ep_logprobs
