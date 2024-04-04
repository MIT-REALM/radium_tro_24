"""Implement sampling algorithms"""

import operator

import jax
import jax.flatten_util
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
from beartype import beartype
from beartype.typing import NamedTuple, Union
from jaxtyping import Array, Bool, Float, Integer, jaxtyped

from radium.types import LogLikelihood, Params, PRNGKeyArray, Sampler


# @jaxtyped(typechecker=beartype)  # Commented out until beartype 0.12.0 release (TODO@dawsonc)
class SamplerState(NamedTuple):
    """State for samplers."""

    position: Params
    logdensity: Float[Array, ""]
    logdensity_grad: Params
    num_accepts: Integer[Array, ""]


def normalize_gradients_with_threshold(
    gradients: Params,
    threshold: Union[float, Float[Array, ""]],
) -> Params:
    """
    Normalize the gradients to a threshold if they are larger than the threshold.

    args:
        gradients: the gradients to normalize
        threshold: the threshold to normalize to
    returns:
        the normalized gradients
    """
    # Compute the L2 norm of the gradients
    block_grad_norms_sq = jtu.tree_map(lambda x: jnp.sum(x**2), gradients)
    grad_norm = jnp.sqrt(jtu.tree_reduce(operator.add, block_grad_norms_sq))

    # Normalize the gradients if they are larger than the threshold
    normalized_gradients = jax.lax.cond(
        grad_norm > threshold,
        lambda: jtu.tree_map(lambda x: x * threshold / grad_norm, gradients),
        lambda: gradients,
    )

    return normalized_gradients


@jaxtyped(typechecker=beartype)
def weight_perturbation_gradient_estimator(
    logdensity_fn: LogLikelihood,
    position: Params,
    step_size: Union[float, Float[Array, ""]],
    key: PRNGKeyArray,
    n_samples: Union[int, Integer[Array, ""]] = 8,
) -> Params:
    """
    Estimate the gradient using a weight perturbation method.

    args:
        logdensity_fn: the non-normalized log likelihood function
        position: the position to estimate the gradient at
        step_size: the size of the step to take
        key: the random key to use for the perturbation
        n_samples: the number of samples to use in the estimate

    returns:
        the log density at the current position and the gradient estimate
    """
    # Compute the log density at the current position
    logdensity = logdensity_fn(position)

    # Make a function to get the gradient estimate from a single perturbation
    def single_perturbation_estimate(subkey):
        # Generate noise with the same shape as the position pytree
        p, unravel_fn = jax.flatten_util.ravel_pytree(position)
        sample = jrandom.normal(subkey, shape=p.shape, dtype=p.dtype)
        perturbation = unravel_fn(sample)

        # Compute the log density at the perturbed position
        perturbed_position = jtu.tree_map(
            lambda x, y: x + jnp.sqrt(2 * step_size) * y, position, perturbation
        )
        perturbed_logdensity = logdensity_fn(perturbed_position)

        # Compute the gradient estimate
        return jtu.tree_map(
            lambda x, y: (perturbed_logdensity - logdensity) * y,
            position,
            perturbation,
        )

    # Compute the gradient estimate by averaging over several perturbations
    keys = jrandom.split(key, n_samples)
    gradient_estimate = jax.vmap(single_perturbation_estimate)(keys)
    gradient_estimate = jtu.tree_map(lambda x: x.mean(axis=0), gradient_estimate)

    return logdensity, gradient_estimate


@jaxtyped(typechecker=beartype)
def init_sampler(
    position: Params,
    logdensity_fn: LogLikelihood,
    normalize_gradients: Union[bool, Bool[Array, ""]] = True,
    gradient_clip: Union[float, Float[Array, ""]] = float("inf"),
    estimate_gradients: bool = False,
) -> SamplerState:
    """
    Initialize a sampler.

    args:
        position: the initial position of the sampler
        logdensity_fn: the non-normalized log likelihood function
        normalize_gradients: if True, normalize gradients to have clipped L2 norm
        gradient_clip: maximum value to clip gradients
        estimate_gradients: if True, estimate gradients using a weight perturbation
    """
    # Generate noise with the same shape as the position pytree, so we can set the
    # gradient clipping threshold appropriately
    p, _ = jax.flatten_util.ravel_pytree(position)
    sample = jrandom.normal(jrandom.PRNGKey(0), shape=p.shape, dtype=p.dtype)
    noise_strength = jnp.linalg.norm(sample)

    grad_fn = jax.value_and_grad(logdensity_fn)

    if estimate_gradients:
        logdensity, logdensity_grad = weight_perturbation_gradient_estimator(
            logdensity_fn, position, 1e-3, jrandom.PRNGKey(0)
        )
    else:
        logdensity, logdensity_grad = grad_fn(position)

    # Drop nans
    logdensity_grad = jtu.tree_map(
        lambda grad: jnp.where(jnp.isnan(grad), 0.0, grad), logdensity_grad
    )

    # Normalize the gradients if requested
    logdensity_grad = jax.lax.cond(
        normalize_gradients,
        lambda: normalize_gradients_with_threshold(
            logdensity_grad, noise_strength * gradient_clip
        ),
        lambda: logdensity_grad,
    )

    return SamplerState(position, logdensity, logdensity_grad, jnp.array(0))


@jaxtyped(typechecker=beartype)
def make_kernel(
    logdensity_fn: LogLikelihood,
    step_size: Union[float, Float[Array, ""]],
    use_gradients: Union[bool, Bool[Array, ""]] = True,
    use_stochasticity: Union[bool, Bool[Array, ""]] = True,
    gradient_clip: Union[float, Float[Array, ""]] = float("inf"),
    normalize_gradients: Union[bool, Bool[Array, ""]] = True,
    use_mh: Union[bool, Bool[Array, ""]] = True,
    estimate_gradients: bool = False,
) -> Sampler:
    """
    Build a kernel for a sampling algorithm (either MALA, RMH, or MLE).

    Here's the behavior of this sampler as a function of the arguments:

    (inside each block indicates behavior with use_mh = True/False if behavior differs)

                                    use_gradients = True | use_gradients = False
                              //====================================================
    use_stochasticity = True  ||         MALA/ULA        |    RMH/random walk
    --------------------------||----------------------------------------------------
    use_stochasticity = False ||      Gradient descent   |     doesn't move
                              \\====================================================

    args:
        logdensity_fn: the non-normalized log likelihood function
        step_size: the size of the step to take
        use_gradients: if True, use gradients in the proposal and acceptance steps
        use_stochasticity: if True, add a Gaussian to the proposal and acceptance steps
        gradient_clip: maximum value to clip gradients
        normalize_gradients: if True, normalize gradients to have unit L2 norm
        use_mh: if True, use Metropolis-Hastings acceptance, otherwise always accept
        estimate_gradients: if True, estimate gradients using a weight perturbation
    """
    # A generic Metropolis-Hastings-style MCMC algorithm has 2 steps: a proprosal and
    # an accept/reject step.

    # Start by defining the proposal log likelihood
    @jaxtyped(typechecker=beartype)
    def transition_log_probability(
        state: SamplerState,
        new_state: SamplerState,
        step_size: Union[float, Float[Array, ""]],
    ):
        """Compute the log likelihood of proposing new_state starting in state."""
        # Based on the MALA implementation in Blackjax, but extended to allow
        # turning it into RMH/gradient descent

        # If we're not using gradients, zero out the gradient in a JIT compatible way
        grad = jax.lax.cond(
            use_gradients,
            lambda x: x,
            lambda x: jtu.tree_map(jnp.zeros_like, x),
            state.logdensity_grad,
        )

        theta = jtu.tree_map(
            lambda new_x, x, g: new_x - x - step_size * g,
            new_state.position,
            state.position,
            grad,
        )
        theta_dot = jtu.tree_reduce(
            operator.add, jtu.tree_map(lambda x: jnp.sum(x * x), theta)
        )

        transition_log_probability = -0.25 * (1.0 / step_size) * theta_dot
        return transition_log_probability

    @jaxtyped(typechecker=beartype)
    def one_step(prng_key: PRNGKeyArray, state: SamplerState) -> SamplerState:
        """Generate a new sample"""
        # Split the key
        proposal_key, estimate_key, acceptance_key = jrandom.split(prng_key, 3)

        # Propose a new state
        delta_x = jtu.tree_map(
            jnp.zeros_like,
            state.position,
        )

        # Add gradients to the proposal if using
        delta_x = jax.lax.cond(
            use_gradients,
            lambda dx: jtu.tree_map(
                lambda leaf, grad: leaf + step_size * grad,
                dx,
                state.logdensity_grad,
            ),
            lambda dx: dx,
            delta_x,
        )

        # Generate noise with the same shape as the position pytree
        p, unravel_fn = jax.flatten_util.ravel_pytree(state.position)
        sample = jrandom.normal(proposal_key, shape=p.shape, dtype=p.dtype)
        noise_strength = jnp.linalg.norm(sample)
        noise = unravel_fn(sample)

        # Add it to the change in state if using stochasticity
        delta_x = jax.lax.cond(
            use_stochasticity,
            lambda dx: jtu.tree_map(
                lambda leaf, d: leaf + jnp.sqrt(2 * step_size) * d,
                dx,
                noise,
            ),
            lambda dx: dx,
            delta_x,
        )

        # Make the state proposal
        new_position = jtu.tree_map(lambda x, dx: x + dx, state.position, delta_x)
        if estimate_gradients:
            new_logdensity, new_grad = weight_perturbation_gradient_estimator(
                logdensity_fn,
                new_position,
                step_size,
                estimate_key,
            )
        else:
            new_logdensity, new_grad = jax.value_and_grad(logdensity_fn)(new_position)

        # Drop nans
        new_grad = jtu.tree_map(
            lambda grad: jnp.where(jnp.isnan(grad), 0.0, grad), new_grad
        )

        # Normalize the gradients if requested.
        new_grad = jax.lax.cond(
            normalize_gradients,
            lambda: normalize_gradients_with_threshold(
                new_grad, noise_strength * gradient_clip
            ),
            lambda: new_grad,
        )

        new_state = SamplerState(
            new_position,
            new_logdensity,
            new_grad,
            state.num_accepts + 1,
        )

        # Accept/reject the state using the Metropolis-Hasting rule
        log_p_accept = (
            new_state.logdensity
            - state.logdensity
            + transition_log_probability(new_state, state, step_size)
            - transition_log_probability(state, new_state, step_size)
        )
        log_p_accept = jnp.where(jnp.isnan(log_p_accept), -jnp.inf, log_p_accept)
        p_accept = jnp.clip(jnp.exp(log_p_accept), a_max=1)
        # If we're not using stochasticity or MH has been disabled, then always accept
        do_accept = jax.lax.cond(
            jnp.logical_and(use_stochasticity, use_mh),
            lambda: jrandom.bernoulli(acceptance_key, p_accept),
            lambda: jnp.array(True),
        )

        accepted_state = jax.lax.cond(do_accept, lambda: new_state, lambda: state)
        return accepted_state

    return one_step
