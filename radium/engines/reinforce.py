"""Implement the reinforce optimization algorithm."""

import jax
import jax.flatten_util
import jax.random as jrandom
import jax.tree_util as jtu
from beartype import beartype
from beartype.typing import NamedTuple, Union
from jaxtyping import Array, Float, jaxtyped

from radium.types import LogLikelihood, Params, PRNGKeyArray, Sampler


# @jaxtyped(typechecker=beartype)  # Commented out until beartype 0.12.0 release (TODO@dawsonc)
class ReinforceState(NamedTuple):
    """State for reinforce optimizer."""

    position: Params
    baseline: Float[Array, ""]
    logdensity: Float[Array, ""]


@jaxtyped(typechecker=beartype)
def init_sampler(
    position: Params,
    logdensity_fn: LogLikelihood,
) -> ReinforceState:
    """
    Initialize the reinforce optimizer state.

    args:
        position: the initial position of the sampler
        logdensity_fn: the non-normalized log likelihood function (not used)
    """
    # The original "learning to collide" paper does not use a baseline, but let's add
    # it to be sporting.
    logdensity = logdensity_fn(position)
    return ReinforceState(position, baseline=logdensity, logdensity=logdensity)


@jaxtyped(typechecker=beartype)
def make_kernel(
    logdensity_fn: LogLikelihood,
    step_size: Union[float, Float[Array, ""]],
    perturbation_stddev: Union[float, Float[Array, ""]],
    baseline_update_rate: Union[float, Float[Array, ""]],
) -> Sampler:
    """
    Build a kernel for the reinforce algorithm.

    args:
        logdensity_fn: the non-normalized log likelihood function
        step_size: the size of the step to take
        perturbation_stddev: the standard deviation of the perturbation to add to the
            solution at each step.
        baseline_update_rate: the rate at which to update the moving average baseline
    """

    @jaxtyped(typechecker=beartype)
    def one_step(prng_key: PRNGKeyArray, state: ReinforceState) -> ReinforceState:
        """Generate a new sample"""
        # Generate noise with the same shape as the position pytree
        p, unravel_fn = jax.flatten_util.ravel_pytree(state.position)
        sample = perturbation_stddev * jrandom.normal(
            prng_key, shape=p.shape, dtype=p.dtype
        )
        noise = unravel_fn(sample)

        # Perturb the state using this noise
        new_position = jtu.tree_map(lambda x, dx: x + dx, state.position, noise)

        # Get the log density at the new position (we want to maximize this)
        logdensity = logdensity_fn(new_position)

        # Compute the advantage by subtracting the baseline
        advantage = logdensity - state.baseline

        # Step in the direction of increased advantage
        new_position = jtu.tree_map(
            # lambda x, dx: x + step_size / perturbation_stddev**2 * advantage * dx,
            lambda x, dx: x + step_size * advantage * dx,
            state.position,
            noise,
        )

        # Update the baseline using a moving average
        updated_baseline = (
            1 - baseline_update_rate
        ) * state.baseline + baseline_update_rate * logdensity

        # Compute the new logdensity at the new state (cheating for pure REINFORCE
        # but we need to be able to log this)
        new_logdensity = logdensity_fn(new_position)

        # Return the new state
        return ReinforceState(new_position, updated_baseline, new_logdensity)

    return one_step
