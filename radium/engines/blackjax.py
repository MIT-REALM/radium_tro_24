"""Implement an interface to `BlackJax <https://blackjax-devs.github.io/blackjax/>`_."""

import blackjax
import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import NamedTuple, Tuple, Union
from jaxtyping import Array, Float, Integer, jaxtyped

from radium.types import LogLikelihood, Params, PRNGKeyArray, Sampler


class BlackJaxState(NamedTuple):
    """A struct for storing the state of a BlackJax-derived optimizer/sampler.

    Attributes:
        position: the current solution.
        logdensity: the log-density of the current solution.
        logdensity_grad: the gradient of the log-density of the current solution.
        num_accepts: the number of accepted proposals.
        blackjax_state: the state of the wrapped BlackJax sampler.
    """

    position: Params
    logdensity: Float[Array, ""]
    logdensity_grad: Params
    num_accepts: int
    blackjax_state: NamedTuple


@jaxtyped(typechecker=beartype)
def make_hmc_step_and_initial_state(
    logdensity_fn: LogLikelihood,
    position: Params,
    step_size: Union[float, Float[Array, ""]],
    num_integration_steps: Union[int, Integer[Array, ""]],
) -> Tuple[Sampler, BlackJaxState]:
    """Initialize the the sampler.

    Args:
        logdensity_fn: the non-normalized log likelihood function
        position: the initial solution
        step_size: the size of the steps to take
        num_steps: the number of integration steps to take for each new sample

    Returns:
        initial_state: The initial state of the sampler.
        step_fn: A function that takes a PRNG key and the current state
        and returns the next state of the optimizer, executing one step of
        the sampling algorithm.
    """
    # Start by computing the inverse mass matrix for this problem
    n_vars = jax.flatten_util.ravel_pytree(position)[0].size
    inverse_mass_matrix = jnp.ones(n_vars)

    sampler = blackjax.hmc(
        logdensity_fn,
        step_size,
        inverse_mass_matrix,
        num_integration_steps,
    )

    # Initialize the sampler
    initial_hmc_state = sampler.init(position)

    # Wrap the state in our own state struct
    logdensity_grad = jax.tree_util.tree_map(
        lambda x: -x, initial_hmc_state.potential_energy_grad
    )
    initial_state = BlackJaxState(
        position=position,
        logdensity=-initial_hmc_state.potential_energy,  # negate to get logdensity
        logdensity_grad=logdensity_grad,
        num_accepts=0,
        blackjax_state=initial_hmc_state,
    )

    # Define the step function wrapping blackjax.
    @jaxtyped(typechecker=beartype)
    def step(key: PRNGKeyArray, state: BlackJaxState) -> BlackJaxState:
        """Take one step.

        Args:
            key: a random number generator key.
            state: the current state of the sampler.

        Returns:
            A sample.
        """
        # Take a step
        (
            next_hmc_state,
            hmc_info,
        ) = sampler.step(  # pylint: disable=no-member
            key, state.blackjax_state
        )

        # Update the cumulative average acceptance rate
        new_num_accepts = (
            state.num_accepts + num_integration_steps * hmc_info.acceptance_probability
        )

        logdensity_grad = jax.tree_util.tree_map(
            lambda x: -x, next_hmc_state.potential_energy_grad
        )
        return BlackJaxState(
            position=next_hmc_state.position,
            logdensity=-next_hmc_state.potential_energy,
            logdensity_grad=logdensity_grad,
            num_accepts=new_num_accepts,
            blackjax_state=next_hmc_state,
        )

    return step, initial_state
