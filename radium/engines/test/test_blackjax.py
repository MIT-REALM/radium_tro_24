import jax
import jax.numpy as jnp
import jax.random as jrandom
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from radium.engines.blackjax import make_hmc_step_and_initial_state


# Define a test likelihood
@jaxtyped(typechecker=beartype)
def quadratic_potential(x: Float[Array, " n"]):
    return -0.5 * (x**2).sum()


def test_make_hmc_step_and_initial_state():
    """Test the make_hmc_step_and_initial_state function for the HMC sampler."""
    step_size = 1e-2
    num_integration_steps = 32

    # Scale up the potential to make the test more consistent
    potential_fn = lambda x: 100 * quadratic_potential(x)

    n = 2
    position = jnp.ones(n)
    kernel, state = make_hmc_step_and_initial_state(
        potential_fn, position, step_size, num_integration_steps
    )

    # Kernel should be initialized correctly
    assert kernel is not None

    # We should be able to call the kernel
    prng_key = jrandom.PRNGKey(0)
    next_state = kernel(prng_key, state)
    assert next_state is not None

    # If we run the kernel for a while, we should get samples that average around the
    # minimum at x = 0
    n_steps = 1000
    tolerance = 1e-2

    @jax.jit
    def one_step(state, rng_key):
        state = kernel(rng_key, state)
        return state, state

    keys = jrandom.split(prng_key, n_steps * 2)
    _, states = jax.lax.scan(one_step, state, keys)

    assert jnp.allclose(
        states.position[n_steps:].mean(axis=0), jnp.zeros(n), atol=tolerance
    )
