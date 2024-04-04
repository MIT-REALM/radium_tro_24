import jax
import jax.numpy as jnp
import jax.random as jrandom
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from radium.engines.reinforce import init_sampler, make_kernel


# Define a test likelihood
@jaxtyped(typechecker=beartype)
def quadratic_potential(x: Float[Array, " n"]):
    return -0.5 * (x**2).sum()


def test_init_sampler():
    """Test the init_sampler function."""
    n = 4
    position = jnp.ones(n)
    state = init_sampler(position, quadratic_potential)

    # State should be initialized correctly
    assert jnp.allclose(state.position, position)
    assert jnp.allclose(state.baseline, 0.0)


def test_make_kernel():
    """Test the make_kernel function for the REINFORCE optimizer."""
    step_size = 1e-2
    perturbation_stddev = 0.1
    baseline_update_rate = 0.5

    # Scale up the potential to make the test more consistent
    potential = lambda x: quadratic_potential(x)

    kernel = make_kernel(
        potential, step_size, perturbation_stddev, baseline_update_rate
    )

    # Kernel should be initialized correctly
    assert kernel is not None

    # We should be able to call the kernel
    n = 2
    position = jnp.ones(n)
    state = init_sampler(position, potential)
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
    _, states = jax.lax.scan(one_step, next_state, keys)

    assert jnp.allclose(
        states.position[n_steps:].mean(axis=0), jnp.zeros(n), atol=tolerance
    )
