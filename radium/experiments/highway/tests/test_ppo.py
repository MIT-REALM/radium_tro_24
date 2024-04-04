"""Test RL subroutines for training the highway agent."""
import jax.numpy as jnp

from radium.experiments.highway.train_highway_agent import (
    generalized_advantage_estimate,
)


def test_generalized_advantage_estimate():
    """Test the GAE implementation."""
    rewards = jnp.array([1.0, 2.0, 3.0])
    values = jnp.array([4.0, 5.0, 6.0, 7.0])
    gamma = 1.0
    lam = 1.0

    # Test GAE with no terminal state.
    dones = jnp.array([False, False, False])
    advantage, returns = generalized_advantage_estimate(
        rewards,
        values,
        dones,
        gamma,
        lam,
    )
    # computed by hand
    expected_returns = jnp.array([2.0, 3.0, 4.0]) + values[:-1]
    expected_advantage = jnp.array([9.0, 7.0, 4.0])
    assert jnp.allclose(returns, expected_returns)
    assert jnp.allclose(advantage, expected_advantage)

    # Test GAE with terminal state.
    dones = jnp.array([False, True, False])
    advantage, returns = generalized_advantage_estimate(
        rewards,
        values,
        dones,
        gamma,
        lam,
    )
    # computed by hand
    expected_returns = jnp.array([2.0, -3.0, 4.0]) + values[:-1]
    expected_advantage = jnp.array([-1.0, -3.0, 4.0])
    assert jnp.allclose(returns, expected_returns)
    assert jnp.allclose(advantage, expected_advantage)
