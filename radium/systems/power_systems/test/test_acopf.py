import jax.numpy as jnp
import jax.random as jrandom
import pytest

from radium.systems.power_systems.acopf import ACOPF
from radium.systems.power_systems.acopf_types import (
    Dispatch,
    GenerationDispatch,
    InterconnectionSpecification,
    LoadDispatch,
    NetworkSpecification,
    NetworkState,
)


@pytest.fixture
def test_system():
    """Define a 2-bus test system"""
    nominal_network_state = NetworkState(jnp.ones((1,)))
    network_spec = NetworkSpecification(
        nominal_network_state,
        lines=jnp.array([[0, 1]]),
        line_conductances=jnp.array([1.0]),
        line_susceptances=jnp.array([1.0]),
        shunt_conductances=jnp.array([0.1, 0.1]),
        shunt_susceptances=jnp.array([0.0, 0.0]),
        transformer_tap_ratios=jnp.array([1.0]),
        transformer_phase_shifts=jnp.array([0.0]),
        charging_susceptances=jnp.array([0.0]),
        prob_line_failure=jnp.array([0.1]),
        line_conductance_stddev=jnp.array([0.1]),
        line_susceptance_stddev=jnp.array([0.1]),
    )
    gen_spec = InterconnectionSpecification(
        buses=jnp.array([0]),
        P_limits=jnp.array([[0, 10.0]]),
        Q_limits=jnp.array([[-10.0, 10.0]]),
        P_linear_costs=jnp.array([1.0]),
        P_quadratic_costs=jnp.array([1.0]),
    )
    load_spec = InterconnectionSpecification(
        buses=jnp.array([1]),
        P_limits=jnp.array([[-10.0, 0.0]]),
        Q_limits=jnp.array([[0.0, 0.0]]),
        P_linear_costs=jnp.array([1.0]),
        P_quadratic_costs=jnp.array([0.0]),
    )
    bus_voltage_limits = jnp.array([[0.95, 1.05], [0.95, 1.05]])
    return ACOPF(
        gen_spec,
        load_spec,
        network_spec,
        bus_voltage_limits,
        ref_bus_idx=0,
        constraint_penalty=100.0,
    )


def test_ACOPF___post_init__(test_system: ACOPF):
    """Test the derived values"""
    assert test_system.n_bus == 2
    assert test_system.n_line == 1
    assert jnp.allclose(test_system.nongen_buses, jnp.array([False, True]))
    assert jnp.allclose(test_system.nonref_buses, jnp.array([False, True]))


def test_ACOPF___call__(test_system: ACOPF):
    """Test the forward evaluation"""
    dispatch = Dispatch(
        GenerationDispatch(
            P=jnp.array([4.0]),
            voltage_amplitudes=jnp.array([1.01]),
        ),
        LoadDispatch(
            P=jnp.array([-4.0]),
            Q=jnp.array([1.0]),
        ),
    )
    result = test_system(dispatch, test_system.network_spec.nominal_network_state)

    assert result is not None


def test_ACOPF_dispatch_prior_logprob(test_system: ACOPF):
    """Test the dispatch prior logprobability"""
    # Since the prior distribution is uniform, two different valid dispatches should
    # have the same log probability
    dispatch_1 = Dispatch(
        GenerationDispatch(
            P=jnp.array([4.0]),
            voltage_amplitudes=jnp.array([0.99]),
        ),
        LoadDispatch(
            P=jnp.array([-4.0]),
            Q=jnp.array([1.0]),
        ),
    )
    dispatch_2 = Dispatch(
        GenerationDispatch(
            P=jnp.array([6.0]),
            voltage_amplitudes=jnp.array([1.01]),
        ),
        LoadDispatch(
            P=jnp.array([-6.0]),
            Q=jnp.array([-1.0]),
        ),
    )

    logprob_1 = test_system.dispatch_prior_logprob(dispatch_1)
    logprob_2 = test_system.dispatch_prior_logprob(dispatch_2)

    assert jnp.allclose(logprob_1, logprob_2)


def test_ACOPF_network_state_prior_logprob(test_system: ACOPF):
    """Test the network prior logprobability"""
    # TODO make a more sophisticated test.
    # Test to make sure the function runs and returns the correct thing
    # (it's typeguarded so if we avoid an error here we're all good)
    logprob = test_system.network_state_prior_logprob(
        test_system.network_spec.nominal_network_state
    )
    assert logprob is not None


def test_ACOPF_sample_random_dispatch(test_system: ACOPF):
    """Test the dispatch sampling function"""
    # TODO make a more sophisticated test.
    # Test to make sure the function runs and returns the correct thing
    # (it's typeguarded so if we avoid an error here we're all good)
    dispatch = test_system.sample_random_dispatch(jrandom.PRNGKey(0))
    assert dispatch is not None


def test_ACOPF_sample_random_network_state(test_system: ACOPF):
    """Test the network sampling function"""
    # TODO make a more sophisticated test.
    # Test to make sure the function runs and returns the correct thing
    # (it's typeguarded so if we avoid an error here we're all good)
    network = test_system.sample_random_network_state(jrandom.PRNGKey(0))
    assert network is not None
