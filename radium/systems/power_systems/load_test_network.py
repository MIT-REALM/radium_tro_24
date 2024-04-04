import json
import os.path
import time

import jax
import jax.numpy as jnp
from beartype import beartype

from radium.systems.power_systems.acopf import ACOPF
from radium.systems.power_systems.acopf_types import (
    InterconnectionSpecification,
    NetworkSpecification,
    NetworkState,
)


@beartype
def load_test_network(
    case_name: str,
    p_failure: float = 0.01,
    g_stddev: float = 0.1,
    b_stddev: float = 0.1,
    penalty: float = 10.0,
    load_dispatchability: float = 0.3,
    maxsteps: int = 30,
) -> ACOPF:
    """
    Load a test network from the MATPOWER suite.

    All bus indices are zero-indexed, so 1 less than typically found in the MATLAB-heavy
    power systems literature :D

    args:
        case_name: the name of the case to load
        p_failure: the probability of any single line failing
        g_stddev: the standard deviation in line conductance
        b_stddev: the standard deviation in line susceptance
        penalty: penalty coefficient
        load_dispatchability: what fraction of demand we can dispatch.
        maxsteps: maximum number of GaussNewton steps to take when solving PF.
    """
    ############################
    # Load data from file
    ############################
    current_path = os.path.abspath(os.path.dirname(__file__))
    datafile_path = os.path.join(
        current_path, "example_system_data/" + case_name + ".json"
    )

    with open(datafile_path) as f:
        case_data = json.load(f)

    ############################
    # General information
    ############################
    base_MVA = case_data["baseMVA"]
    n_line = len(case_data["branch"]["from"])
    bus_voltage_limits = jnp.vstack(
        (jnp.array(case_data["bus"]["VMIN"]), jnp.array(case_data["bus"]["VMAX"]))
    ).T

    # Make sure buses are labelled correctly
    # TODO relabel them somehow
    bus_idxs = jnp.array(case_data["bus"]["i"])
    assert jnp.allclose(jnp.diff(bus_idxs), 1), "Buses should be labelled sequentially"

    ############################
    # Network/branch data
    ############################

    # Offset all bus indices to account for matlab 1-indexing
    lines = jnp.vstack((case_data["branch"]["from"], case_data["branch"]["to"])).T
    lines = lines - 1

    # Get line admittances
    line_resistances = jnp.array(case_data["branch"]["BR_R"])
    line_reactances = jnp.array(case_data["branch"]["BR_X"])
    # The line data is provided as impedance, so we need to convert to admittances
    line_impedance = line_resistances + 1j * line_reactances
    line_admittance = 1 / line_impedance
    line_conductances = line_admittance.real
    line_susceptances = line_admittance.imag

    # Get shunt admittances
    shunt_conductances = jnp.array(case_data["bus"]["GS"])
    shunt_susceptances = jnp.array(case_data["bus"]["BS"])
    # Normalize by base MVA (a quirk of this data format)
    shunt_conductances = shunt_conductances / base_MVA
    shunt_susceptances = shunt_susceptances / base_MVA

    # Get tap ratio (default is tap = 1 if 0 in the data)
    tap_ratios = jnp.ones(n_line)
    taps = jnp.array(case_data["branch"]["TAP"])
    tap_ratios = tap_ratios.at[jnp.nonzero(taps)[0]].set(taps[jnp.nonzero(taps)[0]])
    transformer_phase_shifts = jnp.array(case_data["branch"]["SHIFT"])

    # Get charging susceptances
    charging_susceptances = jnp.array(case_data["branch"]["BR_B"])

    # Add line failure parameters
    prob_line_failure = jnp.zeros(n_line) + p_failure
    line_conductance_stddev = jnp.zeros(n_line) + g_stddev
    line_susceptance_stddev = jnp.zeros(n_line) + b_stddev

    # Now we can load this data into our representation
    nominal_network_state = NetworkState(jnp.ones((n_line,)))
    network_spec = NetworkSpecification(
        nominal_network_state,
        lines,
        line_conductances,
        line_susceptances,
        shunt_conductances,
        shunt_susceptances,
        tap_ratios,
        transformer_phase_shifts,
        charging_susceptances,
        prob_line_failure,
        line_conductance_stddev,
        line_susceptance_stddev,
    )

    ############################
    # Generator data
    ############################
    # Subtract 1 to make up for matlab 1-indexing
    generator_buses = jnp.array(case_data["gen"]["bus"]) - 1

    # Load limit data
    P_gen_limits = jnp.vstack(
        (jnp.array(case_data["gen"]["PMIN"]), jnp.array(case_data["gen"]["PMAX"]))
    ).T
    Q_gen_limits = jnp.vstack(
        (jnp.array(case_data["gen"]["QMIN"]), jnp.array(case_data["gen"]["QMAX"]))
    ).T

    # normalize by the base unit
    P_gen_limits /= base_MVA
    Q_gen_limits /= base_MVA

    # load costs (only quadratic costs supported for now)
    cost_types = jnp.array(case_data["gen"]["cost_model"])
    polynomial_degree = jnp.array(case_data["gen"]["NCOST"]) - 1
    if jnp.any(cost_types != 2):
        raise NotImplementedError("Non-polynomial cost models not yet supported!")
    if jnp.any(polynomial_degree > 2):
        raise NotImplementedError(
            "Polynomial cost models beyond quadratic not yet supported!"
        )

    # Normalize the cost by 1 / base_MVA ** 2
    cost_model_coeffs = jnp.array(case_data["gen"]["coeffs"])
    P_quadratic_costs = cost_model_coeffs[:, 0]
    P_linear_costs = cost_model_coeffs[:, 1] / base_MVA

    gen_spec = InterconnectionSpecification(
        buses=generator_buses,
        P_limits=P_gen_limits,
        Q_limits=Q_gen_limits,
        P_linear_costs=P_linear_costs,
        P_quadratic_costs=P_quadratic_costs,
    )

    ############################
    # Load data
    ############################

    # Add a load anywhere the case shows power demand
    # These cases don't include flexible loads, but let's add some dispatchability
    # to these loads. Also make negative to indicate a load
    P_load = jnp.array(case_data["bus"]["PD"])
    Q_load = jnp.array(case_data["bus"]["QD"])
    load_mask = jnp.logical_and(P_load != 0, Q_load != 0)
    scale = 1.0 - load_dispatchability
    P_load_limits = jnp.vstack((-P_load, -scale * P_load)).T[load_mask]
    Q_load_limits = jnp.vstack((-Q_load, -scale * Q_load)).T[load_mask]
    # We may have shuffled some of these maxes and mins, but that's easy to fix
    P_min, P_max = jnp.min(P_load_limits, axis=-1), jnp.max(P_load_limits, axis=-1)
    P_load_limits = jnp.vstack((P_min, P_max)).T
    Q_min, Q_max = jnp.min(Q_load_limits, axis=-1), jnp.max(Q_load_limits, axis=-1)
    Q_load_limits = jnp.vstack((Q_min, Q_max)).T

    # Normalize to base power
    P_load_limits /= base_MVA
    Q_load_limits /= base_MVA

    # Subtract 1 for matlab 1 indexing
    load_buses = jnp.array(case_data["bus"]["i"])[load_mask] - 1

    load_spec = InterconnectionSpecification(
        buses=load_buses,
        P_limits=P_load_limits,
        Q_limits=Q_load_limits,
        P_linear_costs=jnp.zeros(load_buses.shape[0]),
        P_quadratic_costs=jnp.zeros(load_buses.shape[0]),
    )

    ############################
    # Assemble and return
    ############################

    return ACOPF(
        gen_spec,
        load_spec,
        network_spec,
        bus_voltage_limits,
        ref_bus_idx=0,
        constraint_penalty=penalty,
        maxsteps=maxsteps,
        tolerance=1e-3,
    )


if __name__ == "__main__":
    # Cases that load and get residual << 1:
    # case14, case57, case118, case_ACTIVSg200
    # Cases that load but don't solve
    # case_ACTIVSg500
    case_name = "case_ACTIVSg200"
    sys = load_test_network(case_name)
    print(
        f"Successfully loaded {case_name} with {sys.n_bus} buses and {sys.n_line} lines"
    )
    key = jax.random.PRNGKey(0)
    dispatch = sys.sample_random_dispatch(key)
    network_state = sys.network_spec.nominal_network_state

    # Test without JIT
    start = time.perf_counter()
    r = sys(dispatch, network_state)
    end = time.perf_counter()
    print(
        (
            f"ACOPF completion residual {r.acopf_residual:.2e} "
            f"(took {end - start:.2f} s without JIT)"
        )
    )

    # Test with JIT
    completion_fn = lambda d, ns: sys(d, ns)
    start = time.perf_counter()
    jax.jit(completion_fn)(dispatch, network_state)
    end = time.perf_counter()
    print(f"First call with JIT took {end - start:.2f} s")

    N = 100
    start = time.perf_counter()
    for _ in range(N):
        jax.jit(completion_fn)(dispatch, network_state)
    end = time.perf_counter()
    print(f"Second call with JIT took {(end - start) / N:.3f} s")

    # Test with autodiff
    potential_fn = lambda d, ns: sys(d, ns).potential
    potential_and_grad_fn = jax.value_and_grad(potential_fn)
    start = time.perf_counter()
    potential_and_grad_fn(dispatch, network_state)
    end = time.perf_counter()
    print("Value and grad")
    print(f"First call without JIT took {end - start:.2f} s")
