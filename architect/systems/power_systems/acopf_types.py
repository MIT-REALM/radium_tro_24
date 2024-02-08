import jax.experimental.sparse as jsparse
import jax.numpy as jnp
from beartype.typing import NamedTuple, Tuple
from jax.nn import relu, sigmoid
from jaxtyping import Array, Float, Integer  # jaxtyped,


# @jaxtyped(typechecker=beartype)  # Commented out until beartype 0.12.0 release (TODO@dawsonc)
class GenerationDispatch(NamedTuple):
    """
    Dispatched power and voltage amplitude for each generator

    args:
        P: real power injection at each generator bus
        voltage_amplitudes: voltage amplitude at each generator bus
    """

    P: Float[Array, " n_bus"]
    voltage_amplitudes: Float[Array, " n_bus"]


# @jaxtyped(typechecker=beartype)  # Commented out until beartype 0.12.0 release (TODO@dawsonc)
class LoadDispatch(NamedTuple):
    """
    Dispatched real and reactive power loads.

    args:
        P: real power demand at each load bus
        Q: reactive power demand at each load bus
    """

    P: Float[Array, " n_bus"]
    Q: Float[Array, " n_bus"]


# @jaxtyped(typechecker=beartype)  # Commented out until beartype 0.12.0 release (TODO@dawsonc)
class Dispatch(NamedTuple):
    """
    Combined generation and load dispatch

    args:
        gen: generation dispatch
        load: load dispatch
    """

    gen: GenerationDispatch
    load: LoadDispatch


# @jaxtyped(typechecker=beartype)  # Commented out until beartype 0.12.0 release (TODO@dawsonc)
class InterconnectionSpecification(NamedTuple):
    """
    Specification of interconnection buses, limits, and costs.

    Covers both generators (positive P limits) and loads (negative P limits).

    args:
        buses: the bus index of each generator
        P_limits: n_connected x 2 array of (min, max) active power limits at each
            generator bus
        Q_limits: n_connected x 2 array of (min, max) reactive power limits at each
            generator bus
        P_linear_costs: n_connected-element array of linear cost per p.u. active
            power at the bus (should be positive for both loads and generation)
        P_quadratic_costs: n_connected-element array of quadratic cost per p.u.
            active power squared at each bus (should be positive for generation and
            negative for loads).
    """

    buses: Float[Integer, " n_connected"]
    P_limits: Float[Array, "n_connected 2"]
    Q_limits: Float[Array, "n_connected 2"]

    P_linear_costs: Float[Array, " n_connected"]
    P_quadratic_costs: Float[Array, " n_connected"]

    @property
    def n(self) -> int:
        """Return the number of buses in this interconnection"""
        return self.buses.shape[0]

    def cost(self, P: Float[Array, " n_connected"]) -> Float[Array, ""]:
        """Compute the cost of the given dispatch."""
        quad_cost = self.P_quadratic_costs * P**2
        lin_cost = self.P_linear_costs * P
        return quad_cost.sum() + lin_cost.sum()

    def power_limit_violations(
        self, P: Float[Array, " n_connected"], Q: Float[Array, " n_connected"]
    ) -> Tuple[Float[Array, " n_connected"], Float[Array, " n_connected"]]:
        """Compute the P and Q limit violations at each connected bus"""
        # Active power limits
        P_min, P_max = self.P_limits.T
        P_violation = relu(P - P_max) + relu(P_min - P)

        # Reactive power limits
        Q_min, Q_max = self.Q_limits.T
        Q_violation = relu(Q - Q_max) + relu(Q_min - Q)

        return P_violation, Q_violation


# @jaxtyped(typechecker=beartype)  # Commented out until beartype 0.12.0 release (TODO@dawsonc)
class NetworkState(NamedTuple):
    """
    Representation of network connectivity for AC power flow.

    args:
        line_states: float indicating the strength of the corresponding line. Positive
            indicates that the line is nominal, negative indicates that it has failed.
    """

    line_states: Float[Array, " n_lines"]


# @jaxtyped(typechecker=beartype)  # Commented out until beartype 0.12.0 release (TODO@dawsonc)
class NetworkSpecification(NamedTuple):
    """
    Specification of network parameters.

    args:
        nominal_network_state: the nominal admittances of the network lines
        lines: n_lines x 2 array of integers specifying bus connections. If [i, j]
            is a row in this array, then buses i and j will be connected. Each
            connection should be specified only once (so if [i, j] is included,
            it is not necessary to include [j, i] as well)
        line_conductances: n_lines-element array of nominal conductances between buses.
            If [i, j] is the k-th row of `lines`, then the k-th entry of `conductances`
            contains the conductance connecting bus i and bus j.
        line_susceptances: n_lines-element array of nominal susceptances between buses.
            If [i, j] is the k-th row of `lines`, then the k-th entry of `susceptances`
            contains the susceptance connecting bus i and bus j.
        shunt_conductances: n_bus-element array of shunt conductances connecting
            buses to ground.
        shunt_susceptances: n_bus-element array of shunt susceptances connecting
            buses to ground.
        transformer_tap_ratios: the tap ratios of all lines (1 if no transformer)
        transformer_phase_shifts: the phase shifts of all lines (0 if no transformer)
            Units should be in radians.
        charging_susceptances: n_lines-element array of charging susceptances of lines.
        prob_line_failure: n_line-element array of prior probabilities of line failures
        line_conductance_stddev: n_line-element array of standard deviation of line
            conductances in both the nominal and failure cases.
        line_susceptance_stddev: n_line-element array of standard deviation of line
            susceptances in both the nominal and failure cases.
    """

    nominal_network_state: NetworkState
    lines: Integer[Array, "n_lines 2"]
    line_conductances: Float[Array, " n_lines"]
    line_susceptances: Float[Array, " n_lines"]
    shunt_conductances: Float[Array, " n_bus"]
    shunt_susceptances: Float[Array, " n_bus"]
    transformer_tap_ratios: Float[Array, " n_lines"]
    transformer_phase_shifts: Float[Array, " n_lines"]
    charging_susceptances: Float[Array, " n_lines"]

    prob_line_failure: Float[Array, " n_line"]
    line_conductance_stddev: Float[Array, " n_line"]
    line_susceptance_stddev: Float[Array, " n_line"]

    @property
    def n_bus(self) -> int:
        """Return the number of buses in the network"""
        return self.shunt_conductances.shape[0]

    @property
    def n_line(self) -> int:
        """Return the number of lines in the network"""
        return self.line_conductances.shape[0]

    def Y(self, network_state: NetworkState, smoothing: float = 2.0):
        """
        Compute the nodal admittance matrix for this network.

        args:
            network_state: state of all lines in the network
        """
        # Clip all line and shunt conductances to make sure they're non-negative
        line_conductances = jnp.clip(self.line_conductances, a_min=0.0)
        shunt_conductances = jnp.clip(self.shunt_conductances, a_min=0.0)

        # Assemble conductance and susceptance into complex admittance
        line_admittances = line_conductances + 1j * self.line_susceptances
        shunt_admittances = shunt_conductances + 1j * self.shunt_susceptances

        # Incorporate line failures
        line_strength = sigmoid(smoothing * network_state.line_states)
        line_admittances = line_admittances * line_strength

        # The following is based on the code in MATPOWER, PyPower, and Pandapower
        # see github.com/e2nIEE/pandapower/blob/develop/pandapower/pypower/makeYbus.py
        # and https://matpower.org/docs/manual.pdf

        # Assemble the branch admittance matrices, which give the current injections
        # at the from and to ends of each branch (If, It) as a function of the voltages
        # at the from and to ends (Vf, Vt)
        #
        #      | If |   | Yff  Yft |   | Vf |
        #      |    | = |          | * |    |
        #      | It |   | Ytf  Ytt |   | Vt |

        # Make complex tap ratio
        tap = self.transformer_tap_ratios * jnp.exp(1j * self.transformer_phase_shifts)

        # Assume no line charging susceptance
        Ytt = line_admittances + 0.5j * self.charging_susceptances
        Yff = Ytt / self.transformer_tap_ratios**2
        Yft = -line_admittances / jnp.conj(tap)
        Ytf = -line_admittances / tap

        # To go from branch admittances to bus admittances, we need connection matrices,
        # Cf (from connections) and Ct (to connections). These each have n_lines rows
        # and n_buses columns, and are zero everywhere except that the (i, j) element of
        # Cf and the (i, k) element of Ct are 1 iff line i connects from bus j to bus k
        from_b = self.lines[:, 0]
        to_b = self.lines[:, 1]
        # Turns out we don't need this (they can maybe be used in a performance boost?)
        # line_indices = jnp.arange(0, n_line).reshape(-1, 1)
        # Cf_indices = jnp.hstack((line_indices, from_b.reshape(-1, 1)))
        # Ct_indices = jnp.hstack((line_indices, to_b.reshape(-1, 1)))
        # Cf = jsparse.BCOO((jnp.ones(n_line), Cf_indices), shape=(n_line, n_bus))
        # Ct = jsparse.BCOO((jnp.ones(n_line), Ct_indices), shape=(n_line, n_bus))

        # Make the sparse bus admittance matrix by assembling all of the
        # branch admittances and the shunt admittances (duplicate indices are summed)
        diag_i = jnp.arange(0, self.n_bus)
        Y_bus = jsparse.BCOO(
            (
                jnp.concatenate((Yff, Yft, Ytf, Ytt, shunt_admittances)),  # data
                jnp.hstack(
                    (
                        # Row indices
                        jnp.concatenate((from_b, from_b, to_b, to_b, diag_i)).reshape(
                            -1, 1
                        ),
                        # Column indices
                        jnp.concatenate((from_b, to_b, from_b, to_b, diag_i)).reshape(
                            -1, 1
                        ),
                    )
                ),
            ),
            shape=(self.n_bus, self.n_bus),
        )

        return Y_bus


# @jaxtyped(typechecker=beartype)  # Commented out until beartype 0.12.0 release (TODO@dawsonc)
class ACOPFResult(NamedTuple):
    """
    Result of running the ACOPF algorithm.

    Includes:
        The dispatch and network used to compute this result
        The voltage amplitudes at all buses
        The voltage angles at all buses
        The reactive power demanded from each generator
        The total violation of reactive power limits for generators
        The total violation of voltage limits
        The net generation cost
        The augmented cost ("potential") = cost + penalty * total violation
        The residual that remains in the ACOPF flow constraints
    """

    dispatch: Dispatch
    network_state: NetworkState

    voltage_amplitudes: Float[Array, " n_bus"]
    voltage_angles: Float[Array, " n_bus"]

    Q_gen: Float[Array, " n_gen_bus"]
    P_ref: Float[Array, ""]
    Q_ref: Float[Array, ""]

    P_gen_violation: Float[Array, " n_gen_bus"]
    Q_gen_violation: Float[Array, " n_gen_bus"]
    P_load_violation: Float[Array, " n_load_bus"]
    Q_load_violation: Float[Array, " n_load_bus"]
    V_violation: Float[Array, " n_bus"]

    generation_cost: Float[Array, ""]
    potential: Float[Array, ""]

    acopf_residual: Float[Array, ""]
