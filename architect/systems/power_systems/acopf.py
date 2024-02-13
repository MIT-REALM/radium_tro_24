from dataclasses import field

import diffrax
import equinox as eqx
import jax
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import jax.random as jrandom
from beartype import beartype
from beartype.typing import Dict, Tuple
from jax.nn import log_sigmoid, relu
from jaxtyping import Array, Bool, Float, jaxtyped

from architect.systems.power_systems.acopf_types import (
    ACOPFResult,
    Dispatch,
    GenerationDispatch,
    InterconnectionSpecification,
    LoadDispatch,
    NetworkSpecification,
    NetworkState,
)
from architect.types import PRNGKeyArray


@jaxtyped(typechecker=beartype)
class ACOPF(eqx.Module):
    """
    Defines an autonomous system modelling AC optimal power flow.

    Note: all buses not connected to a generator or load will be assumed to have
    zero active and reactive power injection.

    # TODO support thermal limits on lines

    args:
        gen_spec: specification of interconnected generators
        load_spec: specification of interconnected loads
        network_spec: specification of network structure
        bus_voltage_limits: bus voltage limits
        ref_bus_idx: index of reference/slack bus
        constraint_penalty: scalar penalty coefficient for violating power, voltage, or
            angle constraints
        tolerance: tolerance to use when solving power flow equation
        maxsteps: maximum number of steps to use when solving power flow equation
    """

    gen_spec: InterconnectionSpecification
    load_spec: InterconnectionSpecification
    network_spec: NetworkSpecification

    bus_voltage_limits: Float[Array, "n_bus 2"]

    ref_bus_idx: int

    constraint_penalty: float

    tolerance: float = 1e-3
    maxsteps: int = 10

    # Convenience fields: boolean indexing for subsets of buses
    nongen_buses: Bool[Array, "n_bus"] = field(init=False)
    nonref_buses: Bool[Array, "n_bus"] = field(init=False)

    def __post_init__(self):
        # Populate convenience fields
        self.nongen_buses = (
            jnp.ones(self.n_bus, dtype=bool).at[self.gen_spec.buses].set(False)
        )
        self.nonref_buses = (
            jnp.ones(self.n_bus, dtype=bool).at[self.ref_bus_idx].set(False)
        )

    @property
    def n_bus(self) -> int:
        """Return the number of buses in the network"""
        return self.network_spec.n_bus

    @property
    def n_line(self) -> int:
        """Return the number of lines in the network"""
        return self.network_spec.n_line

    @jaxtyped(typechecker=beartype)
    def power_injections(
        self,
        network_state: NetworkState,
        voltage_amplitudes: Float[Array, " n_bus"],
        voltage_angles: Float[Array, " n_bus"],
    ) -> Tuple[Float[Array, " n_bus"], Float[Array, " n_bus"]]:
        """
        Compute the real and reactive power injections at each bus.

        args:
            network_state: the network state to use to solve the injections
            voltage_amplitudes: voltage amplitudes at each bus
            voltage_angles: voltage angles at each bus

        returns:
            a tuple of real and reactive power injections at each bus
        """
        # Compute the complex voltage at each bus
        V = voltage_amplitudes * jnp.exp(voltage_angles * 1j)

        # Compute the complex current and power that results from the proposed voltages
        Y = self.network_spec.Y(network_state)
        current = jsparse.sparsify(lambda Y, V: Y @ V)(Y, V)
        # Y = Y.todense()
        # current = jnp.dot(Y, V)
        S = V * current.conj()

        # Extract real and reactive power from complex power
        P, Q = S.real, S.imag

        return P, Q

    @jaxtyped(typechecker=beartype)
    def generation_cost(self, dispatch: Dispatch) -> Float[Array, ""]:
        """
        Compute the cost of generation given power injections.
        """
        return self.gen_spec.cost(dispatch.gen.P) + self.load_spec.cost(dispatch.load.P)

    @jaxtyped(typechecker=beartype)
    def constraint_violations(
        self,
        P_gen: Float[Array, " n_gen"],
        Q_gen: Float[Array, " n_gen"],
        P_load: Float[Array, " n_load"],
        Q_load: Float[Array, " n_load"],
        voltage_amplitudes: Float[Array, " n_bus"],
    ) -> Dict[str, Float[Array, "..."]]:
        """
        Compute constraint violations.

        args:
            P_gen: active power generation at generators
            Q_gen: reactive power generation at generators
            P_load: active power generation at loads
            Q_load: reactive power generation at loads
            voltage_amplitudes: voltage amplitude at all buses
        returns:
            a dict of constraint violations:
                - active power at each generator
                - reactive power at each generator
                - active power at each load
                - reactive power at each load
                - voltage amplitude at each bus
        """
        # Power limits
        P_gen_err, Q_gen_err = self.gen_spec.power_limit_violations(P_gen, Q_gen)
        P_load_err, Q_load_err = self.load_spec.power_limit_violations(P_load, Q_load)

        # Voltage amplitude limits
        V_min, V_max = self.bus_voltage_limits.T
        V = voltage_amplitudes
        V_violation = relu(V - V_max) + relu(V_min - V)

        return {
            "P_gen": P_gen_err,
            "Q_gen": Q_gen_err,
            "P_load": P_load_err,
            "Q_load": Q_load_err,
            "V": V_violation,
        }

    @jaxtyped(typechecker=beartype)
    def complete_voltages(
        self,
        network_state: NetworkState,
        P_gen: Float[Array, " n_gen"],
        V_gen: Float[Array, " n_gen"],
        P_load: Float[Array, " n_load"],
        Q_load: Float[Array, " n_load"],
    ) -> Tuple[
        Float[Array, " n_nongen_bus"], Float[Array, " n_nonref_bus"], Float[Array, ""]
    ]:
        """
        Solve for free voltage amplitudes and angles.

        args:
            network_state: the network to use to solve the power flow
            P_gen: generator active power injection
            V_gen: generator active power injection
            P_load: load active power injection
            Q_load: load reactive power injection
        returns:
            - voltage amplitudes at all non-generation buses
            - voltage angles at all non-reference buses
        """
        # Define a function that computes the residuals of the power flow equation
        # (we need to pass in all the relevant arguments so we can use implicit
        # differentiation)
        n_nongen = self.n_bus - self.gen_spec.n
        n_nonref = self.n_bus - 1

        # Compute the static part of the residual power injections
        P_residual = jnp.zeros(self.n_bus)
        P_residual = P_residual.at[self.gen_spec.buses].add(-P_gen)
        P_residual = P_residual.at[self.load_spec.buses].add(-P_load)
        Q_residual = jnp.zeros(self.n_bus)
        Q_residual = Q_residual.at[self.load_spec.buses].add(-Q_load)

        # Pre-populate the static voltage amplitudes and angles
        voltage_amplitudes = jnp.zeros(self.n_bus)
        voltage_amplitudes = voltage_amplitudes.at[self.gen_spec.buses].set(V_gen)
        voltage_angles = jnp.zeros(self.n_bus)

        def power_flow_residuals(x, args):
            # Extract parameters
            (
                network_state,
                voltage_amplitudes,
                voltage_angles,
                P_residual,
                Q_residual,
            ) = args

            # Extract voltage amplitudes and angles from the inputs
            V_nongen = x[:n_nongen]
            voltage_angles_nonref = x[n_nongen:]

            # Combine these amplitudes and angles with the known values
            voltage_amplitudes = voltage_amplitudes.at[self.nongen_buses].set(V_nongen)
            voltage_angles = voltage_angles.at[self.nonref_buses].set(
                voltage_angles_nonref
            )

            # Get the power injections implied by these voltages
            P_implied, Q_implied = self.power_injections(
                network_state, voltage_amplitudes, voltage_angles
            )

            # Compute the difference between these power injections and the dispatched
            # power injections. We want the active power residual to be zero at all
            # non-reference buses, and we want the reactive power residual to be zero at
            # all load buses
            P_residual_i = P_residual + P_implied
            Q_residual_i = Q_residual + Q_implied

            # Filter down to the buses we care about at this stage. We need to balance
            # active power at all buses except the slack buses, and we need to balance
            # reactive power at all buses without generators
            P_residual_i = P_residual_i[self.nonref_buses]
            Q_residual_i = Q_residual_i[self.nongen_buses]

            # Stack the residuals to minimize
            residuals = jnp.hstack((P_residual_i, Q_residual_i))

            return residuals

        # Solve the nonlinear least squares problem
        V_nongen_init = jnp.ones(n_nongen)
        voltage_angles_nonref_init = jnp.zeros(n_nonref)
        x_init = jnp.hstack((V_nongen_init, voltage_angles_nonref_init))
        solver = diffrax.NewtonNonlinearSolver(
            rtol=self.tolerance, atol=self.tolerance, max_steps=self.maxsteps
        )
        sol = solver(
            power_flow_residuals,
            x_init,
            (
                network_state,
                voltage_amplitudes,
                voltage_angles,
                P_residual,
                Q_residual,
            ),
        )

        # Extract solution
        x_sol = sol.root
        V_nongen = x_sol[:n_nongen]
        voltage_angles_nonref = x_sol[n_nongen:]
        acopf_residual = (
            power_flow_residuals(
                x_sol,
                (
                    network_state,
                    voltage_amplitudes,
                    voltage_angles,
                    P_residual,
                    Q_residual,
                ),
            )
            ** 2
        ).sum()

        return V_nongen, voltage_angles_nonref, acopf_residual

    @jaxtyped(typechecker=beartype)
    def complete_powers(
        self,
        network_state: NetworkState,
        P_gen: Float[Array, " n_gen"],
        P_load: Float[Array, " n_load"],
        Q_load: Float[Array, " n_load"],
        voltage_amplitudes: Float[Array, " n_bus"],
        voltage_angles: Float[Array, " n_bus"],
    ) -> Tuple[Float[Array, " n_gen"], Float[Array, ""], Float[Array, ""]]:
        """
        Solve for the free power injections.

        args:
            network_state: the network to use to solve the power flow
            P_gen: generator active power injection
            P_load: load active power injection
            Q_load: load reactive power injection
            voltage_amplitudes: voltage amplitudes at each bus
            voltage_angles: voltage angles at each bus

        returns:
            - Reactive power at each generator
            - Active power injection at reference bus
            - Reactive power injection at reference bus
        """
        # Make a forward pass through the power flow equations to get the net power
        # injections at each bus
        P, Q = self.power_injections(network_state, voltage_amplitudes, voltage_angles)

        # Subtract known power injections
        P = P.at[self.gen_spec.buses].add(-P_gen)
        P = P.at[self.load_spec.buses].add(-P_load)
        Q = Q.at[self.load_spec.buses].add(-Q_load)

        # Get reactive power at each generator
        Q_gen = Q[self.gen_spec.buses]

        # Get injection from the reference bus (whatever's leftover)
        P_ref = P[self.ref_bus_idx]
        Q_ref = Q[self.ref_bus_idx]

        return Q_gen, P_ref, Q_ref

    @jaxtyped(typechecker=beartype)
    def __call__(
        self,
        dispatch: Dispatch,
        network_state: NetworkState,
    ) -> ACOPFResult:
        """
        Computes the power flows associated with the proposed bus voltages and angles.

        args:
            dispatch: the voltage dispatch to use when solving power flow
            network_state: the network to use for solving power flow
            repair_steps: how many steps to take to correct infeasible dispatches
            repair_lr: the size of the repair steps
        returns:
            an ACOPFResult including the repaired dispatch
        """
        # The dispatch includes:
        #   - P and Q at each load
        #   - P and |V| at each generator
        P_gen = dispatch.gen.P
        V_gen = dispatch.gen.voltage_amplitudes
        P_load = dispatch.load.P
        Q_load = dispatch.load.Q

        # We need to complete the dispatch to get
        #   - |V| at the rest of the buses
        #   - voltage angle at all non-reference buses
        #   - Q at each generator
        #   - P injection at reference bus

        # We can do this completion in 2 steps.

        # First, solve the nonlinear power flow equations for a subset of the variables
        #   - |V| at the non-generator buses
        #   - voltage angle at all non-reference buses
        # also track any error in the ACOPF power flow
        V_nongen, voltage_angles_nonref, acopf_residual = self.complete_voltages(
            network_state, P_gen, V_gen, P_load, Q_load
        )

        # Save these into some convenient arrays
        voltage_amplitudes = jnp.zeros(self.n_bus)
        voltage_amplitudes = voltage_amplitudes.at[self.gen_spec.buses].set(
            dispatch.gen.voltage_amplitudes
        )
        voltage_amplitudes = voltage_amplitudes.at[self.nongen_buses].set(V_nongen)

        voltage_angles = jnp.zeros(self.n_bus)
        voltage_angles = voltage_angles.at[self.nonref_buses].set(voltage_angles_nonref)

        # Second, evaluate the power flow equations to get:
        #   - Q at each generator
        #   - Q and P injected from reference bus
        Q_gen, P_ref, Q_ref = self.complete_powers(
            network_state, P_gen, P_load, Q_load, voltage_amplitudes, voltage_angles
        )

        # Compute constraint violations
        constraint_violations = self.constraint_violations(
            P_gen, Q_gen, P_load, Q_load, voltage_amplitudes
        )
        total_violation = sum([v.sum() for v in constraint_violations.values()])

        # Compute the net cost of generation
        generation_cost = self.generation_cost(dispatch)

        # Combine cost and violations to get the potential
        potential = (
            generation_cost
            + self.constraint_penalty * jax.nn.sigmoid(10 * total_violation)
            # + acopf_residual  # Including this makes training unstable
        )

        result = ACOPFResult(
            dispatch,
            network_state,
            voltage_amplitudes,
            voltage_angles,
            Q_gen,
            P_ref,
            Q_ref,
            constraint_violations["P_gen"],
            constraint_violations["Q_gen"],
            constraint_violations["P_load"],
            constraint_violations["Q_load"],
            constraint_violations["V"],
            generation_cost,
            potential,
            acopf_residual,
        )

        return result

    @jaxtyped(typechecker=beartype)
    def dispatch_prior_logprob(self, dispatch: Dispatch) -> Float[Array, ""]:
        """
        Compute the prior log probability of the given dispatch.

        Probability is not necessarily normalized.

        We assume that the prior distribution for the dispatch is uniform within its
        limits.

        We use a smooth approximation of the uniform distribution.
        """
        # Define the smoothing constant
        b = 50.0

        def log_smooth_uniform(x, x_min, x_max):
            return log_sigmoid(b * (x - x_min)) + log_sigmoid(b * (x_max - x))

        P_min, P_max = self.gen_spec.P_limits.T
        logprob_P_gen = log_smooth_uniform(dispatch.gen.P, P_min, P_max).sum()
        V_min, V_max = self.bus_voltage_limits[self.gen_spec.buses].T
        logprob_V = log_smooth_uniform(
            dispatch.gen.voltage_amplitudes, V_min, V_max
        ).sum()

        P_min, P_max = self.load_spec.P_limits.T
        logprob_P_load = log_smooth_uniform(dispatch.load.P, P_min, P_max).sum()
        Q_min, Q_max = self.load_spec.Q_limits.T
        logprob_Q_load = log_smooth_uniform(dispatch.load.Q, Q_min, Q_max).sum()

        logprob = logprob_V + logprob_P_gen + logprob_P_load + logprob_Q_load

        return logprob

    @jaxtyped(typechecker=beartype)
    def sample_random_dispatch(self, key: PRNGKeyArray) -> Dispatch:
        """
        Sample a random dispatch from the uniform distribution within its limits

        args:
            key: PRNG key to use for sampling
        """
        V_key, P_gen_key, P_load_key, Q_load_key = jrandom.split(key, 4)

        P_min, P_max = self.gen_spec.P_limits.T
        P_gen = jrandom.uniform(
            P_gen_key, shape=(self.gen_spec.n,), minval=P_min, maxval=P_max
        )
        V_min, V_max = self.bus_voltage_limits[self.gen_spec.buses].T
        V_gen = jrandom.uniform(
            V_key, shape=(self.gen_spec.n,), minval=V_min, maxval=V_max
        )

        P_min, P_max = self.load_spec.P_limits.T
        P_load = jrandom.uniform(
            P_load_key, shape=(self.load_spec.n,), minval=P_min, maxval=P_max
        )
        Q_min, Q_max = self.load_spec.Q_limits.T
        Q_load = jrandom.uniform(
            Q_load_key, shape=(self.load_spec.n,), minval=Q_min, maxval=Q_max
        )

        return Dispatch(
            GenerationDispatch(P_gen, V_gen),
            LoadDispatch(P_load, Q_load),
        )

    @jaxtyped(typechecker=beartype)
    def network_state_prior_logprob(
        self, network_state: NetworkState
    ) -> Float[Array, ""]:
        """
        Compute the prior log probability of the given network state.

        Line states are positive in the absence of failure and negative when failure
        occurs. We can represent this as a Gaussian distribution with the mean shifted
        so that there is network_spec.prob_line_failure probability mass for negative
        values.

        Failures are assumed to be indepedendent. Probability density is not necessarily
        normalized.

        We assume that shunt admittances are not subject to failures.
        """
        # Figure out how much to shift a standard Gaussian so that there is the right
        # probability of negative values
        shift = -jax.scipy.stats.norm.ppf(self.network_spec.prob_line_failure)

        # Get the log likelihood for this shifted Gaussian
        logprob = jax.scipy.stats.norm.logpdf(
            network_state.line_states, loc=shift
        ).sum()

        return logprob

    @jaxtyped(typechecker=beartype)
    def sample_random_network_state(self, key: PRNGKeyArray) -> NetworkState:
        """
        Sample a random network state

        Line states are positive in the absence of failure and negative when failure
        occurs. We can represent this as a Gaussian distribution with the mean shifted
        so that there is network_spec.prob_line_failure probability mass for negative
        values.

        Failures are assumed to be indepedendent.

        args:
            key: PRNG key to use for sampling
        """
        # Figure out how much to shift a standard Gaussian so that there is the right
        # probability of negative values
        shift = -jax.scipy.stats.norm.ppf(self.network_spec.prob_line_failure)

        # Sample line states (positive = no failure, negative = failure)
        n_line = self.network_spec.n_line
        line_states = jrandom.normal(key, shape=(n_line,)) + shift

        # Return a new network state
        return NetworkState(line_states)
