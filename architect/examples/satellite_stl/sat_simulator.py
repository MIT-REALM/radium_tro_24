"""Define a simulator for the satellite"""
from functools import partial
import time
from typing import Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from architect.components.dynamics.linear_satellite import (
    linear_satellite_next_state_substeps,
    linear_satellite_dt_AB,
)


@partial(jax.jit, static_argnames=["substeps"])
def sat_simulate(
    design_params: jnp.ndarray,
    exogenous_sample: jnp.ndarray,
    time_steps: int,
    dt: float,
    substeps: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Simulate the performance of the satellite rendezvous system.

    To make this function pure, we need to pass in all sources of randomness used.

    args:
        design_params: a (6 * 3 + time_steps * (3 + 6)) array of design parameters.
            The first 6 * 3 values define a gain matrix for state-feedback control
            u = -K(x - x_planned). Each later group of 3 + 6 values is a control input
            and target state at that timestep, defining a trajectory.
        exogenous_sample: (6,) array containing initial states.
                          Can be generated by SatExogenousParameters.sample
        time_steps: the number of steps to simulate
        dt: the duration of each time step
        substeps: how many smaller updates to break this interval into
    returns:
        a tuple of
            - the state trace in a (time_steps, 6) array
            - the total expended actuation effort (sum of 1-norm scaled by dt)
    """
    # To enable easier JIT, this simulation will be implemented as a scan.
    # This requires creating an array with the reference states and controls
    # that we will scan over
    K = design_params[: 6 * 3].reshape(3, 6)
    planned_trajectory = design_params[6 * 3 :].reshape(-1, 3 + 6)

    # No noise for now
    actuation_noise = jnp.zeros(6)

    # Now create a function that executes a single step of the simulation
    @partial(jax.jit, static_argnames=["substeps"])
    def step(current_state, current_plan):
        """Perform a single simulation step

        args:
            current_state: (6,) array of satellite states
            current_plan: (9,) array of 3 planned inputs and 6 planned states
        returns:
            - (6,) array of next state
            - (9,) array of next state and control input used to get there
        """
        # Get the state and control from the plan
        planned_input = current_plan[:3]
        planned_state = current_plan[3:]

        # Compute the control from feedback and planned
        control_input = planned_input - K @ (current_state - planned_state)

        # Update the state
        next_state = linear_satellite_next_state_substeps(
            current_state, control_input, actuation_noise, dt, substeps
        )

        return next_state, jnp.concatenate((next_state, control_input))

    # Simulate by scanning over the plan
    initial_state = exogenous_sample
    state_control_trace: jnp.ndarray
    final_state, state_control_trace = jax.lax.scan(
        step, initial_state, planned_trajectory
    )

    # Get the state trace and total control effort
    state_trace = state_control_trace[:, :6]
    total_effort = jnp.linalg.norm(state_control_trace[:, 6:], ord=1, axis=-1).sum()

    return state_trace, dt * total_effort


@jax.jit
def sat_simulate_dt(
    design_params: jnp.ndarray,
    exogenous_sample: jnp.ndarray,
    time_steps: int,
    dt: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Simulate the performance of the satellite rendezvous system.

    To make this function pure, we need to pass in all sources of randomness used.

    args:
        design_params: a (6 * 3 + time_steps * (3 + 6)) array of design parameters.
            The first 6 * 3 values define a gain matrix for state-feedback control
            u = -K(x - x_planned). Each later group of 3 + 6 values is a control input
            and target state at that timestep, defining a trajectory.
        exogenous_sample: (6,) array containing initial states.
                          Can be generated by SatExogenousParameters.sample
        time_steps: the number of steps to simulate
        dt: the duration of each time step
    returns:
        a tuple of
            - the state trace in a (time_steps, 6) array
            - the total expended actuation effort (sum of 1-norm scaled by dt)
    """
    # To enable easier JIT, this simulation will be implemented as a scan.
    # This requires creating an array with the reference states and controls
    # that we will scan over
    K = design_params[: 6 * 3].reshape(3, 6)
    planned_trajectory = design_params[6 * 3 :].reshape(-1, 3 + 6)

    # Get discrete-time matrices
    A, B = linear_satellite_dt_AB(dt)

    # Now create a function that executes a single step of the simulation
    @jax.jit
    def step(current_state, current_plan):
        """Perform a single simulation step

        args:
            current_state: (6,) array of satellite states
            current_plan: (9,) array of 3 planned inputs and 6 planned states
        returns:
            - (6,) array of next state
            - (9,) array of next state and control input used to get there
        """
        # Get the state and control from the plan
        planned_input = current_plan[:3]
        planned_state = current_plan[3:]

        # Compute the control from feedback and planned
        control_input = planned_input - K @ (current_state - planned_state)

        # Update the state
        next_state = A @ current_state + B @ control_input

        return next_state, jnp.concatenate((next_state, control_input))

    # Simulate by scanning over the plan
    initial_state = exogenous_sample
    state_control_trace: jnp.ndarray
    final_state, state_control_trace = jax.lax.scan(
        step, initial_state, planned_trajectory
    )

    # Get the state trace and total control effort
    state_trace = state_control_trace[:, :6]
    total_effort = jnp.linalg.norm(state_control_trace[:, 6:], ord=1, axis=-1).sum()

    return state_trace, dt * total_effort


if __name__ == "__main__":
    # Test the simulation
    t_sim = 100.0
    dt = 1.0
    substeps = 200
    T = int(t_sim // dt)
    start_state = jnp.zeros((6,)) + 1.0
    K = jnp.array(
        [
            [100.0, 0.0, 0.0, 100.0, 0.0, 0.0],
            [0.0, 100.0, 0.0, 0.0, 100.0, 0.0],
            [0.0, 0.0, 100.0, 0.0, 0.0, 100.0],
        ]
    )
    planned_trajectory = jnp.zeros((T, 9))
    design_params = jnp.concatenate((K.reshape(-1), planned_trajectory.reshape(-1)))

    # Burn-in once to activate JIT (if using)
    state_trace_ct, effort = sat_simulate(design_params, start_state, T, dt, substeps)
    state_trace_dt, effort = sat_simulate_dt(design_params, start_state, T, dt)

    ax = plt.axes(projection="3d")
    ax.plot3D(state_trace_ct[:, 0], state_trace_ct[:, 1], state_trace_ct[:, 2])
    ax.plot3D(state_trace_dt[:, 0], state_trace_dt[:, 1], state_trace_dt[:, 2], "r:")
    ax.plot3D(0.0, 0.0, 0.0, "ko")

    N_tests = 5
    sim_time = 0.0
    for _ in range(N_tests):
        start = time.perf_counter()
        state_trace, _ = sat_simulate(design_params, start_state, T, dt, substeps)
        end = time.perf_counter()
        sim_time += end - start

    print(f"CT: ran {N_tests} sims in {sim_time} s. Average {sim_time / N_tests} s")

    N_tests = 5
    sim_time = 0.0
    for _ in range(N_tests):
        start = time.perf_counter()
        state_trace, _ = sat_simulate_dt(design_params, start_state, T, dt)
        end = time.perf_counter()
        sim_time += end - start

    print(f"DT: ran {N_tests} sims in {sim_time} s. Average {sim_time / N_tests} s")

    plt.show()

    # # Test vectorizing
    # sim_trace_v, effort_v = jax.vmap(sat_simulate_dt, (0, None, None, None))(
    #     jnp.tile(design_params.reshape(1, -1), (2, 1)), start_state, T, dt
    # )
