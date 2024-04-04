"""Define a simulator for drones flying in wind in formation in 2D"""

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import NamedTuple
from jax.nn import softplus
from jaxtyping import Array, Float, jaxtyped

from radium.systems.hide_and_seek.hide_and_seek_types import (
    MultiAgentTrajectory,
    Trajectory2D,
)
from radium.types import PRNGKeyArray
from radium.utils import softmax, softmin


class FormationResult(NamedTuple):
    """
    The result of a formation simulation

    args:
        positions: the positions of the drones over time
        potential: the potential/cost assigned to this rollout
        connectivity: the connectivity of the formation over time
        min_interagent_distance: the minimum distance between any two agents over time
    """

    positions: Float[Array, "T n 2"]
    potential: Float[Array, " "]
    connectivity: Float[Array, "T"]
    min_interagent_distance: Float[Array, "T"]


class WindField(eqx.Module):
    """
    Represents a wind flow field in 2D as the output of an MLP.
    """

    layers: list

    def __init__(self, key):
        key1, key2, key3 = jax.random.split(key, 3)
        n_in, n_hidden, n_out = 2, 32, 2
        self.layers = [
            eqx.nn.Linear(n_in, n_hidden, key=key1),
            eqx.nn.Linear(n_hidden, n_hidden, key=key2),
            eqx.nn.Linear(n_hidden, n_out, key=key3),
        ]

    def __call__(self, x, max_thrust: float = 1.0):
        for layer in self.layers:
            x = jax.nn.tanh(layer(x))

        return max_thrust * x


class KernelWindField(eqx.Module):
    """
    Represents a wind flow field in 2D as a kernel function.
    """

    wind_kernels: list
    kernel_locs: list

    def __init__(self, key, n_kernels: int = 1):
        key1, key2 = jax.random.split(key, 2)
        self.wind_kernels = jax.random.normal(key1, shape=(n_kernels, 2))
        self.kernel_locs = jax.random.normal(key2, shape=(n_kernels, 2))

    def kernel_fn(self, x, loc, wind):
        # Make the kernel stronger on the positive side of the wind
        wind_direction = wind / (jnp.linalg.norm(wind) + 1e-3)
        distance_along_wind = jnp.dot(x - loc, wind_direction)
        distance_perpendicular_to_wind = jnp.linalg.norm(
            x - loc - distance_along_wind * wind_direction
        )
        # Apply a Gaussian kernel
        return jax.lax.cond(
            distance_along_wind >= 0,
            lambda: jnp.exp(
                -20 * distance_perpendicular_to_wind**2
                - distance_along_wind**2 / jnp.linalg.norm(wind)
            ),
            lambda: jnp.exp(
                -20 * distance_perpendicular_to_wind**2
                - 20 * distance_along_wind**2 / jnp.linalg.norm(wind)
            ),
        )

    def __call__(self, x, max_thrust: float = 1.0):
        weights = jax.vmap(self.kernel_fn, in_axes=(None, 0, 0))(
            x, self.kernel_locs, self.wind_kernels
        )
        return max_thrust * jnp.sum(weights[:, None] * self.wind_kernels, axis=0)


class ConstantWindField(eqx.Module):
    """
    Represents a wind flow field in 2D as a constant vector field.
    """

    vector: Float[Array, " 2"]

    def __init__(self, key):
        self.vector = jax.random.normal(key, shape=(2,))

    def __call__(self, x, max_thrust: float = 1.0):
        return max_thrust * jax.nn.tanh(self.vector)


def double_integrator_dynamics(
    q: Float[Array, " 4"], u: Float[Array, " 2"], d: Float[Array, " 2"], mass: float
) -> Float[Array, " 4"]:
    """
    The dynamics of a double integrator with wind.

    args:
        q: the state of the system (x, y, xdot, ydot)
        u: the control input (xddot, yddot)
        d: the wind thrust (fx, fy)
    returns:
        the time derivative of state (xdot, ydot, xddot, yddot)
    """
    qdot = jnp.concatenate([q[2:], (u + d) / mass])
    return qdot


def pd_controller(q: Float[Array, " 4"], target_position: Float[Array, " 2"]):
    """
    A PD controller for a double integrator.
    """
    kp, kd = 10.0, 5.0
    u = -kp * (q[:2] - target_position) - kd * q[2:]
    return u


def closed_loop_dynamics(
    q: Float[Array, " 4"],
    target_position: Float[Array, " 2"],
    wind: eqx.Module,
    max_wind_thrust: float,
    mass: float,
) -> Float[Array, " 4"]:
    """
    The closed loop dynamics of a double integrator with wind.
    """
    u = pd_controller(q, target_position)
    d = wind(q[:2], max_wind_thrust)
    return double_integrator_dynamics(q, u, d, mass)


def algebraic_connectivity(
    positions: Float[Array, "n 2"],
    communication_range: float,
    communication_strengths: Float[Array, "n n"],
    sharpness: float = 20.0,
) -> Float[Array, ""]:
    """Compute the connectivity of a formation of drones"""
    # Get the pairwise distance between drones
    squared_distance_matrix = jnp.sum(
        (positions[:, None, :] - positions[None, :, :]) ** 2, axis=-1
    )

    # Compute adjacency and degree matrices (following Cavorsi RSS 2023 with
    # modification for reduced connection strengths)
    adjacency_matrix = jax.nn.sigmoid(
        sharpness * (communication_range**2 - squared_distance_matrix)
    )
    adjacency_matrix = adjacency_matrix * jax.nn.sigmoid(4 * communication_strengths)
    degree_matrix = jnp.diag(jnp.sum(adjacency_matrix, axis=1))

    # Compute the laplacian
    laplacian = degree_matrix - adjacency_matrix

    # Compute the connectivity (the second smallest eigenvalue of the laplacian)
    connectivity = jnp.linalg.eigvalsh(laplacian)[1]

    return connectivity


def min_interagent_distance(
    positions: Float[Array, "n 2"], sharpness: float = 20.0
) -> Float[Array, ""]:
    """Compute the minimum distance between any two agents"""
    # Get the pairwise distance between drones
    distance_matrix = jnp.sqrt(
        jnp.sum(((positions[:, None, :] - positions[None, :, :]) ** 2), axis=-1) + 1e-3
    )

    # Set the diagonal to something large so we don't have to worry
    distance_matrix += jnp.eye(distance_matrix.shape[0]) * 100.0

    # Compute the minimum distance
    min_distance = softmin(distance_matrix, sharpness)

    return min_distance


def sample_random_connection_strengths(
    key: PRNGKeyArray, n: int
) -> Float[Array, "n n"]:
    """
    Sample a random set of connections strengths.

    Connection strength is positive in the absence of failure and negative when failure
    occurs. We can represent this as a Gaussian distribution with the mean shifted
    so that there is 2% probability mass for negative values.

    Failures are assumed to be indepedendent.

    args:
        key: PRNG key to use for sampling
        n: number of drones
    """
    # Figure out how much to shift a standard Gaussian so that there is the right
    # probability of negative values
    shift = -jax.scipy.stats.norm.ppf(0.02)

    # Sample line states (positive = no failure, negative = failure)
    connection_strengths = jax.random.normal(key, shape=(n, n)) + shift

    # Return a new network state
    return connection_strengths


@jaxtyped(typechecker=beartype)
def connection_strength_prior_logprob(
    connection_strengths: Float[Array, "n n"]
) -> Float[Array, ""]:
    """
    Compute the prior log probability of the given connection strengths.

    Connection strength is positive in the absence of failure and negative when failure
    occurs. We can represent this as a Gaussian distribution with the mean shifted
    so that there is 2% probability mass for negative values.

    Failures are assumed to be indepedendent. Probability density is not necessarily
    normalized.

    args:
        connection_strengths: the connection strengths to evaluate
    """
    # Figure out how much to shift a standard Gaussian so that there is the right
    # probability of negative values
    shift = -jax.scipy.stats.norm.ppf(0.02)

    # Get the log likelihood for this shifted Gaussian
    logprob = jax.scipy.stats.norm.logpdf(connection_strengths, loc=shift).sum()

    return logprob


@jaxtyped(typechecker=beartype)
def simulate(
    target_trajectories: MultiAgentTrajectory,
    initial_states: Float[Array, "n 4"],
    wind: eqx.Module,
    connection_strengths: Float[Array, "n n"],
    goal_com_position: Float[Array, " 2"],
    duration: float = 10.0,
    dt: float = 0.01,
    max_wind_thrust: float = 0.5,
    communication_range: float = 0.5,
    drone_mass: float = 0.2,
    b: float = 20.0,
) -> FormationResult:
    """
    Simulate a formation of drones flying in wind

    args:
        target_positions: the target positions of the drones over time
        initial_states: the initial states of the drones
        wind: the wind vector over time
        connection_strengths: the connection strengths between drones
        goal_com_position: the goal center of mass position for the formation
        duration: the length of the simulation (seconds)
        dt: the timestep of the simulation (seconds)
        max_wind_thrust: the maximum thrust of the wind (N)
        communication_range: the communication range of each drone (meters)
        drone_mass: the mass of each drone (kg)
        b: parameter used to smooth min/max

    returns:
        A FormationResult
    """

    # Define a function to step the simulation
    def step(carry, t):
        # Unpack the carry
        qs = carry["qs"]

        # Get the target positions for this timestep
        target_positions = target_trajectories(t / duration)

        # get the state derivative for each drone
        qdots = jax.vmap(closed_loop_dynamics, in_axes=(0, 0, None, None, None))(
            qs, target_positions, wind, max_wind_thrust, drone_mass
        )

        # Integrate the state derivative
        qs = qs + qdots * dt

        # Pack the carry
        carry = {"qs": qs}

        # Return the carry for the next step and output states as well
        return carry, qs

    # Define the initial carry and inputs
    carry = {"qs": initial_states}

    # Run the loop
    _, qs = jax.lax.scan(step, carry, jnp.arange(0.0, duration, dt))

    # Symmetrize connection strengths
    connection_strengths = 0.5 * (connection_strengths + connection_strengths.T)

    # Compute the potential based on the minimum connectivity of the formation over
    # time. The potential penalizes connectivities that approach zero
    connectivity = jax.vmap(algebraic_connectivity, in_axes=(0, None, None))(
        qs[:, :, :2],
        communication_range,
        connection_strengths,
    )
    inverse_connectivity = 1 / (connectivity + 1e-2)
    potential = softmax(inverse_connectivity, b)

    # Add a term that encourages getting the formation COM to the desired position
    formation_final_com = jnp.mean(qs[-1, :, :2], axis=0)
    potential += 10 * jnp.sqrt(
        jnp.mean((formation_final_com - goal_com_position) ** 2) + 1e-2
    )

    # Add a term that avoids collisions
    min_distance_trace = jax.vmap(min_interagent_distance)(qs[:, :, :2])
    min_distance = softmin(min_distance_trace, b)
    min_distance = softplus(b * min_distance) / b
    potential += 0.1 / min_distance

    # Return the result
    return FormationResult(
        positions=qs,
        potential=potential,
        connectivity=connectivity,
        min_interagent_distance=min_distance_trace,
    )


if __name__ == "__main__":
    # Test the simulation
    key = jax.random.PRNGKey(1)
    wind = KernelWindField(key)
    target_positions = MultiAgentTrajectory(
        [
            Trajectory2D(jnp.array([[1.5, 0.3]])),
            Trajectory2D(jnp.array([[1.5, 0.0]])),
            Trajectory2D(jnp.array([[1.5, -0.3]])),
        ]
    )
    initial_states = jnp.array(
        [
            [-1.5, -0.3, 0.0, 0.0],
            [-1.5, 0.0, 0.0, 0.0],
            [-1.5, 0.3, 0.0, 0.0],
        ]
    )
    goal_com_position = jnp.array([1.5, 0.0])
    result = simulate(
        target_positions,
        initial_states,
        wind,
        connection_strengths=jnp.ones((3, 3)),
        max_wind_thrust=50.0,
        goal_com_position=goal_com_position,
    )
    print(result.potential)

    # # Test that there is a gradient path from the target positions to the potential
    # def potential_fn(x):
    #     target_positions = MultiAgentTrajectory(
    #         [
    #             Trajectory2D(jnp.array([[1.5, 0.3]])),
    #             Trajectory2D(jnp.array([[1.5, 0.0]])),
    #             Trajectory2D(jnp.array([[1.5 + x, -0.3 + x]])),
    #         ]
    #     )
    #     initial_states = jnp.array(
    #         [
    #             [-1.5, -0.3, 0.0, 0.0],
    #             [-1.5, 0.0, 0.0, 0.0],
    #             [-1.5, 0.3, 0.0, 0.0],
    #         ]
    #     )
    #     goal_com_position = jnp.array([1.5, 0.0])
    #     result = simulate(
    #         target_positions,
    #         initial_states,
    #         wind,
    #         connection_strengths=jnp.ones((3, 3)),
    #         max_wind_thrust=1.0,
    #         goal_com_position=goal_com_position,
    #     )
    #     return result.potential

    # grad_fn = jax.grad(potential_fn)
    # grad = grad_fn(0.0)
    # print(grad)

    # plot the results, with a 2D plot of the positions and a time trace of the
    # connectivity on different subplots
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(6, 9))

    # Plot the trajectories
    axes[0].plot(result.positions[:, :, 0], result.positions[:, :, 1])
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].set_title("Positions")
    # Overlay the wind field
    X, Y = jnp.meshgrid(jnp.linspace(-2, 2, 50), jnp.linspace(-1, 1, 50))
    wind_speeds = jax.vmap(jax.vmap(wind))(jnp.stack([X, Y], axis=-1))
    axes[0].quiver(
        X,
        Y,
        wind_speeds[:, :, 0],
        wind_speeds[:, :, 1],
        color="b",
        alpha=0.5,
        angles="xy",
        scale_units="xy",
        scale=10.0,
    )
    axes[0].scatter(
        wind.kernel_locs[:, 0], wind.kernel_locs[:, 1], color="r", marker="x"
    )
    axes[0].scatter(
        wind.kernel_locs[:, 0] + wind.wind_kernels[:, 0],
        wind.kernel_locs[:, 1] + wind.wind_kernels[:, 1],
        color="r",
        marker="o",
    )

    # Plot the connectivity
    axes[1].plot(result.connectivity)
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Connectivity")
    axes[1].set_title("Connectivity")

    # Plot the minimum interagent distance
    min_distance = softplus(20.0 * result.min_interagent_distance) / 20.0
    min_distance_potential = 1 / min_distance
    axes[2].plot(result.min_interagent_distance, label="Minimum Interagent Distance")
    axes[2].plot(min_distance_potential, label="Potential")
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Minimum Interagent Distance")
    axes[2].legend()

    plt.tight_layout()
    plt.show()
