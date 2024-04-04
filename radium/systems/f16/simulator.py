"""Define a simulator for the F16 system"""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype.typing import NamedTuple
from jax_f16.f16 import F16
from jaxtyping import Array, Float

from radium.types import PRNGKeyArray
from radium.utils import softmin

# Constants
FLIGHT_DECK = 1000.0  # ft


class F16Result(NamedTuple):
    """
    The result of an F16 simulation

    args:
        states: the state of the aircraft over time
        potential: the potential/cost assigned to this rollout
    """

    states: Float[Array, "T nx"]
    potential: Float[Array, ""]


def are_wings_level(x_f16):
    "are the wings level?"
    phi = x_f16[F16.PHI]
    radsFromWingsLevel = round(phi / (2 * math.pi))
    return abs(phi - (2 * math.pi) * radsFromWingsLevel) < 0.087


def is_roll_rate_low(x_f16):
    "is the roll rate low enough to switch to pull?"
    p = x_f16[F16.P]
    return abs(p) < 0.17


def is_above_flight_deck(x_f16):
    "is the aircraft above the flight deck?"
    alt = x_f16[F16.H]
    return alt >= FLIGHT_DECK


def is_nose_high_enough(x_f16):
    "is the nose high enough?"

    theta = x_f16[F16.THETA]
    alpha = x_f16[F16.ALPHA]

    # Determine which angle is "level" (0, 360, 720, etc)
    radsFromNoseLevel = round((theta - alpha) / (2 * math.pi))

    # Evaluate boolean
    return (theta - alpha) - 2 * math.pi * radsFromNoseLevel > 0.0


def roll_wings_level(x_f16):
    "get commands to roll level."

    phi = x_f16[F16.PHI]
    p = x_f16[F16.P]

    rv = jnp.array(F16.trim_control())

    # Determine which angle is "level" (0, 360, 720, etc)
    radsFromWingsLevel = round(phi / (2 * math.pi))

    # PD Control until phi == pi * radsFromWingsLevel
    ps = -(phi - (2 * math.pi) * radsFromWingsLevel) * 4.0 - p * 2.0

    # Build commands to roll wings level
    rv = rv.at[1].set(ps)

    return rv


def pitch_level(x_f16):
    "get commands to pitch level."

    theta = x_f16[F16.THETA]
    alpha = x_f16[F16.ALPHA]
    q = x_f16[F16.Q]

    rv = jnp.array(F16.trim_control())

    # Determine which angle is "level" (0, 360, 720, etc)
    radsFromLevel = round(theta / (2 * math.pi))
    theta = theta - (2 * math.pi) * radsFromLevel
    gamma = theta - alpha

    # PD Control until theta == pi * radsFromLevel
    qs = -gamma * 4.0 - q * 2.0

    # Build commands to pull up to level
    rv = rv.at[0].set(qs)

    return rv


def pull_nose_level(self):
    "get commands to pull up"
    rv = jnp.array(F16.trim_control())
    rv = rv.at[0].set(5.0)
    return rv


def gcas_controller(x: Float[Array, " nx"]) -> Float[Array, " nu"]:
    """Implement a version of the GCAS controller from the original F16 implementation.

    github.com/stanleybak/AeroBenchVVPython
    """
    going_to_crash = jnp.logical_and(
        jnp.logical_not(is_nose_high_enough(x)),
        jnp.logical_not(is_above_flight_deck(x)),
    )

    need_to_level_wings = jnp.logical_not(
        jnp.logical_and(is_roll_rate_low(x), are_wings_level(x))
    )

    do_nothing = jnp.array(F16.trim_control())
    pull_up_command = pull_nose_level(x)
    roll_level_command = roll_wings_level(x)

    # If we are going to crash, we need to do something.
    # Roll level if not level, then pull up
    avoid_crash = roll_level_command
    avoid_crash += jnp.where(
        need_to_level_wings, jnp.zeros_like(pull_up_command), pull_up_command
    )

    # If we are going to crash, don't; otherwise, do nothing
    command = jnp.where(going_to_crash, avoid_crash, do_nothing)

    return command


def gcas_pd_controller(x: Float[Array, " nx"]) -> Float[Array, " nu"]:
    """Implement a simplified version of the GCAS controller."""
    pull_up_command = pitch_level(x)
    roll_level_command = roll_wings_level(x)

    gcas_command = pull_up_command + roll_level_command

    return gcas_command


class ResidualControl(eqx.Module):
    """Define an MLP for the residual control input"""

    layers: list
    activation_altitude: float = FLIGHT_DECK

    def __init__(self, key):
        key1, key2, key3 = jax.random.split(key, 3)
        n_in, n_hidden, n_out = F16.NX + len(F16.ANGLES), 32, F16.NU
        self.layers = [
            eqx.nn.Linear(n_in, n_hidden, key=key1),
            eqx.nn.Linear(n_hidden, n_hidden, key=key2),
            eqx.nn.Linear(n_hidden, n_out, key=key3),
        ]

    def __call__(self, x):
        # Sin/cos embedding for angles
        x = x.at[F16.ANGLES].set(jnp.sin(x[F16.ANGLES]))
        x = jnp.concatenate([x, jnp.cos(x[F16.ANGLES])], axis=-1)

        for layer in self.layers[:-1]:
            x = jax.nn.tanh(layer(x))

        # Scale the last layer to be small by default
        u = 1e-2 * self.layers[-1](x)
        u = jnp.tanh(u)

        # Saturate the throttle from -0.5 to 0.5
        u = u.at[F16.THRTL].set(0.5 * u[F16.THRTL])

        # Saturate other controls from -5 to 5
        u = u.at[: F16.THRTL].set(5 * u[: F16.THRTL])

        # Always regulate Ny+R to zero per the original paper
        u = u.at[F16.NYR].set(0.0)

        # Only apply the residual controller below a certain altitude
        u = jnp.where(
            x[F16.H] < self.activation_altitude, u, jnp.array(F16.trim_control())
        )

        return u


def closed_loop_dynamics(
    x: Float[Array, " nx"],
    residual_controller: ResidualControl,
) -> Float[Array, " nx"]:
    """
    The closed loop dynamics with GCAS and residual controller.

    args:
        x: the state of the aircraft
        residual_controller: the residual controller

    returns:
        the derivative of the state
    """
    # u_gcas = gcas_controller(x)
    u_gcas = gcas_pd_controller(x)
    u_residual = residual_controller(x)
    u = u_gcas + u_residual
    # u = u_residual

    return F16().xdot(x, u)


def sample_initial_states(key: PRNGKeyArray) -> Float[Array, " 5"]:
    """
    Sample initial states for the simulation.

    Based on case 3S from Heidlauf et al.
    """
    x = jnp.array(F16.trim_state())

    # Sample a random initial altitude
    h = 1500.0 + jax.random.normal(key) * 200.0

    # Sample random initial roll
    phi = jax.random.normal(key) * jnp.pi / 8

    # Sample random initial pitch
    theta = -jnp.pi / 5 + jax.random.normal(key) * jnp.pi / 8

    # Sample initial roll and pitch rates
    p = jax.random.normal(key) * jnp.pi / 8
    q = jax.random.normal(key) * jnp.pi / 8

    return jnp.array([h, phi, theta, p, q])


def initial_state_logprior(x: Float[Array, " 5"]) -> Float[Array, ""]:
    """Compute the log prior probability for the given initial state."""
    h, phi, theta, p, q = x

    # Initial altitude
    logprior = jax.scipy.stats.norm.logpdf(h, 1500.0, 200.0)

    # Initial roll
    logprior += jax.scipy.stats.norm.logpdf(phi, 0.0, jnp.pi / 8)

    # Initial pitch
    logprior += jax.scipy.stats.norm.logpdf(theta, -jnp.pi / 5, jnp.pi / 8)

    # Initial roll and pitch rate
    logprior += jax.scipy.stats.norm.logpdf(p, 0.0, jnp.pi / 8)
    logprior += jax.scipy.stats.norm.logpdf(q, 0.0, jnp.pi / 8)

    return logprior


# @jaxtyped(typechecker=beartype)
def simulate(
    initial_state: Float[Array, " 5"],
    residual_controller: ResidualControl,
    dt: float = 0.01,
    duration: float = 15.0,
) -> F16Result:
    """
    Simulate an F16 with GCAS and residual control.

    args:
        initial_state: the initial state of the aircraft
        residual_controller: the residual controller

    returns:
        the result of the simulation
    """
    # Load the provided initial state into the full state
    h, phi, theta, p, q = initial_state
    initial_state = jnp.array(F16.trim_state())
    initial_state = initial_state.at[F16.H].set(h)
    initial_state = initial_state.at[F16.PHI].set(phi)
    initial_state = initial_state.at[F16.THETA].set(theta)
    initial_state = initial_state.at[F16.P].set(p)
    initial_state = initial_state.at[F16.Q].set(q)

    # Define a function to step the simulation
    def step(carry, t):
        # Unpack the carry
        x = carry["x"]

        # get the state derivative and integrate
        xdot = closed_loop_dynamics(x, residual_controller)

        # Integrate the state derivative
        x_next = x + xdot * dt

        # Stop if you've hit the ground
        x_next = jnp.where(x_next[F16.H] <= 0.0, x, x_next)

        # Stop if alpha or beta are too large (model no longer valid)
        model_invalid = jnp.logical_or(
            jnp.logical_or(
                x_next[F16.ALPHA] <= -10 * jnp.pi / 180,
                x_next[F16.ALPHA] >= 45 * jnp.pi / 180,
            ),
            jnp.logical_or(
                x_next[F16.BETA] <= -30 * jnp.pi / 180,
                x_next[F16.BETA] >= 30 * jnp.pi / 180,
            ),
        )
        x_next = jnp.where(model_invalid, x, x_next)

        # Pack the carry
        carry = {"x": x_next}

        # Return the carry for the next step and output states as well
        return carry, x_next

    # Define the initial carry and inputs
    carry = {"x": initial_state}

    # Run the loop
    _, x = jax.lax.scan(step, carry, jnp.arange(0.0, duration, dt))

    # Get the minimum altitudetempering
    alt = x[:, F16.H]
    min_alt = softmin(alt, sharpness=10.0)

    # Potential is high when min altitude is low
    potential = (
        100 * jax.nn.elu(200.0 - min_alt) / FLIGHT_DECK
    )  # Normalize by flight deck

    # Add a penalty for deviation from level
    potential += jnp.mean(x[:, F16.PHI] ** 2 / jnp.pi**2)
    potential += jnp.mean(x[:, F16.THETA] ** 2 / jnp.pi**2)
    potential += jnp.mean(x[:, F16.ALPHA] ** 2 / (10 * jnp.pi / 180) ** 2)
    potential += jnp.mean(x[:, F16.BETA] ** 2 / (30 * jnp.pi / 180) ** 2)

    return F16Result(states=x, potential=potential)


if __name__ == "__main__":
    # # Sample a large number of points
    # N = int(5e2)
    # key = jax.random.PRNGKey(0)
    # key, x_key = jax.random.split(key)
    # x_keys = jax.random.split(x_key, N)
    # x = jax.vmap(sample_initial_states)(x_keys)

    # # Simulate all of these initial states using the hand-derived GCAS controller
    # sim_fn = lambda x: simulate(x, gcas_controller).states
    # print("Generating training set...")
    # x_train = jax.vmap(sim_fn)(x)

    # # Train a neural controller to match these
    # controller = ResidualControl(key)
    # lr = 1e-3
    # iters = 500

    # def loss_fn(pi, x_train):
    #     u_gcas = jax.vmap(jax.vmap(gcas_controller))(x_train)
    #     u_pi = jax.vmap(jax.vmap(pi))(x_train)
    #     mse = jnp.mean(jnp.square(u_gcas - u_pi))
    #     return mse

    # def step(controller):
    #     grads = jax.jit(jax.grad(loss_fn))(controller, x_train)
    #     return jax.tree_util.tree_map(lambda pi, g: pi - lr * g, controller, grads)

    # print("Training...")
    # for i in tqdm(range(iters)):
    #     controller = step(controller)
    #     if i % 10 == 0:
    #         print(f"Iter {i}: {loss_fn(controller, x_train)}")

    # eqx.tree_serialise_leaves("gcas_neural.eqx", controller)

    # Test the simulation
    print("Testing...")
    key = jax.random.PRNGKey(0)

    # Load the controller
    key, subkey = jax.random.split(key)
    controller = ResidualControl(subkey)
    # controller = eqx.tree_deserialise_leaves("gcas_neural.eqx", controller)

    # Point towards the ground
    initial_state = jnp.array([800.0, 1.0, -0.8, 0.0, 0.0])  # h phi theta p q
    initial_state = sample_initial_states(key)

    result = simulate(initial_state, gcas_controller, dt=0.01, duration=15.0)
    jax.numpy.set_printoptions(precision=2, linewidth=200)
    print(result.potential)
    print(
        (
            "  vₜ        α         β         ϕ         θ         ψ         P         "
            "Q         R         pn        pe        h         pow       nz        "
            "ps        ny+r"
        )
    )
    print(result.states[-1])

    # plot the results, showing altitude over time
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(6, 6))

    axes[0].plot(result.states[:, F16.H])
    axes[0].set_ylabel("Altitude (ft)")
    axes[0].set_xlabel("Time (s)")

    # Also plot the roll angle and rate
    axes[1].plot(result.states[:, F16.PHI], "r-", label="Roll Angle")
    axes[1].plot(result.states[:, F16.P], "r--", label="Roll Rate")
    axes[1].plot(result.states[:, F16.THETA], "b-", label="Pitch Angle")
    axes[1].plot(result.states[:, F16.Q], "b--", label="Pitch Rate")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend()

    plt.show()
