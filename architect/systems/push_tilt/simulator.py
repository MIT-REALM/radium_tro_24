"""Define a simulator for the pusher-slider system"""

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import NamedTuple
from jaxtyping import Array, Float, jaxtyped

from architect.types import PRNGKeyArray


class PushTiltResult(NamedTuple):
    """
    The result of a pusher-slider (with tip-over) simulation

    args:
        push_height: the height of the planned push
        push_force: the force of the planned push
        net_moment: the net moment applied to the slider
        net_force: the net force applied to the slider
        potential: the potential/cost assigned to this rollout
    """

    push_height: Float[Array, ""]
    push_force: Float[Array, ""]
    net_moment: Float[Array, ""]
    net_force: Float[Array, ""]
    potential: Float[Array, ""]


class Planner(eqx.Module):
    """Define an MLP for the planner"""

    layers: list

    def __init__(self, key):
        key1, key2, key3 = jax.random.split(key, 3)
        # in: observed height and radius, out: push height and force
        n_in, n_hidden, n_out = 2, 32, 2
        self.layers = [
            eqx.nn.Linear(n_in, n_hidden, key=key1),
            eqx.nn.Linear(n_hidden, n_hidden, key=key2),
            eqx.nn.Linear(n_hidden, n_out, key=key3),
        ]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.tanh(layer(x))

        push_height, push_force = self.layers[-1](x)

        # Saturate push_height from 0.1 to 0.9
        push_height = 0.1 + 0.8 * jax.nn.sigmoid(push_height)

        # Saturate push_force from 0 to 10
        push_force = 10 * jax.nn.sigmoid(push_force)

        return push_height, push_force


def sample_object_parameters(key: PRNGKeyArray) -> Float[Array, " 5"]:
    """
    Sample parameters for the pushed object.
    """
    keys = jax.random.split(key, 5)
    height = jax.random.uniform(keys[0], minval=0.1, maxval=0.2)
    radius = jax.random.uniform(keys[1], minval=0.1, maxval=0.25)
    mass = jax.random.uniform(keys[2], minval=0.1, maxval=1.0)
    friction = jax.random.uniform(keys[3], minval=0.1, maxval=1.0)
    com_height = jax.random.uniform(keys[4], minval=0.1, maxval=0.9) * height

    return jnp.array([height, radius, mass, friction, com_height])


def object_parameters_logprior(params: Float[Array, " 5"]) -> Float[Array, ""]:
    """Compute the log prior probability for the given parameters."""

    def log_smooth_uniform(x, x_min, x_max):
        b = 100.0  # sharpness
        return jax.nn.log_sigmoid(b * (x - x_min)) + jax.nn.log_sigmoid(b * (x_max - x))

    height, radius, mass, friction, com_height = params

    logprior = jnp.array(0.0)
    logprior += log_smooth_uniform(height, 0.1, 0.2)
    logprior += log_smooth_uniform(radius, 0.1, 0.25)
    logprior += log_smooth_uniform(mass, 0.1, 1.0)
    logprior += log_smooth_uniform(friction, 0.1, 1.0)
    logprior += log_smooth_uniform(com_height, 0.1 * height, 0.9 * height)

    return logprior


def sample_observations(
    key: PRNGKeyArray, object_params: Float[Array, " 5"]
) -> Float[Array, " 2"]:
    """
    Sample noisy observations of the height and radius.
    """
    keys = jax.random.split(key, 2)
    height = jax.random.normal(keys[0]) * 0.1 + object_params[0]
    radius = jax.random.normal(keys[1]) * 0.1 + object_params[1]

    return jnp.array([height, radius])


def observation_logprior(
    params: Float[Array, " 5"], observation: Float[Array, " 2"]
) -> Float[Array, ""]:
    """Compute the log prior probability for the given observations."""
    h, r = params[:2]

    logprior = jnp.array(0.0)
    logprior += jax.scipy.stats.norm.logpdf(observation[0], loc=h, scale=0.1)
    logprior += jax.scipy.stats.norm.logpdf(observation[1], loc=r, scale=0.1)

    return logprior


@jaxtyped(typechecker=beartype)
def simulate(
    planner: Planner,
    observations: Float[Array, " 2"],
    object_params: Float[Array, " 5"],
) -> PushTiltResult:
    """
    Simulate.

    args:
        planner: the planner
        observations: the observations
        object_params: the parameters of the object

    returns:
        the result of the simulation
    """
    # Get the planned push height and force from the observations
    push_height, push_force = planner(observations)
    push_height = push_height * observations[0]

    # Very simple: we need to push hard enough to make the object move without tipping
    # it over
    _, radius, mass, friction, com_height = object_params
    d_com = jnp.sqrt(radius**2 + com_height**2)
    theta_com = jnp.arctan2(radius, com_height)
    tau_gravity = mass * 9.81 * d_com * jnp.sin(theta_com)
    tau_push = push_force * push_height
    net_moment = tau_push - tau_gravity  # needs to be <= 0
    net_force = push_force - mass * 9.81 * friction  # needs to be >= 0

    # Compute the potential to make sure net_moment <= 0 and net_force >= 1.0 N
    potential = jax.nn.relu(net_moment) + jax.nn.relu(1.0 - net_force)

    # Return the results data
    return PushTiltResult(
        push_height,
        push_force,
        net_moment,
        net_force,
        potential,
    )
