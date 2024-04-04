import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from jax.nn import logsumexp
from jaxtyping import Array, Float, jaxtyped

from radium.systems.hide_and_seek.hide_and_seek_types import (
    HideAndSeekResult,
    MultiAgentTrajectory,
)
from radium.utils import softclip


@jax.jit
def softnorm(x):
    """Compute the 2-norm, but if x is too small replace it with the squared 2-norm
    to make sure it's differentiable. This function is continuous and has a derivative
    that is defined everywhere, but its derivative is discontinuous.
    """
    eps = 1e-5
    scaled_square = lambda x: (eps * (x / eps) ** 2).sum()
    return jax.lax.cond(jnp.linalg.norm(x) >= eps, jnp.linalg.norm, scaled_square, x)


@jaxtyped(typechecker=beartype)
class Game(eqx.Module):
    """
    The hide and seek game engine

    args:
        initial_seeker_positions: initial positions of seekers
        initial_hider_positions: initial positions of hiders
        duration: the length of the game (seconds)
        dt: the timestep of the game's simulation (seconds)
        sensing_range: the sensing range of each seeker (meters)
        seeker_max_speed: the max speed of the seekers (m/s)
        hider_max_speed: the max speed of the hiders (m/s)
        b: parameter used to smooth min/max
    """

    initial_seeker_positions: Float[Array, "n_seeker 2"]
    initial_hider_positions: Float[Array, "n_hider 2"]

    duration: float = 100.0
    dt: float = 0.1

    sensing_range: float = 0.5

    seeker_max_speed: float = 0.5
    hider_max_speed: float = 0.5

    b: float = 100

    width: float = 3.0
    height: float = 2.0

    def softmin(self, x: Float[Array, "..."]) -> Float[Array, ""]:
        return -1 / self.b * logsumexp(-self.b * x)

    @jaxtyped(typechecker=beartype)
    def __call__(
        self,
        seeker_trajectory: MultiAgentTrajectory,
        hider_trajectory: MultiAgentTrajectory,
        seeker_disturbance_trace: MultiAgentTrajectory,
        max_disturbance: float = 0.01,
    ) -> HideAndSeekResult:
        """
        Play out a game of hide and seek.

        args:
            seeker_trajectory: trajectories for the seekers to follow
            hider_trajectory: trajectories for the hiders to follow
            seeker_disturbance: disturbance to add to the seekers
            max_disturbance: maximum disturbance to add to the seekers
        """

        # Define a function to execute one step in the game
        def step(carry, t):
            # Unpack carry
            current_seeker_positions, current_hider_positions = carry

            # Get current target positions for hiders and seekers
            seeker_target = seeker_trajectory(t / self.duration)
            hider_target = hider_trajectory(t / self.duration)
            seeker_disturbance = seeker_disturbance_trace(t / self.duration)
            seeker_disturbance = jax.nn.tanh(seeker_disturbance) * max_disturbance

            # Clamp the targets to be inside the arena (soft clip)
            seeker_target = seeker_target.at[:, 0].set(
                softclip(seeker_target[:, 0], self.width / 2.0)
            )
            seeker_target = seeker_target.at[:, 1].set(
                softclip(seeker_target[:, 1], self.height / 2.0)
            )
            hider_target = hider_target.at[:, 0].set(
                softclip(hider_target[:, 0], self.width / 2.0)
            )
            hider_target = hider_target.at[:, 1].set(
                softclip(hider_target[:, 1], self.height / 2.0)
            )

            # Steer towards these targets (clip with max speeds)
            v_seeker = seeker_target - current_seeker_positions

            # # Add the disturbance to the seekers
            # v_seeker += seeker_disturbance

            # Cap max speed
            seeker_speed = softnorm(v_seeker)
            v_seeker = jnp.where(
                seeker_speed >= self.seeker_max_speed,
                v_seeker * self.seeker_max_speed / (1e-3 + seeker_speed),  # avoid nans
                v_seeker,
            )

            # Same for hiders
            v_hider = hider_target - current_hider_positions
            hider_speed = softnorm(v_hider)
            v_hider = jnp.where(
                hider_speed >= self.hider_max_speed,
                v_hider * self.hider_max_speed / (1e-3 + hider_speed),  # avoid nans
                v_hider,
            )

            seeker_positions = current_seeker_positions + self.dt * v_seeker
            hider_positions = current_hider_positions + self.dt * v_hider

            # Get the distance from each hider to each seeker
            # See https://sparrow.dev/pairwise-distance-in-numpy/
            hider_to_seeker_distance_matrix = jnp.linalg.norm(
                hider_positions[:, None, :] - seeker_positions[None, :, :], axis=-1
            )
            hider_to_closest_seeker_distance = jax.vmap(self.softmin)(
                hider_to_seeker_distance_matrix
            )

            # Return an updated carry and output
            output = (
                seeker_positions,
                hider_positions,
                hider_to_closest_seeker_distance,
            )
            carry = output[:2]
            return carry, output

        # Execute the game simulation to get a trace of hider and seeker positions
        _, (
            seeker_positions,
            hider_positions,
            hider_to_closest_seeker_distance,
        ) = jax.lax.scan(
            step,
            (self.initial_seeker_positions, self.initial_hider_positions),
            jnp.arange(0, self.duration, self.dt),
        )

        # Compute the cost as the (smoothed) minimum squared distance between each hider
        # and its closest seeker
        softmin_hider_to_closest_seeker_distance = jax.vmap(self.softmin)(
            hider_to_closest_seeker_distance.T
        )
        # cost = jax.nn.celu(
        #     (softmin_hider_to_closest_seeker_distance - self.sensing_range)
        #     / self.sensing_range,
        #     alpha=1.0,
        # ).mean()
        cost = -self.softmin(  # max distance outside sensing range
            -(softmin_hider_to_closest_seeker_distance - self.sensing_range)
        )
        num_escaped_hiders = (
            softmin_hider_to_closest_seeker_distance > self.sensing_range
        ).sum()

        return HideAndSeekResult(
            seeker_trajectory=seeker_trajectory,
            hider_trajectory=hider_trajectory,
            game_duration=self.duration,
            seeker_positions=seeker_positions,
            hider_positions=hider_positions,
            potential=cost,
            escaped=num_escaped_hiders,
        )
