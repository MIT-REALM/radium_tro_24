"""Manage the state of the drone and landing environment."""

import jax
import jax.nn
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import NamedTuple, Tuple
from jax.nn import log_sigmoid
from jaxtyping import Array, Bool, Float, jaxtyped

from architect.systems.components.sensing.vision.render import (
    CameraExtrinsics,
    CameraIntrinsics,
)
from architect.systems.components.sensing.vision.util import look_at
from architect.systems.drone_landing.scene import DroneLandingScene
from architect.types import PRNGKeyArray


# @beartype
class DroneObs(NamedTuple):
    """Observations returned from the drone landing environment.

    Attributes:
        image: an image from the drone's camera.
        state: the state of the drone.
    """

    image: Float[Array, "*batch res_x res_y 3"]
    depth_image: Float[Array, "*batch res_x res_y"]
    state: Float[Array, "*batch 4"]


# @beartype
class DroneState(NamedTuple):
    """The state of the drone with single-integrator dynamics in x, y, z, and yaw.

    Attributes:
        drone_state: the state of the drone vehicle
        tree_locations: the locations of the trees in the scene
    """

    drone_state: Float[Array, " 3"]
    tree_locations: Float[Array, "num_trees 2"]
    tree_velocities: Float[Array, "num_trees 2"]
    wind_speed: Float[Array, " 2"]


class DroneLandingEnv:
    """
    The drone landing environment.

    The goal is for the drone to land on the landing pad. The drone has a camera
    mounted to it, and the agent can control the drone's position and orientation
    by controlling the drone's velocity and yaw rate.

    There are a number of trees in the environment; if the drone hits a tree, it
    will crash and the episode will end (with a large negative reward). Ditto for
    crashing into the ground.

    The drone has single integrator dynamics in x, y, z, and yaw.

    Args:
        drone_landing_scene: a representation of the underlying scene.
        camera_intrinsics: the intrinsics of the camera mounted to the drone.
        dt: the time step to use for simulation.
        num_trees: the number of trees in the scene.
        initial_drone_state: the initial state of the drone.
        collision_penalty: the penalty to apply when the ego vehicle collides with
            any obstacle in the scene.
        render_sharpness: the sharpness of the scene.
        moving_obstacles: whether or not to move the obstacles in the scene.
        substeps: the number of substeps to use for simulation.
    """

    _scene: DroneLandingScene
    _camera_intrinsics: CameraIntrinsics
    _dt: float
    _num_trees: int
    _collision_penalty: float
    _render_sharpness: float
    _initial_drone_state_mean: Float[Array, " n_states"]
    _initial_drone_state_stddev: Float[Array, " n_states"]
    _wind_speed_stddev: Float[Array, " "]
    _moving_obstacles: bool
    _substeps: int

    @jaxtyped(typechecker=beartype)
    def __init__(
        self,
        scene: DroneLandingScene,
        camera_intrinsics: CameraIntrinsics,
        dt: float,
        num_trees: int,
        initial_drone_state: Float[Array, " n_states"],
        collision_penalty: float = 100.0,
        render_sharpness: float = 100.0,
        moving_obstacles: bool = False,
        substeps: int = 1,
    ):
        """Initialize the environment."""
        self._scene = scene
        self._camera_intrinsics = camera_intrinsics
        self._dt = dt
        self._num_trees = num_trees
        self._initial_drone_state_mean = initial_drone_state
        self._initial_drone_state_stddev = jnp.array([0.1, 0.1, 0.05, 0.1])
        self._wind_speed_stddev = jnp.array(0.1)
        self._collision_penalty = collision_penalty
        self._render_sharpness = render_sharpness
        self._moving_obstacles = moving_obstacles
        self._substeps = substeps

    @jaxtyped(typechecker=beartype)
    def drone_dynamics(
        self,
        state: Float[Array, " n_states"],
        action: Float[Array, " n_actions"],
        wind_speed: Float[Array, " 2"],
    ) -> Float[Array, " n_states"]:
        """Compute the dynamics of the drone.

        Args:
            state: the current state of the car [x, y, z, yaw]
            action: the control action to take [vx, vy, vz, yaw rate]
            wind_speed: the wind speed in the x and y directions

        Returns:
            The next state of the drone
        """
        # Unpack the state
        x, y, z, yaw = state

        # Clip the velocities and unpack
        action = 2 * jax.nn.tanh(action)
        # vx, vy, vz, yaw_rate = action  todo
        v, yaw_rate = action
        vx = v * jnp.cos(yaw)
        vy = v * jnp.sin(yaw)
        vz = 0.0

        # Compute the next state
        x_next = x + (vx + wind_speed[0]) * self._dt
        y_next = y + (vy + wind_speed[1]) * self._dt
        z_next = z + vz * self._dt
        yaw_next = yaw + yaw_rate * self._dt

        return jnp.array([x_next, y_next, z_next, yaw_next])

    @jaxtyped(typechecker=beartype)
    def step(
        self,
        state: DroneState,
        action: Float[Array, " n_actions"],
        key: PRNGKeyArray,
    ) -> Tuple[DroneState, DroneObs, Float[Array, ""], Bool[Array, ""]]:
        """Take a step in the environment.

        The reward increases as the drone approaches the goal, with a penalty for
        colliding with any obstacle in the scene.

        Args:
            state: the current state of the environment
            action: the control action to take for the ego vehicle (acceleration and
                steering angle)
            key: a random number generator key

        Returns:
            The next state of the environment, the observations, the reward, and a
            boolean indicating whether the episode is done. Episodes end when the drone
            crashes or gets close enough to the goal.
        """
        # Unpack the state
        drone_state, tree_locations, tree_velocities, wind_speed = state

        # Compute the next state of the drone
        next_drone_state = drone_state
        for _ in range(self._substeps):
            next_drone_state = self.drone_dynamics(next_drone_state, action, wind_speed)
            # Compute the next state of the trees
            if self._moving_obstacles:
                tree_locations += self._dt * tree_velocities

        next_state = DroneState(
            next_drone_state, tree_locations, tree_velocities, wind_speed
        )

        # Compute the reward, which increases as the drone gets closer to the goal
        # at the origin and decreases if it collides with anything
        goal = jnp.array([0.0, 0.0, 1.0])  # just above the landing pad
        err = next_drone_state[:3] - goal
        distance_to_goal = jnp.sqrt(jnp.sum(err**2) + 1e-3)
        # And a big reward when we get close (currently disabled)
        distance_reward = (
            0 * self._collision_penalty * jax.nn.sigmoid(5 * (1.0 - distance_to_goal))
        )

        min_distance_to_obstacle = self._scene.check_for_collision(
            next_drone_state[:3],  # trim out yaw; not needed for collision checking
            wind_speed,
            tree_locations,
            self._render_sharpness,
        )
        collision_reward = -self._collision_penalty * jax.nn.sigmoid(
            -5 * min_distance_to_obstacle
        )

        # reward = distance_reward + collision_reward  #  / self._collision_penalty
        reward = collision_reward

        # The episode ends when a collision occurs or we get close to the goal, at which
        # point we reset the environment
        done = jnp.logical_or(min_distance_to_obstacle < 0.05, distance_to_goal < 0.25)
        done = jnp.logical_or(done, distance_to_goal > 15.0)
        done = jnp.logical_or(
            done, next_drone_state[0] > 0.5
        )  # stop if we go too far in x
        next_state = jax.lax.cond(
            done,
            lambda: self.reset(key),
            lambda: next_state,
        )

        # Compute the observations from a camera placed on the drone
        obs = self.get_obs(next_state)

        return next_state, obs, reward, done

    @jaxtyped(typechecker=beartype)
    def reset(self, key: PRNGKeyArray) -> DroneState:
        """Reset the environment.

        Args:
            key: a random number generator key

        Returns:
            The initial state of the environment.
        """
        # Split the PRNG key
        initial_state_key, tree_key, tree_vel_key, wind_key = jax.random.split(key, 4)

        # Sample a new initial state
        initial_drone_state = jax.random.multivariate_normal(
            initial_state_key,
            self._initial_drone_state_mean,
            jnp.diag(self._initial_drone_state_stddev**2),
        )

        # Sample new tree locations
        tree_locations = jax.random.uniform(
            tree_key,
            shape=(self._num_trees, 2),
            minval=jnp.array([-8.0, -4.0]),
            maxval=jnp.array([-1.0, 4.0]),
        )

        # Sample new tree velocities
        tree_vel_keys = jax.random.split(tree_vel_key, self._num_trees)
        tree_velocities = jax.vmap(
            jax.random.multivariate_normal, in_axes=(0, None, None)
        )(tree_vel_keys, jnp.zeros(2), jnp.eye(2) * 0.25**2)
        if not self._moving_obstacles:
            tree_velocities = jnp.zeros_like(tree_velocities)

        # Sample a new wind speed from a multivariate gaussian
        wind_speed = jax.random.multivariate_normal(
            wind_key, jnp.zeros(2), jnp.eye(2) * self._wind_speed_stddev**2
        )

        return DroneState(
            drone_state=initial_drone_state,
            tree_locations=tree_locations,
            tree_velocities=tree_velocities,
            wind_speed=wind_speed,
        )

    @jaxtyped(typechecker=beartype)
    def initial_state_logprior(self, state: DroneState) -> Float[Array, ""]:
        """
        Compute the prior logprobability of the given state.

        Args:
            state: the state to evaluate the prior at

        Returns:
            The logprior of the given state.
        """
        # Unpack the state
        drone_state, tree_locations, tree_velocities, wind_speed = state

        # Compute the prior logprobabilities

        # Drone state is sampled from a normal distribution
        drone_logprior = jax.scipy.stats.multivariate_normal.logpdf(
            drone_state,
            mean=self._initial_drone_state_mean,
            cov=jnp.diag(self._initial_drone_state_stddev**2),
        )

        # Tree locations are sampled from a uniform distribution
        def log_smooth_uniform(x, x_min, x_max):
            b = 50.0  # sharpness
            return log_sigmoid(b * (x - x_min)) + log_sigmoid(b * (x_max - x))

        num_trees = tree_locations.shape[0]
        tree_min_x = -8.0 * jnp.ones(num_trees)
        tree_max_x = -1.0 * jnp.ones(num_trees)
        tree_min_y = -4.0 * jnp.ones(num_trees)
        tree_max_y = 4.0 * jnp.ones(num_trees)
        tree_logprior = log_smooth_uniform(
            tree_locations[:, 0], tree_min_x, tree_max_x
        ).sum()
        tree_logprior += log_smooth_uniform(
            tree_locations[:, 1], tree_min_y, tree_max_y
        ).sum()

        # Tree velocities are sampled from a normal distribution
        tree_vel_logprior = jax.vmap(
            jax.scipy.stats.multivariate_normal.logpdf, in_axes=(0, None, None)
        )(
            tree_velocities,
            jnp.zeros(2),
            jnp.eye(2) * 0.25**2,
        ).sum()
        if not self._moving_obstacles:
            tree_vel_logprior = 0.0

        # Wind speed is also from a normal distribution
        wind_logprior = jax.scipy.stats.multivariate_normal.logpdf(
            wind_speed,
            mean=jnp.zeros(2),
            cov=jnp.eye(2) * self._wind_speed_stddev**2,
        )

        return drone_logprior + tree_logprior + wind_logprior + tree_vel_logprior

    @jaxtyped(typechecker=beartype)
    def get_obs(self, state: DroneState) -> DroneObs:
        """Get the observation from the given state.

        Args:
            state: the current state of the environment

        Returns:
            The observation from the given state.
        """
        # Render the depth image as seen by the drone
        drone_x, drone_y, drone_z, drone_yaw = state.drone_state
        camera_origin = jnp.array([drone_x, drone_y, drone_z])
        drone_heading_vector = jnp.array([jnp.cos(drone_yaw), jnp.sin(drone_yaw), 0])
        extrinsics = CameraExtrinsics(
            camera_origin=camera_origin,
            camera_R_to_world=look_at(
                camera_origin=camera_origin,
                target=camera_origin + drone_heading_vector,
                up=jnp.array([0, 0, 1.0]),
            ),
        )
        depth, color_image = self._scene.render_rgbd(
            self._camera_intrinsics,
            extrinsics,
            state.wind_speed,
            state.tree_locations,
            sharpness=self._render_sharpness,
            shading_light_direction=jnp.array([-1.0, -1.0, 5.0]),
        )
        obs = DroneObs(
            image=color_image,
            depth_image=depth,
            state=state.drone_state,
        )
        return obs
