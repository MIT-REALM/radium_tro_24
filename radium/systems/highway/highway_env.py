"""Manage the state of the ego car and all other vehicles on the highway."""

import jax
import jax.nn
import jax.numpy as jnp
import jax.random as jrandom
from beartype import beartype
from beartype.typing import NamedTuple, Tuple
from jaxtyping import Array, Bool, Float, jaxtyped

from radium.systems.components.sensing.vision.render import (
    CameraExtrinsics,
    CameraIntrinsics,
)
from radium.systems.components.sensing.vision.util import look_at
from radium.systems.highway.highway_scene import HighwayScene
from radium.types import PRNGKeyArray


# @beartype
class HighwayObs(NamedTuple):
    """Observations returned from the highway environment.

    Attributes:
        speed: the forward speed of the ego vehicle
        depth_image: a depth image rendered from the front of the ego vehicle
    """

    speed: Float[Array, " *batch"]
    depth_image: Float[Array, "*batch res_x res_y"]
    color_image: Float[Array, "*batch res_x res_y"]
    ego_state: Float[Array, "*batch n_states"]


# @beartype
class HighwayState(NamedTuple):
    """The state of the ego vehicle and all other vehicles on the highway.

    Attributes:
        ego_state: the state of the ego vehicle
        non_ego_states: the states of all other vehicles
        shading_light_direction: the direction of the light source for rendering
        non_ego_colors: the colors of all non-ego agents
    """

    ego_state: Float[Array, " n_states"]
    non_ego_states: Float[Array, "n_non_ego n_states"]
    shading_light_direction: Float[Array, " 3"]
    non_ego_colors: Float[Array, "n_non_ego 3"]


class HighwayEnv:
    """
    The highway includes one ego vehicle and some number of other vehicles. Each vehicle
    has state [x, y, theta, v] (position, heading, forward speed) and simple dynamics
    [1] with control inputs [a, delta] (acceleration, steering angle).

    The scene yields observations of the current vehicle's forward speed and the
    depth image rendered from the front of the ego vehicle.

    References:
        [1] https://msl.cs.uiuc.edu/planning/node658.html

    Args:
        highway_scene: a representation of the underlying highway scene.
        camera_intrinsics: the intrinsics of the camera mounted to the ego agent.
        dt: the time step to use for simulation.
        initial_ego_state: the mean initial state of the ego vehicle.
        initial_non_ego_states: the mean initial states of all non-ego vehicles.
        initial_state_covariance: the initial state covariance of all vehicles.
        mean_shading_light_direction: the mean direction of the light source for
            rendering.
        shading_light_direction_covariance: the covariance of the position of the light
            source for rendering.
        collision_penalty: the penalty to apply when the ego vehicle collides with
            any obstacle in the scene.
        max_render_dist: the maximum distance to render in the depth image.
        render_sharpness: the sharpness of the scene.
        anti_alias_samples: the number of samples to use for anti-aliasing.
    """

    _highway_scene: HighwayScene
    _camera_intrinsics: CameraIntrinsics
    _dt: float
    _collision_penalty: float
    _max_render_dist: float
    _render_sharpness: float
    _initial_ego_state: Float[Array, " n_states"]
    _initial_non_ego_states: Float[Array, "n_non_ego n_states"]
    _initial_state_covariance: Float[Array, "n_states n_states"]
    _mean_shading_light_direction: Float[Array, " 3"]
    _shading_light_direction_covariance: Float[Array, " 3 3"]

    _axle_length: float = 1.0

    @jaxtyped(typechecker=beartype)
    def __init__(
        self,
        highway_scene: HighwayScene,
        camera_intrinsics: CameraIntrinsics,
        dt: float,
        initial_ego_state: Float[Array, " n_states"],
        initial_non_ego_states: Float[Array, "n_non_ego n_states"],
        initial_state_covariance: Float[Array, "n_states n_states"],
        mean_shading_light_direction: Float[Array, " 3"] = jnp.array([0.0, 0.0, 1.0]),
        shading_light_direction_covariance: Float[Array, " 3 3"] = jnp.eye(3),
        collision_penalty: float = 50.0,
        max_render_dist: float = 30.0,
        render_sharpness: float = 100.0,
        anti_alias_samples: int = 1,
    ):
        """Initialize the environment."""
        self._highway_scene = highway_scene
        self._dt = dt
        self._initial_ego_state = initial_ego_state
        self._initial_non_ego_states = initial_non_ego_states
        self._initial_state_covariance = initial_state_covariance
        self._mean_shading_light_direction = mean_shading_light_direction
        self._shading_light_direction_covariance = shading_light_direction_covariance
        self._collision_penalty = collision_penalty
        self._max_render_dist = max_render_dist
        self._render_sharpness = render_sharpness

        # Increase the resolution so we can down-sample later to anti-alias
        self._camera_intrinsics = CameraIntrinsics(
            resolution=(
                camera_intrinsics.resolution[0] * anti_alias_samples,
                camera_intrinsics.resolution[1] * anti_alias_samples,
            ),
            focal_length=camera_intrinsics.focal_length,
            sensor_size=camera_intrinsics.sensor_size,
        )
        self._anti_alias_samples = anti_alias_samples

    @jaxtyped(typechecker=beartype)
    def car_dynamics(
        self,
        state: Float[Array, " n_states"],
        action: Float[Array, " n_actions"],
    ) -> Float[Array, " n_states"]:
        """Compute the dynamics of the car.

        Args:
            state: the current state of the car [x, y, theta, v]
            action: the control action to take [acceleration, steering angle]

        Returns:
            The next state of the car
        """
        # Unpack the state and action
        x, y, theta, v = state
        a, delta = action

        # Clip the steering angle
        delta = jnp.clip(delta, -0.1, 0.1)

        # Clip the acceleration
        a = jnp.clip(a, -2.0, 2.0)

        # Compute the next state
        x_next = x + v * jnp.cos(theta) * self._dt
        y_next = y + v * jnp.sin(theta) * self._dt
        theta_next = theta + v * jnp.tan(delta) / self._axle_length * self._dt
        v_next = v + a * self._dt

        # Clip the velocity to some maximum
        v_next = jnp.clip(v_next, 0.0, 20.0)

        return jnp.array([x_next, y_next, theta_next, v_next])

    @jaxtyped(typechecker=beartype)
    def step(
        self,
        state: HighwayState,
        ego_action: Float[Array, " n_actions"],
        non_ego_actions: Float[Array, "n_non_ego n_actions"],
        key: PRNGKeyArray,
        reset: bool = True,
        collision_sharpness: float = 5.0,
    ) -> Tuple[HighwayState, HighwayObs, Float[Array, ""], Bool[Array, ""]]:
        """Take a step in the environment.

        The reward is the distance travelled in the positive x direction, minus a
        penalty if the ego vehicle collides with any other object in the scene.

        Args:
            state: the current state of the environment
            ego_action: the control action to take for the ego vehicle (acceleration and
                steering angle)
            non_ego_actions: the control action to take for all other vehicles
                (acceleration and steering angle)
            key: a random number generator key
            reset: whether to reset the environment.
            collision_sharpness: how sharp to make the collision penalty

        Returns:
            The next state of the environment, the observations, the reward, and a
            boolean indicating whether the episode is done. Episodes end when the ego
            car crashes into any other object in the scene.
        """
        # Unpack the state
        ego_state, non_ego_states, _, _ = state

        # Compute the next state of the ego and other vehicles
        next_ego_state = self.car_dynamics(ego_state, ego_action)
        next_non_ego_states = jax.vmap(self.car_dynamics)(
            non_ego_states, non_ego_actions
        )
        next_state = HighwayState(
            next_ego_state,
            next_non_ego_states,
            state.shading_light_direction,
            state.non_ego_colors,
        )

        # Compute the reward, which increases as the vehicle travels farther in
        # the positive x direction and decreases if it collides with anything
        min_distance_to_obstacle = self._highway_scene.check_for_collision(
            next_ego_state[:3],  # trim out speed; not needed for collision checking
            next_non_ego_states[:, :3],
            self._render_sharpness,
            include_ground=False,
        )
        collision_reward = -self._collision_penalty * jax.nn.sigmoid(
            -collision_sharpness * min_distance_to_obstacle
        )
        distance_reward = 1.0 * (next_ego_state[0] - ego_state[0]) / self._dt
        # lane_keeping_reward = (
        #     0.0 * (next_ego_state[1] - self._initial_ego_state[1]) ** 2
        # )
        # reward = distance_reward + lane_keeping_reward + collision_reward
        reward = 0.1 * distance_reward + collision_reward

        # The episode ends when a collision occurs, at which point we reset the
        # environment (or if we run out of road)
        done = jnp.logical_or(min_distance_to_obstacle < 0.0, next_ego_state[0] > 90.0)
        next_state = jax.lax.cond(
            jnp.logical_and(done, reset),
            lambda: self.reset(key),
            lambda: next_state,
        )

        # Compute the observations from a camera placed on the ego vehicle
        obs = self.get_obs(next_state)

        return next_state, obs, reward, done

    @jaxtyped(typechecker=beartype)
    def reset(self, key: PRNGKeyArray) -> HighwayState:
        """Reset the environment.

        Args:
            key: a random number generator key

        Returns:
            The initial state of the environment.
        """
        ego_state_key, non_ego_state_key, light_key, color_key = jrandom.split(key, 4)

        # Sample new the initial states
        non_ego_state = self.sample_initial_non_ego_states(non_ego_state_key)
        ego_state = self.sample_initial_ego_state(ego_state_key)

        # Sample a new lighting direction
        shading_light_direction = self.sample_shading_light_direction(light_key)

        # Sample new colors for the non-ego vehicles
        non_ego_colors = self.sample_non_ego_colors(color_key)

        return HighwayState(
            ego_state=ego_state,
            non_ego_states=non_ego_state,
            shading_light_direction=shading_light_direction,
            non_ego_colors=non_ego_colors,
        )

    @jaxtyped(typechecker=beartype)
    def get_obs(self, state: HighwayState) -> HighwayObs:
        """Get the observation from the given state.

        Args:
            state: the current state of the environment

        Returns:
            The observation from the given state.
        """
        # Render the depth image as seen by the ego agent
        ego_x, ego_y, ego_theta, ego_v = state.ego_state
        camera_origin = jnp.array([ego_x, ego_y, 0.75 * self._highway_scene.car.height])
        ego_heading_vector = jnp.array([jnp.cos(ego_theta), jnp.sin(ego_theta), 0])
        extrinsics = CameraExtrinsics(
            camera_origin=camera_origin,
            camera_R_to_world=look_at(
                camera_origin=camera_origin,
                target=camera_origin + ego_heading_vector,
                up=jnp.array([0, 0, 1.0]),
            ),
        )
        depth_image, color_image = self._highway_scene.render_rgbd(
            self._camera_intrinsics,
            extrinsics,
            state.non_ego_states[:, :3],  # trim out speed; not needed for rendering
            max_dist=self._max_render_dist,
            sharpness=self._render_sharpness,
            shading_light_direction=state.shading_light_direction,
            car_colors=state.non_ego_colors,
        )

        # Down-sample the image to anti-alias
        depth_image = jax.image.resize(
            depth_image,
            (
                self._camera_intrinsics.resolution[0] // self._anti_alias_samples,
                self._camera_intrinsics.resolution[1] // self._anti_alias_samples,
            ),
            method=jax.image.ResizeMethod.LINEAR,
        )
        color_image = jax.image.resize(
            color_image,
            (
                self._camera_intrinsics.resolution[0] // self._anti_alias_samples,
                self._camera_intrinsics.resolution[1] // self._anti_alias_samples,
                3,
            ),
            method=jax.image.ResizeMethod.LINEAR,
        )

        obs = HighwayObs(
            speed=ego_v,
            depth_image=depth_image,
            color_image=color_image,
            ego_state=state.ego_state,
        )
        return obs

    @jaxtyped(typechecker=beartype)
    def sample_initial_ego_state(self, key: PRNGKeyArray) -> Float[Array, " n_states"]:
        """Sample an initial state for the ego vehicle.

        Args:
            key: the random number generator key.

        Returns:
            The initial state of the ego vehicle.
        """
        initial_state = jax.random.multivariate_normal(
            key, self._initial_ego_state, self._initial_state_covariance
        )
        return initial_state

    @jaxtyped(typechecker=beartype)
    def initial_ego_state_prior_logprob(
        self,
        state: Float[Array, " n_states"],
    ) -> Float[Array, ""]:
        """Compute the prior log probability of an initial state for the ego vehicle.

        Args:
            state: the state of the ego vehicle at which to compute the log probability.

        Returns:
            The prior log probability of the givens state.
        """
        logprob = jax.scipy.stats.multivariate_normal.logpdf(
            state, self._initial_ego_state, self._initial_state_covariance
        )
        return logprob

    @jaxtyped(typechecker=beartype)
    def sample_initial_non_ego_states(
        self, key: PRNGKeyArray
    ) -> Float[Array, "n_non_ego n_states"]:
        """Sample initial states for the non-ego vehicles.

        Args:
            key: the random number generator key.

        Returns:
            The initial state of the ego vehicle.
        """
        n_non_ego = self._initial_non_ego_states.shape[0]
        keys = jrandom.split(key, n_non_ego)
        sample_single_state = lambda key, state_mean: jax.random.multivariate_normal(
            key, state_mean, self._initial_state_covariance
        )
        initial_states = jax.vmap(sample_single_state)(
            keys, self._initial_non_ego_states
        )
        return initial_states

    @jaxtyped(typechecker=beartype)
    def initial_non_ego_states_prior_logprob(
        self,
        state: Float[Array, "n_non_ego n_states"],
    ) -> Float[Array, ""]:
        """Compute the prior log probability of initial states for the non-ego vehicles.

        Args:
            state: the state of the non-ego vehicle at which to compute the log
                probability.

        Returns:
            The prior log probability of the givens state.
        """
        single_state_logprob = (
            lambda s, s_mean: jax.scipy.stats.multivariate_normal.logpdf(
                s, s_mean, self._initial_state_covariance
            )
        )
        initial_state_logprobs = jax.vmap(single_state_logprob)(
            state, self._initial_non_ego_states
        )
        return initial_state_logprobs.sum()

    @jaxtyped(typechecker=beartype)
    def sample_non_ego_actions(
        self,
        key: PRNGKeyArray,
        noise_cov: Float[Array, "n_actions n_actions"],
        n_non_ego: int,
    ) -> Float[Array, "n_non_ego n_actions"]:
        """Sample an action for the non-ego vehicles.

        Args:
            key: the random number generator key.
            noise_cov: the covariance matrix of the Gaussian noise to add to the
                zero action.
            n_non_ego: the number of non-ego vehicles in the scene.

        Returns:
            The initial state of the ego vehicle.
        """
        action = jax.random.multivariate_normal(
            key, jnp.zeros((2,)), noise_cov, shape=(n_non_ego,)
        )
        return action

    @jaxtyped(typechecker=beartype)
    def non_ego_actions_prior_logprob(
        self,
        action: Float[Array, "n_non_ego n_actions"],
        noise_cov: Float[Array, "n_actions n_actions"],
    ) -> Float[Array, ""]:
        """Compute the prior log probability of an action for the non-ego vehicles.

        Args:
            action: the action of the non-ego vehicles at which to compute the log
                probability.
            noise_cov: the covariance matrix of the Gaussian noise added to the action.

        Returns:
            The prior log probability of the givens action.
        """
        logprob_fn = lambda a: jax.scipy.stats.multivariate_normal.logpdf(
            a, jnp.zeros_like(a), noise_cov
        )
        logprobs = jax.vmap(logprob_fn)(action)
        return logprobs.sum()

    @jaxtyped(typechecker=beartype)
    def sample_shading_light_direction(self, key: PRNGKeyArray) -> Float[Array, " 3"]:
        """Sample light direction.

        Args:
            key: the random number generator key.

        Returns:
            A direction for the shading light.
        """
        direction = jax.random.multivariate_normal(
            key,
            self._mean_shading_light_direction,
            self._shading_light_direction_covariance,
        )
        return direction

    @jaxtyped(typechecker=beartype)
    def shading_light_direction_prior_logprob(
        self,
        direction: Float[Array, " 3"],
    ) -> Float[Array, ""]:
        """Compute the prior log probability of a shading light direction.

        Args:
            direction: the lighting direction of which to compute the log probability.

        Returns:
            The prior log probability of the givens direction.
        """
        logprob = jax.scipy.stats.multivariate_normal.logpdf(
            direction,
            self._mean_shading_light_direction,
            self._shading_light_direction_covariance,
        )
        return logprob

    @jaxtyped(typechecker=beartype)
    def sample_non_ego_colors(self, key: PRNGKeyArray) -> Float[Array, "n_non_ego 3"]:
        """Sample RGB colors for each non-ego agent uniformly at random.

        Args:
            key: the random number generator key.

        Returns:
            An array of RGB colors for each non-ego agent.
        """
        n_non_ego = self._initial_non_ego_states.shape[0]
        color = jax.random.uniform(key, shape=(n_non_ego, 3))
        return color

    @jaxtyped(typechecker=beartype)
    def non_ego_colors_prior_logprob(
        self,
        color: Float[Array, "n_non_ego 3"],
    ) -> Float[Array, ""]:
        """Compute the prior log probability of non-ego vehicle colors.

        These colors are sampled uniformly at random with each element between 0 and 1.

        Args:
            color: the colors of which to compute the log probability.

        Returns:
            The prior log probability of the given colors.
        """
        # Define the smoothing constant
        b = 50.0

        def log_smooth_uniform(x, x_min, x_max):
            return jax.nn.log_sigmoid(b * (x - x_min)) + jax.nn.log_sigmoid(
                b * (x_max - x)
            )

        logprob = log_smooth_uniform(color, 0.0, 1.0).sum()
        return logprob

    @jaxtyped(typechecker=beartype)
    def overall_prior_logprob(self, state: HighwayState) -> Float[Array, ""]:
        """Compute the overall prior logprobability of the given initial state"""
        return (
            self.initial_ego_state_prior_logprob(state.ego_state)
            + self.initial_non_ego_states_prior_logprob(state.non_ego_states)
            + self.shading_light_direction_prior_logprob(state.shading_light_direction)
            + self.non_ego_colors_prior_logprob(state.non_ego_colors)
        )
