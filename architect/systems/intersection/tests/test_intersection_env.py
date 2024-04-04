"""Test the intersection environment."""
import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest

from architect.systems.components.sensing.vision.render import CameraIntrinsics
from architect.systems.highway.highway_env import HighwayState
from architect.systems.intersection.env import IntersectionEnv
from architect.systems.intersection.scene import IntersectionScene


@pytest.fixture
def intersection_scene():
    """An intersection scene to use in testing."""
    return IntersectionScene()


@pytest.fixture
def intrinsics():
    """Camera intrinsics to use during testing."""
    intrinsics = CameraIntrinsics(
        focal_length=0.1,
        sensor_size=(0.1, 0.1),
        resolution=(64, 64),
    )
    return intrinsics


@pytest.fixture
def intersection_env(intersection_scene, intrinsics):
    """Highway environment system under test."""
    initial_ego_state = jnp.array([-20, -3.75, 0.0, 3.0])
    initial_non_ego_states = jnp.array(
        [
            [-3.75, 10, -jnp.pi / 2, 4.5],
            [-3.75, 20, -jnp.pi / 2, 4.5],
            [3.75, -30, jnp.pi / 2, 4.5],
        ]
    )
    initial_state_covariance = jnp.diag(jnp.array([0.5, 0.5, 0.001, 0.2]) ** 2)

    # Set the direction of light shading
    shading_light_direction = jnp.array([-0.2, -1.0, 1.5])
    shading_light_direction /= jnp.linalg.norm(shading_light_direction)
    shading_direction_covariance = (0.25) ** 2 * jnp.eye(3)

    env = IntersectionEnv(
        intersection_scene,
        intrinsics,
        dt=0.1,
        initial_ego_state=initial_ego_state,
        initial_non_ego_states=initial_non_ego_states,
        initial_state_covariance=initial_state_covariance,
        collision_penalty=5.0,
        mean_shading_light_direction=shading_light_direction,
        shading_light_direction_covariance=shading_direction_covariance,
    )
    return env


def test_intersection_env_init(intersection_env):
    """Test the highway environment initialization."""
    assert intersection_env is not None


def test_intersection_env_step(intersection_env):
    """Test the highway environment step."""
    # Create actions for the ego and non-ego agents (driving straight at fixed speed)
    ego_action = jnp.array([0.0, 0.0])
    non_ego_actions = jnp.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])

    # Initialize a state
    state = HighwayState(
        ego_state=jnp.array([-20, -3.75, 0.0, 3.0]),
        non_ego_states=jnp.array(
            [
                [-3.75, 10, -jnp.pi / 2, 4.5],
                [-3.75, 20, -jnp.pi / 2, 4.5],
                [3.75, -30, jnp.pi / 2, 4.5],
            ]
        ),
        shading_light_direction=jnp.array([1.0, 0.0, 0.0]),
        non_ego_colors=jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    )

    # Take a step
    next_state, obs, reward, done = intersection_env.step(
        state, ego_action, non_ego_actions, jrandom.PRNGKey(0)
    )

    assert next_state is not None
    assert obs is not None
    assert reward is not None
    assert done is not None


def test_intersection_env_step_grad(intersection_env):
    """Test gradients through the highway environment step."""

    def step_and_get_depth(x):
        # Create actions for the ego and non-ego agents (driving straight at fixed speed)
        ego_action = jnp.array([0.0, 0.0 + x])
        non_ego_actions = jnp.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])

        # Initialize a state
        state = HighwayState(
            ego_state=jnp.array([-20 - 0.1, -3.75, 0.0, 3.0]),
            non_ego_states=jnp.array(
                [
                    [-3.75, 10, -jnp.pi / 2, 4.5],
                    [-3.75, 20, -jnp.pi / 2, 4.5],
                    [3.75, -30, jnp.pi / 2, 4.5],
                ]
            ),
            shading_light_direction=jnp.array([1.0, 0.0, 0.0]),
            non_ego_colors=jnp.array(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            ),
        )

        # Take a step
        _, obs, _, _ = intersection_env.step(
            state, ego_action, non_ego_actions, jrandom.PRNGKey(0)
        )

        return obs

    # Test gradients
    render_depth = lambda x: step_and_get_depth(x).depth_image
    depth_image = render_depth(jnp.array(0.0))
    depth_grad = jax.jacfwd(render_depth)(jnp.array(0.0))

    assert depth_image is not None
    assert depth_grad is not None
    assert depth_grad.shape == depth_image.shape

    # import matplotlib.colors as mcolors
    # import matplotlib.pyplot as plt

    # fig, axs = plt.subplots(1, 2)
    # axs[0].imshow(depth_image.T)
    # grad_img = axs[1].imshow(depth_grad.T, cmap="bwr", norm=mcolors.CenteredNorm())
    # # Add a color bar for the gradient
    # fig.colorbar(grad_img)
    # plt.show()


def test_intersection_env_reset(intersection_env):
    """Test the highway environment reset."""
    state = intersection_env.reset(jrandom.PRNGKey(0))
    assert state is not None


def test_intersection_env_sample_non_ego_action(intersection_env):
    """Test sampling a non-ego action."""
    key = jrandom.PRNGKey(0)
    n_non_ego = 3
    n_actions = 2  # per agent
    noise_cov = jnp.diag(0.1 * jnp.ones(n_actions))
    non_ego_actions = intersection_env.sample_non_ego_actions(key, noise_cov, n_non_ego)
    assert non_ego_actions.shape == (n_non_ego, n_actions)


def test_intersection_env_non_ego_action_prior_logprob(intersection_env):
    """Test computing the log probability of a non-ego action."""
    n_non_ego = 3
    n_actions = 2  # per agent
    non_ego_actions = jnp.zeros((n_non_ego, n_actions))
    noise_cov = jnp.diag(0.1 * jnp.ones(n_actions))
    logprob = intersection_env.non_ego_actions_prior_logprob(non_ego_actions, noise_cov)
    assert logprob.shape == ()
