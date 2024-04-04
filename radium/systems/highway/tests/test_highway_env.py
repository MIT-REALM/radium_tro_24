"""Test the highway environment."""

import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest

from radium.systems.components.sensing.vision.render import CameraIntrinsics
from radium.systems.highway.highway_env import HighwayEnv, HighwayState
from radium.systems.highway.highway_scene import HighwayScene


@pytest.fixture
def highway_scene():
    """A highway scene to use in testing."""
    return HighwayScene(num_lanes=3, lane_width=4.0)


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
def highway_env(highway_scene, intrinsics):
    """Highway environment system under test."""
    initial_ego_state = jnp.array([-15.0, 0.0, 0.0, 10.0])
    initial_non_ego_states = jnp.array(
        [
            [7.0, 0.0, 0.0, 10.0],
            [0.0, 4.0, 0.0, 9.0],
            [-5.0, -4.0, 0.0, 11.0],
        ]
    )
    return HighwayEnv(
        highway_scene,
        intrinsics,
        0.1,
        initial_ego_state,
        initial_non_ego_states,
        0.1 * jnp.eye(4),
        anti_alias_samples=1,
    )


def test_highway_env_init(highway_env):
    """Test the highway environment initialization."""
    assert highway_env is not None


def test_highway_env_step(highway_env):
    """Test the highway environment step."""
    # Create actions for the ego and non-ego agents (driving straight at fixed speed)
    ego_action = jnp.array([0.0, 0.0])
    non_ego_actions = jnp.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])

    # Initialize a state
    state = HighwayState(
        ego_state=jnp.array([-15.0, 0.0, 0.0, 10.0]),
        non_ego_states=jnp.array(
            [
                [7.0, 0.0, 0.0, 10.0],
                [0.0, 4.0, 0.0, 9.0],
                [-5.0, -4.0, 0.0, 11.0],
            ]
        ),
        shading_light_direction=jnp.array([1.0, 0.0, 0.0]),
        non_ego_colors=jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
    )

    # Take a step
    next_state, obs, reward, done = highway_env.step(
        state, ego_action, non_ego_actions, jrandom.PRNGKey(0)
    )

    assert next_state is not None
    assert obs is not None
    assert reward is not None
    assert done is not None


def test_highway_env_step_grad(highway_env):
    """Test gradients through the highway environment step."""

    def step_and_get_depth(x):
        # Create actions for the ego and non-ego agents (driving straight at fixed speed)
        ego_action = jnp.array([0.0, 0.0 + x])
        non_ego_actions = jnp.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])

        # Initialize a state
        state = HighwayState(
            ego_state=jnp.array([-15.0, 0.0, 0.0, 10.0]),
            non_ego_states=jnp.array(
                [
                    [7.0, 0.0, 0.0, 10.0],
                    [0.0, 4.0, 0.0, 9.0],
                    [-5.0, -4.0, 0.0, 11.0],
                ]
            ),
            shading_light_direction=jnp.array([1.0, 0.0, 0.0]),
            non_ego_colors=jnp.array(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            ),
        )

        # Take a step
        _, obs, _, _ = highway_env.step(
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

    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(depth_image.T)
    grad_img = axs[1].imshow(depth_grad.T, cmap="bwr", norm=mcolors.CenteredNorm())
    # Add a color bar for the gradient
    fig.colorbar(grad_img)
    plt.show()


def test_highway_env_reset(highway_env):
    """Test the highway environment reset."""
    state = highway_env.reset(jrandom.PRNGKey(0))
    assert state is not None


def test_highway_env_sample_non_ego_action(highway_env):
    """Test sampling a non-ego action."""
    key = jrandom.PRNGKey(0)
    n_non_ego = 3
    n_actions = 2  # per agent
    noise_cov = jnp.diag(0.1 * jnp.ones(n_actions))
    non_ego_actions = highway_env.sample_non_ego_actions(key, noise_cov, n_non_ego)
    assert non_ego_actions.shape == (n_non_ego, n_actions)


def test_highway_env_non_ego_action_prior_logprob(highway_env):
    """Test computing the log probability of a non-ego action."""
    n_non_ego = 3
    n_actions = 2  # per agent
    non_ego_actions = jnp.zeros((n_non_ego, n_actions))
    noise_cov = jnp.diag(0.1 * jnp.ones(n_actions))
    logprob = highway_env.non_ego_actions_prior_logprob(non_ego_actions, noise_cov)
    assert logprob.shape == ()
