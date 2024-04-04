"""Define a neural network policy for driving in the highway environment."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, Float, jaxtyped

from radium.systems.highway.highway_env import HighwayObs
from radium.types import PRNGKeyArray


class DrivingPolicy(eqx.Module):
    """A neural network actor-critic policy for driving in the highway environment.

    The policy has a convolutional encoder to compute a latent embedding of the
    given image observation, and a fully-connected final to compute the action
    based on the latent embedding concatenated with the current forward velocity.
    """

    encoder_conv_1: eqx.nn.Conv2d
    encoder_conv_2: eqx.nn.Conv2d
    encoder_conv_3: eqx.nn.Conv2d

    actor_fcn: eqx.nn.MLP
    critic_fcn: eqx.nn.MLP

    log_action_std: Float[Array, ""]

    @jaxtyped(typechecker=beartype)
    def __init__(
        self,
        key: PRNGKeyArray,
        image_shape: Tuple[int, int],
        image_channels: int = 4,
        cnn_channels: int = 32,
        fcn_layers: int = 2,
        fcn_width: int = 32,
    ):
        """Initialize the policy."""
        cnn_key, fcn_key = jrandom.split(key)

        # Create the convolutional encoder
        cnn_keys = jrandom.split(cnn_key, 3)
        kernel_size = 6
        stride = 1  # 2 for 64x64
        self.encoder_conv_1 = eqx.nn.Conv2d(
            key=cnn_keys[0],
            in_channels=image_channels,
            out_channels=cnn_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        # Get new image shape after convolution
        image_shape = jnp.floor(
            (jnp.array(image_shape) - (kernel_size - 1) - 1) / stride + 1
        )
        self.encoder_conv_2 = eqx.nn.Conv2d(
            key=cnn_keys[1],
            in_channels=cnn_channels,
            out_channels=cnn_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        image_shape = jnp.floor((image_shape - (kernel_size - 1) - 1) / stride + 1)
        self.encoder_conv_3 = eqx.nn.Conv2d(
            key=cnn_keys[2],
            in_channels=cnn_channels,
            out_channels=1,
            kernel_size=kernel_size,
            stride=stride,
        )
        image_shape = jnp.floor((image_shape - (kernel_size - 1) - 1) / stride + 1)
        embedding_size = image_shape[0] * image_shape[1]
        embedding_size = int(embedding_size)

        actor_key, critic_key = jrandom.split(fcn_key)
        # Create the fully connected layers
        self.actor_fcn = eqx.nn.MLP(
            in_size=embedding_size + 1,
            out_size=2,
            width_size=fcn_width,
            depth=fcn_layers,
            key=actor_key,
        )
        self.critic_fcn = eqx.nn.MLP(
            in_size=embedding_size + 1,
            out_size=1,
            width_size=fcn_width,
            depth=fcn_layers,
            key=critic_key,
        )

        # Initialize action standard deviation
        self.log_action_std = jnp.log(jnp.array(0.1))

    def forward(self, obs: HighwayObs) -> Tuple[Float[Array, " 2"], Float[Array, ""]]:
        """Compute the mean action and value estimate for the given state.

        Args:
            obs: The observation of the current state.

        Returns:
            The mean action and value estimate.
        """
        # Compute the image embedding
        depth_image = obs.depth_image
        depth_image = jnp.expand_dims(depth_image, axis=0)
        rgbd = jnp.concatenate(
            (depth_image, obs.color_image.transpose(2, 0, 1)), axis=0
        )
        y = jax.nn.relu(self.encoder_conv_1(rgbd))
        y = jax.nn.relu(self.encoder_conv_2(y))
        y = jax.nn.relu(self.encoder_conv_3(y))
        y = jnp.reshape(y, (-1,))

        # Concatenate the embedding with the forward velocity
        y = jnp.concatenate((y, obs.speed.reshape(-1)))

        # Compute the action and value estimate
        value = self.critic_fcn(y).reshape()  # scalar output
        action_mean = self.actor_fcn(y)

        return action_mean, value

    def __call__(
        self, obs: HighwayObs, key: PRNGKeyArray, deterministic=True
    ) -> Tuple[Float[Array, " 2"], Float[Array, ""], Float[Array, ""]]:
        """Compute the action and value estimate for the given state.

        Args:
            obs: The observation of the current state.
            key: The random key for sampling the action.
            deterministic: Whether to sample the action from the policy distribution
                or to return the mean action.

        Returns:
            The action, action log probability, and value estimate.
        """
        action_mean, value = self.forward(obs)
        action_noise = jnp.exp(self.log_action_std)

        if deterministic:
            # Return the mean action, a constant logprob, and the value estimate
            return action_mean, jnp.array(0.0), value

        action = jrandom.multivariate_normal(
            key, action_mean, action_noise * jnp.eye(action_mean.shape[0])
        )
        action_logp = jax.scipy.stats.multivariate_normal.logpdf(
            action, action_mean, action_noise * jnp.eye(action_mean.shape[0])
        )

        return action, action_logp, value

    def action_log_prob_and_value(
        self, obs: HighwayObs, action: Float[Array, " 2"]
    ) -> Tuple[Float[Array, ""], Float[Array, ""]]:
        """Compute the log probability of an action with the given observation.

        Args:
            obs: The observation of the current state.
            action: The action to compute the log probability of.
            action_noise: The standard deviation of the Gaussian noise to add

        Returns:
            The log probability of the action and the value estimate.
        """
        action_mean, value = self.forward(obs)
        action_noise = jnp.exp(self.log_action_std)

        action_logp = jax.scipy.stats.multivariate_normal.logpdf(
            action, action_mean, action_noise * jnp.eye(action_mean.shape[0])
        )

        return action_logp, value

    def entropy(self) -> Float[Array, ""]:
        """Return the entropy of the action distribution."""
        action_noise = jnp.exp(self.log_action_std)
        return (
            0.5 * jnp.linalg.slogdet(2 * jnp.pi * jnp.e * action_noise * jnp.eye(2))[1]
        )
