"""Define a neural network policy for the drone landing environment."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import Array, Float, jaxtyped

from radium.systems.drone_landing.env import DroneObs
from radium.types import PRNGKeyArray


class DroneLandingPolicy(eqx.Module):
    """A neural network actor-critic policy for drone landing.

    The policy has a convolutional encoder to compute a latent embedding of the
    given image observation, and a fully-connected final to compute the action
    based on the latent embedding concatenated with the current forward velocity.
    """

    encoder_conv_1: eqx.nn.Conv2d
    encoder_conv_2: eqx.nn.Conv2d
    encoder_conv_3: eqx.nn.Conv2d

    actor_fcn: eqx.nn.MLP

    @jaxtyped(typechecker=beartype)
    def __init__(
        self,
        key: PRNGKeyArray,
        image_shape: Tuple[int, int],
        image_channels: int = 4,
        cnn_channels: int = 32,
        fcn_layers: int = 4,
        fcn_width: int = 64,
    ):
        """Initialize the policy."""
        cnn_key, fcn_key = jrandom.split(key)

        # Create the convolutional encoder
        cnn_keys = jrandom.split(cnn_key, 3)
        kernel_size = 7  # 6 for 64x64, 7 for 32x32
        stride = 1  # 2 for 64x64, 1 for 32x32
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
        embedding_size = int(embedding_size) + 5  # for drone state

        # Create the fully connected layers
        self.actor_fcn = eqx.nn.MLP(
            in_size=embedding_size,
            out_size=2,  # todo 4,
            width_size=fcn_width,
            depth=fcn_layers,
            key=fcn_key,
        )

    def forward(self, obs: DroneObs) -> Float[Array, " 2"]:
        """Compute the action for the given observation.

        Args:
            obs: The observation of the current state.

        Returns:
            The predicted action
        """
        # Compute the image embedding
        rgb = obs.image.transpose(2, 0, 1)
        depth_image = obs.depth_image
        depth_image = jnp.expand_dims(depth_image, axis=0)
        rgbd = jnp.concatenate((depth_image, rgb), axis=0)
        y = jax.nn.relu(self.encoder_conv_1(rgbd))
        y = jax.nn.relu(self.encoder_conv_2(y))
        y = jax.nn.relu(self.encoder_conv_3(y))
        y = jnp.reshape(y, (-1,))

        # Concatenate the drone state
        y = jnp.concatenate(
            (
                y,
                obs.state[:3].reshape(-1),
                jnp.cos(obs.state[3]).reshape(-1),
                jnp.sin(obs.state[3]).reshape(-1),
            )
        )

        # Compute the action
        action = self.actor_fcn(y)
        return action

    def __call__(self, obs: DroneObs) -> Float[Array, " 2"]:
        """Compute the action and value estimate for the given state.

        Args:
            obs: The observation of the current state.

        Returns:
            The predicted action
        """
        return self.forward(obs)
