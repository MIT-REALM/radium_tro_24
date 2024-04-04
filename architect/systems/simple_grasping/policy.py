"""Define a neural network policy for the simple grasping environment."""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped

from architect.types import PRNGKeyArray


class AffordancePredictor(eqx.Module):
    """A neural network for predicting grasp affordances in a scene.

    The policy is structured as an image-to-image autoencoder. The input is a depth
    image of the scene and an output is a mask of that image predicting where the
    agent can grasp the object. Also outputs a 3D pose for each finger of the gripper.
    """

    encoder_conv_1: eqx.nn.Conv2d
    encoder_conv_2: eqx.nn.Conv2d
    encoder_conv_3: eqx.nn.Conv2d

    decoder_conv_1: eqx.nn.ConvTranspose2d
    decoder_conv_2: eqx.nn.ConvTranspose2d
    decoder_conv_3: eqx.nn.ConvTranspose2d

    fcn_1: eqx.nn.Linear
    fcn_2: eqx.nn.Linear
    fcn_3: eqx.nn.Linear

    @jaxtyped(typechecker=beartype)
    def __init__(
        self,
        key: PRNGKeyArray,
        cnn_channels: int = 32,
    ):
        """Initialize the policy."""
        encoder_key, decoder_key, fcn_key = jrandom.split(key, 3)

        # Create the convolutional encoder
        encoder_keys = jrandom.split(encoder_key, 3)
        kernel_size = 7
        stride = 2
        self.encoder_conv_1 = eqx.nn.Conv2d(
            key=encoder_keys[0],
            in_channels=1,
            out_channels=cnn_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        # Get new image shape after convolution
        self.encoder_conv_2 = eqx.nn.Conv2d(
            key=encoder_keys[1],
            in_channels=cnn_channels,
            out_channels=cnn_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.encoder_conv_3 = eqx.nn.Conv2d(
            key=encoder_keys[2],
            in_channels=cnn_channels,
            out_channels=cnn_channels,
            kernel_size=kernel_size,
            stride=stride,
        )

        # Create the convolutional decoder with a U-net architecture
        decoder_keys = jrandom.split(decoder_key, 3)
        self.decoder_conv_1 = eqx.nn.ConvTranspose2d(
            key=decoder_keys[0],
            in_channels=cnn_channels,
            out_channels=cnn_channels,
            kernel_size=kernel_size,
            stride=stride,
            output_padding=1,
        )
        self.decoder_conv_2 = eqx.nn.ConvTranspose2d(
            key=decoder_keys[1],
            in_channels=cnn_channels * 2,
            out_channels=cnn_channels,
            kernel_size=kernel_size,
            stride=stride,
            output_padding=0,
        )
        self.decoder_conv_3 = eqx.nn.ConvTranspose2d(
            key=decoder_keys[2],
            in_channels=cnn_channels * 2,
            out_channels=1,
            kernel_size=kernel_size,
            stride=stride,
            output_padding=1,
        )

        # Also create the fully connected layers for predicting the gripper finger poses
        fcn1_key, fcn2_key, fcn3_key = jrandom.split(fcn_key, 3)
        input_features = 32 * 3 * 3  # encoding size
        self.fcn_1 = eqx.nn.Linear(input_features, 32, key=fcn1_key)
        self.fcn_2 = eqx.nn.Linear(32, 32, key=fcn2_key)
        self.fcn_3 = eqx.nn.Linear(32, 2 * 3, key=fcn3_key)

    def forward(self, img: Float[Array, "w h"]) -> Float[Array, "w h"]:
        """Compute the affordance mask for the given image.

        Args:
            img: the input image

        Returns:
            The predicted affordance mask
        """
        # Run the encoder
        img = jnp.expand_dims(img, axis=0)
        # print(f"img shape: {img.shape}")
        enc1 = jax.nn.relu(self.encoder_conv_1(img))
        # print(f"enc1 shape: {enc1.shape}")
        enc2 = jax.nn.relu(self.encoder_conv_2(enc1))
        # print(f"enc2 shape: {enc2.shape}")
        enc3 = jax.nn.relu(self.encoder_conv_3(enc2))
        # print(f"enc3 shape: {enc3.shape}")

        # Run the decoder
        dec3 = jax.nn.relu(self.decoder_conv_1(enc3))
        # print(f"dec3 shape: {dec3.shape}")
        dec2_in = jnp.concatenate((dec3, enc2), axis=0)
        dec2 = jax.nn.relu(self.decoder_conv_2(dec2_in))
        # print(f"dec2 shape: {dec2.shape}")
        dec1_in = jnp.concatenate((dec2, enc1), axis=0)
        y = jax.nn.sigmoid(self.decoder_conv_3(dec1_in))
        # print(f"y shape: {y.shape}")
        affordance_mask = jnp.squeeze(y, axis=0)

        # Concatenate the affordance mask with the depth image
        fcn_input = enc3.reshape(-1)
        # print(f"fcn_input shape: {fcn_input.shape}")
        y = jax.nn.relu(self.fcn_1(fcn_input))
        # print(f"fcn1 shape: {y.shape}")
        y = jax.nn.relu(self.fcn_2(y))
        # print(f"fcn2 shape: {y.shape}")
        predicted_grasp_pose = self.fcn_3(y).reshape(2, 3)
        # print(f"fcn3 shape: {y.shape}")

        return affordance_mask, predicted_grasp_pose

    def __call__(self, img: Float[Array, "w h"]) -> Float[Array, "w h"]:
        """Compute the grasp affordance mask for the given image.

        Args:
            img: the input image

        Returns:
            The predicted affordance mask
        """
        return self.forward(img)
