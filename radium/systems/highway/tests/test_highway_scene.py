"""Test the highway scene."""
import jax
import jax.numpy as jnp

from radium.systems.components.sensing.vision.render import (
    CameraExtrinsics,
    CameraIntrinsics,
)
from radium.systems.components.sensing.vision.util import look_at
from radium.systems.highway.highway_env import HighwayScene


def test_highway_scene():
    # Create a test highway scene and render it
    highway = HighwayScene(num_lanes=3, lane_width=4.0)
    car_states = jnp.array(
        [
            [7.0, 0.0, 0.0],
            [0.0, highway.lane_width, 0.0],
            [-5.0, -highway.lane_width, 0.0],
        ]
    )

    # Set the camera parameters
    intrinsics = CameraIntrinsics(
        focal_length=0.1,
        sensor_size=(0.1, 0.1),
        resolution=(512, 512),
    )
    extrinsics = CameraExtrinsics(
        camera_origin=jnp.array([-10.0, 0.0, 10.0]),
        camera_R_to_world=look_at(jnp.array([-10.0, 0.0, 10.0]), jnp.zeros(3)),
    )

    depth_image = highway.render_depth(intrinsics, extrinsics, car_states)

    assert depth_image is not None


def test_highway_scene_depth_grad():
    # Create a test highway scene and render it
    highway = HighwayScene(num_lanes=3, lane_width=4.0)

    def render_depth(x):
        car_states = jnp.array(
            [
                [7.0, 0.0, 0.0 + x],
                [0.0, highway.lane_width, 0.0],
                [-5.0, -highway.lane_width, 0.0],
            ]
        )

        # Set the camera parameters
        intrinsics = CameraIntrinsics(
            focal_length=0.1,
            sensor_size=(0.1, 0.1),
            resolution=(64, 64),
        )
        extrinsics = CameraExtrinsics(
            camera_origin=jnp.array([-10.0, 0.0, 10.0]),
            camera_R_to_world=look_at(jnp.array([-10.0, 0.0, 10.0]), jnp.zeros(3)),
        )

        depth_image = highway.render_depth(intrinsics, extrinsics, car_states)

        return depth_image

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
