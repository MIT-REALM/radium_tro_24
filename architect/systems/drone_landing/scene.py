"""Define a drone landing scene with a variable number of trees and cars."""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import List, Optional, Tuple
from jaxtyping import Array, Float, jaxtyped

from architect.systems.components.sensing.vision.render import (
    CameraExtrinsics,
    CameraIntrinsics,
    pinhole_camera_rays,
    raycast,
    render_color,
    render_depth,
    render_shadows,
)
from architect.systems.components.sensing.vision.shapes import (
    Box,
    Cylinder,
    Halfspace,
    Scene,
    SDFShape,
)


@beartype
class DroneLandingScene:
    """Represent a drone landing scene with landing pad and wind sock."""

    ground: Halfspace
    landing_pad: List[Box]

    @beartype
    def __init__(
        self,
    ):
        """Initialize the drone landing scene."""
        # Create the ground plane
        self.ground = Halfspace(
            point=jnp.array([0.0, 0.0, 0.0]),
            normal=jnp.array([0.0, 0.0, 1.0]),
            c=jnp.array([229, 220, 197]) / 255.0,
        )

        # Create the landing pad with a light-colored square and a red H
        self.landing_pad = [
            # Pad
            Box(
                center=jnp.array([0.0, 0.0, 0.0]),
                extent=jnp.array([1.0, 1.0, 0.01]),
                rotation=jnp.eye(3),
                c=jnp.array([0.1, 0.1, 0.1]),
            ),
            # H - left vertical bar
            Box(
                center=jnp.array([0.0, -0.25, 0.0]),
                extent=jnp.array([0.5, 0.01, 0.02]),
                rotation=jnp.eye(3),
                c=jnp.array([1.0, 0.0, 0.0]),
            ),
            # H - right vertical bar
            Box(
                center=jnp.array([0.0, 0.25, 0.0]),
                extent=jnp.array([0.5, 0.01, 0.02]),
                rotation=jnp.eye(3),
                c=jnp.array([1.0, 0.0, 0.0]),
            ),
            # H - horizontal bar
            Box(
                center=jnp.array([0.0, 0.0, 0.0]),
                extent=jnp.array([0.01, 0.5, 0.02]),
                rotation=jnp.eye(3),
                c=jnp.array([1.0, 0.0, 0.0]),
            ),
        ]

        # Create the walls on either side of the corridor to the landing pad
        self.walls = [
            # Left wall
            Box(
                center=jnp.array([0.0, 4, 0.5]),
                extent=jnp.array([30.0, 0.5, 1.0]),
                rotation=jnp.eye(3),
                c=jnp.array([0.1, 0.1, 0.1]),
            ),
            # Right wall
            Box(
                center=jnp.array([0.0, -4, 0.5]),
                extent=jnp.array([30.0, 0.5, 1.0]),
                rotation=jnp.eye(3),
                c=jnp.array([0.1, 0.1, 0.1]),
            ),
            # Back wall
            Box(
                center=jnp.array([-15.0, 0.0, 0.5]),
                extent=jnp.array([0.5, 8.0, 1.0]),
                rotation=jnp.eye(3),
                c=jnp.array([0.1, 0.1, 0.1]),
            ),
        ]

    @jaxtyped(typechecker=beartype)
    def _get_shapes(
        self,
        wind_direction_xy: Float[Array, "2"],
        tree_locations: Float[Array, "num_trees 2"],
        sharpness: float = 100.0,
        tree_radius=0.5,
        tree_height=2.0,
        drone_state: Optional[Float[Array, " 4"]] = None,
    ) -> SDFShape:
        """Return an SDF representation this scene.

        Args:
            wind_direction_xy: the [x, y] direction of the wind
            tree_locations: the locations of the trees
            sharpness: the sharpness of the SDF shapes
            tree_radius: the radius of the trees
            tree_height: the height of the trees
            drone_state: the state of the drone (if None, don't render the drone)
        """
        # Rotate the wind sock to point in the wind direction
        wind_angle = jnp.arctan2(wind_direction_xy[1], wind_direction_xy[0])
        wind_sock_rotation = jnp.array(
            [
                [jnp.cos(wind_angle), -jnp.sin(wind_angle), 0.0],
                [jnp.sin(wind_angle), jnp.cos(wind_angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        wind_sock = Box(
            center=jnp.array([0.6, 0.6, 0.1]),
            extent=jnp.array([0.4, 0.04, 0.01]),
            rotation=wind_sock_rotation,
            c=jnp.array([0.9, 0.5, 0.1]),
        )

        # Add trees to the scene
        trees = []
        for tree_location in tree_locations:
            tree = Cylinder(
                center=jnp.array(
                    [tree_location[0], tree_location[1], tree_height / 2.0]
                ),
                radius=jnp.array(tree_radius),
                height=jnp.array(tree_height),
                c=jnp.array([0.0, 0.8, 0.0]),
                rotation=jnp.eye(3),
            )
            trees.append(tree)

        # Add the drone to the scene
        drone_shapes = []
        if drone_state is not None:
            R = jnp.array(
                [
                    [jnp.cos(drone_state[3]), -jnp.sin(drone_state[3]), 0.0],
                    [jnp.sin(drone_state[3]), jnp.cos(drone_state[3]), 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
            drone_shapes.append(
                Box(
                    center=drone_state[:3],
                    extent=jnp.array([0.1, 0.1, 0.1]),
                    rotation=R,
                    c=jnp.array([52.0, 152, 219]) / 255.0,
                )
            )
            rotor_offsets = jnp.array(
                [
                    [0.15, 0.15, 0.0],
                    [0.15, -0.15, 0.0],
                    [-0.15, 0.15, 0.0],
                    [-0.15, -0.15, 0.0],
                ]
            )
            for rotor_offset in rotor_offsets:
                drone_shapes.append(
                    Cylinder(
                        center=drone_state[:3] + R @ rotor_offset,
                        radius=jnp.array(0.15),
                        height=jnp.array(0.05),
                        c=jnp.array([52.0, 152, 219]) / 255.0,
                        rotation=R,
                    )
                )

        shapes = [self.ground] + self.landing_pad + [wind_sock] + trees + self.walls
        shapes += drone_shapes
        return Scene(shapes=shapes, sharpness=sharpness)

    @jaxtyped(typechecker=beartype)
    def check_for_collision(
        self,
        collider_state: Float[Array, " 3"],
        wind_direction_xy: Float[Array, "2"],
        tree_locations: Float[Array, "n_car 2"],
        sharpness: float = 100.0,
    ) -> Float[Array, ""]:
        """Check for collision with any obstacle in the scene.

        Args:
            collider_state: the [x, y, z] position to check.
            wind_direction_xy: the [x, y] direction of the wind
            tree_locations: the locations of the trees
            sharpness: the sharpness of the SDF shapes

        Returns:
            The minimum distance from the car to any obstacle in the scene.
        """
        # Make the scene (a composite of SDF shapes)
        scene = self._get_shapes(wind_direction_xy, tree_locations, sharpness)

        # Return the minimum distance to any obstacle (negative if there's a collision)
        return scene(collider_state) - 0.05  # radius of drone

    @jaxtyped(typechecker=beartype)
    def render_rgbd(
        self,
        intrinsics: CameraIntrinsics,
        extrinsics: CameraExtrinsics,
        wind_direction_xy: Float[Array, "2"],
        tree_locations: Float[Array, "num_trees 2"],
        shading_light_direction: Optional[Float[Array, "3"]] = None,
        sharpness: float = 50.0,
        max_dist: float = 50.0,
        drone_state: Optional[Float[Array, " 4"]] = None,
    ) -> Tuple[Float[Array, "H W"], Float[Array, "H W 3"]]:
        """Render the color and depth image of this scene from the given camera pose.

        Args:
            intrinsics: the camera intrinsics
            extrinsics: the camera extrinsics
            wind_direction_xy: the [x, y] direction of the wind
            tree_locations: the locations of the trees
            shading_light_direction: the direction of the light source for shading.
                If None, no shading is applied.
            sharpness: the sharpness of the scene
            drone_state: if not None, render the drone at this location

        Returns:
            The depth and color images of the scene
        """
        # Make the scene (a composite of SDF shapes)
        scene = self._get_shapes(
            wind_direction_xy,
            tree_locations,
            sharpness=sharpness,
            drone_state=drone_state,
        )

        # Render the scene
        rays = pinhole_camera_rays(intrinsics, extrinsics)
        hit_pts = jax.vmap(raycast, in_axes=(None, None, 0, None, None))(
            scene, extrinsics.camera_origin, rays, 100, 200.0
        )
        depth_image = render_depth(
            hit_pts, intrinsics, extrinsics, max_dist=max_dist
        ).reshape(intrinsics.resolution)
        color_image = render_color(hit_pts, scene).reshape(*intrinsics.resolution, 3)

        # Add shading if requested
        if shading_light_direction is not None:
            # Normalize
            shading_light_direction = shading_light_direction / (
                1e-3 + jnp.linalg.norm(shading_light_direction)
            )
            shadows = render_shadows(
                hit_pts, scene, shading_light_direction, ambient=0.2
            ).reshape(intrinsics.resolution)
            color_image = color_image * shadows[..., None]

        return depth_image, color_image


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Create a test scene and render it
    scene = DroneLandingScene()
    wind_direction_xy = jnp.array([1.0, -0.5])

    # Add trees from a uniform distribution in a square
    # around the landing pad (excluding the landing pad)
    num_trees = 5
    tree_locations = jax.random.uniform(
        jax.random.PRNGKey(2),
        shape=(num_trees, 2),
        minval=jnp.array([-8.0, -4.0]),
        maxval=jnp.array([-1.0, 4.0]),
    )
    tree_locations = jnp.array(
        [
            [-5.0, 0.0],
            [-3.0, 0.0],
            [-1.0, 0.0],
            [-7.0, 0.0],
            [-9.0, 0.0],
        ]
    )

    # Set the camera parameters
    intrinsics = CameraIntrinsics(
        focal_length=0.1,
        sensor_size=(0.1, 0.1),
        resolution=(512, 512),
    )
    camera_pos = jnp.array([-10.0, 0.0, 1.0])
    # R_straight_down = jnp.array(
    #     [
    #         [1.0, 0.0, 0.0],
    #         [0.0, -1.0, 0.0],
    #         [0.0, 0.0, -1.0],
    #     ]
    # )
    R_forward = jnp.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]
    ) @ jnp.array(
        [
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    extrinsics = CameraExtrinsics(
        camera_origin=camera_pos,
        camera_R_to_world=R_forward,
    )

    light_direction = jnp.array([-1.0, -1.0, 1.0])
    _, color_image = scene.render_rgbd(
        intrinsics,
        extrinsics,
        wind_direction_xy,
        tree_locations,
        shading_light_direction=light_direction,
    )

    _, axs = plt.subplots(1, 1)
    axs.imshow(color_image.transpose(1, 0, 2))
    plt.show()
