"""Define an intersection scene."""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import List, Optional, Tuple
from jaxtyping import Array, Float, jaxtyped

from radium.systems.components.sensing.vision.render import (
    CameraExtrinsics,
    CameraIntrinsics,
    pinhole_camera_rays,
    raycast,
    render_color,
    render_depth,
    render_shadows,
)
from radium.systems.components.sensing.vision.shapes import (
    Box,
    Halfspace,
    Scene,
    SDFShape,
)
from radium.systems.highway.highway_scene import Car


@beartype
class IntersectionScene:
    """Represent an intersection scene with multiple cars and lanes."""

    ground: Halfspace
    walls: List[Box]
    car: Car

    @beartype
    def __init__(self):
        """Initialize the highway scene."""
        # Create the ground plane and walls around the intersection
        self.ground = Halfspace(
            point=jnp.array([0.0, 0.0, 0.0]),
            normal=jnp.array([0.0, 0.0, 1.0]),
            c=jnp.array([229, 220, 197]) / 255.0,
        )
        self.walls = [
            Box(  # NE vertical
                center=jnp.array([15.0, -7.5 - 0.5, 0.0]),
                extent=jnp.array([15.0, 1.0, 3.0]),
                rotation=jnp.eye(3),
                c=jnp.array([167, 117, 77]) / 255.0,
                rounding=jnp.array(0.3),
            ),
            Box(  # NW vertical
                center=jnp.array([15.0, 7.5 + 0.5, 0.0]),
                extent=jnp.array([15.0, 1.0, 3.0]),
                rotation=jnp.eye(3),
                c=jnp.array([167, 117, 77]) / 255.0,
                rounding=jnp.array(0.3),
            ),
            Box(  # SE vertical
                center=jnp.array([-15.0, -7.5 - 0.5, 0.0]),
                extent=jnp.array([15.0, 1.0, 3.0]),
                rotation=jnp.eye(3),
                c=jnp.array([167, 117, 77]) / 255.0,
                rounding=jnp.array(0.3),
            ),
            Box(  # SW vertical
                center=jnp.array([-15.0, 7.5 + 0.5, 0.0]),
                extent=jnp.array([15.0, 1.0, 3.0]),
                rotation=jnp.eye(3),
                c=jnp.array([167, 117, 77]) / 255.0,
                rounding=jnp.array(0.3),
            ),
            Box(  # NW horizontal
                center=jnp.array([7.5 + 0.5, 15.0, 0.0]),
                extent=jnp.array([1.0, 15.0, 3.0]),
                rotation=jnp.eye(3),
                c=jnp.array([167, 117, 77]) / 255.0,
                rounding=jnp.array(0.3),
            ),
            Box(  # NE horizontal
                center=jnp.array([-7.5 - 0.5, 15.0, 0.0]),
                extent=jnp.array([1.0, 15.0, 3.0]),
                rotation=jnp.eye(3),
                c=jnp.array([167, 117, 77]) / 255.0,
                rounding=jnp.array(0.3),
            ),
            Box(  # SW horizontal
                center=jnp.array([7.5 + 0.5, -15.0, 0.0]),
                extent=jnp.array([1.0, 15.0, 3.0]),
                rotation=jnp.eye(3),
                c=jnp.array([167, 117, 77]) / 255.0,
                rounding=jnp.array(0.3),
            ),
            Box(  # SE horizontal
                center=jnp.array([-7.5 - 0.5, -15.0, 0.0]),
                extent=jnp.array([1.0, 15.0, 3.0]),
                rotation=jnp.eye(3),
                c=jnp.array([167, 117, 77]) / 255.0,
                rounding=jnp.array(0.3),
            ),
        ]
        self.car = Car()  # re-used for all cars

    @jaxtyped(typechecker=beartype)
    def _get_shapes(
        self,
        car_states: Float[Array, "n_car 3"],
        sharpness: float = 100.0,
        car_colors: Optional[Float[Array, "n_car 3"]] = None,
        include_ground: bool = True,
    ) -> SDFShape:
        """Return an SDF representation this scene.

        Args:
            car_states: the [x, y, heading] state of each car in the scene
            sharpness: the sharpness of the SDF shapes
            car_colors: the color of each car. If None, the default colors are used.
            include_ground: whether to include the ground plane in the scene
        """
        if car_colors is None:
            car_shapes = [self.car.get_shapes(state) for state in car_states]
        else:
            car_shapes = [
                self.car.get_shapes(state, color)
                for state, color in zip(car_states, car_colors)
            ]

        shapes = []
        shapes += [shape for sublist in car_shapes for shape in sublist]
        shapes += self.walls
        if include_ground:
            shapes.append(self.ground)

        return Scene(shapes=shapes, sharpness=sharpness)

    @jaxtyped(typechecker=beartype)
    def check_for_collision(
        self,
        collider_state: Float[Array, " 3"],
        scene_car_states: Float[Array, "n_car 3"],
        sharpness: float = 100.0,
        include_ground: bool = True,
    ) -> Float[Array, ""]:
        """Check for collision with any obstacle in the scene.

        Args:
            collider_state: the [x, y, heading] state of the car to check (note:
                this should not be one of the cars included in the scene, otherwise
                there will always be a collision).
            scene_car_states: the [x, y, heading] state of each car in the scene
            sharpness: the sharpness of the SDF shapes
            include_ground: whether to include the ground plane in the scene

        Returns:
            The minimum distance from the car to any obstacle in the scene.
        """
        # Make the scene (a composite of SDF shapes)
        scene = self._get_shapes(
            scene_car_states, sharpness, include_ground=include_ground
        )

        # Check for collision at four points on the car
        car_R_to_world = jnp.array(
            [
                [jnp.cos(collider_state[2]), -jnp.sin(collider_state[2]), 0],
                [jnp.sin(collider_state[2]), jnp.cos(collider_state[2]), 0],
                [0, 0, 1],
            ]
        )
        collider_pts_car_frame = jnp.array(
            [
                [-self.car.length / 2, self.car.width / 2, self.car.height * 0.75],
                [self.car.length / 2, self.car.width / 2, self.car.height * 0.75],
                [-self.car.length / 2, -self.car.width / 2, self.car.height * 0.75],
                [self.car.length / 2, -self.car.width / 2, self.car.height * 0.75],
            ]
        )
        collider_pts_world = collider_pts_car_frame @ car_R_to_world.T
        collider_pts_world = collider_pts_world.at[:, :2].add(collider_state[:2])

        # Return the minimum distance to any obstacle (negative if there's a collision)
        return jax.vmap(scene)(collider_pts_world).min()

    @jaxtyped(typechecker=beartype)
    def render_depth(
        self,
        intrinsics: CameraIntrinsics,
        extrinsics: CameraExtrinsics,
        car_states: Float[Array, "n_car 3"],
        max_dist: float = 50.0,
        sharpness: float = 100.0,
    ) -> Float[Array, "res_x res_y"]:
        """Render the depth image of this scene from the given camera pose.

        Args:
            intrinsics: the camera intrinsics
            extrinsics: the camera extrinsics
            car_states: the [x, y, heading] state of each car
            max_dist: the maximum distance to render
            sharpness: the sharpness of the scene

        Returns:
            The depth image of the scene
        """
        # Make the scene (a composite of SDF shapes)
        scene = self._get_shapes(car_states, sharpness=sharpness)

        # Render the scene
        rays = pinhole_camera_rays(intrinsics, extrinsics)
        hit_pts = jax.vmap(raycast, in_axes=(None, None, 0))(
            scene, extrinsics.camera_origin, rays, 300, 200.0
        )
        depth_image = render_depth(
            hit_pts, intrinsics, extrinsics, max_dist=max_dist
        ).reshape(intrinsics.resolution)
        return depth_image

    @jaxtyped(typechecker=beartype)
    def render_rgbd(
        self,
        intrinsics: CameraIntrinsics,
        extrinsics: CameraExtrinsics,
        car_states: Float[Array, "n_car 3"],
        shading_light_direction: Optional[Float[Array, "3"]] = None,
        car_colors: Optional[Float[Array, "n_car 3"]] = None,
        max_dist: float = 50.0,
        sharpness: float = 50.0,
    ) -> Tuple[Float[Array, "res_x res_y"], Float[Array, "res_x res_y 3"]]:
        """Render the color + depth images of this scene from the given camera pose.

        Args:
            intrinsics: the camera intrinsics
            extrinsics: the camera extrinsics
            car_states: the [x, y, heading] state of each car
            shading_light_direction: the direction of the light source for shading. If
                None, no shading is applied.
            car_colors: the color of each car. If None, the default colors are used.
            max_dist: the maximum distance to render
            sharpness: the sharpness of the scene

        Returns:
            The depth and color images of the scene
        """
        # Make the scene (a composite of SDF shapes)
        scene = self._get_shapes(car_states, sharpness=sharpness, car_colors=car_colors)

        # Render the scene
        rays = pinhole_camera_rays(intrinsics, extrinsics)
        hit_pts = jax.vmap(raycast, in_axes=(None, None, 0, None, None))(
            scene, extrinsics.camera_origin, rays, 300, 200.0
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

    from radium.systems.components.sensing.vision.util import look_at

    # Create a test highway scene and render it
    highway = IntersectionScene()
    car_states = jnp.array(
        [[-3.75, -7.5, -jnp.pi / 2], [-3.75, 1, -jnp.pi / 2], [3.75, 10, jnp.pi / 2]]
    )

    # Set the camera parameters
    intrinsics = CameraIntrinsics(
        focal_length=0.1,
        sensor_size=(0.1, 0.1),
        resolution=(512, 512),
    )
    extrinsics = CameraExtrinsics(
        camera_origin=jnp.array([-20, -2, 2.5]),
        camera_R_to_world=look_at(jnp.array([-20, -2, 2.5]), jnp.zeros(3)),
    )

    light_direction = jnp.array([-1.0, -1.0, 1.5])
    depth_image, color_image = highway.render_rgbd(
        intrinsics,
        extrinsics,
        car_states,
        shading_light_direction=light_direction,
        car_colors=jax.random.uniform(jax.random.PRNGKey(0), (3, 3)),
    )

    _, axs = plt.subplots(1, 2)
    axs[0].imshow(depth_image.T)
    axs[1].imshow(color_image.transpose(1, 0, 2))
    plt.show()
