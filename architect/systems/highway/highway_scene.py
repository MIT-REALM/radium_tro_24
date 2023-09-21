"""Define a highway scene with a variable number of lanes and cars."""
import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import List, NamedTuple, Optional, Tuple
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
class Car(NamedTuple):
    """Represent a car as a composite of primitive shapes.

    Attributes:
        height: the height of the car
        width: the width of the car
        length: the length of the car
    """

    w_base: Float[Array, ""] = jnp.array(2.8)  # width at base of car
    w_top: Float[Array, ""] = jnp.array(2.3)  # width at top of car

    h_base: Float[Array, ""] = jnp.array(0.4)  # height to undecarriage
    h_chassis: Float[Array, ""] = jnp.array(1.0)  # height of chassis
    h_top: Float[Array, ""] = jnp.array(0.75)  # height of top of car

    l_hood: Float[Array, ""] = jnp.array(0.9)  # length of hood
    l_trunk: Float[Array, ""] = jnp.array(0.4)  # length of trunk
    l_cabin: Float[Array, ""] = jnp.array(3.0)  # length of cabin

    r_wheel: Float[Array, ""] = jnp.array(0.4)  # radius of wheel
    w_wheel: Float[Array, ""] = jnp.array(0.3)  # width of wheel

    @property
    def length(self):
        return self.l_hood + self.l_trunk + self.l_cabin

    @property
    def width(self):
        return self.w_base

    @property
    def height(self):
        return self.h_base + self.h_chassis + self.h_top

    @jaxtyped
    @beartype
    def get_shapes(
        self,
        state: Float[Array, " 3"],
        color: Float[Array, " 3"] = jnp.array([0.972549, 0.4, 0.14117648]),
    ) -> List[SDFShape]:
        """Return a list of primitive shapes representing this car

        Args:
            state: the [x, y, heading] state of the car
            color: the color of the car

        Returns:
            A list of SDF shapes representing the car
        """
        # Unpack state
        x, y, theta = state
        # Most shapes have the same rotation matrix
        rotation = jnp.array(
            [
                [jnp.cos(theta), -jnp.sin(theta), 0],
                [jnp.sin(theta), jnp.cos(theta), 0],
                [0, 0, 1],
            ]
        )

        # Create the primitive shapes that make up the car
        chassis = Box(
            center=jnp.array([x, y, self.h_base + self.h_chassis / 2]),
            extent=jnp.array(
                [
                    self.l_cabin + self.l_hood + self.l_trunk,
                    self.w_base,
                    self.h_chassis,
                ]
            ),
            rotation=rotation,
            c=color,
            rounding=jnp.array(0.1),
        )
        cab = Box(
            center=jnp.array(
                [
                    x + (self.l_trunk - self.l_hood) / 2,
                    y,
                    self.h_base + self.h_chassis + self.h_top / 2,
                ]
            ),
            extent=jnp.array(
                [
                    self.l_cabin,
                    self.w_top,
                    self.h_top,
                ]
            ),
            rotation=rotation,
            c=jnp.array([255, 244, 236]) / 255.0,
            rounding=jnp.array(0.3),
        )

        l_all = self.l_cabin + self.l_trunk + self.l_hood
        # The wheels are rotated 90 degrees around the x axis from the car
        wheel_rotation = rotation @ jnp.array(
            [
                [1, 0, 0],
                [0, 0, -1],
                [0, 1, 0],
            ]
        )
        wheel_color = jnp.array([45, 48, 71]) / 255.0
        wheels = [
            Cylinder(
                jnp.array([x, y, 0.0])
                + rotation
                @ jnp.array([-0.3 * l_all, -0.5 * self.w_base, self.h_base / 2]),
                self.r_wheel,
                self.w_wheel,
                wheel_rotation,
                wheel_color,
            ),
            Cylinder(
                jnp.array([x, y, 0.0])
                + rotation
                @ jnp.array([-0.3 * l_all, 0.5 * self.w_base, self.h_base / 2]),
                self.r_wheel,
                self.w_wheel,
                wheel_rotation,
                wheel_color,
            ),
            Cylinder(
                jnp.array([x, y, 0.0])
                + rotation
                @ jnp.array([0.3 * l_all, -0.5 * self.w_base, self.h_base / 2]),
                self.r_wheel,
                self.w_wheel,
                wheel_rotation,
                wheel_color,
            ),
            Cylinder(
                jnp.array([x, y, 0.0])
                + rotation
                @ jnp.array([0.3 * l_all, +0.4 * self.w_base, self.h_base / 2]),
                self.r_wheel,
                self.w_wheel,
                wheel_rotation,
                wheel_color,
            ),
        ]

        return [chassis, cab] + wheels


@beartype
class HighwayScene:
    """Represent a highway scene with multiple cars and lanes."""

    ground: Halfspace
    walls: List[Box]
    car: Car
    lane_width: float

    @beartype
    def __init__(
        self,
        num_lanes: int,
        lane_width: float = 4.0,
        segment_length: float = 100.0,
    ):
        """Initialize the highway scene.

        Args:
            num_lanes: the number of lanes in the scene
            lane_width: the width of each lane
            segment_length: the length of the highway segment to represent
        """
        # Create the ground plane and walls on each side of the highway
        self.ground = Halfspace(
            point=jnp.array([0.0, 0.0, 0.0]),
            normal=jnp.array([0.0, 0.0, 1.0]),
            c=jnp.array([229, 220, 197]) / 255.0,
        )
        self.walls = [
            Box(
                center=jnp.array([0.0, -lane_width * num_lanes / 2.0 - 0.5, 0.0]),
                extent=jnp.array([segment_length, 1.0, 3.0]),
                rotation=jnp.eye(3),
                c=jnp.array([167, 117, 77]) / 255.0,
                rounding=jnp.array(0.3),
            ),
            Box(
                center=jnp.array([0.0, lane_width * num_lanes / 2 + 0.5, 0.0]),
                extent=jnp.array([segment_length, 1.0, 3.0]),
                rotation=jnp.eye(3),
                c=jnp.array([167, 117, 77]) / 255.0,
                rounding=jnp.array(0.3),
            ),
        ]
        self.car = Car()  # re-used for all cars
        self.lane_width = lane_width

    @jaxtyped
    @beartype
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
            include_ground: whether to include the ground plane
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

    @jaxtyped
    @beartype
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
            include_ground: whether to include the ground plane

        Returns:
            The minimum distance from the car to any obstacle in the scene.
        """
        # Make the scene (a composite of SDF shapes)
        scene = self._get_shapes(
            scene_car_states, sharpness=sharpness, include_ground=include_ground
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
                [-self.car.length / 2, self.car.width / 2, self.car.height / 2],
                [self.car.length / 2, self.car.width / 2, self.car.height / 2],
                [-self.car.length / 2, -self.car.width / 2, self.car.height / 2],
                [self.car.length / 2, -self.car.width / 2, self.car.height / 2],
            ]
        )
        collider_pts_world = collider_pts_car_frame @ car_R_to_world.T
        collider_pts_world = collider_pts_world.at[:, :2].add(collider_state[:2])

        # Return the minimum distance to any obstacle (negative if there's a collision)
        return jax.vmap(scene)(collider_pts_world).min()

    @jaxtyped
    @beartype
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
            scene, extrinsics.camera_origin, rays
        )
        depth_image = render_depth(
            hit_pts, intrinsics, extrinsics, max_dist=max_dist
        ).reshape(intrinsics.resolution)
        return depth_image

    @jaxtyped
    @beartype
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
            scene, extrinsics.camera_origin, rays, 200, 200.0
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

    from architect.systems.components.sensing.vision.util import look_at

    # Create a test highway scene and render it
    highway = HighwayScene(num_lanes=3, lane_width=5.0, segment_length=200.0)
    car_states = jnp.array(
        [
            [-90.0, -3.0, 0.0],
            [-70, 3.0, 0.0],
            # [-5.0, -highway.lane_width, 0.0],
        ]
    )

    # Set the camera parameters
    intrinsics = CameraIntrinsics(
        focal_length=0.1,
        sensor_size=(0.1, 0.1),
        resolution=(512, 512),
    )
    extrinsics = CameraExtrinsics(
        camera_origin=jnp.array([-100.0, -1.5, 2.0]),
        camera_R_to_world=look_at(jnp.array([-100.0, -1.5, 2.0]), jnp.zeros(3)),
    )

    light_direction = jnp.array([-1.0, -1.0, 1.5])
    depth_image, color_image = highway.render_rgbd(
        intrinsics, extrinsics, car_states, shading_light_direction=light_direction
    )

    _, axs = plt.subplots(1, 2)
    axs[0].imshow(depth_image.T)
    axs[1].imshow(color_image.transpose(1, 0, 2))
    plt.show()
