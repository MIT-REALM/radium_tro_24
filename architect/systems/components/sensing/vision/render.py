"""Functions for rendering a scene."""

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import Callable, NamedTuple, Tuple
from jaxtyping import Array, Float, jaxtyped


@beartype
class CameraIntrinsics(NamedTuple):
    """Represent the intrinsic parameters of a camera.

    Attributes:
        focal_length: The focal length of the camera, in meters.
        sensor_size: The (x, y) size of the camera sensor, in meters.
        resolution: The (x, y) resolution of the camera, in pixels.
    """

    focal_length: float
    sensor_size: Tuple[float, float]
    resolution: Tuple[int, int]


@beartype
class CameraExtrinsics(NamedTuple):
    """Represent the extrinsic parameters of a camera.

    Attributes:
        camera_R_to_world: The rotation matrix from camera to world coordinates.
        camera_origin: The location of the camera origin in world coordinates.
    """

    camera_R_to_world: Float[Array, "3 3"]
    camera_origin: Float[Array, " 3"]


@jaxtyped(typechecker=beartype)
def pinhole_camera_rays(
    intrinsics: CameraIntrinsics,
    extrinsics: CameraExtrinsics,
) -> Float[Array, "n_rays 3"]:
    """Return the rays originating from a pinhole camera, in world coordinates.

    Args:
        intrinsics: The intrinsic parameters of the camera.
        extrinsics: The extrinsic parameters of the camera.

    Returns:
        The rays originating from the camera, in world coordinates.
    """
    # Unpack camera parameters
    camera_R_to_world = extrinsics.camera_R_to_world
    sensor_size = intrinsics.sensor_size
    resolution = intrinsics.resolution
    focal_length = intrinsics.focal_length

    # Get the points where the rays pass through the image plane of the camera,
    # in camera coordinates.
    sensor_x, sensor_y = sensor_size
    res_x, res_y = resolution
    x_img, y_img = jnp.mgrid[
        -sensor_x : sensor_x : res_x * 1j,  # complex slice -> linspace-like behavior
        -sensor_y : sensor_y : res_y * 1j,
    ].reshape(2, -1)

    # The rays are the unit vectors from the camera origin to these points
    rays = jnp.c_[x_img, y_img, focal_length * jnp.ones_like(x_img)]
    rays = rays / jnp.linalg.norm(rays, axis=1, keepdims=True)

    # Convert to world coordinates
    return rays @ camera_R_to_world.T


@jaxtyped(typechecker=beartype)
def raycast(
    sdf: Callable[[Float[Array, " 3"]], Float[Array, ""]],
    origin: Float[Array, " 3"],
    ray: Float[Array, " 3"],
    max_steps: int = 40,
    max_dist: float = 200.0,
) -> Float[Array, " 3"]:
    """Find the points at which the given ray intersects the given SDF.

    Args:
        sdf: The signed distance function of the scene.
        origin: The origin of the ray, in world coordinates.
        ray: The ray to cast, in world coordinates.
        max_steps: The maximum number of steps to take along the ray.
        max_dist: The maximum distance to travel along the ray.

    Returns:
        The point at which the ray intersects the SDF, in world coordinates.
    """

    # At each iteration, we can step the ray a distance equal to the current value
    # of the SDF without intersecting the surface. The first argument passed by
    # fori_loop is the iteration number, which we don't need.
    def solve_for_collision_distance(dist_fn, initial_guess):
        """Solve for the distance along the ray where the SDF is 0."""

        # Define a function for executing one step of ray marching
        def step_ray(_, dist_along_ray: Float[Array, ""]) -> Float[Array, ""]:
            h = dist_fn(dist_along_ray)
            new_dist_along_ray = dist_along_ray + h
            return jnp.clip(new_dist_along_ray, 0.0, max_dist)

        # Raymarch
        return jax.lax.fori_loop(0, max_steps, step_ray, initial_guess)

    # We need to treat this as a root-finding problem so we can use implicit
    # differentiation rather than unrolling the whole shebang.
    distance_along_ray = jax.lax.custom_root(
        lambda d: sdf(origin + d * ray),
        initial_guess=jnp.array(1e-2),
        solve=solve_for_collision_distance,
        tangent_solve=lambda g, y: y / g(1.0),
    )

    # # Empirically, implicit autodiff doesn't work as well as just differentiating
    # # through the for loop in solve
    # dist_fn = lambda d: sdf(origin + d * ray)
    # initial_guess = jnp.array(1e-2)

    # # Define a function for executing one step of ray marching
    # @jax.checkpoint
    # def step_ray(_, dist_along_ray: Float[Array, ""]) -> Float[Array, ""]:
    #     h = dist_fn(dist_along_ray)
    #     new_dist_along_ray = dist_along_ray + h
    #     return jnp.clip(new_dist_along_ray, 0.0, max_dist)

    # # Raymarch
    # distance_along_ray = jax.lax.fori_loop(0, max_steps, step_ray, initial_guess)

    return origin + distance_along_ray * ray


@jaxtyped(typechecker=beartype)
def raycast_shadow(
    sdf: Callable[[Float[Array, " 3"]], Float[Array, ""]],
    origin: Float[Array, " 3"],
    ray: Float[Array, " 3"],
    shadow_hardness: float = 1.0,
    max_steps: int = 40,
    max_dist: float = 200.0,
    ambient_light: float = 0.1,
) -> Float[Array, ""]:
    """Accumulate the shading along the given ray.

    Args:
        sdf: The signed distance function of the scene.
        origin: The origin of the ray, in world coordinates.
        ray: The ray to cast, in world coordinates. Should be the direction towards
            the light source.
        shadow_hardness: higher values will yield sharper edges to the shadows.
        max_steps: The maximum number of steps to take along the ray.
        max_dist: The maximum distance to travel along the ray.
        ambient_light: The amount of light that is always present, regardless of
            occlusion (between 0 and 1).

    Returns:
        The total shading of the origin (0 = fully shaded, 1 = fully lit)
    """

    # At each iteration, we can step the ray a distance equal to the current value
    # of the SDF without intersecting the surface. The first argument passed by
    # fori_loop is the iteration number, which we don't need.
    @jax.checkpoint
    def step_shadow_ray(
        _, carry: Tuple[Float[Array, ""], Float[Array, ""]]
    ) -> Tuple[Float[Array, ""], Float[Array, ""]]:
        # Unpack
        distance_along_ray, shading = carry

        # Step the ray according to the SDF at the current point, and make the shadow
        # darker as we get closer to the surface of the scene.
        current_pt = origin + distance_along_ray * ray
        h = sdf(current_pt)
        shading = jnp.clip(
            shadow_hardness * h / distance_along_ray, ambient_light, shading
        )

        # Stop if we've reached the maximum distance
        return jnp.clip(distance_along_ray + h, -max_dist, max_dist), shading

    return jax.lax.fori_loop(0, max_steps, step_shadow_ray, (1e-2, 1.0))[1]


def render_depth(
    hit_pts: Float[Array, "n_rays 3"],
    intrinsics: CameraIntrinsics,
    extrinsics: CameraExtrinsics,
    max_dist: float = 10.0,
) -> Float[Array, " n_rays"]:
    """Render the depth of the given hit points.

    Args:
        hit_pts: The hit points of the rays, in world coordinates.
        intrinsics: The intrinsic parameters of the camera.
        extrinsics: The extrinsic parameters of the camera.
        max_dist: The maximum distance to render, in meters. Set the depth for all rays
            without an intersection within this distance to 0.

    Returns:
        The depth of the hit points, in meters
    """
    # Unpack camera parameters
    camera_R_to_world = extrinsics.camera_R_to_world
    camera_origin = extrinsics.camera_origin

    # Get the depth image by getting the distance along the camera z axis to each hit
    # point
    depth_values = jax.vmap(jnp.dot, in_axes=(0, None))(
        hit_pts - camera_origin, camera_R_to_world @ jnp.array([0.0, 0.0, 1.0])
    )

    # Clip the depth image at a maximum distance
    depth_values = jnp.clip(depth_values, a_min=0.0, a_max=max_dist)
    depth_values = jnp.where(depth_values == max_dist, 0.0, depth_values)

    return depth_values


def render_color(
    hit_pts: Float[Array, "n_rays 3"],
    sdf: Callable[[Float[Array, " 3"]], Float[Array, ""]],
    fog_color: Float[Array, " 3"] = jnp.array([0.5, 0.6, 0.7]),
    fog_strength: float = 0.005,
) -> Float[Array, "n_rays 3"]:
    """Render the color of the given hit points.

    Args:
        hit_pts: The hit points of the rays, in world coordinates.
        sdf: The signed distance function of the scene.
        fog_color: The color to use for fog.
        fog_strength: The strength of the fog, 0 = no fog.

    Returns:
        The RGB color of the hit points, normalized to [0, 1].
    """
    # Get the depth image by getting the scene color at each hit point
    color_values = jax.vmap(sdf.color)(hit_pts)

    # Add fog
    distance = jnp.sqrt(jnp.sum(hit_pts**2, axis=-1) + 1e-3)
    fog_weight = 1 - jnp.exp(-fog_strength * distance)
    color_values = (
        fog_weight[:, None] * fog_color + (1 - fog_weight[:, None]) * color_values
    )

    return color_values


def render_shadows(
    hit_pts: Float[Array, "n_rays 3"],
    sdf: Callable[[Float[Array, " 3"]], Float[Array, ""]],
    light_dir: Float[Array, " 3"],
    shadow_hardness: float = 1.0,
    max_steps: int = 40,
    ambient: float = 0.01,
) -> Float[Array, "n_rays 3"]:
    """Render the shading of the given hit points.

    Args:
        hit_pts: The hit points of the rays, in world coordinates.
        sdf: The signed distance function of the scene.
        light_dir: The direction of the light source, in world coordinates.
        shadow_hardness: higher values will yield sharper edges to the shadows.
        max_steps: The maximum number of steps to take along the ray.
        ambient: amount of ambient lighting (between 0=none and 1=full)

    Returns:
        The shading the hit points (0 = fully shaded, 1 = fully lit).
    """
    # Get the depth image by getting the scene color at each hit point
    shading = jax.vmap(raycast_shadow, in_axes=(None, 0, None, None, None, None))(
        sdf,
        hit_pts,
        light_dir,
        shadow_hardness,
        max_steps,
        ambient,
    )

    return shading
