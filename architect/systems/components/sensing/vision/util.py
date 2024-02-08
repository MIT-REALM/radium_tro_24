"""Vision utility functions."""

import jax.numpy as jnp
from beartype import beartype
from jaxtyping import Array, Float, jaxtyped


@jaxtyped(typechecker=beartype)
def look_at(
    camera_origin: Float[Array, " 3"],
    target: Float[Array, " 3"],
    up: Float[Array, " 3"] = jnp.array([0.0, 0.0, 1.0]),
) -> Float[Array, "3 3"]:
    """Compute the rotation matrix that rotates the camera to look at the target.

    Args:
        camera_origin: The origin of the camera, in world coordinates.
        target: The target point, in world coordinates.
        up: The direction that should be "up" from the camera's point of view.

    Returns:
        The rotation matrix that rotates the camera to look at the target.
    """
    # Construct the rotation matrix from the x, y, and z axes of the camera.
    # The z axis points from the camera origin to the target (forward), the x axis
    # points to the right of the camera, and the y axis points down.
    forward = target - camera_origin
    right = jnp.cross(forward, up)
    down = jnp.cross(forward, right)
    R = jnp.vstack([right, down, forward])
    R = R / jnp.linalg.norm(R, axis=-1, keepdims=True)
    return R.T
