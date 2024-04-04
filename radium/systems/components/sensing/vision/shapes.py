"""Represent various shapes using Signed Distance Functions (SDFs).

Note that these are intended for the purpose of rendering a view of a 3D scene, not
for contact simulation or collision detection. As a result, these functions may not
return a fully accurate SDF, but they should be sufficient for rendering.
"""

from abc import ABC, abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.typing import List, Optional
from jaxtyping import Array, Float, jaxtyped

from radium.utils import softmin


@jax.jit
def softnorm(x):
    """Compute the 2-norm, but if x is too small replace it with the squared 2-norm
    to make sure it's differentiable. This function is continuous and has a derivative
    that is defined everywhere, but its derivative is discontinuous.
    """
    eps = 1e-3
    # scaled_square = lambda x: (eps * (x / eps) ** 2).sum()
    # return jax.lax.cond(jnp.linalg.norm(x) >= eps, jnp.linalg.norm, scaled_square, x)
    return jnp.sqrt(jnp.sum(x**2) + eps)


class SDFShape(ABC, eqx.Module):
    """Abstract base class for shapes defined via signed distance functions."""

    @abstractmethod
    def __call__(self, x: Float[Array, " 3"]) -> Float[Array, ""]:
        """Compute the SDF at a given point.

        Args:
            x: a point in world coordinates

        Returns:
            signed distance from the point to the shape
        """

    def color(self, x: Float[Array, " 3"]) -> Float[Array, " 3"]:
        """Compute the color at a given point.

        Default behavior is a checkerboard pattern.

        Args:
            x: a point in world coordinates

        Returns:
            RGB color at the point, scaled to [0, 1]
        """
        red = jax.nn.sigmoid(10 * jnp.sin(10 * x[0]))
        green = jax.nn.sigmoid(10 * jnp.sin(10 * x[1]))
        blue = jax.nn.sigmoid(10 * jnp.sin(10 * x[2]))
        return jnp.array([red, green, blue])


class Scene(SDFShape):
    """Represent a scene using a SDF.

    Attributes:
        shapes: the shapes in the scene
        sharpness: the sharpness of the SDF
    """

    shapes: List[SDFShape]
    sharpness: float = 1.0

    def __call__(self, x: Float[Array, " 3"]) -> Float[Array, ""]:
        """Compute the SDF at a given point.

        Args:
            x: a point in world coordinates

        Returns:
            signed distance from the point to the scene (positive outside)
        """
        distances = jnp.array([shape(x) for shape in self.shapes])
        return softmin(distances, self.sharpness)

    def color(self, x: Float[Array, " 3"]) -> Float[Array, " 3"]:
        """Compute the color at a given point.

        Args:
            x: a point in world coordinates

        Returns:
            RGB color at the point, scaled to [0, 1]
        """
        colors = jnp.array([shape.color(x) for shape in self.shapes])
        distances = jnp.array([shape(x) for shape in self.shapes])
        return (jax.nn.softplus(-self.sharpness * distances)[:, None] * colors).sum(
            axis=0
        )


class Subtraction(SDFShape):
    """Represent the subtraction of one shape from another using SDFs.

    Attributes:
        shape1: the shape to add (sets the color)
        shape2: the shape to subtract
        sharpness: the sharpness of the SDF
    """

    shape1: SDFShape
    shape2: SDFShape
    sharpness: float = 1.0

    def __call__(self, x: Float[Array, " 3"]) -> Float[Array, ""]:
        """Compute the SDF at a given point.

        Args:
            x: a point in world coordinates

        Returns:
            signed distance from the point to the scene (positive outside)
        """
        distance1 = self.shape1(x)
        distance2 = self.shape2(x)
        return -softmin(jnp.array([-distance1, distance2]), self.sharpness)

    def color(self, x: Float[Array, " 3"]) -> Float[Array, " 3"]:
        """Compute the color at a given point.

        Color is determined by the first shape.

        Args:
            x: a point in world coordinates

        Returns:
            RGB color at the point, scaled to [0, 1]
        """
        return self.shape1.color(x)


@jaxtyped(typechecker=beartype)
class Sphere(SDFShape):
    """Represent a sphere using a SDF.

    Attributes:
        center: the center of the sphere in world coordinates
        radius: Radius of the sphere.
    """

    center: Float[Array, " 3"]
    radius: Float[Array, ""]

    def __call__(self, x: Float[Array, " 3"]) -> Float[Array, ""]:
        """Compute the SDF at a given point.

        Args:
            x: a point in world coordinates

        Returns:
            signed distance from the point to the sphere (positive outside)
        """
        return softnorm(x - self.center) - self.radius


@jaxtyped(typechecker=beartype)
class Halfspace(SDFShape):
    """Represent a halfspace using a SDF.

    Attributes:
        normal: the normal vector pointing to the exterior of the halfspace
        point: a point on the plane bounding the halfspace
        c: the color of the halfspace (or None for a checkerboard pattern)
    """

    normal: Float[Array, " 3"]
    point: Float[Array, " 3"]
    c: Optional[Float[Array, " 3"]] = None

    def __call__(self, x: Float[Array, " 3"]) -> Float[Array, ""]:
        """Compute the SDF at a given point.

        Args:
            x: a point in world coordinates

        Returns:
            signed distance from the point to the halfspace
        """
        return jnp.dot(x - self.point, self.normal)

    def color(self, x: Float[Array, " 3"]) -> Float[Array, " 3"]:
        """Compute the color at a given point.

        Args:
            x: a point in world coordinates

        Returns:
            RGB color at the point, scaled to [0, 1]
        """
        if self.c is not None:
            return self.c

        return super().color(x)


@jaxtyped(typechecker=beartype)
class Box(SDFShape):
    """Represent a box using a SDF.

    Attributes:
        center: the center of the box in world coordinates
        extent: the extent of the box in each dimension
        R_to_world: the 3D rotation matrix from the box frame to the world frame
        c: the color of the box (or None for a checkerboard pattern)
    """

    center: Float[Array, " 3"]
    extent: Float[Array, " 3"]
    rotation: Float[Array, "3 3"]
    c: Optional[Float[Array, " 3"]] = None
    rounding: Float[Array, ""] = jnp.array(0.1)

    def __call__(self, x: Float[Array, " 3"]) -> Float[Array, ""]:
        """Compute the SDF at a given point.

        Args:
            x: a point in world coordinates

        Returns:
            signed distance from the point to the box
        """
        # Get the offset from the box center to the point, and rotate it into the box
        # frame
        offset = self.rotation.T @ (x - self.center)

        # Compute the distance to the box in the box frame (which is axis-aligned)
        distance_to_box = jnp.abs(offset) - self.extent / 2.0

        # Round the corner a bit
        sdf = (
            softnorm(jnp.maximum(distance_to_box, 0.0))
            + jnp.minimum(jnp.max(distance_to_box), 0.0)
            - self.rounding
        )
        return sdf

    def color(self, x: Float[Array, " 3"]) -> Float[Array, " 3"]:
        """Compute the color at a given point.

        Args:
            x: a point in world coordinates

        Returns:
            RGB color at the point, scaled to [0, 1]
        """
        if self.c is not None:
            return self.c

        return super().color(x)


@jaxtyped(typechecker=beartype)
class Cylinder(SDFShape):
    """Represent a capped cylinder using a SDF.

    Attributes:
        center: the center of the cylinder in world coordinates
        radius: the radius of the cylinder
        height: the height of the cylinder
        R_to_world: the 3D rotation matrix from the box frame to the world frame
        c: the color of the cylinder (if None, use a checkerboard pattern)
    """

    center: Float[Array, " 3"]
    radius: Float[Array, ""]
    height: Float[Array, ""]
    rotation: Float[Array, "3 3"]
    c: Optional[Float[Array, " 3"]] = None

    def __call__(self, x: Float[Array, " 3"]) -> Float[Array, ""]:
        """Compute the SDF at a given point.

        Args:
            x: a point in world coordinates

        Returns:
            signed distance from the point to the cylinder
        """
        # Get the offset from the cylinder center to the point, and rotate it into the
        # cylinder frame
        offset = self.rotation.T @ (x - self.center)

        # Compute the distance to the cylinder in the cylinder frame (which is
        # axis-aligned)
        distance_to_axis = softnorm(offset[:2]) - self.radius
        distance_to_caps = jnp.abs(offset[2]) - self.height / 2.0

        sdf = jnp.maximum(distance_to_axis, distance_to_caps)
        return sdf

    def color(self, x: Float[Array, " 3"]) -> Float[Array, " 3"]:
        """Compute the color at a given point.

        Args:
            x: a point in world coordinates

        Returns:
            RGB color at the point, scaled to [0, 1]
        """
        if self.c is not None:
            return self.c

        return super().color(x)
