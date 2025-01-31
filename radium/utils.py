"""Common utility functions."""

import jax
import jax.numpy as jnp
from beartype import beartype
from jax.nn import logsumexp, softplus
from jaxtyping import Array, Float, jaxtyped


@jaxtyped(typechecker=beartype)
def softmax(x: Float[Array, "..."], sharpness: float = 0.05):
    """Return the soft maximum of the given vector"""
    return 1 / sharpness * logsumexp(sharpness * x)


@jaxtyped(typechecker=beartype)
def softmin(x: Float[Array, "..."], sharpness: float = 0.05):
    """Return the soft minimum of the given vector"""
    return -softmax(-x, sharpness)


@jaxtyped(typechecker=beartype)
def softclip(x: Float[Array, "..."], a: float, sharpness: float = 10.0):
    """Return the vector clipped elementwise to the range [-a, a]"""
    return (
        x
        - softplus(sharpness * (x - a)) / sharpness
        + softplus(sharpness * (-a - x)) / sharpness
    )


@jax.custom_vjp
def clip_gradient(lo, hi, x):
    return x  # identity function


def clip_gradient_fwd(lo, hi, x):
    return x, (lo, hi)  # save bounds as residuals


def clip_gradient_bwd(res, g):
    lo, hi = res
    clipped = jax.tree_util.tree_map(lambda x: jnp.clip(x, lo, hi), g)
    return (
        None,
        None,
        clipped,
    )  # use None to indicate zero cotangents for lo and hi


clip_gradient.defvjp(clip_gradient_fwd, clip_gradient_bwd)
