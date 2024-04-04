import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from radium.systems.hide_and_seek.hide_and_seek_types import Trajectory2D


def test_Trajectory2d(plot=False):
    # Define a test spline
    p = jnp.array(
        [
            [0.0, 0.0],
            [-0.1, 1.0],
            [2.0, 1.0],
            [3.0, 0.0],
            [4.0, 1.0],
        ],
        dtype=float,
    )
    traj = Trajectory2D(p)

    # Evaluate it at a single point
    t = jnp.array(0.5)
    pt = traj(t)
    assert pt.shape == (2,)

    # Evaluate it at a bunch of points
    N = 100
    t = jnp.linspace(0, 1, N)
    pts = jax.vmap(traj)(t)
    assert pts.shape == (N, 2)

    # Plot if requested
    if plot:
        plt.scatter(p[:, 0], p[:, 1], color="r", marker="x")
        plt.plot(pts[:, 0], pts[:, 1])
        plt.show()


if __name__ == "__main__":
    test_Trajectory2d(plot=True)
