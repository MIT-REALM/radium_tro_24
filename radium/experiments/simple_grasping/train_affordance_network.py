"""Train the grasp affordance network for the simple grasping environment."""

import os

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import numpy as np
import optax
from beartype import beartype
from beartype.typing import NamedTuple
from jaxtyping import Array, Float, jaxtyped
from matplotlib import colormaps as cm
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from radium.systems.simple_grasping.env import make_grasping_scene, render_rgbd
from radium.systems.simple_grasping.policy import AffordancePredictor
from radium.types import PRNGKeyArray


# Define a type for training data
class TrainingData(NamedTuple):
    """Represent training data for the grasp affordance network."""

    depth_image: Float[Array, "H W"]
    affordance_mask: Float[Array, "H W"]
    gt_grasp_pose: Float[Array, "2 3"]


def generate_example(key: PRNGKeyArray, object: str = "mug") -> TrainingData:
    """Generate a single training example"""
    # Sample a mug position and location
    loc_key, rot_key = jrandom.split(key)
    position = 0.2 * jrandom.normal(loc_key, shape=(2,))
    rotation = jnp.pi / 2 * jrandom.normal(rot_key, shape=()) + jnp.pi / 2

    # Create a scene with an object
    camera_pos = jnp.array([-1.0, 0.0, 1.0])
    scene, gt_grasp = make_grasping_scene(
        mug_location=position,
        mug_rotation=rotation,
        sharpness=50.0,
        object=object,
    )
    depth_image, color_image = render_rgbd(scene, camera_pos)

    # Reduce color image to single channel
    afforance_mask = jnp.mean(color_image, axis=-1)

    return TrainingData(
        depth_image=depth_image,
        affordance_mask=afforance_mask,
        gt_grasp_pose=gt_grasp,
    )


@jaxtyped(typechecker=beartype)
def shuffle_examples(examples: TrainingData, key: PRNGKeyArray) -> TrainingData:
    """Shuffle the examples.

    Args:
        examples: The examples to shuffle.
        key: The PRNG key to use for shuffling.

    Returns:
        The shuffled examples.
    """
    # Shuffle the examples
    examples_len = examples.depth_image.shape[0]
    permutation = jrandom.permutation(key, examples_len)
    examples = jtu.tree_map(lambda x: x[permutation], examples)

    return examples


def train_affordance_network(
    learning_rate: float = 1e-3,
    seed: int = 0,
    n_examples: int = 32 * 50,
    epochs: int = 1000,
    minibatch_size: int = 32,
    logdir: str = "./tmp/affordance+grasp/all",
):
    """
    Train the affordance network.

    Args: various hyperparameters.
    """
    # Set up logging
    writer = SummaryWriter(logdir)

    # Set up the network
    key = jrandom.PRNGKey(seed)
    key, subkey = jrandom.split(key)
    predictor = AffordancePredictor(subkey)

    # Set up the optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(predictor, eqx.is_array))

    # Compile a loss function and optimization update
    @eqx.filter_value_and_grad
    def loss_fn(
        predictor: AffordancePredictor, examples: TrainingData
    ) -> Float[Array, ""]:
        # Compute the predicted affordances for each observation
        predicted_affordances, predicted_grasps = jax.vmap(predictor)(
            examples.depth_image
        )

        # Minimize L2 loss between predicted and actual affordances
        loss = jnp.mean(jnp.square(predicted_affordances - examples.affordance_mask))

        # and L2 loss between predicted and actual grasps
        loss += jnp.mean(jnp.square(predicted_grasps - examples.gt_grasp_pose))

        return loss

    @eqx.filter_jit
    def step_fn(
        opt_state: optax.OptState,
        predictor: AffordancePredictor,
        examples: TrainingData,
    ):
        loss, grad = loss_fn(predictor, examples)
        updates, opt_state = optimizer.update(grad, opt_state)
        predictor = eqx.apply_updates(predictor, updates)
        return (
            loss,
            predictor,
            opt_state,
        )

    # Generate a batch of labeled images
    key, subkey = jrandom.split(key)
    subkeys = jrandom.split(subkey, n_examples)
    mug_examples = jax.vmap(generate_example, in_axes=(0, None))(subkeys, "mug")
    bowl_examples = jax.vmap(generate_example, in_axes=(0, None))(subkeys, "bowl")
    can_examples = jax.vmap(generate_example, in_axes=(0, None))(subkeys, "box")
    examples = jax.tree_util.tree_map(
        lambda *x: jnp.concatenate(x, axis=0),
        mug_examples,
        bowl_examples,
        can_examples,
    )

    # Shuffle the trajectory into minibatches
    key, subkey = jrandom.split(key)
    examples = shuffle_examples(examples, subkey)

    # Training loop
    pbar = tqdm(range(epochs))
    for epoch in pbar:
        # Save regularly
        if epoch % 50 == 0 or epoch == epochs - 1:
            # Save predictor
            eqx.tree_serialise_leaves(
                os.path.join(logdir, f"predictor_{epoch}.eqx"), predictor
            )

            # Predict for the first 10 examples
            predicted_affordances, predicted_grasps = jax.vmap(predictor)(
                examples.depth_image[:10]
            )
            cm_jet = cm.get_cmap("jet")
            displays = [
                jnp.concatenate(
                    [
                        cm_jet(examples.depth_image[i].T),
                        cm_jet(examples.affordance_mask[i].T),
                        cm_jet(predicted_affordances[i].T),
                    ],
                    axis=1,
                )
                for i in range(10)
            ]
            display = np.array(jnp.concatenate(displays, axis=0))[:, :, :3]
            writer.add_image("predicted_affordances", display, epoch, dataformats="HWC")

            # Plot the predicted and ground truth grasps in 3D
            for i in range(5):
                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
                ax.set_zlim(-0.1, 1)
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                ax.set_zlabel("z")
                ax.scatter(
                    predicted_grasps[i, :, 0],
                    predicted_grasps[i, :, 1],
                    predicted_grasps[i, :, 2],
                    c="r",
                    marker="o",
                )
                ax.scatter(
                    examples.gt_grasp_pose[i, :, 0],
                    examples.gt_grasp_pose[i, :, 1],
                    examples.gt_grasp_pose[i, :, 2],
                    c="b",
                    marker="o",
                )
                ax.view_init(elev=45, azim=180)
                writer.add_figure(f"predicted_grasps/{i}", fig, epoch)

        # Compute the loss and gradient
        epoch_loss = 0.0
        batches = 0
        for batch_start in range(0, n_examples, minibatch_size):
            key, subkey = jrandom.split(key)
            (
                loss,
                predictor,
                opt_state,
            ) = step_fn(
                opt_state,
                predictor,
                jtu.tree_map(
                    lambda x: x[batch_start : batch_start + minibatch_size],
                    examples,
                ),
            )
            batches += 1
            epoch_loss += loss.item()

        # Average
        epoch_loss /= batches

        # Log the loss
        pbar.set_description(f"loss: {epoch_loss:.3e}")
        writer.add_scalar("loss", epoch_loss, epoch)


if __name__ == "__main__":
    train_affordance_network()
