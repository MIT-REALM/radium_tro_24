"""Train an agent for the drone environment using behavior cloning."""

import os

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
import matplotlib.image
import optax
from beartype import beartype
from beartype.typing import NamedTuple, Tuple
from jax_tqdm import scan_tqdm
from jaxtyping import Array, Float, jaxtyped
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from radium.systems.components.sensing.vision.render import CameraIntrinsics
from radium.systems.drone_landing.env import DroneLandingEnv, DroneObs
from radium.systems.drone_landing.oracle import oracle_policy
from radium.systems.drone_landing.policy import DroneLandingPolicy
from radium.systems.drone_landing.scene import DroneLandingScene
from radium.types import PRNGKeyArray

#############################################################################
# Define some utilities for generating training trajectories and generalized
# advantage estimation.
#############################################################################


class Trajectory(NamedTuple):
    observations: DroneObs
    actions: Float[Array, " n_actions"]
    expert_actions: Float[Array, " n_actions"] = None


@jaxtyped(typechecker=beartype)
def generate_trajectory_learner(
    env: DroneLandingEnv,
    policy: DroneLandingPolicy,
    key: PRNGKeyArray,
    rollout_length: int,
) -> Trajectory:
    """Rollout the policy and generate a trajectory to train on.

    Args:
        env: The environment to rollout in.
        policy: The policy to rollout.
        key: The PRNG key to use for sampling.
        rollout_length: The length of the trajectory.

    Returns:
        The trajectory generated by the rollout.
    """

    # Create a function to take one step with the policy
    @scan_tqdm(rollout_length, message="Learner rollout")
    def step(carry, scan_input):
        # Unpack the input
        _, key = scan_input  # first element is index for tqdm
        # Unpack the carry
        obs, state = carry

        # Sample an action from the policy
        action = policy(obs)

        # Also get the expert's label for this state
        key, subkey = jrandom.split(key)
        expert_action = oracle_policy(state, 20, env, subkey, 50)

        # Take a step in the environment using that action
        next_state, next_observation, _, _ = env.step(state, action, key)

        next_carry = (next_observation, next_state)
        output = (obs, action, expert_action)
        return next_carry, output

    # Get the initial state
    reset_key, rollout_key = jrandom.split(key)
    initial_state = env.reset(reset_key)
    initial_obs = env.get_obs(initial_state)

    # Transform and rollout!
    keys = jrandom.split(rollout_key, rollout_length)
    _, (obs, learner_actions, expert_actions) = jax.lax.scan(
        step, (initial_obs, initial_state), (jnp.arange(rollout_length), keys)
    )

    # Save all this information in a trajectory object
    trajectory = Trajectory(
        observations=obs,
        actions=learner_actions,
        expert_actions=expert_actions,
    )

    return trajectory


@jaxtyped(typechecker=beartype)
def generate_trajectory_expert(
    env: DroneLandingEnv,
    key: PRNGKeyArray,
    rollout_length: int,
    action_noise: float = 0.5,
) -> Trajectory:
    """Rollout the policy and generate an expert trajectory to train on.

    Args:
        env: The environment to rollout in.
        key: The PRNG key to use for sampling.
        rollout_length: The length of the trajectory.
        action_noise: The amount of noise to add to the actions.

    Returns:
        The trajectory generated by the rollout.
    """

    # Create a function to take one step with the policy
    @scan_tqdm(rollout_length, message="Expert rollout")
    def step(carry, scan_input):
        # Unpack the input
        _, key = scan_input  # first element is index for tqdm
        action_key, noise_key, step_key = jrandom.split(key, 3)

        # Unpack the carry
        obs, state = carry

        # Sample an action from the policy
        horizon = 20
        optim_iters = 50
        action = oracle_policy(state, horizon, env, action_key, optim_iters)
        action += jrandom.normal(noise_key, action.shape) * action_noise

        # Take a step in the environment using that action
        next_state, next_observation, _, _ = env.step(state, action, step_key)

        next_carry = (next_observation, next_state)
        output = (obs, action)
        return next_carry, output

    # Get the initial state
    reset_key, rollout_key = jrandom.split(key)
    initial_state = env.reset(reset_key)
    initial_obs = env.get_obs(initial_state)

    # Transform and rollout!
    keys = jrandom.split(rollout_key, rollout_length)
    _, (obs, actions) = jax.lax.scan(
        step, (initial_obs, initial_state), (jnp.arange(rollout_length), keys)
    )

    # Save all this information in a trajectory object
    trajectory = Trajectory(
        observations=obs,
        actions=actions,
    )

    return trajectory


@jaxtyped(typechecker=beartype)
def shuffle_trajectory(traj: Trajectory, key: PRNGKeyArray) -> Trajectory:
    """Shuffle the trajectory.

    Args:
        traj: The trajectory to shuffle.
        key: The PRNG key to use for shuffling.

    Returns:
        The shuffled trajectory.
    """
    # Shuffle the trajectory
    traj_len = traj.actions.shape[0]
    permutation = jrandom.permutation(key, traj_len)
    traj = jtu.tree_map(lambda x: x[permutation], traj)

    return traj


def save_traj_imgs(trajectory: Trajectory, logdir: str, epoch_num: int) -> None:
    """Save the given trajectory to a gif."""
    color_images = trajectory.observations.image
    img_dir = os.path.join(logdir, f"epoch_{epoch_num}_imgs")
    os.makedirs(img_dir, exist_ok=True)
    for i, img in enumerate(color_images):
        matplotlib.image.imsave(
            os.path.join(img_dir, f"img_{i}.png"), img.transpose(1, 0, 2)
        )


def make_drone_landing_env(
    image_shape: Tuple[int, int], num_trees: int = 10, moving_obstacles: bool = False
):
    """Make the drone landing environment."""
    scene = DroneLandingScene()
    intrinsics = CameraIntrinsics(
        focal_length=0.1,
        sensor_size=(0.1, 0.1),
        resolution=image_shape,
    )

    initial_drone_state = jnp.array([-10.0, 0.0, 1.0, 0.0])
    env = DroneLandingEnv(
        scene,
        intrinsics,
        dt=0.1,
        num_trees=num_trees,
        initial_drone_state=initial_drone_state,
        collision_penalty=25.0,
        moving_obstacles=moving_obstacles,
    )
    return env


def train_ppo_drone(
    image_shape: Tuple[int, int],
    learning_rate: float = 1e-5,
    seed: int = 0,
    steps_per_epoch: int = 32 * 20,
    epochs: int = 200,
    gd_steps_per_update: int = 1000,
    minibatch_size: int = 32,
    moving_obstacles: bool = True,
    logdir: str = "./tmp/oracle_32_moving",
):
    """
    Train the drone using behavior cloning.

    Args: various hyperparameters.moving_obstacles: bool = False
    """
    if moving_obstacles:
        logdir += "_moving_obstacles"

    # Set up logging
    writer = SummaryWriter(logdir)

    # Set up the environment
    env = make_drone_landing_env(image_shape, moving_obstacles=moving_obstacles)

    # Set up the policy
    key = jrandom.PRNGKey(seed)
    key, subkey = jrandom.split(key)
    policy = DroneLandingPolicy(subkey, image_shape)

    # Set up the optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(policy, eqx.is_array))

    # Compile a loss function and optimization update
    @eqx.filter_value_and_grad
    def loss_fn(policy: DroneLandingPolicy, trajectory: Trajectory) -> Float[Array, ""]:
        # Compute the predicted action for each observation
        predicted_actions = jax.vmap(policy)(trajectory.observations)

        # Minimize L2 loss between predicted and actual actions
        action_loss = jnp.mean(
            jnp.square(predicted_actions - trajectory.expert_actions)
        )

        return action_loss

    @eqx.filter_jit
    def step_fn(
        opt_state: optax.OptState, policy: DroneLandingPolicy, trajectory: Trajectory
    ):
        loss, grad = loss_fn(policy, trajectory)
        updates, opt_state = optimizer.update(grad, opt_state)
        policy = eqx.apply_updates(policy, updates)
        return (
            loss,
            policy,
            opt_state,
        )

    # Training loop
    for epoch in range(epochs):
        # Generate a trajectory with the learner policy, getting labels from the expert
        key, subkey = jrandom.split(key)
        trajectory = generate_trajectory_learner(env, policy, subkey, steps_per_epoch)

        # Save regularly
        if epoch % 10 == 0 or epoch == epochs - 1:
            # Save trajectory images; can be converted to video using this command:
            # ffmpeg -framerate 10 -i img_%d.png -c:v libx264 -r 30 -vf \
            #   scale=320x320:flags=neighbor out.mp4
            save_traj_imgs(trajectory, logdir, epoch)
            # Save policy
            eqx.tree_serialise_leaves(
                os.path.join(logdir, f"policy_{epoch}.eqx"), policy
            )

        # Shuffle the trajectory into minibatches
        key, subkey = jrandom.split(key)
        trajectory = shuffle_trajectory(trajectory, subkey)

        # Compute the loss and gradient
        for i in tqdm(range(gd_steps_per_update)):
            epoch_loss = 0.0
            batches = 0
            for batch_start in range(0, steps_per_epoch, minibatch_size):
                key, subkey = jrandom.split(key)
                (
                    loss,
                    policy,
                    opt_state,
                ) = step_fn(
                    opt_state,
                    policy,
                    jtu.tree_map(
                        lambda x: x[batch_start : batch_start + minibatch_size],
                        trajectory,
                    ),
                )
                batches += 1
                epoch_loss += loss.item()

            # Average
            epoch_loss /= batches

        # Log the loss
        print(f"Epoch {epoch:03d}; loss: {epoch_loss:.2f}")
        writer.add_scalar("loss", epoch_loss, epoch)


if __name__ == "__main__":
    train_ppo_drone((32, 32))
