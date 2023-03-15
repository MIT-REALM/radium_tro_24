"""Train an agent for the highway environment using PPO."""
import os
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
from jaxtyping import Float, Array, Bool, jaxtyped
from beartype import beartype
from beartype.typing import NamedTuple, Tuple
from torch.utils.tensorboard import SummaryWriter
import equinox as eqx
import optax
import matplotlib.pyplot as plt

from architect.types import PRNGKeyArray
from architect.systems.components.sensing.vision.render import CameraIntrinsics
from architect.systems.highway.driving_policy import DrivingPolicy
from architect.systems.highway.highway_scene import HighwayScene
from architect.systems.highway.highway_env import (
    HighwayEnv,
    HighwayObs,
)


#############################################################################
# Define some utilities for generating training trajectories and generalized
# advantage estimation.
#############################################################################


class Trajectory(NamedTuple):
    observations: HighwayObs
    actions: Float[Array, " n_actions"]
    action_log_probs: Float[Array, " n_actions"]
    rewards: Float[Array, ""]
    returns: Float[Array, ""]
    advantages: Float[Array, ""]
    done: Bool[Array, ""]


@jaxtyped
@beartype
def generalized_advantage_estimate(
    rewards: Float[Array, " n_steps"],
    values: Float[Array, " n_steps+1"],
    dones: Bool[Array, " n_steps"],
    gamma: float,
    lam: float,
) -> Tuple[Float[Array, " n_steps"], Float[Array, " n_steps"]]:
    """Compute the generalized advantage estimation for a trajectory.

    Args:
        rewards: The rewards for each step in the trajectory.
        values: The values for each step in the trajectory (plus the value for the final
            state).
        dones: True if a step in the trajectory is a terminal state, False otherwise.
        gamma: The discount factor for GAE.
        lam: The lambda factor for GAE.

    Returns:
        The advantages and returns for each step in the trajectory.
    """

    def gae_step(advantage, gae_input):
        # Unpack input
        (
            reward,  # reward in current state
            current_value,  # value estimate in current state
            next_value,  # value estimate in next state
            terminal,  # is the current state terminal?
        ) = gae_input

        # Difference between current value estimate and reward-to-go
        delta = reward + gamma * next_value * (1.0 - terminal) - current_value
        # Advantage estimate
        advantage = delta + gamma * lam * advantage * (1.0 - terminal)
        advantage = advantage.reshape()

        return advantage, advantage  # carry and output

    # Compute the advantage estimate for each step in the trajectory
    _, advantages = jax.lax.scan(
        gae_step, jnp.array(0.0), (rewards, values[:-1], values[1:], dones)
    )

    # The return is the current state value + the advantage
    returns = advantages + values[:-1]

    return advantages, returns


@jaxtyped
@beartype
def generate_trajectory(
    env: HighwayEnv,
    policy: DrivingPolicy,
    non_ego_actions: Float[Array, "n_non_ego n_actions"],
    key: PRNGKeyArray,
    rollout_length: int,
    gamma: float,
    gae_lambda: float,
    action_noise: float,
) -> Trajectory:
    """Rollout the policy and generate a trajectory to train on.

    Args:
        env: The environment to rollout in.
        policy: The policy to rollout.
        non_ego_actions: The actions for the non-ego vehicles (held constant).
        key: The PRNG key to use for sampling.
        rollout_length: The length of the trajectory.
        gamma: The discount factor for GAE.
        gae_lambda: The lambda parameter for GAE.
        action_noise: The standard deviation of the noise to add to the actions.

    Returns:
        The trajectory generated by the rollout.
    """

    # Create a function to take one step with the policy
    def step(carry, key: PRNGKeyArray):
        # Unpack the carry
        obs, state = carry

        # PRNG key management
        step_key, action_subkey = jrandom.split(key)

        # Sample an action from the policy
        action, action_logprob, value = policy(obs, action_subkey, action_noise)

        # Take a step in the environment using that action
        next_state, next_observation, reward, done = env.step(
            state, action, non_ego_actions, key
        )

        next_carry = (next_observation, next_state)
        output = (obs, action, action_logprob, reward, value, done)
        return next_carry, output

    # Get the initial state
    reset_key, rollout_key = jrandom.split(key)
    initial_state = env.reset(reset_key)
    initial_obs = env.get_obs(initial_state)

    # Transform and rollout!
    keys = jrandom.split(rollout_key, rollout_length)
    _, (obs, actions, action_logprobs, rewards, values, dones) = jax.lax.scan(
        step, (initial_obs, initial_state), keys
    )

    # Compute the advantage estimate. This requires the value estimate at the
    # end of the rollout. The key we use doesn't matter here.
    _, _, final_value = policy(jtu.tree_map(lambda x: x[-1], obs), keys[-1])
    values = jnp.concatenate([values, jnp.expand_dims(final_value, 0)], axis=0)
    advantage, returns = generalized_advantage_estimate(
        rewards, values, dones, gamma, gae_lambda
    )

    # Save all this information in a trajectory object
    trajectory = Trajectory(
        observations=obs,
        actions=actions,
        action_log_probs=action_logprobs,
        rewards=rewards,
        returns=returns,
        advantages=advantage,
        done=dones,
    )

    return trajectory


@jaxtyped
@beartype
def shuffle_trajectory(traj: Trajectory, key: PRNGKeyArray) -> Trajectory:
    """Shuffle the trajectory.

    Args:
        traj: The trajectory to shuffle.
        key: The PRNG key to use for shuffling.

    Returns:
        The shuffled trajectory.
    """
    # Shuffle the trajectory
    traj_len = traj.done.shape[0]
    permutation = jrandom.permutation(key, traj_len)
    traj = jtu.tree_map(lambda x: x[permutation], traj)

    return traj


@jaxtyped
@beartype
def ppo_clip_loss_fn(
    policy: DrivingPolicy,
    traj: Trajectory,
    epsilon: float,
    critic_weight: float,
    action_noise: float,
) -> Float[Array, ""]:
    """Compute the clipped PPO loss.

    Args:
        policy: The current policy.
        traj: The training trajectory.
        epsilon: The epsilon parameter for the PPO loss.
        critic_weight: The weight for the critic loss.
        action_noise: The standard deviation of the noise to add to the actions.

    Returns:
        The total PPO loss and a tuple of component losses
    """
    # Get the action log probabilities using the current policy, so we can compute
    # the ratio of the action probabilities under the new and old policies.
    # Also get the value, which we'll use to compute the critic loss
    action_logprobs, value_estimate = jax.vmap(
        policy.action_log_prob_and_value, in_axes=(0, 0, None)
    )(traj.observations, traj.actions, action_noise)
    likelihood_ratio = jnp.exp(action_logprobs - traj.action_log_probs)
    clipped_likelihood_ratio = jnp.clip(likelihood_ratio, 1 - epsilon, 1 + epsilon)

    # The PPO loss for the actor is the average minimum of the product of these ratios
    # with the advantage estimate
    actor_loss = -jnp.minimum(
        likelihood_ratio * traj.advantages, clipped_likelihood_ratio * traj.advantages
    ).mean()

    # The critic loss is the mean squared error between the value estimate and the
    # reward-to-go
    critic_loss = critic_weight * jnp.square(traj.returns - value_estimate).mean()

    return actor_loss + critic_loss, (actor_loss, critic_loss)


def save_traj_imgs(trajectory: Trajectory, logdir: str, epoch_num: int) -> None:
    """Save the given trajectory to a gif."""
    depth_images = trajectory.observations.depth_image
    img_dir = os.path.join(logdir, f"epoch_{epoch_num}_imgs")
    os.makedirs(img_dir, exist_ok=True)
    for i, img in enumerate(depth_images):
        plt.imshow(img.T)
        plt.savefig(os.path.join(img_dir, f"{i:04d}.png"))
        plt.close()


def train_ppo_driver(
    image_shape: Tuple[int, int],
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.97,
    epsilon: float = 0.2,
    critic_weight: float = 0.5,
    seed: int = 0,
    steps_per_epoch: int = 320,  # 640,
    epochs: int = 100,
    gd_steps_per_update: int = 80,
    minibatch_size: int = 32,
    logdir: str = "./tmp/ppo",
) -> DrivingPolicy:
    """Train the driver using PPO.

    Args: various hyperparameters.
    """
    # Set up logging
    writer = SummaryWriter(logdir)

    # Set up the environment
    scene = HighwayScene(num_lanes=3, lane_width=4.0)
    intrinsics = CameraIntrinsics(
        focal_length=0.1,
        sensor_size=(0.1, 0.1),
        resolution=image_shape,
    )
    initial_ego_state = jnp.array([-50.0, 0.0, 0.0, 10.0])
    initial_non_ego_states = jnp.array(
        [
            [-28.0, 0.0, 0.0, 10.0],
            [-35, 4.0, 0.0, 9.0],
            [-40, -4.0, 0.0, 11.0],
        ]
    )
    initial_state_covariance = jnp.diag(jnp.array([0.5, 0.5, 0.001, 0.5]) ** 2)
    env = HighwayEnv(
        scene,
        intrinsics,
        dt=0.1,
        initial_ego_state=initial_ego_state,
        initial_non_ego_states=initial_non_ego_states,
        initial_state_covariance=initial_state_covariance,
    )

    # Set up the policy
    key = jrandom.PRNGKey(seed)
    key, subkey = jrandom.split(key)
    policy = DrivingPolicy(subkey, image_shape)

    # Set up the optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(policy, eqx.is_array))

    # Fix non-ego actions to be constant (drive straight at fixed speed)
    n_non_ego = 3
    non_ego_actions = jnp.zeros((n_non_ego, 2))

    # Compile a loss function and optimization update
    @partial(eqx.filter_value_and_grad, has_aux=True)
    def loss_fn(policy: DrivingPolicy, trajectory: Trajectory) -> Float[Array, ""]:
        return ppo_clip_loss_fn(
            policy, trajectory, epsilon, critic_weight, action_noise
        )

    @eqx.filter_jit
    def step_fn(
        opt_state: optax.OptState, policy: DrivingPolicy, trajectory: Trajectory
    ) -> Tuple[Float[Array, ""], DrivingPolicy, optax.OptState]:
        (loss, (actor_loss, critic_loss)), grad = loss_fn(policy, trajectory)
        updates, opt_state = optimizer.update(grad, opt_state)
        policy = eqx.apply_updates(policy, updates)
        return loss, policy, opt_state, (actor_loss, critic_loss)

    # Training loop
    for epoch in range(epochs):
        # Schedule the action noise
        action_noise = 0.01 ** (epoch / epochs)

        # Generate a trajectory
        key, subkey = jrandom.split(key)
        trajectory = generate_trajectory(
            env,
            policy,
            non_ego_actions,
            key,
            steps_per_epoch,
            gamma,
            gae_lambda,
            action_noise,
        )
        if epoch % 5 == 0:
            save_traj_imgs(trajectory, logdir, epoch)

        # Shuffle the trajectory into minibatches
        key, subkey = jrandom.split(key)
        trajectory = shuffle_trajectory(trajectory, subkey)

        # Compute the loss and gradient
        for i in range(gd_steps_per_update):
            epoch_loss = 0.0
            epoch_actor_loss = 0.0
            epoch_critic_loss = 0.0
            for batch_start in range(0, steps_per_epoch, minibatch_size):
                key, subkey = jrandom.split(key)
                loss, policy, opt_state, actor_loss, critic_loss = step_fn(
                    opt_state,
                    policy,
                    jtu.tree_map(
                        lambda x: x[batch_start : batch_start + minibatch_size],
                        trajectory,
                    ),
                )
                epoch_loss += (loss / minibatch_size).item()
                epoch_actor_loss += (actor_loss / minibatch_size).item()
                epoch_critic_loss += (critic_loss / minibatch_size).item()

        # Log the loss
        print(
            (
                f"Epoch {epoch}; loss: {epoch_loss} "
                f"(actor {epoch_actor_loss}, critic {epoch_critic_loss}) "
                f"total_reward: {trajectory.reward.sum()} "
                f"total_return: {trajectory.returns.sum()}"
            )
        )
        writer.add_scalar("loss", epoch_loss, epoch)
        writer.add_scalar("actor loss", epoch_actor_loss, epoch)
        writer.add_scalar("critic loss", epoch_critic_loss, epoch)
        writer.add_scalar("episode reward", trajectory.reward.sum(), epoch)
        writer.add_scalar("episode return", trajectory.returns.sum(), epoch)


if __name__ == "__main__":
    train_ppo_driver((64, 64))
