"""Code to predict and mitigate failure modes in the grasping scenario."""
import argparse
import json
import os
import shutil
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
import matplotlib.pyplot as plt
from beartype.typing import NamedTuple
from jaxtyping import Array, Float, Shaped

from architect.engines import predict_and_mitigate_failure_modes
from architect.engines.reinforce import init_sampler as init_reinforce_sampler
from architect.engines.reinforce import make_kernel as make_reinforce_kernel
from architect.engines.samplers import init_sampler as init_mcmc_sampler
from architect.engines.samplers import make_kernel as make_mcmc_kernel
from architect.systems.simple_grasping.env import (
    clamp_force_in_friction_cone,
    make_grasping_scene,
    normal_at_point,
    render_rgbd,
)
from architect.systems.simple_grasping.policy import AffordancePredictor
from architect.types import PRNGKeyArray
from architect.utils import softmin


class SimulationResults(NamedTuple):
    """A class for storing the results of a simulation."""

    potential: Float[Array, ""]
    grasp_pt: Float[Array, "2 3"]
    grasp_pt_gt: Float[Array, "2 3"]
    object_position: Float[Array, " 2"]
    object_rotation: Float[Array, ""]
    distractor_position: Float[Array, ""]
    normal_force: Float[Array, "2 3"]
    tangential_force: Float[Array, "2 3"]
    depth_image: Float[Array, "H W"]
    gt_affordance: Float[Array, "H W"]
    predicted_affordance: Float[Array, "H W"]
    max_affordance_gt_xy: Float[Array, " 2"]
    max_affordance_pred_xy: Float[Array, " 2"]


class GraspExogenousParams(NamedTuple):
    """A class for storing the exogenous parameters of a grasp."""

    object_position: Float[Array, " 2"]
    object_rotation: Float[Array, ""]
    distractor_position: Float[Array, ""]  # y position of distractor


def sample_ep(key: PRNGKeyArray) -> GraspExogenousParams:
    """Sample the exogenous parameters of a grasp."""
    loc_key, rot_key, distractor_key = jrandom.split(key, 3)
    object_position = 0.1 * jrandom.normal(loc_key, shape=(2,))
    object_rotation = jnp.pi / 4 * jrandom.normal(rot_key, shape=()) + jnp.pi / 2
    distractor_position = 0.5 * jrandom.normal(distractor_key) - 0.5

    return GraspExogenousParams(
        object_position=object_position,
        object_rotation=object_rotation,
        distractor_position=distractor_position,
    )


def ep_logprior(ep: GraspExogenousParams) -> Float[Array, ""]:
    """Return the log prior probability for the given exogenous parameters."""
    return (
        jax.scipy.stats.norm.logpdf(ep.object_position, scale=0.1).sum()
        + jax.scipy.stats.norm.logpdf(
            ep.object_rotation, scale=jnp.pi / 4, loc=jnp.pi / 2
        )
        + jax.scipy.stats.norm.logpdf(ep.distractor_position, scale=0.5, loc=-0.5)
    )


def simulate(
    object_type: str,
    ep: GraspExogenousParams,
    policy: AffordancePredictor,
    static_policy: AffordancePredictor,
) -> Float[Array, ""]:
    """Simulate the grasping environment.

    Disables randomness in the policy and environment (all randomness should be
    factored out into the initial_state argument).

    Args:
        ep: The exogenous parameters of the grasp.
        policy: The parts of the policy that are design parameters.
        static_policy: the parts of the policy that are not design parameters.

    Returns:
        SimulationResults object
    """
    # Merge the policy back together
    affordance_predictor = eqx.combine(policy, static_policy)

    # Make the scene
    camera_pos = jnp.array([-1.0, 0.0, 1.0])
    scene, grasp_gt = make_grasping_scene(
        mug_location=ep.object_position,
        mug_rotation=ep.object_rotation,
        distractor_location=jnp.array([-0.6, ep.distractor_position]),
        sharpness=50.0,
        object=object_type,
    )
    depth_image, gt_affordance = render_rgbd(scene, camera_pos)
    gt_affordance = jnp.mean(gt_affordance, axis=-1)

    # Predict a grasp pose and affordances
    predicted_affordance, grasp_pose = affordance_predictor(depth_image)

    # Extract the maximum affordance from the ground truth and prediction
    gt_affordance_norm = (gt_affordance - gt_affordance.mean()) / gt_affordance.std()
    predicted_affordance_norm = (
        predicted_affordance - predicted_affordance.mean()
    ) / predicted_affordance.std()
    max_affordance_gt = jax.nn.softmax(gt_affordance_norm, axis=(0, 1))
    max_affordance_pred = jax.nn.softmax(predicted_affordance_norm, axis=(0, 1))
    w, h = max_affordance_gt.shape
    x = jnp.linspace(0, 1.0, w)
    y = jnp.linspace(0, 1.0, h)
    X, Y = jnp.meshgrid(x, y)
    max_affordance_gt_x = jnp.sum(X * max_affordance_gt)
    max_affordance_gt_y = jnp.sum(Y * max_affordance_gt)
    max_affordance_pred_x = jnp.sum(X * max_affordance_pred)
    max_affordance_pred_y = jnp.sum(Y * max_affordance_pred)

    # Compute the grasp potential as the distance between the predicted and
    # ground truth grasp affordances
    potential = jnp.sqrt(
        (max_affordance_gt_x - max_affordance_pred_x) ** 2
        + (max_affordance_gt_y - max_affordance_pred_y) ** 2
        + 1e-3
    )

    # Get the grasp forces at that point
    clamp_strength = 1.0
    grasp1 = grasp_pose[0]
    grasp2 = grasp_pose[1]
    grasp1_to_2 = grasp2 - grasp1
    grasp1_to_2 /= jnp.sqrt(jnp.sum(grasp1_to_2**2) + 1e-3)
    grasp1_clamp_force = clamp_strength * grasp1_to_2
    grasp2_clamp_force = -clamp_strength * grasp1_to_2
    grasp1_normal = normal_at_point(scene, grasp1)
    grasp2_normal = normal_at_point(scene, grasp2)
    grasp1_normal_force, grasp1_tangential_force = clamp_force_in_friction_cone(
        clamp_force=grasp1_clamp_force, normal=grasp1_normal, mu=1.0, sharpness=10.0
    )
    grasp2_normal_force, grasp2_tangential_force = clamp_force_in_friction_cone(
        clamp_force=grasp2_clamp_force, normal=grasp2_normal, mu=1.0, sharpness=10.0
    )

    # # The potential is the distance from the true grasp point in centimeters
    # potential = 10 * jnp.sqrt(jnp.sum((grasp_pose - grasp_gt) ** 2) + 1e-3)
    # # The potential is the total tangential force (slipping)
    # potential = jnp.sum(grasp1_tangential_force**2) + jnp.sum(
    #     grasp2_tangential_force**2
    # )

    return SimulationResults(
        potential=potential,
        grasp_pt=jnp.vstack([grasp1, grasp2]),
        grasp_pt_gt=grasp_gt,
        object_position=ep.object_position,
        object_rotation=ep.object_rotation,
        distractor_position=ep.distractor_position,
        normal_force=jnp.vstack([grasp1_normal_force, grasp2_normal_force]),
        tangential_force=jnp.vstack([grasp1_tangential_force, grasp2_tangential_force]),
        depth_image=depth_image,
        gt_affordance=gt_affordance,
        predicted_affordance=predicted_affordance,
        max_affordance_gt_xy=jnp.array([max_affordance_gt_x, max_affordance_gt_y]),
        max_affordance_pred_xy=jnp.array(
            [max_affordance_pred_x, max_affordance_pred_y]
        ),
    )


if __name__ == "__main__":
    # Set up arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--savename", type=str, default="grasping")
    parser.add_argument("--object_type", type=str, default="box")
    parser.add_argument("--seed", type=int, nargs="?", default=0)
    parser.add_argument("--L", type=float, nargs="?", default=1.0)
    parser.add_argument("--failure_level", type=float, nargs="?", default=0.2)
    parser.add_argument("--dp_logprior_scale", type=float, nargs="?", default=1.0)
    parser.add_argument("--dp_mcmc_step_size", type=float, nargs="?", default=1e-5)
    parser.add_argument("--ep_mcmc_step_size", type=float, nargs="?", default=1e-5)
    parser.add_argument("--num_rounds", type=int, nargs="?", default=100)
    parser.add_argument("--num_steps_per_round", type=int, nargs="?", default=10)
    parser.add_argument("--num_chains", type=int, nargs="?", default=10)
    parser.add_argument("--quench_rounds", type=int, nargs="?", default=0)
    parser.add_argument("--disable_gradients", action="store_true")
    parser.add_argument("--disable_stochasticity", action="store_true")
    parser.add_argument("--disable_mh", action="store_true")
    parser.add_argument("--reinforce", action="store_true")
    parser.add_argument("--moving_obstacles", action="store_true")
    boolean_action = argparse.BooleanOptionalAction
    parser.add_argument("--repair", action=boolean_action, default=False)
    parser.add_argument("--predict", action=boolean_action, default=True)
    parser.add_argument("--temper", action=boolean_action, default=True)
    parser.add_argument("--grad_clip", type=float, nargs="?", default=float("inf"))
    parser.add_argument("--dont_normalize_gradients", action="store_true")
    args = parser.parse_args()

    # Hyperparameters
    L = args.L
    seed = args.seed
    object_type = args.object_type
    failure_level = args.failure_level
    dp_logprior_scale = args.dp_logprior_scale
    dp_mcmc_step_size = args.dp_mcmc_step_size
    ep_mcmc_step_size = args.ep_mcmc_step_size
    num_rounds = args.num_rounds
    num_steps_per_round = args.num_steps_per_round
    num_chains = args.num_chains
    use_gradients = not args.disable_gradients
    use_stochasticity = not args.disable_stochasticity
    use_mh = not args.disable_mh
    reinforce = args.reinforce
    repair = args.repair
    predict = args.predict
    temper = args.temper
    quench_rounds = args.quench_rounds
    grad_clip = args.grad_clip
    normalize_gradients = not args.dont_normalize_gradients

    print("Running prediction/mitigation on grasping with hyperparameters:")
    print(f"\tmodel_path = {args.model_path}")
    print(f"\tobject = {object_type}")
    print(f"\tseed = {seed}")
    print(f"\timage dimensions (w x h) = 64 x 64")
    print(f"\tL = {L}")
    print(f"\tfailure_level = {failure_level}")
    print(f"\tdp_logprior_scale = {dp_logprior_scale}")
    print(f"\tdp_mcmc_step_size = {dp_mcmc_step_size}")
    print(f"\tep_mcmc_step_size = {ep_mcmc_step_size}")
    print(f"\tnum_rounds = {num_rounds}")
    print(f"\tnum_steps_per_round = {num_steps_per_round}")
    print(f"\tnum_chains = {num_chains}")
    print(f"\tuse_gradients = {use_gradients}")
    print(f"\tuse_stochasticity = {use_stochasticity}")
    print(f"\tuse_mh = {use_mh}")
    print(f"\trepair = {repair}")
    print(f"\tpredict = {predict}")
    print(f"\ttemper = {temper}")
    print(f"\tquench_rounds = {quench_rounds}")
    print(f"\tgrad_clip = {grad_clip}")
    print(f"\tnormalize_gradients = {normalize_gradients}")
    print(
        f"\tUsing alternative algorithm? {reinforce}",
        f"(reinforce = {reinforce})",
    )

    # Add exponential tempering if using
    t = jnp.linspace(0, 1, num_rounds) + 0.1
    tempering_schedule = 1 - jnp.exp(-40 * t) if temper else None

    # Make a PRNG key (#sorandom)
    prng_key = jrandom.PRNGKey(seed)

    # Load the model (key doesn't matter; we'll replace all leaves with the saved
    # parameters), duplicating the model for each chain. We'll also split partition
    # out just the continuous parameters, which will be our design parameters
    dummy_policy = AffordancePredictor(jrandom.PRNGKey(0))
    load_policy = lambda _: eqx.tree_deserialise_leaves(args.model_path, dummy_policy)
    get_dps = lambda _: eqx.partition(load_policy(_), eqx.is_array)[0]
    initial_dps = eqx.filter_vmap(get_dps)(jnp.arange(num_chains))
    # Also save out the static part of the policy
    initial_dp, static_policy = eqx.partition(load_policy(None), eqx.is_array)

    # Make a prior logprob for the policy that penalizes large updates to the policy
    # parameters
    def dp_prior_logprob(dp):
        # Take a mean rather than a sum to make this not crazy large
        block_logprobs = jtu.tree_map(
            lambda x_updated, x: jax.scipy.stats.norm.logpdf(
                x_updated - x, scale=dp_logprior_scale
            ).mean(),
            dp,
            initial_dp,
        )
        # Take a block mean rather than the sum of all blocks to avoid a crazy large
        # logprob
        overall_logprob = jax.flatten_util.ravel_pytree(block_logprobs)[0].mean()
        return overall_logprob

    # Initialize some initial states (these serve as our initial exogenous parameters)
    prng_key, ep_keys = jrandom.split(prng_key)
    ep_keys = jrandom.split(ep_keys, num_chains)
    initial_eps = jax.vmap(sample_ep)(ep_keys)

    # Choose which sampler to use
    if reinforce:
        init_sampler_fn = init_reinforce_sampler
        make_kernel_fn = lambda _, logprob_fn, step_size, _: make_reinforce_kernel(
            logprob_fn,
            step_size,
            perturbation_stddev=0.05,
            baseline_update_rate=0.5,
        )
    else:
        # This sampler yields either MALA, GD, or RMH depending on whether gradients
        # and/or stochasticity are enabled
        init_sampler_fn = lambda params, logprob_fn: init_mcmc_sampler(
            params,
            logprob_fn,
            normalize_gradients,
        )
        make_kernel_fn = (
            lambda _, logprob_fn, step_size, stochasticity: make_mcmc_kernel(
                logprob_fn,
                step_size,
                use_gradients,
                stochasticity,
                grad_clip,
                normalize_gradients,
                use_mh,
            )
        )

    if reinforce:
        alg_type = "reinforce_l2c_0.05_step"
    elif use_gradients and use_stochasticity and use_mh:
        alg_type = "mala"
    elif use_gradients and use_stochasticity and not use_mh:
        alg_type = "ula"
    elif use_gradients and not use_stochasticity:
        alg_type = "gd"
    elif not use_gradients and use_stochasticity and use_mh:
        alg_type = "rmh"
    elif not use_gradients and use_stochasticity and not use_mh:
        alg_type = "random_walk"
    else:
        alg_type = "static"

    # Run the prediction+mitigation process
    t_start = time.perf_counter()
    dps, eps, dp_logprobs, ep_logprobs = predict_and_mitigate_failure_modes(
        prng_key,
        initial_dps,
        initial_eps,
        dp_logprior_fn=dp_prior_logprob,
        ep_logprior_fn=ep_logprior,
        ep_potential_fn=lambda dp, ep: -L
        * jax.nn.elu(
            failure_level - simulate(object_type, ep, dp, static_policy).potential
        ),
        dp_potential_fn=lambda dp, ep: -L
        * jax.nn.elu(
            simulate(object_type, ep, dp, static_policy).potential - failure_level
        ),
        init_sampler=init_sampler_fn,
        make_kernel=make_kernel_fn,
        num_rounds=num_rounds,
        num_mcmc_steps_per_round=num_steps_per_round,
        dp_mcmc_step_size=dp_mcmc_step_size,
        ep_mcmc_step_size=ep_mcmc_step_size,
        use_stochasticity=use_stochasticity,
        repair=repair,
        predict=predict,
        quench_rounds=quench_rounds,
        tempering_schedule=tempering_schedule,
        logging_prefix=f"{args.savename}/{alg_type}[{os.getpid()}]",
    )
    t_end = time.perf_counter()
    print(
        f"Ran {num_rounds:,} rounds with {num_chains} chains in {t_end - t_start:.2f} s"
    )

    # Select the policy that performs best against all predicted failures before
    # the final round (choose from all chains)
    if repair:
        most_likely_dps_idx = jnp.argmax(dp_logprobs[-1], axis=-1)
        final_dps = jtu.tree_map(lambda leaf: leaf[-1, most_likely_dps_idx], dps)
    else:
        # Just pick one policy arbitrarily if we didn't optimize the policies.
        final_dps = jtu.tree_map(lambda leaf: leaf[-1, 0], dps)

    # Evaluate this single policy against all failures
    final_eps = jtu.tree_map(lambda leaf: leaf[-1], eps)

    # # TODO for debugging plots, just use the initial policy and eps
    # t_end = 0.0
    # t_start = 0.0
    # final_dps = jtu.tree_map(lambda leaf: leaf[-1], initial_dps)
    # final_eps = initial_eps
    # dp_logprobs = jnp.zeros((num_rounds, num_chains))
    # ep_logprobs = jnp.zeros((num_rounds, num_chains))
    # # TODO debugging bit ends here

    # Evaluate the solutions proposed by the prediction+mitigation algorithm
    result = eqx.filter_vmap(
        lambda dp, ep: simulate(object_type, ep, dp, static_policy),
        in_axes=(None, 0),
    )(final_dps, final_eps)

    # Plot the results
    fig = plt.figure(figsize=(32, 16), constrained_layout=True)
    axs = fig.subplot_mosaic(
        [
            ["trace", "trace", "object_pos", "object_rot", "distractor"],
            ["depth", "depth", "depth", "depth", "depth"],
            ["gt", "gt", "gt", "gt", "gt"],
            ["predicted", "predicted", "predicted", "predicted", "predicted"],
            ["potential", "potential", "potential", "potential", "potential"],
        ],
    )

    # Plot the chain convergence
    if predict:
        axs["trace"].plot(ep_logprobs)
        axs["trace"].set_ylabel("Log probability after contingency update")
    else:
        axs["trace"].plot(dp_logprobs)
        axs["trace"].set_ylabel("Log probability after repair")

    axs["trace"].set_xlabel("# Samples")

    # Plot the exogenous parameters, color-coded by potential
    max_potential = jnp.max(result.potential)
    min_potential = jnp.min(result.potential)
    normalized_potential = (result.potential - min_potential) / (
        max_potential - min_potential
    )
    axs["object_pos"].scatter(
        result.object_position[:, 0],
        result.object_position[:, 1],
        marker="o",
        color=plt.cm.plasma_r(normalized_potential),
    )
    axs["object_pos"].set_xlabel("Object x")
    axs["object_pos"].set_ylabel("Object y")

    axs["object_rot"].scatter(
        result.object_rotation,
        jnp.zeros_like(result.object_rotation),
        marker="o",
        color=plt.cm.plasma_r(normalized_potential),
    )
    axs["object_rot"].set_xlabel("Object rotation")

    axs["distractor"].scatter(
        result.distractor_position,
        jnp.zeros_like(result.distractor_position),
        marker="o",
        color=plt.cm.plasma_r(normalized_potential),
    )
    axs["distractor"].set_xlabel("Distractor y")

    # Plot the potential across all failure cases
    axs["potential"].plot(result.potential, "ko")
    axs["potential"].set_ylabel("Potential")

    # Plot the depth image, ground truth affordance, and predicted affordance
    axs["depth"].imshow(jnp.concatenate(result.depth_image.transpose(0, 2, 1), axis=1))
    axs["gt"].imshow(jnp.concatenate(result.gt_affordance.transpose(0, 2, 1), axis=1))
    axs["predicted"].imshow(
        jnp.concatenate(result.predicted_affordance.transpose(0, 2, 1), axis=1)
    )

    # Overlay the max affordance on these images
    w, h = result.depth_image.shape[1:]
    for i in range(num_chains):
        max_gt = result.max_affordance_gt_xy[i]
        max_pred = result.max_affordance_pred_xy[i]
        axs["gt"].scatter(
            max_gt[1] * h + h * i,
            max_gt[0] * w,
            marker="x",
            color="red",
            s=50,
        )
        axs["predicted"].scatter(
            max_pred[1] * h + h * i,
            max_pred[0] * w,
            marker="x",
            color="red",
            s=50,
        )

    save_dir = (
        f"results/{args.savename}_{object_type}/{'predict' if predict else ''}"
        f"{'_' if repair else ''}{'repair_' + str(dp_logprior_scale) if repair else ''}/"
        f"L_{L:0.1e}/"
        f"{num_rounds * num_steps_per_round}_samples_{num_rounds}x{num_steps_per_round}/"
        f"{num_chains}_chains/"
        f"{quench_rounds}_quench/"
        f"dp_{dp_mcmc_step_size:0.1e}/"
        f"ep_{ep_mcmc_step_size:0.1e}/"
        f"{'grad_norm' if normalize_gradients else 'no_grad_norm'}/"
        f"grad_clip_{grad_clip}/"
    )
    filename = save_dir + f"{alg_type}{'_tempered_40+0.1' if temper else ''}_{seed}"
    print(f"Saving results to: {filename}")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(filename + ".png")

    # Save the final design parameters (joined back into the full policy)
    final_policy = eqx.combine(final_dps, static_policy)
    eqx.tree_serialise_leaves(filename + ".eqx", final_policy)

    # Save the trace of policies
    eqx.tree_serialise_leaves(filename + "_trace.eqx", dps)

    # Save the initial policy
    try:
        shutil.copyfile(
            args.model_path,
            f"results/{args.savename}_{object_type}/" + "initial_policy.eqx",
        )
    except shutil.SameFileError:
        print("Initial policy already copied to results.")

    # Save the hyperparameters
    with open(filename + ".json", "w") as f:
        json.dump(
            {
                "eps": final_eps._asdict(),
                "eps_trace": eps._asdict(),
                "object_type": object_type,
                "L": L,
                "failure_level": failure_level,
                "dp_logprior_scale": dp_logprior_scale,
                "dp_mcmc_step_size": dp_mcmc_step_size,
                "ep_mcmc_step_size": ep_mcmc_step_size,
                "num_rounds": num_rounds,
                "num_steps_per_round": num_steps_per_round,
                "num_chains": num_chains,
                "use_gradients": use_gradients,
                "use_stochasticity": use_stochasticity,
                "repair": repair,
                "predict": predict,
                "quench_rounds": quench_rounds,
                "tempering_schedule": tempering_schedule,
                "ep_logprobs": ep_logprobs,
                "dp_logprobs": dp_logprobs,
            },
            f,
            default=lambda x: x.tolist() if isinstance(x, Shaped[Array, "..."]) else x,
        )
