import argparse
import json
import os
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
import matplotlib
import matplotlib.pyplot as plt
from jax.config import config
from jaxtyping import Array, Shaped

import wandb
from radium.engines import predict_and_mitigate_failure_modes
from radium.engines.reinforce import init_sampler as init_reinforce_sampler
from radium.engines.reinforce import make_kernel as make_reinforce_kernel
from radium.engines.samplers import init_sampler as init_mcmc_sampler
from radium.engines.samplers import make_kernel as make_mcmc_kernel
from radium.systems.formation2d.simulator import (
    KernelWindField,
    connection_strength_prior_logprob,
    sample_random_connection_strengths,
    simulate,
)
from radium.systems.hide_and_seek.hide_and_seek_types import (
    Arena,
    MultiAgentTrajectory,
    Trajectory2D,
)

config.update("jax_debug_nans", True)


def plotting_cb(dp, eps):
    result = jax.vmap(simulate_fn, in_axes=(None, 0))(dp, eps)
    # For later, save the index of the worst contingency
    worst_eps_idx = jnp.argmax(result.potential)

    # Plot the results
    fig = plt.figure(figsize=(32, 16), constrained_layout=True)
    axs = fig.subplot_mosaic(
        [
            ["arena", "arena"],
            ["arena", "arena"],
            ["connectivity", "potential"],
        ]
    )

    # Plot the arena
    axs["arena"].plot(
        [-width / 2, -width / 2, width / 2, width / 2, -width / 2],
        [-height / 2, height / 2, height / 2, -height / 2, -height / 2],
        "k-",
    )

    # Plot initial setup
    axs["arena"].scatter(
        initial_states[:, 0],
        initial_states[:, 1],
        color="k",
        marker="o",
        s=25,
        label="Initial positions",
    )

    axs["arena"].legend()
    axs["arena"].set_aspect("equal")

    # Plot planned trajectories
    t = jnp.linspace(0, 1, 100)
    for traj in dp.trajectories:
        pts = jax.vmap(traj)(t)
        axs["arena"].plot(pts[:, 0], pts[:, 1], "r-")
        axs["arena"].scatter(traj.p[:, 0], traj.p[:, 1], s=25, color="r", marker="x")

    # Plot each agent's trajectory
    for chain in range(num_chains):
        for i in range(n):
            axs["arena"].plot(
                result.positions[chain, :, i, 0],
                result.positions[chain, :, i, 1],
                "k-",
                linewidth=1,
                alpha=0.2,
            )

    # Plot endpoints for each trajectory
    axs["arena"].scatter(
        result.positions[:, -1, :, 0],
        result.positions[:, -1, :, 1],
        s=25,
        color="r",
        marker="o",
    )
    # Plot the goal point
    axs["arena"].scatter(
        goal_com_position[0], goal_com_position[1], s=50, color="g", marker="*"
    )

    # Plot the worst wind speeds
    worst_wind = jax.tree_util.tree_map(lambda leaf: leaf[worst_eps_idx], eps[0])
    wind_speeds = jax.vmap(jax.vmap(worst_wind))(jnp.stack([test_X, test_Y], axis=-1))
    axs["arena"].quiver(
        test_X,
        test_Y,
        wind_speeds[:, :, 0],
        wind_speeds[:, :, 1],
        color="b",
        alpha=0.5,
        angles="xy",
        scale_units="xy",
        scale=10.0,
    )

    # Plot the connectivity
    axs["connectivity"].plot(result.connectivity.T)
    axs["connectivity"].set_xlabel("Time")
    axs["connectivity"].set_ylabel("Connectivity")
    axs["connectivity"].set_title("Connectivity")

    # Plot a scatterplot of min connectivity vs potential
    axs["potential"].scatter(jnp.min(result.connectivity, axis=-1), result.potential)
    axs["potential"].set_xlabel("Min connectivity")
    axs["potential"].set_ylabel("Potential")

    # log the figure to wandb
    wandb.log({"plot": wandb.Image(fig)}, commit=False)

    # Close the figure
    plt.close()


if __name__ == "__main__":
    matplotlib.use("Agg")

    # Set up arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--savename", type=str, default="tro2-formation-collision-halfkernel"
    )
    parser.add_argument("--n", type=int, nargs="?", default=5)
    parser.add_argument("--L", type=float, nargs="?", default=1.0)
    parser.add_argument("--T", type=int, nargs="?", default=3)
    parser.add_argument("--failure_level", type=float, nargs="?", default=50.0)
    parser.add_argument("--seed", type=int, nargs="?", default=0)
    parser.add_argument("--width", type=float, nargs="?", default=3.2)
    parser.add_argument("--height", type=float, nargs="?", default=3.0)
    parser.add_argument("--R", type=float, nargs="?", default=0.5)
    parser.add_argument("--max_wind_thrust", type=float, nargs="?", default=0.5)
    parser.add_argument("--duration", type=float, nargs="?", default=30.0)
    parser.add_argument("--dp_mcmc_step_size", type=float, nargs="?", default=1e-3)
    parser.add_argument("--ep_mcmc_step_size", type=float, nargs="?", default=1e-3)
    parser.add_argument("--num_rounds", type=int, nargs="?", default=50)
    parser.add_argument("--num_mcmc_steps_per_round", type=int, nargs="?", default=50)
    parser.add_argument("--num_chains", type=int, nargs="?", default=10)
    parser.add_argument("--quench_rounds", type=int, nargs="?", default=20)
    parser.add_argument("--disable_gradients", action="store_true")
    parser.add_argument("--disable_stochasticity", action="store_true")
    parser.add_argument("--reinforce", action="store_true")
    parser.add_argument("--num_stress_test_cases", type=int, nargs="?", default=1000)
    boolean_action = argparse.BooleanOptionalAction
    parser.add_argument("--repair", action=boolean_action, default=True)
    parser.add_argument("--predict", action=boolean_action, default=True)
    parser.add_argument("--temper", action=boolean_action, default=False)
    parser.add_argument("--grad_clip", type=float, nargs="?", default=float("inf"))
    args = parser.parse_args()

    # Hyperparameters
    n = args.n
    T = args.T
    L = args.L
    width = args.width
    height = args.height
    failure_level = args.failure_level
    R = args.R
    max_wind_thrust = args.max_wind_thrust
    duration = args.duration
    dp_mcmc_step_size = args.dp_mcmc_step_size
    ep_mcmc_step_size = args.ep_mcmc_step_size
    num_rounds = args.num_rounds
    num_mcmc_steps_per_round = args.num_mcmc_steps_per_round
    num_chains = args.num_chains
    use_gradients = not args.disable_gradients
    use_stochasticity = not args.disable_stochasticity
    repair = args.repair
    predict = args.predict
    temper = args.temper
    quench_rounds = args.quench_rounds
    grad_clip = args.grad_clip
    reinforce = args.reinforce
    num_stress_test_cases = args.num_stress_test_cases

    print("Running Formation with hyperparameters:")
    print(f"\tn = {n}")
    print(f"\tT = {T}")
    print(f"\tL = {L}")
    print(f"\twidth = {width}")
    print(f"\theight = {height}")
    print(f"\tR = {R}")
    print(f"failure_level = {failure_level}")
    print(f"\tmax_wind_thrust = {max_wind_thrust}")
    print(f"\tduration = {duration}")
    print(f"\tdp_mcmc_step_size = {dp_mcmc_step_size}")
    print(f"\tep_mcmc_step_size = {ep_mcmc_step_size}")
    print(f"\tnum_rounds = {num_rounds}")
    print(f"\tnum_mcmc_steps_per_round = {num_mcmc_steps_per_round}")
    print(f"\tnum_chains = {num_chains}")
    print(f"\tuse_gradients = {use_gradients}")
    print(f"\tuse_stochasticity = {use_stochasticity}")
    print(f"\treinforce = {reinforce}")
    print(f"\trepair = {repair}")
    print(f"\tpredict = {predict}")
    print(f"\ttemper = {temper}")
    print(f"\tquench_rounds = {quench_rounds}")
    print(f"\tgrad_clip = {grad_clip}")
    print(f"\tnum_stress_test_cases = {num_stress_test_cases}")

    if reinforce:
        alg_type = "reinforce"
    elif use_gradients and use_stochasticity:
        alg_type = "mala"
    elif use_gradients and not use_stochasticity:
        alg_type = "gd"
    elif not use_gradients and use_stochasticity:
        alg_type = "rmh"
    else:
        alg_type = "static"

    # Initialize logger
    # wandb.init(
    #     project=args.savename + f"-{n}-agents",
    #     group=alg_type
    #     + ("-predict" if predict else "")
    #     + ("-repair" if repair else ""),
    #     config={
    #         "L": L,
    #         "n": n,
    #         "failure_level": failure_level,
    #         "seed": args.seed,
    #         "dp_mcmc_step_size": dp_mcmc_step_size,
    #         "ep_mcmc_step_size": ep_mcmc_step_size,
    #         "num_rounds": num_rounds,
    #         "num_steps_per_round": num_mcmc_steps_per_round,
    #         "num_chains": num_chains,
    #         "use_gradients": use_gradients,
    #         "use_stochasticity": use_stochasticity,
    #         "reinforce": reinforce,
    #         "repair": repair,
    #         "predict": predict,
    #         "temper": temper,
    #         "quench_rounds": quench_rounds,
    #         "grad_clip": grad_clip,
    #         "num_stress_test_cases": num_stress_test_cases,
    #         "max_wind_thrust": max_wind_thrust,
    #     },
    # )

    # Add exponential tempering if using
    t = jnp.linspace(0, 1, num_rounds)
    tempering_schedule = 1 - jnp.exp(-5 * t) if temper else None

    # Set up the simulation
    arena = Arena(width, height, 0.0)
    initial_states = jnp.stack(
        (
            jnp.zeros(n) - width / 2.0 + R,
            jnp.linspace(-height / 2.0 + R, height / 2.0 - R, n),
            jnp.zeros(n),
            jnp.zeros(n),
        )
    ).T
    goal_com_position = jnp.array([width / 2.0 - R, 0.0])

    # Raise a warning if the initial states are too far apart
    d_ij = jnp.linalg.norm(
        initial_states[:, :2, None] - initial_states[:, :2, None].T, axis=1
    )
    d_ij += 1e2 * jnp.eye(n)
    min_d_ij = jnp.min(d_ij, axis=1)
    if jnp.any(min_d_ij > R):
        raise ValueError(
            "Initial states are too far apart, so it is impossible to satisfy"
            + " the connectivity constraint."
        )

    # Make a PRNG key (#sorandom)
    prng_key = jrandom.PRNGKey(args.seed)

    # # Initialize the trajectories randomly (these will be the DPs)
    # prng_key, traj_key = jrandom.split(prng_key)
    # traj_keys = jrandom.split(traj_key, num_chains)
    # init_robot_trajectories = jax.vmap(
    #     lambda key: arena.sample_random_multi_trajectory(
    #         key, initial_states[:, :2], T=T, fixed=False
    #     )
    # )(traj_keys)

    # Initialize trajectories to be straight lines plus some small noise
    prng_key, traj_key = jrandom.split(prng_key)

    waypoints = initial_states[:, :2]
    waypoints = waypoints[:, None, :] + jnp.zeros((n, T, 2))  # add time dimension
    waypoints = (
        jax.random.normal(traj_key, (num_chains, *waypoints.shape)) * 0.1
        + waypoints[None, :, :, :]
    )  # Add chain dimension
    waypoints = waypoints.at[:, :, :, 0].add(jnp.linspace(0, width - 2 * R, T))
    init_robot_trajectories = jax.vmap(
        lambda w: MultiAgentTrajectory([Trajectory2D(p) for p in w])
    )(waypoints)

    # Initialize the wind field randomly
    prng_key, wind_key = jrandom.split(prng_key)
    wind_keys = jrandom.split(wind_key, num_chains)
    wind = jax.vmap(KernelWindField)(wind_keys)

    # Initialize the connection strengths
    prng_key, conn_key = jrandom.split(prng_key)
    conn_keys = jrandom.split(conn_key, num_chains)
    conn = jax.vmap(sample_random_connection_strengths, in_axes=(0, None))(conn_keys, n)

    # Make stress test eps
    prng_key, wind_key = jrandom.split(prng_key)
    wind_keys = jrandom.split(wind_key, num_stress_test_cases)
    wind_stress_test = jax.vmap(KernelWindField)(wind_keys)

    # Initialize the connection strengths
    prng_key, conn_key = jrandom.split(prng_key)
    conn_keys = jrandom.split(conn_key, num_stress_test_cases)
    conn_stress_test = jax.vmap(sample_random_connection_strengths, in_axes=(0, None))(
        conn_keys, n
    )
    stress_test_eps = (wind_stress_test, conn_stress_test)

    # Define a prior over wind fields that says that the average wind thrust
    # should follow a gaussian distribution (maybe not super physical but just a start)
    N_test_points = 25
    test_X, test_Y = jnp.meshgrid(
        jnp.linspace(-width / 2.0, width / 2.0, N_test_points),
        jnp.linspace(-height / 2.0, height / 2.0, N_test_points),
    )

    def wind_logprior(wind):
        # For kernel wind field
        return jax.scipy.stats.norm.logpdf(wind.wind_kernels, 0.0, 1.0).sum()
        # For MLP wind field
        # wind_speeds = jax.vmap(jax.vmap(wind))(jnp.stack([test_X, test_Y], axis=-1))
        # mean_wind_speed = jnp.mean(wind_speeds)
        # return jax.scipy.stats.norm.logpdf(mean_wind_speed, 0.0, 1.0)

    def overall_logprior(ep):
        wind = ep[0]
        connection_strength = ep[1]
        return wind_logprior(wind) + connection_strength_prior_logprob(
            connection_strength
        )

    # Wrap the simulator function
    simulate_fn = lambda dp, ep: simulate(
        dp,
        initial_states,
        ep[0],
        ep[1],
        goal_com_position,
        max_wind_thrust=max_wind_thrust,
        duration=duration,
        dt=0.05,
        communication_range=R,
    )


    # Test runtimes
    dp = jax.tree_util.tree_map(lambda x: x[0], init_robot_trajectories)
    wind = KernelWindField(prng_key)
    conn = sample_random_connection_strengths(prng_key, n)
    ep = (
        wind,
        conn,
    )
    N_trials = 100
    from tqdm import tqdm

    # test without AD
    test_fn = lambda dp, ep: simulate_fn(dp, ep).potential
    test_fn_jit = jax.jit(test_fn)
    result = test_fn_jit(dp, ep)
    result.block_until_ready()
    no_ad_times = []
    for _ in tqdm(range(N_trials)):
        t0 = time.perf_counter()
        result = test_fn_jit(dp, ep)
        result.block_until_ready()
        no_ad_times.append(time.perf_counter() - t0)

    no_ad_times = jnp.array(no_ad_times)
    print(f"No AD: mean {no_ad_times.mean()} std {no_ad_times.std()}")

    # test with AD
    test_fn_jit = jax.jit(jax.value_and_grad(test_fn))
    ad_times = []
    result = test_fn_jit(dp, ep)[0]
    result.block_until_ready()
    for _ in tqdm(range(N_trials)):
        t0 = time.perf_counter()
        result = test_fn_jit(dp, ep)[0]
        result.block_until_ready()
        ad_times.append(time.perf_counter() - t0)

    ad_times = jnp.array(ad_times)
    print(f"W/ AD: mean {ad_times.mean()} std {ad_times.std()}")

    1 / 0

    if reinforce:
        init_sampler_fn = init_reinforce_sampler
        noise_scale = 0.1
        make_kernel_fn = lambda _1, logprob_fn, step_size, _2: make_reinforce_kernel(
            logprob_fn,
            step_size,
            perturbation_stddev=noise_scale,
            baseline_update_rate=0.5,
        )
    else:
        # This sampler yields either MALA, GD, or RMH depending on whether gradients and/or
        # stochasticity are enabled
        init_sampler_fn = lambda params, logprob_fn: init_mcmc_sampler(
            params,
            logprob_fn,
            True,  # TODO don't normalize gradients
            grad_clip,
        )
        make_kernel_fn = (
            lambda _, logprob_fn, step_size, stochasticity: make_mcmc_kernel(
                logprob_fn,
                step_size,
                use_gradients,
                stochasticity,
                grad_clip,
                True,  # TODO don't normalize gradients
                True,  # use metroplis-hastings
            )
        )

    # Run the prediction+mitigation process
    t_start = time.perf_counter()
    dps, eps, dp_logprobs, ep_logprobs = predict_and_mitigate_failure_modes(
        prng_key,
        init_robot_trajectories,
        (wind, conn),
        dp_logprior_fn=arena.multi_trajectory_prior_logprob,
        ep_logprior_fn=overall_logprior,
        ep_potential_fn=lambda dp, ep: -L
        * jax.nn.elu(failure_level - simulate_fn(dp, ep).potential),
        dp_potential_fn=lambda dp, ep: -L
        * jax.nn.elu(simulate_fn(dp, ep).potential - failure_level),
        # ep_potential_fn=lambda dp, ep: -L
        # * (failure_level - simulate_fn(dp, ep).potential),
        # dp_potential_fn=lambda dp, ep: -L
        # * (simulate_fn(dp, ep).potential - failure_level),
        init_sampler=init_sampler_fn,
        make_kernel=make_kernel_fn,
        num_rounds=num_rounds,
        num_mcmc_steps_per_round=num_mcmc_steps_per_round,
        dp_mcmc_step_size=dp_mcmc_step_size,
        ep_mcmc_step_size=ep_mcmc_step_size,
        use_stochasticity=use_stochasticity,
        repair=repair,
        predict=predict,
        quench_rounds=quench_rounds,
        quench_dps_only=False,  # quench both dps and eps
        stress_test_cases=stress_test_eps,
        tempering_schedule=tempering_schedule,
        failure_level=failure_level,
        potential_fn=lambda dp, ep: simulate_fn(dp, ep).potential,
        plotting_cb=plotting_cb,
        # test_every=10,
    )
    t_end = time.perf_counter()
    print(
        f"Ran {num_rounds:,} rounds with {num_chains} chains in {t_end - t_start:.2f} s"
    )

    # Select the seeker trajectory that performs best against all hider strategies
    # predicted before the final round (choose from all chains)
    if repair:
        most_likely_dps_idx = jnp.argmax(dp_logprobs[-1], axis=-1)
        final_dps = jtu.tree_map(lambda leaf: leaf[-1, most_likely_dps_idx], dps)
    else:
        # Just pick one dispatch arbitrarily
        final_dps = jtu.tree_map(lambda leaf: leaf[-1, 0], dps)
    # Evaluate this against all contingencies
    final_eps = jtu.tree_map(lambda leaf: leaf[-1], eps)
    result = jax.vmap(simulate_fn, in_axes=(None, 0))(final_dps, final_eps)
    # For later, save the index of the worst contingency
    worst_eps_idx = jnp.argmax(result.potential)

    # Plot the results
    fig = plt.figure(figsize=(32, 16), constrained_layout=True)
    axs = fig.subplot_mosaic(
        [
            ["arena", "arena"],
            ["arena", "arena"],
            ["trace", "connectivity"],
        ]
    )

    # Plot the arena
    axs["arena"].plot(
        [-width / 2, -width / 2, width / 2, width / 2, -width / 2],
        [-height / 2, height / 2, height / 2, -height / 2, -height / 2],
        "k-",
    )

    # Plot initial setup
    axs["arena"].scatter(
        initial_states[:, 0],
        initial_states[:, 1],
        color="k",
        marker="o",
        s=25,
        label="Initial positions",
    )

    axs["arena"].legend()
    axs["arena"].set_aspect("equal")

    # Plot planned trajectories
    t = jnp.linspace(0, 1, 100)
    for traj in final_dps.trajectories:
        pts = jax.vmap(traj)(t)
        axs["arena"].plot(pts[:, 0], pts[:, 1], "r-")
        axs["arena"].scatter(traj.p[:, 0], traj.p[:, 1], s=25, color="r", marker="x")

    # Plot endpoints for each trajectory
    axs["arena"].scatter(
        result.positions[:, -1, :, 0],
        result.positions[:, -1, :, 1],
        s=25,
        color="r",
        marker="o",
    )
    # for i in range(initial_states.shape[0]):
    #     max_U = result.potential.max()
    #     min_U = result.potential.min()
    #     for j in range(num_chains):
    #         # Make higher-potenial trajectories less transparent
    #         potential = result.potential[j]
    #         alpha = 0.3 + 0.7 * ((potential - min_U) / (1e-3 + max_U - min_U)).item()
    #         axs["arena"].plot(
    #             result.positions[j, :, i, 0],
    #             result.positions[j, :, i, 1],
    #             "r-",
    #             linewidth=1,
    #             alpha=alpha,
    #         )

    # Plot the worst wind speeds
    worst_wind = jax.tree_util.tree_map(lambda leaf: leaf[worst_eps_idx], final_eps[0])
    wind_speeds = jax.vmap(jax.vmap(worst_wind))(jnp.stack([test_X, test_Y], axis=-1))
    axs["arena"].quiver(
        test_X,
        test_Y,
        wind_speeds[:, :, 0],
        wind_speeds[:, :, 1],
        color="b",
        alpha=0.5,
        angles="xy",
        scale_units="xy",
        scale=10.0,
    )

    # Plot the chain convergence
    if predict:
        axs["trace"].plot(ep_logprobs)
        axs["trace"].set_ylabel("Log probability after contingency update")
    else:
        axs["trace"].plot(dp_logprobs)
        axs["trace"].set_ylabel("Log probability after repair")

    axs["trace"].set_xlabel("# Samples")

    # Plot the connectivity
    axs["connectivity"].plot(result.connectivity.T)
    axs["connectivity"].set_xlabel("Time")
    axs["connectivity"].set_ylabel("Connectivity")
    axs["connectivity"].set_title("Connectivity")

    experiment_type = "formation2d_grad_norm_netconn"
    if reinforce:
        alg_type = "reinforce"
    elif use_gradients and use_stochasticity:
        alg_type = "mala"
    elif use_gradients and not use_stochasticity:
        alg_type = "gd"
    elif not use_gradients and use_stochasticity:
        alg_type = "rmh"
    else:
        alg_type = "static"
    case_name = f"{n}"
    path = (
        f"results/{experiment_type}/{case_name}/L_{L:0.1e}/{T}_T/"
        f"{num_rounds * num_mcmc_steps_per_round}_samples/"
        f"{quench_rounds}_quench/{'tempered' if temper else 'no_temper'}"
        f"{num_chains}_chains/dp_{dp_mcmc_step_size:0.1e}/"
        f"ep_{ep_mcmc_step_size:0.1e}/"
        f"{'repair' if repair else 'no_repair'}/"
        f"{'predict' if predict else 'no_predict'}"
    )
    filename = os.path.join(path, alg_type)
    print(f"Saving results to: {filename}")
    os.makedirs(path, exist_ok=True)
    plt.savefig(filename + ".png")

    # Save the dispatch
    with open(filename + ".json", "w") as f:
        json.dump(
            {
                "trajs": {
                    "trajectories": [traj.p.tolist() for traj in final_dps.trajectories]
                },
                "best_eps_idx": worst_eps_idx,
                "time": t_end - t_start,
                "n": n,
                "T": T,
                "width": width,
                "height": height,
                "R": R,
                "max_wind_thrust": max_wind_thrust,
                "duration": duration,
                "dp_mcmc_step_size": dp_mcmc_step_size,
                "ep_mcmc_step_size": ep_mcmc_step_size,
                "num_rounds": num_rounds,
                "num_mcmc_steps_per_round": num_mcmc_steps_per_round,
                "num_chains": num_chains,
                "use_gradients": use_gradients,
                "use_stochasticity": use_stochasticity,
                "repair": repair,
                "predict": predict,
                "quench_rounds": quench_rounds,
                "tempering_schedule": tempering_schedule,
            },
            f,
            default=lambda x: x.tolist() if isinstance(x, Shaped[Array, "..."]) else x,
        )

    # Save the final exogenous parameters
    eqx.tree_serialise_leaves(filename + "_final_eps.eqx", final_eps)

    # Save the trace of design parameters
    with open(filename + "_dp_trace.json", "w") as f:
        json.dump(
            {
                "seeker_trajs": dps.trajectories,
                "num_rounds": num_rounds,
                "num_chains": num_chains,
                "n": n,
                "width": width,
                "height": height,
                "R": R,
                "duration": duration,
            },
            f,
            default=lambda x: x.tolist() if isinstance(x, Shaped[Array, "..."]) else x,
        )
