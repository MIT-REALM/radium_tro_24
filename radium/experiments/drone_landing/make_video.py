"""A script for analyzing the results of the repair experiments."""
import json
import os
import subprocess

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
import seaborn as sns

from radium.experiments.drone_landing.predict_and_mitigate import simulate
from radium.experiments.drone_landing.train_drone_agent import make_drone_landing_env
from radium.systems.components.sensing.vision.render import CameraExtrinsics
from radium.systems.components.sensing.vision.util import look_at
from radium.systems.drone_landing.env import DroneState
from radium.systems.drone_landing.policy import DroneLandingPolicy

SEEDS = [0, 1, 2, 3]
moving_obstacles = True
SAVE_DIR = "tmp/0vids/drone_dynamic"
DATA_SOURCES = {
    # "prediction": {
    #     "path_prefix": f"results/drone_dynamic/predict/L_1.0e+01/25_samples_25x1/5_chains/0_quench/dp_1.0e-02/ep_1.0e-02/grad_norm/grad_clip_inf/mala_tempered_40+0.1",
    # },
    "repair": {
        "path_prefix": f"results/drone_dynamic/predict_repair_1.0/L_1.0e+01/25_samples_5x5/5_chains/0_quench/dp_1.0e-03/ep_1.0e-03/grad_norm/grad_clip_inf/mala_tempered_40+0.1",
    },
}


def load_data_sources_from_json():
    """Load data sources from a JSON file."""
    loaded_data = {}

    # Load the results from each data source
    for alg in DATA_SOURCES:
        loaded_data[alg] = []
        for seed in SEEDS:
            with open(DATA_SOURCES[alg]["path_prefix"] + f"_{seed}" + ".json") as f:
                data = json.load(f)

                new_data = {
                    "eps_trace": jax.tree_util.tree_map(
                        lambda x: jnp.array(x),
                        data["eps_trace"],
                        is_leaf=lambda x: isinstance(x, list),
                    ),
                    "final_eps": jax.tree_util.tree_map(
                        lambda x: jnp.array(x),
                        data["eps"],
                        is_leaf=lambda x: isinstance(x, list),
                    ),
                    "T": data["T"],
                    "num_trees": data["num_trees"],
                    "failure_level": 7.5,  # data["failure_level"],
                }

            # Also load in the design parameters
            image_shape = (32, 32)
            dummy_policy = DroneLandingPolicy(jax.random.PRNGKey(0), image_shape)
            full_policy = eqx.tree_deserialise_leaves(
                DATA_SOURCES[alg]["path_prefix"] + f"_{seed}" + ".eqx", dummy_policy
            )
            dp, static_policy = eqx.partition(full_policy, eqx.is_array)
            new_data["dp"] = dp
            new_data["static_policy"] = static_policy

            loaded_data[alg].append(new_data)

    return loaded_data


def get_trajectories(loaded_data, alg, seed):
    """Get the trajectories from running the given policy on all eps."""
    image_shape = (32, 32)
    env = make_drone_landing_env(
        image_shape, loaded_data[alg][seed]["num_trees"], moving_obstacles
    )
    simulate_fn = lambda ep: simulate(
        env,
        loaded_data[alg][seed]["dp"],
        ep,
        loaded_data[alg][seed]["static_policy"],
        loaded_data[alg][seed]["T"],
    )

    if "tree_velocities" in result["final_eps"]:
        tree_velocities = result["final_eps"]["tree_velocities"]
    else:
        tree_velocities = jnp.zeros_like(result["final_eps"]["tree_locations"])

    eps = DroneState(
        drone_state=result["final_eps"]["drone_state"],
        tree_locations=result["final_eps"]["tree_locations"],
        tree_velocities=tree_velocities,
        wind_speed=result["final_eps"]["wind_speed"],
    )

    return jax.vmap(simulate_fn)(eps)


def render_frame(env, drone_state, tree_locations, wind_speed):
    """Render one frame of the highway environment."""
    # Render the scene
    drone_pos = drone_state[:3]
    camera_origin = drone_pos + jnp.array([-3, -3, 3])
    extrinsics = CameraExtrinsics(
        camera_origin=camera_origin,
        camera_R_to_world=look_at(
            camera_origin=camera_origin,
            target=drone_pos,
            up=jnp.array([0, 0, 1.0]),
        ),
    )

    _, color_image = env._scene.render_rgbd(
        env._camera_intrinsics,
        extrinsics,
        wind_speed,
        tree_locations,
        sharpness=env._render_sharpness,
        shading_light_direction=jnp.array([-1.0, -1.0, 5.0]),
        drone_state=drone_state,
    )

    return color_image


if __name__ == "__main__":
    # Activate seaborn styling
    sns.set_theme(context="paper", style="whitegrid")

    # Load the data
    data = load_data_sources_from_json()

    # Get trajectories for each seed for each algorithm
    for alg in data:
        for seed, result in enumerate(data[alg]):
            print(f"Simulating {alg} seed {seed}")
            # Get the trajectories by simulating
            trajectories = get_trajectories(data, alg, seed)

            # Render the trajectories
            print("Rendering...")
            image_shape = (256, 256)  # Higher res for videos
            env = make_drone_landing_env(
                image_shape, result["num_trees"], moving_obstacles
            )
            render_fn = lambda drone_state, tree_locs, wind_speed: render_frame(
                env, drone_state, tree_locs, wind_speed
            )
            for chain_idx in range(trajectories.drone_traj.shape[0]):
                print(f"\tChain {chain_idx}...")
                # Upsample the trajectories
                t_upsample = 3
                drone_traj = trajectories.drone_traj[chain_idx]
                tree_traj = trajectories.tree_traj[chain_idx]
                T = drone_traj.shape[0]
                t_original = jnp.linspace(0.0, T * env._dt, T)
                t_upsampled = jnp.linspace(0.0, T * env._dt, T * t_upsample)
                upsample_fn = lambda traj: jnp.interp(t_upsampled, t_original, traj)

                drone_traj_upsampled = jax.vmap(upsample_fn, in_axes=(1))(
                    drone_traj
                ).transpose(1, 0)
                tree_traj_upsampled = jax.vmap(
                    jax.vmap(upsample_fn, in_axes=(1)), in_axes=(1)
                )(tree_traj).transpose(2, 0, 1)

                # Render
                rendered_frames = jax.vmap(render_fn, in_axes=(0, 0, None))(
                    drone_traj_upsampled,
                    tree_traj_upsampled,
                    trajectories.wind_speed[chain_idx],
                )

                # Save
                img_dir = os.path.join(
                    SAVE_DIR, alg, f"seed_{seed}", f"chain_{chain_idx}"
                )
                os.makedirs(img_dir, exist_ok=True)
                for i, img in enumerate(rendered_frames):
                    matplotlib.image.imsave(
                        os.path.join(img_dir, f"img_{i}.png"), img.transpose(1, 0, 2)
                    )

                # Convert to video
                subprocess.Popen(
                    [
                        "ffmpeg",
                        "-y",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-framerate",
                        f"{10 * t_upsample}",
                        "-i",
                        "img_%d.png",
                        "-c:v",
                        "libx264",
                        "-r",
                        "30",
                        "-vf",
                        "scale=1024x1024",  # :flags=neighbor",
                        "0_out.mp4",
                    ],
                    cwd=img_dir,
                )
