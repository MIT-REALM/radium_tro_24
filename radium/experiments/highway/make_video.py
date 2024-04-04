"""A script for analyzing the results of the repair experiments."""
import json
import os
import subprocess

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
import seaborn as sns

from radium.experiments.highway.predict_and_mitigate import (
    sample_non_ego_actions,
    simulate,
)
from radium.experiments.highway.train_highway_agent import make_highway_env
from radium.systems.components.sensing.vision.render import CameraExtrinsics
from radium.systems.components.sensing.vision.util import look_at
from radium.systems.highway.driving_policy import DrivingPolicy
from radium.systems.highway.highway_env import HighwayState

lr = 1e-2
lr = f"{lr:.1e}"
SEEDS = [0, 1, 2, 3]
SAVE_DIR = "tmp/highway"
DATA_SOURCES = {
    "prediction": {
        "path_prefix": f"results/highway/predict/noise_5.0e-01/L_1.0e+01/25_samples/5_chains/0_quench/dp_{lr}/ep_{lr}/mala_20tempered+0.1",
    },
    "repair": {
        "path_prefix": f"results/highway/predict_repair_1.0/noise_5.0e-01/L_1.0e+01/25_samples/5_chains/0_quench/dp_{lr}/ep_{lr}/mala_20tempered+0.1",
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
                    "final_eps": jnp.array(data["action_trajectory"]),
                    "eps_trace": jax.tree_util.tree_map(
                        lambda x: jnp.array(x),
                        data["action_trajectory_trace"],
                        is_leaf=lambda x: isinstance(x, list),
                    ),
                    "noise_scale": data["noise_scale"],
                    "initial_state": jax.tree_util.tree_map(
                        lambda x: jnp.array(x),
                        data["initial_state"],
                        is_leaf=lambda x: isinstance(x, list),
                    ),
                }
                new_data["T"] = new_data["eps_trace"].shape[2]

            # Also load in the design parameters
            image_shape = (32, 32)
            dummy_policy = DrivingPolicy(jax.random.PRNGKey(0), image_shape)
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
    env = make_highway_env(image_shape)
    env._collision_penalty = 10.0
    initial_state = HighwayState(
        ego_state=loaded_data[alg][seed]["initial_state"]["ego_state"],
        non_ego_states=loaded_data[alg][seed]["initial_state"]["non_ego_states"],
        shading_light_direction=loaded_data[alg][seed]["initial_state"][
            "shading_light_direction"
        ],
        non_ego_colors=loaded_data[alg][seed]["initial_state"]["non_ego_colors"],
    )

    simulate_fn = lambda ep: simulate(
        env,
        loaded_data[alg][seed]["dp"],
        initial_state,
        ep,
        loaded_data[alg][seed]["static_policy"],
        loaded_data[alg][seed]["T"],
    )

    return jax.vmap(simulate_fn)(loaded_data[alg][seed]["final_eps"])


def render_frame(env, initial_state, ego_state, non_ego_state):
    """Render one frame of the highway environment."""
    # Render the depth image as seen by the ego agent
    ego_x, ego_y, _, _ = ego_state
    ego_vehicle_origin = jnp.array([ego_x, ego_y, 0.75 * env._highway_scene.car.height])
    camera_origin = ego_vehicle_origin + jnp.array([-10, -10, 10])
    extrinsics = CameraExtrinsics(
        camera_origin=camera_origin,
        camera_R_to_world=look_at(
            camera_origin=camera_origin,
            target=ego_vehicle_origin,
            up=jnp.array([0, 0, 1.0]),
        ),
    )

    # Add the ego state to the non ego states so we can render all cars
    car_states = jnp.concatenate([ego_state.reshape(1, -1), non_ego_state], axis=0)
    # Add a nice blue for the ego vehicle and dark grey for cars
    colors = (
        jnp.array(
            [
                [52.0, 152, 219],
                [44.0, 62, 80],
                [44.0, 62, 80],
            ]
        )
        / 255.0
    )
    _, color_image = env._highway_scene.render_rgbd(
        env._camera_intrinsics,
        extrinsics,
        car_states[:, :3],  # trim out speed; not needed for rendering
        max_dist=env._max_render_dist,
        sharpness=env._render_sharpness,
        shading_light_direction=initial_state.shading_light_direction,
        car_colors=colors,
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
            env = make_highway_env(image_shape)
            initial_state = HighwayState(
                ego_state=data[alg][seed]["initial_state"]["ego_state"],
                non_ego_states=data[alg][seed]["initial_state"]["non_ego_states"],
                shading_light_direction=data[alg][seed]["initial_state"][
                    "shading_light_direction"
                ],
                non_ego_colors=data[alg][seed]["initial_state"]["non_ego_colors"],
            )
            render_fn = lambda ego_state, non_ego_state: render_frame(
                env, initial_state, ego_state, non_ego_state
            )
            for chain_idx in range(trajectories.ego_trajectory.shape[0]):
                print(f"\tChain {chain_idx}...")
                # Upsample the trajectories
                t_upsample = 3
                ego_trajectory = trajectories.ego_trajectory[chain_idx]
                non_ego_trajectory = trajectories.non_ego_trajectory[chain_idx]
                T = ego_trajectory.shape[0]
                t_original = jnp.linspace(0.0, T * env._dt, T)
                t_upsampled = jnp.linspace(0.0, T * env._dt, T * t_upsample)
                upsample_fn = lambda traj: jnp.interp(t_upsampled, t_original, traj)

                ego_traj_upsampled = jax.vmap(upsample_fn, in_axes=(1))(
                    ego_trajectory
                ).transpose(1, 0)
                non_ego_traj_upsampled = jax.vmap(
                    jax.vmap(upsample_fn, in_axes=(1)), in_axes=(1)
                )(non_ego_trajectory).transpose(2, 0, 1)

                # Render
                rendered_frames = jax.vmap(render_fn)(
                    ego_traj_upsampled,
                    non_ego_traj_upsampled,
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
