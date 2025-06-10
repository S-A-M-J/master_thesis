import os
from datetime import datetime
from etils import epath
import functools

from typing import Any, Dict, Sequence, Tuple, Union
from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.base import State as PipelineState
from brax.envs.base import Env, PipelineEnv, State
from brax.io import html, mjcf, model
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import networks as sac_networks
from brax.training.agents.sac import train as sac
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np
from flax import struct
from flax.training import orbax_utils
from orbax import checkpoint as ocp

from reachbot.getup import default_config as reachbot_config
from reachbot.getup import Getup as ReachbotGetup
import jax
from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_params


ckpt_path = "/Users/sam/Documents/Master Thesis/reachbot_rl/results/getup_2025-02-24_02-40-31"
# Path to the XML file

env_cfg = reachbot_config()
#env_cfg.energy_termination_threshold = 400  # lower energy termination threshold
#env_cfg.reward_config.energy = -0.003  # non-zero negative `energy` reward
#env_cfg.reward_config.dof_acc = -2.5e-7  # non-zero negative `dof_acc` reward
env_cfg.Kp = 400
env_cfg.Kd = 10

ppo_params = locomotion_params.brax_ppo_config('Go1JoystickFlatTerrain')
ppo_training_params = dict(ppo_params)
ppo_training_params['num_timesteps'] = 10000

FINETUNE_PATH = epath.Path(ckpt_path)
latest_ckpts = list(FINETUNE_PATH.glob("*"))
latest_ckpts = [ckpt for ckpt in latest_ckpts if ckpt.is_dir()]
latest_ckpts.sort(key=lambda x: int(x.name))
latest_ckpt = latest_ckpts[-1]
restore_checkpoint_path = latest_ckpt

times = [datetime.now()]

env = ReachbotGetup(config=env_cfg)

x_data, y_data, y_dataerr = [], [], []
timesteps = []
rewards = []
total_rewards = []
total_rewards_std = []

def progress(num_steps, metrics):
    return
    times.append(datetime.now())
    x_data.append(num_steps)
    y_data.append(metrics["eval/episode_reward"])
    y_dataerr.append(metrics["eval/episode_reward_std"])
    timesteps.append(num_steps)
    metrics["timesteps"] = num_steps
    rewards.append(metrics)
    total_rewards.append(metrics["eval/episode_reward"])
    total_rewards_std.append(metrics["eval/episode_reward_std"])
    print(f"step: {num_steps}, reward: {y_data[-1]:.3f} +/- {y_dataerr[-1]:.3f}")

# Getting the network factory
network_factory = ppo_networks.make_ppo_networks
if "network_factory" in ppo_params:
    if "network_factory" in ppo_training_params:
        del ppo_training_params["network_factory"]
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        **ppo_params.network_factory
    )

datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# Save the frames as a video file
# Create a new folder with datetime_str as name
new_folder_path = "getup_"+datetime_str
#os.makedirs(new_folder_path, exist_ok=True)


def policy_params_fn(current_step, make_policy, params):
    return
    del make_policy  # Unused.
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(params)
    path = os.path.join(new_folder_path, f"{current_step}")
    abs_path = os.path.abspath(path)
    orbax_checkpointer.save(abs_path, params, force=True, save_args=save_args)

train_fn = functools.partial(
    ppo.train, **dict(ppo_training_params),
    network_factory=network_factory,
    progress_fn=progress,
    policy_params_fn=policy_params_fn
)

make_inference_fn, params, metrics = train_fn(
    environment=env,
    eval_env=env,
    wrap_env_fn=wrapper.wrap_for_brax_training,
    restore_checkpoint_path=restore_checkpoint_path,  # restore from the checkpoint!
    seed=1,
)
print(f"time to jit: {times[1] - times[0]}")
print(f"time to train: {times[-1] - times[1]}")

jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)
jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))

rng = jax.random.PRNGKey(0)
rollout = []
n_episodes = 1200

# Rollout policy and record simulation
for _ in range(n_episodes):
    state = jit_reset(rng)
    rollout.append(state)
    for i in range(env_cfg.episode_length):
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_step(state, ctrl)
        rollout.append(state)

    render_every = 1
    frames = env.render(rollout[::render_every], camera='track')
    video_path = os.path.join(new_folder_path, 'posttraining.mp4')
    fps = 1.0 / env.dt

import imageio
imageio.mimsave(video_path, frames, fps=fps)