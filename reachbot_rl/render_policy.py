
import mujoco
import os

import jax
from etils import epath
import functools


from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import model
from jax import numpy as jp

from reachbot.getup import default_config as reachbot_getup_config
from reachbot.getup import Getup as ReachbotGetup
from reachbot.joystick import Joystick as ReachbotJoystick
from reachbot.joystick import default_config as reachbot_joystick_config

import jax
from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_params

from ml_collections import config_dict

script_dir = os.path.dirname(os.path.abspath(__file__))

relative_ckpt_path = "joystick_2025-06-17_15-23-05"

ckpt_path = os.path.join(script_dir, relative_ckpt_path)



# Get the configuration for the environment
import json
with open(os.path.join(ckpt_path, 'config.json'), 'r') as f:
    loaded_config = json.load(f)
# Get the default environment configuration (a ConfigDict).
if 'joystick' in relative_ckpt_path:
    print('Rendering joystick task result')
    env_cfg = reachbot_joystick_config()
elif 'getup' in relative_ckpt_path:
    print('Rendering getup task result')
    env_cfg = reachbot_getup_config()
else:
    print('Unknown task')
    exit()
# Convert the loaded dict to a ConfigDict
json_env_cfg = config_dict.ConfigDict(loaded_config['env_cfg'])
# Update the default config with the values from the JSON.
env_cfg.update(json_env_cfg)

if 'joystick' in relative_ckpt_path:
    env = ReachbotJoystick(config=env_cfg, task="rough_terrain_basic")
elif 'getup' in relative_ckpt_path:
    env = ReachbotGetup(config=env_cfg, task="flat_terrain_basic")

# Get the PPO configuration
ppo_params = locomotion_params.brax_ppo_config('Go1JoystickFlatTerrain')
ppo_training_params = dict(ppo_params)
ppo_training_params['num_timesteps'] = 0

# Getting the network configuration for the policy
if "network_factory" in ppo_params:
    if "network_factory" in ppo_training_params:
        del ppo_training_params["network_factory"]
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        **ppo_params.network_factory
    )


# Building the training function based on the ppo parameters
train_fn = functools.partial(
    ppo.train, **dict(ppo_training_params),
    network_factory=network_factory,
)

# Building the inference function
make_inference_fn, params, _ = train_fn(
    environment=env,
    num_timesteps=0,
    wrap_env_fn=wrapper.wrap_for_brax_training
)

# Load the trained model
params = model.load_params(os.path.join(ckpt_path,'params'))

# Jit everything
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)
jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))

# Reset the environment
rng = jax.random.PRNGKey(3)
rollout = []
n_episodes = 1
episode_length = 1200

# Set the command to be executed by the agent
x_vel = 0.2
y_vel = 0.2
yaw_vel = 0.0
command = jp.array([x_vel, y_vel, yaw_vel])

# Rollout policy and record simulation
for _ in range(n_episodes):
    state = jit_reset(rng)
    state.info["command"] = command
    for i in range(episode_length):
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = jit_inference_fn(state.obs, act_rng)
        state = jit_step(state, ctrl)
        state.info["command"] = command
        rollout.append(state)

    render_every = 1
    width = 1920  # Full HD width (default is usually 640)
    height = 1080  # Full HD height (default is usually 480)
    frames = env.render(rollout[::render_every], camera='track', width=width, height=height)
    video_path = os.path.join(ckpt_path, 'posttraining3.mp4')
    fps = 1.0 / env.dt

import imageio
imageio.mimsave(video_path, frames, fps=fps)