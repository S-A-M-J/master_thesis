
import mujoco
import os
import sys
import imageio

# Add the project root to the Python path to resolve module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import jax
from etils import epath
import functools


from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import model
from jax import numpy as jp

#jax.config.update("jax_debug_nans", True)
jax.config.update("jax_debug_infs", True)

#from reachbot.getup import default_config as reachbot_getup_config
#from reachbot.getup import Getup as ReachbotGetup
#from reachbot.joystick import Joystick as ReachbotJoystick
#from reachbot.joystick import default_config as reachbot_joystick_config

import jax
from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_params

from ml_collections import config_dict

script_dir = os.path.dirname(os.path.abspath(__file__))

relative_ckpt_path = "cave_exploration/logs/cave_exploration-2025-07-15_09-13-47"

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
elif 'cave_exploration' in relative_ckpt_path:
    print('Rendering cave exploration task result')
    from tasks.cave_exploration.cave_exploration import default_config as cave_exploration_config
    env_cfg = cave_exploration_config()
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
elif 'cave_exploration' in relative_ckpt_path:
    from tasks.cave_exploration.cave_exploration import CaveExplore
    env = CaveExplore(config=env_cfg)
    print(env.mj_model.actuator_ctrlrange)
    print(env.mj_model.actuator_ctrllimited)
    print(env.mjx_model.actuator_ctrlrange)
    print(env.mjx_model.actuator_ctrllimited)



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
n_episodes = 3
episode_length = 5000


# Rollout policy and record simulation

print(f"Running rollout for {n_episodes} episode(s) with {episode_length} steps each...")
for episode in range(n_episodes):
    episode_reward = 0.0
    print(f"Episode {episode + 1}/{n_episodes}")
    state = jit_reset(rng)
    rollout.append(state)
    for i in range(episode_length):
        if i % 500 == 0:
            print(f"  Step {i}/{episode_length}")
            
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = jit_inference_fn(state.obs, act_rng)
        
        # Check for numerical issues
        if jp.any(jp.isinf(ctrl)) or jp.any(jp.isnan(ctrl)):
            print(f"Numerical issue detected in control at step {i}. Stopping rollout.")
            break
            
        state = jit_step(state, ctrl)

        # Accumulate reward for this episode
        episode_reward += float(state.reward)
        
        if state.done:
            print(f"Episode {episode + 1} ended at step {i} with reward: {episode_reward:.3f}")
            break
            
        rollout.append(state)

    print(f"Rollout completed with {len(rollout)} states")

    # Render video
    print("Rendering video...")
    render_every = 1  # Render every frame
    width = 1920      # Full HD width
    height = 1080     # Full HD height

    frames = env.render(rollout[::render_every], camera='track_global', width=width, height=height)
    print(f"Rendered {len(frames)} frames")

    # Save video
    video_path = os.path.join(relative_ckpt_path, f'posttraining_{episode_reward:.2f}.mp4')
    fps = 1.0 / env.dt

    print(f"Saving video to {video_path} at {fps} FPS...")
    imageio.mimsave(video_path, frames, fps=fps)
    print(f"Video saved successfully to {video_path}")