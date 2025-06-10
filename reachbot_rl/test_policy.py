import os
import pickle
import functools
from datetime import datetime

import jax
from jax import numpy as jp
import imageio

# Set up MuJoCo GPU rendering if needed
os.environ['MUJOCO_GL'] = 'egl'

# Import your environment creation functions and configurations
from reachbot.integration import get_reachbot_getup_env
from reachbot.getup import default_config as reachbot_config

# Import the PPO inference function generator from your training library
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo

# ---------------------------
# 1. Load the Saved Policy
# ---------------------------
params_path = '/Users/sam/Library/CloudStorage/GoogleDrive-samuel.jahn@gmx.de/My Drive/reachbot_rl/results-reachbot-g_2025-02-11_23-34-53/params.pkl'
with open(params_path, 'rb') as f:
    params = pickle.load(f)
print("Loaded trained policy parameters.")

# ---------------------------
# 2. Recreate the Environment
# ---------------------------
env = get_reachbot_getup_env()        # Your custom Reachbot environment.
env_cfg = reachbot_config()           # The configuration used during training.

# JIT the reset and step functions for speed.
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

# ---------------------------
# 3. Build the Inference Function
# ---------------------------

from brax.training.agents.ppo import networks as ppo_networks

# In training, the network factory was passed (e.g., as network_factory).
# Here we assume you want to use the same function:
network_factory = ppo_networks.make_ppo_networks

# Optionally, you may have used an observations normalization function.
# For simplicity, we use an identity function:
normalize = lambda x, y: x

# Build the policy network with the same parameters as during training.
ppo_network = network_factory(
    obs_shape, env.action_size, preprocess_observations_fn=normalize
)

make_policy = ppo_networks.make_inference_fn(ppo_network)
inference_fn = jax.jit(lambda obs, key: make_policy(params, obs, key))
print("Inference function created.")

# ---------------------------
# 4. Roll Out the Policy
# ---------------------------
rng = jax.random.PRNGKey(0)
state = jit_reset(rng)
rollout = [state]

# Roll out for the episode length defined in your configuration.
for i in range(env_cfg.episode_length):
    act_rng, rng = jax.random.split(rng)
    # The inference function takes observations and a PRNG key, and returns:
    #   (action, auxiliary_info)
    action, _ = inference_fn(state.obs, act_rng)
    state = jit_step(state, action)
    rollout.append(state)

print("Rollout complete.")

# ---------------------------
# 5. Render and Save the Rollout
# ---------------------------
# Render the rollout frames.
frames = env.render(rollout)
# Create a folder to save the video if desired.
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
results_folder = f"results-rollout-{timestamp}"
os.makedirs(results_folder, exist_ok=True)

# Save the video.
video_path = os.path.join(results_folder, 'trained_policy_rollout.mp4')
fps = 1.0 / env.dt  # Use the simulation time step to set FPS.
imageio.mimsave(video_path, frames, fps=fps)
print(f"Rollout video saved to {video_path}")
