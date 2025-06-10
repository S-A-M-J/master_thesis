# @title Run RL simulation

import multiprocessing as mp
import os
import argparse
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

NVIDIA_ICD_CONFIG_PATH = '/usr/share/glvnd/egl_vendor.d/10_nvidia.json'
NVIDIA_ICD_CONFIG_DIR = os.path.dirname(NVIDIA_ICD_CONFIG_PATH)

if not os.path.exists(NVIDIA_ICD_CONFIG_DIR):
    os.makedirs(NVIDIA_ICD_CONFIG_DIR, exist_ok=True)

if not os.path.exists(NVIDIA_ICD_CONFIG_PATH):
    with open(NVIDIA_ICD_CONFIG_PATH, 'w') as f:
        f.write("""{
    "file_format_version" : "1.0.0",
    "ICD" : {
        "library_path" : "libEGL_nvidia.so.0"
    }
}
""")

#Configure MuJoCo to use the EGL rendering backend (requires GPU)
print('Setting environment variable to use GPU rendering:')
os.environ['MUJOCO_GL'] = 'egl'

# Tell XLA to use Triton GEMM, this    pip install numpy<2.0 improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

import jax
from jax import numpy as jp

# This helps with nan values being returned from the model while costing some perf. See github @brax for more info
# Other fix: jax.config.update('jax_enable_x64', True). However, this will slow down the training a lot
jax.config.update('jax_default_matmul_precision', 'high')
jax.config.update("jax_debug_nans", True)
# Check if GPU is available
gpu_available = jax.devices()[0].platform == 'gpu'
print(f"GPU available: {gpu_available}")

# Print GPU details
if gpu_available:
    gpu_device = jax.devices('gpu')[0]
    print(f"GPU device: {gpu_device}")
else:
    print("No GPU device found.")

import signal
import sys

# Function to handle the interrupt signal
def signal_handler(sig, frame):
    print('Program exited via keyboard interrupt')
    sys.exit(0)

# Register the signal handler for SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, signal_handler)

import json

try:
  import mujoco

  mujoco.MjModel.from_xml_string('<mujoco/>')
except Exception as e: 
  raise e from RuntimeError(
      'Could not import mujoco.'
  )

# Importing the necessary libraries
from datetime import datetime
import functools

# Run the code on the CPU rather than the GPU
# Normally the code runs on the GPU or any other accelerator that is available
#os.environ['JAX_PLATFORM_NAME'] = 'cpu'


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
from mujoco_playground.config import locomotion_params

from reachbot.getup import Getup as ReachbotGetup

from  tensorboardX import SummaryWriter

ENV_STR = 'Go1JoystickFlatTerrain'

# Store data from training
x_data, y_data, y_dataerr = [], [], []
times = [datetime.now()]

class JaxArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, jp.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
def save_video(frames, video_path, fps):
    import imageio
    imageio.mimsave(video_path, frames, fps=fps)


def trainModel(ppo_params_input:dict = None, on_sherlock:bool = False):

  # Original implementation
  # env = registry.load('cartpole_balance')
  # env_cfg = registry.get_default_config(cartpole_balance)
  # Create a custom environment
  from reachbot.integration import get_reachbot_getup_env
  from reachbot.getup import default_config as reachbot_config
  env_cfg = reachbot_config()
  env_cfg.episode_length = 1000
  env_cfg.reward_config.scales.shoulder_torque = -0.2 # Default is -0.1
  #env_cfg.action_scale = 1.0
  #env_cfg.reward_config.scales.posture = 0.0
  #env_cfg.reward_config.dof_vel = -0.1
  env = ReachbotGetup(config=env_cfg, task='rough_terrain_basic')

  ppo_params = locomotion_params.brax_ppo_config(ENV_STR)
  # Getting RL configuration parameters
  if ppo_params_input is None:
    ppo_training_params = dict(ppo_params)
    # Modify params for faster training
    ppo_training_params["num_timesteps"] = 50000000 # Reduce from 60000000
    ppo_training_params["episode_length"] = 3000 # Reduce from 1000
    #ppo_training_params["num_envs"] = 1024 # Reduce from 2048
    #ppo_training_params["batch_size"] = 512 # Reduce from 1024
    #ppo_training_params["num_minibatches"] = 16 # Reduce from 32
    #ppo_training_params["num_updates_per_batch"] = 8 # Reduce from 16
  else:
    ppo_training_params = ppo_params_input.copy()

  print("Training parameters:")

  for key, value in ppo_training_params.items():
      print(f"  {key}: {value}")
  #sac_params = dm_control_suite_params.brax_sac_config(ENV_STR)
  timesteps = []
  rewards = []
  total_rewards = []
  total_rewards_std = []

  datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  new_folder_path = "getup_"+datetime_str
  import os
  os.makedirs(new_folder_path, exist_ok=True)
  writer = SummaryWriter('runlogs_getup/'+new_folder_path)

  # Function to display the training progress
  def progress(num_steps, metrics):

    times.append(datetime.now())
    x_data.append(num_steps)
    y_data.append(metrics["eval/episode_reward"])
    y_dataerr.append(metrics["eval/episode_reward_std"])
    timesteps.append(num_steps)
    metrics["timesteps"] = num_steps
    rewards.append(metrics)
    total_rewards.append(metrics["eval/episode_reward"])
    total_rewards_std.append(metrics["eval/episode_reward_std"])
    for key, value in metrics.items():
        writer.add_scalar(key, value, num_steps)
        writer.flush()
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

  # Save the frames as a video file

  def policy_params_fn(current_step, make_policy, params):
    del make_policy  # Unused.
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(params)
    checkpoint_path = os.path.join(new_folder_path, 'checkpoints')
    path = os.path.join(checkpoint_path, f"{current_step}")
    abs_path = os.path.abspath(path)
    orbax_checkpointer.save(abs_path, params, force=True, save_args=save_args)

  # Training the model
  print("Training the model...")
  train_fn = functools.partial(
      ppo.train, **dict(ppo_training_params),
      network_factory=network_factory,
      progress_fn=progress,
      policy_params_fn=policy_params_fn
  )

  # Function to control the trained agents actions in the environment
  # Params: Stores the weights of the trained model
  # Metrics: Contains information about the training process such as performance over time
  from mujoco_playground import wrapper
  import os
  # Run training
  make_inference_fn, params, metrics = train_fn(
      environment=env,
      wrap_env_fn=wrapper.wrap_for_brax_training,
  )
  print(f"time to jit: {times[1] - times[0]}")
  print(f"time to train: {times[-1] - times[1]}")

  if not on_sherlock:
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))


    rng = jax.random.PRNGKey(0)
    rollout = []
    n_episodes = 1

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
    frames = env.render(rollout[::render_every], camera='track', width=1920, height=1080)
    video_path = os.path.join(new_folder_path, 'posttraining.mp4')
    fps = 1.0 / env.dt
    ctx = mp.get_context("spawn")
    p = ctx.Process(target=save_video, args=(frames, video_path, fps))
    p.start()
    p.join()

  # Save configs as json in results
  configs = {
      "env_cfg": env_cfg.to_dict(),
      "ppo_params": ppo_training_params
  }
  config_path = os.path.join(new_folder_path, 'config.json')
  # Replace Infinity with a large number or null
  def replace_infinity(obj):
      if isinstance(obj, dict):
          return {k: replace_infinity(v) for k, v in obj.items()}
      elif isinstance(obj, list):
          return [replace_infinity(v) for v in obj]
      elif isinstance(obj, float) and obj == float('inf'):
          return 1e308
      return obj

  # Update the configs before saving
  configs = replace_infinity(configs)

  # Save environment configuration
  with open(config_path, "w", encoding="utf-8") as fp:
    json.dump(configs, fp, indent=4)


  # Store the results in a file
  results_path = os.path.join(new_folder_path, 'results.txt')
  with open(results_path, 'w') as f:
    for i in range(len(total_rewards)):
      f.write(f"step: {timesteps[i]}, reward: {total_rewards[i]}, reward_std: {total_rewards_std[i]}\n")

  # Save the rewards as JSON
  rewards_path = os.path.join(new_folder_path, 'rewards.json')
  with open(rewards_path, 'w') as fp:
      json.dump(rewards, fp, indent=4, cls=JaxArrayEncoder)

  params_path = os.path.join(new_folder_path, 'params')
  model.save_params(params_path, params)


# Main function
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Train a model with optional Sherlock flag.')
  parser.add_argument('--on_sherlock', action='store_true', help='Flag to indicate if running on Sherlock')
  args = parser.parse_args()

  ppo_params = locomotion_params.brax_ppo_config(ENV_STR)
  ppo_training_params = dict(ppo_params)
  for i in range(1):
    # Modify params for faster training
    ppo_training_params["num_timesteps"] = 20_000_000 # Reduce from 60000000
    ppo_training_params["episode_length"] = 1000 + i * 1000 # Reduce from 1000
    #ppo_training_params["learning_rate"] = 0.0003 # Reduce from 0.0003
    #ppo_training_params["num_envs"] = 2048 # Reduce from 2048
    #ppo_training_params["batch_size"] = 256 # Reduce from 1024
    #ppo_training_params["num_minibatches"] = 16 # Reduce from 32
    #ppo_training_params["num_updates_per_batch"] = 8 # Reduce from 16

  trainModel(ppo_training_params, on_sherlock=args.on_sherlock)