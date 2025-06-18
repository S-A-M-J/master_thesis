# @title Run RL simulation with joystick control

import multiprocessing as mp
import argparse
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


# Tell XLA to use Triton GEMM, this    pip install numpy<2.0 improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

import jax
from jax import numpy as jp



# This helps with nan values being returned from the model while costing some perf. See github @brax for more info
# Other fix: jax.config.update('jax_enable_x64', True). However, this will slow down the training a lot
#jax.config.update('jax_default_matmul_precision', 'high')
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

import mujoco

import json
# Importing the necessary libraries
from datetime import datetime
import functools
# Run the code on the CPU rather than the GPU
# Normally the code runs on the GPU or any other accelerator that is available
#os.environ['JAX_PLATFORM_NAME'] = 'cpu'

from brax.io import model
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
import mujoco
from flax.training import orbax_utils
from orbax import checkpoint as ocp
from mujoco_playground.config import locomotion_params

from tasks.cave_exploration.cave_exploration import CaveExplore
from tasks.common.randomize import domain_randomize as reachbot_randomize
from mujoco_playground import registry

from tensorboardX import SummaryWriter

from pathlib import Path



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

  # Create log directory for training run
  datetime_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  logdir = os.path.join(os.path.dirname(__file__), "logs/cave_exploration-"+datetime_str)
  os.makedirs(logdir, exist_ok=True) 
  


  from tasks.cave_exploration.cave_exploration import default_config as reachbot_config
  env_cfg = reachbot_config()
  
  env_cfg.reward_config.scales.orientation = 0.0 # Disable orientation reward
  env_cfg.reward_config.scales.lin_vel_z = 0.0 # Disable velocity reward
  env_cfg.reward_config.scales.ang_vel_xy = 0.0 # Disable xy velocity reward
  env_cfg.reward_config.scales.feet_slip = 0.0 # Disable z velocity reward
  env_cfg.reward_config.scales.boom_extension_speed = 0.0 # Disable feet height reward
  env_cfg.reward_config.scales.stand_still = 0.0 # Disable feet height reward
  env_cfg.reward_config.scales.torques = 0.0 # Disable LIDAR reward
  env_cfg.reward_config.scales.action_rate = 0.0 # Disable LIDAR reward
  env_cfg.reward_config.scales.energy = 0.0 # Enable
  env_cfg.reward_config.scales.distance_to_target = 2.0 # Disable LIDAR reward

  env = CaveExplore(config=env_cfg, lidar_num_horizontal_rays=20, lidar_max_range=15.0, lidar_horizontal_angle_range=jp.pi * 2, lidar_vertical_angle_range=jp.pi / 6) # Updated LIDAR params for 3D

  ppo_params = locomotion_params.brax_ppo_config(ENV_STR)
  # Getting RL configuration parameters
  if ppo_params_input is None:
    ppo_training_params = dict(ppo_params)
    # Modify params for faster training
    #ppo_training_params["num_timesteps"] = 50000000 # Reduce from 60000000
    #ppo_training_params["episode_length"] = 3000 # Reduce from 1000
    #ppo_training_params["num_envs"] = 1024 # Reduce from 2048
    #ppo_training_params["batch_size"] = 512 # Reduce from 1024
    #ppo_training_params["num_minibatches"] = 16 # Reduce from 32
    #ppo_training_params["num_updates_per_batch"] = 8 # Reduce from 16
  else:
    ppo_training_params = ppo_params_input.copy()

  env_cfg.episode_length = ppo_training_params["episode_length"]

  print("Training parameters:")

  for key, value in ppo_training_params.items():
      print(f"  {key}: {value}")
  timesteps = []
  rewards = []
  total_rewards = []
  total_rewards_std = []

  writer = SummaryWriter(logdir=logdir)


  # Function to display the training progress
  def progress(num_steps, metrics):
    times.append(datetime.now())
    timesteps.append(num_steps)
    total_rewards.append(metrics["eval/episode_reward"])
    total_rewards_std.append(metrics["eval/episode_reward_std"])
    for key, value in metrics.items():
        writer.add_scalar(key, value, num_steps)
        writer.flush()
    metrics["timesteps"] = num_steps
    metrics["time"] = datetime.now().timestamp()
    rewards.append(metrics)
    percent_complete = (num_steps / ppo_training_params["num_timesteps"]) * 100
    print(f"step: {num_steps}/{ppo_training_params['num_timesteps']} ({percent_complete:.1f}%), reward: {total_rewards[-1]:.3f} +/- {total_rewards_std[-1]:.3f}")

  
  # Getting the network factory
  network_factory = ppo_networks.make_ppo_networks(observation_size=env.observation_size, action_size=env.action_size)
  if "network_factory" in ppo_params:
    if "network_factory" in ppo_training_params:
      del ppo_training_params["network_factory"]
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        **ppo_params.network_factory
    )


  def policy_params_fn(current_step, make_policy, params):
    del make_policy  # Unused.
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(params)
    checkpoint_path = os.path.join(logdir, 'checkpoints')
    path = os.path.join(checkpoint_path, f"{current_step}")
    abs_path = os.path.abspath(path)
    orbax_checkpointer.save(abs_path, params, force=True, save_args=save_args)

  #randomizer = registry.get_domain_randomizer(ENV_STR)
  randomizer = reachbot_randomize

  # Training the model
  print("Training the model...")
  train_fn = functools.partial(
      ppo.train, **dict(ppo_training_params),
      network_factory=network_factory,
      progress_fn=progress,
      policy_params_fn=policy_params_fn,
      #randomization_fn=randomizer,
  )

  # Function to control the trained agents actions in the environment
  # Params: Stores the weights of the trained model
  # Metrics: Contains information about the training process such as performance over time
  from mujoco_playground import wrapper
  # Run training
  make_inference_fn, params, metrics = train_fn(
      environment=env,
      wrap_env_fn=wrapper.wrap_for_brax_training,
  )
  print(f"time to jit: {times[1] - times[0]}")
  print(f"time to train: {times[-1] - times[1]}")

  # Save configs as json in results
  configs = {
      "env_cfg": env_cfg.to_dict(),
      "ppo_params": ppo_training_params
  }
  config_path = os.path.join(logdir, 'config.json')
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
  results_path = os.path.join(logdir, 'results.txt')
  with open(results_path, 'w') as f:
    for i in range(len(total_rewards)):
      f.write(f"step: {timesteps[i]}, reward: {total_rewards[i]}, reward_std: {total_rewards_std[i]}\n")
    f.write(f"Time to jit: {times[1] - times[0]}\n")
    f.write(f"Time to train: {times[-1] - times[1]}\n")

  # Save the rewards as JSON
  rewards_path = os.path.join(logdir, 'rewards.json')
  def nest_flat_dict(flat_dict):
    """Converts a dictionary with '/' in keys to a nested dictionary."""
    nested_dict = {}
    for key, value in flat_dict.items():
        parts = key.split('/')
        d = nested_dict
        for i, part in enumerate(parts):
            is_last_part = (i == len(parts) - 1)
            if is_last_part:
                # If a dictionary already exists here, we're setting the 'value' for that group.
                if isinstance(d.get(part), dict):
                    d[part]['value'] = value
                else:
                    d[part] = value
            else:
                # If the path item is not a dict, convert it to one to allow nesting.
                if not isinstance(d.get(part), dict):
                    d[part] = {'value': d[part]} if part in d else {}
                d = d[part]
    return nested_dict
  
  nested_rewards = [nest_flat_dict(r) for r in rewards]
  with open(rewards_path, 'w') as fp:
      json.dump(nested_rewards, fp, indent=4, cls=JaxArrayEncoder)

    
  params_path = os.path.join(logdir, 'params')
  model.save_params(params_path, params)

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
    for i in range(1200):
      act_rng, rng = jax.random.split(rng)
      ctrl, _ = jit_inference_fn(state.obs, act_rng)
      state = jit_step(state, ctrl)
      rollout.append(state)

  render_every = 1
  frames = env.render(rollout[::render_every], camera='track_global', width=1920, height=1080)
  video_path = os.path.join(logdir, 'posttraining.mp4')
  fps = 1.0 / env.dt
  ctx = mp.get_context("spawn")
  p = ctx.Process(target=save_video, args=(frames, video_path, fps))
  p.start()
  p.join()



# Main function
if __name__ == '__main__':
  ppo_params = locomotion_params.brax_ppo_config(ENV_STR)
  ppo_training_params = dict(ppo_params)
  
  # Modify params for faster training
  ppo_training_params["num_timesteps"] = 1_000_000 # Reduce from 60000000
  ppo_training_params["episode_length"] = 2000 # Reduce from 1000
  ppo_training_params["num_envs"] = 2048 # Reduce from 2048
  ppo_training_params["batch_size"] = 256 # Reduce from 1024
  ppo_training_params["num_minibatches"] = 16 # Reduce from 32
  ppo_training_params["num_updates_per_batch"] = 8 # Reduce from 16
  ppo_training_params["unroll_length"] = 50
  ppo_training_params["entropy_cost"] = 0.02
  trainModel(ppo_training_params)