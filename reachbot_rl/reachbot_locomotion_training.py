# @title Run RL simulation

import signal
import sys

# Function to handle the interrupt signal
def signal_handler(sig, frame):
    print('Program exited via keyboard interrupt')
    sys.exit(0)

# Register the signal handler for SIGINT (Ctrl+C)
signal.signal(signal.SIGINT, signal_handler)

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
import os
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

# Importing pickle for load/save functionality of policy function
import pickle


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
from etils import epath
from flax import struct
from flax.training import orbax_utils
from IPython.display import HTML, clear_output, display
import matplotlib.pyplot as plt
import jax
from jax import numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np
from orbax import checkpoint as ocp

# Import Reachbot XML file
from mujoco_playground import registry
REACHBOT_XML = ""


ENV_STR = 'Go1JoystickFlatTerrain'

# Store data from training
x_data, y_data, y_dataerr = [], [], []
times = [datetime.now()]

# Function to display the training progress
def progress(num_steps, metrics):
  clear_output(wait=True)

  times.append(datetime.now())
  x_data.append(num_steps)
  y_data.append(metrics["eval/episode_reward"])
  y_dataerr.append(metrics["eval/episode_reward_std"])

  print(f"step: {num_steps}, reward: {y_data[-1]:.3f} +/- {y_dataerr[-1]:.3f}")

  #plt.xlim([0, ppo_training_params["num_timesteps"] * 1.25])
  #plt.ylim([0, 1100])
  #plt.xlabel("# environment steps")
  #plt.ylabel("reward per episode")
  #plt.title(f"y={y_data[-1]:.3f}")
  #plt.errorbar(x_data, y_data, yerr=y_dataerr, color="blue")

  #display(plt.gcf())


def trainModel(load_params=False):

  # Original implementation
  # env = registry.load('cartpole_balance')
  # env_cfg = registry.get_default_config(cartpole_balance)
  # Create a custom environment
  from reachbot.integration import get_reachbot_env
  from reachbot.joystick import default_config as reachbot_config
  env = get_reachbot_env()
  env_cfg = reachbot_config
  print(reachbot_config)
  sys.exit()


  # Precompiling to make the environment faster
  # jax.jit is a decorator that compiles the function to make it faster
  jit_reset = jax.jit(env.reset) # env.reset is a function that resets the environment
  jit_step = jax.jit(env.step)

  # Randomly initializing the state of the environment
  state = jit_reset(jax.random.PRNGKey(0))
  # Set first state in rollout to be the initial state. Rollout will contain the states of the environment
  rollout = [state]

  # Getting RL configuration parameters
  from mujoco_playground.config import dm_control_suite_params
  ppo_params = dm_control_suite_params.brax_ppo_config(ENV_STR)
  print(ppo_params)
  ppo_training_params = dict(ppo_params)
  print(ppo_training_params)
  sac_params = dm_control_suite_params.brax_sac_config(ENV_STR)


  # Getting the network factory
  network_factory = ppo_networks.make_ppo_networks
  if "network_factory" in ppo_params:
    del ppo_training_params["network_factory"]
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        **ppo_params.network_factory
    )

  # Modify params for faster training
  ppo_training_params["num_timesteps"] = 10000000 # Reduce from 60000000
  ppo_training_params["episode_length"] = 500 # Reduce from 1000
  ppo_training_params["num_envs"] = 1024 # Reduce from 2048
  ppo_training_params["batch_size"] = 512 # Reduce from 1024
  ppo_training_params["num_minibatches"] = 16 # Reduce from 32
  ppo_training_params["num_updates_per_batch"] = 8 # Reduce from 16


  # Training the model
  train_fn = functools.partial(
      ppo.train, **dict(ppo_training_params),
      network_factory=network_factory,
      progress_fn=progress
  )


  # Load the policy function and parameters if available
  if os.path.exists('policy.pkl') and load_params:
      with open('policy.pkl', 'rb') as f:
          make_inference_fn, params = pickle.load(f)
  elif load_params: 
     print ("Policy file not found")
     sys.exit()

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


  # Save the policy function and parameters
  #with open('policy.pkl', 'wb') as f:
  #    pickle.dump((make_inference_fn, params), f)

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


  # Display the trained agent in the environment
  #render_every = 1
  #frames = env.render(rollout[::render_every])
  #rewards = [s.reward for s in rollout]
  # media.show_video(frames, fps=1.0 / env.dt / render_every)

trainModel(load_params=False)