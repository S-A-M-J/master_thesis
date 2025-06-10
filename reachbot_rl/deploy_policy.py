import os
import jax
import functools
import numpy as np

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.io import model

from reachbot.getup import default_config as reachbot_config
from reachbot.getup import Getup as ReachbotGetup

import mujoco
from mujoco_playground import wrapper
from mujoco_playground.config import locomotion_params

# Set paths etc.
script_dir = os.path.dirname(os.path.abspath(__file__))
policy_path = "joystick_2025-02-28_08-28-26"

# Get the configuration for the environment
env_cfg = reachbot_config()
env_cfg.Kp = 200
env_cfg.Kd = 10
env = ReachbotGetup(config=env_cfg, task="flat_terrain_basic")

# Get the PPO configuration
ppo_params = locomotion_params.brax_ppo_config('Go1JoystickFlatTerrain')
ppo_training_params = dict(ppo_params)
ppo_training_params['num_timesteps'] = 0  # indicate no further training

# Get the network configuration for the policy
if "network_factory" in ppo_params:
    if "network_factory" in ppo_training_params:
        del ppo_training_params["network_factory"]
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        **ppo_params.network_factory
    )

# Build the training function (used here only to get the inference function)
train_fn = functools.partial(
    ppo.train, **dict(ppo_training_params),
    network_factory=network_factory,
)

# Build the inference function and initial parameters (unused here)
make_inference_fn, params, _ = train_fn(
    environment=env,
    num_timesteps=0,
    wrap_env_fn=wrapper.wrap_for_brax_training,
    seed=1,
)

# Load the trained model parameters from disk
params = model.load_params(os.path.join(policy_path, 'params'))

# JIT compile Brax functions
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)
jit_inference_fn = jax.jit(make_inference_fn(params, deterministic=True))

def update_sim_from_state(model, data, state):
    # Copy Brax state (positions and velocities) into the MuJoCo simulation.
    # Adjust these assignments if your environment's API differs.
    data.data.qpos[:] = np.asarray(state.qp)
    data.data.qvel[:] = np.asarray(state.qv)
    # Run forward computation to update derived quantities.
    mujoco.mj_forward(model, data)

# Reset the Brax environment
rng = jax.random.PRNGKey(0)
state = jit_reset(rng)

# Retrieve the underlying MuJoCo model and create its MjData.
mj_model = env._mj_model
data = mujoco.MjData(mj_model)

# Launch the viewer using the MjData object.
viewer = mujoco.viewer.launch_passive(mj_model, data)

print("Starting live deployment. Press ESC in the viewer window to exit.")

# Live control loop
while viewer.is_running():
    # Generate a new random key for stochasticity in the policy
    act_rng, rng = jax.random.split(rng)
    # Compute the control action from the current observation.
    ctrl, _ = jit_inference_fn(state.obs, act_rng)
    # Step the Brax environment forward using the control
    state = jit_step(state, ctrl)
    # Update the MuJoCo simulation state from the Brax state.
    update_sim_from_state(mj_model, data, state)
    # Take a simulation step in MuJoCo (updates dynamics using qpos/qvel).
    mujoco.mj_step(mj_model, data)
    # Synchronize the viewer with the new simulation state.
    viewer.sync()
