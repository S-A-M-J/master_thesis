# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Modified by Samuel Jahn for the purpose of the Reachbot project
# Contact: samjahn@stanford.edu
# ==============================================================================
"""Joystick task for Reachbot."""

from typing import Any, Dict, Optional, Union

import sys
import os

import mujoco
import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx, MjModel
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import collision
from mujoco_playground._src import mjx_env
from ..common import reachbot_constants as consts

from models.model_loader import ReachbotModelType
from .environment.env_loader import CaveBatchLoader
import random

def renormalize_quat(qpos):
    """Renormalize the free-joint quaternion to prevent numerical drift.
    
    Args:
        qpos: Position array where indices 3:7 contain the free-joint quaternion
        
    Returns:
        qpos with renormalized quaternion at indices 3:7
    """
    # assumes indices 3:7 are the free‑joint quaternion
    quat = qpos[3:7]
    quat = quat / jp.linalg.norm(quat)
    return qpos.at[3:7].set(quat)

def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      cave_batch_size=1,  # Number of caves to load in a batch
      ctrl_dt=0.02,
      sim_dt=0.004,
      episode_length=10000,
      Kp_rot=25.0,
      Kd_rot=1.0,
      Kp_pri=100.0,
      Kd_pri=20.0,
      action_repeat=1,
      action_scale=0.2,
      history_len=1,
      soft_joint_pos_limit_factor=0.95,
      noise_config=config_dict.create(
          level=1.0,  # Set to 0.0 to disable noise.
          scales=config_dict.create(
              joint_pos=0.03,
              joint_vel=1.5,
              gyro=0.2,
              gravity=0.05,
              linvel=0.1,
          ),
      ),
      reward_config = config_dict.create(
        scales=config_dict.create(
            
            # Base movement rewards
            orientation=-0.2,           # Penalty scale for orientation deviation (based on upwards vector sensor). Default is -5.0
            distance_to_target=5.0,     # Reward scale for distance to target (reduced from 10.0)
            vel_to_target=10.0,          # Reward scale for velocity towards target (reduced from 100.0)
            exploration_rate=1.0,       # Reward scale for exploration rate
            lin_vel_z=-0.1,            # Penalty scale for linear velocity in z-direction
            ang_vel_xy=-0.1,           # Penalty scale for angular velocity in xy-plane
            
            # Other rewards
            dof_pos_limits=-1.0,        # Penalty scale for degree of freedom position limits. Default is -1.0
            pose=0.0,                   # Reward scale for maintaining a specific pose. Default is 0.5
            feet_slip=-0.01,           # Penalty scale for feet slipping
            
            # Termination and stand-still penalties
            termination=-1.0,           # Penalty scale for termination conditions. Default is -1.0
            
            # Regularization terms
            torques=-0.0002,            # Penalty scale for torques applied. Default is -0.0002
            action_rate=-0.01,          # Penalty scale for action rate changes. Default is -0.01
            energy=-0.001,              # Penalty scale for energy consumption. Default is -0.001
            
        ),
      ),
      pert_config=config_dict.create(
          enable=False,
          velocity_kick=[0.0, 3.0],
          kick_durations=[0.05, 0.2],
          kick_wait_times=[1.0, 3.0],
      ),
      stickiness_config=config_dict.create(
          enable=False,  # Enable stickiness forces
          stickiness_force=50.0,  # Force applied when stickiness is activated
          min_activation_threshold=0.5,  # Threshold for activating stickiness
          deactivation_threshold=0.3,  # Threshold for deactivating stickiness (hysteresis)
      ),
      lidar_config=config_dict.create(
          num_horizontal_rays=10,  # Number of horizontal rays
          max_range=10.0,  # Maximum range of LIDAR
          horizontal_angle_range=jp.pi * 2,  # Horizontal angle range in radians
          num_vertical_rays=3,  # Number of vertical rays
          vertical_angle_range=jp.pi / 6,  # Vertical angle range in radians
      ),
  )


class CaveExplore(mjx_env.MjxEnv):
  """Explore the cave environment."""

  def __init__(
      self,
      model: ReachbotModelType = ReachbotModelType.BASIC,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    # Replace default config with provided config and overrides
    self._config = config_dict.ConfigDict(config)
    if config_overrides:
      self._config.update(config_overrides)
    self._envs = CaveBatchLoader(self._config.cave_batch_size, self._config, model)
    self._model = model
    
    # LIDAR parameters
    self._lidar_num_horizontal_rays = self._config.lidar_config.num_horizontal_rays
    self._lidar_num_vertical_rays = self._config.lidar_config.num_vertical_rays
    self._lidar_max_range = self._config.lidar_config.max_range
    self._lidar_horizontal_angle_range = self._config.lidar_config.horizontal_angle_range
    self._lidar_vertical_angle_range = self._config.lidar_config.vertical_angle_range
    
    # Select initial random environment
    self._select_random_env()
    
    # Call parent class __init__
    super().__init__(config, config_overrides)
    
    # Call _post_init to initialize model-dependent attributes
    self._post_init()
    

  def _post_init(self) -> None:
    self._init_q = jp.array(self._active_env["mj_model"].keyframe("home").qpos)
    self._default_pose = jp.array(self._active_env["mj_model"].keyframe("home").qpos[7:])

    # Note: First joint is freejoint.
    self._lowers, self._uppers = self._active_env["mj_model"].jnt_range[1:].T
    self._soft_lowers = self._lowers * self._config.soft_joint_pos_limit_factor
    self._soft_uppers = self._uppers * self._config.soft_joint_pos_limit_factor

    self._torso_body_id = self._active_env["mj_model"].body(consts.ROOT_BODY).id
    self._torso_mass = self._active_env["mj_model"].body_subtreemass[self._torso_body_id]

    self._no_movement_duration = 5.0  # seconds
    self._no_movement_threshold = 0.1  # meters
    self._no_movement_steps = jp.array(self._no_movement_duration / self.sim_dt, dtype=jp.int32)

    self._feet_site_id = np.array(
        [self._active_env["mj_model"].site(name).id for name in consts.FEET_SITES]
    )
    # Collect all geoms whose names start with "cave_wall" as floor geoms
    self._floor_geom_ids = np.array([
      i for i in range(self._active_env["mj_model"].ngeom)
      if mujoco.mj_id2name(self._active_env["mj_model"], mujoco.mjtObj.mjOBJ_GEOM, i).startswith("cave_wall")
    ])
    print("Floor boxes detected:", len(self._floor_geom_ids))
    self._feet_geom_id = np.array(
        [self._active_env["mj_model"].geom(name).id for name in consts.FEET_GEOMS]
    )

    foot_linvel_sensor_adr = []
    for site in consts.FEET_SITES:
      sensor_id = self._active_env["mj_model"].sensor(f"{site}_global_linvel").id
      sensor_adr = self._active_env["mj_model"].sensor_adr[sensor_id]
      sensor_dim = self._active_env["mj_model"].sensor_dim[sensor_id]
      foot_linvel_sensor_adr.append(
          list(range(sensor_adr, sensor_adr + sensor_dim))
      )
    self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)

    # Initialize IMU site ID which is needed for get_gravity method
    self._imu_site_id = self._active_env["mj_model"].site("imu").id

    # Pre-compute boom geom mappings for JAX compatibility and efficient contact filtering
    # Optimizations implemented:
    # 1. Use jp.isin() instead of individual binary searches for vectorized lookups
    # 2. Sort arrays once during initialization for efficient operations
    # 3. Minimize array operations and intermediate allocations
    # 4. Early exit conditions to avoid unnecessary computations
    # 5. Vectorized force calculations instead of per-contact processing
    mj_model = self._active_env["mj_model"]
    
    # Find all boom end geoms and create JAX-compatible arrays
    boom_geom_ids = []
    boom_nums = []
    boom_body_ids = []
    
    for i in range(mj_model.ngeom):
        geom_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if geom_name and "boomEnd" in geom_name:
            boom_geom_ids.append(i)
            # Extract boom number
            try:
                boom_num = int(geom_name.split('boomEnd')[1])
            except (ValueError, IndexError):
                boom_num = 0
            boom_nums.append(boom_num)
            boom_body_ids.append(mj_model.geom_bodyid[i])
    
    # Convert to JAX arrays for JIT compatibility
    self._boom_geom_ids = jp.array(boom_geom_ids)
    self._boom_nums = jp.array(boom_nums)
    self._boom_body_ids = jp.array(boom_body_ids)
    
    # Sort boom arrays for efficient binary search
    if len(boom_geom_ids) > 0:
        sorted_indices = jp.argsort(self._boom_geom_ids)
        self._boom_geom_ids = self._boom_geom_ids[sorted_indices]
        self._boom_nums = self._boom_nums[sorted_indices] 
        self._boom_body_ids = self._boom_body_ids[sorted_indices]
    
    # Also need floor/wall geom IDs - sort these too for efficiency
    floor_geom_ids = []
    for i in range(mj_model.ngeom):
        geom_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if geom_name and ("cave_wall" in geom_name or "floor" in geom_name):
            floor_geom_ids.append(i)
    
    self._floor_geom_ids = jp.array(sorted(floor_geom_ids)) if floor_geom_ids else jp.array([])
    
    print(f"Found {len(boom_geom_ids)} boom end geoms")
    print(f"Found {len(floor_geom_ids)} floor/wall geoms")

    print("CaveExplore task initialized with model:", self._model)
    print("CaveExplore task action space:", self.action_size)
    print("CaveExplore task observation space:", self.observation_size)

    # Track maximum contacts for debugging
    self._max_contacts = 12

  def _select_random_env(self) -> None:
    env_idx = random.randint(0, self._config.cave_batch_size - 1)
    self._active_env = self._envs.envs[env_idx]

  def get_upvector(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self._active_env["mj_model"], data, consts.UPVECTOR_SENSOR)

  def get_gravity(self, data: mjx.Data) -> jax.Array:
    return data.site_xmat[self._imu_site_id].T @ jp.array([0, 0, -1])

  def get_global_linvel(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(
        self._active_env["mj_model"], data, consts.GLOBAL_LINVEL_SENSOR
    )

  def get_global_angvel(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(
        self._active_env["mj_model"], data, consts.GLOBAL_ANGVEL_SENSOR
    )

  def get_local_linvel(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(
        self._active_env["mj_model"], data, consts.LOCAL_LINVEL_SENSOR
    )

  def get_accelerometer(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(
        self._active_env["mj_model"], data, consts.ACCELEROMETER_SENSOR
    )

  def get_gyro(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self._active_env["mj_model"], data, consts.GYRO_SENSOR)

  def get_lidar_pos(self, data: mjx.Data) -> jax.Array: # Added for LIDAR
    return mjx_env.get_sensor_data(self._active_env["mj_model"], data, consts.HEAD_POS_SENSOR) # Added for LIDAR

  def get_feet_pos(self, data: mjx.Data) -> jax.Array:
    return jp.vstack([
        mjx_env.get_sensor_data(self._active_env["mj_model"], data, sensor_name)
        for sensor_name in consts.FEET_POS_SENSOR
    ])

  def _qpos_to_motor_ctrl(self, qpos: jax.Array) -> jax.Array:
    """Convert joint angles to control input format"""
    return qpos[7:7+self.mjx_model.nu]
    
  def reset(self, rng: jax.Array) -> mjx_env.State:
    """Resets the environment to a random state."""
    self._select_random_env()
    qpos = jp.array(self._active_env["initial_qpos"].copy())
    
    qvel = jp.zeros(self.mjx_model.nv)

    # Randomize the initial z axis orientation of the robot
    rng, key = jax.random.split(rng)
    yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
    quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
    new_quat = math.quat_mul(qpos[3:7], quat)

    qpos = qpos.at[3:7].set(new_quat)  # Set the new orientation

    #  Randomize the initial joint velocities
    rng, key = jax.random.split(rng)
    qvel = jp.zeros(self.mjx_model.nv)
    ctrl = jp.zeros(self.mjx_model.nu)
    data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=ctrl)

    rng, key1, key2, key3 = jax.random.split(rng, 4)
    time_until_next_pert = jax.random.uniform(
        key1,
        minval=self._config.pert_config.kick_wait_times[0],
        maxval=self._config.pert_config.kick_wait_times[1],
    )
    steps_until_next_pert = jp.round(time_until_next_pert / self.dt).astype(
        jp.int32
    )
    pert_duration_seconds = jax.random.uniform(
        key2,
        minval=self._config.pert_config.kick_durations[0],
        maxval=self._config.pert_config.kick_durations[1],
    )
    pert_duration_steps = jp.round(pert_duration_seconds / self.dt).astype(
        jp.int32
    )
    pert_mag = jax.random.uniform(
        key3,
        minval=self._config.pert_config.velocity_kick[0],
        maxval=self._config.pert_config.velocity_kick[1],
    )

    rng, key1, key2 = jax.random.split(rng, 3)
  # Compute and store initial distance to target in info dict (JAX-safe)
    init_dist_to_target = jp.linalg.norm(qpos[0:3] - jp.array(self._active_env["target_pos"]))

    pos_history = jp.tile(qpos[0:3], (self._no_movement_steps, 1))
  
    info = {
        "init_dist_to_target": init_dist_to_target,
        "rng": rng,
        "target_pos": self._active_env["target_pos"],
        "last_act": jp.zeros(self.action_size),  # Changed from self.mjx_model.nu to self.action_size
        "last_last_act": jp.zeros(self.action_size),  # Changed from self.mjx_model.nu to self.action_size
        "last_contact": jp.zeros(len(self._feet_geom_id), dtype=bool),
        "steps_until_next_pert": steps_until_next_pert,
        "pert_duration_seconds": pert_duration_seconds,
        "pert_duration": pert_duration_steps,
        "steps_since_last_pert": 0,
        "pert_steps": 0,
        "pert_dir": jp.zeros(3),
        "pert_mag": pert_mag,
        "last_pos": qpos[0:3],
        "pos_history": pos_history,  
        "steps": 0,
        # Stickiness state tracking for each boom
        "boom_stickiness_active": jp.zeros(4, dtype=bool),  # Track which booms are currently stuck
        "last_stickiness_ctrl": jp.zeros(4),  # Track previous stickiness control values
    }

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())
    metrics["swing_peak"] = jp.zeros(())

    obs = self._get_obs(data, info)
    reward, done = jp.zeros(2)
    
    return mjx_env.State(data, obs, reward, done, metrics, info)


  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    """Applies action to the environment and returns the new state."""
    if self._config.pert_config.enable:
      state = self._maybe_apply_perturbation(state)
    
    state_formatted = self._qpos_to_motor_ctrl(state.data.qpos)
    actuator_action = action[:self.mjx_model.nu]
    stickiness_action = action[self.mjx_model.nu:] 
    motor_targets = state_formatted + actuator_action * self._config.action_scale

    # Handle contacts properly - in MJX, contacts is a struct with arrays
    contacts = state.data.contact
    
    # Check if there are any contacts first
    if state.data.ncon > 0:
        
        if state.data.ncon > self._max_contacts:
            self._max_contacts = state.data.ncon
            print(f"Max contacts updated to {self._max_contacts} based on current step.")
        # Ensure contacts.geom is a 2D array with shape (ncon, 2
        # Get contact geom pairs - contacts.geom is shape (ncon, 2)
        contact_geom1 = contacts.geom[:, 0]  # First geom in each contact
        contact_geom2 = contacts.geom[:, 1]  # Second geom in each contact
        
        # Find contacts between feet and floor
        feet_floor_contacts1 = jp.isin(contact_geom1, self._feet_geom_id) & jp.isin(contact_geom2, self._floor_geom_ids)
        feet_floor_contacts2 = jp.isin(contact_geom2, self._feet_geom_id) & jp.isin(contact_geom1, self._floor_geom_ids)
        
        # Get the foot geom IDs that are in contact
        foot_contacts1 = jp.where(feet_floor_contacts1, contact_geom1, -1)
        foot_contacts2 = jp.where(feet_floor_contacts2, contact_geom2, -1)
        
        # Combine both cases and filter out -1 values
        all_foot_contacts = jp.concatenate([foot_contacts1, foot_contacts2])
        
        # Create contact array for each foot by checking if any contact matches each foot geom ID
        # We don't need to filter out -1 values, just check if any valid contact matches
        contact = jp.array([jp.any((all_foot_contacts == geom_id) & (all_foot_contacts >= 0)) for geom_id in self._feet_geom_id])
    else:
        # No contacts detected
        contact = jp.zeros(len(self._feet_geom_id), dtype=bool)

    if self._config.stickiness_config.enable:
      # Apply stickiness force if enabled - pass contact information to avoid recomputing
      xfrc_applied = self._apply_stickiness_forces(state, stickiness_action, contacts)
      
      # Update stickiness state in info
      updated_info = self._update_stickiness_state(state.info, stickiness_action)
      state = state.replace(info=updated_info)
      
      state = state.replace(data=state.data.replace(xfrc_applied=xfrc_applied))
    # ... rest of method ...

    data = mjx_env.step(
        self.mjx_model, state.data, motor_targets, self.n_substeps
    )
    
    
    # Renormalize quaternion to prevent numerical drift
    data = data.replace(qpos=renormalize_quat(data.qpos))

    contact_filt = contact | state.info["last_contact"]
    p_f = data.site_xpos[self._feet_site_id]
    p_fz = p_f[..., -1]

    # Update position history for termination condition
    pos_history = jp.roll(state.info["pos_history"], shift=-1, axis=0)
    pos_history = pos_history.at[-1].set(data.qpos[0:3])
    state.info["pos_history"] = pos_history

    obs = self._get_obs(data, state.info)
    done = self._get_termination(data, state.info)

    rewards = self._get_reward(
        data, action, state.info, state.metrics, done, contact
    )
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    state.info["last_pos"] = data.qpos[0:3]
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v

    done = done.astype(reward.dtype)
    state = state.replace(data=data, obs=obs, reward=reward, done=done)
    state.info["steps"] += 1
    return state
  
  def _prefilter_boom_wall_contacts(self, contacts, ncon: int) -> tuple:
    """Pre-filter contacts to only include boom-wall interactions with optimized lookups."""
    if ncon == 0:
        return jp.array([], dtype=bool), jp.array([], dtype=jp.int32), jp.array([], dtype=jp.int32), jp.array([]).reshape(0, 3)
    
    # Handle both contact field access patterns - MJX uses different structures
    if hasattr(contacts, 'geom'):
        # Standard MJX contact structure
        geom1_ids = contacts.geom[:ncon, 0]
        geom2_ids = contacts.geom[:ncon, 1]
        contact_positions = contacts.pos[:ncon]
    else:
        # Alternative contact structure
        geom1_ids = contacts.geom1[:ncon]
        geom2_ids = contacts.geom2[:ncon]
        contact_positions = contacts.pos[:ncon]
    
    # Efficient vectorized lookup using isin (faster than individual binary searches)
    is_boom1 = jp.isin(geom1_ids, self._boom_geom_ids)
    is_boom2 = jp.isin(geom2_ids, self._boom_geom_ids)
    is_wall1 = jp.isin(geom1_ids, self._floor_geom_ids)
    is_wall2 = jp.isin(geom2_ids, self._floor_geom_ids)
    
    # Find boom-wall contacts: (boom1 & wall2) OR (boom2 & wall1)
    boom_wall_mask = (is_boom1 & is_wall2) | (is_boom2 & is_wall1)
    
    # Extract boom and wall geom IDs for valid contacts using more efficient operations
    boom_geom_ids = jp.where(is_boom1 & is_wall2, geom1_ids,
                            jp.where(is_boom2 & is_wall1, geom2_ids, -1))
    wall_geom_ids = jp.where(is_boom1 & is_wall2, geom2_ids,
                            jp.where(is_boom2 & is_wall1, geom1_ids, -1))
    
    return boom_wall_mask, boom_geom_ids, wall_geom_ids, contact_positions

  def _update_stickiness_state(self, info: Dict[str, Any], stickiness_ctrl: jax.Array) -> Dict[str, Any]:
    """Update boom stickiness activation state with hysteresis."""
    last_ctrl = info["last_stickiness_ctrl"]
    currently_active = info["boom_stickiness_active"]
    
    # Hysteresis thresholds
    activate_threshold = self._config.stickiness_config.min_activation_threshold
    deactivate_threshold = self._config.stickiness_config.deactivation_threshold
    
    # Activation: not currently active AND control signal crosses activation threshold
    newly_activated = (~currently_active) & (last_ctrl <= activate_threshold) & (stickiness_ctrl > activate_threshold)
    
    # Deactivation: currently active AND control signal drops below deactivation threshold  
    newly_deactivated = currently_active & (stickiness_ctrl < deactivate_threshold)
    
    # Update state
    new_active_state = (currently_active | newly_activated) & (~newly_deactivated)
    
    # Update info dict
    info["boom_stickiness_active"] = new_active_state
    info["last_stickiness_ctrl"] = stickiness_ctrl
    
    return info

  def _apply_stickiness_forces(self, state: mjx_env.State, stickiness_ctrl: jax.Array, contacts) -> jax.Array:
    """Apply stickiness forces with efficient pre-filtering and state management."""
    xfrc_applied = jp.zeros((self.mjx_model.nbody, 6))
    
    if state.data.ncon == 0:
        return xfrc_applied
    
    # Update stickiness state with hysteresis - get updated info dict
    updated_info = self._update_stickiness_state(state.info, stickiness_ctrl)
    
    # Pre-filter to only boom-wall contacts
    boom_wall_mask, boom_geom_ids, wall_geom_ids, contact_positions = self._prefilter_boom_wall_contacts(
        contacts, state.data.ncon
    )
    
    # Early exit if no boom-wall contacts
    if jp.sum(boom_wall_mask) == 0:
        return xfrc_applied
    
    # Filter to only valid boom-wall contacts
    valid_boom_geoms = boom_geom_ids[boom_wall_mask]
    valid_contact_pos = contact_positions[boom_wall_mask]
    
    # Create efficient lookup maps for boom geoms to avoid repeated searches
    # Use vectorized searchsorted for all valid boom geoms at once
    boom_indices = jp.searchsorted(self._boom_geom_ids, valid_boom_geoms)
    
    # Validate indices and ensure they point to correct geoms
    valid_idx_mask = (boom_indices < len(self._boom_geom_ids)) & \
                     (self._boom_geom_ids[boom_indices] == valid_boom_geoms)
    
    # Get boom numbers and body IDs for valid contacts
    boom_nums = jp.where(valid_idx_mask, self._boom_nums[boom_indices], -1)
    boom_body_ids = jp.where(valid_idx_mask, self._boom_body_ids[boom_indices], -1)
    
    # Check which booms are in active sticky state using updated info
    boom_active_mask = jp.zeros_like(boom_nums, dtype=bool)
    for i in range(4):  # Assuming 4 booms max
        boom_i_mask = (boom_nums == i) & valid_idx_mask
        boom_active_mask = jp.where(
            boom_i_mask,
            updated_info["boom_stickiness_active"][i],
            boom_active_mask
        )
    
    # Final filter: valid boom contacts that are actually active
    final_valid_mask = valid_idx_mask & boom_active_mask
    
    # Early exit if no active boom contacts
    if jp.sum(final_valid_mask) == 0:
        return xfrc_applied
    
    # Extract final valid data
    final_boom_body_ids = boom_body_ids[final_valid_mask]
    final_contact_pos = valid_contact_pos[final_valid_mask]
    
    # Calculate forces vectorized for all valid contacts
    boom_positions = state.data.xpos[final_boom_body_ids]
    direction_vectors = final_contact_pos - boom_positions
    distances = jp.linalg.norm(direction_vectors, axis=1, keepdims=True)
    
    # Avoid division by zero and normalize
    safe_distances = jp.maximum(distances, 1e-8)
    normalized_directions = direction_vectors / safe_distances
    
    force_magnitude = self._config.stickiness_config.stickiness_force
    forces = force_magnitude * normalized_directions
    
    # Apply forces using efficient JAX operations
    return self._accumulate_forces_efficiently(xfrc_applied, final_boom_body_ids, forces)

  def _accumulate_forces_efficiently(self, xfrc_applied, body_ids, forces):
    """Efficiently accumulate forces using JAX operations."""
    if body_ids.shape[0] == 0:
        return xfrc_applied
    
    # Use JAX's advanced indexing for efficient force accumulation
    # This automatically handles multiple forces on the same body by summing them
    
    # Apply translational forces (first 3 components of xfrc_applied)
    for i in range(forces.shape[0]):
        body_id = body_ids[i]
        force = forces[i]
        # Apply force only if it's significant to avoid numerical noise
        force_mag_sq = jp.sum(jp.square(force))
        xfrc_applied = jp.where(
            force_mag_sq > 1e-12,  # Threshold for significant force
            xfrc_applied.at[body_id, :3].add(force),
            xfrc_applied
        )
    
    return xfrc_applied
  
  def _get_termination(self, data: mjx.Data, info: Dict[str, Any]) -> jax.Array:
     qpos = data.qpos
     
     # Get voxel bounds from active environment with buffer
     voxel_bounds = self._active_env["json_data"]["voxel_bounds"]
     buffer = 0.1  # 0.1 meter buffer
     
     # Check if robot position is outside voxel bounds + buffer
     out_of_bounds = (
         (qpos[0] < voxel_bounds["x_min"] - buffer) | (qpos[0] > voxel_bounds["x_max"] + buffer) |
         (qpos[1] < voxel_bounds["y_min"] - buffer) | (qpos[1] > voxel_bounds["y_max"] + buffer) |
         (qpos[2] < voxel_bounds["z_min"] - buffer) | (qpos[2] > voxel_bounds["z_max"] + buffer)
     )

     # No movement termination
     pos_history = info["pos_history"]
     oldest_pos = pos_history[0]
     movement = jp.linalg.norm(qpos[0:3] - oldest_pos)
     
     # Only check for no movement after enough steps have passed
     has_waited_long_enough = info["steps"] > self._no_movement_steps
     is_not_moving = movement < self._no_movement_threshold
     no_movement = is_not_moving & has_waited_long_enough
     
     # NaN detection termination
     has_nan = (jp.any(jp.isnan(qpos)) | 
                jp.any(jp.isnan(data.qvel)) | 
                jp.any(jp.isnan(data.ctrl)) |
                jp.any(jp.isinf(qpos)) |
                jp.any(jp.isinf(data.qvel)))

     return out_of_bounds | no_movement | has_nan # Include NaN in termination conditions


  def _get_obs(
      self, data: mjx.Data, info: Dict[str, Any]
  ) -> Dict[str, jax.Array]:
    gyro = self.get_gyro(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gyro = (
        gyro
        + (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gyro
    )

    gravity = self.get_gravity(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gravity = (
        gravity
        + (2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gravity
    )

    joint_angles = data.qpos[7:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_angles = (
        joint_angles
        + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_pos
    )

    joint_vel = data.qvel[6:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_vel = (
        joint_vel
        + (2 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_vel
    )

    linvel = self.get_local_linvel(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_linvel = (
        linvel
        + (2 * jax.random.uniform(noise_rng, shape=linvel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.linvel
    )

    # LIDAR data
    lidar_pos = self.get_lidar_pos(data)
    lidar_ranges = self._get_lidar_ranges(data, lidar_pos)

    state = jp.hstack([
        noisy_linvel,  # 3
        noisy_gyro,  # 3
        noisy_gravity,  # 3
        noisy_joint_angles,  # 12
        noisy_joint_vel,  # 12
        info["last_act"],  # 12
        lidar_ranges,  # LIDAR
    ])

    accelerometer = self.get_accelerometer(data)
    angvel = self.get_global_angvel(data)
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel()

    privileged_state = jp.hstack([
        state,
        gyro,  # 3
        accelerometer,  # 3
        gravity,  # 3
        linvel,  # 3
        angvel,  # 3
        noisy_joint_vel,  # 12
        data.actuator_force,  # 12
        info["last_contact"],  # 4
        feet_vel,  # 4*3
        data.xfrc_applied[self._torso_body_id, :3],  # 3
        info["steps_since_last_pert"] >= info["steps_until_next_pert"],  # 1
    ])

    return {
        "state": state,
        "privileged_state": privileged_state,
    }

  def _get_lidar_ranges(self, data: mjx.Data, head_pos: jax.Array) -> jax.Array:
    """Casts rays from the robot's head in a 3D pattern and returns distances to nearest geoms."""
    total_lidar_rays = self._lidar_num_horizontal_rays * self._lidar_num_vertical_rays
    ranges = jp.full(total_lidar_rays, self._lidar_max_range, dtype=jp.float32)
    
    imu_quat = mjx_env.get_sensor_data(self._active_env["mj_model"], data, consts.ORIENTATION_SENSOR)

    horizontal_angles = jp.linspace(-self._lidar_horizontal_angle_range / 2, 
                                     self._lidar_horizontal_angle_range / 2, 
                                     self._lidar_num_horizontal_rays)
    
    vertical_angles = jp.linspace(-self._lidar_vertical_angle_range / 2, 
                                   self._lidar_vertical_angle_range / 2, 
                                   self._lidar_num_vertical_rays)

    ray_idx = 0
    for v_angle in vertical_angles:  # Elevation
        for h_angle in horizontal_angles:  # Azimuth
            # Calculate ray direction in head's local frame
            # x = cos(elevation) * cos(azimuth)
            # y = cos(elevation) * sin(azimuth)
            # z = sin(elevation)
            local_ray_dir_x = jp.cos(v_angle) * jp.cos(h_angle)
            local_ray_dir_y = jp.cos(v_angle) * jp.sin(h_angle)
            local_ray_dir_z = jp.sin(v_angle)
            local_ray_dir = jp.array([local_ray_dir_x, local_ray_dir_y, local_ray_dir_z])
            
            rot_mat = math.quat_to_mat(imu_quat)
            world_ray_dir = rot_mat @ local_ray_dir
            
            norm = jp.linalg.norm(world_ray_dir)
            jp.where(norm == 0, 1e-6, norm)  # Avoid division by zero
            world_ray_dir = world_ray_dir / norm

            # Cast ray - use geomgroup to filter collision detection
            # Create collision mask: only collide with cave walls (contype=2)
            # Bit 1 corresponds to contype=2 (cave walls), exclude bit 0 (contype=1, robot parts)
            geomgroup = [0, 1, 0, 0, 0, 0]  # Only enable group 1 for cave walls
            
            hit_dist, hit_geom_id = mjx.ray(self.mjx_model, data, head_pos, world_ray_dir, 
                               geomgroup)

            current_range = jp.where(hit_dist >= 0.0, 
                                     jp.minimum(hit_dist, self._lidar_max_range), 
                                     self._lidar_max_range)
            ranges = ranges.at[ray_idx].set(current_range)
            ray_idx += 1
            
    return ranges

  def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: Dict[str, Any],
        metrics: Dict[str, Any],
        done: jax.Array,
        contact: jax.Array,
    ) -> Dict[str, jax.Array]:
        del metrics  # Unused.
        #jax.debug.print("CaveExplore step: {qpos}", qpos=data.qpos)
        return {
            "distance_to_target": self._reward_dist_to_target(
                data.qpos[0:3], jp.array(info["last_pos"]), jp.array(info["target_pos"]), jp.array(info["init_dist_to_target"])
            ),
            "vel_to_target": self._reward_vel_to_target(
                data.qpos[0:3], jp.array(info["target_pos"]), jp.array(info["last_pos"])
            ),
            "exploration_rate": self._reward_exploration_rate(data.qpos[0:3]),
            "lin_vel_z": self._cost_lin_vel_z(self.get_global_linvel(data)),
            "ang_vel_xy": self._cost_ang_vel_xy(self.get_global_angvel(data)),
            "orientation": self._cost_orientation(self.get_upvector(data)),
            "termination": self._cost_termination(done),
            "torques": self._cost_torques(data.actuator_force),
            "action_rate": self._cost_action_rate(
                action, info["last_act"], info["last_last_act"]
            ),
            "energy": self._cost_energy(data.qvel[6:], data.actuator_force),
            "feet_slip": self._cost_feet_slip(data, contact, info),
            "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[7:]),
        }

  # Base-related rewards.

  def _cost_lin_vel_z(self, global_linvel) -> jax.Array:
    # Penalize z axis base linear velocity.
    return jp.square(global_linvel[2])

  def _cost_ang_vel_xy(self, global_angvel) -> jax.Array:
    # Penalize xy axes base angular velocity.
    return jp.sum(jp.square(global_angvel[:2]))

  def _cost_orientation(self, torso_zaxis: jax.Array) -> jax.Array:
    # Penalize non flat base orientation.
    return jp.sum(jp.square(torso_zaxis[:2]))

  # Energy related rewards.

  def _cost_torques(self, torques: jax.Array) -> jax.Array:
    # Penalize torques.
    return jp.sqrt(jp.sum(jp.square(torques))) + jp.sum(jp.abs(torques))

  def _cost_energy(
      self, qvel: jax.Array, qfrc_actuator: jax.Array
  ) -> jax.Array:
    # Penalize energy consumption.
    return jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator))

  def _cost_action_rate(
      self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
  ) -> jax.Array:
    del last_last_act  # Unused.
    return jp.sum(jp.square(act - last_act))

  def _cost_termination(self, done: jax.Array) -> jax.Array:
    # Penalize early termination.
    return done
  
  def _reward_vel_to_target(self, qpos: jax.Array, target_pos: jax.Array, last_pos: jax.Array) -> jax.Array:
    # Reward for velocity towards target
    last_dist = jp.linalg.norm(last_pos - target_pos)
    current_dist = jp.linalg.norm(qpos - target_pos)
    distance_diff = last_dist - current_dist
    clipped_dist = jp.clip(distance_diff, -1.0, 1.0)
    return clipped_dist

  
  def _reward_dist_to_target(self, current_pos: jax.Array, last_pos: jax.Array, target_pos: jax.Array, max_dist: jax.Array) -> jax.Array:
    # Reward for being closer to target with configurable shaping to reduce variance
    current_dist = jp.linalg.norm(current_pos - target_pos)
    last_dist = jp.linalg.norm(last_pos - target_pos)
    dist_diff = last_dist - current_dist
    
    # Bounded linear reward with proximity bonus
    clipped_reward = jp.clip(dist_diff, -1.0, 1.0)
    
    normalized_current_dist = jp.clip(current_dist / max_dist, 0.01, 1.0)
    proximity_bonus = 0.1 * (1.0 / normalized_current_dist - 1.0)
        
    return clipped_reward + proximity_bonus
  
  def _reward_exploration_rate(self, qpos: jax.Array) -> jax.Array:
    # Reward for exploration - could be enhanced with visited positions tracking
    # For now, return 0 but this could be expanded to reduce variance
    # by encouraging more consistent exploration behavior
    return 0.0
    
    # Future enhancement: track visited positions and reward novel areas
    # This would require adding visited_positions to the info dict and 
    # implementing a spatial hash or grid-based tracking system

  def _cost_joint_pos_limits(self, qpos: jax.Array) -> jax.Array:
    # Penalize joints if they cross soft limits.
    out_of_limits = -jp.clip(qpos - self._soft_lowers, None, 0.0)
    out_of_limits += jp.clip(qpos - self._soft_uppers, 0.0, None)
    return jp.sum(out_of_limits)

  # Feet related rewards.

  def _cost_feet_slip(
      self, data: mjx.Data, contact: jax.Array, info: Dict[str, Any]
  ) -> jax.Array:
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
    vel_xy = feet_vel[..., :2]
    vel_xy_norm_sq = jp.sum(jp.square(vel_xy), axis=-1)
    return jp.sum(vel_xy_norm_sq * contact)


  # Perturbation

  def _maybe_apply_perturbation(self, state: mjx_env.State) -> mjx_env.State:
    def gen_dir(rng: jax.Array) -> jax.Array:
      angle = jax.random.uniform(rng, minval=0.0, maxval=jp.pi * 2)
      return jp.array([jp.cos(angle), jp.sin(angle), 0.0])

    def apply_pert(state: mjx_env.State) -> mjx_env.State:
      t = state.info["pert_steps"] * self.dt
      u_t = 0.5 * jp.sin(jp.pi * t / state.info["pert_duration_seconds"])
      # kg * m/s * 1/s = m/s^2 = kg * m/s^2 (N).
      force = (
          u_t  # (unitless)
          * self._torso_mass  # kg
          * state.info["pert_mag"]  # m/s
          / state.info["pert_duration_seconds"]  # 1/s
      )
      xfrc_applied = jp.zeros((self.mjx_model.nbody, 6))
      xfrc_applied = xfrc_applied.at[self._torso_body_id, :3].set(
          force * state.info["pert_dir"]
      )
      data = state.data.replace(xfrc_applied=xfrc_applied)
      state = state.replace(data=data)
      state.info["steps_since_last_pert"] = jp.where(
          state.info["pert_steps"] >= state.info["pert_duration"],
          0,
          state.info["steps_since_last_pert"],
      )
      state.info["pert_steps"] += 1
      return state

    def wait(state: mjx_env.State) -> mjx_env.State:
      state.info["rng"], rng = jax.random.split(state.info["rng"])
      state.info["steps_since_last_pert"] += 1
      xfrc_applied = jp.zeros((self.mjx_model.nbody, 6))
      data = state.data.replace(xfrc_applied=xfrc_applied)
      state.info["pert_steps"] = jp.where(
          state.info["steps_since_last_pert"]
          >= state.info["steps_until_next_pert"],
          0,
          state.info["pert_steps"],
      )
      state.info["pert_dir"] = jp.where(
          state.info["steps_since_last_pert"]
          >= state.info["steps_until_next_pert"],
          gen_dir(rng),
          state.info["pert_dir"],
      )
      return state.replace(data=data)

    return jax.lax.cond(
        state.info["steps_since_last_pert"]
        >= state.info["steps_until_next_pert"],
        apply_pert,
        wait,
        state,
    )
  
    
  # Accessors.

  @property
  def xml_path(self) -> str:
    """Path to the xml file for the environment."""
    return self._active_env["xml_path"]

  @property
  def action_size(self) -> int:
    """Size of the action space."""
    joints = self._active_env["mj_model"].nu
    if self._config.stickiness_config.enable:
      return 4 + joints
    else:
      return joints

  @property
  def mj_model(self) -> MjModel:
    return self._active_env["mj_model"]

  @property
  def mjx_model(self) -> mjx.Model:
    return self._active_env["mjx_model"]

