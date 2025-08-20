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

from typing import Any, Dict, Optional, Union, Tuple

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
            
            # Positive rewards
            track_lidar_direction=10.0,  # Reward scale for velocity towards target (reduced from 100.0)
            exploration_rate=1.0,       # Reward scale for exploration rate
            wide_stance=0.1,           # Reward scale for maintaining a wide stance for stability

            # Penalties

            orientation=-0.2,           # Penalty scale for orientation deviation (based on upwards vector sensor). Default is -5.0
            distance_from_start=-5.0,     # Penalty scale for distance to target (reduced from 10.0)
            stability=-0.1,          # Penalty scale for stability (reduced from 10.0)
            min_distance=-1.0,          # Penalty scale for minimum distance to walls (0 cost at 0.4m, -1 at 0m)
            
            # Other rewards
            dof_pos_limits=-1.0,        # Penalty scale for degree of freedom position limits. Default is -1.0
            inactivity=-0.1,           # Penalty scale for being inactive (not moving)
            
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
          stickiness_force=100.0,  # Force applied when stickiness is activated (towards wall when in contact)
          min_activation_threshold=0.5,  # Threshold for activating stickiness
          deactivation_threshold=0.3,  # Threshold for deactivating stickiness (hysteresis)
      ),
      lidar_config=config_dict.create(
          num_horizontal_rays=20,  # Number of horizontal rays
          max_range=20.0,  # Maximum range of LIDAR
          horizontal_angle_range=jp.pi * 2,  # Horizontal angle range in radians
          num_vertical_rays=5,  # Number of vertical rays
          vertical_angle_range=jp.pi / 2,  # Vertical angle range in radians
          frequency_hz=10.0,  # LIDAR update frequency in Hz (default 10Hz = 100ms interval)
      ),
      randomize_starting_pos=False,  # Whether to randomize starting positions
  )


class CaveExplore(mjx_env.MjxEnv):
  """Explore the cave environment."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
      cave_batch_loader: Optional[Any] = None,
      scene_type: str = "training",
      scene_data: Optional[Dict[str, Any]] = None,
      domain_randomization_enabled: bool = False,
  ):
    # Replace default config with provided config and overrides
    self._config = config_dict.ConfigDict(config)
    if config_overrides:
      self._config.update(config_overrides)
    
    # Store domain randomization setting
    self._domain_randomization_enabled = domain_randomization_enabled
    
    # Get the scene data - either from cave_batch_loader or directly provided
    if scene_data is not None:
        # Use directly provided scene data
        self._scene_data = scene_data
        self._cave_batch_loader = None
        self._scene_type = scene_type
    elif cave_batch_loader is not None:
        # Use cave_batch_loader to get scene data
        self._cave_batch_loader = cave_batch_loader
        self._scene_type = scene_type
        self._scene_data = cave_batch_loader.get_scene_data(scene_type)
    else:
        raise ValueError("Either cave_batch_loader or scene_data must be provided")
    
    # Public variable containing all loaded cave IDs from the scene
    self.caveIds = list(self._scene_data["caves"].keys())
    # Convert to JAX array for JAX-compatible indexing
    self._cave_ids_array = jp.array(self.caveIds)
    
    # Initialize cave parameters dictionary indexed by cave_id
    self._cave_params = {}
    for cave_id, cave_data in self._scene_data["caves"].items():
        # Convert target_pos dict to list format for consistency
        target_pos = cave_data["target_pos"]
        if isinstance(target_pos, dict):
            target_pos_list = [target_pos.get("x", 0.0), target_pos.get("y", 0.0), target_pos.get("z", 0.0)]
        else:
            target_pos_list = target_pos if target_pos else [0.0, 0.0, 0.0]
            
        # Convert voxel_bounds dict to list format for consistency
        voxel_bounds = cave_data["voxel_bounds"]
        if isinstance(voxel_bounds, dict):
            voxel_bounds_list = [
                voxel_bounds["x_min"], voxel_bounds["x_max"], 
                voxel_bounds["y_min"], voxel_bounds["y_max"], 
                voxel_bounds["z_min"], voxel_bounds["z_max"]
            ]
        else:
            voxel_bounds_list = voxel_bounds
            
        self._cave_params[cave_id] = {
            "box_count": cave_data["box_count"],
            "starting_pos": cave_data["starting_pos"],
            "target_pos": target_pos_list,
            "voxel_bounds": voxel_bounds_list,
            "voxel_positions": cave_data["voxel_positions"]
        }
    
    # Get master cave info
    self._master_cave_id = self._scene_data["master_cave_id"]
    
    # Initialize cave geom IDs dictionary for all boxes in the master cave
    self._master_cave_geom_ids = []
    self._current_cave_id = None
    self._initialize_master_cave_geom_mapping()
    
    # Current environment parameters (will be set when selecting a cave)
    self._current_env = {
        "cave_id": None,
        "target_pos": None,
        "starting_pos": None,
        "voxel_bounds": None,
        "initial_qpos": None
    }
    
    # LIDAR parameters
    self._lidar_num_horizontal_rays = self._config.lidar_config.num_horizontal_rays
    self._lidar_num_vertical_rays = self._config.lidar_config.num_vertical_rays
    self._lidar_max_range = self._config.lidar_config.max_range
    self._lidar_horizontal_angle_range = self._config.lidar_config.horizontal_angle_range
    self._lidar_vertical_angle_range = self._config.lidar_config.vertical_angle_range
    self._lidar_frequency_hz = self._config.lidar_config.frequency_hz
    
    # Calculate LIDAR update interval in simulation steps
    lidar_update_interval_seconds = 1.0 / self._lidar_frequency_hz
    self._lidar_update_interval_steps = int(lidar_update_interval_seconds / self._config.sim_dt)
    
    # Ensure minimum update interval of 1 step
    self._lidar_update_interval_steps = max(1, self._lidar_update_interval_steps)
    
    # Other paramters
    self._max_ms = 0.1  # Maximum meters per second for velocity of robot
    
    # Prepare cave data arrays for domain randomization if enabled
    if self._domain_randomization_enabled:
        self._prepare_domain_randomization_data()
    
    # Call parent class __init__
    super().__init__(config, config_overrides)
    
    # Call _post_init to initialize model-dependent attributes
    self._post_init()
    
  def get_environment_info(self):
    """Get information about this environment instance."""
    return {
        "scene_type": self._scene_type,
        "num_caves": len(self.caveIds),
        "cave_ids": sorted(self.caveIds),
        "master_cave_id": self._master_cave_id,
        "has_cave_batch_loader": self._cave_batch_loader is not None,
        "domain_randomization_enabled": self._domain_randomization_enabled
    }

  def _prepare_domain_randomization_data(self):
    """Prepare cave data arrays for domain randomization."""
    max_boxes = 7500  # Should match the value used in domain randomization setup
    
    num_caves = len(self.caveIds)
    
    # Initialize arrays
    cave_box_counts = jp.zeros(num_caves, dtype=jp.int32)
    
    # Get all cave target positions and voxel bounds
    all_target_positions = jp.zeros((num_caves, 3))
    all_voxel_bounds = jp.zeros((num_caves, 6))  # x_min, x_max, y_min, y_max, z_min, z_max
    
    # Prepare starting positions arrays - pad to consistent length
    max_starting_positions = 10  # Assume max 10 starting positions per cave
    all_starting_pos_x = jp.zeros((num_caves, max_starting_positions))
    all_starting_pos_y = jp.zeros((num_caves, max_starting_positions))
    all_starting_pos_z = jp.zeros((num_caves, max_starting_positions))
    starting_pos_counts = jp.zeros(num_caves, dtype=jp.int32)
    
    # Fill in cave data
    for cave_idx, cave_id in enumerate(self.caveIds):
        cave_data = self._cave_params[cave_id]
        
        # Voxel positions
        voxel_positions = cave_data["voxel_positions"]
        num_boxes = min(len(voxel_positions), max_boxes)
        cave_box_counts = cave_box_counts.at[cave_idx].set(num_boxes)
        
        # Target positions
        target_pos = jp.array(cave_data["target_pos"])
        all_target_positions = all_target_positions.at[cave_idx].set(target_pos)
        
        # Voxel bounds
        voxel_bounds = jp.array(cave_data["voxel_bounds"])
        all_voxel_bounds = all_voxel_bounds.at[cave_idx].set(voxel_bounds)
        
        # Starting positions
        starting_pos = cave_data["starting_pos"]
        valid_starting_pos = [pos for pos in starting_pos if pos is not None]
        num_starting = min(len(valid_starting_pos), max_starting_positions)
        starting_pos_counts = starting_pos_counts.at[cave_idx].set(num_starting)
        
        for i in range(num_starting):
            pos = valid_starting_pos[i]
            all_starting_pos_x = all_starting_pos_x.at[cave_idx, i].set(pos.get("x", 0.0))
            all_starting_pos_y = all_starting_pos_y.at[cave_idx, i].set(pos.get("y", 0.0))
            all_starting_pos_z = all_starting_pos_z.at[cave_idx, i].set(pos.get("z", 0.0))
    
    # Store the prepared arrays
    self._domain_randomization_data = {
        'cave_box_counts': cave_box_counts,
        'all_target_positions': all_target_positions,
        'all_voxel_bounds': all_voxel_bounds,
        'all_starting_pos_x': all_starting_pos_x,
        'all_starting_pos_y': all_starting_pos_y,
        'all_starting_pos_z': all_starting_pos_z,
        'starting_pos_counts': starting_pos_counts,
        'cave_ids_array': self._cave_ids_array,
    }

  def get_current_cave_data_from_randomization(self, cave_idx: jax.Array) -> Dict[str, jax.Array]:
    """Get cave data for the given cave index from domain randomization arrays."""
    if not self._domain_randomization_enabled:
        raise ValueError("Domain randomization is not enabled")
    
    dr_data = self._domain_randomization_data
    
    return { 
        'starting_pos_x': dr_data['all_starting_pos_x'][cave_idx],
        'starting_pos_y': dr_data['all_starting_pos_y'][cave_idx],
        'starting_pos_z': dr_data['all_starting_pos_z'][cave_idx],
        'target_pos': dr_data['all_target_positions'][cave_idx],
        'starting_pos_length': dr_data['starting_pos_counts'][cave_idx],
        'cave_id': dr_data['cave_ids_array'][cave_idx],
        'voxel_bounds': dr_data['all_voxel_bounds'][cave_idx]
    }
    

  @property
  def mjx_model(self):
    """Return the mjx_model from the scene data."""
    return self._scene_data["mjx_model"]

  def _initialize_master_cave_geom_mapping(self):
    """Initialize mapping of geom IDs for the master cave boxes."""
    mj_model = self._scene_data["mj_model"]
    
    # Find all geoms that belong to the master cave (they should be named with master cave pattern)
    for i in range(mj_model.ngeom):
        geom_name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if geom_name and geom_name.startswith(f"cave_wall_box_{self._master_cave_id}_"):
            self._master_cave_geom_ids.append(i)
    
    # Convert to JAX array for efficiency
    self._master_cave_geom_ids = jp.array(self._master_cave_geom_ids)
    
    print(f"Initialized {len(self._master_cave_geom_ids)} geoms for master cave {self._master_cave_id}")

  def select_cave_environment(self, cave_id: int):
    """Select a specific cave environment by repositioning boxes."""
    if cave_id not in self._cave_params:
        raise ValueError(f"Cave ID {cave_id} not found in available caves: {list(self._cave_params.keys())}")
    
    # Update current environment parameters
    cave_data = self._cave_params[cave_id]
    self._current_cave_id = cave_id
    self._current_env.update({
        "cave_id": cave_id,
        "target_pos": cave_data["target_pos"],
        "starting_pos": cave_data["starting_pos"], 
        "voxel_bounds": cave_data["voxel_bounds"],
    })
    
    # Set initial position from starting positions
    starting_positions = cave_data["starting_pos"]
    if starting_positions and len(starting_positions) > 0:
        # Use first valid starting position
        first_pos = None
        for pos in starting_positions:
            if pos is not None:
                first_pos = [pos.get("x", 0.0), pos.get("y", 0.0), pos.get("z", 0.0)]
                break
        if first_pos is None:
            first_pos = [0.0, 0.0, 0.0]  # Fallback
        self._current_env["initial_qpos"] = jp.array(first_pos)
    else:
        self._current_env["initial_qpos"] = jp.array([0.0, 0.0, 0.0])
    
    print(f"Selected cave environment {cave_id}")
    print(f"Target position: {self._current_env['target_pos']}")
    print(f"Number of starting positions: {len([p for p in self._current_env['starting_pos'] if p is not None])}")
    
    return self._current_env
    

  def _post_init(self) -> None:
    self._init_q = jp.array(self._scene_data["mj_model"].keyframe("home").qpos)
    self._default_pose = jp.array(self._scene_data["mj_model"].keyframe("home").qpos[7:])

    # Note: First joint is freejoint.
    self._lowers, self._uppers = self._scene_data["mj_model"].jnt_range[1:].T
    self._soft_lowers = self._lowers * self._config.soft_joint_pos_limit_factor
    self._soft_uppers = self._uppers * self._config.soft_joint_pos_limit_factor

    self._torso_body_id = self._scene_data["mj_model"].body(consts.ROOT_BODY).id
    self._torso_mass = self._scene_data["mj_model"].body_subtreemass[self._torso_body_id]

    self._no_movement_duration = 5.0  # seconds
    self._no_movement_threshold = 0.1  # meters
    self._no_movement_steps = jp.array(self._no_movement_duration / self.sim_dt, dtype=jp.int32)

    self._feet_site_id = np.array(
        [self._scene_data["mj_model"].site(name).id for name in consts.FEET_SITES]
    )
    # Collect all geoms whose names start with "cave_wall_box" as cave geoms
    self._cave_geom_ids = np.array([
      i for i in range(self._scene_data["mj_model"].ngeom)
      if mujoco.mj_id2name(self._scene_data["mj_model"], mujoco.mjtObj.mjOBJ_GEOM, i).startswith("cave_wall_box")
    ])
    print("Cave boxes detected:", len(self._cave_geom_ids))
    self._feet_geom_id = np.array(
        [self._scene_data["mj_model"].geom(name).id for name in consts.FEET_GEOMS]
    )

    foot_linvel_sensor_adr = []
    for site in consts.FEET_SITES:
      sensor_id = self._scene_data["mj_model"].sensor(f"{site}_global_linvel").id
      sensor_adr = self._scene_data["mj_model"].sensor_adr[sensor_id]
      sensor_dim = self._scene_data["mj_model"].sensor_dim[sensor_id]
      foot_linvel_sensor_adr.append(
          list(range(sensor_adr, sensor_adr + sensor_dim))
      )
    self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)

    # Initialize IMU site ID which is needed for get_gravity method
    self._imu_site_id = self._scene_data["mj_model"].site("imu").id

    mj_model = self._scene_data["mj_model"]
    
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

    self._cave_geom_ids = jp.array(sorted(floor_geom_ids)) if floor_geom_ids else jp.array([])

    print(f"Found {len(boom_geom_ids)} boom end geoms")
    print(f"Found {len(floor_geom_ids)} floor/wall geoms")

    # Pre-compute normalized local LIDAR ray directions
    self._precompute_lidar_directions()

    print("CaveExplore task action space:", self.action_size)

    self._max_dist_per_step = self._max_ms * self._config.sim_dt  # Maximum distance per step based on max speed

  def _precompute_lidar_directions(self) -> None:
    """Precompute normalized local LIDAR ray directions for efficiency and network observation."""
    horizontal_angles = jp.linspace(-self._lidar_horizontal_angle_range / 2,
                                     self._lidar_horizontal_angle_range / 2,
                                     self._lidar_num_horizontal_rays)
    
    vertical_angles = jp.linspace(-self._lidar_vertical_angle_range / 2, 
                                   self._lidar_vertical_angle_range / 2, 
                                   self._lidar_num_vertical_rays)

    # Precompute all ray directions
    local_ray_dirs = []
    for v_angle in vertical_angles:  # Elevation
        for h_angle in horizontal_angles:  # Azimuth
            local_ray_dir_x = jp.cos(v_angle) * jp.cos(h_angle)
            local_ray_dir_y = jp.cos(v_angle) * jp.sin(h_angle)
            local_ray_dir_z = jp.sin(v_angle)
            local_ray_dir = jp.array([local_ray_dir_x, local_ray_dir_y, local_ray_dir_z])
            
            # Normalize the direction vector
            norm = jp.linalg.norm(local_ray_dir)
            norm = jp.where(norm == 0, 1e-6, norm)  # Avoid division by zero
            local_ray_dir = local_ray_dir / norm
            
            local_ray_dirs.append(local_ray_dir)
    
    # Store as JAX array for efficient access
    self._local_ray_directions = jp.stack(local_ray_dirs)
    print(f"Precomputed {len(local_ray_dirs)} LIDAR ray directions")

  def _select_random_env(self, rng: jax.Array) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """Select cave index for this reset (JAX-compatible)."""
    if self._domain_randomization_enabled:
        # Use domain randomization data
        self._rng, key = jax.random.split(rng)
        num_caves = len(self.caveIds)
        cave_idx = jax.random.randint(key, (), 0, num_caves)
        env_data_jax = self.get_current_cave_data_from_randomization(cave_idx)
        return cave_idx, env_data_jax
    else:
        # Original behavior for manual cave selection
        self._rng, key = jax.random.split(rng)
        num_caves = len(self.caveIds)
        cave_idx = jax.random.randint(key, (), 0, num_caves)
        cave_id = self._cave_ids_array[cave_idx]
        
        # Get cave data for the selected cave
        # We need to use the cave_idx to index into our data structures
        # Since we can't use cave_id (a traced value) as a dictionary key,
        # we'll prepare all cave data in arrays and index by cave_idx
        
        # Convert cave parameters to arrays indexed by cave order
        starting_pos_arrays = []
        target_pos_arrays = []
        voxel_bounds_arrays = []
        
        for cave_id_val in self.caveIds:
            cave_data = self._cave_params[cave_id_val]
            
            # Starting positions - pad to same length
            starting_pos = cave_data["starting_pos"]
            max_positions = 10  # Assume max 10 starting positions per cave
            
            start_x = [pos.get("x", 0.0) if pos is not None else 0.0 for pos in starting_pos] + [0.0] * (max_positions - len(starting_pos))
            start_y = [pos.get("y", 0.0) if pos is not None else 0.0 for pos in starting_pos] + [0.0] * (max_positions - len(starting_pos))
            start_z = [pos.get("z", 0.0) if pos is not None else 0.0 for pos in starting_pos] + [0.0] * (max_positions - len(starting_pos))
            valid_count = len([p for p in starting_pos if p is not None])
            
            starting_pos_arrays.append({
                'x': jp.array(start_x[:max_positions]),
                'y': jp.array(start_y[:max_positions]), 
                'z': jp.array(start_z[:max_positions]),
                'count': jp.array(valid_count)
            })
            
            # Target position - should already be in list format
            target_pos = cave_data["target_pos"] if cave_data["target_pos"] else [0.0, 0.0, 0.0]
            target_pos_arrays.append(jp.array(target_pos))
            
            # Voxel bounds - should already be in list format
            voxel_bounds = cave_data["voxel_bounds"] if cave_data["voxel_bounds"] else [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            voxel_bounds_arrays.append(jp.array(voxel_bounds))
        
        # Stack into arrays
        starting_x_stack = jp.stack([sp['x'] for sp in starting_pos_arrays])
        starting_y_stack = jp.stack([sp['y'] for sp in starting_pos_arrays])
        starting_z_stack = jp.stack([sp['z'] for sp in starting_pos_arrays])
        starting_counts = jp.stack([sp['count'] for sp in starting_pos_arrays])
        target_stack = jp.stack(target_pos_arrays)
        voxel_bounds_stack = jp.stack(voxel_bounds_arrays)
        
        # Index by cave_idx
        env_data_jax = {
            'starting_pos_x': starting_x_stack[cave_idx],
            'starting_pos_y': starting_y_stack[cave_idx],
            'starting_pos_z': starting_z_stack[cave_idx],
            'target_pos': target_stack[cave_idx],
            'starting_pos_length': starting_counts[cave_idx],
            'cave_id': cave_id,
            'voxel_bounds': voxel_bounds_stack[cave_idx]
        }
        
        return cave_idx, env_data_jax

  def get_current_cave_data_from_state(self, state: mjx_env.State) -> Dict[str, Any]:
    """Get current cave data from the environment state.
    
    This method can be used to retrieve cave information after domain randomization
    has been applied during training.
    
    Args:
        state: The current environment state
        
    Returns:
        Dictionary containing current cave information
    """
    info = state.info
    
    if self._domain_randomization_enabled:
        cave_idx = info["cave_idx"]
        # Convert JAX arrays back to Python types for easier handling
        cave_data = {
            "cave_id": int(info["cave_id"]),
            "cave_idx": int(cave_idx),
            "target_pos": [float(x) for x in info["target_pos"]],
            "voxel_bounds": [float(x) for x in info["voxel_bounds"]],
            "init_pos": [float(x) for x in info["init_pos"]],
            "domain_randomization_enabled": bool(info["domain_randomization_enabled"])
        }
    else:
        # For manual cave selection
        cave_data = {
            "cave_id": int(info["cave_id"]) if "cave_id" in info else self._current_cave_id,
            "cave_idx": int(info.get("cave_idx", 0)),
            "target_pos": [float(x) for x in info["target_pos"]],
            "voxel_bounds": [float(x) for x in info["voxel_bounds"]],
            "init_pos": [float(x) for x in info["init_pos"]],
            "domain_randomization_enabled": False
        }
    
    return cave_data

  def get_domain_randomization_info(self) -> Dict[str, Any]:
    """Get information about domain randomization setup.
    
    Returns:
        Dictionary containing domain randomization information
    """
    if not self._domain_randomization_enabled:
        return {
            "enabled": False,
            "num_caves": len(self.caveIds),
            "cave_ids": self.caveIds
        }
    
    dr_data = self._domain_randomization_data
    return {
        "enabled": True,
        "num_caves": len(self.caveIds),
        "cave_ids": self.caveIds,
        "max_boxes": dr_data['max_boxes'].shape[1],
        "cave_data_shape": {
            "all_target_positions": dr_data['all_target_positions'].shape,
            "all_voxel_bounds": dr_data['all_voxel_bounds'].shape,
            "starting_pos_counts": dr_data['starting_pos_counts'].shape
        }
    }

  def set_training_mode(self, training: bool = True):
    """Switch between training and evaluation scenes - deprecated in new design."""
    print("Warning: set_training_mode is deprecated. Use separate training and eval environment instances.")

  @property
  def current_mjx_model(self):
    """Return the mjx_model for the currently selected environment."""
    return self.mjx_model

  @property  
  def current_mj_model(self):
    """Return the mj_model for the currently active cave environment."""
    return self._scene_data["mj_model"]

  def get_upvector(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.current_mj_model, data, consts.UPVECTOR_SENSOR)

  def get_gravity(self, data: mjx.Data) -> jax.Array:
    return data.site_xmat[self._imu_site_id].T @ jp.array([0, 0, -1])

  def get_global_linvel(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(
        self.current_mj_model, data, consts.GLOBAL_LINVEL_SENSOR
    )

  def get_global_angvel(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(
        self.current_mj_model, data, consts.GLOBAL_ANGVEL_SENSOR
    )

  def get_local_linvel(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(
        self.current_mj_model, data, consts.LOCAL_LINVEL_SENSOR
    )

  def get_accelerometer(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(
        self.current_mj_model, data, consts.ACCELEROMETER_SENSOR
    )

  def get_gyro(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.current_mj_model, data, consts.GYRO_SENSOR)

  def get_lidar_pos(self, data: mjx.Data) -> jax.Array: # Added for LIDAR
    return mjx_env.get_sensor_data(self.current_mj_model, data, consts.HEAD_POS_SENSOR) # Added for LIDAR

  def get_feet_pos(self, data: mjx.Data) -> jax.Array:
    return jp.vstack([
        mjx_env.get_sensor_data(self.current_mj_model, data, sensor_name)
        for sensor_name in consts.FEET_POS_SENSOR
    ])

  def _qpos_to_motor_ctrl(self, qpos: jax.Array) -> jax.Array:
    """Convert joint angles to control input format"""
    return qpos[7:7+self.mjx_model.nu]
    
  def _get_current_cave_data(self) -> Dict[str, jax.Array]:
    """Get the data for the currently selected cave in JAX format."""
    if self._current_cave_id is None:
        raise ValueError("No cave is currently selected")
    
    cave_data = self._cave_params[self._current_cave_id]
    
    # Convert starting positions to JAX arrays
    starting_pos = cave_data["starting_pos"]
    max_positions = 10  # Same as in _select_random_env
    
    start_x = [pos.get("x", 0.0) if pos is not None else 0.0 for pos in starting_pos] + [0.0] * (max_positions - len(starting_pos))
    start_y = [pos.get("y", 0.0) if pos is not None else 0.0 for pos in starting_pos] + [0.0] * (max_positions - len(starting_pos))
    start_z = [pos.get("z", 0.4) if pos is not None else 0.4 for pos in starting_pos] + [0.0] * (max_positions - len(starting_pos))
    valid_count = len([p for p in starting_pos if p is not None])
    
    return {
        'starting_pos_x': jp.array(start_x[:max_positions]),
        'starting_pos_y': jp.array(start_y[:max_positions]),
        'starting_pos_z': jp.array(start_z[:max_positions]),
        'target_pos': jp.array(cave_data["target_pos"]),
        'starting_pos_length': jp.array(valid_count),
        'cave_id': jp.array(self._current_cave_id),
        'voxel_bounds': jp.array(cave_data["voxel_bounds"])
    }

  def reset(self, rng: jax.Array) -> mjx_env.State:
    """Reset the environment. Works with both domain randomization and manual cave selection."""
    
    if self._domain_randomization_enabled:
        # For domain randomization, select a random cave and get its data
        rng, cave_rng = jax.random.split(rng)
        num_caves = len(self.caveIds)
        cave_idx = jax.random.randint(cave_rng, (), 0, num_caves)
        env_data = self.get_current_cave_data_from_randomization(cave_idx)
    else:
        # For manual cave selection, require a cave to be selected first
        if self._current_cave_id is None:
            raise ValueError("No cave is currently selected. Call select_cave_environment first.")
        env_data = self._get_current_cave_data()
        cave_idx = jp.array(0)  # Not used in manual mode, but needed for info dict
    
    length_starting_pos = env_data['starting_pos_length']
    rng, key1, key2 = jax.random.split(rng, 3)
    qpos = self._init_q.copy()
    
    # 30% chance for random starting position, 70% chance for first starting position
    use_random_pos = jax.random.uniform(key1) < 0.3
    random_index = jax.random.randint(key2, (), 0, length_starting_pos)

    if self._config.randomize_starting_pos:
      selected_index = jp.where(use_random_pos, random_index, 0)
    else:
      selected_index = jp.array(0)

    
    new_position = jp.array([
        env_data['starting_pos_x'][selected_index], 
        env_data['starting_pos_y'][selected_index], 
        env_data['starting_pos_z'][selected_index]
    ])
    qpos = qpos.at[:3].set(new_position)

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

    pos_history = jp.tile(qpos[0:3], (self._no_movement_steps, 1))
  
    info = {
        "rng": rng,
        "init_pos": qpos[0:3],
        "target_pos": env_data['target_pos'],
        "voxel_bounds": env_data['voxel_bounds'],
        "cave_id": env_data['cave_id'],
        "cave_idx": cave_idx,  # Store the cave index for domain randomization
        "domain_randomization_enabled": self._domain_randomization_enabled,
        "lidar_ranges": jp.zeros(self._lidar_num_horizontal_rays * self._lidar_num_vertical_rays),
        "deepest_lidar_direction": jp.zeros(3),  # Direction of the deepest LIDAR ray
        "lidar_step_counter": jp.array(0),  # Counter for LIDAR update frequency
        "last_act": jp.zeros(self.action_size),  # Changed from self.mjx_model.nu to self.action_size
        "last_last_act": jp.zeros(self.action_size),  # Changed from self.mjx_model.nu to self.action_size
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
        "heading_from_imu": 0.0,
        "distance_from_imu": 0.0,
        "torso_contact": 0,  # Track if torso is in contact with cave walls
    }

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())

    # Calculate initial LIDAR data
    lidar_pos = self.get_lidar_pos(data)
    initial_lidar_ranges = self._get_lidar_ranges(data, lidar_pos)
    initial_deepest_lidar_direction = self._get_avg_deepest_lidar_range(
        initial_lidar_ranges, self._local_ray_directions
    )
    info["lidar_ranges"] = initial_lidar_ranges
    info["deepest_lidar_direction"] = initial_deepest_lidar_direction

    obs = self._get_obs(data, info)
    reward, done = jp.zeros(2)
    
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    """Applies action to the environment and returns the new state."""
    if self._config.pert_config.enable:
      state = self._maybe_apply_perturbation(state)
    
    state_formatted = self._qpos_to_motor_ctrl(state.data.qpos)
    actuator_action = action[:self.mjx_model.nu]
    
    # Extract stickiness action if enabled
    stickiness_action = jp.zeros(4)  # Default to zero if not enabled
    if self._config.stickiness_config.enable:
        stickiness_action = action[self.mjx_model.nu:] 
        
    motor_targets = state_formatted + actuator_action * self._config.action_scale

    torso_contact_dist, torso_contact_normal = self.get_collision_with_torso(
        state.data._impl.contact
    )
    state.info["torso_contact"] = jp.where(
        torso_contact_dist < 0.0, 1, 0
    )

    data = mjx_env.step(
        self.mjx_model, state.data, motor_targets, self.n_substeps
    )
    
    # Apply stickiness forces if boom ends are in contact and activated
    if self._config.stickiness_config.enable:
        data = self.apply_stickiness_forces(data, stickiness_action, state.info)
    
    # Renormalize quaternion to prevent numerical drift
    data = data.replace(qpos=renormalize_quat(data.qpos))

    p_f = data.site_xpos[self._feet_site_id]
    p_fz = p_f[..., -1]

    # Update position history for termination condition
    pos_history = jp.roll(state.info["pos_history"], shift=-1, axis=0)
    pos_history = pos_history.at[-1].set(data.qpos[0:3])
    state.info["pos_history"] = pos_history

    obs = self._get_obs(data, state.info)
    done = self._get_termination(data, state.info)

    rewards = self._get_reward(
        data, action, state.info, state.metrics, done
    )
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    #reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)
    reward = sum(rewards.values()) * self.dt

    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    state.info["last_pos"] = data.qpos[0:3]
    state.info["lidar_step_counter"] += 1  # Increment LIDAR step counter
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v

    done = done.astype(reward.dtype)
    state = state.replace(data=data, obs=obs, reward=reward, done=done)
    state.info["steps"] += 1
    return state
  
  def get_collisions_with_boom_ends(
    self,
    contact: Any,
    boom_ends: jp.array,   # int32[M]
    cave_geoms: jp.array,  # int32[K]
) -> Tuple[jp.ndarray, jp.ndarray]:
    """
    contact.geom   -> int32[N,2]
    contact.dist   -> float32[N]
    contact.frame  -> float32[N,2,7]

    Returns:
      dists   : float32[M]    (deepest dist per boom_end)
      normals : float32[M,3]  (outward normal per boom_end)
    """
    geom_pair = contact.geom        # [N,2]
    dist_arr  = contact.dist        # [N]
    frame     = contact.frame       # [N,2,7]

    # Expand dims so we can compare every boom_end against every contact:
    # boom_ends[:,None] has shape [M,1]
    # geom_pair[None,:,:] has shape [1,N,2]
    be = boom_ends[:, None]         # [M,1]
    g0 = geom_pair[None, :, 0]      # [1,N]
    g1 = geom_pair[None, :, 1]      # [1,N]

    # Which contact slots match boom_end?
    is_be0 = (g0 == be)             # [M,N]
    is_be1 = (g1 == be)             # [M,N]

    # Which contact slots match any cave_geom?
    in_cave0 = jp.isin(g0, cave_geoms)  # [1,N] broadcast→[M,N]
    in_cave1 = jp.isin(g1, cave_geoms)  # [1,N] broadcast→[M,N]

    # Final mask: boom_end on one side, cave on the other
    mask = ((is_be0 & in_cave1) | (is_be1 & in_cave0)) & (dist_arr[None, :] < 0.0)  # [M,N]

    # Broadcast dist_arr to [M,N], but push non‑matches to +inf
    dists_b = jp.where(mask, dist_arr[None, :], jp.inf)  # [M,N]

    # For each boom_end (axis=1), pick the *minimum* dist (deepest penetration)
    idx = jp.argmin(dists_b, axis=1)  # [M], indices into contacts

    # Gather the actual deepest distances
    deepest = dists_b[jp.arange(boom_ends.shape[0]), idx]  # [M]

    # Now gather normals.  frame[:,0,:3] is the normal for geom_pair[:,0]
    # We need normals[be] to always point *out* of the boom_end:
    #   if geom_pair[i,0] == boom_end  → use  frame[i,0,:3]
    #   else                           → use -frame[i,0,:3]
    normals0 = frame[:, 0, :3]       # [N,3]
    chosen_norm0 = normals0[idx]     # [M,3]

    # Determine whether the boom_end was at geom_pair[idx,0] (vs geom_pair[idx,1])
    be_at_pos0 = (geom_pair[idx, 0] == boom_ends)
    normals = jp.where(be_at_pos0[:, None],
                        chosen_norm0,
                        -chosen_norm0)  # [M,3]

    return deepest, normals
  
  def apply_stickiness_forces(
      self, 
      data: mjx.Data, 
      stickiness_action: jax.Array,
      state_info: Dict[str, Any]
  ) -> mjx.Data:
    """Apply external forces when boom ends are in contact with walls and activated.
    
    For every step, if a boom end is in contact with one or more wall geoms (actual penetrating 
    contact) and the corresponding boom end is activated by the neural net, there will be an 
    external force of 100N along the contact vector that pushes the boom end towards the wall. 
    For every boom this force shall only be done for the wall geom that is penetrated the 
    deepest by this boom end. This force shall remain active until the boom end is no longer active.
    
    When the network output for a boom switches from 1 to 0, all external stickiness forces 
    related to that boom are deleted.
    
    IMPORTANT: Each boom can only have one stickiness force at a time. When a boom is deactivated,
    ALL external forces on that boom body are cleared to ensure clean force management.
    
    Args:
        data: MuJoCo data containing contact information
        stickiness_action: Array of stickiness activation values [4]
        state_info: State info dictionary containing boom stickiness state
        
    Returns:
        Updated data with applied forces
    """
    if not self._config.stickiness_config.enable:
        return data
        
    # Get contact information between boom ends and cave walls
    deepest_dists, contact_normals = self.get_collisions_with_boom_ends(
        data._impl.contact,
        self._boom_geom_ids,
        self._master_cave_geom_ids
    )
    
    # Check which booms are in penetrating contact (negative distance means penetration)
    in_contact = deepest_dists < 0.0  # [4] boolean array
    
    # Check which booms are activated by neural network
    activated = stickiness_action > self._config.stickiness_config.min_activation_threshold  # [4] boolean array
    
    # Update stickiness state: active if both in contact and activated
    current_stickiness = in_contact & activated
    
    # Get previous stickiness state
    prev_stickiness = state_info.get("boom_stickiness_active", jp.zeros(4, dtype=bool))
    
    # Detect deactivation: boom was active but is now inactive
    deactivated = prev_stickiness & ~current_stickiness
    
    # Start with current external forces
    updated_xfrc = data.xfrc_applied.copy()
    
    # Clear forces for deactivated booms (when network output switches from 1 to 0)
    # Use JAX-compatible operations instead of loops
    def clear_deactivated_force(carry, i):
        xfrc, deactivated_array = carry
        boom_body_id = self._boom_body_ids[i]
        # Use jp.where to conditionally clear forces
        zero_forces = jp.zeros(6)
        current_forces = xfrc[boom_body_id, :]
        new_forces = jp.where(deactivated_array[i], zero_forces, current_forces)
        xfrc = xfrc.at[boom_body_id, :].set(new_forces)
        return (xfrc, deactivated_array), None
    
    (updated_xfrc, _), _ = jax.lax.scan(clear_deactivated_force, 
                                        (updated_xfrc, deactivated), 
                                        jp.arange(len(self._boom_body_ids)))
    
    # Apply new forces for currently active booms using JAX-compatible operations
    def apply_stickiness_force(carry, i):
        xfrc = carry
        # Only apply force if boom is actively sticking and has valid contact
        should_apply_force = current_stickiness[i] & jp.isfinite(deepest_dists[i])
        
        # Calculate force vector: force magnitude * contact normal (towards wall)
        # The contact normal points outward from the boom, so we use it directly 
        # to push the boom towards the wall
        force_vector = (
            self._config.stickiness_config.stickiness_force 
            * contact_normals[i]  # Normal points towards wall when boom is penetrating
        )
        
        # Apply force only if we should (use jp.where to avoid conditional logic)
        force_to_apply = jp.where(should_apply_force, force_vector, jp.zeros(3))
        
        # Apply force to the boom end body
        boom_body_id = self._boom_body_ids[i]
        xfrc = xfrc.at[boom_body_id, :3].add(force_to_apply)
        return xfrc, None
    
    updated_xfrc, _ = jax.lax.scan(apply_stickiness_force, 
                                   updated_xfrc, 
                                   jp.arange(len(self._boom_body_ids)))
    
    # Update state info with current stickiness state
    state_info["boom_stickiness_active"] = current_stickiness
    
    return data.replace(xfrc_applied=updated_xfrc)
  
  def get_collision_with_torso(
        self,
        contact: Any,
    ) -> Tuple[jp.ndarray, jp.ndarray]:
        """
        Returns (dist, normal) for the *deepest* contact
        between `torso_geom` and *any* of the geoms in `cave_geoms`.
        If there are no such contacts, dist will be +1e8 and normal = 0.
        """
        # contact.geom: (Ncont, 2), contact.dist: (Ncont,), contact.frame: (Ncont, 2, 7)
        geom_pair = contact.geom            # int32[Ncont,2]
        dist_arr = contact.dist             # float32[Ncont]
        frame    = contact.frame            # float32[Ncont,2,7]

        # build mask of “this contact involves torso_geom on one side
        # and ANY cave_geom on the other side”
        is_t0 = geom_pair[:, 0] == self._torso_body_id
        is_t1 = geom_pair[:, 1] == self._torso_body_id

        # jnp.isin for membership test: shape (Ncont,)
        # cave_geoms is an array of your 3 000 wall‐ids
        in_cave0 = jp.isin(geom_pair[:, 0], self._cave_geom_ids)
        in_cave1 = jp.isin(geom_pair[:, 1], self._cave_geom_ids)

        mask = (is_t0 & in_cave1) | (is_t1 & in_cave0)  # bool[Ncont]

        # replace all non‐matches with +inf so argmin picks only valid ones
        safe_dists = jp.where(mask, dist_arr, 1e8)

        # index of the minimal distance (i.e. deepest penetration)
        idx = jp.argmin(safe_dists)

        # pull it out
        d = safe_dists[idx]

        # frame[idx,0,:3] is the normal *for geom_pair[idx,0]*;
        # if geom_pair[idx,0] was the torso we leave it, otherwise flip it.
        n0 = frame[idx, 0, :3]
        normal = jp.where(geom_pair[idx,0] == self._torso_body_id, n0, -n0)

              
        # Use jp.where instead of if statement for JAX compatibility
        no_contact_dist = jp.array(1e8, dtype=jp.float32)
        no_contact_normal = jp.zeros(3, dtype=jp.float32)

        # Return based on whether there's contact (d <= 0) or not (d > 0)
        final_dist = jp.where(d > 0, no_contact_dist, d)
        final_normal = jp.where(d > 0, no_contact_normal, normal)

        return final_dist, final_normal


  def _get_termination(self, data: mjx.Data, info: Dict[str, Any]) -> jax.Array:
     qpos = data.qpos
     
     # Get voxel bounds from info (passed from reset) with buffer
     # voxel_bounds format: [x_min, x_max, y_min, y_max, z_min, z_max]
     voxel_bounds = info["voxel_bounds"]
     buffer = 0.1  # 0.1 meter buffer
     
     # Check if robot position is outside voxel bounds + buffer
     out_of_bounds = (
         (qpos[0] < voxel_bounds[0] - buffer) | (qpos[0] > voxel_bounds[1] + buffer) |
         (qpos[1] < voxel_bounds[2] - buffer) | (qpos[1] > voxel_bounds[3] + buffer) |
         (qpos[2] < voxel_bounds[4] - buffer) | (qpos[2] > voxel_bounds[5] + buffer)
     )
     
     # Check if any feet positions are outside voxel bounds + buffer
     feet_positions = data.site_xpos[self._feet_site_id]  # Shape: (n_feet, 3)
     feet_out_of_bounds = jp.any(
         (feet_positions[:, 0] < voxel_bounds[0] - buffer) | (feet_positions[:, 0] > voxel_bounds[1] + buffer) |
         (feet_positions[:, 1] < voxel_bounds[2] - buffer) | (feet_positions[:, 1] > voxel_bounds[3] + buffer) |
         (feet_positions[:, 2] < voxel_bounds[4] - buffer) | (feet_positions[:, 2] > voxel_bounds[5] + buffer)
     )

     # No movement termination
     pos_history = info["pos_history"]
     oldest_pos = pos_history[0]
     movement = jp.linalg.norm(qpos[0:3] - oldest_pos)
     
     # Only check for no movement after enough steps have passed
     has_waited_long_ENOUGH = info["steps"] > self._no_movement_steps
     is_not_moving = movement < self._no_movement_threshold
     no_movement = is_not_moving & has_waited_long_ENOUGH

     # Terminate if upvector is inversed 
     fall_termination = self.get_upvector(data)[-1] < 0.0

     # Terminate if torso is in contact with cave walls
     torso_contact = info["torso_contact"]

     return out_of_bounds | feet_out_of_bounds | torso_contact | fall_termination #| no_movement

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
    info["heading_from_imu"] = info["heading_from_imu"] + noisy_gyro[2] * self._config.sim_dt

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
    info["distance_from_imu"] = info["distance_from_imu"] + jp.linalg.norm(noisy_linvel) * self._config.sim_dt

    # LIDAR data - update only at specified frequency to save compute
    should_update_lidar = (info["lidar_step_counter"] % self._lidar_update_interval_steps) == 0
    
    # Use jp.where instead of jax.lax.cond to avoid memory issues with lambda captures
    lidar_pos = self.get_lidar_pos(data)
    new_lidar_ranges = self._get_lidar_ranges(data, lidar_pos)
    new_deepest_lidar_direction = self._get_avg_deepest_lidar_range(
        new_lidar_ranges, self._local_ray_directions
    )
    
    # Only update cached data when needed, but compute is always done
    # This approach trades some computation for memory efficiency
    lidar_ranges = jp.where(should_update_lidar, new_lidar_ranges, info["lidar_ranges"])
    deepest_lidar_direction = jp.where(should_update_lidar, new_deepest_lidar_direction, info["deepest_lidar_direction"])
    
    # Update cached LIDAR data in info
    info["lidar_ranges"] = lidar_ranges
    info["deepest_lidar_direction"] = deepest_lidar_direction
    
    # Flatten LIDAR directions for network input (each direction is 3D)
    lidar_directions_flat = self._local_ray_directions.flatten()

    state = jp.hstack([
        noisy_linvel,  # 3
        noisy_gyro,  # 3
        noisy_gravity,  # 3
        noisy_joint_angles,  # 12
        noisy_joint_vel,  # 12
        info["last_act"],  # 12
        lidar_ranges,  # LIDAR ranges
        lidar_directions_flat,  # LIDAR ray directions (flattened)
        info["distance_from_imu"],  # 1
        info["heading_from_imu"],  # 1
        #info["deepest_lidar_direction"],  # 3 (direction of the average of top 3 deepest LIDAR ranges)
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
    
    # Get the IMU orientation (robot body orientation)
    imu_quat = mjx_env.get_sensor_data(self.current_mj_model, data, consts.ORIENTATION_SENSOR)
    # Convert quaternion to rotation matrix
    rot_mat = math.quat_to_mat(imu_quat)

    # Use precomputed local ray directions
    for ray_idx in range(total_lidar_rays):
        # Get local ray direction
        local_ray_dir = self._local_ray_directions[ray_idx]
        
        # Transform local direction to world coordinates using robot orientation
        world_ray_dir = rot_mat @ local_ray_dir

        geomgroup_mask = [True, False, False, False, False, False]
        # Cast ray from head position in world direction
        hit_dist, hit_geom_id = mjx.ray(self.mjx_model, data, head_pos, world_ray_dir, geomgroup=geomgroup_mask, bodyexclude=self._torso_body_id)

        # Clamp distance to max range, use max range if no hit
        current_range = jp.where(hit_dist >= 0.0, 
                                 jp.minimum(hit_dist, self._lidar_max_range), 
                                 self._lidar_max_range)
        ranges = ranges.at[ray_idx].set(current_range)
            
    return ranges

  def _at_min_wall_distance(self, lidar_ranges: jax.Array) -> jax.Array:
    """Check if the robot is at the minimum wall distance based on LIDAR ranges."""
    # Check if any LIDAR range is less than or equal to the minimum wall distance
    return jp.any(lidar_ranges <= 0.4)

  def _get_avg_deepest_lidar_range(self, lidar_ranges: jax.Array, norm_lidar_directions: jax.Array) -> jax.Array:
    """Compute the direction of the average of top 3 deepest LIDAR ranges in robot's local frame."""
    # Get indices of top 3 deepest (largest) ranges
    top_3_indices = jp.argsort(lidar_ranges)[-3:]  # Get indices of 3 largest values
    
    # Extract top 3 ranges and their corresponding directions
    top_3_ranges = lidar_ranges[top_3_indices]  # shape: (3,)
    top_3_directions = norm_lidar_directions[top_3_indices]  # shape: (3, 3)
    
    # Weight each direction by its corresponding range
    weighted_directions = top_3_ranges[:, None] * top_3_directions  # shape: (3, 3)
    
    # Sum the weighted directions from top 3
    sum_weighted_direction = jp.sum(weighted_directions, axis=0)  # shape: (3,)
    
    # Normalize to get unit direction vector
    norm = jp.linalg.norm(sum_weighted_direction)
    norm = jp.where(norm < 1e-6, 1e-6, norm)  # Avoid division by zero
    avg_deepest_direction = sum_weighted_direction / norm
    
    return avg_deepest_direction

  def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: Dict[str, Any],
        metrics: Dict[str, Any],
        done: jax.Array,
    ) -> Dict[str, jax.Array]:
        del metrics  # Unused.
        #jax.debug.print("CaveExplore step: {qpos}", qpos=data.qpos)
        stability = self._cost_stability(self.get_feet_pos(data))
        return {
            "stability": stability,
            "distance_from_start": self._cost_dist_from_start(
              jp.array(info["init_pos"]), data.qpos[0:3], info["last_pos"]
            ),
            "track_lidar_direction": self._reward_track_lidar_direction(
                jp.array(info["deepest_lidar_direction"]), self.get_local_linvel(data)
            ),
            "min_distance": self._cost_min_distance(info["lidar_ranges"]),
            "wide_stance": self._reward_wide_stance(self.get_feet_pos(data)),
            "exploration_rate": self._reward_exploration_rate(data.qpos[0:3]),
            "orientation": self._cost_orientation(self.get_upvector(data)),
            "termination": self._cost_termination(done),
            "torques": self._cost_torques(data.actuator_force),
            "action_rate": self._cost_action_rate(
                action, info["last_act"], info["last_last_act"]
            ),
            "energy": self._cost_energy(data.qvel[6:], data.actuator_force),
            "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[7:]),
            "inactivity": self._cost_inactivity(self.get_global_linvel(data))
        }

  # Base-related rewards.


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

  def _reward_track_lidar_direction(self, target_direction_norm: jax.Array, local_vel: jax.Array) -> jax.Array:
    """Reward for velocity alignment with the deepest LIDAR direction."""
    # target_direction_norm is already in local coordinates from _get_avg_deepest_lidar_range
    # local_vel is also in local coordinates from get_local_linvel
    
    # Normalize velocity vector
    vel_norm = jp.linalg.norm(local_vel)
    local_vel_normalized = jp.where(vel_norm < 1e-6, 
                                   jp.zeros_like(local_vel), 
                                   local_vel / vel_norm)
    
    # Compute alignment (dot product of normalized vectors)
    alignment = jp.dot(local_vel_normalized, target_direction_norm)
    
    # Scale by velocity magnitude for more nuanced reward
    vel_scale = jp.tanh(vel_norm / self._max_ms)  # Scale by how fast robot is moving
    reward = alignment * vel_scale

    return reward


  def _cost_min_distance(self, lidar_ranges: jax.Array) -> jax.Array:
    """Cost function for minimum distance to walls. Returns 0 at 0.4m, -1 at 0m."""
    min_distance = jp.min(lidar_ranges)
    
    # Linear interpolation: 0 cost at 0.4m, -1 cost at 0m
    # cost = (0.4 - min_distance) / 0.4
    # Clamp to ensure cost is between 0 and -1
    cost = jp.clip((0.4 - min_distance) / 0.4, 0.0, 1.0)
    return cost


  def _cost_dist_from_start(self, start_pos: jax.Array, current_pos: jax.Array, last_pos: jax.Array) -> jax.Array:
    """Penalty for moving closer to start position."""
    # Calculate change in distance from start
    last_dist = jp.linalg.norm(last_pos - start_pos)
    current_dist = jp.linalg.norm(current_pos - start_pos)
    
    # Positive when moving toward start (bad), negative when moving away (good)
    movement_toward_start = last_dist - current_dist
    
    # Normalize by max possible movement per step
    movement_toward_start_norm = movement_toward_start / (self._max_ms * self._config.sim_dt)
    
    # Only penalize movement toward start (clip negative values to 0)
    return jp.clip(movement_toward_start_norm, -0.2, 1.0)


  def _cost_stability(self, local_feet_pos: jax.Array) -> jax.Array:
    """Fast stability reward using simplified support polygon approximation."""
    
    # Extract x, y coordinates only
    feet_xy = local_feet_pos[:, :2]  # Shape: (n_feet, 2)

    # Fast approximation: use bounding box of feet as support region
    min_x = jp.min(feet_xy[:, 0])
    max_x = jp.max(feet_xy[:, 0])
    min_y = jp.min(feet_xy[:, 1])
    max_y = jp.max(feet_xy[:, 1])
    
    # Distance from COM to edges of bounding rectangle
    dist_to_edges = jp.array([
        - min_x,  # left edge
        max_x,  # right edge
        - min_y,  # bottom edge
        max_y    # top edge
    ])
    
    # Minimum distance to any edge (negative if outside)
    margin = jp.min(dist_to_edges)
    
    # Normalize and convert to reward [0, 1]
    characteristic_length = 0.2  # typical foot spacing
    normalized_margin = - margin / characteristic_length
    
    # Smooth reward function
    cost = jp.clip(jp.tanh((normalized_margin + 1) * 3.0), 0.0, 1.0)


    return cost 


  def _reward_exploration_rate(self, qpos: jax.Array) -> jax.Array:
    # Reward for exploration - could be enhanced with visited positions tracking
    # For now, return 0 but this could be expanded to reduce variance
    # by encouraging more consistent exploration behavior
    return 0.0
    
    # Future enhancement: track visited positions and reward novel areas
    # This would require adding visited_positions to the info dict and 
    # implementing a spatial hash or grid-based tracking system

  def _reward_wide_stance(self, local_feet_pos: jax.Array) -> jax.Array:
    """Reward for maintaining a wide stance for better stability.
    
    Args:
        local_feet_pos: Array of shape (n_feet, 3) with foot positions in local coordinates
        
    Returns:
        Reward value that increases with foot spread (wider stance = higher reward)
    """
    # Extract x, y coordinates only (ignore z for stance width calculation)
    feet_xy = local_feet_pos[:, :2]  # Shape: (n_feet, 2)
    
    # Calculate pairwise distances between all feet
    # This gives us the full spread of the stance
    distances = jp.linalg.norm(feet_xy[:, None] - feet_xy[None, :], axis=2)
    
    # Get unique distances by masking out diagonal and lower triangle
    # Create a mask for upper triangle (k=1 excludes diagonal)
    n_feet = distances.shape[0]
    i_indices = jp.arange(n_feet)[:, None]
    j_indices = jp.arange(n_feet)[None, :]
    upper_mask = j_indices > i_indices
    
    # Use jnp.where to extract only upper triangular distances
    # Pad with zeros where mask is False to maintain fixed size
    masked_distances = jp.where(upper_mask, distances, 0.0)
    
    # Sum all non-zero distances and count them
    total_distance = jp.sum(masked_distances)
    count_distances = jp.sum(upper_mask)
    
    # Calculate average distance (avoid division by zero)
    avg_distance = jp.where(count_distances > 0, total_distance / count_distances, 0.0)
    
    # Combine both measures: avg top 3 distances (overall width) and spread variance
    # Normalize by characteristic robot dimensions
    characteristic_length = 0.8  # typical maximum foot spacing for this robot
    normalized_avg_distance = avg_distance / characteristic_length
    # Weighted combination - prioritize average top distances but also reward even distribution
    stance_quality = normalized_avg_distance

    # Apply smooth saturation - reward plateaus for very wide stances to avoid over-extension
    # Using tanh to provide diminishing returns for extremely wide stances
    reward = jp.tanh(stance_quality * 2.0)  # Scale factor of 2.0 for good responsiveness
    
    return reward

  def _cost_joint_pos_limits(self, qpos: jax.Array) -> jax.Array:
    # Penalize joints if they cross soft limits.
    out_of_limits = -jp.clip(qpos - self._soft_lowers, None, 0.0)
    out_of_limits += jp.clip(qpos - self._soft_uppers, 0.0, None)
    return jp.sum(out_of_limits)

  # Feet related rewards.

  def _cost_inactivity(self, global_linvel: jax.Array) -> jax.Array:
    # Penalize inactivity - only movement in x-y counts
    xy_speed = jp.linalg.norm(global_linvel[:2])
    # Normalize speed by max speed and invert for cost (1 - speed/max_speed)
    normalized_speed = jp.clip(xy_speed / self._max_ms, 0.0, 1.0)
    return 1.0 - normalized_speed
  
  def _cost_3_points_stance(self, feet_in_contact: jax.Array) -> jax.Array:
    """Cost for minimum number of feet in contact with the ground."""
    # Penalize if fewer than 3 feet are in contact
    num_in_contact = jp.sum(feet_in_contact)
    return jp.clip(3 - num_in_contact, 0.0, 1.0)


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
    return "Dynamic cave environment"  # Since we don't have a single XML file anymore

  @property
  def action_size(self) -> int:
    """Size of the action space."""
    joints = self._scene_data["mj_model"].nu
    if self._config.stickiness_config.enable:
      return 4 + joints
    else:
      return joints

  @property
  def mj_model(self) -> MjModel:
    return self._scene_data["mj_model"]

  @property
  def mjx_model(self) -> mjx.Model:
    return self._scene_data["mjx_model"]

