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

from typing import Any, Dict, Optional, Union

from etils import epath
import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco import MjModel  # type: ignore
from mujoco.mjx._src import math
from models.model_loader import ReachbotModelType

from mujoco_playground._src import mjx_env
# Replace this with custom constants
import tasks.common.reachbot_constants as consts


def get_assets() -> Dict[str, bytes]:
  assets = {}
  mjx_env.update_assets(assets, consts.ROOT_PATH / "xmls", "*.xml")
  mjx_env.update_assets(assets, consts.ROOT_PATH / "xmls" / "assets")
  #path = mjx_env.MENAGERIE_PATH / "unitree_go1" # UNKNOWN effect at the moment
  return assets


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
            milestone_reward=0.1,  # Reward for reaching x-direction milestones
            
        ),
      ),
      pert_config=config_dict.create(
          enable=False,
          velocity_kick=[0.0, 3.0],
          kick_durations=[0.05, 0.2],
          kick_wait_times=[1.0, 3.0],
      ),
      stickiness_config=config_dict.create(
          enable=True,  # Enable stickiness forces
          stickiness_force=50.0,  # Force applied when stickiness is activated (towards wall when in contact)
          min_activation_threshold=0.1,  # Threshold for activating stickiness
          deactivation_threshold=-0.1,  # Threshold for deactivating stickiness (hysteresis)
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


class ReachbotEnv(mjx_env.MjxEnv):
  """Base class for Reachbot environments."""

  def __init__(
      self,
      xml_path: epath.Path,
      config: config_dict.ConfigDict,
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ) -> None:
    super().__init__(config, config_overrides)

    # Use from_xml_path to ensure <include file="..."/> statements are resolved relative to the XML file's directory.
    self._mj_model = MjModel.from_xml_path(
        xml_path, assets=get_assets()
    )
    self._mj_model.opt.timestep = self._config.sim_dt

    # Modify PD gains.
    self._mj_model.dof_damping[6:] = config.Kd_rot
    # Modify PD gains for prismatic joints.
    for i in [8, 11, 14, 17]:
      self._mj_model.dof_damping[i] = config.Kd_pri
    self._mj_model.actuator_gainprm[:, 0] = config.Kp_rot
    self._mj_model.actuator_biasprm[:, 1] = -config.Kp_rot
    for i in [2, 5, 8, 11]:
      self._mj_model.actuator_gainprm[i, 0] = config.Kp_pri
      self._mj_model.actuator_biasprm[i, 1] = -config.Kp_pri

    # Increase offscreen framebuffer size to render at higher resolutions.
    self._mj_model.vis.global_.offwidth = 3840
    self._mj_model.vis.global_.offheight = 2160

    self._mjx_model = mjx.put_model(self._mj_model)
    self._xml_path = xml_path
    self._imu_site_id = self._mj_model.site("imu").id

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
    self._max_dist_per_step = self._max_ms * self._config.sim_dt  # Maximum distance per step based on max speed

    self._init_q = jp.array(self._mj_model.keyframe("stand").qpos)
    self._stand_pose = jp.array(self._mj_model.keyframe("stand").qpos[7:])

    # Note: First joint is freejoint.
    self._lowers, self._uppers = self._mj_model.jnt_range[1:].T
    self._soft_lowers = self._lowers * self._config.soft_joint_pos_limit_factor
    self._soft_uppers = self._uppers * self._config.soft_joint_pos_limit_factor

    self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id
    self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]

    # Get torso geom ID for collision detection
    self._torso_geom_id = self._mj_model.geom("mainBody").id

    self._feet_site_id = jp.array(
        [self._mj_model.site(name).id for name in consts.FEET_SITES]
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

    # Pre-compute normalized local LIDAR ray directions
    self._precompute_lidar_directions()

    print("CaveExplore task action space:", self.action_size)


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

  def get_upvector(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.mj_model, data, consts.UPVECTOR_SENSOR)

  def get_gravity(self, data: mjx.Data) -> jax.Array:
    return data.site_xmat[self._imu_site_id].T @ jp.array([0, 0, -1])

  def get_global_linvel(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(
        self.mj_model, data, consts.GLOBAL_LINVEL_SENSOR
    )

  def get_global_angvel(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(
        self.mj_model, data, consts.GLOBAL_ANGVEL_SENSOR
    )

  def get_local_linvel(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(
        self.mj_model, data, consts.LOCAL_LINVEL_SENSOR
    )

  def get_accelerometer(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(
        self.mj_model, data, consts.ACCELEROMETER_SENSOR
    )

  def get_gyro(self, data: mjx.Data) -> jax.Array:
    return mjx_env.get_sensor_data(self.mj_model, data, consts.GYRO_SENSOR)

  def get_lidar_pos(self, data: mjx.Data) -> jax.Array: # Added for LIDAR
    return mjx_env.get_sensor_data(self.mj_model, data, consts.HEAD_POS_SENSOR) # Added for LIDAR
  
  def get_lidar_dirs(self, data: mjx.Data) -> jax.Array: # Added for LIDAR
     return self._local_ray_directions  # Return precomputed local ray directions
  
  
  def _get_lidar_ranges(self, data: mjx.Data, head_pos: jax.Array) -> jax.Array:
    """Casts rays from the robot's head in a 3D pattern and returns distances to nearest geoms."""
    total_lidar_rays = self._lidar_num_horizontal_rays * self._lidar_num_vertical_rays
    ranges = jp.full(total_lidar_rays, self._lidar_max_range, dtype=jp.float32)
    
    # Get the IMU orientation (robot body orientation)
    imu_quat = mjx_env.get_sensor_data(self._mj_model, data, consts.ORIENTATION_SENSOR)
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

  def get_feet_pos(self, data: mjx.Data) -> jax.Array:
    return jp.vstack([
        mjx_env.get_sensor_data(self.mj_model, data, sensor_name)
        for sensor_name in consts.FEET_POS_SENSOR
    ])

  # Accessors.

  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def action_size(self) -> int:
    """Size of the action space."""
    joints = self._mj_model.nu + 4 # Add 4 for grippers at boom ends

  @property
  def mj_model(self) -> MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model
