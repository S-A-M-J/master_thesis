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

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import collision
from mujoco_playground._src import mjx_env
import tasks.common.base as reachbot_base
import tasks.common.reachbot_constants as consts


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.004,
      episode_length=1000,
      Kp_rot=35.0,
      Kd_rot=1,
      Kp_pri=200.0,
      Kd_pri=20.0,
      action_repeat=1,
      action_scale=0.5,
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
            
            # Other rewards
            dof_pos_limits=-1.0,        # Penalty scale for degree of freedom position limits. Default is -1.0
            pose=0.0,                   # Reward scale for maintaining a specific pose. Default is 0.5
            
            # Termination and stand-still penalties
            termination=-1.0,           # Penalty scale for termination conditions. Default is -1.0
            stand_still=-1.0,           # Penalty scale for standing still. Default is -1.0
            
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
          stickiness_force=100.0,  # Force applied when stickiness is activated
      )
  )


class CaveExplore(reachbot_base.ReachbotEnv):
  """Explore the cave environment."""

  _deflection_model_used = False
  observation_size = 0  # Will be set during initialization
  privileged_state_size = 0  # Will be set during initialization
  action_size = 0  # Will be set during initialization
  num_booms = 4  # Default number of booms

  def __init__(
      self,
      task: str = "reachbot_basic",
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
      # LIDAR parameters
      lidar_num_horizontal_rays: int = 10,  # Renamed from lidar_num_rays
      lidar_max_range: float = 10.0,
      lidar_horizontal_angle_range: float = jp.pi * 2,  # 360 degrees by default
      lidar_num_vertical_rays: int = 5,  # New parameter for vertical rays
      lidar_vertical_angle_range: float = jp.pi / 6,  # New parameter for vertical angle range (30 degrees)
      # Cave parameters
      cave_width: float = 1.0,
      cave_height: float = 1.0,
      cave_length: float = 1.0,
      target_pos : jax.Array = jp.array([0.0, 0.0, 0.0]), # x, y, z coordinates of the target position in worldspace
      voxel_positions: jax.Array = jp.zeros((0, 3), dtype=jp.float32), # x, y, z coordinates of multiple voxel positions in worldspace
      action_size: int = 12,  # Optional explicit action size
  ):
    super().__init__(
        xml_path=consts.task_to_xml(task).as_posix(),
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init()
    if "deflection" in task:
      self._deflection_model_used = True
    
    # Calculate observation sizes after post_init
    self._calculate_observation_sizes()
    
    # LIDAR parameters
    self._lidar_num_horizontal_rays = lidar_num_horizontal_rays
    self._lidar_num_vertical_rays = lidar_num_vertical_rays
    self._lidar_max_range = lidar_max_range
    self._lidar_horizontal_angle_range = lidar_horizontal_angle_range
    self._lidar_vertical_angle_range = lidar_vertical_angle_range
    self._head_pos_sensor_id = self._mj_model.sensor(consts.HEAD_POS_SENSOR).id

    # Cave parameters
    self._cave_width = cave_width
    self._cave_height = cave_height
    self._cave_length = cave_length
    self._target_pos = target_pos
    if (voxel_positions is not None):
      self._voxel_positions = voxel_positions
    else: 
      self._voxel_positions = voxel_positions
      
    # Calculate observation and action sizes
    self.action_size = action_size if action_size is not None else self.mjx_model.nu
    
    # We'll calculate the actual observation size after post_init
    # when we have all the required variables

  def _post_init(self) -> None:
    self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
    self._default_pose = jp.array(self._mj_model.keyframe("home").qpos[7:])

    self._z_des = 0.42

    # Note: First joint is freejoint.
    self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
    self._soft_lowers = self._lowers * self._config.soft_joint_pos_limit_factor
    self._soft_uppers = self._uppers * self._config.soft_joint_pos_limit_factor

    self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id
    self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]

    self._feet_site_id = np.array(
        [self._mj_model.site(name).id for name in consts.FEET_SITES]
    )
    self._floor_geom_id = self._mj_model.geom("floor").id
    self._feet_geom_id = np.array(
        [self._mj_model.geom(name).id for name in consts.FEET_GEOMS]
    )

    foot_linvel_sensor_adr = []
    for site in consts.FEET_SITES:
      sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id
      sensor_adr = self._mj_model.sensor_adr[sensor_id]
      sensor_dim = self._mj_model.sensor_dim[sensor_id]
      foot_linvel_sensor_adr.append(
          list(range(sensor_adr, sensor_adr + sensor_dim))
      )
    self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)

    self._imu_site_id = self._mj_model.site("imu").id


  def _qpos_to_motor_ctrl(self, qpos: jax.Array) -> jax.Array:
    """Convert joint angles to control input format"""
    if self._deflection_model_used:
      return jp.concatenate([qpos[7:10], qpos[12:15], qpos[17:20], qpos[22:25]])  # Exclude the deflection joints
    else:
      return qpos[7:-4]
    
  def _qpos_to_stickiness_ctrl(self, qpos: jax.Array) -> jax.Array:
    """Convert joint angles to stickiness control input format"""
    if self._deflection_model_used:
      return jp.concatenate([qpos[10:12], qpos[15:17], qpos[20:22], qpos[25:27]])  # Exclude the deflection joints
    else:
      return qpos[-4:] 

  def reset(self, rng: jax.Array) -> mjx_env.State:
    qpos = self._init_q
    qvel = jp.zeros(self.mjx_model.nv)

    # x=+U(-0.5, 0.5), y=+U(-0.5, 0.5), yaw=U(-3.14, 3.14).
    rng, key = jax.random.split(rng)
    dxy = jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)
    qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
    rng, key = jax.random.split(rng)
    yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
    quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
    new_quat = math.quat_mul(qpos[3:7], quat)
    qpos = qpos.at[3:7].set(new_quat)

    # d(xyzrpy)=U(-0.5, 0.5)
    rng, key = jax.random.split(rng)
    qvel = qvel.at[0:6].set(
        jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5)
    )
    ctrl = self._qpos_to_motor_ctrl(qpos)
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
    time_until_next_cmd = jax.random.exponential(key1) * 5.0
  

    info = {
        "rng": rng,
        "target_pos": self._target_pos,
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
        "steps_until_next_pert": steps_until_next_pert,
        "pert_duration_seconds": pert_duration_seconds,
        "pert_duration": pert_duration_steps,
        "steps_since_last_pert": 0,
        "pert_steps": 0,
        "pert_dir": jp.zeros(3),
        "pert_mag": pert_mag,
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
    # state = self._reset_if_outside_bounds(state)
    state_formatted = self._qpos_to_motor_ctrl(self._init_q)
    stickiness_ctrl = self._qpos_to_stickiness_ctrl(self._init_q)
    motor_targets = state_formatted + action * self._config.action_scale
    contact = jp.array([
            collision.geoms_colliding(state.data, geom_id, self._floor_geom_id)
            for geom_id in self._feet_geom_id
        ])
    
    if self._config.stickiness_config.enable:
      # Apply stickiness force if enabled
      xfrc_applied = self._apply_stickiness_forces(state, stickiness_ctrl)
      state = state.replace(data=state.data.replace(xfrc_applied=xfrc_applied))


    data = mjx_env.step(
        self.mjx_model, state.data, motor_targets, self.n_substeps
    )

    contact_filt = contact | state.info["last_contact"]
    p_f = data.site_xpos[self._feet_site_id]
    p_fz = p_f[..., -1]

    obs = self._get_obs(data, state.info)
    done = self._get_termination(data)

    rewards = self._get_reward(
        data, action, state.info, state.metrics, done, contact
    )
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v

    done = done.astype(reward.dtype)
    state = state.replace(data=data, obs=obs, reward=reward, done=done)
    return state
  
  def _apply_stickiness_forces(self, state: mjx_env.State, stickiness_ctrl: jax.Array) -> jax.Array:
    """Applies a stickiness force to the robot when a boom end touches a cave wall and 
    the corresponding stickiness control from the neural network is activated"""
    
    # Initialize external forces array
    xfrc_applied = jp.zeros((self.mjx_model.nbody, 6))
    
    # Check contacts for each boom
    boom_idxs = []
    wall_idxs = []
    
    # Find contacts between boom ends and cave walls
    for i, contact in enumerate(state.data.contact):
      geom1_id = contact.geom1
      geom2_id = contact.geom2
      
      # Get geom names
      geom1_name = self.mjx_model.geom_id2name[geom1_id] if geom1_id < len(self.mjx_model.geom_id2name) else ""
      geom2_name = self.mjx_model.geom_id2name[geom2_id] if geom2_id < len(self.mjx_model.geom_id2name) else ""
      
      # Check if one object is a boom end and the other is a cave wall
      if "boomEnd" in geom1_name and "cave_wall" in geom2_name:
        boom_idxs.append(geom1_id)
        wall_idxs.append(geom2_id)
      elif "boomEnd" in geom2_name and "cave_wall" in geom1_name:
        boom_idxs.append(geom2_id)
        wall_idxs.append(geom1_id)
    
    # Apply stickiness forces for each contact where stickiness is activated
    for i, (boom_id, wall_id) in enumerate(zip(boom_idxs, wall_idxs)):
      # Get the boom number from its name to index into stickiness_signals
      boom_name = self.mjx_model.geom_id2name[boom_id]
      boom_num = int(boom_name.split('boomEnd')[1]) if len(boom_name.split('boomEnd')) > 1 else i % self.num_booms
      
      # Check if stickiness is activated for this boom (threshold > 0.5)
      if boom_num < len(stickiness_ctrl) and stickiness_ctrl[boom_num] > 0.5:
        # Get positions
        boom_pos = state.data.geom_xpos[boom_id]
        wall_pos = state.data.geom_xpos[wall_id]
        
        # Calculate direction and distance
        direction_vector = wall_pos - boom_pos
        distance = jp.linalg.norm(direction_vector)
        normalized_direction = direction_vector / (distance + 1e-6)
        
        # Apply force to the body containing this boom end
        boom_body_id = self.mjx_model.geom_bodyid[boom_id]
        force_magnitude = 100.0  # Adjust this value to control stickiness strength
        
        # Add force to the existing forces
        current_forces = xfrc_applied[boom_body_id]
        new_forces = current_forces.at[:3].add(force_magnitude * normalized_direction)
        xfrc_applied = xfrc_applied.at[boom_body_id].set(new_forces)
    
    return xfrc_applied
  
  def _get_termination(self, state: mjx_env.State) -> jax.array:
     qpos = state.data.qpos
     out_of_bounds = jp.abs(qpos[0]) > 9.5 or jp.abs(qpos[1]) > 9.5 or jp.abs(qpos[2]) > 0.5 # Check if x, y, or z position is out of bounds
     has_fallen = self.get_upvector(state.data)[-1] < 0.0 # Check if the robot has fallen
     return out_of_bounds or has_fallen # Return True if either condition is met


  def _get_obs(
      self, data: mjx.Data, info: dict[str, Any]
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
        noisy_joint_angles - self._default_pose,  # 12
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
        info["feet_air_time"],  # 4
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
    
    imu_quat = mjx_env.get_sensor_data(self.mj_model, data, consts.ORIENTATION_SENSOR)

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
            
            world_ray_dir = math.quat_rotate(imu_quat, local_ray_dir)
            
            norm = jp.linalg.norm(world_ray_dir)
            world_ray_dir = world_ray_dir / (norm + 1e-6)

            hit_dist = mjx.ray(self.mjx_model, data, head_pos, world_ray_dir, 
                               None, True, self._torso_body_id, -1)

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
      info: dict[str, Any],
      metrics: dict[str, Any],
      done: jax.Array,
      contact: jax.Array,
  ) -> dict[str, jax.Array]:
    del metrics  # Unused.
    return {
        "distance_to_target": self._reward_target_distance(
            data.qpos[0:3], info["target_pos"]
        ),
        "exploration_rate": self._reward_exploration_rate(data.qpos[0:3]),
        "lin_vel_z": self._cost_lin_vel_z(self.get_global_linvel(data)),
        "ang_vel_xy": self._cost_ang_vel_xy(self.get_global_angvel(data)),
        "orientation": self._cost_orientation(self.get_upvector(data)),
        "stand_still": self._cost_stand_still(data.qvel[6:]),
        "termination": self._cost_termination(done),
        "torques": self._cost_torques(data.actuator_force),
        "action_rate": self._cost_action_rate(
            action, info["last_act"], info["last_last_act"]
        ),
        "energy": self._cost_energy(data.qvel[6:], data.actuator_force),
        "feet_slip": self._cost_feet_slip(data, contact, info),
        "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[7:]),
        "boom_extension_speed": self._cost_boom_extension_speed(data.qvel[6:]),
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

  def _cost_stand_still(
      self,
      qvel: jax.Array,
  ) -> jax.Array:
    
    return jp.sum(jp.square(qvel)) * self._config.reward_config.scales.stand_still

  def _cost_termination(self, done: jax.Array) -> jax.Array:
    # Penalize early termination.
    return done
  
  def _reward_target_distance(self, qpos: jax.Array, target_pos: jax.Array) -> jax.Array:
    # Reward for distance to target.
    return jp.linalg.norm(qpos - target_pos)
  
  def _reward_exploration_rate(self, qpos: jax.Array) -> jax.Array:
    # Reward for exploration rate.
    return 0 # Placeholder for exploration rate reward, can be modified later.
    return jp.linalg.norm(qpos - self._target_pos) / (self._cave_width * self._cave_height * self._cave_length)

  def _cost_joint_pos_limits(self, qpos: jax.Array) -> jax.Array:
    # Penalize joints if they cross soft limits.
    out_of_limits = -jp.clip(qpos - self._soft_lowers, None, 0.0)
    out_of_limits += jp.clip(qpos - self._soft_uppers, 0.0, None)
    return jp.sum(out_of_limits)

  # Feet related rewards.

  def _cost_feet_slip(
      self, data: mjx.Data, contact: jax.Array, info: dict[str, Any]
  ) -> jax.Array:
    cmd_norm = jp.linalg.norm(info["command"])
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
    vel_xy = feet_vel[..., :2]
    vel_xy_norm_sq = jp.sum(jp.square(vel_xy), axis=-1)
    return jp.sum(vel_xy_norm_sq * contact) * (cmd_norm > 0.01)


  # Perturbation and command sampling.

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
  
  
  def _cost_boom_extension_speed(self, qvel: jax.Array) -> jax.Array:
    """Returns a penalty value if the boom extension speed is too high."""
    # Penalize the boom extension speed.
    boom_indices = jp.array([9, 12, 15, 18])
    boom_extension_speed = jp.sum(jp.abs(qvel[boom_indices]))
    return boom_extension_speed

  def _calculate_observation_sizes(self):
    """Calculate the sizes of observation and privileged state arrays"""
    # Basic components size calculation
    linvel_size = 3  # Local linear velocity
    gyro_size = 3    # Gyroscope data
    gravity_size = 3  # Gravity direction
    
    # Joint information
    joint_angles_size = len(self._default_pose)  # Joint angles
    joint_vel_size = self.mjx_model.nv - 6  # Joint velocities (excluding root)
    
    # Last action size
    last_act_size = self.mjx_model.nu
    
    # LIDAR data size
    lidar_size = self._lidar_num_horizontal_rays * self._lidar_num_vertical_rays
    
    # Calculate state size (regular observation)
    self.observation_size = (
        linvel_size + 
        gyro_size + 
        gravity_size + 
        joint_angles_size + 
        joint_vel_size + 
        last_act_size + 
        lidar_size
    )
    
    # Additional privileged state components
    accelerometer_size = 3
    additional_gyro_size = 3
    additional_gravity_size = 3
    additional_linvel_size = 3
    angvel_size = 3
    additional_joint_vel_size = joint_vel_size
    actuator_force_size = self.mjx_model.nu
    feet_contact_size = len(self._feet_geom_id)
    feet_vel_size = len(self._feet_geom_id) * 3
    feet_air_time_size = len(self._feet_geom_id)
    xfrc_applied_size = 3  # Only using the force part, not torque
    pert_indicator_size = 1
    
    # Calculate privileged state size
    self.privileged_state_size = (
        self.observation_size +  # Regular observation
        additional_gyro_size + 
        accelerometer_size + 
        additional_gravity_size + 
        additional_linvel_size + 
        angvel_size + 
        additional_joint_vel_size + 
        actuator_force_size +
        feet_contact_size + 
        feet_vel_size + 
        feet_air_time_size + 
        xfrc_applied_size + 
        pert_indicator_size
    )
