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
"""Fall recovery task for the Go1."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env

import reachbot.base as reachbot_base
import reachbot.reachbot_constants as consts


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.004, # Default: 0.004
      Kp_rot=35.0,
      Kd_rot=1,
      Kp_pri=200.0,
      Kd_pri=20.0,
      episode_length=1200,
      drop_from_height_prob=0.6,
      settle_time=0.5,
      action_repeat=1, 
      action_scale=0.5, # Scale the output of the policy
      soft_joint_pos_limit_factor=0.95,
      energy_termination_threshold=np.inf,
      noise_config=config_dict.create(
          level=1.0,
          scales=config_dict.create(
              joint_pos=0.03,
              joint_vel=1.5,
              gyro=0.2,
              gravity=0.05,
          ),
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              orientation=0.2,
              torso_height=2,
              posture=3,
              stand_still=5,
              action_rate=-0, # Default: -0.001
              dof_pos_limits=0,
              torques=0, # Default: -1e-5
              dof_acc=0, # Default: -2.5e-7
              dof_vel=-0, # Default: -0.1
              shoulder_torque=-0.1,
              vertical_velocity=-0,
          ),
      ),
  )


class Getup(reachbot_base.ReachbotEnv):
  """Recover from a fall and stand up.

  Observation space:
      - Gyroscope readings (3)
      - Gravity vector (3)
      - Joint angles (12)
      - Last action (12)

  Action space: Joint angles (12) scaled by a factor and added to the current
  joint angles. We tried using the same action space used in the joystick task
  where the output of the policy is added to the nominal "home" pose but it
  didn't work as well as adding to the current joint configuration. I suspect
  this is because the latter gives the policy a wider initial range of motion.

  Reward function:
      - Orientation: The torso should be upright.
      - Torso height: The torso should be at a desired height. This is to
          prevent the robot from flipping over and just lying on the ground.
      - Posture: The robot should be in the neural pose. This reward is only
          given when the robot is upright and at the desired height.
      - Stand still: Policy outputs should be zero once the robot is upright
          and at the desired height. This minimizes jittering.
      The next two rewards aren't really needed but promote better sim2real
          transfer (in theory):
      - Torques: Minimize joint torques.
      - Action rate: Minimize the first and second derivative of actions.
  """

  _deflection_model_used = False

  def __init__(
      self,
      task: str = "flat_terrain_deflection",
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(
        xml_path=consts.task_to_xml(task).as_posix(),
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init()
    if "deflection" in task:
      self._deflection_model_used = True
    print(f"Deflection model used: {self._deflection_model_used}")

  def _post_init(self) -> None:
    self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
    self._default_pose = jp.array(self._mj_model.keyframe("home").qpos[7:])

    self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
    c = (self._lowers + self._uppers) / 2
    r = self._uppers - self._lowers
    self._soft_lowers = c - 0.5 * r * self._config.soft_joint_pos_limit_factor
    self._soft_uppers = c + 0.5 * r * self._config.soft_joint_pos_limit_factor

    self._settle_steps = int(self._config.settle_time / self.sim_dt)
    self._z_des = 0.42
    self._up_vec = jp.array([0.0, 0.0, -1.0])
    self._imu_site_id = self._mj_model.site("imu").id

  def _qpos_to_ctrl(self, qpos: jax.Array) -> jax.Array:
    """Convert joint angles to control input format"""
    if self._deflection_model_used:
      return jp.concatenate([qpos[7:10], qpos[12:15], qpos[17:20], qpos[22:25]])  # Exclude the deflection joints
    else:
      return qpos[7:]
    
  def _qvel_to_ctrl(self, qvel: jax.Array) -> jax.Array:
    """Convert joint velocities to control input format"""
    if self._deflection_model_used:
      return jp.concatenate([qvel[6:9], qvel[11:14], qvel[16:19], qvel[21:24]])  # Exclude the deflection joints
    else:
      return qvel[6:]

  def _get_random_qpos(self, rng: jax.Array) -> jax.Array:
    """Generate an initial configuration where the robot is at a height of 0.5m
    with a random orientation and joint angles.

    Note(kevin): We could also randomize the root height but experiments on
    real hardware show that this works just fine.
    """
    rng, orientation_rng, qpos_rng = jax.random.split(rng, 3)

    qpos = jp.zeros(self.mjx_model.nq)

    # Initialize height and orientation of the root body.
    height = 0.5
    qpos = qpos.at[2].set(height)
    quat = jax.random.normal(orientation_rng, (4,))
    quat /= jp.linalg.norm(quat) + 1e-6
    qpos = qpos.at[3:7].set(quat)

    # Randomize joint angles.
    num_joints = self.mjx_model.nq - 7
    qpos = qpos.at[7:].set(
      jax.random.uniform(
        qpos_rng, (num_joints,), minval=self._lowers, maxval=self._uppers
      )
    )
    if(self._deflection_model_used):
      #Set deflection joints to 0 = no deflection
      qpos = qpos.at[11].set(0)
      qpos = qpos.at[16].set(0)
      qpos = qpos.at[21].set(0)
      qpos = qpos.at[26].set(0)
      #Set second prismatic to mimic the first
      qpos = qpos.at[12].set(qpos[10])
      qpos = qpos.at[17].set(qpos[15])
      qpos = qpos.at[22].set(qpos[20])
      qpos = qpos.at[27].set(qpos[25])
    
    return qpos

  def reset(self, rng: jax.Array) -> mjx_env.State:
    # Sample a random initial configuration with some probability.
    #rng = jax.random.PRNGKey(0) if rng.ndim > 1 else rng
    rng, key1, key2 = jax.random.split(rng, 3)
    qpos = jp.where(
        jax.random.bernoulli(key1, self._config.drop_from_height_prob),
        self._get_random_qpos(key2),
        self._init_q,
    )

    # Sample a random root velocity.
    rng, key = jax.random.split(rng)
    qvel = jp.zeros(self.mjx_model.nv)
    qvel = qvel.at[0:6].set(
        jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5)
    )

    ctrl = self._qpos_to_ctrl(qpos)
    data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=ctrl)

    # Let the robot settle for a few steps.
    data = mjx_env.step(self.mjx_model, data, ctrl, self._settle_steps)
    data = data.replace(time=0.0)

    info = {
        "rng": rng,
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
    }

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())

    obs = self._get_obs(data, info)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    state_in_ctrl_format = self._qpos_to_ctrl(state.data.qpos)
    motor_targets = state_in_ctrl_format + action * self._config.action_scale
    data = mjx_env.step(
        self.mjx_model, state.data, motor_targets, self.n_substeps
    )

    obs = self._get_obs(data, state.info)
    done = self._get_termination(data)

    rewards = self._get_reward(data, action, state.info, state.metrics, done)
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

    # Bookkeeping.
    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v

    done = jp.float32(done)
    state = state.replace(data=data, obs=obs, reward=reward, done=done)
    return state

  def _get_termination(self, data: mjx.Data) -> jax.Array:
    qvel_formatted = self._qvel_to_ctrl(data.qvel)
    energy = jp.sum(jp.abs(data.actuator_force * qvel_formatted))
    energy_termination = energy > self._config.energy_termination_threshold
    return energy_termination

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

    state = jp.concatenate([
        noisy_gyro,  # 3
        noisy_gravity,  # 3
        noisy_joint_angles - self._default_pose,  # 12
        noisy_joint_vel,  # 12
        info["last_act"],  # 12
        jp.array([self._z_des]),  
    ])

    accelerometer = self.get_accelerometer(data)
    linvel = self.get_local_linvel(data)
    angvel = self.get_global_angvel(data)
    torso_height = data.site_xpos[self._imu_site_id][2]

    privileged_state = jp.hstack([
        state,
        gyro,
        accelerometer,
        linvel,
        angvel,
        joint_angles,
        joint_vel,
        data.actuator_force,
        torso_height,
    ])

    return {
        "state": state,
        "privileged_state": privileged_state,
    }

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
      done: jax.Array,
  ) -> dict[str, jax.Array]:
    del done, metrics  # Unused.

    torso_height = data.site_xpos[self._imu_site_id][2]
    joint_angles = data.qpos[7:]
    joint_torques = data.actuator_force
    gravity = self.get_gravity(data)
    is_upright = self._is_upright(gravity)
    is_at_desired_height = self._is_at_desired_height(torso_height)
    gate = is_upright * is_at_desired_height
    height_weight = is_upright * self._height_weight(torso_height)
    rewards = {
      "orientation": self._reward_orientation(gravity, gate),
      "torso_height": self._reward_height(torso_height),
      "posture": self._reward_posture(joint_angles, is_upright),
      "stand_still": self._reward_stand_still(action, gate),
      "action_rate": self._cost_action_rate(action, info),
      "torques": self._cost_torques(joint_torques),
      "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[7:]),
      "dof_acc": self._cost_dof_acc(data.qacc[6:]),
      "dof_vel": self._cost_dof_vel(data.qvel[6:]),
      "shoulder_torque": self._cost_shoulder_torque(joint_torques, gate),
      "vertical_velocity": self._cost_vertical_velocity(data),
    }
    return rewards
    # Sanitize each reward component
    #safe_rewards = {k: safe_reward(k,v) for k, v in rewards.items()}
    #return safe_rewards

  # Check if the robot is upright by comparing the gravity vector to the up vector.
  def _is_upright(self, gravity: jax.Array, ori_tol: float = 0.01) -> jax.Array:
    ori_error = jp.sum(jp.square(self._up_vec - gravity))
    return ori_error < ori_tol

  # Check if the torso height is at the desired height within a tolerance.
  def _is_at_desired_height(
      self, torso_height: jax.Array, pos_tol: float = 0.05
  ) -> jax.Array:
    height_error = jp.abs(self._z_des - torso_height)
    return height_error < pos_tol

  # Reward for maintaining the correct orientation.
  def _reward_height(self, torso_height: jax.Array) -> jax.Array:
    height = jp.min(jp.array([torso_height, self._z_des]))
    reward = jp.exp(height) - 1.0
    #reward = height
    return reward

  def _reward_orientation(self, up_vec: jax.Array, gate: jax.Array) -> jax.Array:
    error = jp.sum(jp.square(self._up_vec - up_vec))
    reward = jp.exp(-2.0 * error)
    return reward


  # Reward for maintaining the default posture when upright and at the desired height.
  def _reward_posture(self, joint_angles: jax.Array, gate: jax.Array) -> jax.Array:
      cost = jp.sum(jp.square(joint_angles - self._default_pose))
      reward = jp.exp(-0.5 * cost)
      return reward * gate

  # Reward for minimizing action outputs when upright and at the desired height.
  def _reward_stand_still(self, act: jax.Array, gate: jax.Array) -> jax.Array:
    cost = jp.sum(jp.square(act))
    reward = jp.exp(-0.5 * cost)
    return reward * gate

  # Cost for high torques.
  def _cost_torques(self, torques: jax.Array) -> jax.Array:
    return jp.sqrt(jp.sum(jp.square(torques))) + jp.sum(jp.abs(torques))

  # Cost for high action rate (first and second derivatives of actions).
  def _cost_action_rate(
      self, act: jax.Array, info: dict[str, Any]
  ) -> jax.Array:
    c1 = jp.sum(jp.square(act - info["last_act"]))
    c2 = jp.sum(jp.square(act - 2 * info["last_act"] + info["last_last_act"]))
    return c1 + c2

  # Cost for joint positions outside soft limits.
  def _cost_joint_pos_limits(self, qpos: jax.Array) -> jax.Array:
    out_of_limits = -jp.clip(qpos - self._soft_lowers, None, 0.0)
    out_of_limits += jp.clip(qpos - self._soft_uppers, 0.0, None)
    return jp.sum(out_of_limits)

  # Cost for high degrees of freedom velocities.
  def _cost_dof_vel(self, qvel: jax.Array) -> jax.Array:
    max_velocity = 2.0 * jp.pi  # rad/s
    cost = jp.maximum(jp.abs(qvel) - max_velocity, 0.0)
    return jp.sum(jp.square(cost))

  # Cost for high degrees of freedom accelerations.
  def _cost_dof_acc(self, qacc: jax.Array) -> jax.Array:
    return jp.sum(jp.square(qacc))
  
  def _cost_shoulder_torque(self, torques: jax.Array, gate: jax.Array) -> jax.Array:
    """Returns a penalty for high torques at the shoulder joints."""
    shoulder_torques = jp.array([torques[0], torques[3], torques[6], torques[9]]) / 4
    return jp.sum(jp.abs(shoulder_torques)) * gate
  
  def _height_weight(self, torso_height: jax.Array) -> jax.Array:
    # A Gaussian weighting centered at self._z_des.
    sigma = 0.1  # adjust sigma to widen/narrow the band
    error = torso_height - self._z_des
    weight = jp.exp(-0.5 * jp.square(error / sigma))
    return weight
  
  def _cost_vertical_velocity(self, data: mjx.Data) -> jax.Array:
    # Assuming data.qvel[2] is the vertical (z) velocity.
    z_vel = data.qvel[2]
    return jp.square(z_vel)

  
def safe_reward(key, reward_component):
  return jp.nan_to_num(reward_component, nan=0.0, posinf=100000.0, neginf=0.0)