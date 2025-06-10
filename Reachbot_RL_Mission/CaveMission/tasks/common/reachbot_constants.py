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
"""Defines Reachbot constants."""

from etils import epath

from mujoco_playground._src import mjx_env

#ROOT_PATH = mjx_env.ROOT_PATH / "reachbot"
from etils import epath
ROOT_PATH = mjx_env.ROOT_PATH / "xmls" / "scenes"

STANDARD_XML = (ROOT_PATH / "scene_basic.xml")
FEET_ONLY_XML = (
    ROOT_PATH / "scene_feetonly.xml"
)
DEFLECTION_XML = (
    ROOT_PATH / "scene_reachbot_deflection.xml"
)
BASIC_FRICTION_XML = (
    ROOT_PATH  / "scene_reachbot_basic_friction.xml"
)
DEFLECTION_FRICTION_XML = (
    ROOT_PATH  / "scene_reachbot_deflection_friction.xml"
)


def task_to_xml(task_name: str) -> epath.Path:
  print("task name: " + task_name)
  return {
      "standard": STANDARD_XML,
      "reachbot_basic_feet_only": FEET_ONLY_XML,
      "reachbot_deflection": DEFLECTION_XML,
      "reachbot_friction_basic": BASIC_FRICTION_XML,
      "reachbot_friction_deflection": DEFLECTION_FRICTION_XML,
  }[task_name]

FEET_SITES = [
    "boomEndSite1",
    "boomEndSite2",
    "boomEndSite3",
    "boomEndSite4",
]

FEET_GEOMS = [
    "boomEnd1",
    "boomEnd2",
    "boomEnd3",
    "boomEnd4",
]


FEET_POS_SENSOR = [f"{site}_pos" for site in FEET_SITES]

ROOT_BODY = "mainBody"

UPVECTOR_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "accelerometer"
GYRO_SENSOR = "gyro"
HEAD_POS_SENSOR = "head_pos" # Added for LIDAR
ORIENTATION_SENSOR = "orientation" # Added for LIDAR
