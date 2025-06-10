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

ROOT_PATH = epath.Path("reachbot")
FEET_ONLY_FLAT_TERRAIN_XML = (
    ROOT_PATH / "xmls" / "scene_mjx_feetonly_flat_terrain.xml"
)
FEET_ONLY_ROUGH_TERRAIN_XML = (
    ROOT_PATH / "xmls" / "scene_mjx_feetonly_rough_terrain.xml"
)
BASIC_FLAT_TERRAIN_XML = ROOT_PATH / "xmls" / "scene_mjx_basic_flat_terrain.xml"
DEFLECTION_FLAT_TERRAIN_XML = ROOT_PATH / "xmls" / "scene_mjx_deflection_flat_terrain.xml"
FULL_COLLISIONS_FLAT_TERRAIN_XML = (
    ROOT_PATH / "xmls" / "scene_mjx_fullcollisions_flat_terrain.xml"
)
BASIC_ROUGH_TERRAIN_XML = ( ROOT_PATH / "xmls" / "scene_mjx_basic_rough_terrain.xml" )


def task_to_xml(task_name: str) -> epath.Path:
  print("task name: " + task_name)
  return {
      "flat_terrain_basic": BASIC_FLAT_TERRAIN_XML,
      "flat_terrain_deflection": DEFLECTION_FLAT_TERRAIN_XML,
      "rough_terrain_basic": BASIC_ROUGH_TERRAIN_XML,
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

#FEET_SITES = [
#    "FR",
#    "FL",
#    "RR",
#    "RL",
#]

#FEET_GEOMS = [
#    "FR",
#    "FL",
#    "RR",
#    "RL",
#]

FEET_POS_SENSOR = [f"{site}_pos" for site in FEET_SITES]

ROOT_BODY = "mainBody"

UPVECTOR_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "accelerometer"
GYRO_SENSOR = "gyro"
