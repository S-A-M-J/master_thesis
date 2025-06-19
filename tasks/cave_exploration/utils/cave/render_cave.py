import os
import sys
import mujoco
import mujoco.viewer
import argparse
import numpy as np

# Add the project root and cave exploration directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
cave_exploration_dir = os.path.join(current_dir, '..', '..')
project_root = os.path.join(cave_exploration_dir, '..', '..')
sys.path.append(project_root)
sys.path.append(cave_exploration_dir)

from environment.env_loader import CaveBatchLoader
from cave_exploration import default_config as reachbot_config

def render_mesh( scale=1.0):
    """
    Render a mesh STL file in MuJoCo.
    """
    xml_path = "xmls/scene_reachbot_cave.xml"
    
    # Load model and create data
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    mujoco.mj_resetDataKeyframe(model, data,  0)

    # Launch viewer
    with mujoco.viewer.launch(model, data) as viewer:
        # Initial camera setup
        viewer.cam.lookat[:] = [0, 0, 0]  # Looking at the center
        viewer.cam.distance = 20.0  # Adjusted distance
        viewer.cam.azimuth = 122.8  # Based on xyaxes rotation
        viewer.cam.elevation = -40.3  # Based on xyaxes elevation
        # Position approximately matches pos="-12.870 -7.169 14.348"
        # Orientation approximately matches xyaxes="0.542 -0.841 0.000 0.532 0.343 0.774"

        # Render loop
        while viewer.is_running():
            viewer.sync()
            mujoco.mj_step(model, data)

        

    


def main():
    print("MuJoCo Mesh Viewer")
    config = reachbot_config()  # Call the function to get the config object
    envs = CaveBatchLoader(3, config)
    rng = np.random.randint(3)  # Initialize random number generator
    print(f"Rendering environment {envs.envs[rng]['folder']}")
    model = envs.envs[rng]["mj_model"]
    initial_qpos = envs.envs[rng]["initial_qpos"]
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[:] = initial_qpos  # Set the initial position
    with mujoco.viewer.launch(model, data) as viewer:
        # Initial camera setup
        viewer.cam.lookat[:] = [0, 0, 0]  # Looking at the center
        viewer.cam.distance = 30.0  # Adjusted distance
        viewer.cam.azimuth = 122.8  # Based on xyaxes rotation
        viewer.cam.elevation = -40.3  # Based on xyaxes elevation
        # Position approximately matches pos="-12.870 -7.169 14.348"
        # Orientation approximately matches xyaxes="0.542 -0.841 0.000 0.532 0.343 0.774"

        # Render loop
        while viewer.is_running():
            viewer.sync()
            mujoco.mj_step(model, data)

if __name__ == "__main__":
    main()