import os
import random
import glob
import json
from mujoco import mjx, MjModel  # Make sure mujoco's mjx Python bindings are installed
from models.model_loader import ReachbotModelType, ReachbotModel

CAVES_DIR = "/home/sam/master_thesis/tasks/cave_exploration/environment/caves"

class CaveBatchLoader:
    def __init__(self, batch_size, config, reachbot_model_type=ReachbotModelType.BASIC):
        """
        root_dir: Directory containing folders for each cave environment.
        batch_size: Number of environments to load per epoch.
        reachbot_model_type: Type of Reachbot model to use (BASIC or DEFLECTION).

        Load a batch of cave environments for exploration tasks.
        """
        self.batch_size = batch_size
        self.reachbot_model = ReachbotModel(reachbot_model_type)
        self.envs = []  # List of dicts: {mjx_model, json_data, target_pos, voxel_positions, folder}

        # Folders directly under root_dir
        cave_folders = [os.path.join(CAVES_DIR, d) for d in os.listdir(CAVES_DIR)
                if os.path.isdir(os.path.join(CAVES_DIR, d))]

        print(f"Found {len(cave_folders)} cave folders in {CAVES_DIR}.")
        chosen_folders = random.sample(cave_folders, min(self.batch_size, len(cave_folders)))
        print(f"Loading {len(chosen_folders)} cave environments.")

        # Load the scene_reachbot_cave.xml template
        template_path = "/home/sam/master_thesis/tasks/cave_exploration/environment/scene_reachbot_cave.xml"
        with open(template_path, "r") as f:
            scene_template = f.read()
        for folder in chosen_folders:
            # Prepare paths
            cave_boxes_path = os.path.join(folder, "cave_tunnel_boxes.xml")
            # Replace placeholders in template
            scene_xml = scene_template.replace("{REACHBOT_MODEL_PATH}", self.reachbot_model.model_path)
            scene_xml = scene_xml.replace("{CAVE_BOXES_PATH}", cave_boxes_path)
            # Save the generated scene.xml in the cave folder
            xml_file = os.path.join(folder, "scene.xml")
            with open(xml_file, "w") as f:
                f.write(scene_xml)

            # Use from_xml_path to ensure <include file="..."/> statements are resolved relative to the XML file's directory.
            mj_model = MjModel.from_xml_path(
                xml_file,
            )
            mj_model.opt.timestep = config.sim_dt

            # Modify PD gains.
            mj_model.dof_damping[6:] = config.Kd_rot
            # Modify PD gains for prismatic joints.
            for i in [8, 11, 14, 17]:
                mj_model.dof_damping[i] = config.Kd_pri
                mj_model.actuator_gainprm[:, 0] = config.Kp_rot
                mj_model.actuator_biasprm[:, 1] = -config.Kp_rot
            for i in [2, 5, 8, 11]:
                mj_model.actuator_gainprm[i, 0] = config.Kp_pri
                mj_model.actuator_biasprm[i, 1] = -config.Kp_pri

            # Increase offscreen framebuffer size to render at higher resolutions.
            mj_model.vis.global_.offwidth = 3840
            mj_model.vis.global_.offheight = 2160

            mjx_model = mjx.put_model(mj_model)

            # Find JSON file
            json_file = os.path.join(folder, "cave_tunnel_info.json")
            # Load XML into mjx (GPU memory)
            # Delete the XML file to save space
            os.remove(xml_file)
            # Load JSON
            with open(json_file, "r") as f:
                json_data = json.load(f)
            # Extract target_pos and voxel_positions if present
            target_pos = None
            voxel_positions = None
            if "target_pos" in json_data:
                tp = json_data["target_pos"]
                target_pos = [tp.get("x", 0.0), tp.get("y", 0.0), tp.get("z", 0.0)]
            if "voxel_positions" in json_data:
                voxel_positions = json_data["voxel_positions"]
            self.envs.append({
                "cave_id": json_data.get("cave_id"),
                "mj_model": mj_model,
                "mjx_model": mjx_model,
                "json_data": json_data,
                "target_pos": target_pos,
                "voxel_positions": voxel_positions,
                "folder": folder
            })

    