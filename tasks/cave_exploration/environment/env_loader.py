import os
import random
import glob
import json
import xml.etree.ElementTree as ET
import mujoco  # Make sure mujoco is installed and configured correctly
from mujoco import mjx, MjModel  # Make sure mujoco's mjx Python bindings are installed
from models.model_loader import ReachbotModelType, ReachbotModel
import numpy as np

CAVES_DIR = os.path.join(os.path.dirname(__file__), "caves")

class CaveBatchLoader:
    def __init__(self, config, reachbot_model_type=ReachbotModelType.BASIC, caves_directory=CAVES_DIR):
        """
        config: Configuration dictionary containing simulation parameters.
        reachbot_model_type: Type of Reachbot model to use (BASIC or DEFLECTION).
        caves_directory: Directory containing cave environments.

        Load cave environments and split them into training (90%) and evaluation (10%) scenes.
        """
        self.reachbot_model = ReachbotModel(reachbot_model_type)
        self.caves_directory = caves_directory
        
        # Initialize scene properties
        self.training_scene = {"mj_model": None, "mjx_model": None, "caves": {}}
        self.eval_scene = {"mj_model": None, "mjx_model": None, "caves": {}}
        
        # Get all cave folders
        cave_folders = [os.path.join(caves_directory, d) for d in os.listdir(caves_directory)
                if os.path.isdir(os.path.join(caves_directory, d)) and d.startswith('cave_')]
        
        cave_folders.sort()  # Ensure consistent ordering
        print(f"Found {len(cave_folders)} cave folders in {caves_directory}.")
        
        # Split caves: 90% for training, 10% for evaluation (deterministic split)
        num_train_caves = int(len(cave_folders) * 0.9)
        train_caves = cave_folders[:num_train_caves]
        eval_caves = cave_folders[num_train_caves:]
        
        print(f"Dataset split (deterministic):")
        print(f"  Training caves: {len(train_caves)} ({len(train_caves)/len(cave_folders)*100:.1f}%)")
        print(f"  Evaluation caves: {len(eval_caves)} ({len(eval_caves)/len(cave_folders)*100:.1f}%)")
        print(f"  Total caves: {len(cave_folders)}")
        
        # Ensure no overlap
        train_set = set(train_caves)
        eval_set = set(eval_caves)
        if train_set.intersection(eval_set):
            raise ValueError("Training and evaluation sets overlap!")
        print("✓ Verified no overlap between training and evaluation sets")
        
        # Load cave metadata for all caves
        all_cave_data = self._load_all_cave_metadata(cave_folders)
        
        # Create training scene
        self._create_scene(train_caves, all_cave_data, config, "training")
        
        # Create evaluation scene  
        self._create_scene(eval_caves, all_cave_data, config, "eval")

    def _load_all_cave_metadata(self, cave_folders):
        """Load metadata from all caves using cave_config.json files."""
        cave_data = {}
        
        for folder in cave_folders:
            config_file = os.path.join(folder, "cave_config.json")
            if not os.path.exists(config_file):
                print(f"Warning: cave_config.json file not found for {folder}, skipping...")
                continue
                
            # Load JSON
            with open(config_file, "r") as f:
                config_data = json.load(f)
                
            cave_id = config_data["generation"]["cave_id"]
            
            # Extract relevant data
            target_pos = None
            starting_pos = None
            voxel_bounds = None
            voxel_size = config_data.get("voxel_size", 0.2)
            box_count = config_data.get("box_count", 0)
            boxes = config_data.get("boxes", [])
            
            if "target_pos" in config_data:
                tp = config_data["target_pos"]
                target_pos = [tp.get("x", 0.0), tp.get("y", 0.0), tp.get("z", 0.0)]
            if "starting_pos" in config_data:
                starting_pos = config_data["starting_pos"]
            if "voxel_bounds" in config_data:
                voxel_bounds = config_data["voxel_bounds"]
            
            cave_data[cave_id] = {
                "folder": folder,
                "target_pos": target_pos,
                "starting_pos": starting_pos,
                "box_count": box_count,
                "voxel_bounds": voxel_bounds,
                "voxel_size": voxel_size,
                "boxes": boxes
            }
            
        return cave_data

    def _find_master_cave(self, all_cave_data):
        """Find the cave with the most voxels to use as master cave."""
        if not all_cave_data:
            raise ValueError("No cave data available")
        
        master_cave_id = max(all_cave_data.keys(), key=lambda k: all_cave_data[k]["box_count"])
        master_cave_data = all_cave_data[master_cave_id]
        
        print(f"Selected cave {master_cave_id} as master cave with {master_cave_data['box_count']} voxels")
        return master_cave_id, master_cave_data

    def _create_master_cave_xml(self, master_cave_data):
        """Create XML content for the master cave from its voxel positions."""
        voxel_size = master_cave_data["voxel_size"]
        box_size = voxel_size / 2.0  # MuJoCo uses half-extents for box geometry
        boxes = master_cave_data["boxes"]
        cave_id = master_cave_data["cave_id"]
        
        # Create XML structure
        root = ET.Element("mujoco")
        body = ET.SubElement(root, "body", name="master_cave")
        
        # Add each box as a geom
        for i, box in enumerate(boxes):
            pos = box["position"]
            
            geom = ET.SubElement(body, "geom")
            geom.set("name", f"cave_wall_box_{cave_id}_{i}")
            geom.set("type", "box")
            geom.set("pos", f"{pos['x']} {pos['y']} {pos['z']}")
            geom.set("size", f"{box_size} {box_size} {box_size}")
            geom.set("material", "cave_wall")
            geom.set("contype", "2")
            geom.set("conaffinity", "1")
            geom.set("friction", "5.0 0.01 0.001")
            #geom.set("solref", "0.001 1")
            #geom.set("solimp", "0.99 0.99 0.001")
            geom.set("solref", "0.003 1")
            geom.set("solimp", "0.9 0.995 0.03 0.5 2")
            geom.set("margin", "0.005")  # Added margin for better collision handling

        # Convert to string
        xml_string = ET.tostring(root, encoding='unicode')
        return xml_string

    def _create_scene(self, cave_folders, all_cave_data, config, scene_type):
        """Create a scene (training or eval) using the master cave approach."""
        print(f"Creating {scene_type} scene with {len(cave_folders)} caves...")
        
        # Find the master cave (cave with most voxels)
        print(f"Step 1: Finding master cave...")
        master_cave_id, master_cave_data = self._find_master_cave(all_cave_data)
        print(f"Step 1 completed: Master cave {master_cave_id} selected")
        
        # Add the cave_id to master_cave_data for XML generation
        master_cave_data["cave_id"] = master_cave_id
        
        # Create XML for master cave
        print(f"Step 2: Creating master cave XML...")
        master_cave_xml = self._create_master_cave_xml(master_cave_data)
        print(f"Step 2 completed: Master cave XML generated")
        
        # Save master cave XML
        print(f"Step 3: Saving master cave XML...")
        master_cave_xml_file = os.path.join(self.caves_directory, f"{scene_type}_master_cave.xml")
        try:
            with open(master_cave_xml_file, "w") as f:
                f.write(master_cave_xml)
            print(f"Step 3 completed: Master cave XML saved to {master_cave_xml_file}")
        except Exception as e:
            print(f"Error saving master cave XML: {e}")
            raise
        
        # Create scene XML with master cave
        print(f"Step 4: Creating scene XML...")
        scene_xml = self._create_scene_xml(master_cave_xml_file)
        print(f"Step 4 completed: Scene XML created")
        
        # Save scene XML
        print(f"Step 5: Saving scene XML...")
        scene_xml_file = os.path.join(self.caves_directory, f"scene_{scene_type}.xml")
        with open(scene_xml_file, "w") as f:
            f.write(scene_xml)
        print(f"Step 5 completed: Scene XML saved to {scene_xml_file}")
        
        # Create MuJoCo models
        print(f"Step 6: Creating MuJoCo model...")
        mj_model = self._create_mujoco_model(scene_xml_file, config)
        print(f"Step 6 completed: MuJoCo model created")
        
        print(f"Step 7: Creating MJX model...")
        mjx_model = mjx.put_model(mj_model)
        print(f"Step 7 completed: MJX model created")
        
        # Clean up temporary files
        print(f"Step 8: Cleaning up temporary files...")
        os.remove(scene_xml_file)
        print(f"Step 8 completed: Temporary files cleaned up")
        
        # Store in appropriate scene
        scene_dict = self.training_scene if scene_type == "training" else self.eval_scene
        scene_dict["mj_model"] = mj_model
        scene_dict["mjx_model"] = mjx_model
        scene_dict["num_caves"] = len(cave_folders)
        scene_dict["master_cave_id"] = master_cave_id
        
        # Store voxel positions for all caves in this scene
        print(f"Step 9: Storing voxel positions for all caves...")
        for folder in cave_folders:
            cave_id = self._get_cave_id_from_folder(folder)
            if cave_id in all_cave_data:
                cave_info = all_cave_data[cave_id]
                scene_dict["caves"][cave_id] = {
                    "box_count": cave_info["box_count"],
                    "starting_pos": cave_info["starting_pos"],
                    "target_pos": cave_info["target_pos"],
                    "voxel_bounds": cave_info["voxel_bounds"],
                    "voxel_size": cave_info["voxel_size"],
                    "voxel_positions": [box["position"] for box in cave_info["boxes"]], # Store all voxel position
                }
        print(f"Step 9 completed: Voxel positions stored for {len(scene_dict['caves'])} caves")
        
        print(f"Created {scene_type} scene with master cave {master_cave_id} and {len(scene_dict['caves'])} total caves")

    def _get_cave_id_from_folder(self, folder):
        """Extract cave ID from folder path."""
        folder_name = os.path.basename(folder)
        return int(folder_name.split('_')[1])

    def get_cave_voxel_positions(self, cave_id, scene_type="training"):
        """Get voxel positions for a specific cave."""
        scene_dict = self.training_scene if scene_type == "training" else self.eval_scene
        
        if cave_id not in scene_dict["caves"]:
            raise ValueError(f"Cave {cave_id} not found in {scene_type} scene")
        
        return scene_dict["caves"][cave_id]["voxel_positions"]
    
    
    def get_all_cave_ids(self, scene_type="training"):
        """Get all cave IDs for a scene."""
        scene_dict = self.training_scene if scene_type == "training" else self.eval_scene
        return list(scene_dict["caves"].keys())
    
    def get_master_cave_id(self, scene_type="training"):
        """Get the master cave ID for a scene."""
        scene_dict = self.training_scene if scene_type == "training" else self.eval_scene
        return scene_dict.get("master_cave_id", None)
    
    def get_scene_data(self, scene_type="training"):
        """Get the complete scene data for training or evaluation.
        
        Args:
            scene_type: Either "training" or "eval"
            
        Returns:
            Dictionary containing all scene data including models and cave information
        """
        if scene_type == "training":
            return self.training_scene
        elif scene_type == "eval":
            return self.eval_scene
        else:
            raise ValueError(f"Invalid scene_type: {scene_type}. Must be 'training' or 'eval'")
    
    def get_training_scene_data(self):
        """Get the training scene data."""
        return self.training_scene
    
    def get_eval_scene_data(self):
        """Get the evaluation scene data."""
        return self.eval_scene
    
    def get_dataset_summary(self):
        """Get a summary of the dataset split."""
        training_cave_ids = sorted(list(self.training_scene["caves"].keys()))
        eval_cave_ids = sorted(list(self.eval_scene["caves"].keys()))
        
        return {
            "total_caves": len(training_cave_ids) + len(eval_cave_ids),
            "training_caves": {
                "count": len(training_cave_ids),
                "cave_ids": training_cave_ids
            },
            "eval_caves": {
                "count": len(eval_cave_ids), 
                "cave_ids": eval_cave_ids
            },
            "training_master_cave": self.training_scene.get("master_cave_id"),
            "eval_master_cave": self.eval_scene.get("master_cave_id"),
            "no_overlap": len(set(training_cave_ids).intersection(set(eval_cave_ids))) == 0
        }
    
    def create_cave_xml_from_positions(self, cave_id, scene_type="training"):
        """Create XML string for a specific cave from its voxel positions."""
        voxel_positions = self.get_cave_voxel_positions(cave_id, scene_type)
        scene_dict = self.training_scene if scene_type == "training" else self.eval_scene
        voxel_size = scene_dict["caves"][cave_id]["voxel_size"]
        
        # Create XML structure
        root = ET.Element("mujoco")
        body = ET.SubElement(root, "body", name=f"cave_{cave_id}")
        
        # Convert voxel_size to half-extents for MuJoCo box geometry
        half_extent = voxel_size / 2.0
        
        # Add each voxel as a geom
        for i, pos in enumerate(voxel_positions):
            geom = ET.SubElement(body, "geom")
            geom.set("name", f"cave_wall_box_{cave_id}_{i}")
            geom.set("type", "box")
            geom.set("pos", f"{pos['x']} {pos['y']} {pos['z']}")
            geom.set("size", f"{half_extent} {half_extent} {half_extent}")
            geom.set("material", "cave_wall")
            geom.set("contype", "2")
            geom.set("conaffinity", "1")
            geom.set("friction", "5.0 0.01 0.001")
            geom.set("solref", "0.001 1")
            geom.set("solimp", "0.99 0.99 0.001")
        
        # Convert to string
        xml_string = ET.tostring(root, encoding='unicode')
        return xml_string

    def _create_scene_xml(self, master_cave_xml_file):
        """Create the full scene XML with reachbot and master cave."""
        # Load the scene template
        template_path = os.path.join(os.path.dirname(__file__), "scene_reachbot_cave.xml")
        with open(template_path, "r") as f:
            scene_template = f.read()
        
        # Replace placeholders in template
        scene_xml = scene_template.replace("{REACHBOT_MODEL_PATH}", self.reachbot_model.model_path)
        scene_xml = scene_xml.replace("{CAVE_BOXES_PATH}", master_cave_xml_file)
        
        return scene_xml

    def _create_mujoco_model(self, scene_xml_file, config):
        """Create and configure MuJoCo model from scene XML."""
        # Use from_xml_path to ensure <include file="..."/> statements are resolved
        mj_model = MjModel.from_xml_path(scene_xml_file)
        
        # Load model and let robot settle to get initial state
        data = mujoco.MjData(mj_model)
        steps = 5 / config.sim_dt  # Number of steps to let the robot settle
        for _ in range(int(steps)):
            mujoco.mj_step(mj_model, data)

        # Modify PD gains
        mj_model.dof_damping[6:] = config.Kd_rot
        # Modify PD gains for prismatic joints
        for i in [8, 11, 14, 17]:
            mj_model.dof_damping[i] = config.Kd_pri
            mj_model.actuator_gainprm[:, 0] = config.Kp_rot
            mj_model.actuator_biasprm[:, 1] = -config.Kp_rot
        for i in [2, 5, 8, 11]:
            mj_model.actuator_gainprm[i, 0] = config.Kp_pri
            mj_model.actuator_biasprm[i, 1] = -config.Kp_pri

        # Increase offscreen framebuffer size to render at higher resolutions
        mj_model.vis.global_.offwidth = 3840
        mj_model.vis.global_.offheight = 2160

        return mj_model

