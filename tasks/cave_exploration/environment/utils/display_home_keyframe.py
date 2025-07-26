
from etils import epath
import mujoco
import numpy as np

xml_path = mujoco.task_to_xml("flat_terrain_deflection").as_posix()
#xml_path = "reachbot/xmls/reachbot_basic/reachbot.xml"
# Read the XML file content
with open(xml_path, 'r') as file:
    xml = file.read()

# Load the model
model = mujoco.MjModel.from_xml_string(
        epath.Path(xml_path).read_text(), assets=get_assets()
    )
#model = mujoco.MjModel.from_xml_string(xml)

# Get joint range
joint_range = model.jnt_range
print(joint_range)

# Check specific joint range values
joint_name = "revolver11"
joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
print(f"Joint '{joint_name}' range: {model.jnt_range[joint_id]}")


data = mujoco.MjData(model)
# Set the position of the reachbot model to the scene's home keyframe
mujoco.mj_resetDataKeyframe(model, data, 0)

# Create a Mujoco viewer to display the model
viewer = mujoco.viewer.launch_passive(model, data)

# Function to convert local position to global position
def local_to_global(model, data, local_pos, site_name):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    site_pos = data.site_xpos[site_id]
    site_mat = data.site_xmat[site_id].reshape(3, 3)
    global_pos = site_pos + site_mat @ local_pos
    return global_pos

print("Before:", model.actuator_gainprm)
print("Before:", model.actuator_dyntype)
print("Before:", model.actuator_biasprm)
#print("Before:", model.actuator_ctrlrange)
print("Before:", model.dof_damping)
model.actuator_gainprm[:, 0] = 800
model.actuator_biasprm[:, 1] = -800
#model.dof_damping[6:] = 10
#print("After:", model.actuator_gainprm)
#print("After:", model.actuator_dyntype)
#print("After:", model.actuator_biasprm)
#print("After:", model.actuator_ctrlrange)
print("After:", model.dof_damping)

# Get and print joint min and max values
range = model.jnt_range[1:].T
print("Joint range:", range)

print("KP values:")
print(model.actuator_gainprm)
print(model.actuator_biasprm)
print("KD values:")
print(model.dof_damping)

# Print out qvel
print("qvel:", data.qvel)


# Render the scene
while True:
    mujoco.mj_step(model, data)
    viewer.sync()

    # Get the center of mass of the entire model
    com = data.subtree_com[0]
    #print(f"Center of Mass of the entire model: {com}")
    
    # Get the local position from the sensor
    imu_site_id = model.site("imu").id
    torso_height = data.site_xpos[imu_site_id][2]
    #print("Torso height:", torso_height)

    
    # Convert local position to global position
    #global_pos = local_to_global(model, data, qpos, 'imu')

    actuator_forces = data.actuator_force
    #print("Actuator forces:", actuator_forces)

    # Print kp and kd values
    #print("Actuator kp values:", model.actuator_gainprm[:, 0])
    #print("Actuator kd values:", model.actuator_gainprm[:, 1])
    
    # Print actuator force range
    actuator_forcerange = model.actuator_forcerange
    #print("Actuator force range:", actuator_forcerange)
    #print("Local position:", qpos, end=',')
    #print("Global position:", global_pos)
